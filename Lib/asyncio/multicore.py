"""Multi-core stream servers: one event loop per CPU core (experimental).

``asyncio.start_server(..., workers=N)`` serves connections from event loops
running in parallel, each in its own subinterpreter with its own GIL
(PEP 684, PEP 734), all inside one process.  The main interpreter accepts
connections and hands each socket to a worker loop; nothing is pickled,
because interpreters share the process file-descriptor table.

Each worker runs at most as many connections at once as an adaptive
concurrency limit allows (TCP-Vegas style, as in Netflix concurrency-limits).
Connections beyond the limit wait in a FIFO queue on the worker; when every
worker is full the acceptor pauses, so further clients wait in the kernel's
accept queue.  Nothing is rejected unless ``overload='shed'`` or
``max_queue_wait`` asks for it.

Workers publish their state into a shared buffer that the acceptor reads
lock-free (CRC-checked records, since interpreters really run in parallel)
to route each connection to the least loaded worker, avoiding one whose loop
is stalled right now.  With ``workers='auto'`` the number of workers follows
the CPUs the process may use, including a container's CPU quota.

The client callback must be a module-level function: each worker imports it
by name and runs it in its own interpreter, so it does not share global state
with the other workers or with the caller.  Unlike asyncio.start_server(),
the connection is closed when the callback returns: that return is what
frees the worker's slot and gives the limiter its latency sample.
"""

__all__ = ('ConcurrencyLimit', 'MulticoreServer', 'WorkerStats',
           'current_worker', 'start_server')

import collections
import contextlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import traceback
import zlib
from concurrent import interpreters

from . import events
from . import exceptions
from . import streams
from . import tasks
from . import timeouts


# --- Adaptive concurrency limit -------------------------------------------


class ConcurrencyLimit:
    """Parameters of the per-worker adaptive concurrency limit."""

    __slots__ = ('initial', 'minimum', 'maximum', 'backoff', 'probe')

    def __init__(self, initial=100, minimum=2, maximum=1000, *,
                 backoff=0.9, probe=30):
        if not 1 <= minimum <= initial <= maximum:
            raise ValueError('need 1 <= minimum <= initial <= maximum')
        if not 0 < backoff < 1:
            raise ValueError('backoff must be between 0 and 1')
        self.initial = initial
        self.minimum = minimum
        self.maximum = maximum
        self.backoff = backoff
        self.probe = probe

    def _as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return (f'ConcurrencyLimit(initial={self.initial}, '
                f'minimum={self.minimum}, maximum={self.maximum}, '
                f'backoff={self.backoff})')


class _VegasLimiter:
    """TCP-Vegas queue estimation with multiplicative backoff.

    With L the limit, queue = ceil(L * (1 - rtt_noload / rtt)) and
    threshold, alpha, beta = log10(L) * (1, 3, 6): a sample taken while the
    worker uses less than half its limit changes nothing; otherwise the
    limit grows by beta below the threshold, by the threshold below alpha,
    holds up to beta and shrinks by the threshold above it.  Timeouts and
    loop stalls multiply it by `backoff`.
    """

    def __init__(self, config):
        self.config = config
        self.limit = float(config.initial)
        self.in_flight = 0
        self.rtt_noload = math.inf
        self.rtt_short = 0.0
        self._samples = 0

    @property
    def effective_limit(self):
        return max(self.config.minimum,
                   min(self.config.maximum, int(self.limit)))

    def has_capacity(self):
        return self.in_flight < self.effective_limit

    def acquire(self):
        self.in_flight += 1

    def release(self, rtt, dropped=False):
        in_flight = self.in_flight
        self.in_flight = max(0, self.in_flight - 1)
        if dropped:
            self.backoff()
            return
        if rtt <= 0.0:
            return
        self._observe(rtt)
        self._update(rtt, in_flight)

    def backoff(self):
        self.limit = max(float(self.config.minimum),
                         math.floor(self.limit * self.config.backoff))

    def _observe(self, rtt):
        if self.rtt_short == 0.0:
            self.rtt_short = rtt
        else:
            self.rtt_short = 0.3 * rtt + 0.7 * self.rtt_short
        self._samples += 1
        # Forget the minimum now and then, so a permanently slower
        # workload gets a new baseline instead of shrinking the limit.
        if self._samples >= self.config.probe * self.effective_limit:
            self._samples = 0
            self.rtt_noload = rtt
        elif rtt < self.rtt_noload:
            self.rtt_noload = rtt

    def _update(self, rtt, in_flight):
        limit = self.limit
        if in_flight * 2 < limit:
            return
        log_limit = max(1.0, math.log10(limit))
        threshold, alpha, beta = log_limit, 3 * log_limit, 6 * log_limit
        queue = math.ceil(limit * (1 - self.rtt_noload / rtt))
        if queue <= threshold:
            limit += beta
        elif queue < alpha:
            limit += threshold
        elif queue > beta:
            limit -= threshold
        else:
            return
        self.limit = max(float(self.config.minimum),
                         min(float(self.config.maximum), limit))


# --- Telemetry shared between interpreters --------------------------------

# Covers the cache-line size of x86-64 (64 bytes) and Apple silicon (128).
_STRIDE = 128
_CRC = struct.Struct('<I')
_BODY = struct.Struct('<IIIIIQQQQdddddd')
_RECORD = _CRC.size + _BODY.size
assert _RECORD <= _STRIDE
_NEVER_WRITTEN = bytes(_RECORD)

_STARTING, _READY, _STOPPING, _STOPPED, _FAILED = range(5)
_STATE_NAMES = ('starting', 'ready', 'stopping', 'stopped', 'failed')

WorkerStats = collections.namedtuple('WorkerStats', [
    'worker_id', 'state', 'in_flight', 'queued', 'limit', 'cpu',
    'received', 'completed', 'shed', 'stalls', 'rtt', 'rtt_noload',
    'loop_lag', 'max_loop_lag', 'steal', 'heartbeat',
])
WorkerStats.__doc__ = """\
A worker's state as last published.

Times are in seconds.  `state` is one of 'starting', 'ready', 'stopping',
'stopped' and 'failed'; `cpu` is -1 when the worker is not pinned.
`steal` is the total time the worker's thread spent off the CPU while it
should have been running, beyond normal timer slack."""


class _Slot:
    """One worker's record in the shared buffer; the worker is its only writer.

    A reader in another interpreter runs truly in parallel and Python has no
    memory fences, so a copy can catch a record half written (a sequence
    counter alone was seen to fail on weakly ordered CPUs).  Each record is
    written with one copy and carries a CRC32 of its fields; a reader takes
    one copy and accepts it only if the checksum matches.
    """

    def __init__(self, buffer, worker_id):
        self._buffer = buffer
        self._offset = worker_id * _STRIDE

    def publish(self, state, in_flight=0, queued=0, limit=0, cpu=-1,
                received=0, completed=0, shed=0, stalls=0, rtt=0.0,
                rtt_noload=0.0, loop_lag=0.0, max_loop_lag=0.0, steal=0.0):
        body = _BODY.pack(
            state, in_flight, queued, limit, cpu & 0xFFFFFFFF, received,
            completed, shed, stalls, rtt,
            0.0 if rtt_noload == math.inf else rtt_noload,
            loop_lag, max_loop_lag, steal, time.monotonic())
        end = self._offset + _RECORD
        self._buffer[self._offset:end] = _CRC.pack(zlib.crc32(body)) + body


def _read_slot(buffer, worker_id, retries=64):
    """The worker's last intact record, or None if none was seen in time."""
    offset = worker_id * _STRIDE
    for _ in range(retries):
        record = bytes(buffer[offset:offset + _RECORD])
        body = record[_CRC.size:]
        if record == _NEVER_WRITTEN or (
                zlib.crc32(body) == _CRC.unpack_from(record)[0]):
            fields = list(_BODY.unpack(body))
            fields[0] = _STATE_NAMES[fields[0]]
            if fields[4] == 0xFFFFFFFF:
                fields[4] = -1
            return WorkerStats(worker_id, *fields)
    return None


# --- CPUs ------------------------------------------------------------------

# A Windows affinity mask covers one processor group of 64 logical CPUs.
_WINDOWS_GROUP_SIZE = 64


def _available_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    count = os.process_cpu_count() or 1
    if sys.platform == 'win32':
        count = min(count, _WINDOWS_GROUP_SIZE)
    return list(range(count))


def _physical_cpus():
    """One logical CPU per physical core where that is known."""
    cpus = _available_cpus()
    if sys.platform == 'win32':
        return _windows_physical_cpus(cpus)
    if not sys.platform.startswith('linux'):
        return cpus
    chosen, seen = [], set()
    for cpu in cpus:
        path = (f'/sys/devices/system/cpu/cpu{cpu}/topology/'
                'thread_siblings_list')
        try:
            with open(path) as f:
                key = f.read().strip()
        except OSError:
            key = str(cpu)
        if key not in seen:
            seen.add(key)
            chosen.append(cpu)
    return chosen


def _windows_physical_cpus(cpus):
    import ctypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel32.GetLogicalProcessorInformationEx
    query.argtypes = [ctypes.c_int, ctypes.c_void_p,
                      ctypes.POINTER(ctypes.c_uint32)]
    query.restype = ctypes.c_int
    relation_processor_core = 0
    size = ctypes.c_uint32(0)
    query(relation_processor_core, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not size.value or not query(relation_processor_core, buffer,
                                   ctypes.byref(size)):
        return cpus
    allowed = set(cpus)
    chosen = []
    offset = 0
    raw = buffer.raw
    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, then a
    # PROCESSOR_RELATIONSHIP whose first GROUP_AFFINITY starts 32 bytes in.
    while offset < size.value:
        record_size = int.from_bytes(raw[offset + 4:offset + 8], 'little')
        mask = int.from_bytes(raw[offset + 32:offset + 40], 'little')
        group = int.from_bytes(raw[offset + 40:offset + 42], 'little')
        if group == 0:
            core = [cpu for cpu in range(_WINDOWS_GROUP_SIZE)
                    if mask >> cpu & 1 and cpu in allowed]
            if core:
                chosen.append(core[0])
        offset += record_size or size.value
    return sorted(chosen) or cpus


def _cpu_quota(cgroup_root='/sys/fs/cgroup', proc_cgroup='/proc/self/cgroup'):
    """CPUs' worth of time the process's cgroup allows, or None.

    os.process_cpu_count() ignores this limit, so a container limited to
    two CPUs on a 64-core host would otherwise look like 64 CPUs.
    """
    try:
        with open(proc_cgroup) as f:
            entries = f.read().splitlines()
    except OSError:
        return None
    limits = []
    for entry in entries:
        hierarchy, controllers, path = entry.split(':', 2)
        if hierarchy == '0':
            limits += _cgroup_v2_limits(cgroup_root, path)
        elif 'cpu' in controllers.split(','):
            limits += _cgroup_v1_limits(
                os.path.join(cgroup_root, controllers), path)
    return min(limits) if limits else None


def _cgroup_v2_limits(root, path):
    limits = []
    for directory in _cgroup_dirs(root, path):
        try:
            with open(os.path.join(directory, 'cpu.max')) as f:
                quota, period = f.read().split()
        except (OSError, ValueError):
            continue
        if quota != 'max':
            limits.append(int(quota) / int(period))
    return limits


def _cgroup_v1_limits(root, path):
    limits = []
    for directory in _cgroup_dirs(root, path):
        try:
            with open(os.path.join(directory, 'cpu.cfs_quota_us')) as f:
                quota = int(f.read())
            with open(os.path.join(directory, 'cpu.cfs_period_us')) as f:
                period = int(f.read())
        except (OSError, ValueError):
            continue
        if quota > 0 and period > 0:
            limits.append(quota / period)
    return limits


def _cgroup_dirs(root, path):
    """The process's cgroup and its ancestors; a limit on any of them applies.

    Inside a container the host-side path is not mounted and the container's
    own cgroup is the root.
    """
    root = os.path.normpath(root)
    directory = os.path.normpath(os.path.join(root, path.lstrip('/')))
    if not os.path.isdir(directory):
        return [root]
    dirs = [directory]
    while directory != root and directory.startswith(root + os.sep):
        directory = os.path.dirname(directory)
        dirs.append(directory)
    return dirs


def _cpu_budget():
    """The CPUs workers should use: physical cores, capped by the quota."""
    cpus = _physical_cpus()
    quota = _cpu_quota()
    if quota is not None:
        cpus = cpus[:max(1, math.ceil(quota))]
    return cpus


def _pin_current_thread(cpu):
    """Restrict the calling thread to `cpu`; False where unsupported."""
    if hasattr(os, 'sched_setaffinity'):
        # On Linux pid 0 means the calling thread, not the whole process.
        os.sched_setaffinity(0, {cpu})
        return True
    if sys.platform == 'win32':
        _windows_set_thread_affinity(1 << cpu)
        return True
    return False


def _current_thread_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return os.sched_getaffinity(0)
    if sys.platform == 'win32':
        # There is no getter: setting returns the previous mask.
        mask = _windows_set_thread_affinity(
            (1 << len(_available_cpus())) - 1)
        _windows_set_thread_affinity(mask)
        return {cpu for cpu in range(_WINDOWS_GROUP_SIZE) if mask >> cpu & 1}
    return None


def _windows_set_thread_affinity(mask):
    import ctypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p,
                                               ctypes.c_size_t]
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    previous = kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(),
                                              mask)
    if previous == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return previous


# --- Finding the callback in another interpreter --------------------------

_MAIN_ALIAS = '__asyncio_multicore_main__'


def _reference(target, role):
    """'module:qualname' that a worker interpreter can import."""
    if isinstance(target, str):
        return target
    module = getattr(target, '__module__', None)
    qualname = getattr(target, '__qualname__', None)
    if (not callable(target) or inspect.ismethod(target) or not module
            or not qualname or '<' in qualname):
        raise TypeError(
            f'{role} must be a module-level function, because each worker '
            f'is a separate interpreter that imports it by name; lambdas, '
            f'closures and bound methods cannot be shared: got {target!r}')
    if module == '__main__':
        if _main_path() is None:
            raise TypeError(f'{role} {qualname!r} is defined in an '
                            f'interactive session; define it in a file')
        module = _MAIN_ALIAS
    return f'{module}:{qualname}'


def _main_path():
    return getattr(sys.modules.get('__main__'), '__file__', None)


def _resolve(reference, main_path):
    module_name, _, qualname = reference.partition(':')
    if module_name == _MAIN_ALIAS:
        target = _load_main(main_path)
    else:
        target = importlib.import_module(module_name)
    for part in qualname.split('.'):
        target = getattr(target, part)
    return target


def _load_main(main_path):
    # Like multiprocessing's __mp_main__: the script runs again under
    # another name, so its "if __name__ == '__main__':" block does not.
    if _MAIN_ALIAS in sys.modules:
        return sys.modules[_MAIN_ALIAS]
    spec = importlib.util.spec_from_file_location(_MAIN_ALIAS, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MAIN_ALIAS] = module
    spec.loader.exec_module(module)
    return module


# --- Worker interpreters ----------------------------------------------------

_STOP = -1
_SHED_LINGER = 1.0
# Idle timer wake-ups run up to ~1 ms late; only off-CPU time beyond this
# counts as steal.
_TIMER_SLACK = 0.002
# time.thread_time() advances in scheduler ticks (1/64 s) on Windows.
_CPU_CLOCK_STEP = 1 / 64 if sys.platform == 'win32' else 0.0


class _WorkerContext:
    __slots__ = ('worker_id', 'cpu', 'limiter')

    def __init__(self, worker_id, cpu, limiter):
        self.worker_id = worker_id
        self.cpu = cpu
        self.limiter = limiter

    def __repr__(self):
        return f'<worker {self.worker_id} cpu={self.cpu}>'


_current = None


def current_worker():
    """The worker running the caller: its `worker_id` and `cpu`.

    Raises RuntimeError outside a multi-core server's worker.
    """
    if _current is None:
        raise RuntimeError('not running inside a multi-core server worker')
    return _current


def _worker_main(config_json, buffer, inbox, doorbell_fd):
    config = json.loads(config_json)
    slot = _Slot(buffer, config['worker_id'])
    # Own the doorbell first, so it is closed however the worker ends.
    doorbell = socket.socket(fileno=doorbell_fd)
    try:
        doorbell.setblocking(False)
        with _fine_timers():
            from . import runners
            runners.run(_Worker(config, slot, inbox, doorbell).run())
    except BaseException:
        slot.publish(_FAILED)
        raise
    finally:
        doorbell.close()


@contextlib.contextmanager
def _fine_timers():
    """Ask Windows for 1 ms timers while the worker runs.

    By default Windows wakes event loops on a ~15.6 ms tick, which the
    watchdog would read as loop lag and steal.
    """
    if sys.platform != 'win32':
        yield
        return
    import ctypes

    winmm = ctypes.WinDLL('winmm')
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)


class _Worker:

    def __init__(self, config, slot, inbox, doorbell):
        self.config = config
        self.slot = slot
        self.inbox = inbox
        self.doorbell = doorbell
        self.state = _STARTING
        self.cpu = -1
        limit = config['limit']
        self.limiter = (_VegasLimiter(ConcurrencyLimit(**limit))
                        if limit else None)
        self.shed_response = config['shed_response'].encode('latin-1')
        self.waiting = collections.deque()
        self.lingering = set()
        self.tasks = set()
        self.received = self.completed = self.shed = self.stalls = 0
        self.loop_lag = self.max_loop_lag = self.steal = 0.0

    async def run(self):
        global _current
        self.loop = events.get_running_loop()
        self.stopping = self.loop.create_future()
        if self.config['cpu'] >= 0 and _pin_current_thread(self.config['cpu']):
            self.cpu = self.config['cpu']
        _current = _WorkerContext(self.config['worker_id'], self.cpu,
                                  self.limiter)
        main_path = self.config['main_path']
        self.callback = _resolve(self.config['callback'], main_path)
        if self.config['on_start']:
            result = _resolve(self.config['on_start'], main_path)(_current)
            if inspect.isawaitable(result):
                await result
        reader = self.loop.create_task(self._read_inbox(self.doorbell))
        watchdog = self.loop.create_task(self._watchdog())
        self.state = _READY
        self.publish()
        try:
            await self.stopping
        finally:
            self.state = _STOPPING
            self.publish()
            reader.cancel()
            await self._drain()
            watchdog.cancel()
            await tasks.gather(watchdog, reader, return_exceptions=True)
            self.state = _STOPPED
            self.publish()

    async def _drain(self):
        # Connections handed to this worker, queued ones included, are
        # finished within the grace period.
        deadline = self.loop.time() + self.config['shutdown_grace']
        while (self.tasks or self.waiting) and self.loop.time() < deadline:
            await tasks.sleep(0.01)
        for sock, expiry in self.waiting:
            if expiry is not None:
                expiry.cancel()
            sock.close()
        self.waiting.clear()
        for task in self.tasks:
            task.cancel()
        await tasks.gather(*self.tasks, return_exceptions=True)
        for sock in self.lingering:
            sock.close()

    def publish(self):
        limiter = self.limiter
        self.slot.publish(
            self.state,
            in_flight=limiter.in_flight if limiter else len(self.tasks),
            queued=len(self.waiting),
            limit=limiter.effective_limit if limiter else 0,
            cpu=self.cpu, received=self.received, completed=self.completed,
            shed=self.shed, stalls=self.stalls,
            rtt=limiter.rtt_short if limiter else 0.0,
            rtt_noload=limiter.rtt_noload if limiter else 0.0,
            loop_lag=self.loop_lag, max_loop_lag=self.max_loop_lag,
            steal=self.steal)

    async def _read_inbox(self, doorbell):
        while True:
            if not await self.loop.sock_recv(doorbell, 4096):
                self._stop()
                return
            while True:
                try:
                    fd = self.inbox.get_nowait()
                except interpreters.QueueEmpty:
                    break
                if fd == _STOP:
                    self._stop()
                    return
                sock = socket.socket(fileno=fd)
                sock.setblocking(False)
                self._admit(sock)

    def _stop(self):
        if not self.stopping.done():
            self.stopping.set_result(None)

    def _admit(self, sock):
        self.received += 1
        if self._has_capacity() and not self.waiting:
            self._start(sock)
        elif self.config['overload'] == 'shed':
            self._shed(sock)
        else:
            expiry = None
            if self.config['max_queue_wait'] is not None:
                expiry = self.loop.call_later(self.config['max_queue_wait'],
                                              self._expire, sock)
            self.waiting.append((sock, expiry))
        self.publish()

    def _has_capacity(self):
        return self.limiter is None or self.limiter.has_capacity()

    def _start(self, sock):
        if self.limiter:
            self.limiter.acquire()
        task = self.loop.create_task(self._serve(sock))
        self.tasks.add(task)
        task.add_done_callback(self._done)

    def _expire(self, sock):
        for entry in self.waiting:
            if entry[0] is sock:
                self.waiting.remove(entry)
                break
        self._shed(sock)
        self.publish()

    def _done(self, task):
        self.tasks.discard(task)
        while self.waiting and self._has_capacity():
            sock, expiry = self.waiting.popleft()
            if expiry is not None:
                expiry.cancel()
            self._start(sock)
        self.publish()

    def _shed(self, sock):
        self.shed += 1
        _shed(sock, self.shed_response, self.loop, self.lingering)

    async def _serve(self, sock):
        started = self.loop.time()
        dropped = False
        try:
            reader = streams.StreamReader(limit=self.config['stream_limit'],
                                          loop=self.loop)
            protocol = streams.StreamReaderProtocol(reader, loop=self.loop)
            transport, _ = await self.loop.connect_accepted_socket(
                lambda: protocol, sock)
            writer = streams.StreamWriter(transport, protocol, reader,
                                          self.loop)
            try:
                async with timeouts.timeout(self.config['request_timeout']):
                    result = self.callback(reader, writer)
                    if inspect.isawaitable(result):
                        await result
            except TimeoutError:
                dropped = True
            finally:
                writer.close()
        except Exception as exc:
            self.loop.call_exception_handler({
                'message': 'Unhandled exception in client_connected_cb',
                'exception': exc,
            })
        finally:
            self.completed += 1
            if self.limiter:
                self.limiter.release(self.loop.time() - started, dropped)
            self.publish()

    async def _watchdog(self):
        interval = self.config['watchdog_interval']
        threshold = self.config['stall_threshold'] + _CPU_CLOCK_STEP
        last_wall, last_cpu = self.loop.time(), time.thread_time()
        while True:
            await tasks.sleep(interval)
            wall, cpu = self.loop.time(), time.thread_time()
            elapsed = wall - last_wall
            # Lag spent running our own code is load, which the limiter
            # sees in its latency samples; only time off the CPU is a stall.
            off_cpu = elapsed - (cpu - last_cpu) - interval
            last_wall, last_cpu = wall, cpu
            self.loop_lag = max(0.0, elapsed - interval)
            self.max_loop_lag = max(self.max_loop_lag, self.loop_lag)
            self.steal += max(0.0, off_cpu - _TIMER_SLACK)
            if off_cpu >= threshold:
                self.stalls += 1
                if self.limiter:
                    self.limiter.backoff()
            self.publish()


def _shed(sock, response, loop, lingering):
    # Half-close and close later: closing with unread request bytes would
    # send an RST that can destroy the reply.  `lingering` holds the sockets
    # still waiting, so a shutdown can close them.
    try:
        if response:
            sock.send(response)
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        sock.close()
        return
    lingering.add(sock)

    def close():
        lingering.discard(sock)
        sock.close()

    loop.call_later(_SHED_LINGER, close)


# --- Deciding how many workers take traffic --------------------------------


class _Autoscaler:
    """Parks and unparks workers to follow the CPU budget.

    With park_on_steal, a worker whose CPU is being taken by other
    processes (steal ratio above 0.3 for two intervals) is also parked, and
    returns after three intervals below 0.1.  That is off by default: under
    fair-share scheduling a parked worker's CPU time goes to the competing
    processes, which lowered throughput when measured.
    """

    def __init__(self, park_on_steal):
        self.park_on_steal = park_on_steal
        self._high = collections.Counter()
        self._low = collections.Counter()

    def decide(self, samples, budget):
        """Return (park, unpark, spawn) for [(worker_id, active, ratio)]."""
        budget = max(1, budget)
        for worker_id, _, ratio in samples:
            high, low = self._high[worker_id], self._low[worker_id]
            self._high[worker_id] = high + 1 if ratio > 0.3 else 0
            self._low[worker_id] = low + 1 if ratio < 0.1 else 0
        active = [s for s in samples if s[1]]
        parked = [s for s in samples if not s[1]]
        spawn = max(0, budget - len(samples))
        if len(active) > budget:
            excess = sorted(active, key=lambda s: (-s[2], -s[0]))
            return {s[0] for s in excess[:len(active) - budget]}, set(), 0
        if self.park_on_steal and len(active) > 1:
            stolen = [s for s in active if self._high[s[0]] >= 2]
            if stolen:
                worst = max(stolen, key=lambda s: (s[2], s[0]))
                self._high[worst[0]] = 0
                return {worst[0]}, set(), spawn
        ready = sorted((s for s in parked if self._low[s[0]] >= 3),
                       key=lambda s: (s[2], s[0]))
        return set(), {s[0] for s in ready[:budget - len(active)]}, spawn


# --- The server -------------------------------------------------------------

_BOOTSTRAP = """\
import sys
sys.path[:] = list(_asyncio_path)
from asyncio import multicore
multicore._worker_main(_asyncio_config, _asyncio_buffer, _asyncio_inbox,
                       _asyncio_doorbell)
"""

_INBOX_CAPACITY = 1 << 16
_CAPACITY_POLL = 0.001
_UNLIMITED = 1 << 30


async def start_server(client_connected_cb, host=None, port=None, *,
                       limit=streams._DEFAULT_LIMIT, workers='auto',
                       sock=None, backlog=100, start_serving=True,
                       concurrency=ConcurrencyLimit(), overload='queue',
                       worker_queue=1024, max_queue_wait=None,
                       shed_response=b'', request_timeout=None, pin=True,
                       routing='least_loaded', on_start=None,
                       park_on_steal=False, adapt_interval=1.0,
                       shutdown_grace=5.0, startup_timeout=60.0):
    """Start a TCP server that serves connections on every CPU core.

    Takes the arguments of asyncio.start_server() that apply (host, port,
    limit, sock, backlog, start_serving) and returns a MulticoreServer,
    which behaves like asyncio.Server.

    workers is a number of worker loops, or 'auto' for one per usable CPU,
    followed while the server runs.  concurrency is a ConcurrencyLimit, or
    None for no limit.  overload is 'queue' (connections over the limit
    wait) or 'shed' (they get shed_response and are closed).
    max_queue_wait sheds connections that waited longer than that.
    request_timeout cancels a callback that runs longer.  pin binds each
    worker thread to its own CPU where the OS allows it.  routing is
    'least_loaded' or 'round_robin'.  on_start is called in each worker
    with its context before it serves.
    """
    server = MulticoreServer(
        client_connected_cb, limit=limit, workers=workers,
        concurrency=concurrency, overload=overload,
        worker_queue=worker_queue, max_queue_wait=max_queue_wait,
        shed_response=shed_response, request_timeout=request_timeout,
        pin=pin, routing=routing, on_start=on_start,
        park_on_steal=park_on_steal, adapt_interval=adapt_interval,
        shutdown_grace=shutdown_grace, startup_timeout=startup_timeout)
    try:
        await server._start(host, port, sock, backlog, start_serving)
    except BaseException:
        server.close()
        await server.wait_closed()
        raise
    return server


class MulticoreServer(events.AbstractServer):
    """A stream server whose connections are served on several CPU cores.

    Created by start_server(); behaves like asyncio.Server.
    """

    def __init__(self, client_connected_cb, *, limit, workers, concurrency,
                 overload, worker_queue, max_queue_wait, shed_response,
                 request_timeout, pin, routing, on_start, park_on_steal,
                 adapt_interval, shutdown_grace, startup_timeout):
        if overload not in ('queue', 'shed'):
            raise ValueError(
                f"overload must be 'queue' or 'shed', not {overload!r}")
        if routing not in ('least_loaded', 'round_robin'):
            raise ValueError(f"routing must be 'least_loaded' or "
                             f"'round_robin', not {routing!r}")
        self._auto = workers == 'auto'
        if self._auto:
            budget = _cpu_budget()
            self._capacity = max(len(_available_cpus()), len(budget))
            self._initial_cpus = budget if pin else [-1] * len(budget)
        elif isinstance(workers, int) and workers > 0:
            cpus = _physical_cpus()
            self._capacity = workers
            self._initial_cpus = [cpus[i % len(cpus)] if pin else -1
                                  for i in range(workers)]
        else:
            raise ValueError(
                f"workers must be a positive int or 'auto', not {workers!r}")
        self._loop = events.get_running_loop()
        self._pin = pin
        self._routing = routing
        self._overload = overload
        self._worker_queue = worker_queue if overload == 'queue' else 0
        self._shed_response = shed_response
        self._startup_timeout = startup_timeout
        self._adapt_interval = adapt_interval
        watchdog_interval, stall_threshold = 0.005, 0.02
        self._stall_threshold = stall_threshold
        self._stale_after = max(stall_threshold, 4 * watchdog_interval)
        self._config = {
            'callback': _reference(client_connected_cb, 'client_connected_cb'),
            'on_start': _reference(on_start, 'on_start') if on_start else None,
            'main_path': _main_path(),
            'stream_limit': limit,
            'limit': concurrency._as_dict() if concurrency else None,
            'overload': overload,
            'max_queue_wait': max_queue_wait,
            'shed_response': shed_response.decode('latin-1'),
            'request_timeout': request_timeout,
            'watchdog_interval': watchdog_interval,
            'stall_threshold': stall_threshold,
            'shutdown_grace': shutdown_grace,
        }
        self._buffer = memoryview(bytearray(_STRIDE * self._capacity))
        self._interps = []
        self._threads = []
        self._inboxes = []
        self._doorbells = []
        self._cpus = []
        self._dispatched = []
        self._active = []
        self._failures = {}
        self._autoscaler = _Autoscaler(park_on_steal)
        self._steal_seen = {}
        self._adapter = None
        self._acceptor = None
        self._listener = None
        self._sockets = ()
        self._router_shed = 0
        self._lingering = set()
        self._paused = 0.0
        self._next = 0
        self._closed = None
        self._serving_forever = None

    def __repr__(self):
        return f'<{self.__class__.__name__} sockets={self.sockets!r}>'

    # asyncio.AbstractServer

    @property
    def sockets(self):
        return () if self._closed is not None else self._sockets

    def get_loop(self):
        return self._loop

    def is_serving(self):
        return self._closed is None and self._acceptor is not None

    async def start_serving(self):
        if self._closed is not None:
            raise RuntimeError(f'server {self!r} is closed')
        if self._listener is not None and self._acceptor is None:
            self._acceptor = self._loop.create_task(self._accept_loop())

    async def serve_forever(self):
        if self._serving_forever is not None:
            raise RuntimeError(
                f'server {self!r} is already being awaited on '
                f'serve_forever()')
        await self.start_serving()
        self._serving_forever = self._loop.create_future()
        try:
            await self._serving_forever
        except exceptions.CancelledError:
            self.close()
            await self.wait_closed()
            raise
        finally:
            self._serving_forever = None

    def close(self):
        """Stop listening now; wait_closed() waits for accepted connections."""
        if self._closed is not None:
            return
        acceptor = self._stop_accepting()
        self._closed = self._loop.create_task(self._shutdown(acceptor))
        if (self._serving_forever is not None
                and not self._serving_forever.done()):
            self._serving_forever.cancel()

    async def wait_closed(self):
        if self._closed is not None:
            await tasks.shield(self._closed)

    # Statistics

    @property
    def workers(self):
        """The number of worker loops started so far."""
        return len(self._threads)

    def worker_stats(self):
        """A WorkerStats for each worker."""
        return [_read_slot(self._buffer, w, retries=100_000)
                for w in range(len(self._threads))]

    def active_workers(self):
        """For each worker, whether it currently receives connections."""
        return list(self._active)

    # Start-up and shutdown

    async def _start(self, host, port, sock, backlog, start_serving):
        for cpu in self._initial_cpus:
            self._spawn(cpu)
        failure = await self._wait_ready(range(len(self._threads)))
        if failure:
            raise RuntimeError(f'multi-core server workers failed to '
                               f'start: {failure}')
        if sock is None:
            if host is None and socket.has_dualstack_ipv6():
                sock = socket.create_server(
                    ('', port or 0), family=socket.AF_INET6, backlog=backlog,
                    dualstack_ipv6=True)
            else:
                sock = socket.create_server((host or '', port or 0),
                                            backlog=backlog)
        sock.setblocking(False)
        self._listener = sock
        self._sockets = (sock,)
        if self._auto:
            self._adapter = self._loop.create_task(self._adapt_loop())
        if start_serving:
            await self.start_serving()

    def _stop_accepting(self):
        acceptor, self._acceptor = self._acceptor, None
        if acceptor is not None:
            acceptor.cancel()
        if self._listener is not None:
            # Unregister before closing: the next socket may get the same
            # descriptor number while the selector still watches it.
            with contextlib.suppress(NotImplementedError, ValueError):
                self._loop.remove_reader(self._listener.fileno())
            self._listener.close()
            self._listener = None
        return acceptor

    async def _shutdown(self, acceptor):
        if self._adapter is not None:
            self._adapter.cancel()
            await tasks.gather(self._adapter, return_exceptions=True)
        if acceptor is not None:
            await tasks.gather(acceptor, return_exceptions=True)
        for worker_id, inbox in enumerate(self._inboxes):
            inbox.put(_STOP)
            self._ring(worker_id)
        for thread in self._threads:
            await self._loop.run_in_executor(None, thread.join)
        for interp in self._interps:
            interp.close()
        for inbox in self._inboxes:
            # Connections handed over as the worker stopped never reached it.
            while True:
                try:
                    fd = inbox.get_nowait()
                except interpreters.QueueEmpty:
                    break
                if fd != _STOP:
                    socket.socket(fileno=fd).close()
        for doorbell in self._doorbells:
            doorbell.close()
        for sock in self._lingering:
            sock.close()

    def _spawn(self, cpu):
        worker_id = len(self._threads)
        inbox = interpreters.create_queue(maxsize=_INBOX_CAPACITY)
        doorbell, worker_end = socket.socketpair()
        doorbell.setblocking(False)
        config = dict(self._config, worker_id=worker_id, cpu=cpu)
        interp = interpreters.create()
        interp.prepare_main(
            _asyncio_path=tuple(sys.path),
            _asyncio_config=json.dumps(config),
            _asyncio_buffer=self._buffer,
            _asyncio_inbox=inbox,
            _asyncio_doorbell=worker_end.detach(),
        )
        thread = threading.Thread(
            target=self._run_worker, args=(worker_id, interp),
            name=f'asyncio-multicore-worker-{worker_id}')
        self._interps.append(interp)
        self._threads.append(thread)
        self._inboxes.append(inbox)
        self._doorbells.append(doorbell)
        self._cpus.append(cpu)
        self._dispatched.append(0)
        self._active.append(True)
        thread.start()
        return worker_id

    def _run_worker(self, worker_id, interp):
        try:
            interp.exec(_BOOTSTRAP)
        except interpreters.ExecutionFailed as exc:
            self._failures[worker_id] = exc

    async def _wait_ready(self, worker_ids):
        """None once every given worker is ready, else what went wrong."""
        deadline = time.monotonic() + self._startup_timeout
        while True:
            states = [_read_slot(self._buffer, w, retries=100_000).state
                      for w in worker_ids]
            if (any(w in self._failures for w in worker_ids)
                    or 'failed' in states):
                for w in worker_ids:
                    await events.get_running_loop().run_in_executor(
                        None, self._threads[w].join, 1.0)
                return '; '.join(f'worker {w}: {self._failures[w]}'
                                 for w in worker_ids if w in self._failures)
            if all(state == 'ready' for state in states):
                return None
            if time.monotonic() > deadline:
                return f'not ready after {self._startup_timeout}s: {states}'
            await tasks.sleep(0.005)

    # Following the CPU budget

    async def _adapt_loop(self):
        while True:
            await tasks.sleep(self._adapt_interval)
            try:
                await self._adapt()
            except Exception as exc:
                # Adaptation is an optimisation; a failed round must not
                # take the server down.
                self._loop.call_exception_handler({
                    'message': 'multi-core server adaptation failed',
                    'exception': exc,
                })

    async def _adapt(self):
        now = time.monotonic()
        samples = []
        for worker_id in range(len(self._threads)):
            stats = _read_slot(self._buffer, worker_id)
            if stats is None or stats.state != 'ready':
                continue
            steal, seen_at = self._steal_seen.get(worker_id,
                                                  (stats.steal, now))
            self._steal_seen[worker_id] = (stats.steal, now)
            elapsed = now - seen_at
            ratio = (stats.steal - steal) / elapsed if elapsed > 0 else 0.0
            samples.append((worker_id, self._active[worker_id], ratio))
        budget = _cpu_budget()
        park, unpark, spawn = self._autoscaler.decide(samples, len(budget))
        for worker_id in park:
            self._active[worker_id] = False
        for worker_id in unpark:
            self._active[worker_id] = True
        for _ in range(min(spawn, self._capacity - len(self._threads))):
            free = [cpu for cpu in budget if cpu not in self._cpus]
            if self._pin and not free:
                break
            worker_id = self._spawn(free[0] if self._pin else -1)
            failure = await self._wait_ready([worker_id])
            if failure:
                self._active[worker_id] = False
                self._loop.call_exception_handler({
                    'message': f'multi-core server could not add a worker: '
                               f'{failure}',
                })

    # Accepting and routing

    async def _accept_loop(self):
        listener = self._listener
        while True:
            if self._overload == 'queue':
                await self._wait_for_capacity()
            try:
                sock, _ = await self._loop.sock_accept(listener)
            except OSError as exc:
                self._loop.call_exception_handler({
                    'message': 'multi-core server accept failed',
                    'exception': exc,
                })
                await tasks.sleep(0.01)
                continue
            try:
                self._dispatch(sock)
            except Exception as exc:
                # One bad connection must never stop the acceptor.
                sock.close()
                self._loop.call_exception_handler({
                    'message': 'multi-core server could not route a '
                               'connection',
                    'exception': exc,
                })

    async def _wait_for_capacity(self):
        # While every worker is full, stop accepting: further clients wait
        # in the kernel's accept queue instead of being refused.
        if self._pick(peek=True) is not None:
            return
        started = time.monotonic()
        while self._pick(peek=True) is None:
            await tasks.sleep(_CAPACITY_POLL)
        self._paused += time.monotonic() - started

    def _dispatch(self, sock):
        worker_id = self._pick()
        if worker_id is None and self._overload == 'queue':
            worker_id = self._pick(ignore_capacity=True)
        if worker_id is None:
            self._router_shed += 1
            _shed(sock, self._shed_response, self._loop, self._lingering)
            return None
        fd = sock.detach()
        try:
            self._inboxes[worker_id].put_nowait(fd)
        except interpreters.QueueFull:
            self._router_shed += 1
            _shed(socket.socket(fileno=fd), self._shed_response, self._loop,
                  self._lingering)
            return None
        self._dispatched[worker_id] += 1
        self._ring(worker_id)
        return worker_id

    def _ring(self, worker_id):
        try:
            self._doorbells[worker_id].send(b'\x01')
        except BlockingIOError:
            # Full of unread rings: the worker is already due to wake.
            pass
        except OSError:
            # The worker has exited; shutdown closes what is in its inbox.
            pass

    def _pick(self, *, ignore_capacity=False, peek=False):
        now = time.monotonic()
        healthy = fallback = None
        threshold = self._stall_threshold
        count = len(self._threads)
        for offset in range(count):
            worker_id = (self._next + offset) % count
            if not self._active[worker_id]:
                continue
            stats = _read_slot(self._buffer, worker_id)
            if stats is None or stats.state != 'ready':
                continue
            limit = stats.limit or _UNLIMITED
            outstanding = (self._dispatched[worker_id] - stats.completed
                           - stats.shed)
            capacity = (_UNLIMITED if ignore_capacity or not stats.limit
                        else limit + self._worker_queue)
            if outstanding >= capacity:
                continue
            if self._routing == 'round_robin':
                if not peek:
                    self._next = (worker_id + 1) % count
                return worker_id
            # Queue depth dominates; recent loop lag scales it, so lag
            # jitter below the stall threshold cannot outweigh a request.
            score = ((outstanding + 1) / limit
                     * (1 + stats.loop_lag / threshold))
            candidate = (score, worker_id)
            if now - stats.heartbeat <= self._stale_after:
                healthy = min(healthy, candidate) if healthy else candidate
            else:
                fallback = min(fallback, candidate) if fallback else candidate
        if not peek:
            self._next = (self._next + 1) % max(1, count)
        chosen = healthy or fallback
        return chosen[1] if chosen else None

"""Tests for asyncio.multicore (asyncio.start_server(..., workers=N))."""

import asyncio
import os
import socket
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from test import support
from test.support import import_helper
from test.support import script_helper
from test.support import socket_helper

if support.Py_GIL_DISABLED:
    raise unittest.SkipTest('not yet validated on the free-threaded build')
import_helper.import_module('_interpreters')

from asyncio import multicore
from concurrent import interpreters
from test.test_asyncio import multicore_handlers as handlers

HOST = socket_helper.HOST
EXACT_3 = multicore.ConcurrencyLimit(3, 3, 3)
EXACT_1 = multicore.ConcurrencyLimit(1, 1, 1)


def tearDownModule():
    asyncio.set_event_loop(None)


async def request(address, line):
    started = time.monotonic()
    reader, writer = await asyncio.open_connection(*address)
    try:
        writer.write(line)
        await writer.drain()
        reply = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    return reply, time.monotonic() - started


def address(server):
    return server.sockets[0].getsockname()[:2]


async def wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f'condition not met within {timeout}s')
        await asyncio.sleep(0.02)


class LimiterTests(unittest.TestCase):

    def limiter(self, limit=100, in_flight=60):
        limiter = multicore._VegasLimiter(multicore.ConcurrencyLimit(limit))
        limiter.rtt_noload = 0.001
        limiter.in_flight = in_flight
        return limiter

    def rtt_for_queue(self, limit, queue):
        return 0.001 / (1 - queue / limit)

    def test_no_queue_grows_by_beta(self):
        limiter = self.limiter()
        limiter.release(0.001)
        self.assertAlmostEqual(limiter.limit, 112)

    def test_small_queue_grows_by_log(self):
        limiter = self.limiter()
        limiter.release(self.rtt_for_queue(100, 4))
        self.assertAlmostEqual(limiter.limit, 102)

    def test_queue_between_alpha_and_beta_holds(self):
        limiter = self.limiter()
        limiter.release(self.rtt_for_queue(100, 9))
        self.assertEqual(limiter.limit, 100)

    def test_queue_above_beta_shrinks_by_log(self):
        limiter = self.limiter()
        limiter.release(self.rtt_for_queue(100, 20))
        self.assertAlmostEqual(limiter.limit, 98)

    def test_a_lightly_used_limit_does_not_move(self):
        limiter = self.limiter(in_flight=10)
        limiter.release(self.rtt_for_queue(100, 50))
        self.assertEqual(limiter.limit, 100)

    def test_drop_and_backoff_are_multiplicative_and_bounded(self):
        limiter = self.limiter()
        limiter.release(0.001, dropped=True)
        self.assertEqual(limiter.limit, 90)
        for _ in range(100):
            limiter.backoff()
        self.assertEqual(limiter.effective_limit, 2)

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            multicore.ConcurrencyLimit(10, 20, 30)
        with self.assertRaises(ValueError):
            multicore.ConcurrencyLimit(backoff=1.5)


class TelemetryTests(unittest.TestCase):

    def test_records_round_trip(self):
        buffer = memoryview(bytearray(multicore._STRIDE * 2))
        multicore._Slot(buffer, 1).publish(multicore._READY, in_flight=3,
                                           queued=2, limit=40, cpu=-1,
                                           completed=7, steal=0.5)
        stats = multicore._read_slot(buffer, 1)
        self.assertEqual((stats.state, stats.in_flight, stats.queued,
                          stats.limit, stats.cpu, stats.completed,
                          stats.steal),
                         ('ready', 3, 2, 40, -1, 7, 0.5))
        self.assertEqual(multicore._read_slot(buffer, 0).state, 'starting')

    def test_a_torn_record_is_never_returned(self):
        buffer = memoryview(bytearray(multicore._STRIDE))
        multicore._Slot(buffer, 0).publish(multicore._READY, limit=40)
        buffer[10] ^= 0xFF
        self.assertIsNone(multicore._read_slot(buffer, 0))

    def test_records_written_by_another_interpreter_are_shared(self):
        buffer = memoryview(bytearray(multicore._STRIDE))
        interp = interpreters.create()
        try:
            interp.prepare_main(buffer=buffer)
            interp.exec(textwrap.dedent('''
                from asyncio import multicore
                multicore._Slot(buffer, 0).publish(multicore._READY,
                                                   limit=9)
            '''))
        finally:
            interp.close()
        self.assertEqual(multicore._read_slot(buffer, 0).limit, 9)


class CpuBudgetTests(unittest.TestCase):

    def cgroup(self, files, proc):
        root = tempfile.mkdtemp()
        self.addCleanup(support.os_helper.rmtree, root)
        for name, text in files.items():
            path = os.path.join(root, 'fs', name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(text)
        proc_path = os.path.join(root, 'proc')
        with open(proc_path, 'w') as f:
            f.write(proc)
        return os.path.join(root, 'fs'), proc_path

    def test_cgroup_v2_quota(self):
        self.assertEqual(multicore._cpu_quota(*self.cgroup(
            {'cpu.max': '200000 100000\n'}, '0::/\n')), 2.0)
        self.assertIsNone(multicore._cpu_quota(*self.cgroup(
            {'cpu.max': 'max 100000\n'}, '0::/\n')))

    def test_tightest_limit_along_the_cgroup_path_wins(self):
        self.assertEqual(multicore._cpu_quota(*self.cgroup(
            {'app/cpu.max': '150000 100000\n',
             'app/web/cpu.max': 'max 100000\n'}, '0::/app/web\n')), 1.5)

    def test_cgroup_v1_quota(self):
        self.assertEqual(multicore._cpu_quota(*self.cgroup(
            {'cpu,cpuacct/cpu.cfs_quota_us': '50000\n',
             'cpu,cpuacct/cpu.cfs_period_us': '100000\n'},
            '4:cpu,cpuacct:/\n')), 0.5)

    def test_no_cgroup_information(self):
        self.assertIsNone(multicore._cpu_quota('/nonexistent', '/nonexistent'))

    def test_budget_is_capped_by_the_quota(self):
        with mock.patch.object(multicore, '_physical_cpus',
                               return_value=[0, 1, 2, 3]):
            with mock.patch.object(multicore, '_cpu_quota', return_value=2.5):
                self.assertEqual(multicore._cpu_budget(), [0, 1, 2])
            with mock.patch.object(multicore, '_cpu_quota', return_value=0.1):
                self.assertEqual(multicore._cpu_budget(), [0])


class AutoscalerTests(unittest.TestCase):

    def test_steal_parking_is_off_by_default(self):
        scaler = multicore._Autoscaler(park_on_steal=False)
        for _ in range(5):
            samples = [(0, True, 0.9), (1, True, 0.0)]
            self.assertEqual(scaler.decide(samples, 2), (set(), set(), 0))

    def test_sustained_steal_parks_and_quiet_unparks(self):
        scaler = multicore._Autoscaler(park_on_steal=True)
        stolen = [(0, True, 0.6), (1, True, 0.0)]
        self.assertEqual(scaler.decide(stolen, 2), (set(), set(), 0))
        self.assertEqual(scaler.decide(stolen, 2), ({0}, set(), 0))
        quiet = [(0, False, 0.0), (1, True, 0.0)]
        self.assertEqual(scaler.decide(quiet, 2), (set(), set(), 0))
        self.assertEqual(scaler.decide(quiet, 2), (set(), set(), 0))
        self.assertEqual(scaler.decide(quiet, 2), (set(), {0}, 0))

    def test_budget_changes(self):
        scaler = multicore._Autoscaler(park_on_steal=False)
        samples = [(0, True, 0.0), (1, True, 0.2), (2, True, 0.0)]
        self.assertEqual(scaler.decide(samples, 1), ({1, 2}, set(), 0))
        self.assertEqual(scaler.decide(samples, 5), (set(), set(), 2))


class ServerTests(unittest.TestCase):

    def run_server(self, scenario, callback, **kwargs):
        async def main():
            server = await asyncio.start_server(callback, HOST, 0, **kwargs)
            async with server:
                return await scenario(server)

        return asyncio.run(main())

    def test_workers_are_interpreters_and_threads_in_one_process(self):
        async def scenario(server):
            replies = await asyncio.gather(
                *(request(address(server), b'?\n') for _ in range(24)))
            return server, [reply.split() for reply, _ in replies]

        server, rows = self.run_server(scenario, handlers.whoami, workers=3,
                                       routing='round_robin')
        self.assertIsInstance(server, asyncio.AbstractServer)
        self.assertIsInstance(server, multicore.MulticoreServer)
        self.assertEqual(len({row[0] for row in rows}), 3)
        self.assertEqual(len({row[1] for row in rows}), 3)
        self.assertEqual(len({row[2] for row in rows}), 3)
        self.assertNotIn(str(interpreters.get_main().id).encode(),
                         {row[1] for row in rows})
        self.assertEqual({row[3] for row in rows}, {str(os.getpid()).encode()})
        self.assertFalse(server.is_serving())

    def test_many_clients_each_get_their_own_reply(self):
        async def scenario(server):
            lines = [f'client {i}\n'.encode() for i in range(200)]
            replies = await asyncio.gather(
                *(request(address(server), line) for line in lines))
            return (lines, [reply for reply, _ in replies],
                    server.worker_stats())

        lines, replies, stats = self.run_server(scenario, handlers.echo,
                                                workers=2)
        self.assertEqual(replies, lines)
        self.assertEqual(sum(s.completed for s in stats), 200)

    def test_a_plain_function_callback_works(self):
        async def scenario(server):
            # Nothing is sent: the callback never reads, and closing a
            # connection with unread data resets it.
            reply, _ = await request(address(server), b'')
            return reply

        self.assertEqual(self.run_server(scenario, handlers.plain_echo,
                                         workers=1), b'plain\n')

    def test_excess_connections_wait_and_all_complete(self):
        async def scenario(server):
            started = time.monotonic()
            replies = await asyncio.gather(
                *(request(address(server), b'x\n') for _ in range(10)))
            return ([int(reply) for reply, _ in replies],
                    time.monotonic() - started, server.worker_stats()[0])

        peaks, elapsed, stats = self.run_server(
            scenario, handlers.peak_concurrency, workers=1,
            concurrency=EXACT_3)
        self.assertEqual(len(peaks), 10)
        self.assertEqual(max(peaks), 3)
        self.assertGreaterEqual(elapsed, 0.75)
        self.assertEqual((stats.shed, stats.queued), (0, 0))

    def test_the_acceptor_pauses_when_every_worker_is_full(self):
        async def scenario(server):
            replies = await asyncio.gather(
                *(request(address(server), b'x\n') for _ in range(9)))
            return [int(reply) for reply, _ in replies], server._paused

        peaks, paused = self.run_server(
            scenario, handlers.peak_concurrency, workers=1,
            concurrency=EXACT_3, worker_queue=1)
        self.assertEqual(len(peaks), 9)
        self.assertEqual(max(peaks), 3)
        self.assertGreater(paused, 0)

    def test_shed_mode_rejects_only_the_excess(self):
        async def scenario(server):
            replies = await asyncio.gather(
                *(request(address(server), b'x\n') for _ in range(10)))
            return [reply for reply, _ in replies]

        replies = self.run_server(scenario, handlers.slow, workers=1,
                                  concurrency=EXACT_3, overload='shed',
                                  shed_response=b'BUSY\n')
        self.assertEqual(replies.count(b'done\n'), 3)
        self.assertEqual(replies.count(b'BUSY\n'), 7)

    def test_max_queue_wait_sheds_only_what_waited_too_long(self):
        async def scenario(server):
            first = asyncio.create_task(request(address(server), b'x\n'))
            await asyncio.sleep(0.05)
            late = await asyncio.gather(
                *(request(address(server), b'x\n') for _ in range(2)))
            return (await first)[0], late

        first, late = self.run_server(
            scenario, handlers.slow, workers=1, concurrency=EXACT_1,
            max_queue_wait=0.1, shed_response=b'BUSY\n')
        self.assertEqual(first, b'done\n')
        self.assertEqual([reply for reply, _ in late], [b'BUSY\n'] * 2)

    def test_request_timeout_backs_the_limit_off(self):
        async def scenario(server):
            reply, latency = await request(address(server), b'x\n')
            await asyncio.sleep(0.05)
            return reply, latency, server.worker_stats()[0]

        reply, latency, stats = self.run_server(
            scenario, handlers.slow, workers=1, request_timeout=0.05,
            concurrency=multicore.ConcurrencyLimit(20))
        self.assertEqual(reply, b'')
        self.assertLess(latency, 0.19)
        self.assertEqual(stats.limit, 18)

    def test_a_crashing_callback_does_not_stop_its_worker(self):
        async def scenario(server):
            crashed, _ = await request(address(server), b'crash\n')
            echoed, _ = await request(address(server), b'still here\n')
            return crashed, echoed

        crashed, echoed = self.run_server(
            scenario, handlers.crash_or_echo, workers=1,
            on_start=handlers.ignore_loop_errors)
        self.assertEqual((crashed, echoed), (b'', b'still here\n'))

    def test_a_frozen_worker_gets_no_new_connections(self):
        async def scenario(server):
            blocked = asyncio.create_task(
                request(address(server), b'BLOCK 0.5\n'))
            await asyncio.sleep(0.1)
            quick = await asyncio.gather(
                *(request(address(server), b'go\n') for _ in range(10)))
            frozen = int((await blocked)[0])
            await asyncio.sleep(0.05)
            return frozen, quick, server.worker_stats()[frozen]

        frozen, quick, stats = self.run_server(
            scenario, handlers.block_on_command, workers=2)
        self.assertNotIn(frozen, [int(reply) for reply, _ in quick])
        self.assertGreaterEqual(stats.stalls, 1)

    def test_a_busy_loop_is_not_a_stall(self):
        async def scenario(server):
            for _ in range(8):
                await request(address(server), b'2000000\n')
            await asyncio.sleep(0.05)
            return server.worker_stats()[0]

        stats = self.run_server(scenario, handlers.cpu_bound, workers=1,
                                concurrency=multicore.ConcurrencyLimit(50))
        self.assertGreaterEqual(stats.max_loop_lag, 0.03)
        # Treating lag as a stall would count 8 or more; real preemption on
        # a loaded machine may add one or two.
        self.assertLessEqual(stats.stalls, 2)

    @unittest.skipIf(sys.platform == 'darwin',
                     'macOS has no thread-affinity API')
    def test_each_worker_thread_is_pinned_to_its_own_cpu(self):
        if len(multicore._available_cpus()) < 2:
            self.skipTest('needs two CPUs')

        async def scenario(server):
            replies = await asyncio.gather(
                *(request(address(server), b'?\n') for _ in range(10)))
            return {reply.split()[2]: reply.split()[4:6]
                    for reply, _ in replies}

        by_thread = self.run_server(scenario, handlers.whoami, workers=2,
                                    routing='round_robin')
        self.assertEqual(len(by_thread), 2)
        for cpu, affinity in by_thread.values():
            self.assertEqual(affinity, cpu)


class LifecycleTests(unittest.TestCase):

    def test_close_refuses_new_connections_and_finishes_accepted_ones(self):
        async def main():
            server = await asyncio.start_server(handlers.slow, HOST, 0,
                                                workers=1)
            where = address(server)
            in_flight = asyncio.create_task(request(where, b'x\n'))
            await asyncio.sleep(0.05)
            server.close()
            with self.assertRaises(ConnectionRefusedError):
                await request(where, b'x\n')
            await server.wait_closed()
            return (await in_flight)[0]

        self.assertEqual(asyncio.run(main()), b'done\n')

    def test_serve_forever_cancellation_closes_and_cleans_up(self):
        threads_before = threading.active_count()

        async def main():
            server = await asyncio.start_server(handlers.echo, HOST, 0,
                                                workers=2)
            forever = asyncio.create_task(server.serve_forever())
            await asyncio.sleep(0.05)
            reply, _ = await request(address(server), b'hi\n')
            forever.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await forever
            return reply, server

        reply, server = asyncio.run(main())
        self.assertEqual(reply, b'hi\n')
        self.assertFalse(server.is_serving())
        self.assertEqual(server.sockets, ())
        self.assertEqual(threading.active_count(), threads_before)
        self.assertEqual(interpreters.list_all(), [interpreters.get_main()])

    def test_start_serving_false_holds_clients_until_start_serving(self):
        async def main():
            server = await asyncio.start_server(handlers.echo, HOST, 0,
                                                workers=1,
                                                start_serving=False)
            async with server:
                pending = asyncio.create_task(
                    request(address(server), b'later\n'))
                await asyncio.sleep(0.2)
                early = pending.done()
                await server.start_serving()
                return early, (await pending)[0]

        self.assertEqual(asyncio.run(main()), (False, b'later\n'))

    def test_a_worker_failing_to_start_is_reported_and_cleaned_up(self):
        async def main():
            await asyncio.start_server(handlers.echo, HOST, 0, workers=2,
                                       on_start=handlers.fail_on_start)

        with support.captured_stderr():
            with self.assertRaisesRegex(RuntimeError,
                                        'worker 1 refuses to start'):
                asyncio.run(main())
        self.assertEqual(interpreters.list_all(), [interpreters.get_main()])

    def test_callbacks_that_cannot_be_imported_by_name_are_rejected(self):
        for callback in (lambda reader, writer: None,
                         (lambda: lambda reader, writer: None)(),
                         handlers.Greeter().handle):
            with self.subTest(callback=callback):
                async def main():
                    await asyncio.start_server(callback, HOST, 0, workers=1)

                with self.assertRaisesRegex(TypeError,
                                            'module-level function'):
                    asyncio.run(main())
        self.assertEqual(interpreters.list_all(), [interpreters.get_main()])

    def test_invalid_options_are_rejected(self):
        for kwargs in ({'workers': 0}, {'workers': 'many'},
                       {'overload': 'drop'}, {'routing': 'random'}):
            with self.subTest(**kwargs):
                async def main():
                    await asyncio.start_server(handlers.echo, HOST, 0,
                                               **{'workers': 1, **kwargs})

                with self.assertRaises(ValueError):
                    asyncio.run(main())

    def test_without_workers_start_server_is_unchanged(self):
        async def main():
            server = await asyncio.start_server(handlers.echo, HOST, 0)
            async with server:
                return type(server), (await request(address(server),
                                                    b'single\n'))[0]

        server_type, reply = asyncio.run(main())
        self.assertIs(server_type, asyncio.base_events.Server)
        self.assertEqual(reply, b'single\n')

    def test_current_worker_outside_a_worker(self):
        with self.assertRaises(RuntimeError):
            multicore.current_worker()

    def test_a_callback_defined_in_the_main_script(self):
        script = textwrap.dedent('''
            import asyncio
            print('top level', flush=True)

            async def handle(reader, writer):
                line = await reader.readline()
                writer.write(__name__.encode() + b' ' + line)
                await writer.drain()

            async def main():
                server = await asyncio.start_server(handle, '127.0.0.1', 0,
                                                    workers=2)
                async with server:
                    host, port = server.sockets[0].getsockname()[:2]
                    reader, writer = await asyncio.open_connection(host, port)
                    writer.write(b'hi\\n')
                    print('REPLY', (await reader.read()).decode().strip())
                    writer.close()
                    await writer.wait_closed()

            if __name__ == '__main__':
                print('main block', flush=True)
                asyncio.run(main())
        ''')
        with tempfile.TemporaryDirectory() as directory:
            path = script_helper.make_script(directory, 'app', script)
            _, out, err = script_helper.assert_python_ok(path)
        out = out.decode()
        self.assertEqual(out.count('main block'), 1)
        self.assertEqual(out.count('top level'), 3)
        self.assertIn('REPLY __asyncio_multicore_main__ hi', out)


class AdaptationTests(unittest.TestCase):

    async def workers_serving(self, server, count):
        replies = await asyncio.gather(
            *(request(address(server), b'?\n') for _ in range(count)))
        return {reply.split()[0] for reply, _ in replies}

    def test_the_worker_count_follows_the_cpu_budget(self):
        budget = [0, 1]

        async def main():
            server = await asyncio.start_server(
                handlers.whoami, HOST, 0, workers='auto', pin=False,
                adapt_interval=0.05)
            async with server:
                budget[:] = [0]
                await wait_until(
                    lambda: server.active_workers().count(True) == 1, 2.0)
                shrunk = await self.workers_serving(server, 8)
                budget[:] = [0, 1, 2]
                await wait_until(
                    lambda: server.active_workers() == [True] * 3, 5.0)
                grown = await self.workers_serving(server, 30)
                return shrunk, grown

        with mock.patch.object(multicore, '_cpu_budget',
                               side_effect=lambda: list(budget)):
            shrunk, grown = asyncio.run(main())
        self.assertEqual(len(shrunk), 1)
        self.assertEqual(grown, {b'0', b'1', b'2'})

    def test_a_worker_whose_cpu_is_taken_is_parked_only_when_asked(self):
        async def main(park_on_steal):
            server = await asyncio.start_server(
                handlers.block_on_command, HOST, 0, workers='auto',
                pin=False, adapt_interval=0.1, park_on_steal=park_on_steal,
                on_start=handlers.take_worker_zeros_cpu)
            async with server:
                if not park_on_steal:
                    await asyncio.sleep(0.8)
                    return server.active_workers(), None
                await wait_until(lambda: not server.active_workers()[0], 2.0)
                served = [int(reply) for reply, _ in await asyncio.gather(
                    *(request(address(server), b'go\n') for _ in range(10)))]
                await wait_until(lambda: server.active_workers()[0], 5.0)
                return server.active_workers(), served

        with mock.patch.object(multicore, '_cpu_budget',
                               return_value=[0, 1, 2]):
            active, served = asyncio.run(main(park_on_steal=True))
            self.assertNotIn(0, served)
            self.assertEqual(active, [True] * 3)
            active, _ = asyncio.run(main(park_on_steal=False))
            self.assertEqual(active, [True] * 3)

    @support.requires_resource('cpu')
    def test_cpu_bound_callbacks_run_in_parallel(self):
        if len(multicore._available_cpus()) < 4:
            self.skipTest('needs four CPUs')

        async def elapsed(workers):
            server = await asyncio.start_server(handlers.cpu_bound, HOST, 0,
                                                workers=workers)
            async with server:
                await request(address(server), b'10\n')
                started = time.monotonic()
                await asyncio.gather(*(request(address(server), b'1000000\n')
                                       for _ in range(8)))
                return time.monotonic() - started

        one = asyncio.run(elapsed(1))
        four = asyncio.run(elapsed(4))
        self.assertGreater(one / four, 1.5)


if __name__ == '__main__':
    unittest.main()

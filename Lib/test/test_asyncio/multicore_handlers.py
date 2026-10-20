"""Client callbacks for test_multicore, imported by the worker interpreters."""

import asyncio
import os
import threading
import time
from asyncio import multicore
from concurrent import interpreters


async def echo(reader, writer):
    writer.write(await reader.readline())
    await writer.drain()
    # Plain asyncio.start_server() leaves the connection open otherwise.
    writer.close()
    await writer.wait_closed()


def plain_echo(reader, writer):
    # asyncio.start_server() also accepts a plain function.
    writer.write(b'plain\n')


async def whoami(reader, writer):
    await reader.readline()
    worker = multicore.current_worker()
    interp = interpreters.get_current()
    cpus = multicore._current_thread_cpus()
    affinity = ','.join(map(str, sorted(cpus))) if cpus is not None else '-'
    writer.write(f'{worker.worker_id} {interp.id} {threading.get_native_id()} '
                 f'{os.getpid()} {worker.cpu} {affinity}\n'.encode())
    await writer.drain()


async def slow(reader, writer):
    await reader.readline()
    await asyncio.sleep(0.2)
    writer.write(b'done\n')
    await writer.drain()


async def cpu_bound(reader, writer):
    n = int(await reader.readline())
    total = 0
    for i in range(n):
        total += i * i
    writer.write(f'{total}\n'.encode())
    await writer.drain()


async def block_on_command(reader, writer):
    line = await reader.readline()
    if line.startswith(b'BLOCK'):
        time.sleep(float(line.split()[1]))
    writer.write(f'{multicore.current_worker().worker_id}\n'.encode())
    await writer.drain()


async def crash_or_echo(reader, writer):
    line = await reader.readline()
    if line == b'crash\n':
        raise ValueError('boom')
    writer.write(line)
    await writer.drain()


_active = 0
_peak = 0


async def peak_concurrency(reader, writer):
    # Replies with the most requests this worker ever ran at once.
    global _active, _peak
    await reader.readline()
    _active += 1
    _peak = max(_peak, _active)
    try:
        await asyncio.sleep(0.2)
    finally:
        _active -= 1
    writer.write(f'{_peak}\n'.encode())
    await writer.drain()


def ignore_loop_errors(worker):
    # The crash test's error is expected; keep it out of the test output.
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda loop, context: None)


def fail_on_start(worker):
    if worker.worker_id == 1:
        raise RuntimeError('worker 1 refuses to start')


def take_worker_zeros_cpu(worker):
    # Keeps worker 0 off its CPU 15 ms in every 20 ms for about a second.
    if worker.worker_id != 0:
        return
    loop = asyncio.get_running_loop()
    until = loop.time() + 1.0

    def block():
        time.sleep(0.015)
        if loop.time() < until:
            loop.call_later(0.005, block)

    loop.call_soon(block)


class Greeter:
    async def handle(self, reader, writer):
        writer.write(b'hello\n')

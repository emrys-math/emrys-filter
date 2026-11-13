"""Times the compiles of the compile-cost units (docs/design/benchmarks.md, claim 7).

    python tools/compile_cost.py <build directory> [--repeat N] [--time-trace] [--match REGEX]
                                 [--json FILE]

Reads the units from compile_cost_units.json in the build directory, which the benchmarks'
CMakeLists.txt writes, and their commands from compile_commands.json. Runs each command --repeat
times and keeps the least CPU time and wall time, since a busy machine only adds to them, and the
largest peak memory. The ratios divide a unit's times by its reference's: an EmrysFilter unit's
plain-Eigen partner, or for an umbrella header, Eigen/Dense alone. MSVC generates code on several
threads, so its CPU time exceeds its wall time, and its wall ratio is the one a user waits through.

With --match, it times every compile command whose source path matches REGEX instead, such as the
slowest tests. With --time-trace, a Clang build also compiles each unit once more under -ftime-trace
and reports its frontend, function instantiation, and backend seconds; that run isn't timed.

Peak memory is the largest resident set of the compiler's processes on Linux, and the largest
commit charge of one of them on Windows, so it compares within a platform, not across. The
numbers only mean something on a quiet machine. Prints a Markdown table, and writes every number,
with the commit, compiler, and machine, to compile_cost.json in the build directory or to --json.
"""
import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys
import time


def run_measured(command, directory):
    """Runs a shell command; returns its exit status, output, wall and CPU seconds, peak bytes."""
    if os.name == 'nt':
        return _run_windows(command, directory)
    start = time.perf_counter()
    process = subprocess.Popen(
        command, shell=True, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = process.stdout.read()
    # wait4 reports the shell's usage with that of every descendant it waited for, the compiler's.
    _, status, usage = os.wait4(process.pid, 0)
    wall = time.perf_counter() - start
    process.returncode = os.waitstatus_to_exitcode(status)
    peak = usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    return process.returncode, output, wall, usage.ru_utime + usage.ru_stime, peak


def _run_windows(command, directory):
    """run_measured() for Windows: the command runs inside a job object, whose accounting covers
    every process it starts, such as GCC's cc1plus. It starts suspended, so that no child can start
    before the job holds it."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'reads', 'writes', 'others', 'readBytes', 'writeBytes', 'otherBytes')]

    class BasicLimit(ctypes.Structure):
        _fields_ = [('perProcessUserTime', ctypes.c_int64), ('perJobUserTime', ctypes.c_int64),
                    ('flags', wintypes.DWORD), ('minimumWorkingSet', ctypes.c_size_t),
                    ('maximumWorkingSet', ctypes.c_size_t), ('activeProcesses', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priorityClass', wintypes.DWORD),
                    ('schedulingClass', wintypes.DWORD)]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [('basic', BasicLimit), ('io', IoCounters), ('processMemory', ctypes.c_size_t),
                    ('jobMemory', ctypes.c_size_t), ('peakProcessMemory', ctypes.c_size_t),
                    ('peakJobMemory', ctypes.c_size_t)]

    class Accounting(ctypes.Structure):
        _fields_ = [('userTime', ctypes.c_int64), ('kernelTime', ctypes.c_int64),
                    ('periodUserTime', ctypes.c_int64), ('periodKernelTime', ctypes.c_int64),
                    ('pageFaults', wintypes.DWORD), ('processes', wintypes.DWORD),
                    ('activeProcesses', wintypes.DWORD), ('terminatedProcesses', wintypes.DWORD)]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll = ctypes.WinDLL('ntdll')
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    basicAccounting, extendedLimit = 1, 9
    killOnClose, createSuspended = 0x2000, 0x4
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limit = ExtendedLimit()
        limit.basic.flags = killOnClose  # an interrupted run leaves no compiler behind
        kernel32.SetInformationJobObject(job, extendedLimit, ctypes.byref(limit), ctypes.sizeof(limit))
        start = time.perf_counter()
        # The command line goes to CreateProcess as it is, without cmd.exe and its length limit.
        process = subprocess.Popen(
            command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=createSuspended
        )
        if not kernel32.AssignProcessToJobObject(job, int(process._handle)):
            process.kill()
            raise ctypes.WinError(ctypes.get_last_error())
        ntdll.NtResumeProcess(int(process._handle))
        output, _ = process.communicate()
        wall = time.perf_counter() - start
        accounting = Accounting()
        kernel32.QueryInformationJobObject(
            job, basicAccounting, ctypes.byref(accounting), ctypes.sizeof(accounting), None)
        kernel32.QueryInformationJobObject(
            job, extendedLimit, ctypes.byref(limit), ctypes.sizeof(limit), None)
    finally:
        kernel32.CloseHandle(job)
    cpu = (accounting.userTime + accounting.kernelTime) * 1e-7  # 100 ns ticks
    return process.returncode, output, wall, cpu, limit.peakProcessMemory


def compiler_of(command):
    """The compiler in a compile command, and the first line it gives of its version."""
    compiler = shlex.split(command, posix=os.name != 'nt')[0].strip('"')
    name = pathlib.Path(compiler).stem.lower()
    # cl prints its version to standard error, and only when run with no arguments.
    if name == 'cl':
        lines = subprocess.run([compiler], capture_output=True, text=True).stderr.strip().splitlines()
    else:
        lines = subprocess.run([compiler, '--version'], capture_output=True, text=True).stdout.strip().splitlines()
    return compiler, lines[0] if lines else name


def trace_totals(trace):
    """Frontend, function instantiation, and backend seconds from a Clang -ftime-trace file. Each
    of Clang's totals counts an event nested in one of its own name once; function instantiation is
    part of the frontend, and class instantiation, inside both parsing and function instantiation,
    isn't reported."""
    totals = {}
    for event in json.loads(trace.read_text(encoding='utf-8'))['traceEvents']:
        name = event.get('name', '')
        if name.startswith('Total '):
            totals[name[len('Total '):]] = event['dur'] * 1e-6
    return {
        'frontend': totals.get('Frontend', 0.0),
        'instantiation': totals.get('InstantiateFunction', 0.0),
        'backend': totals.get('Backend', 0.0),
    }


def git_commit(directory):
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=directory,
                            capture_output=True, text=True)
    return result.stdout.strip() or 'unknown'


def cache_value(build, name):
    match = re.search(rf'^{name}:\w+=(.*)$', (build / 'CMakeCache.txt').read_text(encoding='utf-8'),
                      re.MULTILINE)
    return match.group(1) if match else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('build', type=pathlib.Path, help='the build directory')
    parser.add_argument('--repeat', type=int, default=3, help='runs of each command (3)')
    parser.add_argument('--time-trace', action='store_true', help='add Clang -ftime-trace totals')
    parser.add_argument('--match', help='time the commands whose source matches this instead')
    parser.add_argument('--json', type=pathlib.Path, help='where to write the numbers')
    arguments = parser.parse_args()
    build = arguments.build.resolve()

    commands = json.loads((build / 'compile_commands.json').read_text(encoding='utf-8'))
    if arguments.match:
        pattern = re.compile(arguments.match)
        units = [{'label': entry['file'], 'reference': '', 'entry': entry}
                 for entry in commands if pattern.search(entry['file'].replace('\\', '/'))]
    else:
        units = json.loads((build / 'compile_cost_units.json').read_text(encoding='utf-8'))
        for unit in units:
            marker = f'/{unit["target"]}.dir/'
            found = [entry for entry in commands
                     if marker in '/' + entry.get('output', '').replace('\\', '/')]
            if len(found) != 1:
                sys.exit(f'{unit["target"]}: {len(found)} compile commands; build the target first')
            unit['entry'] = found[0]
    if not units:
        sys.exit('no compile commands to time')

    compiler, version = compiler_of(units[0]['entry']['command'])
    clang = 'clang' in version.lower() and pathlib.Path(compiler).stem.lower() != 'clang-cl'
    if arguments.time_trace and not clang:
        sys.exit('--time-trace needs Clang')

    for unit in units:
        entry = unit['entry']
        runs = []
        for _ in range(arguments.repeat):
            status, output, wall, cpu, peak = run_measured(entry['command'], entry['directory'])
            if status != 0:
                sys.stdout.write(output.decode(errors='replace'))
                sys.exit(f'{unit["label"]}: the compile failed')
            runs.append((wall, cpu, peak))
        unit['wall'] = min(run[0] for run in runs)
        unit['cpu'] = min(run[1] for run in runs)
        unit['peak'] = max(run[2] for run in runs)
        if arguments.time_trace:
            status, output, *_ = run_measured(entry['command'] + ' -ftime-trace', entry['directory'])
            if status != 0:
                sys.stdout.write(output.decode(errors='replace'))
                sys.exit(f'{unit["label"]}: the compile under -ftime-trace failed')
            objectFile = pathlib.Path(entry['directory'], entry['output'])
            unit['trace'] = trace_totals(objectFile.with_suffix('.json'))
        print(f'{unit["label"]}: {unit["cpu"]:.2f} s', file=sys.stderr)

    byTarget = {unit.get('target'): unit for unit in units}
    for unit in units:
        if unit['reference']:
            reference = byTarget[unit['reference']]
            unit['cpu ratio'] = unit['cpu'] / reference['cpu']
            unit['wall ratio'] = unit['wall'] / reference['wall']

    machine = {
        'commit': git_commit(pathlib.Path(__file__).resolve().parent),
        'compiler': version,
        'build type': cache_value(build, 'CMAKE_BUILD_TYPE'),
        'processor': platform.processor() or platform.machine(),
        'system': platform.platform(),
        'date': datetime.date.today().isoformat(),
        'runs': arguments.repeat,
    }
    print('; '.join(f'{key} {value}' for key, value in machine.items()))
    print()
    header = '| Unit | CPU s | wall s | peak MB | CPU ratio | wall ratio |'
    rule = '|---|---:|---:|---:|---:|---:|'
    if arguments.time_trace:
        header += ' frontend s | function instantiation s | backend s |'
        rule += '---:|---:|---:|'
    print(header)
    print(rule)
    for unit in units:
        ratios = ' | '.join(f'{unit[key]:.2f}' if key in unit else '' for key in ('cpu ratio', 'wall ratio'))
        row = (f'| {unit["label"]} | {unit["cpu"]:.2f} | {unit["wall"]:.2f} | '
               f'{unit["peak"] / 2**20:.0f} | {ratios} |')
        if arguments.time_trace:
            trace = unit['trace']
            row += f' {trace["frontend"]:.2f} | {trace["instantiation"]:.2f} | {trace["backend"]:.2f} |'
        print(row)

    for unit in units:
        del unit['entry']
    output = arguments.json or build / 'compile_cost.json'
    output.write_text(json.dumps({'machine': machine, 'units': units}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

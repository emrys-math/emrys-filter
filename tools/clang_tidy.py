"""Runs clang-tidy, with the checks in .clang-tidy, over the tests, examples, and benchmarks of a build
(decisions.md [static-analysis]).

The headers are analyzed through the translation units that include them, so a finding in a header
is reported once however many units reach it. A finding in a file outside the repository, which the
static analyzer makes when a path that starts in our code ends in Eigen's, is left out, and so are
the negative compile tests' files, which without a case compile only what their cases share. The
build needs only to be configured: its compile_commands.json is read, and the module map arguments
that CMake adds, which name files the build writes, are dropped. Prints each finding and a count by
check; exit status 1 when there is one.

    python tools/clang_tidy.py build/clang
    python tools/clang_tidy.py build/clang -j 2 --clang-tidy clang-tidy-21 tests/Filters
"""
import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYZED = ('tests/', 'examples/', 'benchmarks/')
MODULE_MAP = re.compile(r'\s@\S+\.modmap')
DIAGNOSTIC = re.compile(r'^(.+?):(\d+):(\d+): (warning|error): (.*) \[([\w.,-]+)\]$')


def relative(path):
    """`path` relative to the repository, with forward slashes, or None when it is outside it."""
    try:
        return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def units(build, selected):
    """The compile database's entries for the units analyzed, with the module maps dropped."""
    entries = json.loads((build / 'compile_commands.json').read_text(encoding='utf-8'))
    kept = []
    for entry in entries:
        path = relative(pathlib.Path(entry['directory'], entry['file']))
        if path is None or not path.startswith(ANALYZED) or '/_deps/' in path or path.endswith('compile_errors.cpp'):
            continue
        if selected and not any(path.startswith(prefix) for prefix in selected):
            continue
        entry = dict(entry)
        if 'command' in entry:
            entry['command'] = MODULE_MAP.sub('', entry['command'])
        else:
            entry['arguments'] = [a for a in entry['arguments'] if not re.fullmatch(r'@\S+\.modmap', a)]
        kept.append((path, entry))
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('build', type=pathlib.Path, help='a configured build directory, such as build/clang')
    parser.add_argument('paths', nargs='*', help='analyze only the units under these paths')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.environ.get('EMRYS_FILTER_TIDY_JOBS', 3)),
                        help='units analyzed at once; each takes up to 4 GB (default 3)')
    parser.add_argument('--clang-tidy', default='clang-tidy', help='the clang-tidy to run')
    arguments = parser.parse_args()

    selected = [p.replace('\\', '/').rstrip('/') for p in arguments.paths]
    kept = units(arguments.build.resolve(), selected)
    if not kept:
        print('No translation unit to analyze.')
        return 1
    database = arguments.build.resolve() / 'clang-tidy'
    database.mkdir(exist_ok=True)
    (database / 'compile_commands.json').write_text(json.dumps([e for _, e in kept], indent=1), encoding='utf-8')

    def analyze(unit):
        path, entry = unit
        start = time.monotonic()
        result = subprocess.run([arguments.clang_tidy, '--quiet', '-p', str(database), entry['file']],
                                capture_output=True, encoding='utf-8', errors='replace')
        return path, result, time.monotonic() - start

    findings = {}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(arguments.jobs) as pool:
        for count, (path, result, seconds) in enumerate(pool.map(analyze, kept), 1):
            print(f'[{count}/{len(kept)}] {path} ({seconds:.0f} s)', flush=True)
            reported = False
            for line in result.stdout.splitlines():
                match = DIAGNOSTIC.match(line)
                if not match:
                    continue
                reported = True
                where = relative(match.group(1))
                if where is None:
                    continue
                key = (where, int(match.group(2)), int(match.group(3)), match.group(6))
                findings.setdefault(key, match.group(5))
            if result.returncode != 0 and not reported:
                failed.append((path, (result.stderr.strip() or 'no output').splitlines()[-3:]))

    for (where, line, column, check), message in sorted(findings.items()):
        print(f'{where}:{line}:{column}: {message} [{check}]')
    for path, tail in failed:
        print(f'{path}: clang-tidy failed: {" / ".join(tail)}')
    counts = collections.Counter(check for *_, check in findings)
    if counts:
        print(f'{len(findings)} findings:')
        for check, number in counts.most_common():
            print(f'  {number:5}  {check}')
    else:
        print(f'No findings in {len(kept)} translation units.')
    return 1 if findings or failed else 0


if __name__ == '__main__':
    sys.exit(main())

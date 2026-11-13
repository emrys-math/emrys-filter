"""Lists the checks a change needs at a stage of its way to a release (decisions.md [check-stages]).

The stages are `land`, before work goes onto `next`, and `pr`, before `next` goes to `dev` in a pull
request; the CI jobs not run here run at `ci`, when the user dispatches them. A check is required
at its own stage and every later one when the change touches anything it tests, and at an earlier
stage when the change touches something only it tests: a deduction guide brings MSVC to `land`, a
claimed compile error Clang's negative compile tests, and a `throw` the build without exceptions to
`pr`. The change is the diff from the merge base
with the stage's branch to the working tree, untracked files included. Prints each required check,
its reason, and its command, then the checks a later stage adds.

    python tools/required_checks.py land        # against next
    python tools/required_checks.py pr          # against origin/dev
    python tools/required_checks.py pr --base <branch>
"""
import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STAGES = ['land', 'pr', 'ci']
BASES = {'land': 'next', 'pr': 'origin/dev'}

COMPILED = r'^(include|tests|examples|benchmarks|skills|cmake)/|(^|/)CMakeLists\.txt$|^CMakePresets\.json$'
HEADERS = r'^include/'
SOURCES = r'\.(h|cpp)$'


@dataclasses.dataclass
class Trigger:
    """Brings a check to `stage` when a changed path matches `path`, or a changed line of a path
    matching `within` matches `line`."""
    stage: str
    reason: str
    path: str | None = None
    line: str | None = None
    within: str = HEADERS


@dataclasses.dataclass
class Check:
    name: str
    stage: str
    command: str
    tests: str  # the paths whose change the check tests at all
    triggers: list[Trigger] = dataclasses.field(default_factory=list)


GUIDES = Trigger('land', 'a deduction guide changed, which MSVC deduces differently',
                 line=r'\)\s*->\s*[A-Z]\w*\s*<|^\s*->\s*[A-Z]\w*\s*<')
THREADS = Trigger('land', 'execution policies changed, which only MSVC runs on threads here',
                  path=r'^include/EmrysFilter/Core/Execution\.h$|^tests/(parallel|Core/execution)\.cpp$',
                  line=r'std::execution|\bpar(_unseq)?\b')
CLAIMS = Trigger('land', 'a claimed compile error changed', path=r'^tests/(\w+/)?compile_errors\.(cpp|cmake)$',
                 line=r'static_assert|=\s*delete\b')
# What the negative compile tests check: a compiler's result, so the headers, the cases, and the
# build, and no sanitizer or coverage flag.
COMPILE_ERRORS = (HEADERS + r'|^tests/(\w+/)?compile_errors\.(cpp|cmake)$|^tests/CMakeLists\.txt$'
                  r'|^CMakeLists\.txt$|^CMakePresets\.json$|^cmake/')

CHECKS = [
    Check('clang-format', 'land', "git ls-files '*.h' '*.cpp' | xargs clang-format --dry-run --Werror",
          SOURCES),
    Check('layers', 'land', 'python tools/check_layers.py', HEADERS + r'|^tools/check_layers\.py$'),
    Check('decisions', 'land', 'python tools/check_decisions.py', r'.'),
    Check('compile-error claims', 'land', 'python tools/check_compile_errors.py',
          HEADERS + r'|^tests/(\w+/)?compile_errors\.|^tools/check_compile_errors\.py$'),
    Check('clang', 'land', 'cmake --workflow --preset clang', COMPILED),
    Check('msvc', 'pr', 'cmake --workflow --preset msvc (from an x64 developer environment)', COMPILED,
          [GUIDES, THREADS]),
    Check('gcc', 'pr', 'cmake --workflow --preset gcc', COMPILED),
    Check('clang compile errors', 'pr', 'cmake --workflow --preset clang-compile-errors', COMPILE_ERRORS,
          [CLAIMS]),
    Check('gcc compile errors', 'pr', 'cmake --workflow --preset gcc-compile-errors', COMPILE_ERRORS),
    Check('msvc compile errors', 'pr',
          'cmake --workflow --preset msvc-compile-errors (from an x64 developer environment)', COMPILE_ERRORS),
    Check('doxygen', 'pr', 'cmake --build build/docs --target docs (configured with EMRYS_FILTER_BUILD_DOCS)',
          HEADERS + r'|^docs/|^README\.md$'),
    Check('policies', 'pr', 'python tools/check_policies.py build/docs/docs/xml (after the docs target)',
          HEADERS + r'|^tests/policies\.cpp$|^tools/check_policies\.py$'),
    Check('clang-sanitize', 'pr', 'cmake --workflow --preset clang-sanitize (in WSL)', COMPILED),
    Check('coverage', 'pr',
          'cmake --workflow --preset clang-coverage && '
          'python tools/coverage.py build/clang-coverage --doxygen build/docs/docs/xml',
          HEADERS + r'|^tests/|^tools/coverage\.py$'),
    Check('gcc-no-exceptions', 'ci', 'gh workflow run CI --ref next -f gcc_no_exceptions=true', COMPILED,
          [Trigger('pr', 'exceptions or RTTI changed',
                   line=r'\b(throw|try|catch|typeid|dynamic_cast|__cpp_exceptions|__cpp_rtti)\b|<stdexcept>')]),
    Check('clang-libcxx', 'ci', 'gh workflow run CI --ref next -f clang_libcxx=true -f macos=true', COMPILED,
          [Trigger('pr', 'code depends on the standard library in use',
                   line=r'__cpp_lib_\w+|_LIBCPP|__GLIBCXX__|<execution>|std::execution')]),
    Check('cortex-m', 'ci', 'gh workflow run CI --ref next -f cortex_m=true', COMPILED,
          [Trigger('pr', 'code a microcontroller build cannot have changed',
                   path=r'^benchmarks/cortex_m/',
                   line=r'\bdouble\b|<iostream>|<cstdio>|std::(cout|cerr|printf|thread|mutex|function)\b'
                        r'|\bmalloc\b|\bthread_local\b')]),
    Check('vcpkg', 'ci', 'gh workflow run CI --ref next -f vcpkg=true',
          r'^packaging/vcpkg/|^tools/vcpkg_test_port\.py$|^cmake/EmrysFilterConfig|^examples/consumer/'
          r'|^CMakeLists\.txt$',
          [Trigger('pr', 'the vcpkg port or the package it installs changed',
                   path=r'^packaging/vcpkg/|^tools/vcpkg_test_port\.py$|^cmake/EmrysFilterConfig')]),
    Check('conan', 'ci', 'gh workflow run CI --ref next -f conan=true',
          r'^packaging/conan/|^tools/conan_test_recipe\.py$|^cmake/EmrysFilter|^CMakeLists\.txt$',
          [Trigger('pr', 'the Conan recipe or the package it installs changed',
                   path=r'^packaging/conan/|^tools/conan_test_recipe\.py$|^cmake/EmrysFilter')]),
    Check('gcc-arm64', 'ci', 'gh workflow run CI --ref next -f gcc_arm64=true', COMPILED,
          [Trigger('pr', 'alignment or vectorization changed',
                   line=r'\balignas\b|EIGEN_(MAX_)?(STATIC_)?ALIGN|__aarch64__|__ARM_NEON|reinterpret_cast')]),
]


def git(*arguments):
    return subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True, encoding='utf-8',
                          errors='replace', check=True).stdout


def change(base):
    """Returns the changed paths and, for each, its added and removed lines."""
    merge_base = git('merge-base', 'HEAD', base).strip()
    paths = {}
    for entry in git('diff', '--name-only', merge_base).splitlines():
        paths[entry] = []
    current = None
    for line in git('diff', '-U0', merge_base).splitlines():
        if line.startswith('+++ ') or line.startswith('--- '):
            if line.startswith('+++ b/') or (line.startswith('--- a/') and current is None):
                current = line[6:]
            continue
        if line.startswith('diff --git'):
            current = None
        elif current is not None and line[:1] in '+-':
            paths.setdefault(current, []).append(line[1:])
    for entry in git('ls-files', '--others', '--exclude-standard').splitlines():
        text = (ROOT / entry).read_text(encoding='utf-8', errors='replace')
        paths[entry] = text.splitlines()
    return paths


def reason(check, stage, paths):
    """Returns why `check` is required at `stage` for the change `paths`, or None."""
    if not any(re.search(check.tests, path) for path in paths):
        return None
    if STAGES.index(check.stage) <= STAGES.index(stage):
        return f'a {check.stage} check, and the change touches what it checks'
    for trigger in check.triggers:
        if STAGES.index(trigger.stage) > STAGES.index(stage):
            continue
        for path, lines in paths.items():
            if trigger.path and re.search(trigger.path, path):
                return f'{trigger.reason} ({path})'
            if trigger.line and re.search(trigger.within, path):
                for line in lines:
                    if re.search(trigger.line, line):
                        return f'{trigger.reason} ({path}: {line.strip()[:60]})'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=STAGES[:2])
    parser.add_argument('--base', help='the branch to compare with (default: next for land, origin/dev for pr)')
    arguments = parser.parse_args()
    paths = change(arguments.base or BASES[arguments.stage])
    if not paths:
        print('No change.')
        return 0
    later = STAGES[STAGES.index(arguments.stage) + 1]
    required = [(check, reason(check, arguments.stage, paths)) for check in CHECKS]
    required = [(check, why) for check, why in required if why]
    print(f'{len(paths)} paths changed; required at {arguments.stage}:')
    for check, why in required:
        print(f'  {check.name}: {why}\n      {check.command}')
    if not required:
        print('  nothing')
    names = {check.name for check, _ in required}
    added = [check.name for check in CHECKS if check.name not in names and reason(check, later, paths)]
    if added:
        print(f'Added at {later}: {", ".join(added)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

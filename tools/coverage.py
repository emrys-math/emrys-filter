"""Measures the coverage of the headers by the tests (decisions.md [coverage]), with Clang's
source-based coverage.

    python tools/coverage.py <build directory> [-j N] [--summary FILE] [--html DIRECTORY]

The build directory is a clang-coverage build (CMakePresets.json), whose programs write a profile
when they exit. The script clears the profiles of earlier runs, runs every test with ctest, merges
the profiles with llvm-profdata, and reads from llvm-cov the coverage of include/EmrysFilter/ by
every program ctest ran: the tests, the benchmarks' agreement tests, and the examples. It finds the
LLVM tools beside the compiler that built the programs, else on PATH.

The counts are per instantiation, merged by source location. A region is executed when it runs in
at least one instantiation, and a branch outcome likewise; a function is run when at least one of
its instantiations is; lines are llvm-cov's. A template nothing instantiates has no code and is in
no count, so the percentages are of what the tests instantiate. Regions and branches inside a
macro's expansion are left out, as llvm-cov's own report does not: they are almost all the failure
path of an EMRYS_FILTER_ASSERT, which correct code never takes; the assert-override tests check
that the assertions fire. So are regions inside the initializer of a constexpr variable, which the
compiler evaluates and never runs; a function only ever evaluated by the compiler is consteval,
which has no code to count. llvm-cov keeps one program's record of an inline function whose code differs between programs,
such as one that asserts in the assert-override tests, and warns how many it dropped.

With --doxygen, the XML directory of a Doxygen build of the headers (docs/CMakeLists.txt), it also
checks that the tests instantiate every public entity: each function with a body outside
Emrys::internal, public in its class, is instantiated when some instantiation the programs hold
begins inside its body. A class counts through its member functions, and a concept, alias, or
variable has no code to measure. A function whose body is a static_assert of internal::alwaysFalse,
or of internal::noSensorFor, sensor()'s variadic form of it, is a claimed compile error, the
negative compile tests' to check, not the instantiations'.

Prints a Markdown table by layer, the files with the most regions not executed, and the number of
negative compile tests in the sources, which the compile-errors presets run, and appends them to
--summary too (CI passes $GITHUB_STEP_SUMMARY). Writes every region not executed, as
file:line:column, to coverage_unexecuted.txt in the build directory, every public function not
instantiated to coverage_uninstantiated.txt, and llvm-cov's annotated sources to --html. Exit status
1 when a test fails, after the report.
"""
import argparse
import bisect
import collections
import functools
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
HEADERS = REPOSITORY / 'include' / 'EmrysFilter'
LAYERS = ['Core', 'Spaces', 'Distributions', 'Models', 'Filters', 'Smoothers']
UMBRELLA = 'umbrella headers'
# The variables that are false for every argument, whose static_assert makes a claimed compile error.
ALWAYS_FALSE = ('internal::alwaysFalse<', 'internal::noSensorFor<')
# llvm-cov's RegionKind for code, as against gaps and skipped code, and for a macro's use.
CODE_REGION = 0
EXPANSION_REGION = 1


def llvm_tool(name, compiler):
    """Returns the LLVM tool beside the compiler, following its links, or the one on PATH."""
    directory = pathlib.Path(os.path.realpath(shutil.which(compiler) or compiler)).parent
    for candidate in (directory / name, directory / f'{name}.exe'):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found is None:
        sys.exit(f'{name} is neither beside {compiler} nor on PATH')
    return found


def cached_compiler(build):
    """Returns CMAKE_CXX_COMPILER from the build directory's cache."""
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    match = re.search(r'^CMAKE_CXX_COMPILER:\w+=(.+)$', cache, re.MULTILINE)
    if match is None:
        sys.exit(f'{build} has no CMAKE_CXX_COMPILER; configure it with the clang-coverage preset')
    return match.group(1).strip()


def test_listing(build):
    """Returns ctest's list of the tests in the build directory."""
    listing = subprocess.run(['ctest', '--test-dir', str(build), '--show-only=json-v1'],
                             check=True, capture_output=True, text=True)
    return json.loads(listing.stdout)['tests']


def negative_tests():
    """Returns the number of negative compile tests: the `#if defined(CASE)` blocks of the
    tests/*/compile_errors.cpp files, as tests/CMakeLists.txt finds them. A coverage build does not
    run them, since a case compiles only, and the clang build's result is the same."""
    return sum(len(re.findall(r'^#if defined\(\w+\)', source.read_text(encoding='utf-8'), re.MULTILINE))
               for source in (REPOSITORY / 'tests').rglob('compile_errors.cpp'))


def test_programs(build, tests):
    """Returns the programs in the build directory that ctest runs, each once."""
    programs = []
    for test in tests:
        command = test.get('command')
        if not command:
            continue
        program = pathlib.Path(command[0]).resolve()
        if program.is_relative_to(build) and program.is_file() and program not in programs:
            programs.append(program)
    return programs


@functools.cache
def area(filename):
    """Returns the layer of a header, or UMBRELLA, or None for a file outside the headers."""
    try:
        parts = pathlib.Path(os.path.realpath(filename)).relative_to(HEADERS).parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else UMBRELLA


def code_regions(function):
    """Returns an instantiation's code regions, leaving out those inside a macro's expansion."""
    # A region whose file is the expansion of a macro use is the macro's, not the call site's.
    expansions = {r[6] for r in function['regions'] if r[7] == EXPANSION_REGION}
    return expansions, [r for r in function['regions']
                        if r[7] == CODE_REGION and r[5] not in expansions]


@functools.cache
def constant_initializers(filename):
    """Returns the sorted (start, end) positions, as (line, column) from 1, of the initializers of
    the file's constexpr variables, which the compiler evaluates and never runs."""
    text = pathlib.Path(filename).read_text(encoding='utf-8')
    starts = [0] + [m.end() for m in re.finditer('\n', text)]

    def position(offset):
        line = bisect.bisect_right(starts, offset)
        return line, offset - starts[line - 1] + 1

    spans = []
    for match in re.finditer(r'^[ \t]*(?:static |inline )*constexpr\b', text, re.MULTILINE):
        # A variable's name is followed by = or { before any parenthesis; a function's by one.
        declaration = re.compile(r'[^=({;]*(?:\{|=(?!=))').match(text, match.end())
        if declaration is None or re.search(r'[<>!]$', declaration.group()[:-1].rstrip()):
            continue
        # A brace initializer starts at its brace.
        depth, end = 0, declaration.end() - (declaration.group()[-1] == '{')
        while end < len(text) and not (text[end] == ';' and depth == 0):
            depth += text[end] in '([{'
            depth -= text[end] in ')]}'
            end += 1
        spans.append((position(declaration.end()), position(end)))
    return spans


def in_constant_initializer(filename, line, column):
    spans = constant_initializers(filename)
    index = bisect.bisect_right(spans, ((line, column), (line, column))) - 1
    return index >= 0 and spans[index][0] <= (line, column) <= spans[index][1]


def tally(export):
    """Merges the instantiations of llvm-cov's export by source location; returns the counts by
    area, the regions not executed by file, and the number of instantiations. A region or branch
    is keyed by where it starts: a program whose macros differ, as the assertion override's does,
    can end the same region elsewhere."""
    regions = {}         # (file, line, column) -> executed
    outcomes = {}        # (file, line, column, outcome) -> executed
    functions = {}       # (file, line, column) of the function -> run
    instantiations = collections.Counter()  # area -> [instantiated, run]
    run_instantiations = collections.Counter()
    for function in export['functions']:
        filenames = function['filenames']
        expansions, code = code_regions(function)
        if not code:
            continue
        # A function's first code region is its body, in its own file.
        first = code[0]
        home = area(filenames[first[5]])
        if home is None:
            continue
        key = (filenames[first[5]], first[0], first[1])
        functions[key] = functions.get(key, False) or function['count'] > 0
        instantiations[home] += 1
        run_instantiations[home] += function['count'] > 0
        for line, column, _, _, count, file_id in (r[:6] for r in code):
            if area(filenames[file_id]) is not None and not in_constant_initializer(filenames[file_id], line, column):
                key = (filenames[file_id], line, column)
                regions[key] = regions.get(key, False) or count > 0
        for branch in function.get('branches', []):
            line, column, _, _, true_count, false_count, file_id = branch[:7]
            if (file_id not in expansions and area(filenames[file_id]) is not None
                    and not in_constant_initializer(filenames[file_id], line, column)):
                for outcome, count in (('true', true_count), ('false', false_count)):
                    key = (filenames[file_id], line, column, outcome)
                    outcomes[key] = outcomes.get(key, False) or count > 0

    counts = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for name, table in (('regions', regions), ('branches', outcomes), ('functions', functions)):
        for key, executed in table.items():
            entry = counts[area(key[0])][name]
            entry[0] += executed
            entry[1] += 1
    for name in instantiations:
        counts[name]['instantiations'] = [run_instantiations[name], instantiations[name]]
    for file in export['files']:
        name = area(file['filename'])
        if name is not None:
            lines = file['summary']['lines']
            entry = counts[name]['lines']
            entry[0] += lines['covered']
            entry[1] += lines['count']
    unexecuted = collections.defaultdict(list)
    for key, executed in regions.items():
        if not executed:
            unexecuted[key[0]].append(key[1:3])
    return counts, unexecuted


def public_functions(xml):
    """Yields the file, first and last line of the body, and the name of each public function in
    the Doxygen XML with a body: outside Emrys::internal, public in its class, and neither defaulted
    nor deleted. Leaves out the claimed compile errors, and consteval functions, which the
    compiler evaluates and never emits."""
    seen = set()
    for path in sorted(xml.glob('*.xml')):
        for member in ElementTree.parse(path).getroot().iter('memberdef'):
            if member.get('kind') != 'function' or member.get('id') in seen:
                continue
            seen.add(member.get('id'))
            name = member.findtext('qualifiedname')
            location = member.find('location')
            if (member.get('prot') != 'public' or 'internal::' in name or location is None
                    or location.get('bodystart', '-1') == '-1'
                    or 'consteval' in member.findtext('definition', '').split()):
                continue
            file = HEADERS.parent / location.get('bodyfile')
            first, last = int(location.get('bodystart')), int(location.get('bodyend'))
            body = source_lines(file)[first - 1:last]
            if any(marker in line for line in body for marker in ALWAYS_FALSE):
                continue
            yield file, first, last, name + member.findtext('argsstring')


@functools.cache
def source_lines(file):
    return file.read_text(encoding='utf-8').splitlines()


def uninstantiated(export, xml):
    """Returns the number of public functions by area and those not instantiated, by file."""
    starts = collections.defaultdict(list)  # file -> the first line of each instantiation
    for function in export['functions']:
        _, code = code_regions(function)
        if code:
            starts[os.path.realpath(function['filenames'][code[0][5]])].append(code[0][0])
    for lines in starts.values():
        lines.sort()
    counts = collections.defaultdict(lambda: [0, 0])
    missing = collections.defaultdict(list)
    for file, first, last, name in public_functions(xml):
        lines = starts.get(os.path.realpath(file), [])
        index = bisect.bisect_left(lines, first)
        instantiated = index < len(lines) and lines[index] <= last
        entry = counts[area(file)]
        entry[0] += instantiated
        entry[1] += 1
        if not instantiated:
            missing[str(file)].append((first, name))
    return counts, missing


def percent(entry):
    covered, count = entry
    return f'{covered / count:.1%} of {count}' if count else '-'


def report(counts, unexecuted, entities):
    """Returns the Markdown report, with a column of public functions instantiated if entities."""
    columns = ['regions', 'branches', 'lines', 'functions', 'instantiations']
    header = '| Area | Regions | Branch outcomes | Lines | Functions | Instantiations run |'
    if entities:
        columns.append('entities')
        header += ' Public functions instantiated |'
    rows = [header, '|---' * (len(columns) + 1) + '|']
    total = collections.defaultdict(lambda: [0, 0])
    for name in LAYERS + [UMBRELLA]:
        if name not in counts:
            continue
        for column in columns:
            total[column][0] += counts[name][column][0]
            total[column][1] += counts[name][column][1]
        rows.append(f'| {name} | ' + ' | '.join(percent(counts[name][c]) for c in columns) + ' |')
    rows.append('| **All** | ' + ' | '.join(f'**{percent(total[c])}**' for c in columns) + ' |')
    rows += ['', '| File | Regions not executed |', '|---|---|']
    worst = sorted(unexecuted.items(), key=lambda item: -len(item[1]))[:15]
    for filename, places in worst:
        relative = pathlib.Path(os.path.realpath(filename)).relative_to(HEADERS.parent).as_posix()
        rows.append(f'| {relative} | {len(places)} |')
    return '\n'.join(['## Coverage of include/EmrysFilter/', ''] + rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('build', type=pathlib.Path)
    parser.add_argument('-j', '--jobs', type=int, help='tests ctest runs at once')
    parser.add_argument('--summary', type=pathlib.Path, help='a file to append the report to')
    parser.add_argument('--html', type=pathlib.Path, help='a directory for the annotated sources')
    parser.add_argument('--doxygen', type=pathlib.Path,
                        help='the XML directory of a Doxygen build, to check the public entities')
    arguments = parser.parse_args()
    build = arguments.build.resolve()

    if arguments.doxygen and not (arguments.doxygen / 'index.xml').is_file():
        sys.exit(f'{arguments.doxygen} holds no Doxygen XML; build the docs target first')
    compiler = cached_compiler(build)
    profdata = llvm_tool('llvm-profdata', compiler)
    cov = llvm_tool('llvm-cov', compiler)

    profiles = build / 'profiles'
    shutil.rmtree(profiles, ignore_errors=True)
    profiles.mkdir()
    # %m merges the profiles of every run of one program into one file as they exit.
    environment = dict(os.environ, LLVM_PROFILE_FILE=str(profiles / '%m.profraw'))
    command = ['ctest', '--test-dir', str(build), '--output-on-failure']
    if arguments.jobs:
        command += ['-j', str(arguments.jobs)]
    passed = subprocess.run(command, env=environment).returncode == 0

    raw = sorted(profiles.glob('*.profraw'))
    if not raw:
        sys.exit('the tests wrote no profiles; build with the clang-coverage preset')
    merged = build / 'coverage.profdata'
    subprocess.run([profdata, 'merge', '-sparse', '-o', str(merged)] + [str(p) for p in raw],
                   check=True)

    tests = test_listing(build)
    programs = test_programs(build, tests)
    objects = [str(programs[0])] + [a for p in programs[1:] for a in ('-object', str(p))]
    common = [f'-instr-profile={merged}'] + objects + [str(HEADERS)]
    exported = subprocess.run([cov, 'export', '-skip-expansions'] + common,
                              check=True, stdout=subprocess.PIPE, text=True)
    export = json.loads(exported.stdout)['data'][0]
    counts, unexecuted = tally(export)
    if arguments.doxygen:
        entities, missing = uninstantiated(export, arguments.doxygen)
        for name, entry in entities.items():
            counts[name]['entities'] = entry
        with (build / 'coverage_uninstantiated.txt').open('w', encoding='utf-8') as listing:
            for filename in sorted(missing):
                relative = pathlib.Path(filename).relative_to(HEADERS.parent).as_posix()
                for line, name in sorted(missing[filename]):
                    listing.write(f'{relative}:{line}: {name}\n')

    text = report(counts, unexecuted, arguments.doxygen is not None)
    text += (f'\nNegative compile tests: {negative_tests()}, each a misuse that must fail to '
             'compile with its claimed diagnostic (tools/check_compile_errors.py).\n')
    print(text)
    if arguments.summary:
        with arguments.summary.open('a', encoding='utf-8') as summary:
            summary.write(text)
    with (build / 'coverage_unexecuted.txt').open('w', encoding='utf-8') as listing:
        for filename in sorted(unexecuted):
            relative = pathlib.Path(os.path.realpath(filename)).relative_to(HEADERS.parent)
            for line, column in sorted(unexecuted[filename]):
                listing.write(f'{relative.as_posix()}:{line}:{column}\n')
    if arguments.html:
        subprocess.run([cov, 'show', '-format=html', f'-output-dir={arguments.html}',
                        '-show-line-counts-or-regions', '-show-branches=count',
                        '-show-instantiations=false', '-project-title=EmrysFilter'] + common,
                       check=True)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

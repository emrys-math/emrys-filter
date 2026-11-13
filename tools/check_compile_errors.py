"""Checks that every compile error the headers claim has a negative compile test (decisions.md
[coverage]).

A claim is a static_assert with a message, which a misuse can reach, or a deleted function, such as
marginal() of a temporary. A static_assert without a message checks the library, not its user. A
case is a block of a tests/*/compile_errors.cpp file opened by `#if defined(CASE)` and a comment
naming its claim (tests/compile_errors.cmake runs it):

    // static_assert: <header>: <text>   covers a static_assert of <header> whose message holds <text>
    // deleted: <header>: <declaration>  covers the deleted <declaration> of <header>
    // constraint: <header>: <concept>    a function of <header> constrained by <concept> refuses a misuse

A constraint is not a claim the headers make, so it needs no case; a case of one must name a
concept that <header> uses. A message repeated within one header is one claim. Prints the claims with no case and the cases
that name no claim, then the counts; exit status 1 when there is either.
"""
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADERS = ROOT / 'include' / 'EmrysFilter'
CASE = re.compile(r'^#if defined\((\w+)\)\s*// (static_assert|deleted|constraint): ([^:]+): (.+)$')
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
# Claims no case can reach, and why.
UNREACHABLE = {
    ('Version.h', 'EmrysFilter requires Eigen 5.0 or newer'):
        'the build has the Eigen it requires, and a case cannot include another',
    ('Indexing.h', 'marginal and conditional indices must select a set, not one index'):
        'every selection comes from a tag path through asIndexSet(), which makes one index a set',
    ('Consistency.h', 'a cumulative window is not a dynamic one'):
        'it checks two constants of the library',
    ('FormSteps.h', 'a linearization at a point is for a method that linearizes once at the means'):
        'every caller passes Linearized or checks isLinearMethod first',
    ('Linearization.h', 'a repaired innovation covariance replaces the one given'):
        'every caller passes an innovation covariance it owns and may change',
    ('TransformArgs.h', 'withCross() marks at most one input'):
        'transform() requires it of its arguments, so a call with two is no match instead',
    ('Measurement.h', 'conditioning in place takes noise of one component'):
        'a filter conditions in place only an update of one branch, which mixture noise is not',
    ('MeasurementReads.h', 'the function takes one vector per uncertain input'):
        'the library calls evaluate() with one vector per input of the measurement it resolved',
    ('MeasurementReads.h', 'the function takes one vector per uncertain input, or none to read each at its mean'):
        'the library calls evaluateAtState() with the inputs of the measurement it resolved, or none',
    ('MeasurementReads.h', 'the draw takes one vector per uncertain input, or none to read each at its mean'):
        'the library calls sampleAtState() with the inputs of the measurement it resolved',
    ('MeasurementReads.h', 'a measurement model known by its likelihood alone has no function to evaluate'):
        'every caller checks HasFunction first, with a message of its own',
    ('MeasurementReads.h',
     'additive noise needs the model\'s output space, from writes(), outputSpace(), or its function\'s result'):
        'only a model without a function has no output space, and measurementModel() refuses it additive noise',
    ('RbUpdate.h', 'a measurement weighted by its likelihood alone has logLikelihood(g)'):
        'the filters call updateGivenLikelihood() only for a model with its own likelihood',
    ('ProcessModel.h', 'the step takes one vector per uncertain input, or none to read each at its mean'):
        'the library calls propagate() with the inputs of the process model it resolved, or none',
    ('ProcessModel.h', 'the step takes one vector per uncertain input'):
        'the library calls gatherColumn() with one vector per input of the process model it resolved',
    ('TransformArgs.h', 'a function with more than one uncertain input must return a tuple of Jacobian blocks'):
        'transform() and the models split a Jacobian over the whole input into a block per uncertain input',
}


def assertions(text):
    """Yields the line and message of each static_assert in the text that has a message."""
    for match in re.finditer(r'\bstatic_assert\s*\(', text):
        depth, index, top_comma, strings = 1, match.end(), False, []
        while depth:
            character = text[index]
            if character == '"':
                literal = STRING.match(text, index)
                if top_comma:
                    strings.append(literal.group(1))
                index = literal.end()
                continue
            if character in '([{':
                depth += 1
            elif character in ')]}':
                depth -= 1
            elif character == ',' and depth == 1:
                top_comma = True
            index += 1
        if strings:
            yield text.count('\n', 0, match.start()) + 1, ''.join(strings)


def normalize(declaration):
    return ' '.join(declaration.replace('= delete', '').rstrip('; ').split())


def claims():
    """Returns {(header, kind, text): [locations]}: messages, and the declarations deleted."""
    found = collections.defaultdict(list)
    names = collections.Counter()
    for header in sorted(HEADERS.rglob('*.h')):
        names[header.name] += 1
        text = header.read_text(encoding='utf-8')
        where = header.relative_to(ROOT).as_posix()
        for line, message in assertions(text):
            found[(header.name, 'static_assert', message)].append(f'{where}:{line}')
        for line, source in enumerate(text.splitlines(), 1):
            code = source.split('//')[0]
            if re.search(r'=\s*delete\s*;', code):
                found[(header.name, 'deleted', normalize(code))].append(f'{where}:{line}')
    repeated = [name for name, count in names.items() if count > 1]
    if repeated:
        sys.exit(f'header names are not unique, so a case cannot name one: {repeated}')
    return found


def cases():
    """Yields the location, kind, header, and text of each case."""
    for source in sorted((ROOT / 'tests').rglob('compile_errors.cpp')):
        for line, text in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
            match = CASE.match(text)
            if match:
                where = f'{source.relative_to(ROOT).as_posix()}:{line}'
                yield where, match.group(2), match.group(3), match.group(4).strip()


def covers(claim, kind, header, text):
    claim_header, claim_kind, claim_text = claim
    if (claim_header, claim_kind) != (header, kind):
        return False
    return text in claim_text if kind == 'static_assert' else normalize(text) == claim_text


def main():
    found = claims()
    covered = set()
    problems = []
    for where, kind, header, text in cases():
        if kind == 'constraint':
            sources = list(HEADERS.rglob(header))
            if len(sources) != 1 or not re.search(rf'\b{re.escape(text)}\b', sources[0].read_text(encoding='utf-8')):
                problems.append(f'{where}: {header} uses no concept {text}')
            continue
        matched = [claim for claim in found if covers(claim, kind, header, text)]
        if not matched:
            problems.append(f'{where}: names no {kind} of {header}: {text}')
        elif len(matched) > 1:
            problems.append(f'{where}: names more than one {kind} of {header}: {text}')
        covered.update(matched)
    for claim, places in sorted(found.items()):
        header, kind, text = claim
        if claim in covered:
            if (header, text) in UNREACHABLE:
                problems.append(f'{places[0]}: has a case, so it is not unreachable: {text}')
        elif (header, text) not in UNREACHABLE:
            problems.append(f'{places[0]}: no case for the {kind}: {text}')
    claimed = {(header, text) for header, _, text in found}
    for header, text in sorted(UNREACHABLE):
        if (header, text) not in claimed:
            problems.append(f'UNREACHABLE names no claim of {header}: {text}')
    for problem in problems:
        print(problem)
    reachable = len(found) - len(UNREACHABLE)
    print(f'{len(covered)} of {reachable} claimed compile errors have a case '
          f'({len(UNREACHABLE)} unreachable)')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()

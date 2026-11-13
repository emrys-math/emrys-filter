"""Checks that each policy namespace holds only its kind (decisions.md [policy-factories]).

Emrys::methods holds transform methods, updates covariance update forms, reductions reduction
policies, resamplers resamplers, upkeep upkeep operations, execution execution policies, and
jacobians the marks of a batched function's Jacobians, and
each kind has a concept. C++ can't enumerate a namespace, so this reads the Doxygen XML of a build
of the headers (docs/CMakeLists.txt writes it) for every class, alias, function, and variable each
namespace holds, and checks that tests/policies.cpp has a static_assert of the kind's concept
naming it: a class, an alias, or a variable as `ns::Name`, a factory as `ns::name(` or `ns::name<`. It also
lists an assertion naming a member the XML doesn't hold, which catches a rename. One line per
violation and exit status 1 when there is one.

    python tools/check_policies.py build/docs/docs/xml
"""
import pathlib
import re
import sys
import xml.etree.ElementTree as ElementTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST = ROOT / 'tests' / 'policies.cpp'
CONCEPTS = {
    'methods': ['TransformMethod'],
    'updates': ['CovarianceUpdate'],
    'reductions': ['ReductionPolicy', 'MixtureReductionPolicy'],
    'resamplers': ['Resampler'],
    'upkeep': ['UpkeepOperation'],
    'execution': ['ExecutionPolicy'],
    'jacobians': ['JacobianMark'],
}


def members(xml):
    """Returns {namespace: {member: kind}} over the policy namespaces, from every compound of the
    Doxygen XML: a class or struct by its compound name, a function, a variable, or an alias by its
    qualified name, since a member in a group is listed in the group's compound and not the namespace's."""
    found = {namespace: {} for namespace in CONCEPTS}
    qualified = re.compile(r'^Emrys::(' + '|'.join(CONCEPTS) + r')::(\w+)$')
    for path in sorted(xml.glob('*.xml')):
        root = ElementTree.parse(path).getroot()
        for compound in root.iter('compounddef'):
            if compound.get('kind') in ('class', 'struct'):
                match = qualified.match(compound.findtext('compoundname', ''))
                if match:
                    found[match.group(1)][match.group(2)] = compound.get('kind')
            for member in compound.iter('memberdef'):
                if member.get('kind') in ('function', 'variable', 'typedef'):
                    match = qualified.match(member.findtext('qualifiedname', ''))
                    if match:
                        found[match.group(1)][match.group(2)] = member.get('kind')
    return found


def main():
    if len(sys.argv) != 2 or not (pathlib.Path(sys.argv[1]) / 'index.xml').is_file():
        sys.exit(f'usage: {pathlib.Path(sys.argv[0]).name} <doxygen xml directory>; build the docs target first')
    found = members(pathlib.Path(sys.argv[1]))
    test = TEST.relative_to(ROOT).as_posix()
    text = TEST.read_text(encoding='utf-8')
    # Each static_assert statement whole, with the line it starts on, since clang-format may wrap one.
    asserts = [(text.count('\n', 0, match.start()) + 1, match.group(0))
               for match in re.finditer(r'^\s*static_assert\(.*?\);', text, re.M | re.S)]
    violations = []
    for namespace, entries in sorted(found.items()):
        concepts = CONCEPTS[namespace]
        if not entries:
            violations.append(f'{namespace}: the XML holds no member of Emrys::{namespace}')
        for member, kind in sorted(entries.items()):
            # A factory is called, with template arguments or without; a type or a variable is named.
            needle = re.compile(re.escape(f'{namespace}::{member}') + (r'\s*[<(]' if kind == 'function' else r'\b'))
            if not any(needle.search(statement) and any(f'{concept}<' in statement for concept in concepts)
                       for _, statement in asserts):
                violations.append(f'{test}: no {" or ".join(concepts)} assertion names {namespace}::{member}')
    named = re.compile(r'\b(' + '|'.join(CONCEPTS) + r')::(\w+)')
    for number, statement in asserts:
        for namespace, member in named.findall(statement):
            if member not in found[namespace]:
                violations.append(f'{test}:{number}: {namespace}::{member} is not in the XML')
    for violation in violations:
        print(violation)
    total = sum(len(entries) for entries in found.values())
    print(f'{total} members in {len(found)} namespaces, {len(violations)} violations')
    sys.exit(1 if violations else 0)


main()

"""Checks the decisions in docs/design/decisions.md and every citation of them.

A decision is a heading `### name` in kebab case, and its index row links to it. A citation is the
name in brackets, `[name]`, anywhere in the headers, tests, benchmarks, tools, and documents. This
fails on a decision with no index row, an index row with no decision, and a citation whose name
has a hyphen but names no decision, which catches typos and names left behind by a rename. It warns
on a decision longer than 30 lines. Exit status 1 and one line per violation when there is one.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECISIONS = ROOT / 'docs' / 'design' / 'decisions.md'
SOURCES = ['include', 'tests', 'benchmarks', 'tools', 'docs', '.github']
SUFFIXES = {'.h', '.cpp', '.md', '.py', '.yml', '.txt', '.dox'}
CITATION = re.compile(r'\[([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\](?!\()')

text = DECISIONS.read_text(encoding='utf-8')
lines = text.splitlines()
headings = [(number, line[4:]) for number, line in enumerate(lines, 1) if line.startswith('### ')]
names = {name for _, name in headings}
indexed = set(re.findall(r'^\| \[([a-z0-9-]+)\]\(#\1\) \|', text, re.M))

violations = []
warnings = []
for number, name in headings:
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', name):
        violations.append(f'{DECISIONS.relative_to(ROOT).as_posix()}:{number}: "{name}" is not a kebab-case name')
    if name not in indexed:
        violations.append(f'{DECISIONS.relative_to(ROOT).as_posix()}:{number}: {name} has no row in the index')
for name in sorted(indexed - names):
    violations.append(f'{DECISIONS.relative_to(ROOT).as_posix()}: the index lists {name}, which has no section')
ends = [number for number, _ in headings[1:]] + [len(lines) + 1]
for (number, name), end in zip(headings, ends):
    body = [line for line in lines[number:end - 1] if not line.startswith('## ')]
    while body and not body[-1].strip():
        body.pop()
    if len(body) > 30:
        warnings.append(f'{DECISIONS.relative_to(ROOT).as_posix()}:{number}: {name} is {len(body)} lines, over 30')

roots = [ROOT / source for source in SOURCES] + sorted(ROOT.glob('*.md'))
paths = {p for r in roots for p in ([r] if r.is_file() else r.rglob('*')) if p.suffix in SUFFIXES}
for path in sorted(paths - {pathlib.Path(__file__).resolve()}):
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        for match in CITATION.finditer(line):
            if match.group(1) not in names:
                violations.append(f'{path.relative_to(ROOT).as_posix()}:{number}: [{match.group(1)}] names no decision')

for w in warnings:
    print(f'warning: {w}')
for v in violations:
    print(v)
print(f'{len(names)} decisions, {len(violations)} violations, {len(warnings)} warnings')
sys.exit(1 if violations else 0)

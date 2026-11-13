"""Checks that no header includes from a later layer (decisions.md [layered-headers]).

The directories under include/EmrysFilter/ are layers, in this order; a header may include from
its own layer and the ones before it. Umbrella headers beside the directories may include anything.
Exit status 1 and one line per violation when there is one.
"""
import pathlib
import re
import sys

LAYERS = ['Core', 'Spaces', 'Distributions', 'Models', 'Filters', 'Smoothers']
ROOT = pathlib.Path(__file__).resolve().parent.parent / 'include' / 'EmrysFilter'
INCLUDE = re.compile(r'#include\s+(?:<EmrysFilter/(\w+)/\w+\.h>|"[^"]*")')

violations = []
for layer, directory in enumerate(LAYERS):
    for header in sorted((ROOT / directory).glob('*.h')):
        for number, line in enumerate(header.read_text(encoding='utf-8').splitlines(), 1):
            match = INCLUDE.match(line.strip())
            if not match:
                continue
            target = match.group(1)
            where = f'{header.relative_to(ROOT.parent.parent)}:{number}'
            if target is None:
                violations.append(f'{where}: quoted include; use <EmrysFilter/Layer/Name.h>')
            elif LAYERS.index(target) > layer:
                violations.append(f'{where}: {directory} includes from the later layer {target}')
for extra in sorted(p for p in ROOT.glob('*/*.h') if p.parent.name not in LAYERS):
    violations.append(f'{extra.relative_to(ROOT.parent.parent)}: not in a layer directory')
for v in violations:
    print(v)
print(f'{len(violations)} layer violations')
sys.exit(1 if violations else 0)

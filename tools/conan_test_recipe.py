"""Writes a copy of the Conan recipe that builds from this checkout, for local runs and CI.

The recipe downloads the release tag's archive, and the tag does not exist until the launch. The
copy's source() takes the files git tracks in this checkout instead, with their working-tree
contents; everything else in the recipe is unchanged. Usage: conan_test_recipe.py <output
directory>, then `conan create <output directory>/all --version 0.1.0`.
"""
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECIPE = ROOT / 'packaging' / 'conan' / 'recipes' / 'emrys-filter' / 'all'
DOWNLOAD = 'get(self, **self.conan_data["sources"][self.version], strip_root=True)'

if len(sys.argv) != 2:
    sys.exit(__doc__)
output = pathlib.Path(sys.argv[1]).resolve()
shutil.rmtree(output, ignore_errors=True)
shutil.copytree(RECIPE, output / 'all')

source = output / 'source'
tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True)
for name in tracked.stdout.decode('utf-8').split('\0'):
    if name and (ROOT / name).is_file():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, source / name)

conanfile = output / 'all' / 'conanfile.py'
text = conanfile.read_text(encoding='utf-8')
if DOWNLOAD not in text:
    sys.exit('conan_test_recipe.py: no get() call in the recipe to replace')
text = text.replace(DOWNLOAD, f'copy(self, "*", src={source.as_posix()!r}, dst=self.source_folder)')
conanfile.write_text(text, encoding='utf-8')
print(f'wrote {output / "all"}, building from {ROOT}')

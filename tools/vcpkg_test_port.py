"""Writes a copy of the vcpkg port that builds from this checkout, for the `vcpkg` CI job.

The port fetches the release tag from GitHub, and the tag does not exist until the launch. The copy
replaces the portfile's vcpkg_from_github() call with this checkout, so vcpkg installs what is being
tested; everything else in the port is unchanged. Usage: vcpkg_test_port.py <output directory>,
which gets ports/emrys-filter/, for `vcpkg install --overlay-ports=<output directory>/ports`.
"""
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = ROOT / 'packaging' / 'vcpkg' / 'ports' / 'emrys-filter'

if len(sys.argv) != 2:
    sys.exit(__doc__)
destination = pathlib.Path(sys.argv[1]) / 'ports' / 'emrys-filter'
shutil.rmtree(destination, ignore_errors=True)
shutil.copytree(PORT, destination)

portfile = destination / 'portfile.cmake'
text = portfile.read_text(encoding='utf-8')
text, replaced = re.subn(r'vcpkg_from_github\(.*?\n\)\n',
                         lambda _: f'set(SOURCE_PATH "{ROOT.as_posix()}")\n', text, count=1, flags=re.S)
if replaced != 1:
    sys.exit('vcpkg_test_port.py: no vcpkg_from_github() call in the portfile to replace')
portfile.write_text(text, encoding='utf-8')
print(f'wrote {destination}, building from {ROOT}')

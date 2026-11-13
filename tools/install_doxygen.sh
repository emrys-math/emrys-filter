#!/usr/bin/env bash
# Installs the Doxygen release the reference is built with, for CI's Linux jobs, and puts it on
# GITHUB_PATH. Ubuntu 24.04's package is 1.9.8, which misparses the template code in
# Models/Placement.h and so fails the build that treats warnings as errors; the maintainer builds
# with 1.18.0, so CI does too. The archive is checked against its SHA-256 before it is unpacked.
set -euo pipefail

version=1.18.0
sha256=14fa81bdc34171edb5f1f02b1d60e74802f0439b77fa44e592565d517d72df90
dir="${RUNNER_TEMP:-/tmp}/doxygen"
archive="$dir/doxygen-$version.linux.bin.tar.gz"

mkdir -p "$dir"
curl -fsSL -o "$archive" \
  "https://github.com/doxygen/doxygen/releases/download/Release_${version//./_}/doxygen-$version.linux.bin.tar.gz"
echo "$sha256  $archive" | sha256sum --check --quiet
tar -xzf "$archive" -C "$dir"
"$dir/doxygen-$version/bin/doxygen" --version
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$dir/doxygen-$version/bin" >> "$GITHUB_PATH"
fi

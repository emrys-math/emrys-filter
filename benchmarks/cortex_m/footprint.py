"""Prints the Cortex-M programs' footprint as Markdown: each core's .text, .data, and .bss from
arm-none-eabi-size, and the largest stack frames from GCC's -fstack-usage files.

usage: footprint.py <size tool> (<core> <elf> <object dir>)...
"""

import pathlib
import subprocess
import sys


def sizes(size_tool, elf):
    """The text, data, and bss sizes of elf, in bytes."""
    output = subprocess.run([size_tool, elf], check=True, capture_output=True, text=True).stdout
    text, data, bss = output.splitlines()[1].split()[:3]
    return int(text), int(data), int(bss)


def frames(object_dir):
    """Every function's stack frame in object_dir's .su files, largest first, as (bytes, name, kind)."""
    result = []
    for su in pathlib.Path(object_dir).rglob("*.su"):
        for line in su.read_text().splitlines():
            # path:line:column:function, and the function's signature may hold colons too
            location, size, kind = line.rsplit("\t", 2)
            function = location.split(":", 3)[-1]
            result.append((int(size), function if len(function) <= 120 else function[:117] + "...", kind))
    return sorted(result, reverse=True)


def main(arguments):
    size_tool, rest = arguments[0], arguments[1:]
    cores = [rest[i : i + 3] for i in range(0, len(rest), 3)]
    print("| Core | .text | .data | .bss | Largest stack frame |")
    print("|---|---:|---:|---:|---|")
    for core, elf, object_dir in cores:
        text, data, bss = sizes(size_tool, elf)
        largest = frames(object_dir)
        frame = f"{largest[0][0]} B, {largest[0][1]}" if largest else "none"
        print(f"| {core} | {text} | {data} | {bss} | {frame} |")
    for core, _, object_dir in cores:
        print(f"\n{core}, largest frames (bytes, function, GCC's kind):\n")
        for size, name, kind in frames(object_dir)[:10]:
            print(f"- {size} `{name}` ({kind})")


if __name__ == "__main__":
    main(sys.argv[1:])

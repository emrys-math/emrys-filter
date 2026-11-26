"""Prints the Cortex-M footprint programs' sizes as Markdown: each program's .text, .data, and .bss
from arm-none-eabi-size, its .text and .bss less the simulator program's of the same core, which
every program shares, and the largest stack frame from GCC's -fstack-usage files.

usage: footprint.py <size tool> (<core> <program> <elf> <object dir>)...

The program named "simulator" runs the simulation alone; the others add one filter to it.
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
    programs = [rest[i : i + 4] for i in range(0, len(rest), 4)]
    measured = {(core, name): (sizes(size_tool, elf), frames(object_dir)) for core, name, elf, object_dir in programs}
    print("| Core | Program | .text | .data | .bss | .text less the simulator's | .bss less the simulator's | Largest stack frame |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for core, name, _, _ in programs:
        (text, data, bss), largest = measured[(core, name)]
        base = measured.get((core, "simulator"))
        if base is None or name == "simulator":
            extra_text = extra_bss = ""
        else:
            extra_text = str(text - base[0][0])
            extra_bss = str(bss - base[0][2])
        frame = f"{largest[0][0]} B, {largest[0][1]}" if largest else "none"
        print(f"| {core} | {name} | {text} | {data} | {bss} | {extra_text} | {extra_bss} | {frame} |")
    for core, name, _, _ in programs:
        print(f"\n{core}, {name}, largest frames (bytes, function, GCC's kind):\n")
        for size, function, kind in measured[(core, name)][1][:5]:
            print(f"- {size} `{function}` ({kind})")


if __name__ == "__main__":
    main(sys.argv[1:])

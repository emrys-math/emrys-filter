/* Every program here runs from the MPS2 boards' 4 MB of SSRAM at address 0, which QEMU's
 * mps2-an386 (Cortex-M4) and mps2-an500 (Cortex-M7) both have; QEMU loads the image there and
 * starts from the vector table at its base. The stack is the top of the memory. */

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 4M
}

ENTRY(resetHandler)

SECTIONS
{
    .vectors : { KEEP(*(.vectors)) } > RAM

    .text :
    {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > RAM

    .ARM.exidx : { *(.ARM.exidx*) } > RAM

    .init_array :
    {
        __init_array_start = .;
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        __init_array_end = .;
    } > RAM

    .data : { *(.data*) . = ALIGN(4); } > RAM

    .bss (NOLOAD) :
    {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}

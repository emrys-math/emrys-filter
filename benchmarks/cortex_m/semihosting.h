#pragma once

// Arm semihosting, the debugger's channel to the host, which QEMU serves with
// -semihosting-config enable=on: text out, and the program's exit status. The programs here use
// it in place of a UART and of newlib's stdio, which would bring the heap in.

namespace Semihosting
{

/// Performs semihosting operation \a operation with argument \a argument.
inline int
call(int operation, void const * argument)
{
    register int r0 asm("r0") = operation;
    register void const * r1 asm("r1") = argument;
    asm volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

/// Writes the null-terminated \a text to the host's console.
inline void
write(char const * text)
{
    call(0x04, text);  // SYS_WRITE0
}

/// Writes \a value in decimal.
inline void
write(long value)
{
    char digits[24];
    char * end = digits + sizeof digits;
    char * p = end;
    *--p = '\0';
    bool const negative = value < 0;
    unsigned long magnitude = negative ? 0ul - static_cast<unsigned long>(value) : static_cast<unsigned long>(value);
    do
    {
        *--p = static_cast<char>('0' + magnitude % 10);
        magnitude /= 10;
    }
    while (magnitude != 0);
    if (negative)
    {
        *--p = '-';
    }
    write(p);
}

/// Ends the program, reporting success to the host (QEMU's exit status 0) when \a success holds
/// and failure (status 1) when it does not.
[[noreturn]] inline void
exit(bool success)
{
    constexpr int applicationExit = 0x20026;  // ADP_Stopped_ApplicationExit
    constexpr int runTimeError = 0x20023;     // ADP_Stopped_RunTimeErrorUnknown
    call(0x18, reinterpret_cast<void const *>(success ? applicationExit : runTimeError));  // SYS_EXIT
    for (;;)
    {}
}

}  // namespace Semihosting

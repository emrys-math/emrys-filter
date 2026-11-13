// The startup code of a bare-metal Cortex-M program: the vector table, the reset handler that turns
// on the floating-point unit, clears .bss, runs the static constructors, and calls program(), and
// the handlers that end the program as a failure. The heap is one of them: newlib's malloc takes
// its first memory from _sbrk, which here ends the program, so a program that allocates fails, the
// no-heap check [embedded-builds]. A link-time check would be stricter than the rule: Eigen keeps
// a heap branch for scratch buffers over EIGEN_STACK_ALLOCATION_LIMIT, which no fixed size takes.

#include "semihosting.h"

#include <cstdint>

/// The program, which returns zero when every check passed. C++ forbids calling main().
int
program();

extern "C"
{
    extern std::uint32_t __bss_start[];
    extern std::uint32_t __bss_end[];
    extern void (*__init_array_start[])();
    extern void (*__init_array_end[])();
    extern std::uint32_t __stack_top[];

    [[noreturn]] void
    resetHandler();
    [[noreturn]] void
    faultHandler();

    /// newlib's abort() raises a signal through syscalls a bare-metal program lacks; failing here
    /// keeps them out.
    [[noreturn]] void
    abort()
    {
        Semihosting::write("abort\n");
        Semihosting::exit(false);
    }

    /// The heap's growth, which newlib's malloc calls for its first block: allocating is a failure.
    [[noreturn]] void *
    _sbrk(int)
    {
        Semihosting::write("heap used\n");
        Semihosting::exit(false);
    }

    /// Every fault and unexpected interrupt ends the program as a failure.
    [[noreturn]] void
    faultHandler()
    {
        Semihosting::write("fault\n");
        Semihosting::exit(false);
    }

    /// Enables the floating-point unit before any code that might use it, then runs the program.
    [[noreturn]] void
    resetHandler()
    {
        // CPACR: full access to coprocessors 10 and 11, the FPU.
        auto & cpacr = *reinterpret_cast<std::uint32_t volatile *>(0xE000ED88u);
        cpacr = cpacr | (0xFu << 20);
        asm volatile("dsb\n\tisb" ::: "memory");
        for (std::uint32_t * word = __bss_start; word < __bss_end; ++word)
        {
            *word = 0;
        }
        for (auto constructor = __init_array_start; constructor < __init_array_end; ++constructor)
        {
            (*constructor)();
        }
        Semihosting::exit(program() == 0);
    }
}

/// The initial stack pointer, the reset handler, and the fault handlers for the exceptions every
/// Cortex-M has; nothing here enables an interrupt.
__attribute__((section(".vectors"), used)) void const * const vectorTable[16] = {
    __stack_top,
    reinterpret_cast<void const *>(&resetHandler),
    reinterpret_cast<void const *>(&faultHandler),  // NMI
    reinterpret_cast<void const *>(&faultHandler),  // HardFault
    reinterpret_cast<void const *>(&faultHandler),  // MemManage
    reinterpret_cast<void const *>(&faultHandler),  // BusFault
    reinterpret_cast<void const *>(&faultHandler),  // UsageFault
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    reinterpret_cast<void const *>(&faultHandler),  // SVCall
    reinterpret_cast<void const *>(&faultHandler),  // DebugMonitor
    nullptr,
    reinterpret_cast<void const *>(&faultHandler),  // PendSV
    reinterpret_cast<void const *>(&faultHandler),  // SysTick
};

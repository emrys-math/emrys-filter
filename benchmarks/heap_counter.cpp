// The replacement of the global operator new that counts heap allocations for the benchmarks'
// allocation tests (see heap_counter.h). A program may replace operator new once, so this lives in
// its own translation unit, shared by every problem's allocation test in the test target.

#include "heap_counter.h"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace Benchmark
{

bool countingHeap = false;
int heapAllocations = 0;

}  // namespace Benchmark

void *
operator new(std::size_t size)
{
    if (Benchmark::countingHeap)
    {
        ++Benchmark::heapAllocations;
    }
    if (void * p{std::malloc(size == 0 ? 1 : size)})
    {
        return p;
    }
#if defined(__cpp_exceptions)
    throw std::bad_alloc{};
#else
    std::abort();
#endif
}

void
operator delete(void * p) noexcept
{
    std::free(p);
}

void
operator delete(void * p, std::size_t /*size*/) noexcept
{
    std::free(p);
}

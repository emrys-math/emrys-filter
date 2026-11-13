#pragma once

// A count of what reaches the global heap while a benchmark's steps run, for the allocation tests:
// heap_counter.cpp replaces the global operator new, once per test target, to count while
// countingHeap is set. Eigen's own allocations are caught separately, by EIGEN_RUNTIME_NO_MALLOC
// and Eigen::internal::set_is_malloc_allowed(false).

namespace Benchmark
{

/// Whether operator new counts; set it around the steps under test.
extern bool countingHeap;

/// The allocations counted since it was last reset.
extern int heapAllocations;

}  // namespace Benchmark

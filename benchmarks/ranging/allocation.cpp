// Neither ranging EKF, the EmrysFilter one of emrys_ranging.h nor the plain-Eigen one of
// eigen_ranging.h, allocates on the heap in its steps, so the benchmark's timings contain no
// allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC, so Eigen asserts on any heap
// allocation while it is forbidden, and heap_counter.cpp's operator new counts what anything else
// asks of the heap. Roadmap: Verification, Benchmarks.

#include "../heap_counter.h"
#include "cases.h"
#include "eigen_ranging.h"
#include "emrys_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class RangingAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingAllocation, Ranging::Cases);

// After one step to warm up, twenty predict-update steps of each implementation make no heap
// allocation: Eigen's assertion catches its own, and the replaced operator new counts the rest.
TYPED_TEST(RangingAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    auto const scenario = Ranging::makeScenario<D, A>();
    using Scalar = typename TypeParam::Scalar;
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 21, 1));
    auto emrys = Ranging::EmrysRanging<D, A, TypeParam::Kind, Scalar>{scenario};
    auto eigen = Ranging::EigenRanging<D, A, TypeParam::Kind, Scalar>{scenario};
    emrys.predict();
    emrys.update(ranges[0]);
    eigen.predict();
    eigen.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        emrys.predict();
        emrys.update(ranges[k]);
        eigen.predict();
        eigen.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

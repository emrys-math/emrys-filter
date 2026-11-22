// mherb/kalman's ranging EKF of mherb_ranging.h does not allocate on the heap in its steps, so the
// benchmark against it times no allocator on either side; allocation.cpp checks the EmrysFilter
// side. The target is built with EIGEN_RUNTIME_NO_MALLOC, so Eigen asserts on any heap allocation
// while it is forbidden, and heap_counter.cpp's operator new counts what anything else asks of the
// heap. Roadmap: the benchmarks against other libraries.

#include "../heap_counter.h"
#include "cases.h"
#include "mherb_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class RangingMherbAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingMherbAllocation, Ranging::MherbCases);

// After one step to warm up, twenty predict-update steps make no heap allocation: Eigen's
// assertion catches its own, and the replaced operator new counts the rest.
TYPED_TEST(RangingMherbAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    using Filter = Ranging::MherbRanging<D, A, Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Filter::rangesOf(Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 21, 1)));
    auto mherb = Filter{scenario};
    mherb.predict();
    mherb.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        mherb.predict();
        mherb.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

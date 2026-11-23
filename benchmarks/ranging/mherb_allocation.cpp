// The heap allocations in the steps of the ranging benchmark against mherb/kalman: each of the four
// filters, with EmrysFilter (emrys_ranging_kinds.h) and with mherb/kalman (mherb_ranging.h), in
// double and in float. EmrysFilter's make none, and neither do mherb/kalman's but its square-root
// UKF, whose prediction downdates its factor by Eigen's LLT::rankUpdate(), which allocates a
// temporary vector of dynamic size on each call. That filter is not timed (cases.h), so the timings
// contain no allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC, so Eigen asserts on any
// heap allocation while it is forbidden, and heap_counter.cpp's operator new counts what anything
// else asks of the heap. Roadmap: the benchmarks against other libraries.

#include "../heap_counter.h"
#include "cases.h"
#include "emrys_ranging_kinds.h"
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

// After one step to warm up, twenty predict-update steps of each implementation make no heap
// allocation: Eigen's assertion catches its own, and the replaced operator new counts the rest.
// mherb/kalman's square-root UKF runs with Eigen's allocations allowed, since its prediction makes
// one in LLT::rankUpdate(); the replaced operator new still counts anything else it asks for.
TYPED_TEST(RangingMherbAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    using Mherb = Ranging::MherbRanging<D, A, TypeParam::Kind, Scalar>;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 21, 1));
    auto const mherbRanges = Mherb::rangesOf(ranges);
    auto emrys = Ranging::EmrysRangingKind<D, A, TypeParam::Kind, Scalar>{scenario};
    auto mherb = Mherb{scenario};
    emrys.predict();
    emrys.update(ranges[0]);
    mherb.predict();
    mherb.update(mherbRanges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        emrys.predict();
        emrys.update(ranges[k]);
        if constexpr (TypeParam::Kind == Ranging::FilterKind::SqrtUkf)
        {
            Eigen::internal::set_is_malloc_allowed(true);
        }
        mherb.predict();
        mherb.update(mherbRanges[k]);
        Eigen::internal::set_is_malloc_allowed(false);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

// Neither side of the ranging benchmark against FrancoisCarouge/Kalman allocates on the heap in its
// steps: the EKF with EmrysFilter (emrys_ranging_kinds.h) and with the library (carouge_ranging.h),
// in double and in float, so the timings contain no allocator. The library's filter is allocated
// once, when CarougeRanging is constructed, before the counting starts. The target is built with
// EIGEN_RUNTIME_NO_MALLOC, carouge_ranging.cpp included, so Eigen asserts on any heap allocation
// while it is forbidden, and heap_counter.cpp's operator new counts what anything else asks of the
// heap. Roadmap: the benchmarks against other libraries.

#include "../heap_counter.h"
#include "carouge_ranging.h"
#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class RangingCarougeAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingCarougeAllocation, Ranging::CarougeCases);

// After one step to warm up, twenty predict-update steps of each implementation make no heap
// allocation: Eigen's assertion catches its own, and the replaced operator new counts the rest.
TYPED_TEST(RangingCarougeAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 21, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    auto carouge = Ranging::CarougeRanging<D, A, Scalar>{scenario};
    emrys.predict();
    emrys.update(ranges[0]);
    carouge.predict();
    carouge.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        emrys.predict();
        emrys.update(ranges[k]);
        carouge.predict();
        carouge.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

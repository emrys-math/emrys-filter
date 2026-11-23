// Neither side of the ranging benchmark against TinyEKF allocates on the heap in its steps: the
// EKF with EmrysFilter (emrys_ranging_kinds.h) and with TinyEKF (tinyekf_ranging.h), in double and
// in float, so the timings contain no allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC,
// so Eigen asserts on any heap allocation while it is forbidden, and heap_counter.cpp's operator new
// counts what anything else asks of the heap. Roadmap: the benchmarks against other libraries.

#include "../heap_counter.h"
#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"
#include "tinyekf_ranging.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class RangingTinyEkfAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingTinyEkfAllocation, Ranging::TinyEkfCases);

// After one step to warm up, twenty predict-update steps of each implementation make no heap
// allocation: Eigen's assertion catches its own, and the replaced operator new counts the rest.
TYPED_TEST(RangingTinyEkfAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, 21, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    auto tiny = Ranging::TinyEkfRanging<D, A, Scalar>{scenario};
    emrys.predict();
    emrys.update(ranges[0]);
    tiny.predict();
    tiny.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        emrys.predict();
        emrys.update(ranges[k]);
        tiny.predict();
        tiny.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

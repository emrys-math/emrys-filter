// The heap allocations in the steps of the ranging benchmark against OpenCV: EmrysFilter's EKF
// (emrys_ranging_kinds.h) makes none, and cv::KalmanFilter run as an EKF (opencv_ranging.h) is
// counted and recorded, not required to be zero, since the pair times what its users run and the
// OpenCV it is built against is whichever is installed. With OpenCV 4.14.0 its steps make no
// allocation with 4 or 6 anchors and one per step with 8 or more in double, and with 12 or more in
// float, so the timings of those cases include its allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC,
// so Eigen asserts on any heap allocation while it is forbidden, and heap_counter.cpp's operator new
// counts what anything else asks of the heap; cv::Mat buffers come from OpenCV's own allocator,
// and none is made in a step. Roadmap: the benchmarks against other libraries.

#include "../heap_counter.h"
#include "cases.h"
#include "emrys_ranging_kinds.h"
#include "opencv_ranging.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <string>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class RangingOpenCvAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(RangingOpenCvAllocation, Ranging::OpenCvCases);

/// The steps after warm-up that each test counts.
constexpr std::size_t CountedSteps = 20;

// After one step to warm up, twenty predict-update steps of EmrysFilter's EKF make no heap
// allocation: Eigen's assertion catches its own, and the replaced operator new counts the rest.
TYPED_TEST(RangingOpenCvAllocation, EmrysFilterStepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, int(CountedSteps) + 1, 1));
    auto emrys = Ranging::EmrysRangingKind<D, A, Ranging::FilterKind::Ekf, Scalar>{scenario};
    emrys.predict();
    emrys.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        emrys.predict();
        emrys.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

// The same twenty steps of OpenCV's EKF, whose allocations are recorded as the test's property
// heap_allocations_per_step and printed. OpenCV's arithmetic is in its library, outside Eigen, so
// only the replaced operator new sees it.
TYPED_TEST(RangingOpenCvAllocation, OpenCvStepsAreCounted)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    using Scalar = typename TypeParam::Scalar;
    auto const scenario = Ranging::makeScenario<D, A>();
    auto const ranges = Ranging::rangesIn<Scalar>(Ranging::simulate(scenario, int(CountedSteps) + 1, 1));
    auto cv = Ranging::OpenCvRanging<D, A, Scalar>{scenario};
    cv.predict();
    cv.update(ranges[0]);

    Benchmark::heapAllocations = 0;
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < ranges.size(); ++k)
    {
        cv.predict();
        cv.update(ranges[k]);
    }
    Benchmark::countingHeap = false;
    double const perStep = double(Benchmark::heapAllocations) / double(CountedSteps);
    ::testing::Test::RecordProperty("heap_allocations_per_step", std::to_string(perStep));
    std::printf("OpenCV, %dD with %d anchors: %g heap allocations per step\n", D, A, perStep);
}

}  // namespace

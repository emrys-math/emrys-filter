// Neither attitude EKF, the EmrysFilter one of emrys_attitude.h nor the plain-Eigen one of
// eigen_attitude.h, allocates on the heap in its steps, the reset included, so the benchmark's
// timings contain no allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC, so Eigen asserts
// on any heap allocation while it is forbidden, and heap_counter.cpp's operator new counts what
// anything else asks of the heap. Roadmap: step 13e and Verification, Benchmarks.

#include "../heap_counter.h"
#include "cases.h"
#include "eigen_attitude.h"
#include "emrys_attitude.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class AttitudeAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(AttitudeAllocation, Orientation::Cases);

// After one step to warm up, twenty steps of each implementation, a prediction and two updates
// each, make no heap allocation: Eigen's assertion catches its own, and the replaced operator new
// counts the rest.
TYPED_TEST(AttitudeAllocation, StepsDoNotAllocate)
{
    auto const scenario = Orientation::makeScenario();
    auto const run = Orientation::simulate(scenario, 21, 1);
    auto emrys = Orientation::EmrysAttitude<TypeParam::Kind>{scenario};
    auto eigen = Orientation::EigenAttitude<TypeParam::Kind>{scenario};
    emrys.predict(run.gyro[0]);
    emrys.update(run.gravity[0], run.magneticField[0]);
    eigen.predict(run.gyro[0]);
    eigen.update(run.gravity[0], run.magneticField[0]);

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = 1; k < run.gyro.size(); ++k)
    {
        emrys.predict(run.gyro[k]);
        emrys.update(run.gravity[k], run.magneticField[k]);
        eigen.predict(run.gyro[k]);
        eigen.update(run.gravity[k], run.magneticField[k]);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

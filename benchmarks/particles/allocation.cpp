// Neither ranging bootstrap filter, the EmrysFilter one of emrys_particles.h nor the plain-Eigen
// one of eigen_particles.h, allocates on the heap in its steps once it has run a few, so the
// benchmark's timings contain no allocator. The target is built with EIGEN_RUNTIME_NO_MALLOC, so
// Eigen asserts on any heap allocation while it is forbidden, and heap_counter.cpp's operator new
// counts what anything else asks of the heap. Roadmap: step 18 and Verification, Benchmarks.

#include "../heap_counter.h"
#include "cases.h"
#include "eigen_particles.h"
#include "emrys_particles.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <cstddef>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace
{

template <typename C>
class ParticlesAllocation : public ::testing::Test
{};

TYPED_TEST_SUITE(ParticlesAllocation, Particles::Cases);

// After five steps to warm up, which include at least one resample so that the resampling scratch
// of both filters exists, twenty predict-update-housekeep steps of each implementation make no
// heap allocation: Eigen's assertion catches its own, and the replaced operator new counts the
// rest.
TYPED_TEST(ParticlesAllocation, StepsDoNotAllocate)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    constexpr std::size_t warmUp = 5;
    auto const scenario = Particles::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, warmUp + 20, 1);
    auto const initial = Particles::drawInitialParticles(scenario, TypeParam::ParticleCount, 2);
    auto emrys = Particles::EmrysParticles<D, A, TypeParam::Kind>{scenario, initial, 3};
    auto eigen = Particles::EigenParticles<D, A>{scenario, initial, 3};
    auto const step = [&](std::size_t k) {
        emrys.predict();
        emrys.update(run.ranges[k]);
        emrys.housekeep();
        eigen.predict();
        eigen.update(run.ranges[k]);
        eigen.housekeep();
    };
    for (std::size_t k = 0; k < warmUp; ++k)
    {
        step(k);
    }
    ASSERT_GT(emrys.state().resampleCount(), 0) << "the warm-up never resampled";

    Benchmark::heapAllocations = 0;
    Eigen::internal::set_is_malloc_allowed(false);
    Benchmark::countingHeap = true;
    for (std::size_t k = warmUp; k < run.ranges.size(); ++k)
    {
        step(k);
    }
    Benchmark::countingHeap = false;
    Eigen::internal::set_is_malloc_allowed(true);
    EXPECT_EQ(Benchmark::heapAllocations, 0);
}

}  // namespace

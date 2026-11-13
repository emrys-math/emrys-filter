// The particle benchmark's preconditions [benchmarks]: before any timing counts, the EmrysFilter and
// plain-Eigen bootstrap filters of emrys_particles.h and eigen_particles.h, seeded alike, must
// agree particle by particle and weight by weight to rounding, under both likelihoods, and the
// EmrysFilter one must be consistent by an NEES test. Covers the planar and spatial cases.
// Roadmap: step 18 and Verification, Benchmarks; benchmarks.md, claim 3.

#include "cases.h"
#include "eigen_particles.h"
#include "emrys_particles.h"
#include "scenario.h"

#include <EmrysFilter/Core/ChiSquared.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>

namespace
{

using Particles::EigenParticles;
using Particles::EmrysParticles;

template <typename C>
class ParticlesTest : public ::testing::Test
{};

TYPED_TEST_SUITE(ParticlesTest, Particles::Cases);

// Both implementations, started from the same particles and seeded alike, hold the same particles
// with the same log weights at every step. They draw the same normals and the same resampling
// offset in the same order and compute the same products in nearly the same order, so they
// differ by rounding; 1e-9 relative to the size of the set leaves room for rounding accumulated
// over 300 steps and nothing more, since a modeling difference, such as a wrong likelihood or a
// resampling decision that went the other way, moves whole particles. The moments agree too,
// which is what the timing rows claim.
TYPED_TEST(ParticlesTest, ImplementationsAgree)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    auto const scenario = Particles::makeScenario<D, A>();
    auto const run = Ranging::simulate(scenario, 300, 1);
    auto const initial = Particles::drawInitialParticles(scenario, TypeParam::ParticleCount, 2);
    auto emrys = EmrysParticles<D, A, TypeParam::Kind>{scenario, initial, 3};
    auto eigen = EigenParticles<D, A>{scenario, initial, 3};
    for (std::size_t k = 0; k < run.ranges.size(); ++k)
    {
        emrys.predict();
        eigen.predict();
        emrys.update(run.ranges[k]);
        eigen.update(run.ranges[k]);
        emrys.housekeep();
        eigen.housekeep();
        double const stateScale = std::max(1.0, eigen.states().norm());
        double const weightScale = std::max(1.0, eigen.logWeights().matrix().norm());
        ASSERT_LT((emrys.state().states() - eigen.states()).norm(), 1e-9 * stateScale) << "at step " << k;
        ASSERT_LT((emrys.state().logWeights() - eigen.logWeights()).matrix().norm(), 1e-9 * weightScale)
            << "at step " << k;
        if ((k + 1) % 50 == 0)
        {
            auto const moments = emrys.state().moments();
            ASSERT_TRUE(moments.has_value()) << "at step " << k;
            double const meanScale = std::max(1.0, eigen.mean().norm());
            double const covScale = std::max(1.0, eigen.covariance().norm());
            EXPECT_LT((moments->mean() - eigen.mean()).norm(), 1e-9 * meanScale) << "at step " << k;
            EXPECT_LT((moments->covarianceLower() - eigen.covariance()).norm(), 1e-9 * covScale) << "at step " << k;
        }
    }
    EXPECT_GT(emrys.state().resampleCount(), 0) << "the run never resampled, so the test did not exercise it";
}

// The EmrysFilter bootstrap filter is consistent: over 50 independent runs, the average NEES of
// its particles' moments at each checked step lies in the two-sided 99.9% interval of a
// chi-squared with 50 times the state's degrees of freedom, divided by 50 (Bar-Shalom, Li, and
// Kirubarajan (2001), §5.4). A particle filter is consistent only in the limit of many particles,
// so the interval is widened by a quarter on each side for the sample-size bias of a thousand;
// a wrong likelihood or a lost resampling step moves the NEES by a factor, not a quarter. The
// agreement test carries the result to the plain-Eigen filter.
TYPED_TEST(ParticlesTest, Consistent)
{
    constexpr int D = TypeParam::Dimension;
    constexpr int A = TypeParam::AnchorCount;
    constexpr int runs = 50;
    constexpr int steps = 100;
    constexpr int checkEvery = 25;
    auto const scenario = Particles::makeScenario<D, A>();
    double const dof = runs * 2.0 * D;
    double const lower = 0.75 * Emrys::chiSquaredQuantile<double>(0.0005, dof) / runs;
    double const upper = 1.25 * Emrys::chiSquaredQuantile<double>(0.9995, dof) / runs;

    auto nees = std::array<double, steps / checkEvery>{};
    for (int r = 0; r < runs; ++r)
    {
        auto const seed = static_cast<unsigned>(r);
        auto const run = Ranging::simulate(scenario, steps, 100 + seed);
        auto const initial = Particles::drawInitialParticles(scenario, TypeParam::ParticleCount, 200 + seed);
        auto filter = EmrysParticles<D, A, TypeParam::Kind>{scenario, initial, 300 + seed};
        for (int k = 0; k < steps; ++k)
        {
            filter.predict();
            filter.update(run.ranges[k]);
            filter.housekeep();
            if ((k + 1) % checkEvery == 0)
            {
                auto const moments = filter.state().moments();
                ASSERT_TRUE(moments.has_value()) << "at step " << k << " of run " << r;
                auto const error = (run.truth[k] - moments->mean()).eval();
                nees[(k + 1) / checkEvery - 1] += error.dot(moments->covarianceLower().ldlt().solve(error)) / runs;
            }
        }
    }
    for (std::size_t c = 0; c < nees.size(); ++c)
    {
        EXPECT_GT(nees[c], lower) << "at step " << (c + 1) * checkEvery;
        EXPECT_LT(nees[c], upper) << "at step " << (c + 1) * checkEvery;
    }
}

}  // namespace

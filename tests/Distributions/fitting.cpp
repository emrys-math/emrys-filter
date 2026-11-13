// fitMixture() of Distributions/Fitting.h (step 7): expectation-maximization recovers a mixture
// from samples or weighted particles, reduces a mixture without sampling, is seeded by k-means++ or
// a given mixture, never decreases the log-likelihood, respects the per-dimension variance floor,
// and reports degenerate inputs as errors.

#include <EmrysFilter/Distributions/Fitting.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "user_policies.h"

using Emrys::FitError;
using Emrys::FitOptions;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ParticleSet;

namespace
{

using Mixture2 = GaussianMixtureParams<Linear<2>>;
using Particles2 = ParticleSet<Linear<2>>;

std::array<Eigen::Vector2d, 3> const trueMeans{
    Eigen::Vector2d{-6.0, 0.0},
    Eigen::Vector2d{0.0, 6.0},
    Eigen::Vector2d{6.0, 0.0}
};

std::array<Eigen::Matrix2d, 3> const trueCovs{
    Eigen::Matrix2d{{1.0, 0.3}, {0.3, 0.5}},
    Eigen::Matrix2d{{0.6, -0.2}, {-0.2, 1.2}},
    Eigen::Matrix2d{{1.5, 0.0}, {0.0, 0.4}}
};

std::array<double, 3> const trueWeights{0.2, 0.3, 0.5};

Mixture2
makeTrueMixture()
{
    auto mixture = Mixture2{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        mixture.add(std::log(trueWeights[c]), trueMeans[c], trueCovs[c]);
    }
    return mixture;
}

// Expects \a fitted to have one component close to each true component.
void
expectRecovers(Mixture2 const & fitted, double meanTolerance, double weightTolerance, double covTolerance)
{
    ASSERT_EQ(fitted.size(), 3);
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto nearest = fitted.begin();
        for (auto it = fitted.begin(); it != fitted.end(); ++it)
        {
            if ((it->component.mean() - trueMeans[c]).norm() < (nearest->component.mean() - trueMeans[c]).norm())
            {
                nearest = it;
            }
        }
        EXPECT_LT((nearest->component.mean() - trueMeans[c]).norm(), meanTolerance) << "component " << c;
        EXPECT_NEAR(std::exp(nearest->logWeight), trueWeights[c], weightTolerance) << "component " << c;
        EXPECT_LT((nearest->component.covarianceLower() - trueCovs[c]).norm(), covTolerance * trueCovs[c].norm())
            << "component " << c;
    }
}

}  // namespace

// Fitting three components to 20,000 samples of a three-component mixture converges to a normalized
// mixture with each true component recovered to within 0.1 in mean, 0.02 in weight, and 10% in
// covariance.
TEST(FitMixture, RecoversMixtureFromSamples)
{
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{41};
    auto const particles = Particles2{Emrys::drawSamples(truth.toDensity().value(), 20000, rng)};
    auto const fit = Emrys::fitMixture(particles, 3, rng);
    ASSERT_TRUE(fit.has_value());
    EXPECT_TRUE(fit->converged);
    EXPECT_NEAR(fit->mixture.logTotalWeight().value(), 0.0, 1e-12);
    expectRecovers(fit->mixture, 0.1, 0.02, 0.1);
}

// With samples from a broad proposal and importance weights for the true mixture, the fit recovers
// the components from the weights alone, with looser tolerances for the noisier weighted sample.
TEST(FitMixture, UsesParticleWeights)
{
    // Importance sampling from a broad proposal: the weights alone carry the target's shape.
    auto const truth = makeTrueMixture();
    auto const target = truth.toDensity();
    auto const proposal = GaussianParams<Linear<2>>{Eigen::Vector2d{0.0, 2.0}, 49.0 * Eigen::Matrix2d::Identity()};
    auto rng = std::mt19937_64{43};
    auto const drawn = Emrys::drawSamplesWithLogDensity(proposal.toDensity().value(), 60000, rng);
    auto logWeights = Eigen::ArrayXd{drawn.logDensity.size()};
    for (Eigen::Index i = 0; i < logWeights.size(); ++i)
    {
        logWeights(i) = target->logDensity(drawn.samples.col(i)) - drawn.logDensity(i);
    }
    auto const particles = Particles2{drawn.samples, logWeights};
    auto const fit = Emrys::fitMixture(particles, 3, rng);
    ASSERT_TRUE(fit.has_value());
    expectRecovers(fit->mixture, 0.25, 0.04, 0.3);
}

// Fitting three components to a 999-component mixture (narrow children jittered inside each true
// component) through the moment-matching path recovers the true components and preserves the
// overall mean exactly and the covariance up to the variance floor.
TEST(FitMixture, ReducesJitteredMixtureWithoutSampling)
{
    // 999 children, each a narrowed copy of a true component jittered within it: the
    // moment-matched groups are the true components.
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{47};
    auto children = Mixture2{};
    children.reserve(999);
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const jitter = GaussianParams<Linear<2>>{trueMeans[c], 0.5 * trueCovs[c]};
        auto const centers = Emrys::drawSamples(jitter.toDensity().value(), 333, rng);
        for (Eigen::Index j = 0; j < centers.cols(); ++j)
        {
            children.add(std::log(trueWeights[c] / 333.0), centers.col(j), 0.5 * trueCovs[c]);
        }
    }
    auto const source = children.toDensity();
    auto const fit = Emrys::fitMixture(*source, 3, rng);
    ASSERT_TRUE(fit.has_value());
    EXPECT_TRUE(fit->converged);
    expectRecovers(fit->mixture, 0.2, 0.01, 0.2);

    // Moment matching preserves the overall mean exactly and the covariance up to the floor.
    auto const before = source->moments();
    auto const after = fit->mixture.toDensity()->moments();
    EXPECT_LT((after.mean() - before.mean()).norm(), 1e-9);
    EXPECT_LT((after.covarianceLower() - before.covarianceLower()).norm(), 1e-4);
}

// A mixture of ten identical copies of two components seeds each distinct location once, so the two
// fitted means are the two locations exactly rather than two copies of one.
TEST(FitMixture, StackedComponentsAreSeededOnce)
{
    auto stacked = Mixture2{};
    for (int copy = 0; copy < 10; ++copy)
    {
        stacked.add(0.0, trueMeans[0], trueCovs[0]);
        stacked.add(0.0, trueMeans[2], trueCovs[2]);
    }
    auto rng = std::mt19937_64{53};
    auto const fit = Emrys::fitMixture(stacked.toDensity().value(), 2, rng);
    ASSERT_TRUE(fit.has_value());
    ASSERT_EQ(fit->mixture.size(), 2);
    double const a = fit->mixture.begin()->component.mean()(0);
    double const b = fit->mixture[1].component.mean()(0);
    EXPECT_NEAR(std::min(a, b), -6.0, 1e-9);
    EXPECT_NEAR(std::max(a, b), 6.0, 1e-9);
}

// Asking for five components from particles at only two distinct points returns two: the fit cannot
// make more components than there are distinct seeds.
TEST(FitMixture, DuplicateParticlesLimitTheComponentCount)
{
    auto states = Eigen::Matrix<double, 2, 100>{};
    states.leftCols<50>().colwise() = Eigen::Vector2d{-6.0, -1.0};
    states.rightCols<50>().colwise() = Eigen::Vector2d{6.0, 1.0};
    auto const particles = Particles2{states};
    auto rng = std::mt19937_64{59};
    auto const fit = Emrys::fitMixture(particles, 5, rng);
    ASSERT_TRUE(fit.has_value());
    EXPECT_EQ(fit->mixture.size(), 2);
}

// With one dimension a million times narrower than the other, the regularization floor scales per
// dimension, so the narrow dimension's variance stays small rather than taking the wide one's
// floor.
TEST(FitMixture, FloorScalesWithEachDimension)
{
    // Dimensions a million times apart in scale: each component's floor follows its own dimension.
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{83};
    Eigen::Matrix2Xd samples = Emrys::drawSamples(truth.toDensity().value(), 5000, rng);
    samples.row(1) *= 1e-6;
    auto const particles = Particles2{samples};
    auto const fit = Emrys::fitMixture(particles, 3, rng, FitOptions<double>{.regularization = 1e-2});
    ASSERT_TRUE(fit.has_value());
    for (auto const & component : fit->mixture)
    {
        EXPECT_LT(component.component.covarianceLower()(1, 1), 1e-10) << "floor leaked across dimensions";
    }
}

// Particles with no spread in one dimension are reported as Degenerate rather than fitted.
TEST(FitMixture, NoSpreadInOneDimensionIsDegenerate)
{
    Eigen::Matrix<double, 2, 10> states = Eigen::Matrix<double, 2, 10>::Random();
    states.row(1).setConstant(3.0);
    auto const particles = Particles2{states};
    auto rng = std::mt19937_64{89};
    EXPECT_EQ(Emrys::fitMixture(particles, 2, rng).error(), FitError::Degenerate);
}

// Particles whose weights are all zero fail with InvalidWeights.
TEST(FitMixture, InvalidWeightsFail)
{
    auto particles = Particles2{Eigen::Matrix<double, 2, 3>::Random()};
    particles.scaleWeights(-std::numeric_limits<double>::infinity());
    auto rng = std::mt19937_64{83};
    EXPECT_EQ(Emrys::fitMixture(particles, 2, rng).error(), FitError::InvalidWeights);
}

// Seeding the fit with the true mixture instead of k-means++ converges and recovers the components
// to the same tolerances as the sampled fit, under a parallel policy too.
TEST(FitMixture, SeedsFromAMixture)
{
    auto const truth = makeTrueMixture();
    auto const view = truth.toDensity();
    auto rng = std::mt19937_64{61};
    auto const particles = Particles2{Emrys::drawSamples(*view, 20000, rng)};
    auto const fit = Emrys::fitMixture(particles, *view, 3, rng);
    ASSERT_TRUE(fit.has_value());
    EXPECT_TRUE(fit->converged);
    expectRecovers(fit->mixture, 0.1, 0.02, 0.1);
    auto const threaded = Emrys::fitMixture(EmrysTest::ThreadsPolicy{}, particles, *view, 3, rng);
    ASSERT_TRUE(threaded.has_value());
    EXPECT_TRUE(threaded->converged);
    expectRecovers(threaded->mixture, 0.1, 0.02, 0.1);
}

// Running the fit with 1 to 12 iterations from the same seed gives a log-likelihood that never
// decreases from one iteration count to the next: the EM monotonicity guarantee.
TEST(FitMixture, LogLikelihoodNeverDecreases)
{
    auto const truth = makeTrueMixture();
    auto sampler = std::mt19937_64{67};
    auto const particles = Particles2{Emrys::drawSamples(truth.toDensity().value(), 2000, sampler)};
    double previous = -std::numeric_limits<double>::infinity();
    for (int iterations = 1; iterations <= 12; ++iterations)
    {
        auto rng = std::mt19937_64{71};  // same seeding every time
        auto const fit = Emrys::fitMixture(particles, 4, rng, FitOptions<double>{.maxIterations = iterations});
        ASSERT_TRUE(fit.has_value());
        EXPECT_GE(fit->logLikelihood, previous - 1e-9) << "iterations " << iterations;
        previous = fit->logLikelihood;
    }
}

// A dynamic-size particle set fits to a dynamic-size mixture with the particles' dimension.
TEST(FitMixture, DynamicSize)
{
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{73};
    auto const particles =
        ParticleSet<Linear<Eigen::Dynamic>>{Eigen::MatrixXd{Emrys::drawSamples(truth.toDensity().value(), 5000, rng)}};
    auto const fit = Emrys::fitMixture(particles, 3, rng);
    ASSERT_TRUE(fit.has_value());
    static_assert(std::is_same_v<decltype(fit->mixture), GaussianMixtureParams<Linear<Eigen::Dynamic>>>);
    EXPECT_EQ(fit->mixture.valueDimension(), 2);
}

// Coincident particles are reported as Degenerate.
TEST(FitMixture, DegenerateSourceFails)
{
    auto const collapsed = Particles2{Eigen::Matrix<double, 2, 10>::Ones()};
    auto rng = std::mt19937_64{79};
    auto const fit = Emrys::fitMixture(collapsed, 2, rng);
    ASSERT_FALSE(fit.has_value());
    EXPECT_EQ(fit.error(), FitError::Degenerate);
}

// A source of more components than a chunk, under a parallel policy, sums the components' own
// covariances chunk by chunk, and fits as Sequential does to rounding.
TEST(FitMixture, ManyComponentsUnderAPolicyMatchSequential)
{
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{97};
    auto children = Mixture2{};
    constexpr Eigen::Index perComponent = 800;
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const centers = Emrys::drawSamples(truth.toDensity().value(), perComponent, rng);
        for (Eigen::Index j = 0; j < centers.cols(); ++j)
        {
            children.add(std::log(trueWeights[c] / double(perComponent)), centers.col(j), 0.5 * trueCovs[c]);
        }
    }
    auto const source = children.toDensity();
    ASSERT_TRUE(source);
    auto sequentialRng = std::mt19937_64{101};
    auto threadedRng = std::mt19937_64{101};
    auto const sequential = Emrys::fitMixture(*source, 3, sequentialRng);
    auto const threaded = Emrys::fitMixture(EmrysTest::ThreadsPolicy{}, *source, 3, threadedRng);
    ASSERT_TRUE(sequential && threaded);
    EXPECT_NEAR(threaded->logLikelihood, sequential->logLikelihood, 1e-9 * std::abs(sequential->logLikelihood));
}

// Particles of weight zero take no part in the fit, the particles' weights are checked before a fit
// seeded from a mixture or run under a policy, a minimum weight that every component falls below
// leaves none, and so do components that collapse onto points with no regularization.
TEST(FitMixture, WeightlessParticlesAndComponentsThatVanish)
{
    auto const truth = makeTrueMixture();
    auto rng = std::mt19937_64{103};
    auto particles = Particles2{Emrys::drawSamples(truth.toDensity().value(), 2000, rng)};
    particles.setLogWeight(0, -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(Emrys::fitMixture(particles, 3, rng));
    EXPECT_EQ(
        Emrys::fitMixture(particles, 3, rng, FitOptions<double>{.minLogWeight = 0.0}).error(),
        FitError::Degenerate
    );

    auto invalid = Particles2{Eigen::Matrix<double, 2, 3>::Random()};
    invalid.scaleWeights(-std::numeric_limits<double>::infinity());
    auto const seeds = truth.toDensity();
    EXPECT_EQ(Emrys::fitMixture(invalid, *seeds, 2, rng).error(), FitError::InvalidWeights);
    EXPECT_EQ(Emrys::fitMixture(EmrysTest::ThreadsPolicy{}, invalid, 2, rng).error(), FitError::InvalidWeights);
    EXPECT_EQ(Emrys::fitMixture(EmrysTest::ThreadsPolicy{}, invalid, *seeds, 2, rng).error(), FitError::InvalidWeights);

    auto points = Eigen::Matrix<double, 2, 15>{};
    for (Eigen::Index j = 0; j < 15; ++j)
    {
        points.col(j) = j < 5  ? Eigen::Vector2d{0.0, 0.0}
                      : j < 10 ? Eigen::Vector2d{10.0, 0.0}
                               : Eigen::Vector2d{0.0, 10.0};
    }
    EXPECT_EQ(
        Emrys::fitMixture(Particles2{points}, 3, rng, FitOptions<double>{.regularization = 0.0}).error(),
        FitError::Degenerate
    );
}

// The divergences of Distributions/Divergence.h (step 12b, [divergences]): Kullback-Leibler, Bhattacharyya,
// Hellinger, Jensen-Shannon, and Wasserstein between Gaussians against their closed forms, Monte
// Carlo estimates where there is none, categoricals against hand sums and their bounds, the
// Runnalls merge cost and the bracket it gives on the divergence of a mixture, a hybrid state, or a
// mixture of hybrids from its collapse, a
// Wasserstein ground metric that measures a transformed block, periodic means across the seam, and
// expectedSquaredError of each distribution. The closed forms between square-root forms are the covariance forms'.

#include <EmrysFilter/Distributions/Divergence.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/KernelDensity.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Distributions/Uniform.h>

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>

using Emrys::bhattacharyya;
using Emrys::CategoricalParams;
using Emrys::Composite;
using Emrys::expectedSquaredError;
using Emrys::Finite;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::GroundMetric;
using Emrys::hellinger;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::jensenShannon;
using Emrys::kullbackLeibler;
using Emrys::kullbackLeiblerCollapseBounds;
using Emrys::Linear;
using Emrys::MixtureParams;
using Emrys::MonteCarlo;
using Emrys::ParticleSet;
using Emrys::RadiansPi;
using Emrys::wasserstein;

namespace
{

constexpr double pi = std::numbers::pi;
constexpr double inf = std::numeric_limits<double>::infinity();

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Heading : RadiansPi<1>
{};

using State = Composite<Position, Velocity>;
using Plane = GaussianParams<Linear<2>>;

Plane
first()
{
    return Plane{Eigen::Vector2d{1.0, -0.5}, Eigen::Matrix2d{{2.0, 0.3}, {0.3, 1.0}}};
}

Plane
second()
{
    return Plane{Eigen::Vector2d{0.2, 0.4}, Eigen::Matrix2d{{1.5, -0.4}, {-0.4, 0.8}}};
}

Eigen::Matrix2d
inverse(Eigen::Matrix2d const & m)
{
    return m.ldlt().solve(Eigen::Matrix2d::Identity());
}

Eigen::Matrix2d
sqrt(Eigen::Matrix2d const & m)
{
    auto const solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>{m};
    return solver.eigenvectors() * solver.eigenvalues().cwiseSqrt().asDiagonal() * solver.eigenvectors().transpose();
}

double
handKullbackLeibler(Plane const & p, Plane const & q)
{
    Eigen::Vector2d const d = p.mean() - q.mean();
    Eigen::Matrix2d const qInverse = inverse(q.covarianceLower());
    return 0.5
         * ((qInverse * p.covarianceLower()).trace() + d.dot(qInverse * d) - 2
            + std::log(q.covarianceLower().determinant()) - std::log(p.covarianceLower().determinant()));
}

double
handBhattacharyya(Plane const & p, Plane const & q)
{
    Eigen::Vector2d const d = p.mean() - q.mean();
    Eigen::Matrix2d const average = 0.5 * (p.covarianceLower() + q.covarianceLower());
    return d.dot(inverse(average) * d) / 8
         + 0.5
               * std::log(
                   average.determinant()
                   / std::sqrt(p.covarianceLower().determinant() * q.covarianceLower().determinant())
               );
}

double
handWasserstein(Plane const & p, Plane const & q)
{
    Eigen::Matrix2d const root = sqrt(q.covarianceLower());
    Eigen::Matrix2d const cross = sqrt(root * p.covarianceLower() * root);
    return std::sqrt(
        (p.mean() - q.mean()).squaredNorm() + p.covarianceLower().trace() + q.covarianceLower().trace()
        - 2 * cross.trace()
    );
}

template <typename Estimate>
void
expectWithin(Estimate const & estimate, double expected, double standardErrors = 5)
{
    EXPECT_GT(estimate.standardError, 0);
    EXPECT_NEAR(estimate.value, expected, standardErrors * estimate.standardError)
        << "standard error " << estimate.standardError;
}

// Expects the lower end of kullbackLeiblerCollapseBounds() to be the upper end less the entropy of
// the weights, computed here, and the Monte Carlo divergence of mixture from its collapse to lie
// within five standard errors of the bracket.
void
expectCollapseBracketed(GaussianMixtureParams<Linear<2>> const & mixture, std::mt19937 & rng)
{
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    Plane const collapsed = density->collapsed();
    auto const collapsedDensity = collapsed.toDensity();
    ASSERT_TRUE(collapsedDensity);
    double weightEntropy = 0;
    for (auto const & entry : *density)
    {
        double const weight = std::exp(entry.logWeight - density->logTotalWeight());
        weightEntropy -= weight * std::log(weight);
    }
    auto const bounds = kullbackLeiblerCollapseBounds(*density);
    EXPECT_NEAR(bounds.lower, std::max(bounds.upper - weightEntropy, 0.0), 1e-14);
    auto const estimate = kullbackLeibler(MonteCarlo{200'000}, *density, *collapsedDensity, rng);
    EXPECT_GT(estimate.standardError, 0);
    EXPECT_LE(estimate.value, bounds.upper + 5 * estimate.standardError) << "upper " << bounds.upper;
    EXPECT_GE(estimate.value, bounds.lower - 5 * estimate.standardError) << "lower " << bounds.lower;
}

}  // namespace

// kullbackLeibler() between two Gaussians matches the closed form computed with plain Eigen in both
// directions, is asymmetric, and is exactly zero for a density against itself.
TEST(KullbackLeibler, GaussiansMatchTheClosedForm)
{
    Plane const p = first();
    Plane const q = second();
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    ASSERT_TRUE(pDensity && qDensity);
    EXPECT_NEAR(kullbackLeibler(*pDensity, *qDensity), handKullbackLeibler(p, q), 1e-12);
    EXPECT_NEAR(kullbackLeibler(*qDensity, *pDensity), handKullbackLeibler(q, p), 1e-12);
    EXPECT_GT(std::abs(kullbackLeibler(*pDensity, *qDensity) - kullbackLeibler(*qDensity, *pDensity)), 0.02);
    EXPECT_EQ(kullbackLeibler(*pDensity, *pDensity), 0);
}

// The Monte Carlo estimate with 100,000 samples reports its count and a standard error under 0.01
// and lies within five standard errors of the closed form.
TEST(KullbackLeibler, MonteCarloAgreesWithTheClosedForm)
{
    Plane const p = first();
    Plane const q = second();
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    auto rng = std::mt19937{11};
    // The estimate reports its own standard error, and expectWithin() judges by it; the bound on
    // the error itself checks that the count was used.
    auto const estimate = kullbackLeibler(MonteCarlo{100'000}, *pDensity, *qDensity, rng);
    EXPECT_EQ(estimate.count, 100'000);
    EXPECT_LT(estimate.standardError, 0.01);
    expectWithin(estimate, kullbackLeibler(*pDensity, *qDensity));
}

// The Monte Carlo estimate from a one-component mixture agrees with the Gaussian closed form.
TEST(KullbackLeibler, OneComponentMixtureMatchesItsGaussian)
{
    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(0.0, first().mean(), first().covarianceLower());
    Plane const q = second();
    auto const mixtureDensity = mixture.toDensity();
    auto const qDensity = q.toDensity();
    ASSERT_TRUE(mixtureDensity && qDensity);
    auto rng = std::mt19937{12};
    expectWithin(kullbackLeibler(MonteCarlo{50'000}, *mixtureDensity, *qDensity, rng), handKullbackLeibler(first(), q));
}

// A kernel density of 2,000 samples of q is close to q but, widened by its bandwidth, not at it:
// the estimate is positive and below 0.1.
TEST(KullbackLeibler, KernelDensityOfParticlesIsEstimated)
{
    Plane const q = second();
    auto const qDensity = q.toDensity();
    auto rng = std::mt19937{13};
    auto const particles = ParticleSet<Linear<2>>{drawSamples(*qDensity, 2'000, rng)};
    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde);
    // The kernel density is q widened by its bandwidth, so it is close to q but not at it.
    auto const estimate = kullbackLeibler(MonteCarlo{2'000}, *kde, *qDensity, rng);
    EXPECT_GT(estimate.value, 0);
    EXPECT_LT(estimate.value, 0.1);
}

// Between categoricals the divergence is the hand sum over values, infinite when q has support that
// p lacks, and zero against itself.
TEST(KullbackLeibler, CategoricalsSumOverValues)
{
    auto const pWeights = CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.5, 0.3, 0.2, 0.0});
    auto const qWeights =
        CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.25, 0.25, 0.25, 0.25});
    auto const p = pWeights.toDensity().value();
    auto const q = qWeights.toDensity().value();
    double const expected = 0.5 * std::log(2.0) + 0.3 * std::log(1.2) + 0.2 * std::log(0.8);
    EXPECT_NEAR(kullbackLeibler(p, q), expected, 1e-15);
    // q gives the last value a probability that p doesn't.
    EXPECT_EQ(kullbackLeibler(q, p), inf);
    EXPECT_EQ(kullbackLeibler(p, p), 0);
}

// Runnalls' merge cost, half of (w_i + w_j) log|S| - w_i log|S_i| - w_j log|S_j| for the
// moment-matched merge, equals the weighted divergences of the two components from their merge,
// computed here with kullbackLeibler().
TEST(KullbackLeibler, RunnallsCostIsTheWeightedDivergenceOfTheMergedComponent)
{
    // B = (w_i + w_j) log|S| - w_i log|S_i| - w_j log|S_j|, halved, for the moment-matched merge
    // with covariance S, is the cost MixtureParams::merged() documents: the weighted divergences of
    // the two components from their merge.
    Plane const a = first();
    Plane const b = second();
    double const wa = 0.3;
    double const wb = 0.2;
    double const sharedA = wa / (wa + wb);
    double const sharedB = wb / (wa + wb);
    Eigen::Vector2d const mean = sharedA * a.mean() + sharedB * b.mean();
    Eigen::Vector2d const da = a.mean() - mean;
    Eigen::Vector2d const db = b.mean() - mean;
    Eigen::Matrix2d const cov =
        sharedA * (a.covarianceLower() + da * da.transpose()) + sharedB * (b.covarianceLower() + db * db.transpose());
    auto const merged = Plane{mean, cov};
    auto const mergedDensity = merged.toDensity();
    double const cost = 0.5
                      * ((wa + wb) * std::log(cov.determinant()) - wa * std::log(a.covarianceLower().determinant())
                         - wb * std::log(b.covarianceLower().determinant()));
    EXPECT_NEAR(
        wa * kullbackLeibler(a.toDensity().value(), *mergedDensity)
            + wb * kullbackLeibler(b.toDensity().value(), *mergedDensity),
        cost,
        1e-14
    );
}

// The upper end of kullbackLeiblerCollapseBounds() of two components is half of
// log|S| - w_a log|S_a| - w_b log|S_b|
// with S the moment-matched covariance computed with plain Eigen, and equals the components'
// weighted divergences from the collapse, computed with kullbackLeibler().
TEST(KullbackLeiblerCollapseBounds, IsTheWeightedDivergenceOfTheComponentsFromTheCollapse)
{
    Plane const a = first();
    Plane const b = second();
    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(std::log(0.3), a.mean(), a.covarianceLower());
    mixture.add(std::log(0.7), b.mean(), b.covarianceLower());
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    Eigen::Vector2d const mean = 0.3 * a.mean() + 0.7 * b.mean();
    Eigen::Vector2d const da = a.mean() - mean;
    Eigen::Vector2d const db = b.mean() - mean;
    Eigen::Matrix2d const cov =
        0.3 * (a.covarianceLower() + da * da.transpose()) + 0.7 * (b.covarianceLower() + db * db.transpose());
    double const expected = 0.5
                          * (std::log(cov.determinant()) - 0.3 * std::log(a.covarianceLower().determinant())
                             - 0.7 * std::log(b.covarianceLower().determinant()));
    double const bound = kullbackLeiblerCollapseBounds(*density).upper;
    EXPECT_NEAR(bound, expected, 1e-14);
    Plane const collapsed = density->collapsed();
    auto const collapsedDensity = collapsed.toDensity();
    ASSERT_TRUE(collapsedDensity);
    EXPECT_NEAR(
        0.3 * kullbackLeibler(a.toDensity().value(), *collapsedDensity)
            + 0.7 * kullbackLeibler(b.toDensity().value(), *collapsedDensity),
        bound,
        1e-14
    );
}

// A mixture of one component, or of differently weighted copies of one Gaussian, is its own
// collapse, so both ends of the bracket are zero up to rounding in the log-determinants.
TEST(KullbackLeiblerCollapseBounds, CopiesOfOneGaussianGiveZero)
{
    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(0.0, first().mean(), first().covarianceLower());
    {
        auto const density = mixture.toDensity();
        ASSERT_TRUE(density);
        EXPECT_NEAR(kullbackLeiblerCollapseBounds(*density).upper, 0, 1e-14);
        EXPECT_EQ(kullbackLeiblerCollapseBounds(*density).lower, 0);
    }
    mixture.add(std::log(2.0), first().mean(), first().covarianceLower());
    mixture.add(std::log(0.5), first().mean(), first().covarianceLower());
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    EXPECT_NEAR(kullbackLeiblerCollapseBounds(*density).upper, 0, 1e-14);
    EXPECT_EQ(kullbackLeiblerCollapseBounds(*density).lower, 0);
}

// For an overlapping pair, a pair sharing a mean with different covariances, and three components
// of unequal weights, the Monte Carlo divergence from the collapse over 200,000 draws lies between
// the two ends of the bracket, within five standard errors.
TEST(KullbackLeiblerCollapseBounds, BracketsTheMonteCarloDivergence)
{
    auto rng = std::mt19937{14};
    auto overlapping = GaussianMixtureParams<Linear<2>>{2};
    overlapping.add(std::log(0.3), first().mean(), first().covarianceLower());
    overlapping.add(std::log(0.7), second().mean(), second().covarianceLower());
    expectCollapseBracketed(overlapping, rng);

    auto sharedMean = GaussianMixtureParams<Linear<2>>{2};
    sharedMean.add(std::log(0.5), Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    sharedMean.add(std::log(0.5), Eigen::Vector2d::Zero(), Eigen::Matrix2d{{9.0, 0.0}, {0.0, 0.25}});
    expectCollapseBracketed(sharedMean, rng);

    auto three = GaussianMixtureParams<Linear<2>>{2};
    three.add(std::log(0.6), first().mean(), first().covarianceLower());
    three.add(std::log(0.3), second().mean(), second().covarianceLower());
    three.add(std::log(0.1), Eigen::Vector2d{-3.0, 2.0}, Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.5}});
    expectCollapseBracketed(three, rng);
}

// Two equally weighted components 50 standard deviations apart overlap nowhere, so the mixture's
// entropy is the components' plus log 2, and the divergence from the collapse is the bound less
// log 2, the lower end of the bracket: the Monte Carlo estimate agrees within five standard errors.
TEST(KullbackLeiblerCollapseBounds, SeparatedComponentsOvershootByTheEntropyOfTheWeights)
{
    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(0.0, Eigen::Vector2d{-25.0, 0.0}, first().covarianceLower());
    mixture.add(0.0, Eigen::Vector2d{25.0, 0.0}, first().covarianceLower());
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    Plane const collapsed = density->collapsed();
    auto const collapsedDensity = collapsed.toDensity();
    ASSERT_TRUE(collapsedDensity);
    auto rng = std::mt19937{15};
    expectWithin(
        kullbackLeibler(MonteCarlo{100'000}, *density, *collapsedDensity, rng),
        kullbackLeiblerCollapseBounds(*density).lower
    );
}

// A hybrid state's bracket is that of its toMixture(), the mixture of its conditionals weighted by
// the mode probabilities, to rounding; a mode of zero probability adds nothing to either end. The
// two modes' means are far enough apart that the lower end is above zero.
TEST(KullbackLeiblerCollapseBounds, HybridIsTheMixtureOfItsConditionals)
{
    using Mode = Finite<3>;
    auto const far = Plane{Eigen::Vector2d{10.0, 0.0}, second().covarianceLower()};
    auto hybrid = HybridGaussianParams<Linear<2>, Mode>{
        CategoricalParams<Mode>::fromWeights(Eigen::Vector3d{0.3, 0.7, 0.0}),
        first()
    };
    hybrid.setConditional(1, far);
    hybrid.setConditional(2, Plane{Eigen::Vector2d{-3.0, 2.0}, Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.5}}});
    auto const density = hybrid.toDensity();
    ASSERT_TRUE(density);
    auto const bounds = kullbackLeiblerCollapseBounds(*density);

    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(std::log(0.3), first().mean(), first().covarianceLower());
    mixture.add(std::log(0.7), far.mean(), far.covarianceLower());
    auto const mixtureDensity = mixture.toDensity();
    ASSERT_TRUE(mixtureDensity);
    auto const expected = kullbackLeiblerCollapseBounds(*mixtureDensity);
    EXPECT_GT(expected.lower, 0);
    EXPECT_NEAR(bounds.upper, expected.upper, 1e-14);
    EXPECT_NEAR(bounds.lower, expected.lower, 1e-14);
}

// A mixture of hybrid states is bracketed as the mixture of every component's conditionals, each
// weighted by its component's weight times its mode's probability, to rounding. Every mode's
// Gaussian differs, since copies of one Gaussian would raise the entropy of the weights and so lower
// the lower end, which depends on how the mixture is written.
TEST(KullbackLeiblerCollapseBounds, MixtureOfHybridsIsTheMixtureOfAllConditionals)
{
    using Mode = Finite<2>;
    auto const third = Plane{Eigen::Vector2d{-3.0, 2.0}, Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.5}}};
    auto const fourth = Plane{Eigen::Vector2d{6.0, -1.0}, Eigen::Matrix2d::Identity()};
    auto left =
        HybridGaussianParams<Linear<2>, Mode>{CategoricalParams<Mode>::fromWeights(Eigen::Vector2d{0.4, 0.6}), first()};
    left.setConditional(1, second());
    auto right =
        HybridGaussianParams<Linear<2>, Mode>{CategoricalParams<Mode>::fromWeights(Eigen::Vector2d{0.9, 0.1}), third};
    right.setConditional(1, fourth);
    auto hybrids = MixtureParams<HybridGaussian<Linear<2>, Mode>>{Linear<2>{}};
    hybrids.add(std::log(0.25), left);
    hybrids.add(std::log(0.75), right);
    auto const density = hybrids.toDensity();
    ASSERT_TRUE(density);
    auto const bounds = kullbackLeiblerCollapseBounds(*density);

    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(std::log(0.25 * 0.4), first().mean(), first().covarianceLower());
    mixture.add(std::log(0.25 * 0.6), second().mean(), second().covarianceLower());
    mixture.add(std::log(0.75 * 0.9), third.mean(), third.covarianceLower());
    mixture.add(std::log(0.75 * 0.1), fourth.mean(), fourth.covarianceLower());
    auto const mixtureDensity = mixture.toDensity();
    ASSERT_TRUE(mixtureDensity);
    auto const expected = kullbackLeiblerCollapseBounds(*mixtureDensity);
    EXPECT_GT(expected.lower, 0);
    EXPECT_NEAR(bounds.upper, expected.upper, 1e-13);
    EXPECT_NEAR(bounds.lower, expected.lower, 1e-13);
}

// bhattacharyya() between Gaussians matches the closed form, is symmetric, is zero against itself,
// and its Monte Carlo estimate agrees within five standard errors.
TEST(Bhattacharyya, GaussiansMatchTheClosedForm)
{
    Plane const p = first();
    Plane const q = second();
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(bhattacharyya(*pDensity, *qDensity), handBhattacharyya(p, q), 1e-14);
    EXPECT_NEAR(bhattacharyya(*qDensity, *pDensity), bhattacharyya(*pDensity, *qDensity), 1e-14);
    EXPECT_EQ(bhattacharyya(*pDensity, *pDensity), 0);
    auto rng = std::mt19937{21};
    expectWithin(bhattacharyya(MonteCarlo{100'000}, *pDensity, *qDensity, rng), handBhattacharyya(p, q));
}

// hellinger() between one-dimensional Gaussians matches the textbook formula, is zero against
// itself, and its Monte Carlo estimate agrees.
TEST(Hellinger, OneDimensionalGaussiansMatchTheTextbookForm)
{
    // H^2 = 1 - sqrt(2 s t / (s^2 + t^2)) exp(-(m - n)^2 / (4 (s^2 + t^2))).
    double const s = 0.7;
    double const t = 1.6;
    double const m = 0.4;
    double const n = -0.9;
    auto const p = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{m}, Eigen::Matrix<double, 1, 1>{s * s}};
    auto const q = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{n}, Eigen::Matrix<double, 1, 1>{t * t}};
    double const expected =
        std::sqrt(1 - std::sqrt(2 * s * t / (s * s + t * t)) * std::exp(-(m - n) * (m - n) / (4 * (s * s + t * t))));
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(hellinger(*pDensity, *qDensity), expected, 1e-14);
    EXPECT_EQ(hellinger(*pDensity, *pDensity), 0);
    auto rng = std::mt19937{31};
    expectWithin(hellinger(MonteCarlo{100'000}, *pDensity, *qDensity, rng), expected);
}

// For Gaussians 100 standard deviations apart the Bhattacharyya distance is d^2 / 8 and the
// Hellinger distance rounds to exactly one.
TEST(Hellinger, FarApartGaussiansApproachOne)
{
    auto const p = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    auto const q = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{100.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(bhattacharyya(*pDensity, *qDensity), 100.0 * 100.0 / 8, 1e-9);
    EXPECT_EQ(hellinger(*pDensity, *qDensity), 1);
}

// Between categoricals, Bhattacharyya is minus the log of the Bhattacharyya coefficient, Hellinger
// is the square root of one minus it, and Jensen-Shannon matches the hand sum, is symmetric, and is
// zero against itself.
TEST(Divergences, CategoricalsMatchHandSums)
{
    auto const pWeights = CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.5, 0.3, 0.2, 0.0});
    auto const qWeights = CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.1, 0.2, 0.3, 0.4});
    auto const p = pWeights.toDensity().value();
    auto const q = qWeights.toDensity().value();
    auto const pv = Eigen::Vector4d{0.5, 0.3, 0.2, 0.0};
    auto const qv = Eigen::Vector4d{0.1, 0.2, 0.3, 0.4};
    double const coefficient = pv.cwiseProduct(qv).cwiseSqrt().sum();
    EXPECT_NEAR(bhattacharyya(p, q), -std::log(coefficient), 1e-15);
    EXPECT_NEAR(hellinger(p, q), std::sqrt(1 - coefficient), 1e-15);
    double js = 0;
    for (int i = 0; i < 4; ++i)
    {
        double const m = 0.5 * (pv(i) + qv(i));
        js += pv(i) > 0 ? 0.5 * pv(i) * std::log(pv(i) / m) : 0;
        js += qv(i) > 0 ? 0.5 * qv(i) * std::log(qv(i) / m) : 0;
    }
    EXPECT_NEAR(jensenShannon(p, q), js, 1e-15);
    EXPECT_NEAR(jensenShannon(q, p), js, 1e-15);
    EXPECT_EQ(jensenShannon(p, p), 0);
}

// Disjoint categoricals give infinite Kullback-Leibler and Bhattacharyya, Hellinger one, and
// Jensen-Shannon log 2.
TEST(Divergences, DisjointCategoricalsReachTheirBounds)
{
    auto const pWeights = CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.5, 0.5, 0.0, 0.0});
    auto const qWeights = CategoricalParams<Finite<4>>::fromWeights(Finite<4>{}, Eigen::Vector4d{0.0, 0.0, 0.5, 0.5});
    auto const p = pWeights.toDensity().value();
    auto const q = qWeights.toDensity().value();
    EXPECT_EQ(kullbackLeibler(p, q), inf);
    EXPECT_EQ(bhattacharyya(p, q), inf);
    EXPECT_EQ(hellinger(p, q), 1);
    EXPECT_NEAR(jensenShannon(p, q), std::log(2.0), 1e-15);
}

// The Monte Carlo Jensen-Shannon estimate is zero against itself, between zero and log 2 for two
// Gaussians, symmetric within the two estimates' errors, and log 2 for Gaussians far apart.
TEST(JensenShannon, GaussiansAreEstimated)
{
    Plane const p = first();
    Plane const q = second();
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    auto rng = std::mt19937{41};
    auto const same = jensenShannon(MonteCarlo{20'000}, *pDensity, *pDensity, rng);
    EXPECT_NEAR(same.value, 0, 1e-12);
    auto const between = jensenShannon(MonteCarlo{50'000}, *pDensity, *qDensity, rng);
    EXPECT_GT(between.value, 0);
    EXPECT_LT(between.value, std::log(2.0));
    // Symmetric: the reversed estimate agrees within the two estimates' errors.
    auto const reversed = jensenShannon(MonteCarlo{50'000}, *qDensity, *pDensity, rng);
    EXPECT_NEAR(between.value, reversed.value, 5 * std::hypot(between.standardError, reversed.standardError));

    auto const near = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    auto const far = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{50.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    EXPECT_NEAR(
        jensenShannon(MonteCarlo{1'000}, near.toDensity().value(), far.toDensity().value(), rng).value,
        std::log(2.0),
        1e-12
    );
}

// The 2-Wasserstein distance between one-dimensional Gaussians is the hypotenuse of the mean
// difference and the standard deviation difference.
TEST(Wasserstein, OneDimensionalGaussiansMatchTheTextbookForm)
{
    auto const p = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.4}, Eigen::Matrix<double, 1, 1>{0.49}};
    auto const q = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{-0.9}, Eigen::Matrix<double, 1, 1>{2.56}};
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(wasserstein(*pDensity, *qDensity), std::hypot(1.3, 1.6 - 0.7), 1e-14);
}

// wasserstein() between Gaussians matches the matrix square-root formula computed with an
// eigendecomposition, is symmetric, and is zero against itself up to rounding in the trace terms.
TEST(Wasserstein, GaussiansMatchTheMatrixSqrtForm)
{
    Plane const p = first();
    Plane const q = second();
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(wasserstein(*pDensity, *qDensity), handWasserstein(p, q), 1e-12);
    EXPECT_NEAR(wasserstein(*qDensity, *pDensity), handWasserstein(p, q), 1e-12);
    // Rounding in the trace terms leaves a residue of order the square root of epsilon.
    EXPECT_NEAR(wasserstein(*pDensity, *pDensity), 0, 1e-7);
}

// A weight matrix W gives the Wasserstein distance between W p and W q, while the divergences that
// are invariant under invertible maps are unchanged by W.
TEST(Wasserstein, WeightsActAsAMapAppliedToBoth)
{
    Plane const p = first();
    Plane const q = second();
    auto const w = Eigen::Matrix2d{{2.0, 0.5}, {-1.0, 3.0}};
    auto const mappedP = Plane{Eigen::Vector2d{w * p.mean()}, Eigen::Matrix2d{w * p.covarianceLower() * w.transpose()}};
    auto const mappedQ = Plane{Eigen::Vector2d{w * q.mean()}, Eigen::Matrix2d{w * q.covarianceLower() * w.transpose()}};
    EXPECT_NEAR(wasserstein(p.toDensity().value(), q.toDensity().value(), w), handWasserstein(mappedP, mappedQ), 1e-12);
    // The divergences don't change under the map.
    EXPECT_NEAR(
        kullbackLeibler(mappedP.toDensity().value(), mappedQ.toDensity().value()),
        handKullbackLeibler(p, q),
        1e-12
    );
    EXPECT_NEAR(
        hellinger(mappedP.toDensity().value(), mappedQ.toDensity().value()),
        hellinger(p.toDensity().value(), q.toDensity().value()),
        1e-12
    );
}

// combine<Position, Velocity>(T) then scale<Velocity>(0) builds the matrix [I, T I; 0, 0] of the
// position T ahead, and the operations apply in the order given.
TEST(GroundMetric, BuildsTheLookAheadMap)
{
    double const lookAhead = 2.5;
    auto metric = GroundMetric{State{}};
    EXPECT_EQ(metric.space(), State{});
    metric.combine<Position, Velocity>(lookAhead);
    metric.scale<Velocity>(0);
    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    expected.topLeftCorner<2, 2>().setIdentity();
    expected.topRightCorner<2, 2>() = lookAhead * Eigen::Matrix2d::Identity();
    EXPECT_EQ(metric.matrix(), expected);

    // Applied in order: scaling Velocity first leaves nothing of it to combine.
    auto reversed = GroundMetric{State{}};
    reversed.scale<Velocity>(0);
    reversed.combine<Position, Velocity>(lookAhead);
    Eigen::Matrix4d positionOnly = Eigen::Matrix4d::Zero();
    positionOnly.topLeftCorner<2, 2>().setIdentity();
    EXPECT_EQ(reversed.matrix(), positionOnly);
}

// Wasserstein under the look-ahead metric is the distance between the Gaussians of the predicted
// positions, a reduced-row matrix measures the same, and scaling Velocity to zero alone is the
// distance between the position marginals.
TEST(GroundMetric, MeasuresPredictedPosition)
{
    double const lookAhead = 2.5;
    auto pCov =
        Eigen::Matrix4d{{1.0, 0.2, 0.3, 0.0}, {0.2, 2.0, 0.0, -0.1}, {0.3, 0.0, 0.5, 0.05}, {0.0, -0.1, 0.05, 0.4}};
    auto qCov =
        Eigen::Matrix4d{{1.5, -0.3, 0.1, 0.0}, {-0.3, 1.0, 0.0, 0.2}, {0.1, 0.0, 0.3, 0.0}, {0.0, 0.2, 0.0, 0.6}};
    auto const p = GaussianParams<State>{Eigen::Vector4d{1.0, 2.0, 0.5, -0.3}, pCov};
    auto const q = GaussianParams<State>{Eigen::Vector4d{0.4, 2.5, 0.1, 0.2}, qCov};
    auto metric = GroundMetric{State{}};
    metric.combine<Position, Velocity>(lookAhead);
    metric.scale<Velocity>(0);

    // The position predicted lookAhead ahead, p + T v, as a Gaussian of its own.
    auto predict = Eigen::Matrix<double, 2, 4>{};
    predict << Eigen::Matrix2d::Identity(), lookAhead * Eigen::Matrix2d::Identity();
    auto const predictedP =
        Plane{Eigen::Vector2d{predict * p.mean()}, Eigen::Matrix2d{predict * pCov * predict.transpose()}};
    auto const predictedQ =
        Plane{Eigen::Vector2d{predict * q.mean()}, Eigen::Matrix2d{predict * qCov * predict.transpose()}};
    double const expected = handWasserstein(predictedP, predictedQ);

    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    EXPECT_NEAR(wasserstein(*pDensity, *qDensity, metric), expected, 1e-12);
    // A matrix of fewer rows measures the same.
    EXPECT_NEAR(wasserstein(*pDensity, *qDensity, predict), expected, 1e-12);

    // Scaling Velocity to zero alone is the distance between the position marginals.
    auto positionOnly = GroundMetric{State{}};
    positionOnly.scale<Velocity>(0);
    GaussianParams<Position> const pPosition = p.marginal<Position>().copy();
    GaussianParams<Position> const qPosition = q.marginal<Position>().copy();
    EXPECT_NEAR(
        wasserstein(*pDensity, *qDensity, positionOnly),
        wasserstein(pPosition.toDensity().value(), qPosition.toDensity().value()),
        1e-12
    );
}

// Gaussians on headings either side of the seam are 0.2 apart to every divergence and to
// expectedSquaredError(), as a linear pair 0.2 apart would be.
TEST(Divergences, PeriodicMeansDifferAcrossTheSeam)
{
    // Headings on either side of the seam are 0.2 apart, as in a linear space.
    auto const p = GaussianParams<Heading>{Eigen::Matrix<double, 1, 1>{pi - 0.1}, Eigen::Matrix<double, 1, 1>{0.01}};
    auto const q = GaussianParams<Heading>{Eigen::Matrix<double, 1, 1>{-pi + 0.1}, Eigen::Matrix<double, 1, 1>{0.02}};
    auto const linearP = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.01}};
    auto const linearQ = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.2}, Eigen::Matrix<double, 1, 1>{0.02}};
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    auto const linearPDensity = linearP.toDensity();
    auto const linearQDensity = linearQ.toDensity();
    EXPECT_NEAR(kullbackLeibler(*pDensity, *qDensity), kullbackLeibler(*linearPDensity, *linearQDensity), 1e-12);
    EXPECT_NEAR(bhattacharyya(*pDensity, *qDensity), bhattacharyya(*linearPDensity, *linearQDensity), 1e-12);
    EXPECT_NEAR(wasserstein(*pDensity, *qDensity), wasserstein(*linearPDensity, *linearQDensity), 1e-12);
    EXPECT_NEAR(expectedSquaredError(p, Eigen::Matrix<double, 1, 1>{-pi + 0.1}), 0.2 * 0.2 + 0.01, 1e-12);
}

// expectedSquaredError() of a Gaussian is the squared mean error plus the covariance trace, with a
// weight matrix applied to both, and equals the squared Wasserstein distance to a point mass at the
// truth.
TEST(ExpectedSquaredError, GaussianMatchesItsMoments)
{
    Plane const p = first();
    auto const truth = Eigen::Vector2d{0.5, 0.5};
    EXPECT_NEAR(expectedSquaredError(p, truth), (p.mean() - truth).squaredNorm() + p.covarianceLower().trace(), 1e-14);
    auto const w = Eigen::Matrix2d{{2.0, 0.5}, {-1.0, 3.0}};
    EXPECT_NEAR(
        expectedSquaredError(p, truth, w),
        (w * (p.mean() - truth)).squaredNorm() + (w * p.covarianceLower() * w.transpose()).trace(),
        1e-13
    );
    // The squared distance to a narrow Gaussian at the truth approaches it.
    auto const point = Plane{truth, Eigen::Matrix2d{1e-14 * Eigen::Matrix2d::Identity()}};
    double const distance = wasserstein(p.toDensity().value(), point.toDensity().value());
    EXPECT_NEAR(distance * distance, expectedSquaredError(p, truth), 1e-5);
}

// For a mixture under a ground metric, expectedSquaredError() lies within five standard errors of
// the sample average of the weighted squared errors over 100,000 draws.
TEST(ExpectedSquaredError, MixtureMatchesTheSampleAverage)
{
    auto mixture = GaussianMixtureParams<State>{4};
    mixture.add(std::log(0.7), Eigen::Vector4d{1.0, 0.0, 0.5, 0.0}, Eigen::Matrix4d{0.2 * Eigen::Matrix4d::Identity()});
    mixture
        .add(std::log(0.3), Eigen::Vector4d{-2.0, 1.0, 0.0, 1.0}, Eigen::Matrix4d{0.5 * Eigen::Matrix4d::Identity()});
    auto const truth = Eigen::Vector4d{0.0, 0.0, 0.0, 0.0};
    auto metric = GroundMetric{State{}};
    metric.combine<Position, Velocity>(2.0);
    metric.scale<Velocity>(0);
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    double const expected = expectedSquaredError(*density, truth, metric);

    auto rng = std::mt19937{51};
    auto const samples = drawSamples(*density, 100'000, rng);
    Eigen::ArrayXd const errors =
        (metric.matrix() * (samples.colwise() - truth)).colwise().squaredNorm().transpose().array();
    double const mean = errors.mean();
    double const standardError =
        std::sqrt((errors - mean).square().sum() / double(errors.size() - 1) / double(errors.size()));
    EXPECT_NEAR(expected, mean, 5 * standardError);
}

// For a particle set the result is an expected (the weights may be invalid) equal to the squared
// mean error plus the sample covariance trace.
TEST(ExpectedSquaredError, ParticleSetReturnsAnExpected)
{
    auto rng = std::mt19937{61};
    Plane const p = first();
    auto const particles = ParticleSet<Linear<2>>{drawSamples(p.toDensity().value(), 500, rng)};
    auto const truth = Eigen::Vector2d{0.0, 1.0};
    auto const error = expectedSquaredError(particles, truth);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(error)>, std::expected<double, Emrys::WeightError>>);
    ASSERT_TRUE(error);
    auto const moments = particles.moments();
    EXPECT_NEAR(*error, (moments->mean() - truth).squaredNorm() + moments->covarianceLower().trace(), 1e-12);
}

// A particle set whose weights fail to define a distribution has no moments, and so no expected
// squared error.
TEST(ExpectedSquaredError, ParticleSetWithInvalidWeightsFails)
{
    auto const particles =
        ParticleSet<Linear<2>>{Linear<2>{}, Eigen::Matrix<double, 2, 2>{{0, 1}, {0, 1}}, Eigen::Array2d{-inf, -inf}};
    EXPECT_EQ(expectedSquaredError(particles, Eigen::Vector2d::Zero()).error(), Emrys::WeightError::ZeroSum);
}

// The Monte Carlo divergence is infinite as soon as a draw from p has no density under q, and the
// Hellinger distance between a density and itself is zero, with a standard error of zero, since
// every term of the coefficient is one.
TEST(Divergences, MonteCarloEdges)
{
    auto rng = std::mt19937{71};
    Plane const params = first();
    auto const p = params.toDensity().value();
    auto const narrow = Emrys::Uniform<Linear<2>>::create(Eigen::Vector2d{50.0, 50.0}, Eigen::Vector2d{51.0, 51.0});
    ASSERT_TRUE(narrow);
    auto const infinite = kullbackLeibler(MonteCarlo{10}, p, *narrow, rng);
    EXPECT_EQ(infinite.value, std::numeric_limits<double>::infinity());

    auto const same = hellinger(MonteCarlo{10}, p, p, rng);
    EXPECT_EQ(same.value, 0.0);
    EXPECT_LT(same.standardError, 1e-6);
}

// Kullback-Leibler, Bhattacharyya, Hellinger, and Wasserstein between two square-root forms, read
// from their factors, are those between the covariance forms, to rounding.
TEST(Divergences, SqrtFormsMatchTheCovarianceForms)
{
    Plane const pCovariance = first();
    Plane const qCovariance = second();
    auto const p = Emrys::toSqrt(pCovariance).value();
    auto const q = Emrys::toSqrt(qCovariance).value();
    auto const pSqrt = p.toDensity();
    auto const qSqrt = q.toDensity();
    auto const pDensity = pCovariance.toDensity();
    auto const qDensity = qCovariance.toDensity();
    ASSERT_TRUE(pSqrt && qSqrt && pDensity && qDensity);
    EXPECT_NEAR(kullbackLeibler(*pSqrt, *qSqrt), kullbackLeibler(*pDensity, *qDensity), 1e-12);
    EXPECT_NEAR(kullbackLeibler(*qSqrt, *pSqrt), kullbackLeibler(*qDensity, *pDensity), 1e-12);
    EXPECT_NEAR(bhattacharyya(*pSqrt, *qSqrt), bhattacharyya(*pDensity, *qDensity), 1e-12);
    EXPECT_NEAR(hellinger(*pSqrt, *qSqrt), hellinger(*pDensity, *qDensity), 1e-12);
    EXPECT_NEAR(wasserstein(*pSqrt, *qSqrt), wasserstein(*pDensity, *qDensity), 1e-12);
    EXPECT_EQ(kullbackLeibler(*pSqrt, *pSqrt), 0);
}

// Each divergence in closed form takes the densities of the four forms in any pair, and gives
// what it gives between the two covariance forms, to rounding: a square-root filter's state
// against a covariance-form reference, an information form against either, in either order, and
// Wasserstein under a ground metric too.
TEST(Divergences, AnyPairOfFormsMatchesTheCovarianceForms)
{
    Plane const pCovariance = first();
    Plane const qCovariance = second();
    auto const pSqrtParams = Emrys::toSqrt(pCovariance).value();
    auto const qSqrtParams = Emrys::toSqrt(qCovariance).value();
    auto const pInfoParams = Emrys::toInfo(pCovariance).value();
    auto const qInfoParams = Emrys::toInfo(qCovariance).value();
    auto const pSqrtInfoParams = Emrys::toSqrtInfo(pCovariance).value();
    auto const qSqrtInfoParams = Emrys::toSqrtInfo(qCovariance).value();
    auto const p = pCovariance.toDensity().value();
    auto const q = qCovariance.toDensity().value();
    auto const pSqrt = pSqrtParams.toDensity().value();
    auto const qSqrt = qSqrtParams.toDensity().value();
    auto const pInfo = pInfoParams.toDensity().value();
    auto const qInfo = qInfoParams.toDensity().value();
    auto const pSqrtInfo = pSqrtInfoParams.toDensity().value();
    auto const qSqrtInfo = qSqrtInfoParams.toDensity().value();
    Eigen::Matrix2d const weights{{2.0, 0.5}, {0.0, 1.0}};

    double const kullbackLeiblerPQ = kullbackLeibler(p, q);
    double const kullbackLeiblerQP = kullbackLeibler(q, p);
    double const bhattacharyyaPQ = bhattacharyya(p, q);
    double const hellingerPQ = hellinger(p, q);
    double const wassersteinPQ = wasserstein(p, q);
    double const weightedPQ = wasserstein(p, q, weights);
    auto const expectMatch = [&](auto const & pForm, auto const & qForm) {
        EXPECT_NEAR(kullbackLeibler(pForm, qForm), kullbackLeiblerPQ, 1e-12);
        EXPECT_NEAR(kullbackLeibler(qForm, pForm), kullbackLeiblerQP, 1e-12);
        EXPECT_NEAR(bhattacharyya(pForm, qForm), bhattacharyyaPQ, 1e-12);
        EXPECT_NEAR(bhattacharyya(qForm, pForm), bhattacharyyaPQ, 1e-12);
        EXPECT_NEAR(hellinger(pForm, qForm), hellingerPQ, 1e-12);
        EXPECT_NEAR(wasserstein(pForm, qForm), wassersteinPQ, 1e-12);
        EXPECT_NEAR(wasserstein(qForm, pForm), wassersteinPQ, 1e-12);
        EXPECT_NEAR(wasserstein(pForm, qForm, weights), weightedPQ, 1e-12);
    };
    expectMatch(pSqrt, q);
    expectMatch(p, qSqrt);
    expectMatch(pInfo, qInfo);
    expectMatch(pInfo, q);
    expectMatch(pSqrt, qInfo);
    expectMatch(pSqrtInfo, qSqrtInfo);
    expectMatch(pSqrtInfo, q);
    expectMatch(pSqrtInfo, qInfo);
    EXPECT_NEAR(kullbackLeibler(pInfo, p), 0, 1e-12);
    EXPECT_NEAR(kullbackLeibler(pSqrtInfo, pSqrt), 0, 1e-12);
}

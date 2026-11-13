// The non-Gaussian distributions Uniform, Gamma, and StudentT (step 12, [density-names]): each log-density
// against its formula, the factories' checks of their parameters [checked-factories], the Gamma special cases
// (exponential, chi-squared) and its mean/standard-deviation factory, the Student-t's Gaussian
// limit, moments, and sampling; intervals from the space, across a periodic seam, and on a
// resizable space; and the marginals and conditionals each keeps its parameters through; and
// drawSample(), one draw at a time, for each [allocation].

#include <EmrysFilter/Distributions/Gamma.h>
#include <EmrysFilter/Distributions/StudentT.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Distributions/Uniform.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>

#include "Distributions/draw_stream.h"

using Emrys::Composite;
using Emrys::Degrees360;
using Emrys::Gamma;
using Emrys::GammaError;
using Emrys::Linear;
using Emrys::MomentsError;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::StudentT;
using Emrys::StudentTError;
using Emrys::TagPath;
using Emrys::Uniform;
using Emrys::UniformError;

namespace
{

constexpr double pi = std::numbers::pi;
constexpr double inf = std::numeric_limits<double>::infinity();

struct Heading : RadiansPi<>
{};

struct Compass : Degrees360<>
{};

struct Range : Linear<1>
{};

struct Speed : Linear<1>
{};

// Non-Gaussian densities report moments() but not mean()/covarianceLower(), so transform() never mistakes them for
// Gaussians.
static_assert(!Emrys::UncertainInput<Uniform<Linear<2>>>);

static_assert(!Emrys::UncertainInput<Gamma<Linear<2>>>);
static_assert(!Emrys::UncertainInput<StudentT<Linear<2>>>);

// Parameters that can fail to define the distribution are given only to a factory that checks
// them: Gamma and a Uniform box have no public constructor, while a Uniform over a whole period
// cannot fail and keeps one [checked-factories].
static_assert(!std::is_constructible_v<Gamma<Range>, double, double>);
static_assert(!std::is_constructible_v<Uniform<Linear<2>>, Eigen::Vector2d, Eigen::Vector2d>);
static_assert(std::is_constructible_v<Uniform<Heading>>);
static_assert(!std::is_constructible_v<StudentT<Linear<2>>, Emrys::GaussianParams<Linear<2>>, double>);

// A Gamma needs every coordinate to hold [0, inf) unfolded, which the space's type decides, so
// Gamma<Heading> fails to compile: every block of a Composite or Repeated must hold it.
static_assert(Emrys::internal::holdsHalfLine<Composite<Range, Repeated<Speed>>>());
static_assert(!Emrys::internal::holdsHalfLine<Heading>());
static_assert(!Emrys::internal::holdsHalfLine<Composite<Range, Heading>>());
static_assert(!Emrys::internal::holdsHalfLine<Repeated<Compass>>());

}  // namespace

// A Uniform with no interval covers each periodic coordinate's period, and contains every
// value, since a value outside the period wraps into it.
TEST(Uniform, DefaultIntervalComesFromTheSpace)
{
    auto const heading = Uniform<Heading>{};
    EXPECT_NEAR(heading.lower()(0), -pi, 1e-15);
    EXPECT_NEAR(heading.width()(0), 2 * pi, 1e-15);
    EXPECT_NEAR(heading.upper()(0), pi, 1e-15);
    EXPECT_NEAR(heading.logDensity(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(2 * pi), 1e-15);

    using Box = Composite<Heading, Compass>;
    auto const box = Uniform<Box>{Box{}};
    EXPECT_NEAR(box.logVolume(), std::log(2 * pi * 360.0), 1e-14);
    EXPECT_TRUE(box.contains(Eigen::Vector2d{0.0, 90.0}));
    EXPECT_TRUE(box.contains(Eigen::Vector2d{7.0, 450.0}));
}

// A uniform on an explicit closed box has log-density minus the log of its volume inside and minus
// infinity outside, the moments of a uniform (center, width squared over 12), and samples inside
// the box with matching sample moments.
TEST(Uniform, ExplicitBoxInALinearSpace)
{
    auto const uniform = Uniform<Linear<2>>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    EXPECT_NEAR(uniform.logDensity(Eigen::Vector2d{0.5, 3.0}), -std::log(8.0), 1e-15);
    EXPECT_EQ(uniform.logDensity(Eigen::Vector2d{1.5, 3.0}), -inf);
    EXPECT_NEAR(uniform.density(Eigen::Vector2d{-1.0, 4.0}), 0.125, 1e-15);  // the box is closed

    auto const moments = uniform.moments();
    EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector2d{0.0, 2.0}));
    EXPECT_TRUE(moments.covarianceLower().isApprox(Eigen::Vector2d{4.0 / 12, 16.0 / 12}.asDiagonal().toDenseMatrix()));

    auto rng = std::mt19937_64{7};
    auto const samples = Emrys::drawSamples(uniform, 40000, rng);
    EXPECT_TRUE((samples.row(0).array() >= -1.0).all() && (samples.row(0).array() < 1.0).all());
    EXPECT_TRUE((samples.row(1).array() >= 0.0).all() && (samples.row(1).array() < 4.0).all());
    // The seed is fixed; the tolerances are about seven standard errors of the sample mean at
    // this count (the standard deviations are 0.58 and 1.15).
    Eigen::Vector2d const mean = samples.rowwise().mean();
    EXPECT_NEAR(mean(0), 0.0, 0.02);
    EXPECT_NEAR(mean(1), 2.0, 0.04);
    Eigen::Matrix<double, 2, Eigen::Dynamic> const centered = samples.colwise() - mean;
    EXPECT_NEAR((centered.row(1).array().square().mean()), 16.0 / 12, 0.03);
}

// An interval on an angle that runs past pi wraps: contains() and the samples cover both sides of
// the seam, the mean is the normalized center, and every sample has the interval's density.
TEST(Uniform, PeriodicIntervalCrossesTheSeam)
{
    // [3, 3.5] on an angle wraps past π to 3.5 - 2π.
    auto const arc =
        Uniform<Heading>::create(Heading{}, Eigen::Matrix<double, 1, 1>{3.0}, Eigen::Matrix<double, 1, 1>{3.5}).value();
    EXPECT_TRUE(arc.contains(Eigen::Matrix<double, 1, 1>{3.1}));
    EXPECT_TRUE(arc.contains(Eigen::Matrix<double, 1, 1>{3.4 - 2 * pi}));
    EXPECT_FALSE(arc.contains(Eigen::Matrix<double, 1, 1>{0.0}));
    EXPECT_FALSE(arc.contains(Eigen::Matrix<double, 1, 1>{2.9}));
    EXPECT_NEAR(arc.moments().mean()(0), 3.25 - 2 * pi, 1e-14);  // the center, normalized

    auto rng = std::mt19937_64{3};
    auto const drawn = Emrys::drawSamplesWithLogDensity(arc, 2000, rng);
    bool wrapped = false;
    for (Eigen::Index j = 0; j < drawn.samples.cols(); ++j)
    {
        double const x = drawn.samples(0, j);
        EXPECT_TRUE(x >= -pi && x < pi);
        EXPECT_TRUE(arc.contains(drawn.samples.col(j)));
        EXPECT_NEAR(drawn.logDensity(j), -std::log(0.5), 1e-14);
        wrapped = wrapped || x < 0;
    }
    EXPECT_TRUE(wrapped);
}

// On a dynamic space the box's size sets the dimension.
TEST(Uniform, ResizableSpaceTakesTheBoxSize)
{
    auto const uniform =
        Uniform<Linear<Eigen::Dynamic>>::create(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()).value();
    EXPECT_EQ(uniform.valueDimension(), 3);
    EXPECT_EQ(uniform.space().valueDimension(), 3);
    EXPECT_NEAR(uniform.logDensity(Eigen::Vector3d::Constant(0.5)), 0.0, 1e-15);
}

// create() reports bounds that fail to define a box: an upper bound at or below its lower one,
// a bound that is not finite, and an interval on an angle wider than its period.
TEST(Uniform, CreateChecksTheBox)
{
    auto const at = [](double lower, double upper) {
        return Uniform<Linear<2>>::create(Eigen::Vector2d{0.0, lower}, Eigen::Vector2d{1.0, upper});
    };
    EXPECT_TRUE(at(0.0, 1.0));
    EXPECT_EQ(at(1.0, 1.0).error(), UniformError::Empty);
    EXPECT_EQ(at(2.0, 1.0).error(), UniformError::Empty);
    EXPECT_EQ(at(0.0, inf).error(), UniformError::NonFinite);
    EXPECT_EQ(at(NAN, 1.0).error(), UniformError::NonFinite);
    auto const angle = [](double lower, double upper) {
        return Uniform<Heading>::create(Eigen::Matrix<double, 1, 1>{lower}, Eigen::Matrix<double, 1, 1>{upper});
    };
    EXPECT_TRUE(angle(-pi, pi));
    EXPECT_EQ(angle(-pi, pi + 0.1).error(), UniformError::WiderThanPeriod);
}

// The factories report a shape or scale that is not positive and finite, and fromMeanAndStdDev()
// a mean or standard deviation that is not, before dividing by it.
TEST(Gamma, FactoriesCheckTheParams)
{
    EXPECT_TRUE(Gamma<Range>::create(2.0, 3.0));
    EXPECT_EQ(Gamma<Range>::create(0.0, 3.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::create(2.0, -1.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::create(inf, 3.0).error(), GammaError::NonFinite);
    EXPECT_EQ(Gamma<Range>::create(2.0, NAN).error(), GammaError::NonFinite);
    EXPECT_EQ(Gamma<Linear<2>>::create(Eigen::Vector2d{1.0, 0.0}, 1.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::exponential(0.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::chiSquared(-2.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::fromMeanAndStdDev(0.0, 1.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::fromMeanAndStdDev(1.0, 0.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::fromMeanAndStdDev(-1.0, 1.0).error(), GammaError::NotPositive);
    EXPECT_EQ(Gamma<Range>::fromMeanAndStdDev(inf, 1.0).error(), GammaError::NonFinite);
}

// The log-density matches the shape-scale gamma formula at a point, is minus infinity at and below
// zero, and integrates numerically to one.
TEST(Gamma, LogDensityMatchesTheFormula)
{
    auto const gamma = Gamma<Range>::create(2.0, 3.0).value();
    // x^(k-1) e^(-x/θ) / (Γ(k) θ^k) with k = 2, θ = 3.
    EXPECT_NEAR(gamma.logDensity(Eigen::Matrix<double, 1, 1>{1.5}), std::log(1.5) - 0.5 - std::log(9.0), 1e-14);
    EXPECT_EQ(gamma.logDensity(Eigen::Matrix<double, 1, 1>{-0.1}), -inf);
    EXPECT_EQ(gamma.logDensity(Eigen::Matrix<double, 1, 1>{0.0}), -inf);

    // It integrates to one.
    auto const skewed = Gamma<Linear<1>>::create(2.5, 1.5).value();
    double integral = 0;
    double const h = 1e-3;
    for (double x = h / 2; x < 60; x += h)
    {
        integral += skewed.density(Eigen::Matrix<double, 1, 1>{x}) * h;
    }
    EXPECT_NEAR(integral, 1.0, 1e-6);
}

// exponential(mean) and chiSquared(k) are the gamma special cases, with the exponential's density
// and both distributions' moments.
TEST(Gamma, ExponentialAndChiSquared)
{
    auto const exponential = Gamma<Range>::exponential(2.0).value();
    EXPECT_NEAR(exponential.logDensity(Eigen::Matrix<double, 1, 1>{0.0}), -std::log(2.0), 1e-15);
    EXPECT_NEAR(exponential.logDensity(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(2.0) - 1.5, 1e-15);
    EXPECT_NEAR(exponential.moments().mean()(0), 2.0, 1e-15);
    EXPECT_NEAR(exponential.moments().covarianceLower()(0, 0), 4.0, 1e-15);

    auto const chiSquared = Gamma<Linear<1>>::chiSquared(4.0).value();
    EXPECT_NEAR(chiSquared.moments().mean()(0), 4.0, 1e-15);
    EXPECT_NEAR(chiSquared.moments().covarianceLower()(0, 0), 8.0, 1e-15);
}

// fromMeanAndStdDev() solves shape = (mean / sd)^2 per coordinate, gives independent coordinates
// with those moments, and its samples are non-negative with matching sample means and
// log-densities.
TEST(Gamma, FromMeanAndStdDevPerCoordinate)
{
    auto const gamma =
        Gamma<Linear<2>>::fromMeanAndStdDev(Eigen::Vector2d{100.0, 5.0}, Eigen::Vector2d{20.0, 5.0}).value();
    EXPECT_NEAR(gamma.shape()(0), 25.0, 1e-12);
    EXPECT_NEAR(gamma.shape()(1), 1.0, 1e-12);
    auto const moments = gamma.moments();
    EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector2d{100.0, 5.0}));
    EXPECT_TRUE(moments.covarianceLower().isApprox(Eigen::Vector2d{400.0, 25.0}.asDiagonal().toDenseMatrix()));
    EXPECT_EQ(moments.covarianceLower()(0, 1), 0.0);

    auto rng = std::mt19937_64{11};
    auto const drawn = Emrys::drawSamplesWithLogDensity(gamma, 40000, rng);
    EXPECT_TRUE((drawn.samples.array() >= 0).all());
    // Four to five standard errors of the sample mean (standard deviations 20 and 5) at this count.
    Eigen::Vector2d const mean = drawn.samples.rowwise().mean();
    EXPECT_NEAR(mean(0), 100.0, 0.5);
    EXPECT_NEAR(mean(1), 5.0, 0.1);
    for (Eigen::Index j = 0; j < 10; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), gamma.logDensity(drawn.samples.col(j)), 1e-12);
    }
}

// The one-dimensional log-density matches the Student-t formula with the lgamma normalizer, and the
// density is its exponential, whether the distribution is made from a location and scale or from a
// Gaussian's.
TEST(StudentT, LogDensityMatchesTheFormula)
{
    auto const t = StudentT<Linear<1>>::create(Eigen::Matrix<double, 1, 1>{1.0}, Eigen::Matrix<double, 1, 1>{4.0}, 3.0);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(t->valueDimension(), 1);
    auto const fromGaussian = StudentT<Linear<1>>::create(
        Emrys::GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{1.0}, Eigen::Matrix<double, 1, 1>{4.0}},
        3.0
    );
    ASSERT_TRUE(fromGaussian.has_value());
    // Γ(2) / (Γ(3/2) √(3π) σ) (1 + ((x - μ)/σ)² / 3)^-2 with σ = 2.
    double const x = 2.5;
    double const z = (x - 1.0) / 2.0;
    double const expected = -std::lgamma(1.5) - 0.5 * std::log(3 * pi) - std::log(2.0) - 2.0 * std::log1p(z * z / 3);
    EXPECT_NEAR(t->logDensity(Eigen::Matrix<double, 1, 1>{x}), expected, 1e-14);
    EXPECT_NEAR(fromGaussian->density(Eigen::Matrix<double, 1, 1>{x}), std::exp(expected), 1e-14);
}

// At 1e8 degrees of freedom the density matches the Gaussian's near the center; at 2 it is far
// above the Gaussian's in the tail.
TEST(StudentT, ApproachesTheGaussianAndHasHeavierTails)
{
    auto const mu = Eigen::Vector2d{1.0, -1.0};
    auto const sigma = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto const nearlyGaussian = StudentT<Linear<2>>::create(mu, sigma, 1e8);
    auto const heavy = StudentT<Linear<2>>::create(mu, sigma, 2.0);
    auto const gaussian = Emrys::GaussianParams<Emrys::Linear<2>>{mu, sigma};
    auto const density = gaussian.toDensity();
    ASSERT_TRUE(nearlyGaussian && heavy && density);
    auto const near = Eigen::Vector2d{1.5, -0.5};
    auto const far = Eigen::Vector2d{9.0, 6.0};
    EXPECT_NEAR(nearlyGaussian->logDensity(near), density->logDensity(near), 1e-6);
    EXPECT_GT(heavy->logDensity(far), density->logDensity(far) + 10);
}

// moments() is the scale times nu / (nu - 2) above two degrees of freedom and fails with Infinite
// at two and below (the Cauchy case).
TEST(StudentT, MomentsExistOnlyAboveTwoDegreesOfFreedom)
{
    auto const sigma = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto const t = StudentT<Linear<2>>::create(Eigen::Vector2d::Zero(), sigma, 5.0);
    ASSERT_TRUE(t.has_value());
    auto const moments = t->moments();
    ASSERT_TRUE(moments.has_value());
    EXPECT_TRUE(moments->covarianceLower().isApprox(sigma * 5.0 / 3.0));

    auto const cauchy = StudentT<Linear<2>>::create(Eigen::Vector2d::Zero(), sigma, 1.0);
    ASSERT_TRUE(cauchy.has_value());
    EXPECT_EQ(cauchy->moments().error(), MomentsError::Infinite);
    auto const two = StudentT<Linear<2>>::create(Eigen::Vector2d::Zero(), sigma, 2.0);
    ASSERT_TRUE(two.has_value());
    EXPECT_EQ(two->moments().error(), MomentsError::Infinite);
}

// create() reports each way its parameters fail: an indefinite scale, a scale that is not finite,
// and degrees of freedom that are not positive or not finite, which it checks before the scale.
TEST(StudentT, CreateChecksTheScaleAndTheDegreesOfFreedom)
{
    auto const identity = Eigen::Matrix2d::Identity();
    auto const indefinite = Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}};
    auto const create = [](auto const & scale, double dof) {
        return StudentT<Linear<2>>::create(Eigen::Vector2d::Zero(), scale, dof);
    };
    EXPECT_TRUE(create(identity, 4.0));
    EXPECT_EQ(create(indefinite, 4.0).error(), StudentTError::NotPositiveDefinite);
    EXPECT_EQ(create(Eigen::Matrix2d{{1.0, 0.0}, {0.0, NAN}}, 4.0).error(), StudentTError::NonFiniteScale);
    EXPECT_EQ(create(identity, 0.0).error(), StudentTError::NotPositiveDegreesOfFreedom);
    EXPECT_EQ(create(identity, -3.0).error(), StudentTError::NotPositiveDegreesOfFreedom);
    EXPECT_EQ(create(identity, inf).error(), StudentTError::NonFiniteDegreesOfFreedom);
    EXPECT_EQ(create(indefinite, NAN).error(), StudentTError::NonFiniteDegreesOfFreedom);
}

// 100,000 samples carry the density's own log-density and have the mean and the covariance nu / (nu
// - 2) times the scale, to statistical tolerance; drawSamples() draws the same samples from the same
// generator.
TEST(StudentT, SamplesMatchTheDistribution)
{
    auto const sigma = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto const t = StudentT<Linear<2>>::create(Eigen::Vector2d{1.0, 2.0}, sigma, 8.0);
    ASSERT_TRUE(t.has_value());
    auto rng = std::mt19937_64{5};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*t, 100000, rng);
    for (Eigen::Index j = 0; j < 20; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), t->logDensity(drawn.samples.col(j)), 1e-10);
    }
    // The covariance of a Student-t is nu / (nu - 2) times the scale. The mean tolerance is four
    // standard errors at this count; the heavy tails make the sample covariance noisier, hence 3%.
    Eigen::Vector2d const mean = drawn.samples.rowwise().mean();
    EXPECT_NEAR(mean(0), 1.0, 0.02);
    EXPECT_NEAR(mean(1), 2.0, 0.02);
    Eigen::Matrix<double, 2, Eigen::Dynamic> const centered = drawn.samples.colwise() - mean;
    Eigen::Matrix2d const cov = centered * centered.transpose() / double(centered.cols());
    EXPECT_TRUE(cov.isApprox(sigma * 8.0 / 6.0, 0.03)) << cov;

    auto withDensity = std::mt19937_64{6};
    auto alone = std::mt19937_64{6};
    auto const few = Emrys::drawSamplesWithLogDensity(*t, 20, withDensity);
    auto const samples = Emrys::drawSamples(*t, 20, alone);
    Eigen::Matrix2Xd const withoutDensity = samples;
    EXPECT_EQ(withoutDensity, few.samples);
}

// On an angle the density is symmetric about its location and samples stay in range with matching
// log-densities.
TEST(StudentT, WrapsInAPeriodicSpace)
{
    auto const t = StudentT<Heading>::create(Eigen::Matrix<double, 1, 1>{3.0}, Eigen::Matrix<double, 1, 1>{0.04}, 4.0);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(
        t->logDensity(Eigen::Matrix<double, 1, 1>{3.1}),
        t->logDensity(Eigen::Matrix<double, 1, 1>{2.9}),
        1e-12
    );
    auto rng = std::mt19937_64{9};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*t, 2000, rng);
    EXPECT_TRUE((drawn.samples.array() >= -pi).all() && (drawn.samples.array() < pi).all());
    for (Eigen::Index j = 0; j < 20; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), t->logDensity(drawn.samples.col(j)), 1e-12);
    }
}

// A uniform's tag marginal is the uniform on that block's interval and, the coordinates being
// independent, its conditional is the uniform on the remaining block.
TEST(Uniform, MarginalsAndConditionalsKeepTheIntervals)
{
    using Box = Composite<Heading, Speed>;
    auto const box = Uniform<Box>::create(Box{}, Eigen::Vector2d{-pi, 2.0}, Eigen::Vector2d{pi, 5.0}).value();

    auto const heading = box.marginal<Heading>();
    static_assert(std::is_same_v<decltype(heading), Uniform<Heading> const>);
    EXPECT_NEAR(heading.width()(0), 2 * pi, 1e-15);

    auto const speed = box.marginal<Speed>();
    static_assert(std::is_same_v<decltype(speed), Uniform<Speed> const>);
    EXPECT_NEAR(speed.logDensity(Eigen::Matrix<double, 1, 1>{3.0}), -std::log(3.0), 1e-15);

    // Independent coordinates: conditioning on the heading leaves the speed's interval.
    auto const given = box.conditional<Heading>(Eigen::Matrix<double, 1, 1>{1.0});
    static_assert(std::is_same_v<decltype(given), Uniform<Speed> const>);
    EXPECT_NEAR(given.lower()(0), 2.0, 1e-15);
    EXPECT_NEAR(given.width()(0), 3.0, 1e-15);
    auto const headingGiven = box.conditional<Speed>(Eigen::Matrix<double, 1, 1>{4.0});
    static_assert(std::is_same_v<decltype(headingGiven), Uniform<Heading> const>);
    EXPECT_NEAR(headingGiven.logDensity(Eigen::Matrix<double, 1, 1>{-3.0}), -std::log(2 * pi), 1e-15);
}

// A gamma's marginal keeps the selected coordinates' shapes and scales and its conditional is the
// remaining coordinates', equal to the joint over the marginal.
TEST(Gamma, MarginalsAndConditionalsKeepTheParams)
{
    struct Pair : Linear<2>
    {};

    struct Last : Linear<1>
    {};

    using Three = Composite<Pair, Last>;
    auto const gamma = Gamma<Three>::create(Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.5, 1.0, 1.5}).value();
    auto const picked = gamma.marginal<Pair>();
    static_assert(std::is_same_v<decltype(picked), Gamma<Pair> const>);
    EXPECT_TRUE(picked.shape().isApprox(Eigen::Vector2d{1.0, 2.0}));
    EXPECT_TRUE(picked.scale().isApprox(Eigen::Vector2d{0.5, 1.0}));
    auto const rest = gamma.conditional<Pair>(Eigen::Vector2d{0.4, 4.0});
    static_assert(std::is_same_v<decltype(rest), Gamma<Last> const>);
    EXPECT_TRUE(rest.shape().isApprox(Eigen::Matrix<double, 1, 1>{3.0}));
    auto const x = Eigen::Vector3d{0.4, 4.0, 2.0};
    EXPECT_NEAR(
        rest.logDensity(Eigen::Matrix<double, 1, 1>{2.0}),
        gamma.logDensity(x) - picked.logDensity(Eigen::Vector2d{0.4, 4.0}),
        1e-13
    );
}

namespace
{

struct First : Linear<1>
{};

struct Second : Linear<1>
{};

struct Third : Linear<1>
{};

struct Lead : Composite<First, Second>
{};

using Ordered = Composite<Lead, Third>;  // First, Second, Third are coordinates 0, 1, 2

}  // namespace

// A Student-t's tag marginal keeps the degrees of freedom and selects the location and scale block.
TEST(StudentT, MarginalKeepsTheDegreesOfFreedom)
{
    auto const sigma = Eigen::Matrix3d{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.5}};
    auto const t = StudentT<Ordered>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 4.0);
    ASSERT_TRUE(t.has_value());
    auto const marginal = t->marginal<Lead>();
    EXPECT_EQ(marginal.dof(), 4.0);
    EXPECT_TRUE(marginal.location().isApprox(Eigen::Vector2d{1.0, -1.0}));
    EXPECT_TRUE(marginal.scale().isApprox(Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}));
}

// A Student-t conditional on a nested path has nu + (given dimension) degrees of freedom, equals
// the joint over the marginal, and has the Gaussian conditional's location and its covariance
// stretched by (nu + delta^2) / (nu + given dimension).
TEST(StudentT, ConditionalIsTheJointOverTheMarginal)
{
    auto const sigma = Eigen::Matrix3d{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.5}};
    auto const t = StudentT<Ordered>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 4.0);
    ASSERT_TRUE(t.has_value());
    using Given = TagPath<Lead, Second>;
    auto const v = Eigen::Matrix<double, 1, 1>{2.5};  // far from the location -1, so the conditional widens
    auto const conditional = t->conditional<Given>(v);
    ASSERT_TRUE(conditional.has_value());
    EXPECT_EQ(conditional->dof(), 5.0);
    auto const marginal = t->marginal<Given>();
    for (Eigen::Vector2d const & a : {Eigen::Vector2d{1.0, 0.5}, Eigen::Vector2d{3.0, -2.0}})
    {
        auto const x = Eigen::Vector3d{a(0), v(0), a(1)};
        EXPECT_NEAR(conditional->logDensity(a), t->logDensity(x) - marginal.logDensity(v), 1e-12);
    }
    // The scale is the Gaussian conditional's, stretched by (ν + δ²) / (ν + 1) with δ² = 3.5² / 1.
    auto const gaussian = Emrys::GaussianParams<Ordered>{Ordered{}, Eigen::Vector3d{1.0, -1.0, 0.5}, sigma};
    auto const gaussianConditional = gaussian.conditional<Given>(v);
    EXPECT_TRUE(conditional->scale().isApprox(gaussianConditional.covarianceLower() * (4.0 + 12.25) / 5.0));
    EXPECT_TRUE(conditional->location().isApprox(gaussianConditional.mean()));
}

// Conditioning on a top-level block gives a Student-t over the remaining block, equal to the joint
// over the marginal.
TEST(StudentT, TagConditionalReturnsTheRemainingBlock)
{
    struct Offset : Linear<2>
    {};

    using Pose = Composite<Offset, Heading>;
    auto const sigma = Eigen::Matrix3d{{2.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 0.5}};
    auto const t = StudentT<Pose>::create(Eigen::Vector3d{1.0, -1.0, 0.5}, sigma, 6.0);
    ASSERT_TRUE(t.has_value());
    auto const offset = t->conditional<Heading>(Eigen::Matrix<double, 1, 1>{0.6});
    ASSERT_TRUE(offset.has_value());
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(*offset)>, StudentT<Offset>>);
    EXPECT_EQ(offset->dof(), 7.0);
    auto const heading = t->marginal<Heading>();
    static_assert(std::is_same_v<decltype(heading), StudentT<Heading> const>);
    auto const a = Eigen::Vector2d{0.5, -0.5};
    EXPECT_NEAR(
        offset->logDensity(a),
        t->logDensity(Eigen::Vector3d{0.5, -0.5, 0.6}) - heading.logDensity(Eigen::Matrix<double, 1, 1>{0.6}),
        1e-12
    );
}

// marginal<Repeated>(i) of a uniform, a gamma, and a Student-t over a Repeated space is over the
// element space, and a path into a repetition gives the leaf's distribution; a uniform's box
// repeats the element's intervals.
TEST(Densities, MarginalOfOneRepetitionIsInTheBlockSpace)
{
    // The box repeats one cell's intervals: the whole heading, and speeds from 2 to 5.
    struct Cell : Composite<Heading, Speed>
    {};

    struct Cells : Repeated<Cell, 4>
    {
        using Repeated::Repeated;
    };

    Eigen::VectorXd const lower = Eigen::Vector2d{-pi, 2.0}.replicate(3, 1);
    Eigen::VectorXd const upper = Eigen::Vector2d{pi, 5.0}.replicate(3, 1);
    auto const cells = Uniform<Cells>::create(Cells{Cell{}, 3}, lower, upper).value();
    EXPECT_EQ(cells.valueDimension(), 6);
    auto const cell = cells.marginal<Cells>(1);
    static_assert(std::is_same_v<decltype(cell), Uniform<Cell> const>);
    EXPECT_NEAR(cell.logDensity(Eigen::Vector2d{0.0, 3.0}), -std::log(2 * pi * 3.0), 1e-14);
    auto const speed = cells.marginal<TagPath<Cells, Speed>>(2);
    static_assert(std::is_same_v<decltype(speed), Uniform<Speed> const>);
    EXPECT_EQ(speed.lower()(0), 2.0);
    EXPECT_EQ(speed.width()(0), 3.0);

    struct Ranges : Repeated<Range, 4>
    {
        using Repeated::Repeated;
    };

    auto const ranges = Gamma<Ranges>::create(Ranges{Range{}, 3}, Eigen::Vector3d{2.0, 3.0, 4.0}, 1.0).value();
    auto const second = ranges.marginal<Ranges>(1);
    static_assert(std::is_same_v<decltype(second), Gamma<Range> const>);
    EXPECT_EQ(second.shape()(0), 3.0);

    struct Headings : Repeated<Heading, 4>
    {
        using Repeated::Repeated;
    };

    auto const t =
        StudentT<Headings>::create(Headings{Heading{}, 2}, Eigen::Vector2d{0.5, 1.5}, Eigen::Matrix2d::Identity(), 3.0);
    ASSERT_TRUE(t.has_value());
    auto const last = t->marginal<Headings>(1);
    static_assert(std::is_same_v<decltype(last), StudentT<Heading> const>);
    EXPECT_EQ(last.location()(0), 1.5);
}

// Single draws, one drawSample() at a time, from each distribution here have its moments: 40,000
// from a box and from a gamma, whose draws also stay in its support, and 100,000 from a Student-t,
// whose heavy tails make the sample covariance noisier, as in SamplesMatchTheDistribution. Single
// draws with their log-densities are the same draws, each with logDensity() at it: exactly for the
// box and the gamma, and to rounding for the Student-t, whose distance comes from its normals.
TEST(Densities, StreamsOfSingleDrawsMatchEachDistribution)
{
    auto rng = std::mt19937_64{43};

    auto const box = Emrys::Uniform<Linear<2>>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    Eigen::MatrixXd const inBox = EmrysTest::streamOfDraws(box, 40'000, rng);
    EXPECT_TRUE((inBox.row(0).array() >= -1.0).all() && (inBox.row(0).array() < 1.0).all());
    EXPECT_TRUE((inBox.row(1).array() >= 0.0).all() && (inBox.row(1).array() < 4.0).all());
    EmrysTest::expectSampleMoments(inBox, box.moments().mean(), box.moments().covarianceLower(), 0.05);

    auto const gamma =
        Gamma<Linear<2>>::fromMeanAndStdDev(Eigen::Vector2d{100.0, 5.0}, Eigen::Vector2d{20.0, 5.0}).value();
    Eigen::MatrixXd const positive = EmrysTest::streamOfDraws(gamma, 40'000, rng);
    EXPECT_TRUE((positive.array() >= 0).all());
    EmrysTest::expectSampleMoments(positive, gamma.moments().mean(), gamma.moments().covarianceLower(), 0.05);

    auto const sigma = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto const t = StudentT<Linear<2>>::create(Eigen::Vector2d{1.0, 2.0}, sigma, 8.0);
    ASSERT_TRUE(t.has_value());
    Eigen::MatrixXd const heavy = EmrysTest::streamOfDraws(*t, 100'000, rng);
    EmrysTest::expectSampleMoments(heavy, Eigen::Vector2d{1.0, 2.0}, sigma * 8.0 / 6.0, 0.05);

    EmrysTest::expectSingleDrawsCarryTheirLogDensity(box, 50, 44, 0.0);
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(gamma, 50, 45, 0.0);
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(*t, 50, 46, 1e-12);
}

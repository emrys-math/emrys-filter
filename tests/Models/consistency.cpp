// ChiSquared, NisMonitor, nees(), and NeesMonitor of Models/Consistency.h (step 2b and the
// Agent-friendly use section of the roadmap): the chi-squared CDF and quantile against closed
// forms and each other, the monitors' sliding window, and the NEES against a plain-Eigen
// Mahalanobis distance, across a periodic seam, and for every form of the Gaussian.

#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Models/Consistency.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <type_traits>
#include <utility>

using Emrys::chiSquaredCdf;
using Emrys::chiSquaredQuantile;
using Emrys::NisMonitor;

// The chi-squared CDF with 1, 2, and 4 degrees of freedom matches its closed forms (erf, the
// exponential, and the Erlang sum), the quantile with 2 degrees of freedom inverts the exponential,
// and three quantiles match tabulated values.
TEST(ChiSquared, MatchesClosedForms)
{
    for (double const x : {0.01, 0.5, 2.0, 7.5, 30.0})
    {
        EXPECT_NEAR(chiSquaredCdf<double>(x, 1), std::erf(std::sqrt(x / 2)), 1e-14) << x;
        EXPECT_NEAR(chiSquaredCdf<double>(x, 2), 1 - std::exp(-x / 2), 1e-14) << x;
        EXPECT_NEAR(chiSquaredCdf<double>(x, 4), 1 - std::exp(-x / 2) * (1 + x / 2), 1e-14) << x;
    }
    EXPECT_EQ(chiSquaredCdf<double>(0, 3), 0.0);
    for (double const p : {0.01, 0.5, 0.95, 0.997})
    {
        EXPECT_NEAR(chiSquaredQuantile<double>(p, 2), -2 * std::log(1 - p), 1e-12);
    }
    EXPECT_NEAR(chiSquaredQuantile<double>(0.95, 1), 3.841458820694124, 1e-12);
    EXPECT_NEAR(chiSquaredQuantile<double>(0.99, 10), 23.209251158954338, 1e-11);
    EXPECT_NEAR(chiSquaredQuantile<double>(0.05, 10), 3.9402991361190605, 1e-11);
}

// Across fractional, small, and large degrees of freedom and tail probabilities down to 1e-6, the
// CDF of the quantile returns the probability to 1e-12.
TEST(ChiSquared, QuantileInvertsTheCdf)
{
    for (double const dof : {0.5, 1.0, 3.0, 7.0, 50.0, 400.0})
    {
        for (double const p : {1e-6, 0.01, 0.5, 0.99, 1 - 1e-6})
        {
            double const x = chiSquaredQuantile<double>(p, dof);
            EXPECT_GT(x, 0.0);
            EXPECT_NEAR(chiSquaredCdf<double>(x, dof), p, 1e-12) << "dof " << dof << ", p " << p;
        }
    }
}

// At the degrees of freedom a cumulative monitor reaches over a long run, 1e4 to 1e6, the quantile
// matches values computed to 40 digits with mpmath, at the tails a 0.999 test reads and at the
// median, to 1e-12 relative.
TEST(ChiSquared, QuantileHoldsAtLargeDegreesOfFreedom)
{
    struct Case
    {
        double dof;
        double probability;
        double quantile;
    };

    for (auto const & c : {
             Case{1e4, 5e-4, 9541.1908574959507844},
             Case{1e4, 0.5, 9999.3333412351448272},
             Case{1e4, 0.9995, 10471.912157626601739},
             Case{1e5, 5e-4, 98534.980270548135346},
             Case{1e5, 0.5, 99999.333334123462559},
             Case{1e5, 0.9995, 101478.1231103704511},
             Case{1e6, 5e-4, 995353.0431881858826},
             Case{1e6, 0.5, 999999.33333341234574},
             Case{1e6, 0.9995, 1004660.0602293107867},
         })
    {
        EXPECT_NEAR(chiSquaredQuantile<double>(c.probability, c.dof), c.quantile, 1e-12 * c.quantile)
            << "dof " << c.dof << ", p " << c.probability;
    }
}

// A small probability with a degree of freedom or less puts the quantile hundreds of orders of
// magnitude below one, where P(a, x) is x^a / Gamma(a + 1) to the precision of a double. The
// quantile reaches it, and comes out at most the smallest normal double where it underflows.
TEST(ChiSquared, QuantileFarBelowOne)
{
    for (auto const & [dof, p] :
         {std::pair{0.1, 1e-10}, std::pair{0.5, 1e-10}, std::pair{0.5, 1e-30}, std::pair{1.0, 1e-30}})
    {
        double const a = dof / 2;
        double const expected = 2 * std::pow(p * std::tgamma(a + 1), 1 / a);
        EXPECT_NEAR(chiSquaredQuantile<double>(p, dof), expected, 1e-12 * expected) << "dof " << dof << ", p " << p;
    }
    double const underflowed = chiSquaredQuantile<double>(1e-30, 0.01);
    EXPECT_GE(underflowed, 0.0);
    EXPECT_LE(underflowed, std::numeric_limits<double>::min());
}

// Far down the lower tail at many degrees of freedom Newton's steps from above shrink by a
// fraction of the quantile each, and the search bisects instead; the CDF of the quantile returns
// the probability.
TEST(ChiSquared, QuantileDeepInTheLowerTail)
{
    for (double const p : {1e-200, 1e-100})
    {
        double const x = chiSquaredQuantile<double>(p, 1000);
        EXPECT_NEAR(chiSquaredCdf<double>(x, 1000), p, 1e-11 * p) << p;
    }
}

// The CDF is 1 at infinity and NaN at NaN, and stays a probability where rounding takes its
// expansions past [0, 1]: the series' sum past 1 with almost no degrees of freedom, and the
// continued fraction's complement below 0 at a million degrees of freedom in float.
TEST(ChiSquared, CdfAtTheEdges)
{
    EXPECT_EQ(chiSquaredCdf<double>(std::numeric_limits<double>::infinity(), 3), 1.0);
    EXPECT_TRUE(std::isnan(chiSquaredCdf<double>(std::numeric_limits<double>::quiet_NaN(), 3)));
    EXPECT_EQ(chiSquaredCdf<double>(0.01, 1e-30), 1.0);
    float const rounded = chiSquaredCdf<float>(1588658.5f, 1588656.5f);
    EXPECT_GE(rounded, 0.0f);
    EXPECT_LE(rounded, 1.0f);
}

// In float the CDF and the quantile compute in float, return float, and agree with the double
// evaluation to a few float epsilons; the tolerance, 1e-5 relative, is about 80 of them, for the
// rounding the series and the Newton iteration accumulate in single precision.
TEST(ChiSquared, ComputesInFloat)
{
    static_assert(std::is_same_v<decltype(chiSquaredCdf<float>(1.0f, 2.0f)), float>);
    static_assert(std::is_same_v<decltype(chiSquaredQuantile<float>(0.5f, 2.0f)), float>);
    for (float const dof : {0.5f, 1.0f, 3.0f, 7.0f, 50.0f})
    {
        for (float const p : {0.01f, 0.5f, 0.95f, 0.997f})
        {
            float const x = chiSquaredQuantile<float>(p, dof);
            double const reference = chiSquaredQuantile<double>(p, dof);
            EXPECT_NEAR(x, reference, 1e-5 * reference) << "dof " << dof << ", p " << p;
            EXPECT_NEAR(chiSquaredCdf<float>(x, dof), p, 1e-5f) << "dof " << dof << ", p " << p;
        }
    }
    auto monitor = NisMonitor<2, 2, float>{};
    monitor.add(1.5f, 2);
    static_assert(
        std::is_same_v<decltype(monitor.bounds(0.9f)), std::expected<std::pair<float, float>, Emrys::NumericError>>
    );
    EXPECT_NEAR(monitor.bounds(0.9f)->second, chiSquaredQuantile<double>(0.95, 2), 1e-5);
}

// A monitor of window three keeps the last three NIS values and their degrees of freedom, its
// bounds are the chi-squared quantiles of the summed degrees of freedom, verdict() compares the
// sum against them, and reset() empties it, whether the window is fixed, bounded, or unbounded.
TEST(NisMonitor, KeepsTheMostRecentWindow)
{
    auto const check = [](auto monitor) {
        EXPECT_EQ(monitor.window(), 3);
        monitor.add(1.0, 2);
        monitor.add(2.0, 2);
        monitor.add(3.0, 1);
        EXPECT_EQ(monitor.count(), 3);
        EXPECT_EQ(monitor.sum(), 6.0);
        monitor.add(4.0, 2);  // replaces the first
        EXPECT_EQ(monitor.count(), 3);
        EXPECT_EQ(monitor.sum(), 9.0);
        EXPECT_EQ(monitor.degreesOfFreedom(), 5);
        EXPECT_NEAR(monitor.average().value(), 1.8, 1e-15);

        auto const [lower, upper] = monitor.bounds(0.9).value();
        EXPECT_NEAR(lower, chiSquaredQuantile<double>(0.05, 5), 1e-15);
        EXPECT_NEAR(upper, chiSquaredQuantile<double>(0.95, 5), 1e-15);
        EXPECT_EQ(monitor.verdict(0.9), Emrys::Consistency::Consistent);  // 9 is within [1.15, 11.07]
        monitor.add(40.0, 2);
        EXPECT_EQ(monitor.verdict(0.9), Emrys::Consistency::Overconfident);

        monitor.reset();
        EXPECT_EQ(monitor.count(), 0);
    };
    auto const dynamic = NisMonitor{3};
    static_assert(std::is_same_v<decltype(dynamic), NisMonitor<> const>);
    check(dynamic);
    check(NisMonitor<3>{});
    check(NisMonitor<Eigen::Dynamic, 8>{3});
}

// A monitor whose window has no bound copies into a window from another allocator with its
// values, the NIS and the NEES monitor alike.
TEST(NisMonitor, CopiesIntoAnotherAllocator)
{
    using Pmr = std::pmr::polymorphic_allocator<>;
    auto first = std::pmr::monotonic_buffer_resource{};
    auto second = std::pmr::monotonic_buffer_resource{};
    auto const check = [&](auto monitor) {
        monitor.add(1.0, 2);
        monitor.add(2.0, 1);
        auto const copy = decltype(monitor){monitor, Pmr{&second}};
        EXPECT_EQ(copy.window(), 3);
        EXPECT_EQ(copy.count(), 2);
        EXPECT_EQ(copy.sum(), 3.0);
        EXPECT_EQ(copy.degreesOfFreedom(), 3);
    };
    check(NisMonitor<Eigen::Dynamic, Eigen::Dynamic, double, Pmr>{3, Pmr{&first}});
    check(Emrys::NeesMonitor<Eigen::Dynamic, Eigen::Dynamic, double, Pmr>{3, Pmr{&first}});
}

// A Cumulative monitor drops nothing: its sum and degrees of freedom run over every value since
// construction or reset(), its bounds are those of the total, and a NaN stays until reset(), so
// that average() and verdict() fail with NonFinite while bounds(), which reads no sum, does not.
TEST(NisMonitor, CumulativeKeepsEveryUpdate)
{
    auto monitor = NisMonitor<Emrys::Cumulative>{};
    for (int k = 0; k < 1000; ++k)
    {
        monitor.add(2.0, 2);
    }
    EXPECT_EQ(monitor.count(), 1000);
    EXPECT_EQ(monitor.sum(), 2000.0);
    EXPECT_EQ(monitor.degreesOfFreedom(), 2000);
    EXPECT_EQ(monitor.average(), 1.0);
    auto const [lower, upper] = monitor.bounds(0.99).value();
    EXPECT_NEAR(lower, chiSquaredQuantile<double>(0.005, 2000), 1e-9);
    EXPECT_NEAR(upper, chiSquaredQuantile<double>(0.995, 2000), 1e-9);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);

    monitor.add(std::numeric_limits<double>::quiet_NaN(), 2);
    for (int k = 0; k < 1000; ++k)
    {
        monitor.add(2.0, 2);
    }
    EXPECT_EQ(monitor.average().error(), Emrys::NumericError::NonFinite);
    EXPECT_EQ(monitor.verdict(0.99).error(), Emrys::NumericError::NonFinite);
    EXPECT_TRUE(monitor.bounds(0.99));
    monitor.reset();
    EXPECT_EQ(monitor.count(), 0);
    monitor.add(2.0, 2);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);
}

// An empty monitor has no average, bounds, or verdict, and says so with Empty, whatever its
// window: a monitor fed only the runs a test trusts is empty when the test trusts none.
TEST(NisMonitor, EmptyHasNoVerdict)
{
    auto const check = [](auto monitor) {
        EXPECT_EQ(monitor.average().error(), Emrys::NumericError::Empty);
        EXPECT_EQ(monitor.bounds(0.99).error(), Emrys::NumericError::Empty);
        EXPECT_EQ(monitor.verdict(0.99).error(), Emrys::NumericError::Empty);
        monitor.add(2.0, 2);
        monitor.reset();
        EXPECT_EQ(monitor.verdict(0.99).error(), Emrys::NumericError::Empty);
    };
    check(NisMonitor{3});
    check(NisMonitor<3>{});
    check(NisMonitor<Emrys::Cumulative>{});
    check(Emrys::NeesMonitor<Emrys::Cumulative>{});
}

// The verdict says which side of the bounds the sum is on: below the lower an underconfident
// filter's, above the upper an overconfident one's, and each tail holds (1 - p) / 2.
TEST(NisMonitor, VerdictNamesTheSide)
{
    auto monitor = NisMonitor<Emrys::Cumulative>{};
    double const lower = chiSquaredQuantile<double>(0.005, 10);
    double const upper = chiSquaredQuantile<double>(0.995, 10);
    monitor.add(0.99 * lower, 10);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Underconfident);
    monitor.reset();
    monitor.add(1.01 * lower, 10);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);
    monitor.reset();
    monitor.add(0.99 * upper, 10);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);
    monitor.reset();
    monitor.add(1.01 * upper, 10);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Overconfident);
}

// An unbounded window takes its ring from the allocator once, at construction, so a monitor
// over an arena with no upstream keeps adding past its window; its copy keeps the allocator.
TEST(NisMonitor, UnboundedWindowUsesItsAllocator)
{
    auto buffer = std::array<std::byte, 1024>{};
    auto arena = std::pmr::monotonic_buffer_resource{buffer.data(), buffer.size(), std::pmr::null_memory_resource()};
    using Allocator = std::pmr::polymorphic_allocator<double>;
    auto monitor = NisMonitor<Eigen::Dynamic, Eigen::Dynamic, double, Allocator>{4, Allocator{&arena}};
    for (int k = 0; k < 10; ++k)
    {
        monitor.add(1.0, 1);
    }
    EXPECT_EQ(monitor.sum(), 4.0);
    EXPECT_EQ(monitor.get_allocator().resource(), &arena);
    auto const copy = monitor;
    EXPECT_EQ(copy.get_allocator().resource(), &arena);
    EXPECT_EQ(copy.sum(), 4.0);
}

namespace
{

struct Bearing : Emrys::RadiansPi<1>
{};

struct Range : Emrys::Linear<1>
{};

using Polar = Emrys::Composite<Range, Bearing>;

struct Rate : Emrys::Linear<2>
{};

using Track = Emrys::Composite<Range, Rate>;

Emrys::GaussianParams<Emrys::Linear<3>>
makeEstimate()
{
    auto const cov = Eigen::Matrix3d{{2, 0.3, 0.1}, {0.3, 1, 0.2}, {0.1, 0.2, 0.5}};
    return {Eigen::Vector3d{1, -2, 0.5}, cov};
}

}  // namespace

// nees() is the squared Mahalanobis distance of the truth from the estimate, d^T P^-1 d in plain
// Eigen, and every form of the Gaussian gives it to rounding.
TEST(Nees, IsTheMahalanobisDistanceOfTheTruth)
{
    auto const estimate = makeEstimate();
    auto const truth = Eigen::Vector3d{1.5, -1, 0.3};
    Eigen::Vector3d const d = truth - estimate.mean();
    double const expected = d.dot(estimate.covarianceLower().ldlt().solve(d));
    auto const value = Emrys::nees(truth, estimate);
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, expected, 1e-13);
    EXPECT_NEAR(Emrys::nees(truth, Emrys::toSqrt(estimate).value()).value(), expected, 1e-12);
    EXPECT_NEAR(Emrys::nees(truth, Emrys::toInfo(estimate).value()).value(), expected, 1e-12);
    EXPECT_NEAR(Emrys::nees(truth, Emrys::toSqrtInfo(estimate).value()).value(), expected, 1e-12);
}

// The error is taken by minus() in the estimate's space: a bearing of 3.1 against an estimate of
// -3.1 is 0.083 rad across the seam, not 6.2 rad, so the NEES is that error squared over the
// bearing's variance.
TEST(Nees, TakesTheErrorInTheSpace)
{
    auto const estimate = Emrys::GaussianParams<Polar>{
        Polar{},
        Eigen::Vector2d{10, -3.1},
        Eigen::Vector2d{1, 0.01}.asDiagonal().toDenseMatrix()
    };
    double const across = 2 * std::numbers::pi - 6.2;
    EXPECT_NEAR(Emrys::nees(Eigen::Vector2d{10, 3.1}, estimate).value(), across * across / 0.01, 1e-10);
}

// nees<Path>() of a block is the Mahalanobis distance over that block's entries alone, d^T P_bb^-1 d,
// read from the truth and the estimate by the same path; the block here starts at entry 1, not 0.
// It agrees with nees() of the block's marginal view, which has no density of its own, and with
// every form's marginal, the information forms' a Schur complement.
TEST(Nees, OfABlockReadsTheTruthByThePath)
{
    auto const cov = Eigen::Matrix3d{{2, 0.3, 0.1}, {0.3, 1, 0.2}, {0.1, 0.2, 0.5}};
    auto const estimate = Emrys::GaussianParams<Track>{Track{}, Eigen::Vector3d{1, -2, 0.5}, cov};
    auto const truth = Eigen::Vector3d{1.5, -1, 0.3};
    Eigen::Vector2d const d = truth.tail<2>() - estimate.mean().tail<2>();
    double const expected = d.dot(cov.bottomRightCorner<2, 2>().ldlt().solve(d));

    auto const value = Emrys::nees<Rate>(truth, estimate);
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, expected, 1e-13);
    EXPECT_NEAR(Emrys::nees(truth.tail<2>(), estimate.marginal<Rate>()).value(), expected, 1e-13);
    EXPECT_NEAR(Emrys::nees<Rate>(truth, Emrys::toSqrt(estimate).value()).value(), expected, 1e-12);
    EXPECT_NEAR(Emrys::nees<Rate>(truth, Emrys::toInfo(estimate).value()).value(), expected, 1e-12);
    EXPECT_NEAR(Emrys::nees<Rate>(truth, Emrys::toSqrtInfo(estimate).value()).value(), expected, 1e-12);

    auto monitor = Emrys::NeesMonitor<2>{};
    monitor.add<Rate>(truth, estimate);
    EXPECT_EQ(monitor.degreesOfFreedom(), 2);
    EXPECT_EQ(monitor.sum(), value.value());
}

// A block of one coordinate is a vector of one entry, whose index alone is a scalar: nees<Path>()
// and add<Path>() read it as the block of two above, with one degree of freedom.
TEST(Nees, OfABlockOfOneCoordinate)
{
    auto const cov = Eigen::Matrix3d{{2, 0.3, 0.1}, {0.3, 1, 0.2}, {0.1, 0.2, 0.5}};
    auto const estimate = Emrys::GaussianParams<Track>{Track{}, Eigen::Vector3d{1, -2, 0.5}, cov};
    auto const truth = Eigen::Vector3d{1.5, -1, 0.3};
    double const expected = 0.5 * 0.5 / 2;

    auto const value = Emrys::nees<Range>(truth, estimate);
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, expected, 1e-13);

    auto monitor = Emrys::NeesMonitor<1>{};
    monitor.add<Range>(truth, estimate);
    EXPECT_EQ(monitor.degreesOfFreedom(), 1);
    EXPECT_EQ(monitor.sum(), value.value());
}

// A covariance that isn't positive definite has no NEES, and says why.
TEST(Nees, NeedsAPositiveDefiniteCovariance)
{
    auto const broken =
        Emrys::GaussianParams<Emrys::Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1, 2}, {2, 1}}};
    auto const value = Emrys::nees(Eigen::Vector2d{1, 0}, broken);
    ASSERT_FALSE(value);
    EXPECT_EQ(value.error(), Emrys::GaussianError::NotPositiveDefinite);
}

// The NEES monitor shares the NIS monitor's window, adds each estimate's NEES with its difference
// dimension as the degrees of freedom, and adds NaN for an estimate without one, which leaves
// the monitor with no verdict until it leaves the window.
TEST(NeesMonitor, AddsEachEstimateWithItsDimension)
{
    auto const estimate = makeEstimate();
    auto const truth = Eigen::Vector3d{1.5, -1, 0.3};
    auto monitor = Emrys::NeesMonitor<2>{};
    monitor.add(truth, estimate);
    EXPECT_EQ(monitor.count(), 1);
    EXPECT_EQ(monitor.degreesOfFreedom(), 3);
    EXPECT_EQ(monitor.sum(), Emrys::nees(truth, estimate).value());
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);

    auto const broken = Emrys::GaussianParams<Emrys::Linear<3>>{Eigen::Vector3d::Zero(), -Eigen::Matrix3d::Identity()};
    monitor.add(truth, broken);
    EXPECT_EQ(monitor.verdict(0.99).error(), Emrys::NumericError::NonFinite);
    monitor.add(truth, estimate);
    monitor.add(truth, estimate);
    EXPECT_EQ(monitor.verdict(0.99), Emrys::Consistency::Consistent);
    EXPECT_EQ(monitor.degreesOfFreedom(), 6);
}

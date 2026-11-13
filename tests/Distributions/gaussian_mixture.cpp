// GaussianMixtureParams and its GaussianMixture (step 4; [mixture-filter], [sizes-and-dimensions], [density-names],
// [forwarded-matrices]): log-density and moments against references, weight scale invariance, effective sample size,
// error reporting that names the failing component, and the editing operations (remove, merged, merge, split, removeIf,
// the prunes, normalize, assign, append, resize) that upkeep and the filters build on, the
// lineage those operations keep for the mixture smoother (step 14; [smoothing]), and the view checks and death
// tests Gaussian has [view-stamp].

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <array>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::ComponentError;
using Emrys::GaussianError;
using Emrys::GaussianMixtureError;
using Emrys::GaussianMixtureParams;
using Emrys::Linear;
using Emrys::WeightError;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

// Lets std::vector move rather than copy components on reallocation.
static_assert(std::is_nothrow_move_constructible_v<GaussianMixtureParams<Linear<Eigen::Dynamic>>::Entry>);

static_assert(std::is_nothrow_move_constructible_v<GaussianMixtureParams<Linear<Eigen::Dynamic>>>);

// Each config names a GaussianMixtureParams type and the component dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config
{
    using Type = GaussianMixtureParams<Linear<Size, MaxSize>, Scalar>;
    static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<
    Config<double, 1, 1, 1>,
    Config<double, 3, 3, 3>,
    Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
    Config<double, Eigen::Dynamic, 6, 4>,
    Config<float, 2, 2, 2>>;

template <typename C>
class GaussianMixtureTest : public ::testing::Test
{
protected:

    using MixtureType = typename C::Type;
    using Scalar = typename MixtureType::Scalar;
    using VectorType = typename MixtureType::VectorType;
    using MatrixType = typename MixtureType::MatrixType;
    static constexpr Eigen::Index n = C::n;

    static Scalar
    tolerance()
    {
        return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10);
    }

    static VectorType
    makeVector(double phase)
    {
        return VectorType::NullaryExpr(n, [phase](Eigen::Index i) {
            return Scalar(std::sin(1.3 * i + phase));
        });
    }

    // Deterministic symmetric positive definite matrix: A·Aᵀ + n·I.
    static MatrixType
    makeSpd(double phase)
    {
        MatrixType const a = MatrixType::NullaryExpr(n, n, [phase](Eigen::Index i, Eigen::Index j) {
            return Scalar(std::sin(7.0 * i + 3.0 * j + phase));
        });
        MatrixType p = a * a.transpose();
        p.diagonal().array() += Scalar(n);
        return p;
    }

    // Three components with unnormalized weights 0.2, 1.5, 0.3.
    static MixtureType
    make()
    {
        auto mixture = MixtureType{n};
        mixture.add(Scalar(std::log(0.2)), makeVector(0.0), makeSpd(1.0));
        mixture.add(Scalar(std::log(1.5)), makeVector(1.0), makeSpd(2.0));
        mixture.add(Scalar(std::log(0.3)), makeVector(2.0), makeSpd(3.0));
        return mixture;
    }

    // Direct evaluation through determinants and inverses, in double precision.
    static double
    referenceLogDensity(MixtureType const & mixture, VectorType const & x)
    {
        double density = 0;
        double total = 0;
        for (auto const & c : mixture)
        {
            Eigen::MatrixXd const p = c.component.covarianceLower().template cast<double>();
            Eigen::VectorXd const r = (x - c.component.mean()).template cast<double>();
            double const logN =
                -0.5
                * (double(n) * std::log(2.0 * std::numbers::pi) + std::log(p.determinant()) + r.dot(p.inverse() * r));
            double const w = std::exp(double(c.logWeight));
            density += w * std::exp(logN);
            total += w;
        }
        return std::log(density / total);
    }

    static std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    referenceMoments(MixtureType const & mixture)
    {
        double total = 0;
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
        for (auto const & c : mixture)
        {
            double const w = std::exp(double(c.logWeight));
            total += w;
            mean += w * c.component.mean().template cast<double>();
        }
        mean /= total;
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
        for (auto const & c : mixture)
        {
            double const w = std::exp(double(c.logWeight));
            Eigen::VectorXd const d = c.component.mean().template cast<double>() - mean;
            cov += w * (c.component.covarianceLower().template cast<double>() + d * d.transpose());
        }
        return {mean, cov / total};
    }
};

}  // namespace

TYPED_TEST_SUITE(GaussianMixtureTest, Configs);

// logDensity() matches a direct evaluation of the weighted sum of component densities in double,
// density() is its exponential, and logTotalWeight() is the log of the weights' sum.
TYPED_TEST(GaussianMixtureTest, LogDensityMatchesReference)
{
    auto const mixture = this->make();
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());

    auto const x = this->makeVector(0.7);
    double const expected = this->referenceLogDensity(mixture, x);
    EXPECT_NEAR(g->logDensity(x), expected, this->tolerance() * std::abs(expected));
    EXPECT_NEAR(g->density(x), std::exp(g->logDensity(x)), this->tolerance());
    EXPECT_NEAR(g->logTotalWeight(), std::log(2.0), this->tolerance());
}

// moments() matches the moment-matching formulas (weighted means, weighted covariances plus spread
// of the means) computed with plain Eigen.
TYPED_TEST(GaussianMixtureTest, MomentsMatchReference)
{
    auto const mixture = this->make();
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());

    auto const [mean, cov] = this->referenceMoments(mixture);
    auto const moments = g->moments();
    EXPECT_LT((moments.mean().template cast<double>() - mean).norm(), this->tolerance() * 10);
    EXPECT_LT((moments.covarianceLower().template cast<double>() - cov).norm(), this->tolerance() * 10 * cov.norm());
}

// Overwriting every component's strict upper triangle leaves the moments unchanged.
TYPED_TEST(GaussianMixtureTest, MomentsReadOnlyLowerTriangle)
{
    auto mixture = this->make();
    auto const expected = mixture.toDensity()->moments();

    for (Eigen::Index i = 0; i < mixture.size(); ++i)
    {
        auto cov = mixture[i].component.covarianceLower();
        cov.template triangularView<Eigen::StrictlyUpper>().setConstant(12345);
        mixture.setCovariance(i, cov);
    }
    auto const moments = mixture.toDensity()->moments();
    EXPECT_EQ(moments.mean(), expected.mean());
    EXPECT_EQ(moments.covarianceLower(), expected.covarianceLower());
}

// Scaling every log-weight by the same amount leaves the density unchanged and shifts the total
// weight.
TYPED_TEST(GaussianMixtureTest, WeightScaleDoesNotChangeDensity)
{
    using Scalar = typename TestFixture::Scalar;
    auto mixture = this->make();
    auto const x = this->makeVector(0.7);
    auto const logDensity = mixture.toDensity()->logDensity(x);

    mixture.scaleWeights(Scalar(5));
    auto const g = mixture.toDensity();
    EXPECT_NEAR(g->logDensity(x), logDensity, this->tolerance() * std::abs(logDensity));
    EXPECT_NEAR(g->logTotalWeight(), std::log(2.0) + 5.0, this->tolerance() * 10);
}

// ess() is one over the sum of squared normalized weights: the hand value for
// weights 0.2, 1.5, 0.3, the count for equal weights, and one for a single nonzero weight.
TYPED_TEST(GaussianMixtureTest, Ess)
{
    using Scalar = typename TestFixture::Scalar;
    auto mixture = this->make();
    // Normalized weights 0.1, 0.75, 0.15.
    EXPECT_NEAR(mixture.toDensity()->ess(), Scalar(1 / (0.01 + 0.5625 + 0.0225)), this->tolerance() * 10);

    for (Eigen::Index i = 0; i < mixture.size(); ++i)
    {
        mixture.setLogWeight(i, Scalar(2));
    }
    EXPECT_NEAR(mixture.toDensity()->ess(), Scalar(3), this->tolerance() * 10);

    mixture.setLogWeight(0, -Eigen::NumTraits<Scalar>::infinity());
    mixture.setLogWeight(1, -Eigen::NumTraits<Scalar>::infinity());
    EXPECT_EQ(mixture.toDensity()->ess(), Scalar(1));
}

// The parameters' ess() equals the density's, to the bit: the density's calls the same code with
// the total weight it checked.
TYPED_TEST(GaussianMixtureTest, ParamsEssMatchesTheDensity)
{
    auto const mixture = this->make();
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density.has_value());
    EXPECT_EQ(mixture.ess().value(), density->ess());
}

namespace
{

using Mixture2 = GaussianMixtureParams<Linear<2>>;

// A 2-D mixture with the given weights, each component at a distinct mean.
Mixture2
makeWeighted(std::vector<double> const & weights)
{
    auto mixture = Mixture2{};
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        mixture.add(std::log(weights[i]), Eigen::Vector2d::Constant(double(i)), Eigen::Matrix2d::Identity());
    }
    return mixture;
}

// Expects the mixture's weights, in any order, to match the ascending list `expected`.
void
expectWeights(Mixture2 const & mixture, std::vector<double> const & expected)
{
    auto weights = std::vector<double>{};
    for (auto const & c : mixture)
    {
        weights.push_back(std::exp(c.logWeight));
    }
    std::ranges::sort(weights);
    ASSERT_EQ(weights.size(), expected.size());
    for (std::size_t i = 0; i < weights.size(); ++i)
    {
        EXPECT_NEAR(weights[i], expected[i], 1e-12) << "index " << i;
    }
}

}  // namespace

// An empty mixture reports Empty from logTotalWeight() and toDensity().
TEST(GaussianMixture, EmptyFails)
{
    Mixture2 const mixture;
    EXPECT_EQ(mixture.logTotalWeight().error(), WeightError::Empty);
    EXPECT_EQ(mixture.toDensity().error(), GaussianMixtureError{WeightError::Empty});
}

// All-zero weights fail with ZeroSum.
TEST(GaussianMixture, ZeroWeightFails)
{
    auto mixture = makeWeighted({1.0, 1.0});
    mixture.scaleWeights(-inf);
    EXPECT_EQ(mixture.toDensity().error(), GaussianMixtureError{WeightError::ZeroSum});
}

// A NaN or infinite log-weight fails with NonFinite.
TEST(GaussianMixture, NonFiniteWeightFails)
{
    auto mixture = makeWeighted({1.0, 1.0});
    mixture.setLogWeight(0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(mixture.toDensity().error(), GaussianMixtureError{WeightError::NonFinite});
    mixture.setLogWeight(0, inf);
    EXPECT_EQ(mixture.toDensity().error(), GaussianMixtureError{WeightError::NonFinite});
}

// The parameters' ess() fails with the WeightError of logTotalWeight(): Empty, ZeroSum, and
// NonFinite.
TEST(GaussianMixture, ParamsEssReportsTheWeightError)
{
    Mixture2 const empty;
    EXPECT_EQ(empty.ess().error(), WeightError::Empty);

    auto zero = makeWeighted({1.0, 1.0});
    zero.scaleWeights(-inf);
    EXPECT_EQ(zero.ess().error(), WeightError::ZeroSum);

    auto nonFinite = makeWeighted({1.0, 1.0});
    nonFinite.setLogWeight(0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(nonFinite.ess().error(), WeightError::NonFinite);
}

// The parameters' ess() reads only the weights, so it is there for a mixture whose density
// can't be made, here for a component covariance that isn't positive definite.
TEST(GaussianMixture, ParamsEssNeedsNoFactorization)
{
    auto mixture = Mixture2{};
    mixture.add(0.0, Eigen::Vector2d{0.0, 0.0}, Eigen::Matrix2d::Identity());
    mixture.add(0.0, Eigen::Vector2d{2.0, 0.0}, -Eigen::Matrix2d::Identity());
    ASSERT_FALSE(mixture.toDensity().has_value());
    EXPECT_DOUBLE_EQ(mixture.ess().value(), 2.0);
}

// A component that fails to factorize is reported as a ComponentError with its index and the
// Gaussian error.
TEST(GaussianMixture, ComponentFailureNamesComponent)
{
    auto mixture = makeWeighted({1.0, 1.0, 1.0});
    mixture.setCovariance(1, -Eigen::Matrix2d::Identity());
    auto const g = mixture.toDensity();
    ASSERT_FALSE(g.has_value());
    EXPECT_EQ(g.error(), GaussianMixtureError{(ComponentError{1, GaussianError::NotPositiveDefinite})});
}

// toDensity(roots::ridge()) repairs only the failing component, adding one ridge to its diagonal, and
// leaves the others as they were.
TEST(GaussianMixture, RidgeRepairUpdatesFailingComponent)
{
    auto mixture = makeWeighted({1.0, 1.0, 1.0});
    Eigen::Matrix2d const first = mixture.begin()->component.covarianceLower();
    auto const indefinite = Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}};
    mixture.setCovariance(1, indefinite);
    ASSERT_TRUE(mixture.toDensity(Emrys::roots::ridge()).has_value());
    EXPECT_EQ(mixture.begin()->component.covarianceLower(), first);
    Eigen::Matrix2d const ridge = mixture[1].component.covarianceLower() - indefinite;
    EXPECT_GT(ridge(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(ridge(1, 1), ridge(0, 0));
    EXPECT_EQ(ridge(1, 0), 0.0);
}

// A ridge repair that cannot succeed names the failing component, and leaves the mixture as it
// was: the repairable component before it keeps its covariance.
TEST(GaussianMixture, RidgeRepairFailureNamesComponent)
{
    auto mixture = makeWeighted({1.0, 1.0});
    auto const indefinite = Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}};
    mixture.setCovariance(0, indefinite);
    mixture.setCovariance(1, Eigen::Matrix2d::Zero());
    EXPECT_EQ(
        mixture.factorize(Emrys::roots::ridge()).error(),
        GaussianMixtureError{(ComponentError{1, GaussianError::NotPositiveDefinite})}
    );
    EXPECT_EQ(mixture.begin()->component.covarianceLower(), indefinite);
}

// A component with log-weight minus infinity contributes nothing to the moments.
TEST(GaussianMixture, ZeroWeightComponentIsIgnored)
{
    auto mixture = makeWeighted({1.0, 1.0});
    mixture.setLogWeight(0, -inf);
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());
    auto const moments = g->moments();
    EXPECT_EQ(moments.mean(), Eigen::Vector2d::Constant(1.0));
    EXPECT_EQ(moments.covarianceLower(), Eigen::Matrix2d::Identity());
    auto const other = Emrys::GaussianParams<Linear<2>>{Eigen::Vector2d::Constant(1.0), Eigen::Matrix2d::Identity()};
    EXPECT_NEAR(
        g->logDensity(Eigen::Vector2d{0.5, 2.0}),
        other.toDensity()->logDensity(Eigen::Vector2d{0.5, 2.0}),
        1e-12
    );
}

// The operations that need the total weight report weights that fail to define one, changing
// nothing: normalize(), the prunes, and toDensity(roots::ridge()), which also reports a component
// that no ridge repairs.
TEST(GaussianMixture, OperationsOnTheTotalReportItsFailure)
{
    auto zero = makeWeighted({1.0, 1.0});
    zero.scaleWeights(-inf);
    EXPECT_EQ(zero.normalize().error(), WeightError::ZeroSum);
    EXPECT_EQ(zero.pruneToSize(1).error(), WeightError::ZeroSum);
    EXPECT_EQ(zero.size(), 2);

    auto broken = makeWeighted({1.0, 1.0});
    broken.setCovariance(1, Eigen::Matrix2d::Zero());
    EXPECT_EQ(
        broken.toDensity(Emrys::roots::ridge()).error(),
        GaussianMixtureError{(ComponentError{1, GaussianError::NotPositiveDefinite})}
    );

    auto withNaN = makeWeighted({1.0, 2.0, 3.0});
    withNaN.setLogWeight(2, std::numeric_limits<double>::quiet_NaN());
    // A NaN weight is reported, and nothing is removed.
    EXPECT_EQ(withNaN.pruneToSize(2).error(), WeightError::NonFinite);
    EXPECT_EQ(withNaN.size(), 3);
}

// Assigning a mixture to itself changes nothing, and a move between allocators that differ and
// do not propagate, as two pmr resources, copies the entries into the target's own storage, as
// the allocator-extended move does.
TEST(GaussianMixture, SelfAssignmentAndMovesBetweenAllocators)
{
    auto mixture = makeWeighted({1.0, 3.0});
    auto & alias = mixture;
    mixture = alias;
    expectWeights(mixture, {1.0, 3.0});
    mixture = std::move(alias);
    expectWeights(mixture, {1.0, 3.0});

    using Pooled = GaussianMixtureParams<Linear<2>, double, std::pmr::polymorphic_allocator<>>;
    auto first = std::pmr::monotonic_buffer_resource{};
    auto second = std::pmr::monotonic_buffer_resource{};
    auto source = Pooled{std::pmr::polymorphic_allocator<>{&first}};
    source.add(0.0, Eigen::Vector2d::Ones(), Eigen::Matrix2d::Identity());
    auto target = Pooled{std::pmr::polymorphic_allocator<>{&second}};
    target = std::move(source);
    ASSERT_EQ(target.size(), 1);
    EXPECT_EQ(target.begin()->component.mean(), Eigen::Vector2d::Ones());
    EXPECT_EQ(target.get_allocator().resource(), &second);

    // The allocator-extended move does the same.
    auto moved = Pooled{std::move(target), std::pmr::polymorphic_allocator<>{&first}};
    ASSERT_EQ(moved.size(), 1);
    EXPECT_EQ(moved.begin()->component.mean(), Eigen::Vector2d::Ones());
    EXPECT_EQ(moved.get_allocator().resource(), &first);
}

// remove(i) moves the last component into place i, and removing the last component moves nothing.
TEST(GaussianMixture, RemoveMovesLastIntoPlace)
{
    auto mixture = makeWeighted({1.0, 2.0, 3.0});
    mixture.remove(0);
    ASSERT_EQ(mixture.size(), 2);
    EXPECT_DOUBLE_EQ(mixture[0].logWeight, std::log(3.0));
    EXPECT_DOUBLE_EQ(mixture[1].logWeight, std::log(2.0));
    mixture.remove(1);
    ASSERT_EQ(mixture.size(), 1);
    EXPECT_DOUBLE_EQ(mixture[0].logWeight, std::log(3.0));
}

// forEach() lets the callback read each entry and change its component's mean, covariance, or
// moments, and reweight() also scales each weight by the log-factor it returns; forEach() leaves
// the weights alone. The mixture sets a component, its moments, or its weight's scale by position.
TEST(GaussianMixture, ForEachScalesWeightsAndModifiesComponents)
{
    auto mixture = makeWeighted({1.0, 2.0});
    mixture.reweight([](auto & c) {
        c.setMean(Eigen::Vector2d::Zero());
        return std::log(2.0);
    });
    expectWeights(mixture, {2.0, 4.0});
    for (auto const & c : mixture)
    {
        EXPECT_EQ(c.component.mean(), Eigen::Vector2d::Zero());
    }

    mixture.forEach([](auto & c) {
        c.setCovariance(2.0 * Eigen::Matrix2d::Identity());
    });
    expectWeights(mixture, {2.0, 4.0});
    for (auto const & c : mixture)
    {
        EXPECT_EQ(c.component.covarianceLower(), 2.0 * Eigen::Matrix2d::Identity());
    }

    auto const unit = Emrys::GaussianParams<Linear<2>>{Eigen::Vector2d::Ones(), Eigen::Matrix2d::Identity()};
    mixture.forEach([&](auto & c) {
        EXPECT_DOUBLE_EQ(c.logWeight(), std::log(2.0 * double(c.index() + 1)));
        EXPECT_EQ(c.component().mean(), Eigen::Vector2d::Zero());
        EXPECT_EQ(c.moments().mean(), c.mean());
        EXPECT_EQ(c.covarianceLower(), 2.0 * Eigen::Matrix2d::Identity());
        c.setMoments(unit.moments());
    });
    for (auto const & c : mixture)
    {
        EXPECT_EQ(c.component.mean(), Eigen::Vector2d::Ones());
        EXPECT_EQ(c.component.covarianceLower(), Eigen::Matrix2d::Identity());
    }

    mixture.setComponent(0, Emrys::GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), unit.covarianceLower()});
    mixture.setMoments(1, unit.moments());
    mixture.scaleWeight(0, std::log(0.5));
    expectWeights(mixture, {1.0, 4.0});
    EXPECT_EQ(mixture.begin()->component.mean(), Eigen::Vector2d::Zero());
    EXPECT_EQ(mixture[1].component.mean(), Eigen::Vector2d::Ones());
}

// merge(a, b, moments) replaces a with the given moments and the summed weight and removes b,
// returning the merged component.
TEST(GaussianMixture, MergeSumsWeights)
{
    auto mixture = makeWeighted({1.0, 3.0, 5.0});
    auto const merged = Emrys::Moments<Emrys::Linear<2>>{Eigen::Vector2d{7.0, 8.0}, Eigen::Matrix2d::Identity()};
    auto const kept = mixture.merge(0, 1, merged);
    EXPECT_EQ(mixture.size(), 2);
    EXPECT_EQ(kept, 0);
    EXPECT_DOUBLE_EQ(mixture[kept].logWeight, std::log(4.0));
    EXPECT_EQ(mixture[kept].component.mean(), merged.mean());
    expectWeights(mixture, {4.0, 5.0});
}

// Merging into the last component, where removing the other moves the last into its place, returns
// the index the merged component moved to, with the summed weight.
TEST(GaussianMixture, MergeIntoLastComponent)
{
    auto mixture = makeWeighted({1.0, 3.0, 5.0});
    auto const kept = mixture.merge(2, 0, mixture[2].component.moments());
    EXPECT_EQ(mixture.size(), 2);
    EXPECT_EQ(kept, 0);
    EXPECT_DOUBLE_EQ(mixture[kept].logWeight, std::log(6.0));
    EXPECT_EQ(mixture[kept].component.mean(), Eigen::Vector2d::Constant(2.0));
}

// merged(a, b) is the moment-matched merge, by hand for two one-dimensional components of
// unnormalized weights 0.45 and 0.10 (scaled by e^3): weight 0.55, the weighted mean, and the
// weighted variances plus the spread of the means. It leaves the mixture as it was, and merge(a, b)
// then installs exactly that component with the summed weight.
TEST(GaussianMixture, MergedIsTheMomentMatchedPairAndChangesNothing)
{
    using Matrix1 = Eigen::Matrix<double, 1, 1>;
    auto mixture = GaussianMixtureParams<Linear<1>>{1};
    mixture.add(std::log(0.45) + 3.0, Matrix1{1.0}, Matrix1{1.0});
    mixture.add(std::log(0.10) + 3.0, Matrix1{2.5}, Matrix1{2.0});
    double const mean = (0.45 * 1.0 + 0.10 * 2.5) / 0.55;
    double const variance = (0.45 * 1.0 + 0.10 * 2.0) / 0.55 + 0.45 * 0.10 / (0.55 * 0.55) * 1.5 * 1.5;

    auto const merged = mixture.merged(0, 1);
    EXPECT_NEAR(merged.mean()(0), mean, 1e-14);
    EXPECT_NEAR(merged.covarianceLower()(0, 0), variance, 1e-14);
    ASSERT_EQ(mixture.size(), 2);
    EXPECT_EQ(mixture.begin()->component.mean()(0), 1.0);
    EXPECT_EQ(mixture[1].component.mean()(0), 2.5);

    auto const kept = mixture.merge(0, 1);
    EXPECT_EQ(mixture.size(), 1);
    EXPECT_NEAR(mixture[kept].logWeight, std::log(0.55) + 3.0, 1e-14);
    EXPECT_EQ(mixture[kept].component.mean(), merged.mean());
    EXPECT_EQ(mixture[kept].component.covarianceLower(), merged.covarianceLower());
}

// Merging two components either side of the seam of a periodic coordinate gives mean pi and the
// variances plus the spread of the means, measured around the circle.
TEST(GaussianMixture, MergedAveragesAcrossAPeriodicSeam)
{
    struct Heading : Emrys::RadiansPi<1>
    {};

    using Matrix1 = Eigen::Matrix<double, 1, 1>;
    double const pi = std::numbers::pi;
    auto mixture = GaussianMixtureParams<Heading>{Heading{}};
    mixture.add(std::log(0.5), Matrix1{pi - 0.1}, Matrix1{0.01});
    mixture.add(std::log(0.5), Matrix1{-pi + 0.1}, Matrix1{0.01});
    auto const merged = mixture.merged(0, 1);
    EXPECT_NEAR(std::abs(merged.mean()(0)), pi, 1e-12);
    EXPECT_NEAR(merged.covarianceLower()(0, 0), 0.01 + 0.01, 1e-12);
}

// Merging pairs of a random 12-component mixture down to one component keeps the mixture's mean,
// covariance, and total weight, to rounding, at every size: moment matching preserves them.
TEST(GaussianMixture, MergingKeepsTheMixtureMoments)
{
    auto rng = std::mt19937_64{6};
    auto gaussian = std::normal_distribution<double>{0.0, 3.0};
    auto uniform = std::uniform_real_distribution<double>{0.1, 1.0};
    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    for (int i = 0; i < 12; ++i)
    {
        auto const mean = Eigen::Vector2d{gaussian(rng), gaussian(rng)};
        auto const root = Eigen::Matrix2d{{uniform(rng), 0.0}, {uniform(rng) - 0.5, uniform(rng)}};
        mixture.add(std::log(uniform(rng)), mean, Eigen::Matrix2d{root * root.transpose()});
    }
    auto const before = mixture.toDensity()->moments();
    double const totalBefore = mixture.logTotalWeight().value();
    while (mixture.size() > 1)
    {
        // The last component into the first, then the first two: both ends of the order.
        static_cast<void>(mixture.merge(0, mixture.size() - 1));
        if (mixture.size() > 1)
        {
            static_cast<void>(mixture.merge(0, 1));
        }
        auto const after = mixture.toDensity()->moments();
        EXPECT_LT((after.mean() - before.mean()).norm(), 1e-12) << mixture.size();
        EXPECT_LT((after.covarianceLower() - before.covarianceLower()).norm(), 1e-12) << mixture.size();
        EXPECT_NEAR(mixture.logTotalWeight().value(), totalBefore, 1e-12) << mixture.size();
    }
}

// split(it, children) replaces a component with children whose given weights are scaled to share
// the parent's weight, keeping the total.
TEST(GaussianMixture, SplitRedistributesWeight)
{
    auto mixture = makeWeighted({6.0, 1.0});
    auto const a = Emrys::Moments<Emrys::Linear<2>>{Eigen::Vector2d{-1.0, 0.0}, Eigen::Matrix2d::Identity()};
    auto const b = Emrys::Moments<Emrys::Linear<2>>{Eigen::Vector2d{1.0, 0.0}, Eigen::Matrix2d::Identity()};
    auto const children = std::array{std::pair{std::log(1.0), a}, std::pair{std::log(2.0), b}};
    mixture.split(0, children);

    ASSERT_EQ(mixture.size(), 3);
    EXPECT_DOUBLE_EQ(mixture.begin()->logWeight, std::log(2.0));
    EXPECT_EQ(mixture.begin()->component.mean(), a.mean());
    EXPECT_DOUBLE_EQ(mixture[2].logWeight, std::log(4.0));
    EXPECT_EQ(mixture[2].component.mean(), b.mean());
    EXPECT_NEAR(mixture.logTotalWeight().value(), std::log(7.0), 1e-12);
}

// removeIf(predicate) drops the selected components, and removeIf<true>() also returns the log of
// the weight fraction removed.
TEST(GaussianMixture, RemoveIf)
{
    auto mixture = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    auto const light = [](Mixture2::Entry const & c) {
        return c.logWeight < std::log(2.5);
    };

    auto copy = mixture;
    copy.removeIf(light);
    expectWeights(copy, {3.0, 4.0, 5.0});

    auto const removed = mixture.removeIf<true>(light);
    ASSERT_TRUE(removed.has_value());
    EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
    expectWeights(mixture, {3.0, 4.0, 5.0});
}

// removeIf<true>() on all-zero weights fails with ZeroSum and removes nothing, while the plain
// removeIf() still removes.
TEST(GaussianMixture, RemoveIfReturningWeightFailsWithoutRemoving)
{
    auto mixture = makeWeighted({1.0, 1.0});
    mixture.scaleWeights(-inf);
    auto const removed = mixture.removeIf<true>([](auto const &) {
        return true;
    });
    EXPECT_EQ(removed.error(), WeightError::ZeroSum);
    EXPECT_EQ(mixture.size(), 2);

    mixture.removeIf([](auto const &) {
        return true;
    });
    EXPECT_TRUE(mixture.empty());
}

// pruneToSize(k) keeps the k heaviest components and normalizes them: of weights 1, 5, 2, 4, and
// 3, summing to 15, three leave 3, 4, and 5 over 12 and return the log of the 3/15 removed; pruning
// to the size again removes nothing and returns -inf.
TEST(GaussianMixture, PruneToSizeKeepsTheHeaviest)
{
    auto mixture = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    auto const removed = mixture.pruneToSize(3);
    ASSERT_TRUE(removed.has_value());
    EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
    expectWeights(mixture, {3.0 / 12.0, 4.0 / 12.0, 5.0 / 12.0});

    EXPECT_EQ(mixture.pruneToSize(3).value(), -inf);
    EXPECT_EQ(mixture.size(), 3);
}

// pruneTail(log f) removes the lightest components while their combined weight stays below f: of
// weights 1, 5, 2, 4, and 3 over 15, f = 0.25 removes 1 and 2 (0.2), since adding 3 would reach
// 0.4, and f = 0.15 removes only 1, since 1 and 2 reach 0.2.
TEST(GaussianMixture, PruneTailRemovesTheLightestBelowAFraction)
{
    auto mixture = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    auto copy = mixture;
    auto const removed = mixture.pruneTail(std::log(0.25));
    ASSERT_TRUE(removed.has_value());
    EXPECT_NEAR(*removed, std::log(3.0 / 15.0), 1e-12);
    expectWeights(mixture, {3.0 / 12.0, 4.0 / 12.0, 5.0 / 12.0});

    EXPECT_NEAR(copy.pruneTail(std::log(0.15)).value(), std::log(1.0 / 15.0), 1e-12);
    expectWeights(copy, {2.0 / 14.0, 3.0 / 14.0, 4.0 / 14.0, 5.0 / 14.0});
}

// mostProbable() is the index of the heaviest component, the first of tied ones, from the
// parameters and the density alike; a mixture of one has it at zero.
TEST(GaussianMixture, MostProbable)
{
    auto mixture = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0});
    EXPECT_EQ(mixture.mostProbable(), 1);
    EXPECT_EQ(mixture.toDensity().value().mostProbable(), 1);

    mixture.setLogWeight(3, std::log(5.0));
    EXPECT_EQ(mixture.mostProbable(), 1);
    EXPECT_EQ(mixture.toDensity().value().mostProbable(), 1);

    EXPECT_EQ(makeWeighted({2.0}).mostProbable(), 0);
}

// pruneBelow(log p) removes the components whose normalized weight is below p and normalizes the
// rest, which keep their order: of weights 1, 5, 2, 4, 3, and 0.01, summing to 15.01, p = 0.1
// removes 1 and 0.01, leaving 5, 2, 4, and 3 over 14. The weights' scale, here e^7, doesn't
// matter, and a second prune removes nothing.
TEST(GaussianMixture, PruneBelowRemovesImprobableComponentsAndNormalizes)
{
    auto mixture = makeWeighted({1.0, 5.0, 2.0, 4.0, 3.0, 0.01});
    mixture.scaleWeights(7.0);
    EXPECT_NEAR(mixture.pruneBelow(std::log(0.1)).value(), std::log(1.01 / 15.01), 1e-12);
    auto const expected = std::vector<std::array<double, 2>>{{1.0, 5.0}, {2.0, 2.0}, {3.0, 4.0}, {4.0, 3.0}};
    ASSERT_EQ(mixture.size(), Eigen::Index(expected.size()));
    for (std::size_t k = 0; k < expected.size(); ++k)
    {
        auto const i = Eigen::Index(k);
        EXPECT_EQ(mixture[i].component.mean()(0), expected[k][0]);
        EXPECT_NEAR(mixture[i].logWeight, std::log(expected[k][1] / 14.0), 1e-12);
    }
    EXPECT_EQ(mixture.pruneBelow(std::log(0.1)).value(), -inf);
}

// pruneBelow() keeps the most probable component whatever the threshold, the first of tied ones,
// so a threshold above every probability leaves that one, with weight one.
TEST(GaussianMixture, PruneBelowKeepsTheMostProbable)
{
    auto mixture = makeWeighted({1.0, 3.0, 3.0, 2.0});
    EXPECT_NEAR(mixture.pruneBelow(std::log(0.99)).value(), std::log(6.0 / 9.0), 1e-12);
    ASSERT_EQ(mixture.size(), 1);
    EXPECT_EQ(mixture[0].component.mean()(0), 1.0);
    EXPECT_DOUBLE_EQ(mixture[0].logWeight, 0.0);
}

// Each prune reports the WeightError of logTotalWeight() and removes nothing: Empty, ZeroSum, and
// NonFinite for a NaN weight, which has no place in the order by weight.
TEST(GaussianMixture, PrunesReportTheWeightError)
{
    auto const prunes = [](Mixture2 & mixture) {
        return std::array{
            mixture.pruneBelow(std::log(0.1)).error(),
            mixture.pruneTail(std::log(0.1)).error(),
            mixture.pruneToSize(1).error(),
        };
    };
    auto empty = Mixture2{};
    for (WeightError const error : prunes(empty))
    {
        EXPECT_EQ(error, WeightError::Empty);
    }

    auto zero = makeWeighted({1.0, 2.0});
    zero.scaleWeights(-inf);
    for (WeightError const error : prunes(zero))
    {
        EXPECT_EQ(error, WeightError::ZeroSum);
    }
    EXPECT_EQ(zero.size(), 2);

    auto nan = makeWeighted({1.0, 5.0, 2.0});
    nan.setLogWeight(1, std::numeric_limits<double>::quiet_NaN());
    for (WeightError const error : prunes(nan))
    {
        EXPECT_EQ(error, WeightError::NonFinite);
    }
    EXPECT_EQ(nan.size(), 3);
}

// normalize() scales the weights to sum to one.
TEST(GaussianMixture, Normalize)
{
    auto mixture = makeWeighted({1.0, 3.0});
    ASSERT_TRUE(mixture.normalize().has_value());
    EXPECT_NEAR(mixture.logTotalWeight().value(), 0.0, 1e-15);
    EXPECT_NEAR(std::exp(mixture.begin()->logWeight), 0.25, 1e-15);
}

// assign() replaces the components from a range of (log-weight, Moments) pairs, and append() adds
// another mixture's components.
TEST(GaussianMixture, AssignAndAppend)
{
    using Moments = Emrys::Moments<Emrys::Linear<2>>;
    auto mixture = Mixture2{};
    auto const components = std::vector{
        std::pair{0.0, Moments{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}},
        std::pair{1.0, Moments{Eigen::Vector2d::Ones(), Eigen::Matrix2d::Identity()}}
    };
    mixture.assign(components);
    EXPECT_EQ(mixture.size(), 2);
    EXPECT_EQ(mixture[1].component.mean(), Eigen::Vector2d::Ones());

    mixture.append(mixture);
    EXPECT_EQ(mixture.size(), 4);
    EXPECT_EQ(mixture[3].component.mean(), Eigen::Vector2d::Ones());

    // assign() replaces the lineage's rows too, giving each new component an orphan row.
    mixture.startLineage();
    mixture.assign(components);
    EXPECT_EQ(mixture.size(), 2);
    EXPECT_EQ(mixture.lineage().size(), 2);
}

// A mixture in a fixed-size space takes a capacity without the space; one whose space holds its
// size takes the space too.
TEST(GaussianMixture, CapacityWithoutASpace)
{
    auto const mixture = Mixture2{Emrys::Capacity{5}};
    EXPECT_EQ(mixture.capacity(), 5);
    EXPECT_EQ(mixture.size(), 0);
    EXPECT_EQ(mixture.valueDimension(), 2);
    static_assert(!std::is_constructible_v<GaussianMixtureParams<Linear<Eigen::Dynamic>>, Emrys::Capacity>);
    auto const sized = GaussianMixtureParams<Linear<Eigen::Dynamic>>{Linear<Eigen::Dynamic>{3}, Emrys::Capacity{5}};
    EXPECT_EQ(sized.capacity(), 5);
}

// resizeComponents(n) changes a dynamic mixture's dimension, after which components are set and
// added at the new size.
TEST(GaussianMixture, ResizeComponentsDynamic)
{
    auto mixture = GaussianMixtureParams<Linear<Eigen::Dynamic>>{2};
    mixture.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    EXPECT_EQ(mixture.valueDimension(), 2);

    mixture.resizeComponents(3);
    EXPECT_EQ(mixture.valueDimension(), 3);
    mixture.setMean(0, Eigen::Vector3d::Zero());
    mixture.setCovariance(0, Eigen::Matrix3d::Identity());
    mixture.add(0.0, Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity());
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());
    EXPECT_EQ(g->moments().mean(), Eigen::Vector3d::Constant(0.5));
}

// add(), setMean(), and setCovariance() move an rvalue of the stored type (same data pointer) into the
// component [forwarded-matrices].
TEST(GaussianMixtureParams, MovesRvalueMeanAndCov)
{
    auto mixture = GaussianMixtureParams<Linear<Eigen::Dynamic>>{2};
    Eigen::VectorXd mean = Eigen::Vector2d{1.0, 2.0};
    Eigen::MatrixXd cov = Eigen::Matrix2d::Identity();
    double const * meanData = mean.data();
    double const * covData = cov.data();
    auto const added = mixture.add(0.0, std::move(mean), std::move(cov));
    EXPECT_EQ(mixture[added].component.mean().data(), meanData);
    EXPECT_EQ(mixture[added].component.covarianceLower().data(), covData);

    Eigen::VectorXd newMean = Eigen::Vector2d{3.0, 4.0};
    Eigen::MatrixXd newCov = 2.0 * Eigen::Matrix2d::Identity();
    meanData = newMean.data();
    covData = newCov.data();
    mixture.setMean(added, std::move(newMean));
    mixture.setCovariance(added, std::move(newCov));
    EXPECT_EQ(mixture[added].component.mean().data(), meanData);
    EXPECT_EQ(mixture[added].component.covarianceLower().data(), covData);
}

// Before startLineage() a mixture has no lineage; after it each component is its own parent, a
// component added later has none, split() children take their parent's row, and merge() sums two
// rows by weight: merging components 1 and 2, of weights 2 and 3, gives the merge shares 0.4 and
// 0.6, and then 1 and the merge, of weights 0.5 (half of 1, split) and 5, give shares of 1/11, 4/11
// (0.4 of 10/11), and 6/11.
TEST(GaussianMixtureLineage, StartsAndFollowsAddSplitAndMerge)
{
    auto mixture = makeWeighted({1.0, 2.0, 3.0});
    EXPECT_EQ(mixture.lineage().count(), 0);
    EXPECT_EQ(mixture.lineage().size(), 0);
    mixture.startLineage();
    auto const & lineage = mixture.lineage();
    EXPECT_EQ(lineage.count(), 1);
    ASSERT_EQ(lineage.parentCount(), 3);
    ASSERT_EQ(lineage.size(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        for (Eigen::Index n = 0; n < 3; ++n)
        {
            EXPECT_EQ(lineage.logShare(i, n), i == n ? 0.0 : -inf);
        }
    }

    static_cast<void>(mixture.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()));
    ASSERT_EQ(lineage.size(), 4);
    for (Eigen::Index n = 0; n < 3; ++n)
    {
        EXPECT_EQ(lineage.logShare(3, n), -inf);
    }

    auto const halves = std::array<std::pair<double, Emrys::GaussianParams<Linear<2>>>, 2>{
        std::pair{0.0, Emrys::GaussianParams<Linear<2>>{Eigen::Vector2d{-0.1, 0.0}, Eigen::Matrix2d::Identity()}},
        std::pair{0.0, Emrys::GaussianParams<Linear<2>>{Eigen::Vector2d{0.1, 0.0}, Eigen::Matrix2d::Identity()}}
    };
    mixture.split(0, halves);
    ASSERT_EQ(mixture.size(), 5);
    EXPECT_EQ(lineage.logShare(4, 0), 0.0);
    EXPECT_EQ(lineage.logShare(4, 1), -inf);

    // Merging 1 and 2: the last component (the second half of 0) moves into 2's place.
    static_cast<void>(mixture.merge(1, 2));
    ASSERT_EQ(lineage.size(), 4);
    EXPECT_NEAR(std::exp(lineage.logShare(1, 1)), 0.4, 1e-15);
    EXPECT_NEAR(std::exp(lineage.logShare(1, 2)), 0.6, 1e-15);
    EXPECT_EQ(lineage.logShare(1, 0), -inf);
    EXPECT_EQ(lineage.logShare(2, 0), 0.0);

    static_cast<void>(mixture.merge(2, 1));
    ASSERT_EQ(lineage.size(), 3);
    EXPECT_NEAR(std::exp(lineage.logShare(2, 0)), 1.0 / 11, 1e-15);
    EXPECT_NEAR(std::exp(lineage.logShare(2, 1)), 4.0 / 11, 1e-15);
    EXPECT_NEAR(std::exp(lineage.logShare(2, 2)), 6.0 / 11, 1e-15);
}

// carry() takes a value per parent to the components: a component with one parent takes its value,
// a merge the share-weighted mean (0.4 of parent 1's value and 0.6 of parent 2's, from weights 2 and
// 3), and a component added with no parent the orphan value; the values are what a running NIS
// per component would be [user-extension-points].
TEST(GaussianMixtureLineage, CarriesValuesByShares)
{
    auto mixture = makeWeighted({1.0, 2.0, 3.0});
    mixture.startLineage();
    static_cast<void>(mixture.merge(1, 2));
    static_cast<void>(mixture.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()));
    ASSERT_EQ(mixture.size(), 3);
    auto const parents = Eigen::Vector3d{10.0, 20.0, 30.0};
    auto children = Eigen::VectorXd(mixture.size());
    mixture.lineage().carry(parents, children, -1.0);
    EXPECT_DOUBLE_EQ(children(0), 10.0);
    EXPECT_NEAR(children(1), 0.4 * 20.0 + 0.6 * 30.0, 1e-12);
    EXPECT_DOUBLE_EQ(children(2), -1.0);
}

// removeIf(), pruneToSize(), pruneBelow(), and remove() drop the removed components' rows and keep the rest
// with their components; clear() drops every row but keeps the mark.
TEST(GaussianMixtureLineage, RemovalKeepsRowsWithTheirComponents)
{
    auto mixture = makeWeighted({1.0, 2.0, 3.0, 4.0, 5.0});
    mixture.startLineage();
    auto const & lineage = mixture.lineage();
    // Each row's own parent is the component's original position, its mean.
    auto const parentOf = [&](Eigen::Index i) {
        for (Eigen::Index n = 0; n < lineage.parentCount(); ++n)
        {
            if (lineage.logShare(i, n) == 0.0)
            {
                return n;
            }
        }
        return Eigen::Index(-1);
    };
    auto const expectAligned = [&] {
        ASSERT_EQ(lineage.size(), mixture.size());
        for (Eigen::Index i = 0; i < mixture.size(); ++i)
        {
            EXPECT_EQ(parentOf(i), Eigen::Index(mixture[i].component.mean()(0)));
        }
    };
    mixture.removeIf([](auto const & entry) {
        return entry.component.mean()(0) == 1.0;
    });
    expectAligned();
    ASSERT_TRUE(mixture.pruneToSize(3));
    expectAligned();
    // Weights 3, 4, and 5 over 12: the first is below 0.3.
    EXPECT_NEAR(mixture.pruneBelow(std::log(0.3)).value(), std::log(0.25), 1e-12);
    expectAligned();
    static_cast<void>(mixture.remove(0));
    expectAligned();
    mixture.clear();
    EXPECT_EQ(lineage.size(), 0);
    EXPECT_EQ(lineage.count(), 1);
    EXPECT_EQ(lineage.parentCount(), 5);
}

// A copy is a new mixture and has no lineage, by construction or assignment; a move, by
// construction or assignment, and swap() take it.
// Merging two components of weight zero leaves the kept row's shares as they were, since a share
// of no weight is undefined.
TEST(GaussianMixtureLineage, MergingWeightlessComponentsKeepsTheShares)
{
    auto mixture = makeWeighted({1.0, 1.0, 1.0});
    mixture.setLogWeight(0, -inf);
    mixture.setLogWeight(1, -inf);
    mixture.startLineage();
    auto const merged = Emrys::Moments<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    static_cast<void>(mixture.merge(0, 1, merged));
    EXPECT_EQ(mixture.lineage().logShare(0, 0), 0.0);
    EXPECT_EQ(mixture.lineage().logShare(0, 1), -inf);
}

TEST(GaussianMixtureLineage, CopiesHaveNoneAndMovesKeepIt)
{
    auto mixture = makeWeighted({1.0, 2.0});
    mixture.startLineage();
    mixture.startLineage();
    Mixture2 const copy = mixture;
    EXPECT_EQ(copy.lineage().count(), 0);
    EXPECT_EQ(copy.lineage().size(), 0);
    auto assigned = makeWeighted({1.0});
    assigned.startLineage();
    assigned = mixture;
    EXPECT_EQ(assigned.lineage().count(), 0);

    Mixture2 moved = std::move(mixture);
    EXPECT_EQ(moved.lineage().count(), 2);
    EXPECT_EQ(moved.lineage().size(), 2);
    auto moveAssigned = Mixture2{};
    moveAssigned = std::move(moved);
    EXPECT_EQ(moveAssigned.lineage().count(), 2);
    moveAssigned.swap(assigned);
    EXPECT_EQ(assigned.lineage().count(), 2);
    EXPECT_EQ(moveAssigned.lineage().count(), 0);
}

// Copies and moves of a view read the same density, and neither they nor a finished forEach()
// keep the mixture from changing once they are no longer read [view-stamp].
TEST(GaussianMixture, ViewsCopyAndMoveFreely)
{
    auto mixture = makeWeighted({1.0, 2.0});
    {
        auto g = mixture.toDensity().value();
        auto const copy = g;
        auto const moved = std::move(g);
        EXPECT_EQ(copy.logDensity(Eigen::Vector2d::Zero()), moved.logDensity(Eigen::Vector2d::Zero()));
    }
    mixture.scaleWeights(1.0);
    mixture.forEach([](auto &) {});
    mixture.scaleWeights(1.0);
    EXPECT_TRUE(mixture.toDensity().has_value());
}

/// Whether \a M, with its value category, can make a density.
template <typename M>
concept MakesDensity = requires(M && m) { std::forward<M>(m).toDensity(); };

// A density of a temporary mixture would dangle, so toDensity() on an rvalue does not compile [view-stamp].
TEST(GaussianMixture, NoDensityOfATemporary)
{
    static_assert(MakesDensity<Mixture2 &>);
    static_assert(MakesDensity<Mixture2 const &>);
    static_assert(!MakesDensity<Mixture2>);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// An empty mixture has no most probable component, and asking for one dies.
TEST(GaussianMixtureDeathTest, MostProbableOfAnEmptyMixture)
{
    EXPECT_DEATH(std::ignore = Mixture2{}.mostProbable(), "empty mixture");
}

// Reading a density after any mutation of its mixture dies [view-stamp].
TEST(GaussianMixtureDeathTest, ReadAfterChange)
{
    auto mixture = makeWeighted({1.0, 2.0});
    auto const g = mixture.toDensity();
    Eigen::Vector2d const x = Eigen::Vector2d::Zero();
    std::string const message = "changed while a view";
    EXPECT_DEATH(
        {
            mixture.add(0.0, x, Eigen::Matrix2d::Identity());
            std::ignore = g->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.remove(0);
            std::ignore = g->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.setMean(0, x);
            std::ignore = g->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.scaleWeights(1.0);
            std::ignore = g->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.forEach([](auto &) {});
            std::ignore = g->logDensity(x);
        },
        message
    );
    EXPECT_DEATH(
        {
            (void)mixture.pruneToSize(1);
            std::ignore = g->size();
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.clear();
            std::ignore = g->size();
        },
        message
    );
}

// Changing the mixture from inside its own forEach() or removeIf() callback dies when the callback
// returns.
TEST(GaussianMixtureDeathTest, ChangeMixtureInsideCallback)
{
    auto mixture = makeWeighted({1.0, 2.0});
    EXPECT_DEATH(
        mixture.forEach([&](auto &) {
            mixture.scaleWeights(1.0);
        }),
        "changed inside their own callback"
    );
    EXPECT_DEATH(
        mixture.removeIf([&](auto const &) {
            mixture.scaleWeights(1.0);
            return false;
        }),
        "changed inside their own callback"
    );
}

// Reading a density of a mixture that was moved from dies.
TEST(GaussianMixtureDeathTest, ReadAfterMove)
{
    auto mixture = makeWeighted({1.0, 2.0});
    auto const g = mixture.toDensity();
    EXPECT_DEATH(
        {
            auto stolen = std::move(mixture);
            std::ignore = g->size();
        },
        "changed while a view"
    );
}

#endif

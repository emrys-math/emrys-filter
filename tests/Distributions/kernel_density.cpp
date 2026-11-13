// KernelDensity over a ParticleSet (step 7): the log-density is the sum of the kernels, the moments
// add the bandwidth, Silverman's rule gives the default bandwidth, a single particle is a Gaussian,
// degenerate particles fail, the view checks its set [view-stamp], and samples match the estimate, in a
// batch or one draw at a time [allocation].

#include <EmrysFilter/Distributions/KernelDensity.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "Distributions/draw_stream.h"

using Emrys::GaussianError;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ParticleSet;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

// Each config names a ParticleSet type and the particle dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config
{
    using Type = ParticleSet<Linear<Size, MaxSize>, Scalar>;
    static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<
    Config<double, 1, 1, 1>,
    Config<double, 3, 3, 3>,
    Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
    Config<double, Eigen::Dynamic, 6, 4>,
    Config<float, 2, 2, 2>>;

template <typename C>
class KernelDensityTest : public ::testing::Test
{
protected:

    using ParticleSetType = typename C::Type;
    using Scalar = typename ParticleSetType::Scalar;
    using VectorType = typename ParticleSetType::VectorType;
    using MatrixType = typename ParticleSetType::MatrixType;
    using SampleSetType = typename ParticleSetType::SampleSetType;
    using WeightArrayType = typename ParticleSetType::WeightArrayType;
    static constexpr Eigen::Index n = C::n;
    static constexpr Eigen::Index count = 9;

    static Scalar
    tolerance()
    {
        return std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10);
    }

    static ParticleSetType
    make()
    {
        SampleSetType const states = SampleSetType::NullaryExpr(n, count, [](Eigen::Index i, Eigen::Index j) {
            return Scalar(std::sin(1.3 * double(i * j) + 0.7 * double(j) + double(i)));
        });
        WeightArrayType const logWeights = WeightArrayType::NullaryExpr(count, [](Eigen::Index j) {
            return Scalar(std::log(double(j + 1)));
        });
        return ParticleSetType{states, logWeights};
    }

    static MatrixType
    makeBandwidth()
    {
        MatrixType h = MatrixType::Identity(n, n) * Scalar(0.3);
        if (n > 1)
        {
            h(1, 0) = h(0, 1) = Scalar(0.1);
        }
        return h;
    }

    static VectorType
    makePoint()
    {
        return VectorType::NullaryExpr(n, [](Eigen::Index i) {
            return Scalar(0.2 * double(i) - 0.1);
        });
    }
};

}  // namespace

TYPED_TEST_SUITE(KernelDensityTest, Configs);

// The kernel density at a point is the log of the weighted average of one Gaussian kernel per
// particle, computed here with plain Gaussians, and density() is its exponential.
TYPED_TEST(KernelDensityTest, LogDensityMatchesSumOfKernels)
{
    using Scalar = typename TestFixture::Scalar;
    auto const set = this->make();
    auto const kde = set.toDensity(this->makeBandwidth());
    ASSERT_TRUE(kde.has_value());

    auto const x = this->makePoint();
    double density = 0;
    double total = 0;
    for (Eigen::Index i = 0; i < set.size(); ++i)
    {
        typename TestFixture::VectorType const center = set.state(i);
        auto const kernel = GaussianParams<
            Linear<
                TestFixture::ParticleSetType::ValueDimensionAtCompileTime,
                TestFixture::ParticleSetType::MaxValueDimensionAtCompileTime>,
            Scalar>{center, this->makeBandwidth()};
        double const w = std::exp(double(set.logWeight(i)));
        density += w * std::exp(double(kernel.toDensity()->logDensity(x)));
        total += w;
    }
    double const expected = std::log(density / total);
    EXPECT_NEAR(kde->logDensity(x), expected, this->tolerance() * std::abs(expected));
    EXPECT_NEAR(kde->density(x), std::exp(kde->logDensity(x)), this->tolerance());
}

// The estimate's mean is the particles' weighted mean and its covariance is theirs plus the
// bandwidth.
TYPED_TEST(KernelDensityTest, MomentsAddBandwidth)
{
    auto const set = this->make();
    auto const kde = set.toDensity(this->makeBandwidth());
    ASSERT_TRUE(kde.has_value());
    auto const particleMoments = set.moments().value();
    auto const moments = kde->moments();
    EXPECT_EQ(moments.mean(), particleMoments.mean());
    EXPECT_LT(
        (moments.covarianceLower() - particleMoments.covarianceLower() - this->makeBandwidth()).norm(),
        this->tolerance()
    );
}

// silvermanBandwidth() is the particle covariance scaled by Silverman's factor (4 / (ESS (d +
// 2)))^(2 / (d + 4)) with the effective sample size as the count, and it is the default bandwidth
// of toDensity().
TYPED_TEST(KernelDensityTest, SilvermanBandwidth)
{
    using Scalar = typename TestFixture::Scalar;
    auto const set = this->make();
    double const d = double(this->n);
    double const factor = std::pow(4.0 / (double(set.ess().value()) * (d + 2)), 2 / (d + 4));
    auto const expected = (set.moments()->covarianceLower() * Scalar(factor)).eval();
    auto const silverman = Emrys::silvermanBandwidth(set);
    ASSERT_TRUE(silverman.has_value());
    EXPECT_LT((*silverman - expected).norm(), this->tolerance() * expected.norm());

    auto const kde = set.toDensity();
    ASSERT_TRUE(kde.has_value());
    EXPECT_EQ(kde->bandwidth(), *silverman);
}

// A set of one particle with bandwidth H has the density of a Gaussian centered on it with
// covariance H.
TEST(KernelDensity, SingleParticleIsGaussian)
{
    auto const center = Eigen::Vector2d{1.0, -2.0};
    auto const h = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    auto const set = ParticleSet<Linear<2>>{Eigen::Matrix<double, 2, 1>{center}};
    auto const kde = set.toDensity(h);
    ASSERT_TRUE(kde.has_value());

    auto const gaussian = GaussianParams<Linear<2>>{center, h};
    auto const x = Eigen::Vector2d{0.3, 0.4};
    EXPECT_NEAR(kde->logDensity(x), gaussian.toDensity()->logDensity(x), 1e-12);
}

// Coincident particles give a singular Silverman bandwidth, reported as a not-positive-definite
// Gaussian error.
TEST(KernelDensity, DegenerateParticlesFail)
{
    auto const set = ParticleSet<Linear<2>>{Eigen::Matrix<double, 2, 4>::Ones()};
    auto const kde = set.toDensity();
    ASSERT_FALSE(kde.has_value());
    EXPECT_EQ(kde.error(), GaussianError::NotPositiveDefinite);
}

// A KernelDensity checks its set: reading it after a weight change dies [view-stamp], and an estimate
// that is no longer read leaves the set free to change.
TEST(KernelDensity, ChecksItsParticleSet)
{
    auto set = ParticleSet<Linear<2>>{Eigen::Matrix<double, 2, 3>::Random()};
    {
        auto kde = std::optional<Emrys::KernelDensity<Emrys::Linear<2>>>{};
        kde.emplace(set.toDensity(Eigen::Matrix2d::Identity()).value());
#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
        EXPECT_DEATH(
            {
                set.scaleWeights(1.0);
                std::ignore = kde->logDensity(Eigen::Vector2d::Zero());
            },
            "changed while a view"
        );
#endif
    }
    set.scaleWeights(1.0);
    EXPECT_TRUE(set.toDensity(Eigen::Matrix2d::Identity()).has_value());
}

// 40,000 draws from the estimate have a sample mean within five standard errors of its mean and a
// sample covariance within 5% of its covariance in norm.
TYPED_TEST(KernelDensityTest, SampleMomentsConverge)
{
    auto const set = this->make();
    auto const kde = set.toDensity(this->makeBandwidth());
    ASSERT_TRUE(kde.has_value());
    auto rng = std::mt19937_64{13};
    constexpr Eigen::Index count = 40000;
    auto const samples = Emrys::drawSamples(*kde, count, rng);
    ASSERT_EQ(samples.rows(), this->n);
    ASSERT_EQ(samples.cols(), count);

    auto const moments = kde->moments();
    Eigen::MatrixXd const x = samples.template cast<double>();
    Eigen::VectorXd const mean = x.rowwise().mean();
    Eigen::MatrixXd const centered = x.colwise() - mean;
    Eigen::MatrixXd const cov = centered * centered.transpose() / double(count - 1);
    Eigen::MatrixXd const p = moments.covarianceLower().template cast<double>();
    for (Eigen::Index i = 0; i < this->n; ++i)
    {
        EXPECT_NEAR(mean(i), double(moments.mean()(i)), 5 * std::sqrt(p(i, i) / double(count))) << "mean " << i;
    }
    EXPECT_LT((cov - p).norm(), 0.05 * p.norm());
}

// 40,000 single draws, one drawSample() at a time, match the estimate's moments as the batch does;
// with their log-densities they are the same draws, each with exactly the estimate's logDensity().
TYPED_TEST(KernelDensityTest, StreamOfSingleDrawsMatchesTheEstimate)
{
    auto const set = this->make();
    auto const kde = set.toDensity(this->makeBandwidth());
    ASSERT_TRUE(kde.has_value());
    auto rng = std::mt19937_64{17};
    auto const moments = kde->moments();
    Eigen::MatrixXd const draws = EmrysTest::streamOfDraws(*kde, 40'000, rng);
    ASSERT_EQ(draws.rows(), this->n);
    EmrysTest::expectSampleMoments(
        draws,
        moments.mean().template cast<double>(),
        moments.covarianceLower().template cast<double>(),
        0.05
    );
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(*kde, 50, 18, 0.0);
}

// drawSamplesWithLogDensity() draws the same samples as drawSamples() from the same seed and
// reports the estimate's own log-density at each, in the set's scalar type.
TYPED_TEST(KernelDensityTest, SampledLogDensityMatchesEstimate)
{
    using Scalar = typename TestFixture::Scalar;
    auto const set = this->make();
    auto const kde = set.toDensity(this->makeBandwidth());
    auto a = std::mt19937_64{17};
    auto b = std::mt19937_64{17};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*kde, 20, a);
    EXPECT_EQ(drawn.samples, Emrys::drawSamples(*kde, 20, b));
    ASSERT_EQ(drawn.logDensity.size(), 20);
    for (Eigen::Index j = 0; j < 20; ++j)
    {
        EXPECT_EQ(drawn.logDensity(j), kde->logDensity(drawn.samples.col(j))) << "sample " << j;
    }
    EXPECT_TRUE(drawn.logDensity.isFinite().all());
    static_assert(std::is_same_v<typename decltype(drawn.logDensity)::Scalar, Scalar>);
}

// With all the weight on one particle and a narrow kernel, every sample lies within 0.1 of that
// particle.
TEST(KernelDensity, SamplesStayNearWeightedParticles)
{
    // Only the middle particle has weight, and the kernel is narrow.
    auto const states = Eigen::Matrix<double, 2, 3>{{-10.0, 0.0, 10.0}, {0.0, 5.0, 0.0}};
    auto const set = ParticleSet<Linear<2>>{states, Eigen::Array3d{-inf, 0.0, -inf}};
    auto const kde = set.toDensity(Eigen::Matrix2d::Identity() * 1e-4);
    auto rng = std::mt19937_64{19};
    auto const samples = Emrys::drawSamples(*kde, 100, rng, Emrys::resamplers::systematic());
    EXPECT_LT((samples.colwise() - states.col(1)).cwiseAbs().maxCoeff(), 0.1);
}

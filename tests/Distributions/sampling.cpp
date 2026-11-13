// Sampling from a Gaussian and a GaussianMixtureParams (step 7): sample moments converge, the
// samples' log-density matches the source, the two overloads draw the same samples from the same
// generator, samples feed a ParticleSet, zero samples work, and systematic mixture sampling gives
// each component its share; and drawSample(), one draw at a time, matches the moments of either, with
// drawSampleWithLogDensity() drawing the same and carrying the density at each.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <type_traits>

#include "Distributions/draw_stream.h"

using Emrys::GaussianParams;
using Emrys::Linear;

namespace resamplers = Emrys::resamplers;

namespace
{

// Each config names a Gaussian type and the dimension to test it at.
template <typename Scalar, int Size, int MaxSize, int N>
struct Config
{
    using Type = GaussianParams<Linear<Size, MaxSize>, Scalar>;
    static constexpr Eigen::Index n = N;
};

using Configs = ::testing::Types<
    Config<double, 1, 1, 1>,
    Config<double, 3, 3, 3>,
    Config<double, Eigen::Dynamic, Eigen::Dynamic, 5>,
    Config<double, Eigen::Dynamic, 6, 4>,
    Config<float, 2, 2, 2>>;

template <typename C>
class SamplingTest : public ::testing::Test
{
protected:

    using MG = typename C::Type;
    using Scalar = typename MG::Scalar;
    using VectorType = typename MG::VectorType;
    using MatrixType = typename MG::MatrixType;
    static constexpr Eigen::Index n = C::n;

    // Deterministic symmetric positive definite matrix: A·Aᵀ + I.
    static MatrixType
    makeSpd()
    {
        MatrixType const a = MatrixType::NullaryExpr(n, n, [](Eigen::Index i, Eigen::Index j) {
            return Scalar(std::sin(7.0 * i + 3.0 * j + 1.0));
        });
        MatrixType p = a * a.transpose();
        p.diagonal().array() += Scalar(1);
        return p;
    }

    static MG
    make()
    {
        return MG{
            VectorType::NullaryExpr(
                n,
                [](Eigen::Index i) {
                    return Scalar(1.0 + double(i));
                }
            ),
            makeSpd()
        };
    }
};

}  // namespace

TYPED_TEST_SUITE(SamplingTest, Configs);

// 40,000 draws from each configured Gaussian (fixed, dynamic, bounded, float) have a sample mean
// within five standard errors of the mean per coordinate and a sample covariance within 5% of the
// covariance in norm.
TYPED_TEST(SamplingTest, SampleMomentsConverge)
{
    auto const mg = this->make();
    auto const g = mg.toDensity();
    ASSERT_TRUE(g.has_value());
    auto rng = std::mt19937_64{7};
    constexpr Eigen::Index count = 40000;
    auto const samples = Emrys::drawSamples(*g, count, rng);
    ASSERT_EQ(samples.rows(), this->n);
    ASSERT_EQ(samples.cols(), count);

    Eigen::MatrixXd const x = samples.template cast<double>();
    Eigen::VectorXd const mean = x.rowwise().mean();
    Eigen::MatrixXd const centered = x.colwise() - mean;
    Eigen::MatrixXd const cov = centered * centered.transpose() / double(count - 1);
    Eigen::MatrixXd const p = mg.covarianceLower().template cast<double>();
    for (Eigen::Index i = 0; i < this->n; ++i)
    {
        EXPECT_NEAR(mean(i), double(mg.mean()(i)), 5 * std::sqrt(p(i, i) / double(count))) << "mean " << i;
    }
    EXPECT_LT((cov - p).norm(), 0.05 * p.norm());
}

// drawSamplesWithLogDensity() returns, for each sample, the Gaussian's own log-density at it, to a
// relative tolerance that allows float.
TYPED_TEST(SamplingTest, LogDensityMatchesGaussian)
{
    using Scalar = typename TestFixture::Scalar;
    auto const mg = this->make();
    auto const g = mg.toDensity();
    auto rng = std::mt19937_64{11};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*g, 50, rng);
    ASSERT_EQ(drawn.logDensity.size(), 50);
    Scalar const tolerance = std::is_same_v<Scalar, float> ? Scalar(1e-4) : Scalar(1e-10);
    for (Eigen::Index i = 0; i < 50; ++i)
    {
        Scalar const expected = g->logDensity(drawn.samples.col(i));
        EXPECT_NEAR(drawn.logDensity(i), expected, tolerance * std::abs(expected)) << "sample " << i;
    }
}

// drawSamples() and drawSamplesWithLogDensity() consume the generator identically, so the same seed
// gives the same samples.
TYPED_TEST(SamplingTest, BothVersionsDrawTheSameSamples)
{
    auto const mg = this->make();
    auto const g = mg.toDensity();
    auto a = std::mt19937_64{3};
    auto b = std::mt19937_64{3};
    EXPECT_EQ(Emrys::drawSamples(*g, 20, a), Emrys::drawSamplesWithLogDensity(*g, 20, b).samples);
}

// The samples and the negated log-densities construct a ParticleSet of the same size with those
// log-weights: the importance-sampling use.
TEST(Sampling, FeedsParticleSet)
{
    auto const mg = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    auto rng = std::mt19937_64{5};
    auto const drawn = Emrys::drawSamplesWithLogDensity(mg.toDensity().value(), 10, rng);
    // Importance weights for a uniform target, as a particle filter proposal would use.
    auto const set = Emrys::ParticleSet<Emrys::Linear<2>>{drawn.samples, -drawn.logDensity};
    EXPECT_EQ(set.size(), 10);
    EXPECT_TRUE((set.logWeights() == -drawn.logDensity).all());
}

// Asking for zero samples returns an n-by-0 matrix, not an error.
TEST(Sampling, ZeroSamples)
{
    auto const mg = GaussianParams<Linear<Eigen::Dynamic>>{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    auto rng = std::mt19937_64{5};
    auto const samples = Emrys::drawSamples(mg.toDensity().value(), 0, rng);
    EXPECT_EQ(samples.rows(), 3);
    EXPECT_EQ(samples.cols(), 0);
}

namespace
{

// Three well-separated, narrow 2-D components with unnormalized weights 1, 2, 5.
Emrys::GaussianMixtureParams<Emrys::Linear<2>>
makeSeparatedMixture(double spread)
{
    auto mixture = Emrys::GaussianMixtureParams<Emrys::Linear<2>>{};
    auto const cov = Eigen::Matrix2d{{spread, 0.3 * spread}, {0.3 * spread, 0.5 * spread}};
    mixture.add(std::log(1.0), Eigen::Vector2d{-10.0, 0.0}, cov);
    mixture.add(std::log(2.0), Eigen::Vector2d{0.0, 10.0}, cov);
    mixture.add(std::log(5.0), Eigen::Vector2d{10.0, 0.0}, cov);
    return mixture;
}

}  // namespace

// 40,000 draws from a three-component mixture have a sample mean within five standard errors of the
// mixture's moment-matched mean and a covariance within 5% of its covariance.
TEST(MixtureSampling, SampleMomentsConverge)
{
    auto const mixture = makeSeparatedMixture(4.0);
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());
    auto rng = std::mt19937_64{23};
    constexpr Eigen::Index count = 40000;
    auto const samples = Emrys::drawSamples(*g, count, rng);
    ASSERT_EQ(samples.cols(), count);

    auto const moments = g->moments();
    Eigen::Vector2d const mean = samples.rowwise().mean();
    Eigen::Matrix2Xd const centered = samples.colwise() - mean;
    Eigen::Matrix2d const cov = centered * centered.transpose() / double(count - 1);
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(mean(i), moments.mean()(i), 5 * std::sqrt(moments.covarianceLower()(i, i) / double(count)))
            << "mean " << i;
    }
    EXPECT_LT((cov - moments.covarianceLower()).norm(), 0.05 * moments.covarianceLower().norm());
}

// With a resamplers::Systematic choosing the components and weights 1:2:5, 80 draws fall on the narrow
// components exactly 10, 20, and 50 times.
TEST(MixtureSampling, SystematicGivesEachComponentItsShare)
{
    auto const mixture = makeSeparatedMixture(0.01);
    auto const g = mixture.toDensity();
    auto rng = std::mt19937_64{29};
    auto const samples = Emrys::drawSamples(*g, 80, rng, Emrys::resamplers::systematic());
    auto counts = std::array<int, 3>{};
    for (Eigen::Index j = 0; j < samples.cols(); ++j)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            if ((samples.col(j) - (*g)[Eigen::Index(c)].component.mean()).norm() < 1.0)
            {
                ++counts[c];
            }
        }
    }
    EXPECT_EQ(counts, (std::array<int, 3>{10, 20, 50}));
}

// The mixture's drawSamplesWithLogDensity() draws the same samples as drawSamples() and reports the
// mixture's log-density at each exactly.
TEST(MixtureSampling, LogDensityMatchesMixture)
{
    auto const mixture = makeSeparatedMixture(4.0);
    auto const g = mixture.toDensity();
    auto a = std::mt19937_64{31};
    auto b = std::mt19937_64{31};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*g, 25, a);
    EXPECT_EQ(drawn.samples, Emrys::drawSamples(*g, 25, b));
    for (Eigen::Index j = 0; j < 25; ++j)
    {
        EXPECT_EQ(drawn.logDensity(j), g->logDensity(drawn.samples.col(j))) << "sample " << j;
    }
}

// 40,000 single draws from a Gaussian, one drawSample() at a time, have a sample mean within five
// standard errors of its mean and a sample covariance within 5% of its covariance in norm. Single
// draws with their log-densities are the same draws, each with logDensity() at it to rounding: in a
// linear space it comes from the draw's normals rather than a solve.
TEST(Sampling, StreamOfSingleDrawsMatchesTheGaussian)
{
    auto const params = GaussianParams<Linear<2>>{Eigen::Vector2d{1.0, -2.0}, Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
    auto const gaussian = params.toDensity();
    ASSERT_TRUE(gaussian.has_value());
    auto rng = std::mt19937_64{37};
    Eigen::MatrixXd const draws = EmrysTest::streamOfDraws(*gaussian, 40'000, rng);
    EmrysTest::expectSampleMoments(draws, params.mean(), params.covarianceLower(), 0.05);
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(*gaussian, 50, 38, 1e-12);
}

// 40,000 single draws from a three-component mixture match its moment-matched mean and covariance,
// as SampleMomentsConverge checks of the batch; with their log-densities they are the same draws, each
// with exactly the mixture's logDensity(), which is how it is computed.
TEST(MixtureSampling, StreamOfSingleDrawsMatchesTheMixture)
{
    auto const mixture = makeSeparatedMixture(4.0);
    auto const g = mixture.toDensity();
    ASSERT_TRUE(g.has_value());
    auto rng = std::mt19937_64{41};
    auto const moments = g->moments();
    Eigen::MatrixXd const draws = EmrysTest::streamOfDraws(*g, 40'000, rng);
    EmrysTest::expectSampleMoments(draws, moments.mean(), moments.covarianceLower(), 0.05);
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(*g, 50, 42, 0.0);
}

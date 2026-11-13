#pragma once

// Helpers for the tests of drawSample() and drawSampleWithLogDensity() [allocation]: a stream of single
// draws gathered into the columns of a matrix, a check of their sample moments against a distribution's,
// and a check that the draws with their log-densities are drawSample()'s, each with its density.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <random>

namespace EmrysTest
{

// Draws count samples from distribution one at a time, as code with no heap would, and returns them
// as the columns of a double matrix. drawSample() is found by argument-dependent lookup, so this
// works for any distribution whose header the test includes.
template <typename Distribution, typename G>
[[nodiscard]] Eigen::MatrixXd
streamOfDraws(Distribution const & distribution, Eigen::Index count, G & rng)
{
    auto draws = Eigen::MatrixXd{};
    for (Eigen::Index k = 0; k < count; ++k)
    {
        auto const draw = drawSample(distribution, rng);
        if (k == 0)
        {
            draws.resize(draw.size(), count);
        }
        draws.col(k) = draw.template cast<double>();
    }
    return draws;
}

// The draws' sample mean is within five standard errors of mean in each coordinate, and their
// sample covariance is within the relative tolerance of cov, in norm.
inline void
expectSampleMoments(
    Eigen::MatrixXd const & draws,
    Eigen::VectorXd const & mean,
    Eigen::MatrixXd const & cov,
    double tolerance
)
{
    auto const count = double(draws.cols());
    Eigen::VectorXd const sampleMean = draws.rowwise().mean();
    Eigen::MatrixXd const centered = draws.colwise() - sampleMean;
    Eigen::MatrixXd const sampleCov = centered * centered.transpose() / (count - 1);
    for (Eigen::Index i = 0; i < mean.size(); ++i)
    {
        EXPECT_NEAR(sampleMean(i), mean(i), 5 * std::sqrt(cov(i, i) / count)) << "mean " << i;
    }
    EXPECT_LT((sampleCov - cov).norm(), tolerance * cov.norm()) << sampleCov;
}

// count draws of drawSampleWithLogDensity() from a generator seeded with seed are exactly the draws
// of drawSample() from another seeded alike, so the two consume the generator alike, and each carries
// the distribution's own logDensity() at it to within tolerance. Both functions are found by
// argument-dependent lookup, as in streamOfDraws().
template <typename Distribution>
void
expectSingleDrawsCarryTheirLogDensity(
    Distribution const & distribution,
    int count,
    std::uint64_t seed,
    double tolerance
)
{
    auto plain = std::mt19937_64{seed};
    auto withDensity = std::mt19937_64{seed};
    for (int k = 0; k < count; ++k)
    {
        auto const expected = drawSample(distribution, plain);
        auto const draw = drawSampleWithLogDensity(distribution, withDensity);
        EXPECT_TRUE(draw.sample == expected) << "draw " << k;
        EXPECT_NEAR(double(draw.logDensity), double(distribution.logDensity(draw.sample)), tolerance) << "draw " << k;
    }
}

}  // namespace EmrysTest

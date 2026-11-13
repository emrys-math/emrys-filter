#pragma once

// The ranging problem of [benchmarks] for a particle filter: the anchors, motion, and simulator of
// ../ranging/scenario.h, with wider range noise and a tighter initial estimate, so that a
// bootstrap filter with a thousand particles keeps tens of them within the likelihood of the
// first ranges, where the EKF's figures (0.3 m ranges, a 5 m prior) would leave it one or two,
// and with a longer step and more process noise, so that resampling does not exhaust the cloud;
// and the initial particles, drawn in plain Eigen so that both filters start from the same set.

#include "../ranging/scenario.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <random>

namespace Particles
{

template <int Dimension, int AnchorCount>
using Scenario = Ranging::Scenario<Dimension, AnchorCount>;

/// How the EmrysFilter filter gets the likelihood of the ranges: from the density of the
/// measurement's additive Gaussian noise, which the library evaluates by factorizing R once and
/// solving against it at each particle; from the measurement's own closed form for independent
/// ranges, which is what the plain-Eigen filter computes; or from the noise's density at the ranges
/// of a batched function, which computes every particle's in one call (decisions.md
/// [batched-functions]).
enum class Likelihood
{
    Noise,
    Own,
    Batched
};

/// The particle filter's problem: the ranging scenario with 2 m range noise, a 3 m initial
/// position uncertainty, ranges every second, and an acceleration density of 0.5 m²/s³. The last
/// two keep the bootstrap filter alive: its only source of diversity after a resample is the
/// process noise, which under the EKF's 0.1 s step and 0.1 m²/s³ moves a particle 6 mm per step,
/// so after a few resamples every particle descends from one ancestor and the cloud collapses.
/// Here the noise moves a particle 0.4 m per step, as far as the ranges locate it.
template <int Dimension, int AnchorCount>
[[nodiscard]] Scenario<Dimension, AnchorCount>
makeScenario()
{
    using Block = Eigen::Matrix<double, Dimension, Dimension>;
    auto scenario = Ranging::makeScenario<Dimension, AnchorCount>();
    scenario.rangeStdDev = 2.0;
    scenario.timeStep = 1.0;
    scenario.accelDensity = 0.5;
    scenario.initialCov.template topLeftCorner<Dimension, Dimension>() = Block{9.0 * Block::Identity()};
    return scenario;
}

/// The initial particles: \a count draws from \a scenario's initial estimate, one per column, with
/// the Cholesky factor of its covariance and standard normals from a generator seeded with \a seed.
template <int Dimension, int AnchorCount>
[[nodiscard]] Eigen::Matrix<double, Scenario<Dimension, AnchorCount>::StateSize, Eigen::Dynamic>
drawInitialParticles(Scenario<Dimension, AnchorCount> const & scenario, Eigen::Index count, unsigned seed)
{
    using S = Scenario<Dimension, AnchorCount>;
    auto engine = std::mt19937{seed};
    auto normal = std::normal_distribution<double>{};
    typename S::StateMatrix const factor = scenario.initialCov.llt().matrixL();
    auto particles = Eigen::Matrix<double, S::StateSize, Eigen::Dynamic>{S::StateSize, count};
    auto z = typename S::StateVector{};
    for (Eigen::Index i = 0; i < count; ++i)
    {
        for (auto & coefficient : z)
        {
            coefficient = normal(engine);
        }
        particles.col(i) = scenario.initialMean + factor * z;
    }
    return particles;
}

}  // namespace Particles

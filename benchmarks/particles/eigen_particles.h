#pragma once

// The ranging problem solved from scratch in plain Eigen, as a careful user would write the
// bootstrap particle filter [benchmarks]: the particles in the columns of one fixed-row matrix, their log
// weights in an array, a loop that moves each particle and adds a draw of the process noise, a
// loop that weights each by the closed-form likelihood of independent Gaussian ranges, the
// weights normalized in log space, and systematic resampling into scratch that is kept between
// steps, when the effective sample size falls below half.
//
// Two choices follow the library's so that the two filters, seeded alike, move the same particles
// the same way and agree particle by particle to rounding: the factor of the process noise is the
// library's default root (internal::defaultRoot(), the Cholesky factor, or a pivoted LDLT factor
// where no Cholesky factor exists), and the normals and the resampling offset are drawn in the library's order
// from the same kind of generator. Compare emrys_particles.h.

#include "scenario.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <vector>

namespace Particles
{

/// The ranging bootstrap filter in plain Eigen.
template <int Dimension, int AnchorCount>
class EigenParticles
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    using StateVector = typename ScenarioType::StateVector;
    using StateMatrix = typename ScenarioType::StateMatrix;
    using Ranges = typename ScenarioType::Ranges;
    using States = Eigen::Matrix<double, ScenarioType::StateSize, Eigen::Dynamic>;
    using Weights = Eigen::Array<double, Eigen::Dynamic, 1>;

    /// A filter on \a scenario starting from the particles \a initial, one per column, drawing
    /// from a generator seeded with \a seed.
    EigenParticles(ScenarioType const & scenario, States const & initial, unsigned seed)
        : m_anchors{scenario.anchors}
        , m_rangeVariance{scenario.rangeStdDev * scenario.rangeStdDev}
        , m_logNormalizer{-0.5 * AnchorCount * std::log(2 * std::numbers::pi * m_rangeVariance)}
        , m_timeStep{scenario.timeStep}
        , m_noiseRoot{noiseRoot(processNoiseOver(scenario.timeStep, scenario.accelDensity))}
        , m_states{initial}
        , m_logWeights{Weights::Constant(initial.cols(), -std::log(double(initial.cols())))}
        , m_scratch{StateSize, initial.cols()}
        , m_weights{initial.cols()}
        , m_ancestors(std::size_t(initial.cols()))
        , m_rng{seed}
    {}

    /// Moves every particle forward by the scenario's time step: x = F x + L z with z standard
    /// normal and L L^T = Q, with L factored once at construction, since the step is fixed.
    void
    predict()
    {
        auto normal = std::normal_distribution<double>{};
        auto z = StateVector{};
        for (Eigen::Index i = 0; i < m_states.cols(); ++i)
        {
            for (double & coefficient : z)
            {
                coefficient = normal(m_rng);
            }
            auto particle = m_states.col(i);
            particle.template head<Dimension>() += m_timeStep * particle.template tail<Dimension>();
            particle.noalias() += m_noiseRoot * z;
        }
    }

    /// Weights every particle by the likelihood of the ranges \a z, one per anchor, and
    /// normalizes the weights.
    void
    update(Ranges const & z)
    {
        for (Eigen::Index i = 0; i < m_states.cols(); ++i)
        {
            PositionVector const p = m_states.col(i).template head<Dimension>();
            Ranges const residual = z - (m_anchors.colwise() - p).colwise().norm().transpose();
            m_logWeights(i) += m_logNormalizer - 0.5 * residual.squaredNorm() / m_rangeVariance;
        }
        double const peak = m_logWeights.maxCoeff();
        m_logWeights -= peak + std::log((m_logWeights - peak).exp().sum());
    }

    /// Resamples systematically when the effective sample size is below half the particle count:
    /// the positions (j + u) / N for one uniform u, walked through the cumulative weights, and the
    /// chosen particles gathered into the scratch, which then becomes the set.
    void
    housekeep()
    {
        Eigen::Index const n = m_states.cols();
        m_weights = m_logWeights.exp();
        double const ess = 1.0 / m_weights.square().sum();
        if (ess >= 0.5 * double(n))
        {
            return;
        }

        // The total and the last positive weight in one pass, so that the walk never runs onto a
        // particle whose weight underflowed to zero.
        double total = 0.0;
        Eigen::Index lastPositive = 0;
        for (Eigen::Index i = 0; i < n; ++i)
        {
            total += m_weights(i);
            if (m_weights(i) > 0.0)
            {
                lastPositive = i;
            }
        }
        double const offset = std::min(std::uniform_real_distribution<double>{}(m_rng), std::nextafter(1.0, 0.0));
        Eigen::Index i = 0;
        double cumulative = m_weights(0);
        for (Eigen::Index j = 0; j < n; ++j)
        {
            double const position = (double(j) + offset) / double(n) * total;
            while (i < lastPositive && position >= cumulative)
            {
                cumulative += m_weights(++i);
            }
            m_ancestors[std::size_t(j)] = i;
        }
        for (Eigen::Index j = 0; j < n; ++j)
        {
            m_scratch.col(j) = m_states.col(m_ancestors[std::size_t(j)]);
        }
        m_states.swap(m_scratch);
        m_logWeights.setConstant(-std::log(double(n)));
    }

    /// The particles, one per column.
    [[nodiscard]] States const &
    states() const
    {
        return m_states;
    }

    /// The particles' log weights, normalized after every update.
    [[nodiscard]] Weights const &
    logWeights() const
    {
        return m_logWeights;
    }

    /// The weighted mean of the particles.
    [[nodiscard]] StateVector
    mean() const
    {
        Weights const w = m_logWeights.exp();
        return m_states * w.matrix();
    }

    /// The weighted covariance of the particles about their mean.
    [[nodiscard]] StateMatrix
    covariance() const
    {
        Weights const w = m_logWeights.exp();
        StateVector const m = m_states * w.matrix();
        States const centered = m_states.colwise() - m;
        return centered * w.matrix().asDiagonal() * centered.transpose();
    }

private:

    static constexpr int StateSize = ScenarioType::StateSize;
    using PositionVector = Eigen::Matrix<double, Dimension, 1>;
    using PositionMatrix = Eigen::Matrix<double, Dimension, Dimension>;

    /// The process noise Q over the step \a dt, the exact discretization of white-noise
    /// acceleration of spectral density \a accelDensity.
    [[nodiscard]] static StateMatrix
    processNoiseOver(double dt, double accelDensity)
    {
        PositionMatrix const identity = PositionMatrix::Identity();
        auto processNoise = StateMatrix{};
        processNoise << dt * dt * dt / 3 * identity, dt * dt / 2 * identity, dt * dt / 2 * identity, dt * identity;
        return accelDensity * processNoise;
    }

    /// A factor L with L L^T = \a a for a positive semidefinite \a a, as the library's
    /// internal::defaultRoot() computes it: the Cholesky factor, or where it fails P^T L D^{1/2}
    /// from a pivoted LDLT, with a negative pivot treated as zero.
    [[nodiscard]] static StateMatrix
    noiseRoot(StateMatrix const & a)
    {
        auto const llt = Eigen::LLT<StateMatrix>{a};
        if (llt.info() == Eigen::Success)
        {
            return llt.matrixL();
        }
        auto const ldlt = Eigen::LDLT<StateMatrix>{a};
        StateVector const d = ldlt.vectorD().cwiseMax(0.0).cwiseSqrt();
        StateMatrix lower = ldlt.matrixL();
        lower *= d.asDiagonal();
        return ldlt.transpositionsP().transpose() * lower;
    }

    typename ScenarioType::Anchors m_anchors;  ///< one anchor per column
    double m_rangeVariance;                    ///< the variance of each range
    double m_logNormalizer;                    ///< the log of the ranges' density's normalizer
    double m_timeStep;                         ///< the fixed time between measurements
    StateMatrix m_noiseRoot;                   ///< L with L L^T = Q over the step
    States m_states;                           ///< the particles, one per column
    Weights m_logWeights;                      ///< their log weights
    States m_scratch;                          ///< the resampled particles, swapped in
    Weights m_weights;                         ///< the linear weights, for the sample size and the walk
    std::vector<Eigen::Index> m_ancestors;     ///< the ancestor of each resampled particle
    std::mt19937_64 m_rng;                     ///< draws the noise and the resampling offset
};

}  // namespace Particles

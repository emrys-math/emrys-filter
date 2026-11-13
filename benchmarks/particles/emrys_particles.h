#pragma once

// The ranging problem solved with EmrysFilter's particle filter: the bootstrap filter over a tagged
// position-velocity state, moved by the constant-velocity model with a draw of its noise, weighted
// by the likelihood of every anchor's range, and resampled systematically when the effective
// sample size falls below half the particle count. Compare eigen_particles.h, the same filter
// written from scratch.

#include "scenario.h"

#include <cmath>
#include <functional>
#include <numbers>
#include <random>

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

namespace Particles
{

/// The state's blocks, named so that the model and the measurement address them by tag rather
/// than by index.
template <int Dimension>
struct Position : Emrys::Linear<Dimension>
{};

template <int Dimension>
struct Velocity : Emrys::Linear<Dimension>
{};

/// The ranging bootstrap filter with EmrysFilter, weighting the particles as \a Kind says.
template <int Dimension, int AnchorCount, Likelihood Kind>
class EmrysParticles
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    using State = Emrys::Composite<Position<Dimension>, Velocity<Dimension>>;
    using Ranges = typename ScenarioType::Ranges;
    using States = Eigen::Matrix<double, ScenarioType::StateSize, Eigen::Dynamic>;
    using ParticlesType = Emrys::ParticleSet<State>;

    /// A filter on \a scenario starting from the particles \a initial, one per column, drawing
    /// from a generator seeded with \a seed.
    EmrysParticles(ScenarioType const & scenario, States const & initial, unsigned seed)
        : m_anchors{scenario.anchors}
        , m_rangeVariance{scenario.rangeStdDev * scenario.rangeStdDev}
        , m_logNormalizer{-0.5 * AnchorCount * std::log(2 * std::numbers::pi * m_rangeVariance)}
        , m_rangeNoise{Ranges::Zero(), RangeMatrix{m_rangeVariance * RangeMatrix::Identity()}}
        , m_filter{
              Model{
                  Emrys::ConstantVelocity<Position<Dimension>, Velocity<Dimension>>{scenario.accelDensity},
                  State{},
                  scenario.timeStep
              },
              makeParticles(initial)
          }
        , m_rng{seed}
    {}

    /// Moves every particle forward by the scenario's time step through the model and a draw of
    /// its noise. The step is fixed, so the model holds the noise's factor from construction.
    void
    predict()
    {
        m_filter.predict(m_rng);
    }

    /// Weights every particle by the likelihood of the ranges \a z, one per anchor.
    void
    update(Ranges const & z)
    {
        // The ranges h(p), generic in their scalar. The model refers to the anchors and the noise,
        // which outlive it.
        auto const ranges = [&]<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, Dimension, 1> const position = Emrys::block<Position<Dimension>>(in, x);
            return Eigen::Matrix<T, AnchorCount, 1>{
                (m_anchors.template cast<T>().colwise() - position).colwise().norm().transpose()
            };
        };
        auto const reads = Emrys::reads(Emrys::path<Position<Dimension>>);
        auto const noise = Emrys::additiveNoise(std::cref(m_rangeNoise));
        if constexpr (Kind == Likelihood::Batched)
        {
            // Every particle's ranges in one call, a row per anchor, the particles the columns of X;
            // a batched function writes into the output Y the library gives it, so the model names
            // its output space.
            auto const batchedRanges = Emrys::batched([&](auto const & in, auto const &, auto const & x, auto & y) {
                auto const positions = Emrys::block<Position<Dimension>>(in, x);
                for (Eigen::Index a = 0; a < AnchorCount; ++a)
                {
                    y.values().row(a) = (positions.colwise() - m_anchors.col(a)).colwise().norm();
                }
            });
            auto const model =
                Emrys::measurementModel(batchedRanges, reads, Emrys::outputSpace(Emrys::Linear<AnchorCount>{}), noise);
            static_cast<void>(m_filter.update(Emrys::observe(model, z)));
        }
        else if constexpr (Kind == Likelihood::Noise)
        {
            // The filter evaluates the noise's density at the residual z - h(p).
            auto const model = Emrys::measurementModel(ranges, reads, noise);
            static_cast<void>(m_filter.update(Emrys::observe(model, z)));
        }
        else
        {
            // The model supplies its own likelihood: the ranges are independent and Gaussian, so
            // log p(z | p) = -(A/2) log(2 pi sigma^2) - |z - h(p)|^2 / (2 sigma^2), the closed form
            // the plain-Eigen filter computes. The filter weights by it and leaves the noise, which
            // a Kalman update would use, alone.
            auto const own = Emrys::logLikelihood([&](auto const & in, auto const & x, Ranges const & observed) {
                Eigen::Matrix<double, Dimension, 1> const position = Emrys::block<Position<Dimension>>(in, x);
                Ranges const residual = observed - (m_anchors.colwise() - position).colwise().norm().transpose();
                return m_logNormalizer - 0.5 * residual.squaredNorm() / m_rangeVariance;
            });
            auto const model = Emrys::measurementModel(ranges, reads, noise, own);
            static_cast<void>(m_filter.update(Emrys::observe(model, z)));
        }
    }

    /// Resamples when the effective sample size is below half the particle count.
    void
    housekeep()
    {
        static_cast<void>(Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(m_rng))(m_filter.state()));
    }

    /// The current particles.
    [[nodiscard]] ParticlesType const &
    state() const
    {
        return m_filter.state();
    }

private:

    using RangeMatrix = Eigen::Matrix<double, AnchorCount, AnchorCount>;
    using Motion = Emrys::ConstantVelocity<Position<Dimension>, Velocity<Dimension>>;
    using Model = Emrys::FixedStepModel<State, Motion>;
    using Filter = Emrys::ParticleFilter<State, Model>;

    /// The particle set from \a initial, with its capacity set so that the steps allocate nothing.
    [[nodiscard]] static ParticlesType
    makeParticles(States const & initial)
    {
        auto particles = ParticlesType{State{}, initial};
        particles.setCapacity(initial.cols());
        return particles;
    }

    typename ScenarioType::Anchors m_anchors;                        ///< one anchor per column
    double m_rangeVariance;                                          ///< the variance of each range
    double m_logNormalizer;                                          ///< the log of the ranges' density's normalizer
    Emrys::GaussianParams<Emrys::Linear<AnchorCount>> m_rangeNoise;  ///< independent noise on each range
    Filter m_filter;                                                 ///< the filter
    std::mt19937_64 m_rng;                                           ///< draws the noise and the resampling offset
};

}  // namespace Particles

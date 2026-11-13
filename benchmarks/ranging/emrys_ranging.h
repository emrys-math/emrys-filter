#pragma once

// The ranging problem solved with EmrysFilter: an extended Kalman filter over a tagged
// position-velocity state, with a constant-velocity process model and one measurement of every
// anchor's range per step, in double or in float. Compare eigen_ranging.h, the same filter written
// from scratch.

#include "scenario.h"

#include <cstdlib>
#include <functional>

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

namespace Ranging
{

/// The state's blocks, named so that models and measurements address them by tag rather than by
/// index.
template <int Dimension>
struct Position : Emrys::Linear<Dimension>
{};

template <int Dimension>
struct Velocity : Emrys::Linear<Dimension>
{};

/// The ranging EKF with EmrysFilter in \a Scalar, linearizing the ranges as \a Kind says.
template <int Dimension, int AnchorCount, Jacobian Kind, typename Scalar = double>
class EmrysRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    using State = Emrys::Composite<Position<Dimension>, Velocity<Dimension>>;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;

    explicit EmrysRanging(ScenarioType const & scenario)
        : m_anchors{scenario.anchors.template cast<Scalar>()}
        , m_rangeNoise{Ranges::Zero(), RangeMatrix{Scalar(scenario.rangeStdDev * scenario.rangeStdDev) * RangeMatrix::Identity()}}
        , m_filter{
              StepModel{Model{Scalar(scenario.accelDensity)}, State{}, Scalar(scenario.timeStep)},
              Emrys::GaussianParams<State, Scalar>{
                  State{},
                  StateVector{scenario.initialMean.template cast<Scalar>()},
                  StateMatrix{scenario.initialCov.template cast<Scalar>()}
              }
          }
    {}

    /// Moves the estimate forward by the scenario's time step. The constant-velocity block is
    /// affine, so the fixed-step model holds its transition F and noise Q from construction, and
    /// the prediction is F P F^T + Q without linearizing.
    void
    predict()
    {
        m_filter.predict();
    }

    /// Conditions the estimate on the ranges \a z, one per anchor, and stops the program if the
    /// innovation covariance is not positive definite.
    void
    update(Ranges const & z)
    {
        // The ranges h(p), generic in their scalar so that they can be differentiated. They read
        // only the position block, so only position is differentiated.
        auto const ranges = [&]<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, Dimension, 1> const p = Emrys::block<Position<Dimension>>(in, x);
            Eigen::Matrix<T, AnchorCount, 1> r =
                (m_anchors.template cast<T>().colwise() - p).colwise().norm().transpose();
            return r;
        };
        if constexpr (Kind == Jacobian::Analytic)
        {
            // The ranges with their Jacobian, computed together: each row is the unit direction from
            // its anchor to p.
            auto const withJacobian =
                [&](auto const & in, auto const &, Eigen::Matrix<Scalar, Dimension, 1> const & x) {
                    Eigen::Matrix<Scalar, Dimension, 1> const p = Emrys::block<Position<Dimension>>(in, x);
                    Eigen::Matrix<Scalar, Dimension, AnchorCount> const toAnchor = m_anchors.colwise() - p;
                    Ranges const distances = toAnchor.colwise().norm().transpose();
                    return Emrys::withJacobian(
                        distances,
                        (-(toAnchor.array().rowwise() / distances.transpose().array())).matrix().transpose()
                    );
                };
            if (!m_filter.update(Emrys::observe(Emrys::methods::linearized(), model(withJacobian), z)))
            {
                std::abort();
            }
        }
        else if (!m_filter.update(Emrys::observe(Emrys::methods::linearized(), model(ranges), z)))
        {
            std::abort();
        }
    }

    /// The current estimate, position then velocity.
    [[nodiscard]] Emrys::GaussianParams<State, Scalar> const &
    state() const
    {
        return m_filter.state();
    }

private:

    using RangeMatrix = Eigen::Matrix<Scalar, AnchorCount, AnchorCount>;
    using StateVector = Eigen::Matrix<Scalar, ScenarioType::StateSize, 1>;
    using StateMatrix = Eigen::Matrix<Scalar, ScenarioType::StateSize, ScenarioType::StateSize>;
    using Model = Emrys::ConstantVelocity<Position<Dimension>, Velocity<Dimension>, Scalar>;
    using StepModel = Emrys::FixedStepModel<State, Model, Emrys::methods::Linearized, Scalar>;
    using Filter = Emrys::KalmanFilter<Emrys::Gaussian<State, Scalar>, StepModel>;

    /// The measurement model of the position by the function \a function. The noise outlives the
    /// model, so the model refers to it rather than copying its covariance.
    template <typename Function>
    [[nodiscard]] auto
    model(Function const & function) const
    {
        return Emrys::measurementModel(
            function,
            Emrys::reads(Emrys::path<Position<Dimension>>),
            Emrys::additiveNoise(std::cref(m_rangeNoise))
        );
    }

    Eigen::Matrix<Scalar, Dimension, AnchorCount> m_anchors;                 ///< one anchor per column
    Emrys::GaussianParams<Emrys::Linear<AnchorCount>, Scalar> m_rangeNoise;  ///< independent noise on each range
    Filter m_filter;                                                         ///< the filter
};

}  // namespace Ranging

#pragma once

// The ranging problem solved with EmrysFilter by each of the four filters mherb/kalman has, so
// that mherb_ranging.h has a counterpart for each: the extended and the unscented Kalman filter,
// each over a covariance (KalmanFilter) or a square root of it (SqrtKalmanFilter). The prediction
// is the fixed-step constant-velocity model of emrys_ranging.h in every kind, which is exact for
// this linear model as sigma points through it are. The extended kinds update by the ranges'
// analytic Jacobian over the position, as emrys_ranging.h does. The unscented kinds update by
// methods::unscented() at alpha = 1, beta = 2, kappa = 0, the defaults of both libraries, with a
// measurement model that reads the whole state, so that the sigma points spread over the whole
// state as mherb/kalman's do; one that read only the position would spread fewer points with
// other weights. Compare mherb_ranging.h.

#include "emrys_ranging.h"
#include "scenario.h"

#include <cstdlib>
#include <functional>
#include <type_traits>

#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

namespace Ranging
{

/// The ranging filter \a Kind with EmrysFilter in \a Scalar.
template <int Dimension, int AnchorCount, FilterKind Kind, typename Scalar = double>
class EmrysRangingKind
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    using State = Emrys::Composite<Position<Dimension>, Velocity<Dimension>>;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;
    using StateVector = Eigen::Matrix<Scalar, ScenarioType::StateSize, 1>;
    using StateMatrix = Eigen::Matrix<Scalar, ScenarioType::StateSize, ScenarioType::StateSize>;

    explicit EmrysRangingKind(ScenarioType const & scenario)
        : m_anchors{scenario.anchors.template cast<Scalar>()}
        , m_rangeNoise{Ranges::Zero(), RangeMatrix{Scalar(scenario.rangeStdDev * scenario.rangeStdDev) * RangeMatrix::Identity()}}
        , m_filter{makeFilter(scenario)}
    {}

    /// Moves the estimate forward by the scenario's time step. The square-root filter's prediction
    /// fails only for sigma points with a negative center weight, which a linearized prediction
    /// has none of, so a failure stops the program.
    void
    predict()
    {
        if constexpr (isSqrt(Kind))
        {
            if (!m_filter.predict())
            {
                std::abort();
            }
        }
        else
        {
            m_filter.predict();
        }
    }

    /// Conditions the estimate on the ranges \a z, one per anchor, and stops the program if the
    /// update fails.
    void
    update(Ranges const & z)
    {
        if constexpr (isUnscented(Kind))
        {
            // The ranges h(x) of the whole state, which only the position enters.
            auto const ranges = [&](auto const & in, auto const &, StateVector const & x) {
                Eigen::Matrix<Scalar, Dimension, 1> const p = Emrys::block<Position<Dimension>>(in, x);
                Ranges r = (m_anchors.colwise() - p).colwise().norm().transpose();
                return r;
            };
            auto const model = Emrys::measurementModel(ranges, Emrys::additiveNoise(std::cref(m_rangeNoise)));
            if (!m_filter.update(Emrys::observe(Emrys::methods::unscented(), model, z)))
            {
                std::abort();
            }
        }
        else
        {
            // The ranges with their Jacobian, computed together: each row is the unit direction from
            // its anchor to p.
            auto const withJacobian = Emrys::writesJacobians(
                [&](auto const & in, auto const &, Eigen::Matrix<Scalar, Dimension, 1> const & x, auto & output) {
                    Eigen::Matrix<Scalar, Dimension, 1> const p = Emrys::block<Position<Dimension>>(in, x);
                    Eigen::Matrix<Scalar, Dimension, AnchorCount> const toAnchor = m_anchors.colwise() - p;
                    Ranges const distances = toAnchor.colwise().norm().transpose();
                    Emrys::value(output) = distances;
                    if (Emrys::wantsJacobians(output))
                    {
                        Emrys::jacobian(output) =
                            (-(toAnchor.array().rowwise() / distances.transpose().array())).matrix().transpose();
                    }
                }
            );
            auto const model = Emrys::measurementModel(
                withJacobian,
                Emrys::reads(Emrys::path<Position<Dimension>>),
                Emrys::additiveNoise(std::cref(m_rangeNoise))
            );
            if (!m_filter.update(Emrys::observe(Emrys::methods::linearized(), model, z)))
            {
                std::abort();
            }
        }
    }

    /// The current mean, position then velocity.
    [[nodiscard]] StateVector const &
    mean() const
    {
        return m_filter.state().mean();
    }

    /// The current covariance, whose lower triangle the library guarantees; formed from the root in
    /// the square-root kinds.
    [[nodiscard]] StateMatrix
    covarianceLower() const
    {
        if constexpr (isSqrt(Kind))
        {
            return m_filter.state().covariance().value();
        }
        else
        {
            return m_filter.state().covarianceLower();
        }
    }

private:

    using RangeMatrix = Eigen::Matrix<Scalar, AnchorCount, AnchorCount>;
    using Model = Emrys::ConstantVelocity<Position<Dimension>, Velocity<Dimension>, Scalar>;
    using StepModel = Emrys::FixedStepModel<State, Model, Emrys::methods::Linearized, Scalar>;
    using Filter = std::conditional_t<
        isSqrt(Kind),
        Emrys::SqrtKalmanFilter<State, StepModel, Emrys::methods::Linearized, Scalar>,
        Emrys::KalmanFilter<Emrys::Gaussian<State, Scalar>, StepModel>>;

    /// The filter at the scenario's initial estimate, in the form \a Kind keeps.
    [[nodiscard]] static Filter
    makeFilter(ScenarioType const & scenario)
    {
        auto step = StepModel{Model{Scalar(scenario.accelDensity)}, State{}, Scalar(scenario.timeStep)};
        auto initial = Emrys::GaussianParams<State, Scalar>{
            State{},
            StateVector{scenario.initialMean.template cast<Scalar>()},
            StateMatrix{scenario.initialCov.template cast<Scalar>()}
        };
        if constexpr (isSqrt(Kind))
        {
            // The initial covariance is diagonal and positive, so its root exists.
            return Filter{std::move(step), Emrys::toSqrt(initial).value()};
        }
        else
        {
            return Filter{std::move(step), std::move(initial)};
        }
    }

    Eigen::Matrix<Scalar, Dimension, AnchorCount> m_anchors;                 ///< one anchor per column
    Emrys::GaussianParams<Emrys::Linear<AnchorCount>, Scalar> m_rangeNoise;  ///< independent noise on each range
    Filter m_filter;                                                         ///< the filter
};

}  // namespace Ranging

#pragma once

// The ranging problem solved with mherb/kalman (https://github.com/mherb/kalman), the third
// implementation of [benchmarks]: its ExtendedKalmanFilter with a system model and a measurement
// model written as its own examples write them. Unlike eigen_ranging.h, it keeps the library's own
// arithmetic, which is what its users run: the prediction F P F^T + W Q W^T with W the identity,
// the gain P H^T S^-1 through an explicit inverse of S, and the covariance update P - K H P, with
// no triangle mirrored afterward. Its models compute the Jacobians in updateJacobians() and the
// ranges again in h(), since the filter calls them separately. It has no automatic
// differentiation, so its Jacobian is analytic only. Compare emrys_ranging.h.

#include "scenario.h"

#include <Eigen/Core>

#include <vector>

#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/LinearizedMeasurementModel.hpp>
#include <kalman/LinearizedSystemModel.hpp>

namespace Ranging
{

/// The ranging EKF with mherb/kalman in \a Scalar, with the analytic Jacobian.
template <int Dimension, int AnchorCount, typename Scalar = double>
class MherbRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    static constexpr int StateSize = ScenarioType::StateSize;
    using State = Kalman::Vector<Scalar, StateSize>;
    using Ranges = Kalman::Vector<Scalar, AnchorCount>;

    explicit MherbRanging(ScenarioType const & scenario)
        : m_system{Scalar(scenario.timeStep), Scalar(scenario.accelDensity)}
        , m_measurement{scenario.anchors.template cast<Scalar>(), Scalar(scenario.rangeStdDev)}
    {
        m_filter.init(State{scenario.initialMean.template cast<Scalar>()});
        m_filter.setCovariance(Kalman::Covariance<State>{scenario.initialCov.template cast<Scalar>()});
    }

    /// Moves the estimate forward by the scenario's time step.
    void
    predict()
    {
        m_filter.predict(m_system);
    }

    /// Conditions the estimate on the ranges \a z, one per anchor.
    void
    update(Ranges const & z)
    {
        m_filter.update(m_measurement, z);
    }

    /// The ranges \a ranges of a run as this filter takes them, in mherb/kalman's vector type,
    /// converted before any filter runs so that no conversion is timed.
    [[nodiscard]] static std::vector<Ranges>
    rangesOf(std::vector<Eigen::Matrix<Scalar, AnchorCount, 1>> const & ranges)
    {
        auto result = std::vector<Ranges>{};
        result.reserve(ranges.size());
        for (auto const & z : ranges)
        {
            result.push_back(Ranges{z});
        }
        return result;
    }

    /// The current mean, position then velocity.
    [[nodiscard]] State const &
    mean() const
    {
        return m_filter.getState();
    }

    /// The current covariance, whose two triangles can differ by rounding.
    [[nodiscard]] Kalman::Covariance<State> const &
    covariance() const
    {
        return m_filter.getCovariance();
    }

private:

    using Control = Kalman::Vector<Scalar, 1>;

    /// The constant-velocity model. Its step is fixed, so F, W, and Q are set once here, and
    /// updateJacobians(), which the filter calls before every prediction, has nothing to do.
    class System : public Kalman::LinearizedSystemModel<State, Control>
    {
    public:

        System(Scalar dt, Scalar accelDensity)
            : m_timeStep{dt}
        {
            using PositionMatrix = Eigen::Matrix<Scalar, Dimension, Dimension>;
            PositionMatrix const identity = PositionMatrix::Identity();
            this->F.setIdentity();
            this->F.template topRightCorner<Dimension, Dimension>() = dt * identity;
            this->W.setIdentity();
            auto processNoise = Kalman::Covariance<State>{};
            processNoise << dt * dt * dt / 3 * identity, dt * dt / 2 * identity, dt * dt / 2 * identity, dt * identity;
            this->setCovariance(accelDensity * processNoise);
        }

        [[nodiscard]] State
        f(State const & x, Control const &) const override
        {
            State next = x;
            next.template head<Dimension>() += m_timeStep * x.template tail<Dimension>();
            return next;
        }

        void
        updateJacobians(State const &, Control const &) override
        {}

    private:

        Scalar m_timeStep;  ///< the fixed time between measurements
    };

    /// The ranges to the anchors, with their Jacobian in position.
    class Measurement : public Kalman::LinearizedMeasurementModel<State, Ranges>
    {
    public:

        Measurement(Eigen::Matrix<Scalar, Dimension, AnchorCount> const & anchors, Scalar rangeStdDev)
            : m_anchors{anchors}
        {
            this->H.setZero();
            this->V.setIdentity();
            this->setCovariance(rangeStdDev * rangeStdDev * Kalman::Covariance<Ranges>::Identity());
        }

        [[nodiscard]] Ranges
        h(State const & x) const override
        {
            return Ranges{(m_anchors.colwise() - x.template head<Dimension>()).colwise().norm().transpose()};
        }

        /// The Jacobian in position, each anchor's unit line of sight to the position, negated; the
        /// velocity columns stay zero.
        void
        updateJacobians(State const & x) override
        {
            Eigen::Matrix<Scalar, Dimension, AnchorCount> const toAnchor =
                m_anchors.colwise() - x.template head<Dimension>();
            Eigen::Matrix<Scalar, 1, AnchorCount> const ranges = toAnchor.colwise().norm();
            this->H.template leftCols<Dimension>() =
                -(toAnchor.array().rowwise() / ranges.array()).matrix().transpose();
        }

    private:

        Eigen::Matrix<Scalar, Dimension, AnchorCount> m_anchors;  ///< one anchor per column
    };

    System m_system;                                 ///< the process model
    Measurement m_measurement;                       ///< the measurement model
    Kalman::ExtendedKalmanFilter<State> m_filter{};  ///< the filter
};

}  // namespace Ranging

#pragma once

// The ranging problem solved with mherb/kalman (https://github.com/mherb/kalman), the third
// implementation of [benchmarks]: each of its four filters, ExtendedKalmanFilter,
// UnscentedKalmanFilter, SquareRootExtendedKalmanFilter, and SquareRootUnscentedKalmanFilter, with
// a system model and a measurement model written as its own examples write them. Unlike
// eigen_ranging.h, it keeps the library's own arithmetic, which is what its users run:
//
// - the EKF predicts F P F^T + W Q W^T with W the identity, forms the gain P H^T S^-1 through an
//   explicit inverse of S, and updates the covariance by P - K H P, with no triangle mirrored;
// - the square-root EKF predicts the factor by a QR decomposition, and updates by forming
//   P - K H P from the factor and factorizing it again;
// - both UKFs spread 2n + 1 sigma points over the whole state with alpha = 1, beta = 2, kappa = 0,
//   the defaults of both libraries, and update with the points the prediction moved, drawn
//   before the process noise was added, rather than with points drawn from the predicted
//   covariance; the covariance form inverts S explicitly, and the square-root form downdates its
//   factor one column of K S_y at a time.
//
// The models compute the Jacobians in updateJacobians() and the ranges again in h(), since the
// filter calls them separately; the UKFs never call updateJacobians(). There is no automatic
// differentiation, so the Jacobian is analytic only. Compare emrys_ranging.h and
// emrys_ranging_kinds.h.

#include "scenario.h"

#include <Eigen/Core>

#include <type_traits>
#include <vector>

#include <kalman/ExtendedKalmanFilter.hpp>
#include <kalman/LinearizedMeasurementModel.hpp>
#include <kalman/LinearizedSystemModel.hpp>
#include <kalman/SquareRootExtendedKalmanFilter.hpp>
#include <kalman/SquareRootUnscentedKalmanFilter.hpp>
#include <kalman/UnscentedKalmanFilter.hpp>

namespace Ranging
{

/// The constant-velocity model in \a Dimension dimensions as a mherb/kalman system model over
/// \a Base, Kalman::StandardBase or Kalman::SquareRootBase. Its step is fixed, so F, W, and Q are
/// set once in the constructor, and updateJacobians(), which the extended filters call before
/// every prediction, has nothing to do.
template <int Dimension, typename Scalar, template <class> class Base>
class MherbConstantVelocity
    : public Kalman::LinearizedSystemModel<Kalman::Vector<Scalar, 2 * Dimension>, Kalman::Vector<Scalar, 1>, Base>
{
public:

    using State = Kalman::Vector<Scalar, 2 * Dimension>;
    using Control = Kalman::Vector<Scalar, 1>;

    /// The model with the time step \a dt and the white-noise acceleration's spectral density
    /// \a accelDensity.
    MherbConstantVelocity(Scalar dt, Scalar accelDensity)
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

/// The ranging filter \a Kind with mherb/kalman in \a Scalar, with the analytic Jacobian.
template <int Dimension, int AnchorCount, FilterKind Kind = FilterKind::Ekf, typename Scalar = double>
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
        , m_filter{makeFilter()}
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

    /// The current covariance, formed from the factor in a square-root filter; in the covariance
    /// form its two triangles can differ by rounding.
    [[nodiscard]] Kalman::Covariance<State>
    covariance() const
    {
        return m_filter.getCovariance();
    }

private:

    /// The ranges to the anchors, with their Jacobian in position.
    template <template <class> class Base>
    class MeasurementOver : public Kalman::LinearizedMeasurementModel<State, Ranges, Base>
    {
    public:

        MeasurementOver(Eigen::Matrix<Scalar, Dimension, AnchorCount> const & anchors, Scalar rangeStdDev)
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

    using System = std::conditional_t<
        isSqrt(Kind),
        MherbConstantVelocity<Dimension, Scalar, Kalman::SquareRootBase>,
        MherbConstantVelocity<Dimension, Scalar, Kalman::StandardBase>>;
    using Measurement = std::
        conditional_t<isSqrt(Kind), MeasurementOver<Kalman::SquareRootBase>, MeasurementOver<Kalman::StandardBase>>;
    using Filter = std::conditional_t<
        Kind == FilterKind::Ekf,
        Kalman::ExtendedKalmanFilter<State>,
        std::conditional_t<
            Kind == FilterKind::Ukf,
            Kalman::UnscentedKalmanFilter<State>,
            std::conditional_t<
                Kind == FilterKind::SqrtEkf,
                Kalman::SquareRootExtendedKalmanFilter<State>,
                Kalman::SquareRootUnscentedKalmanFilter<State>>>>;

    /// The filter, the unscented ones with alpha = 1, beta = 2, and kappa = 0 written out.
    [[nodiscard]] static Filter
    makeFilter()
    {
        if constexpr (isUnscented(Kind))
        {
            return Filter{Scalar(1), Scalar(2), Scalar(0)};
        }
        else
        {
            return Filter{};
        }
    }

    System m_system;            ///< the process model
    Measurement m_measurement;  ///< the measurement model
    Filter m_filter;            ///< the filter
};

}  // namespace Ranging

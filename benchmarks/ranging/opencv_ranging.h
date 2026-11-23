#pragma once

// The ranging problem solved with OpenCV's cv::KalmanFilter (https://opencv.org, the video
// module), another implementation of [benchmarks]. Its filter is linear: predict() forms A x and
// A P A^T + Q, and correct() takes a measurement z and forms z - H x. Its users run an EKF through
// it the way this does: after each prediction they set its measurement matrix H to the Jacobian at
// the predicted mean x, and pass correct() the pseudo-measurement z - h(x) + H x, whose innovation
// z - h(x) + H x - H x is the EKF's. The motion here is linear, so the transition matrix is set
// once. Like mherb_ranging.h, it keeps the library's own arithmetic, which is what its users run:
// correct() solves S K^T = H P for the gain by a singular value decomposition of S, then updates
// the covariance as P - K (H P), with no triangle mirrored, and predict() copies its prediction
// into the posterior, for a step without a measurement. Every matrix is a cv::Mat, sized at run
// time. There is no automatic differentiation, so the Jacobian is analytic only. Compare
// emrys_ranging_kinds.h's EKF.

#include "scenario.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>

namespace Ranging
{

/// The ranging EKF with OpenCV's cv::KalmanFilter in \a Scalar, double or float, with the analytic
/// Jacobian.
template <int Dimension, int AnchorCount, typename Scalar = double>
class OpenCvRanging
{
public:

    using ScenarioType = Scenario<Dimension, AnchorCount>;
    static constexpr int StateSize = ScenarioType::StateSize;
    using Ranges = Eigen::Matrix<Scalar, AnchorCount, 1>;
    using State = Eigen::Matrix<Scalar, StateSize, 1>;
    using Covariance = Eigen::Matrix<Scalar, StateSize, StateSize>;

    /// The filter at \a scenario's initial estimate. The step is fixed, so the transition, the
    /// process noise, and the range noise are set once here, and the velocity columns of the
    /// measurement matrix, which stay zero.
    explicit OpenCvRanging(ScenarioType const & scenario)
        : m_filter{StateSize, AnchorCount, 0, cv::traits::Type<Scalar>::value}
        , m_pseudoMeasurement{AnchorCount, 1, cv::traits::Type<Scalar>::value}
    {
        for (int i = 0; i < AnchorCount; ++i)
        {
            for (int j = 0; j < Dimension; ++j)
            {
                m_anchors(j, i) = Scalar(scenario.anchors(j, i));
            }
        }
        Scalar const dt = Scalar(scenario.timeStep);
        Scalar const q = Scalar(scenario.accelDensity);
        cv::setIdentity(m_filter.transitionMatrix);
        m_filter.processNoiseCov.setTo(0);
        for (int i = 0; i < Dimension; ++i)
        {
            int const p = i;              // the position coordinate
            int const v = Dimension + i;  // its velocity
            m_filter.transitionMatrix.template at<Scalar>(p, v) = dt;
            m_filter.processNoiseCov.template at<Scalar>(p, p) = q * dt * dt * dt / 3;
            m_filter.processNoiseCov.template at<Scalar>(p, v) = q * dt * dt / 2;
            m_filter.processNoiseCov.template at<Scalar>(v, p) = q * dt * dt / 2;
            m_filter.processNoiseCov.template at<Scalar>(v, v) = q * dt;
        }
        Scalar const rangeStdDev = Scalar(scenario.rangeStdDev);
        cv::setIdentity(m_filter.measurementNoiseCov, cv::Scalar::all(double(rangeStdDev * rangeStdDev)));
        for (int i = 0; i < StateSize; ++i)
        {
            m_filter.statePost.template at<Scalar>(i) = Scalar(scenario.initialMean(i));
            for (int j = 0; j < StateSize; ++j)
            {
                m_filter.errorCovPost.template at<Scalar>(i, j) = Scalar(scenario.initialCov(i, j));
            }
        }
    }

    /// Moves the estimate forward by the scenario's time step.
    void
    predict()
    {
        m_filter.predict();
    }

    /// Conditions the estimate on the ranges \a z, one per anchor: sets the measurement matrix to
    /// the Jacobian of the ranges at the predicted mean, and corrects by the pseudo-measurement.
    void
    update(Ranges const & z)
    {
        Scalar const * const x = m_filter.statePre.template ptr<Scalar>();
        for (int i = 0; i < AnchorCount; ++i)
        {
            Scalar squared = 0;
            for (int j = 0; j < Dimension; ++j)
            {
                Scalar const d = x[j] - m_anchors(j, i);
                squared += d * d;
            }
            Scalar const range = std::sqrt(squared);
            // h(x) is the range, and H x of its row is the position's part, its velocity columns zero.
            Scalar linear = 0;
            for (int j = 0; j < Dimension; ++j)
            {
                Scalar const h = (x[j] - m_anchors(j, i)) / range;
                m_filter.measurementMatrix.template at<Scalar>(i, j) = h;
                linear += h * x[j];
            }
            m_pseudoMeasurement.template at<Scalar>(i) = z(i) - range + linear;
        }
        m_filter.correct(m_pseudoMeasurement);
    }

    /// The current mean, position then velocity.
    [[nodiscard]] State
    mean() const
    {
        return Eigen::Map<State const>{m_filter.statePost.template ptr<Scalar>()};
    }

    /// The current covariance; its two triangles can differ by rounding.
    [[nodiscard]] Covariance
    covariance() const
    {
        return Eigen::Map<Eigen::Matrix<Scalar, StateSize, StateSize, Eigen::RowMajor> const>{
            m_filter.errorCovPost.template ptr<Scalar>()
        };
    }

private:

    cv::KalmanFilter m_filter;                                ///< OpenCV's filter and its matrices
    cv::Mat m_pseudoMeasurement;                              ///< z - h(x) + H x, one row per anchor
    Eigen::Matrix<Scalar, Dimension, AnchorCount> m_anchors;  ///< one anchor per column
};

}  // namespace Ranging

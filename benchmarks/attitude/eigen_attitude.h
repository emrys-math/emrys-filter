#pragma once

// The attitude problem solved from scratch in plain Eigen, as a careful user would write it [benchmarks]:
// the multiplicative EKF of Markley (2003) in the body-frame form of Solà (2017), with a quaternion
// mean, a 6×6 covariance over the attitude error and the bias, an analytic transition, fixed sizes,
// noalias() products, only the attitude columns of the covariance in the update, and fixed-size
// derivative vectors for autodiff. It takes the library's covariance update form and
// factorization (P - K Pxy^T, with a Cholesky factor of S and the gain solved a row at a time) and
// its exact reset, the right Jacobian of SO(3) at the correction, where Markley leaves the
// covariance as it is and Solà keeps the reset to first order, so the two agree to rounding.
// Compare emrys_attitude.h.

#include "scenario.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/AutoDiff>

namespace Orientation
{

/// The attitude EKF in plain Eigen, linearizing the reference vectors as \a Kind says.
template <Jacobian Kind>
class EigenAttitude
{
public:

    using DifferenceMatrix = Scenario::DifferenceMatrix;

    explicit EigenAttitude(Scenario const & scenario)
        : m_gravity{scenario.gravity}
        , m_magneticField{scenario.magneticField}
        , m_gravityVariance{scenario.gravityStdDev * scenario.gravityStdDev}
        , m_magneticVariance{scenario.magneticStdDev * scenario.magneticStdDev}
        , m_timeStep{scenario.timeStep}
        , m_gyroNoise{scenario.gyroNoiseDensity * scenario.timeStep}
        , m_biasNoise{scenario.biasDensity * scenario.timeStep}
        , m_attitude{Eigen::Vector4d{scenario.initialMean.head<4>()}}
        , m_bias{scenario.initialMean.tail<3>()}
        , m_cov{scenario.initialCov}
    {}

    /// Moves the estimate forward by the scenario's time step dt through the gyroscope's reading
    /// \a gyro: the attitude turns by u = (gyro - b) dt, and the error moves by
    /// F = [R(u)^T, -J_r(u) dt; 0, I], where R(u) is the rotation of the turn and J_r its right
    /// Jacobian; then P = F P F^T + Q, with Q computed once at construction, since the step is
    /// fixed.
    void
    predict(Eigen::Vector3d const & gyro)
    {
        double const dt = m_timeStep;
        Eigen::Vector3d const turn = (gyro - m_bias) * dt;
        Eigen::Quaterniond const step = exponential(turn);
        DifferenceMatrix transition = DifferenceMatrix::Identity();
        transition.topLeftCorner<3, 3>() = step.toRotationMatrix().transpose();
        transition.topRightCorner<3, 3>() = -dt * rightJacobian(turn);

        m_attitude = (m_attitude * step).normalized();
        auto propagated = DifferenceMatrix{};
        propagated.noalias() = transition * m_cov;
        m_cov.noalias() = propagated * transition.transpose();
        m_cov.diagonal().template head<3>().array() += m_gyroNoise;
        m_cov.diagonal().template tail<3>().array() += m_biasNoise;
    }

    /// Conditions the estimate on the accelerometer's reading \a gravity, then on the
    /// magnetometer's reading \a magneticField.
    void
    update(Eigen::Vector3d const & gravity, Eigen::Vector3d const & magneticField)
    {
        updateBy(gravity, m_gravity, m_gravityVariance);
        updateBy(magneticField, m_magneticField, m_magneticVariance);
    }

    /// The current attitude.
    [[nodiscard]] Eigen::Quaterniond const &
    attitude() const
    {
        return m_attitude;
    }

    /// The current bias.
    [[nodiscard]] Eigen::Vector3d const &
    bias() const
    {
        return m_bias;
    }

    /// The current covariance, over the attitude error then the bias.
    [[nodiscard]] DifferenceMatrix const &
    covarianceLower() const
    {
        return m_cov;
    }

private:

    using Matrix3 = Eigen::Matrix3d;
    using Vector6 = Eigen::Matrix<double, 6, 1>;

    /// Conditions the estimate on the reading \a z, of variance \a variance in each coordinate, of
    /// the world-frame unit vector \a reference in the body frame.
    void
    updateBy(Eigen::Vector3d const & z, Eigen::Vector3d const & reference, double variance)
    {
        // The residual z - h(q) and h's Jacobian in the attitude error; h does not depend on bias.
        auto residual = Eigen::Vector3d{};
        auto h = Matrix3{};
        linearize(z, reference, residual, h);

        // Pxy = P H^T, where H is h's Jacobian padded with zero bias columns.
        auto cross = Eigen::Matrix<double, 6, 3>{};
        cross.noalias() = m_cov.template leftCols<3>() * h.transpose();
        Matrix3 innovationCov = variance * Matrix3::Identity();
        innovationCov.noalias() += h * cross.topRows<3>();

        auto const llt = Eigen::LLT<Matrix3>{innovationCov};
        // A row of the gain at a time, as the library solves it: Eigen unrolls a solve with a small
        // fixed vector and sends a matrix right-hand side to its blocked solver.
        auto gain = Eigen::Matrix<double, 6, 3>{};
        for (int i = 0; i < 6; ++i)
        {
            Eigen::Vector3d const row = cross.row(i).transpose();
            gain.row(i) = llt.solve(row).transpose();
        }
        auto correction = Vector6{};
        correction.noalias() = gain * residual;
        m_cov.noalias() -= gain * cross.transpose();

        // The correction moves the attitude by its own exponential, and the reset carries the
        // covariance from the error about the old attitude to the error about the new one: the
        // attitude rows and columns by G = J_r(d).
        Eigen::Vector3d const turn = correction.head<3>();
        m_attitude = (m_attitude * exponential(turn)).normalized();
        m_bias += correction.tail<3>();
        Matrix3 const reset = rightJacobian(turn);
        Eigen::Matrix<double, 3, 6> const rows = reset * m_cov.template topRows<3>();
        m_cov.template topRows<3>() = rows;
        Eigen::Matrix<double, 6, 3> const columns = m_cov.template leftCols<3>() * reset.transpose();
        m_cov.template leftCols<3>() = columns;
    }

    /// Writes the residual z - h(q) at the mean to \a residual and h's Jacobian in the attitude
    /// error to \a h, where h(q) = R(q)^T r is the reference \a reference seen from the body.
    void
    linearize(
        Eigen::Vector3d const & z,
        Eigen::Vector3d const & reference,
        Eigen::Vector3d & residual,
        Matrix3 & h
    ) const
    {
        if constexpr (Kind == Jacobian::Analytic)
        {
            // Turning the body by a small d moves the seen reference v by v × d, so h = [v]×.
            Eigen::Vector3d const v = m_attitude.conjugate() * reference;
            residual = z - v;
            h << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
        }
        else
        {
            // Forward mode over the attitude error at zero, q ⊗ Exp(d), with a fixed-size derivative
            // vector, so nothing allocates.
            using Dual = Eigen::AutoDiffScalar<Eigen::Vector3d>;
            auto error = Eigen::Matrix<Dual, 3, 1>{};
            for (int j = 0; j < 3; ++j)
            {
                error(j) = Dual{0.0, 3, j};
            }
            Eigen::Quaternion<Dual> const turned = m_attitude.template cast<Dual>() * exponential(error);
            Eigen::Matrix<Dual, 3, 1> const v = turned.conjugate() * reference.cast<Dual>();
            for (int i = 0; i < 3; ++i)
            {
                residual(i) = z(i) - v(i).value();
                h.row(i) = v(i).derivatives().transpose();
            }
        }
    }

    /// The unit quaternion of the rotation vector \a d, by its series below a small angle, where
    /// it stays differentiable.
    template <typename T>
    [[nodiscard]] static Eigen::Quaternion<T>
    exponential(Eigen::Matrix<T, 3, 1> const & d)
    {
        using std::cos;
        using std::sin;
        using std::sqrt;
        T const angle2 = d.squaredNorm();
        if (angle2 < T(1e-12))
        {
            T const scale = T(0.5) - angle2 / T(48);
            return Eigen::Quaternion<T>{T(1) - angle2 / T(8), d(0) * scale, d(1) * scale, d(2) * scale};
        }
        T const angle = sqrt(angle2);
        T const scale = T(sin(T(angle / T(2))) / angle);
        return Eigen::Quaternion<T>{cos(T(angle / T(2))), d(0) * scale, d(1) * scale, d(2) * scale};
    }

    /// The right Jacobian of SO(3) at the rotation vector \a d, Solà, Deray, and Atchuthan (2018),
    /// eq. (143), by its series below a small angle.
    [[nodiscard]] static Matrix3
    rightJacobian(Eigen::Vector3d const & d)
    {
        auto skew = Matrix3{};
        skew << 0.0, -d(2), d(1), d(2), 0.0, -d(0), -d(1), d(0), 0.0;
        double const angle2 = d.squaredNorm();
        if (angle2 < 1e-8)
        {
            return Matrix3::Identity() - 0.5 * skew + (skew * skew) / 6.0;
        }
        double const angle = std::sqrt(angle2);
        return Matrix3::Identity() - (1 - std::cos(angle)) / angle2 * skew
             + (angle - std::sin(angle)) / (angle2 * angle) * (skew * skew);
    }

    Eigen::Vector3d m_gravity;        ///< the accelerometer's reference in the world frame
    Eigen::Vector3d m_magneticField;  ///< the magnetometer's reference in the world frame
    double m_gravityVariance;         ///< variance of each accelerometer coordinate
    double m_magneticVariance;        ///< variance of each magnetometer coordinate
    double m_timeStep;                ///< the fixed time between readings
    double m_gyroNoise;               ///< the attitude's noise variance per axis over the step
    double m_biasNoise;               ///< the bias's noise variance per axis over the step
    Eigen::Quaterniond m_attitude;    ///< the estimate's attitude
    Eigen::Vector3d m_bias;           ///< the estimate's bias
    DifferenceMatrix m_cov;           ///< the estimate's covariance, over the attitude error and the bias
};

}  // namespace Orientation

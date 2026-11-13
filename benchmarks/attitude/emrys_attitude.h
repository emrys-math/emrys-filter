#pragma once

// The attitude problem solved with EmrysFilter: an extended Kalman filter over a unit quaternion
// and a gyroscope bias, predicting through a process model that integrates the gyroscope's reading,
// and updating by each reference vector in turn. The update moves the quaternion by its own plus()
// and carries the covariance to the new mean by the reset of [covariance-reset]. Compare eigen_attitude.h, the
// same filter written from scratch.

#include "scenario.h"

#include <cstdlib>
#include <functional>

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

namespace Orientation
{

/// The state's blocks: the attitude, which takes body-frame vectors to the world frame, and the
/// gyroscope's bias.
struct Attitude : Emrys::UnitQuaternion
{};

struct Bias : Emrys::Linear<3>
{};

/// The gyroscope's reading, in radians per second, a sure input of each prediction.
struct Gyro : Emrys::Linear<3>
{};

/// The process model that turns the attitude by the gyroscope's reading less the bias over the
/// step, with the reading's white noise integrated into the attitude. It is what a user writes for
/// a gyroscope: the library has none built in. Its function reads the attitude, the bias, and the
/// reading, and writes the attitude, \f$ q \otimes \mathrm{Exp}((\omega - b) \Delta t) \f$. It is
/// affine in neither the state nor its input, so a fixed step caches no transition, but it returns
/// its Jacobian at the state with its step, from one exponential of the turn, so the Linearized
/// prediction needs no differentiation [returned-jacobians]: for the turn
/// \f$ u = (\omega - b) \Delta t \f$, \f$ R(u)^T \f$ in the attitude, \f$ -J_r(u) \Delta t \f$ in
/// the bias, as Solà (2017) gives them, and zero in the reading, which is sure. \a noiseDensity is
/// the spectral density of the gyroscope's white noise, rad²/s, integrated as
/// \f$ N \Delta t \f$ on each axis of the attitude.
[[nodiscard]] inline auto
gyroIntegration(double noiseDensity)
{
    return Emrys::processModel(
        [](auto const & in, auto const & out, auto const & x, double dt) {
            Eigen::Vector3d const turn = (Emrys::block<Gyro>(in, x) - Emrys::block<Bias>(in, x)) * dt;
            auto const step = Emrys::UnitQuaternion::exponential(turn);
            Eigen::Vector4d y;
            Emrys::UnitQuaternion::map(Emrys::block<Attitude>(out, y)) =
                Emrys::UnitQuaternion::map(Emrys::block<Attitude>(in, x)) * step;
            Eigen::Matrix<double, 3, 9> jacobian = Eigen::Matrix<double, 3, 9>::Zero();
            jacobian.leftCols<3>() = step.toRotationMatrix().transpose();
            jacobian.middleCols<3>(3) = -dt * Emrys::UnitQuaternion::rightJacobian(turn);
            return Emrys::withJacobian(y, jacobian);
        },
        Emrys::reads(Emrys::path<Attitude>, Emrys::path<Bias>, Emrys::path<Gyro>),
        Emrys::writes(Emrys::path<Attitude>),
        Emrys::continuousNoise(Emrys::on<Attitude>(noiseDensity))
    );
}

/// The process model of \a scenario: the gyroscope's integration beside a random walk of the bias.
[[nodiscard]] inline auto
makeModel(Scenario const & scenario)
{
    return Emrys::compose(gyroIntegration(scenario.gyroNoiseDensity), Emrys::RandomWalk<Bias>{scenario.biasDensity});
}

/// The attitude EKF with EmrysFilter, linearizing the reference vectors as \a Kind says.
template <Jacobian Kind>
class EmrysAttitude
{
public:

    using State = Emrys::Composite<Attitude, Bias>;

    explicit EmrysAttitude(Scenario const & scenario)
        : m_gravity{scenario.gravity}
        , m_magneticField{scenario.magneticField}
        , m_gravityNoise{noiseOf(scenario.gravityStdDev)}
        , m_magneticNoise{noiseOf(scenario.magneticStdDev)}
        , m_filter{
              Model{makeModel(scenario), State{}, scenario.timeStep},
              Emrys::GaussianParams<State>{State{}, scenario.initialMean, scenario.initialCov}
          }
    {}

    /// Moves the estimate forward by the scenario's time step through the gyroscope's reading
    /// \a gyro over the step, a sure input. The step is fixed, so the model holds the process noise
    /// from construction; the transition depends on the state and the reading, so it is not cached.
    void
    predict(Eigen::Vector3d const & gyro)
    {
        m_filter.predict(Emrys::sure<Gyro>(gyro));
    }

    /// Conditions the estimate on the accelerometer's reading \a gravity, then on the
    /// magnetometer's reading \a magneticField, and stops the program if either innovation
    /// covariance is not positive definite.
    void
    update(Eigen::Vector3d const & gravity, Eigen::Vector3d const & magneticField)
    {
        if (!m_filter.update(measure(gravity, m_gravity, m_gravityNoise))
            || !m_filter.update(measure(magneticField, m_magneticField, m_magneticNoise)))
        {
            std::abort();
        }
    }

    /// The current estimate, attitude then bias.
    [[nodiscard]] Emrys::GaussianParams<State> const &
    state() const
    {
        return m_filter.state();
    }

private:

    using Model = Emrys::FixedStepModel<State, decltype(makeModel(std::declval<Scenario const &>()))>;
    using Filter = Emrys::KalmanFilter<Emrys::Gaussian<State>, Model>;
    using Noise = Emrys::GaussianParams<Emrys::Linear<3>>;

    /// Independent noise of standard deviation \a stdDev on each coordinate of a reading.
    [[nodiscard]] static Noise
    noiseOf(double stdDev)
    {
        return Noise{Eigen::Vector3d::Zero(), Eigen::Matrix3d{stdDev * stdDev * Eigen::Matrix3d::Identity()}};
    }

    /// The measurement of the attitude by the reading \a z of the world-frame unit vector \a
    /// reference in the body frame. The noise outlives the measurement, so the measurement refers
    /// to it rather than copying its covariance.
    [[nodiscard]] auto
    measure(Eigen::Vector3d const & z, Eigen::Vector3d const & reference, Noise const & noise) const
    {
        if constexpr (Kind == Jacobian::Analytic)
        {
            // The predicted reading R(q)^T r with its Jacobian in the attitude's differences:
            // turning the body by a small rotation vector d moves v = R(q)^T r by v × d, so the
            // Jacobian is [v]×.
            auto const withJacobian = [&](auto const & in, auto const &, Eigen::Vector4d const & x) {
                Eigen::Vector4d const q = Emrys::block<Attitude>(in, x);
                Eigen::Vector3d const v = Emrys::UnitQuaternion::map(q).conjugate() * reference;
                auto skew = Eigen::Matrix3d{};
                skew << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
                return Emrys::withJacobian(v, skew);
            };
            return observe(
                Emrys::methods::linearized(),
                Emrys::measurementModel(
                    withJacobian,
                    Emrys::reads(Emrys::path<Attitude>),
                    Emrys::additiveNoise(std::cref(noise))
                ),
                z
            );
        }
        else
        {
            // The predicted reading R(q)^T r, generic in its scalar so that it can be differentiated.
            auto const reading = [&]<typename T>(auto const & in, auto const &, auto const & x) {
                Eigen::Matrix<T, 4, 1> const q = Emrys::block<Attitude>(in, x);
                Eigen::Matrix<T, 3, 1> v = Emrys::UnitQuaternion::map(q).conjugate() * reference.cast<T>();
                return v;
            };
            return observe(
                Emrys::methods::linearized(),
                Emrys::measurementModel(
                    reading,
                    Emrys::reads(Emrys::path<Attitude>),
                    Emrys::additiveNoise(std::cref(noise))
                ),
                z
            );
        }
    }

    Eigen::Vector3d m_gravity;        ///< the accelerometer's reference in the world frame
    Eigen::Vector3d m_magneticField;  ///< the magnetometer's reference in the world frame
    Noise m_gravityNoise;             ///< the accelerometer's noise
    Noise m_magneticNoise;            ///< the magnetometer's noise
    Filter m_filter;                  ///< the filter
};

}  // namespace Orientation

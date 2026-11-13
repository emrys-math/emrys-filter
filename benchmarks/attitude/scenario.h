#pragma once

// The attitude problem of [benchmarks], shared by both implementations: a body turning at a varying rate
// carries a gyroscope with a slowly drifting bias, which drives the prediction, and two
// reference-vector sensors, an accelerometer that sees gravity and a magnetometer that sees the
// magnetic field, which correct it. The problem is that of Markley (2003) and Solà (2017). This
// header holds the problem's parameters and a simulator of the true attitude, the true bias, and
// the readings, in plain Eigen, so neither filter's code is involved in producing its own test data.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <random>
#include <vector>

namespace Orientation
{

/// How an implementation linearizes the reference-vector measurements: an analytic Jacobian or
/// automatic differentiation, the second axis of [benchmarks]'s 2×2 design. Either way the prediction
/// through the gyroscope is linearized as each implementation must: the hand-rolled filter writes
/// its Jacobian, and EmrysFilter differentiates the process model, which has no way to give one.
enum class Jacobian
{
    Analytic,
    AutoDiff
};

/// The problem's reference vectors, noise levels, and initial estimate. The state is the attitude,
/// a unit quaternion (x, y, z, w) that takes body-frame vectors to the world frame, then the
/// gyroscope's bias; its differences are a body-frame rotation vector, then the bias.
struct Scenario
{
    using StateVector = Eigen::Matrix<double, 7, 1>;
    using DifferenceMatrix = Eigen::Matrix<double, 6, 6>;

    Eigen::Vector3d gravity;        ///< the accelerometer's reference, a unit vector in the world frame
    Eigen::Vector3d magneticField;  ///< the magnetometer's reference, a unit vector in the world frame
    double gravityStdDev;           ///< standard deviation of each accelerometer coordinate
    double magneticStdDev;          ///< standard deviation of each magnetometer coordinate
    double gyroNoiseDensity;        ///< spectral density of the gyroscope's white noise, rad²/s
    double biasDensity;             ///< spectral density of the bias's random walk, rad²/s³
    double timeStep;                ///< time between readings, in seconds
    StateVector initialMean;        ///< the true initial state's mean, and the filters' first estimate
    DifferenceMatrix initialCov;    ///< its covariance, over differences
};

/// The unit quaternion of the rotation vector \a d, by its series near zero.
[[nodiscard]] inline Eigen::Quaterniond
exponential(Eigen::Vector3d const & d)
{
    double const angle = d.norm();
    if (angle < 1e-8)
    {
        return Eigen::Quaterniond{1.0, d(0) / 2, d(1) / 2, d(2) / 2}.normalized();
    }
    return Eigen::Quaterniond{Eigen::AngleAxisd{angle, d / angle}};
}

/// The accelerometer's reference along the world's vertical z axis, a magnetic field dipping 60
/// degrees below the horizontal, a 100 Hz gyroscope with an angle random walk of 0.06 degrees per root second and a
/// bias that drifts by 0.006 degrees per second per root second, and an initial estimate 6 degrees
/// and 0.6 degrees per second uncertain in each axis.
[[nodiscard]] inline Scenario
makeScenario()
{
    auto mean = Scenario::StateVector{};
    mean << exponential(Eigen::Vector3d{0.3, -0.2, 1.0}).coeffs(), Eigen::Vector3d::Zero();
    auto variances = Eigen::Matrix<double, 6, 1>{};
    variances << Eigen::Vector3d::Constant(0.01), Eigen::Vector3d::Constant(1e-4);
    return Scenario{
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d{0.5, 0.0, -std::sqrt(0.75)},
        0.02,
        0.02,
        1e-6,
        1e-8,
        0.01,
        mean,
        Scenario::DifferenceMatrix{variances.asDiagonal()}
    };
}

/// A simulated run: at each step the gyroscope's reading over the step, then the true state and
/// the two reference vectors measured at its end.
struct Trajectory
{
    std::vector<Eigen::Vector3d> gyro;           ///< the gyroscope's reading over each step
    std::vector<Scenario::StateVector> truth;    ///< the true state at the end of each step
    std::vector<Eigen::Vector3d> gravity;        ///< the accelerometer's reading there
    std::vector<Eigen::Vector3d> magneticField;  ///< the magnetometer's reading there
};

/// The body's true rate at time \a t, in radians per second about its own axes.
[[nodiscard]] inline Eigen::Vector3d
trueRate(double t)
{
    return Eigen::Vector3d{0.5 * std::sin(0.7 * t), 0.4 * std::cos(0.5 * t), 0.2 + 0.3 * std::sin(0.3 * t)};
}

/// Simulates \a steps steps of \a scenario from a true initial state drawn from its initial
/// estimate. The gyroscope reads the true rate plus the bias plus white noise, and the attitude
/// turns by the true rate, so the filters' model, which turns by the reading less the bias and
/// adds the noise's integral to the attitude, is exact to first order in the step.
[[nodiscard]] inline Trajectory
simulate(Scenario const & scenario, int steps, unsigned seed)
{
    auto engine = std::mt19937{seed};
    auto normal = std::normal_distribution<double>{};
    auto const draw = [&] {
        return Eigen::Vector3d{normal(engine), normal(engine), normal(engine)};
    };

    double const dt = scenario.timeStep;
    double const gyroStdDev = std::sqrt(scenario.gyroNoiseDensity / dt);
    double const biasStepStdDev = std::sqrt(scenario.biasDensity * dt);
    Scenario::DifferenceMatrix const initialFactor = scenario.initialCov.llt().matrixL();
    Eigen::Matrix<double, 6, 1> const initialError =
        initialFactor * (Eigen::Matrix<double, 6, 1>{} << draw(), draw()).finished();

    auto result = Trajectory{};
    result.gyro.reserve(steps);
    result.truth.reserve(steps);
    result.gravity.reserve(steps);
    result.magneticField.reserve(steps);
    Eigen::Quaterniond attitude =
        Eigen::Quaterniond{Eigen::Vector4d{scenario.initialMean.head<4>()}} * exponential(initialError.head<3>());
    Eigen::Vector3d bias = scenario.initialMean.tail<3>() + initialError.tail<3>();
    for (int k = 0; k < steps; ++k)
    {
        Eigen::Vector3d const rate = trueRate(k * dt);
        result.gyro.push_back(rate + bias + gyroStdDev * draw());
        attitude = (attitude * exponential(rate * dt)).normalized();
        bias += biasStepStdDev * draw();
        Eigen::Quaterniond const toBody = attitude.conjugate();
        result.gravity.push_back(toBody * scenario.gravity + scenario.gravityStdDev * draw());
        result.magneticField.push_back(toBody * scenario.magneticField + scenario.magneticStdDev * draw());
        result.truth.push_back((Scenario::StateVector{} << attitude.coeffs(), bias).finished());
    }
    return result;
}

}  // namespace Orientation

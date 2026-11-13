#pragma once

// The ranging geolocation problem of [benchmarks], shared by both implementations: a receiver moving at
// nearly constant velocity ranges to known, fixed anchors. This header holds the problem's
// parameters and a simulator of the true trajectory and its measurements, in plain Eigen, so
// neither filter's code is involved in producing its own test data.

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

namespace Ranging
{

/// How an implementation linearizes the range measurement: an analytic Jacobian or automatic
/// differentiation, the second axis of [benchmarks]'s 2×2 design.
enum class Jacobian
{
    Analytic,
    AutoDiff
};

/// How far the two implementations in \a Scalar may differ, relative to the size of the state and
/// of the covariance, for the difference to be rounding: 1e-9 in double, 1e-4 in float. Either is
/// far below a modeling difference, which is of order one.
template <typename Scalar>
inline constexpr double agreementTolerance = 1e-9;

template <>
inline constexpr double agreementTolerance<float> = 1e-4;

/// The problem in Dimension spatial dimensions with AnchorCount anchors: the anchors, the noise
/// levels, and the initial estimate. The state is position then velocity.
template <int Dimension, int AnchorCount>
struct Scenario
{
    static constexpr int StateSize = 2 * Dimension;
    using StateVector = Eigen::Matrix<double, StateSize, 1>;
    using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
    using Anchors = Eigen::Matrix<double, Dimension, AnchorCount>;
    using Ranges = Eigen::Matrix<double, AnchorCount, 1>;

    Anchors anchors;          ///< one anchor position per column, in meters
    double rangeStdDev;       ///< standard deviation of each range, in meters
    double accelDensity;      ///< spectral density of the white-noise acceleration, m²/s³
    double timeStep;          ///< time between measurements, in seconds
    StateVector initialMean;  ///< the true initial state's mean, and the filters' first estimate
    StateMatrix initialCov;   ///< its covariance
};

/// The anchors spaced evenly on a circle of radius 100 m around the origin; in 3D, alternately
/// at heights 0 and 30 m, so that height is observable from the ranges alone.
template <int Dimension, int AnchorCount>
[[nodiscard]] Scenario<Dimension, AnchorCount>
makeScenario()
{
    static_assert(Dimension == 2 || Dimension == 3, "the scenario is planar or spatial");
    using S = Scenario<Dimension, AnchorCount>;
    typename S::Anchors anchors = S::Anchors::Zero();
    for (int i = 0; i < AnchorCount; ++i)
    {
        double const angle = 2 * std::numbers::pi * i / AnchorCount;
        anchors(0, i) = 100 * std::cos(angle);
        anchors(1, i) = 100 * std::sin(angle);
        if constexpr (Dimension == 3)
        {
            anchors(2, i) = i % 2 == 0 ? 0.0 : 30.0;
        }
    }
    typename S::StateVector mean = S::StateVector::Zero();
    mean.template tail<Dimension>().setConstant(1.0);
    auto variances = typename S::StateVector{};
    variances << Eigen::Matrix<double, Dimension, 1>::Constant(25.0),
        Eigen::Matrix<double, Dimension, 1>::Constant(1.0);
    return S{anchors, 0.3, 0.1, 0.1, mean, typename S::StateMatrix{variances.asDiagonal()}};
}

/// A simulated run: the true states and the ranges measured at each of them.
template <int Dimension, int AnchorCount>
struct Trajectory
{
    using S = Scenario<Dimension, AnchorCount>;
    std::vector<typename S::StateVector> truth;  ///< the true state at each step
    std::vector<typename S::Ranges> ranges;      ///< the ranges measured at that state
};

/// The simulation of \a scenario one step at a time, from a true initial state drawn from its
/// initial estimate, with process noise drawn from the exact discretization of the white-noise
/// acceleration model, so that a correct filter is consistent from the first step. It holds only
/// the current true state, so a target without a heap [embedded-builds] runs it as it goes;
/// simulate() records it.
template <int Dimension, int AnchorCount>
class Simulator
{
public:

    using S = Scenario<Dimension, AnchorCount>;

    Simulator(S const & scenario, unsigned seed)
        : m_scenario{scenario}
        , m_engine{seed}
    {
        using Block = Eigen::Matrix<double, Dimension, Dimension>;
        double const dt = scenario.timeStep;
        Block const identity = Block::Identity();
        m_transition.template topRightCorner<Dimension, Dimension>() = dt * identity;
        auto processNoise = typename S::StateMatrix{};
        processNoise << dt * dt * dt / 3 * identity, dt * dt / 2 * identity, dt * dt / 2 * identity, dt * identity;
        processNoise *= scenario.accelDensity;
        m_processFactor = processNoise.llt().matrixL();
        typename S::StateMatrix const initialFactor = scenario.initialCov.llt().matrixL();
        m_truth = scenario.initialMean + initialFactor * draw();
    }

    /// Moves the true state one step and returns the ranges measured there.
    [[nodiscard]] typename S::Ranges
    step()
    {
        m_truth = (m_transition * m_truth + m_processFactor * draw()).eval();
        auto z = typename S::Ranges{};
        for (int i = 0; i < AnchorCount; ++i)
        {
            z(i) = (m_truth.template head<Dimension>() - m_scenario.anchors.col(i)).norm()
                 + m_scenario.rangeStdDev * m_normal(m_engine);
        }
        return z;
    }

    /// The true state after the last step.
    [[nodiscard]] typename S::StateVector const &
    truth() const
    {
        return m_truth;
    }

private:

    /// A standard normal draw for every coefficient of the state.
    [[nodiscard]] typename S::StateVector
    draw()
    {
        auto w = typename S::StateVector{};
        for (auto & coefficient : w)
        {
            coefficient = m_normal(m_engine);
        }
        return w;
    }

    S m_scenario;                                                       ///< the problem simulated
    std::mt19937 m_engine;                                              ///< the noise's generator
    std::normal_distribution<double> m_normal;                          ///< a standard normal
    typename S::StateMatrix m_transition = S::StateMatrix::Identity();  ///< the transition over a step
    typename S::StateMatrix m_processFactor;                            ///< a square root of the process noise
    typename S::StateVector m_truth;                                    ///< the true state
};

/// Simulates \a steps steps of \a scenario as Simulator does, recording each true state and the
/// ranges measured at it.
template <int Dimension, int AnchorCount>
[[nodiscard]] Trajectory<Dimension, AnchorCount>
simulate(Scenario<Dimension, AnchorCount> const & scenario, int steps, unsigned seed)
{
    auto simulator = Simulator<Dimension, AnchorCount>{scenario, seed};
    auto result = Trajectory<Dimension, AnchorCount>{};
    result.truth.reserve(steps);
    result.ranges.reserve(steps);
    for (int k = 0; k < steps; ++k)
    {
        result.ranges.push_back(simulator.step());
        result.truth.push_back(simulator.truth());
    }
    return result;
}

/// The ranges of \a run in \a Scalar, as a filter in that scalar receives them, converted before
/// any filter runs so that no conversion is timed.
template <typename Scalar, int Dimension, int AnchorCount>
[[nodiscard]] std::vector<Eigen::Matrix<Scalar, AnchorCount, 1>>
rangesIn(Trajectory<Dimension, AnchorCount> const & run)
{
    auto result = std::vector<Eigen::Matrix<Scalar, AnchorCount, 1>>{};
    result.reserve(run.ranges.size());
    for (auto const & z : run.ranges)
    {
        result.push_back(z.template cast<Scalar>());
    }
    return result;
}

}  // namespace Ranging

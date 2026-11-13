#pragma once

// The fixtures shared by the tests of the filters over the square-root and information forms of
// the Gaussian (tests/Filters/info_filter.cpp and sqrt_filters.cpp; roadmap steps 16
// and 17, decisions.md [gaussian-forms]): the constant-velocity state, model, and initial Gaussian of
// tests/Filters/kalman_filter.cpp, a position fix and a range-bearing measurement by any method, a
// velocity fix, the comparison of a form filter's state with a Kalman filter's through
// toGaussian(), an attitude state with a random walk on a unit quaternion and a measured
// direction, from tests/unequal_dimensions_filters.cpp, and a gyroscope-driven attitude with a
// bias, whose turn returns its Jacobian with its step or is differentiated [returned-jacobians].

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include <gtest/gtest.h>

#include "unequal_dimensions.h"

#include <cmath>

namespace EmrysTest::FormFilters
{

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double accelDensity = 0.1;
constexpr double timeStep = 1.0;

inline auto
makeModel(double density = accelDensity)
{
    return ConstantVelocity<Position, Velocity>{density};
}

inline GaussianParams<State>
makeInitial()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, cov};
}

Eigen::Matrix2d const fixNoise{{1.0, 0.2}, {0.2, 0.8}};
Eigen::Matrix2d const velocityNoise{{0.3, 0.0}, {0.0, 0.5}};

// A fix of the block named by P, z = x + v, with noise r.
template <typename P = Position, typename Method = methods::Linearized>
auto
makeFix(Eigen::Vector2d const & z, Method const & method = {}, Eigen::Matrix2d const & r = fixNoise)
{
    auto const read = [](auto const & in, auto const &, auto const & x) {
        return block<P>(in, x);
    };
    return observe(
        method,
        measurementModel(read, reads(path<P>), additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), r})),
        z
    );
}

// Range and bearing from the origin, with a periodic bearing.
struct Range : Linear<1>
{};

struct Bearing : RadiansPi<1>
{};

using RangeBearingSpace = Composite<Range, Bearing>;
Eigen::Matrix2d const rangeBearingNoise{{0.04, 0.0}, {0.0, 0.0009}};

// The range-bearing model with the noise given, additive over the measurement space.
template <typename Noise>
auto
makeRangeBearingModel(Noise noise)
{
    return measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            return Eigen::Matrix<T, 2, 1>{
                Eigen::numext::sqrt(T(p.squaredNorm())),
                Eigen::numext::atan2(T(p(1)), T(p(0)))
            };
        },
        reads(path<Position>),
        outputSpace(RangeBearingSpace{}),
        additiveNoise(std::move(noise))
    );
}

// The range and bearing z observed by method, with any further arguments of observe(), such as a
// covariance update form.
template <typename Method, typename... Args>
auto
makeRangeBearing(Eigen::Vector2d const & z, Method const & method, Args const &... args)
{
    return observe(
        method,
        makeRangeBearingModel(
            GaussianParams<RangeBearingSpace>{RangeBearingSpace{}, Eigen::Vector2d::Zero(), rangeBearingNoise}
        ),
        z,
        args...
    );
}

// A target moving at the initial velocity, seen at step k, and its range and bearing.
inline Eigen::Vector2d
fixAt(int k)
{
    return Eigen::Vector2d{1.0, 0.5} * k + 0.3 * Eigen::Vector2d{std::cos(2.4 * k), std::sin(2.4 * k)};
}

inline Eigen::Vector2d
rangeBearingAt(int k)
{
    Eigen::Vector2d const p = fixAt(k) + Eigen::Vector2d{3.0, 1.0};
    return Eigen::Vector2d{p.norm(), std::atan2(p(1), p(0))};
}

// The state of a filter over a form, as a Gaussian, must equal expected to tolerance.
template <typename Form, typename Space>
void
expectSameState(Form const & form, GaussianParams<Space> const & expected, double tolerance)
{
    auto const gaussian = Emrys::toGaussian(form);
    ASSERT_TRUE(gaussian) << "the form has no moments";
    EXPECT_LT(Emrys::minus(expected.space(), gaussian->mean(), expected.mean()).norm(), tolerance);
    EXPECT_LT((gaussian->covarianceLower() - expected.covarianceLower()).norm(), tolerance);
}

// The attitude state of tests/unequal_dimensions.h: a unit quaternion, a position, and a velocity.
using Motion = EmrysTest::UnequalDimensions::Motion;

using Orientation = EmrysTest::UnequalDimensions::Orientation;

// Random-walk noise on the rotation vector of the quaternion at Path, which does not move.
template <typename Path>
auto
attitudeWalk(double spectralDensity)
{
    return Emrys::processModel(
        [](auto const &, auto const &, auto const & x, auto) {
            return x;
        },
        Emrys::reads(Emrys::path<Path>),
        Emrys::writes(Emrys::path<Path>),
        Emrys::continuousNoise(Emrys::on<Path>(spectralDensity))
    );
}

inline auto
makeAttitudeModel()
{
    return compose(
        attitudeWalk<Orientation>(0.01),
        ConstantVelocity<EmrysTest::UnequalDimensions::Position, EmrysTest::UnequalDimensions::Velocity>{0.1}
    );
}

inline GaussianParams<Motion>
makeAttitudeInitial()
{
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << EmrysTest::UnequalDimensions::turned, 10.0, -5.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Zero();
    cov.topLeftCorner<5, 5>() = EmrysTest::UnequalDimensions::poseCov();
    cov.bottomRightCorner<2, 2>() = 0.25 * Eigen::Matrix2d::Identity();
    return GaussianParams<Motion>{mean, cov};
}

// A gyroscope-driven attitude: a unit quaternion turned over each step by a body rate, a sure
// input of the call read less a bias, and the bias, which walks.
struct Attitude : Emrys::UnitQuaternion
{};

struct GyroBias : Linear<3>
{};

// The body rate the gyroscope reads, an input of the call.
struct GyroRate : Linear<3>
{};

using GyroState = Composite<Attitude, GyroBias>;

// The turned attitude, q' = q Exp((w - b) dt), written into y in the scalar of x.
template <typename In, typename Out, typename X, typename TimeScalar>
auto
turnAttitude(In const & in, Out const & out, X const & x, TimeScalar dt)
{
    using T = ScalarOf<X>;
    auto y = Emrys::zeros<T>(out);
    Eigen::Matrix<T, 3, 1> const turn = (block<GyroRate>(in, x) - block<GyroBias>(in, x)) * T(dt);
    Emrys::UnitQuaternion::map(block<Attitude>(out, y)) =
        Emrys::UnitQuaternion::map(block<Attitude>(in, x)) * Emrys::UnitQuaternion::exponential(turn);
    return y;
}

// The model that turns the attitude by the rate less the bias with no Jacobian of its own, so that
// the Linearized prediction differentiates it.
struct DifferentiatedGyroTurn
{
    static auto
    make()
    {
        return Emrys::processModel(
            [](auto const & in, auto const & out, auto const & x, auto dt) {
                return turnAttitude(in, out, x, dt);
            },
            reads(path<Attitude>, path<GyroBias>, path<GyroRate>),
            Emrys::writes(path<Attitude>),
            Emrys::continuousNoise(Emrys::on<Attitude>(1e-4))
        );
    }
};

// The same model returning its Jacobian with its value [returned-jacobians]: R(u)^T in the attitude
// and -J_r(u) dt in the bias, for the turn u, from Solà, Deray, and Atchuthan (2018), and zero in
// the rate, a sure input.
struct GyroTurn
{
    static auto
    make()
    {
        return Emrys::processModel(
            [](auto const & in, auto const & out, auto const & x, double dt) {
                Eigen::Vector3d const turn = (block<GyroRate>(in, x) - block<GyroBias>(in, x)) * dt;
                Eigen::Matrix<double, 3, 9> jacobian = Eigen::Matrix<double, 3, 9>::Zero();
                jacobian.leftCols<3>() = Emrys::UnitQuaternion::exponential(turn).toRotationMatrix().transpose();
                jacobian.middleCols<3>(3) = -dt * Emrys::UnitQuaternion::rightJacobian(turn);
                return Emrys::withJacobian(turnAttitude(in, out, x, dt), jacobian);
            },
            reads(path<Attitude>, path<GyroBias>, path<GyroRate>),
            Emrys::writes(path<Attitude>),
            Emrys::continuousNoise(Emrys::on<Attitude>(1e-4))
        );
    }
};

template <typename Turn>
auto
makeGyroModel()
{
    return compose(Turn::make(), Emrys::RandomWalk<GyroBias>{1e-6});
}

inline GaussianParams<GyroState>
makeGyroInitial()
{
    auto mean = Eigen::Matrix<double, 7, 1>{};
    mean << EmrysTest::UnequalDimensions::turned, 0.02, -0.01, 0.03;
    auto spread = Eigen::Matrix<double, 6, 6>{};
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            spread(i, j) = 0.02 * (i + 1) - 0.015 * j + (i == j ? 0.1 : 0.0);
        }
    }
    Eigen::Matrix<double, 6, 6> const cov =
        spread * spread.transpose() + 1e-4 * Eigen::Matrix<double, 6, 6>::Identity();
    return GaussianParams<GyroState>{mean, cov};
}

// The model of a measured body direction, the axis body in the body frame of the attitude at Path.
// iterated.cpp calls its function directly, so it deduces its scalar rather than taking it as a
// template parameter.
template <typename Path = Orientation>
auto
makeDirectionModel(Eigen::Vector3d const & body)
{
    return measurementModel(
        [body](auto const & in, auto const &, auto const & x) {
            using T = ScalarOf<decltype(x)>;
            return EmrysTest::rotate(Eigen::Matrix<T, 4, 1>{block<Path>(in, x)}, body).eval();
        },
        reads(path<Path>),
        additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 0.001 * Eigen::Matrix3d::Identity()})
    );
}

// A measured body direction, observed from an attitude turned from the initial one by a step that
// grows with k, so that the corrections are large enough for the reset to matter. Path is the
// attitude's block.
template <typename Method = methods::Linearized, typename Path = Orientation>
auto
makeDirection(int k, Method const & method = {})
{
    using namespace EmrysTest::UnequalDimensions;
    Eigen::Vector3d const body = k % 2 == 0 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitZ();
    Eigen::Vector3d const observed =
        rotate(multiply(turned, exponential(0.05 * k * Eigen::Vector3d{1.0, -0.5, 0.8})), body);
    return observe(method, makeDirectionModel<Path>(body), observed);
}

}  // namespace EmrysTest::FormFilters

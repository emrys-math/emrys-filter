// The unit quaternion space of Spaces/UnitQuaternion.h (roadmap step 15b, decisions.md [unit-quaternion]): its
// storage order, which is Eigen's (x, y, z, w); its body-frame plus() and minus(), their inverse
// relation, and the double cover; the exponential, logarithm, and right Jacobian; and map(), which
// views a block as an Eigen::Quaternion without copying it, by itself, by tag path, and by
// repetition, writable or const, and inside a process model and a measurement run by the Kalman
// filter under each method. Expected values come from Eigen's own AngleAxis and quaternion product,
// an implementation independent of the space's. The block a path names is found with block()
// (TagPath.h), which the step's accessor tests cover; here it feeds map().

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <array>
#include <numbers>
#include <type_traits>
#include <utility>

#include "Spaces/unit_quaternion.h"

using Emrys::block;
using Emrys::BlockRows;
using Emrys::compose;
using Emrys::Composite;
using Emrys::DifferenceSpace;
using Emrys::GaussianParams;
using Emrys::hasIdentityReset;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::minus;
using Emrys::plus;
using Emrys::Repeated;
using Emrys::Space;
using Emrys::TagPath;
using Emrys::UnitQuaternion;

namespace methods = Emrys::methods;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Position : Linear<3>
{};

struct Velocity : Linear<3>
{};

using Body = Composite<Position, Attitude, Velocity>;

struct Vehicle : Composite<Position, Attitude>
{};

struct Fleet : Repeated<Vehicle, 4>
{
    using Repeated::Repeated;
};

struct Attitudes : Repeated<Attitude, 4>
{
    using Repeated::Repeated;
};

static_assert(Space<UnitQuaternion>);
static_assert(Space<Body>);
static_assert(std::is_same_v<DifferenceSpace<UnitQuaternion>, Linear<3>>);
static_assert(!hasIdentityReset<UnitQuaternion>);
static_assert(
    Emrys::valueDimensionAtCompileTime<Body> == 10 && Emrys::valueDimensionAtCompileTime<DifferenceSpace<Body>> == 9
);

Eigen::Vector3d const axis = Eigen::Vector3d{1.0, 2.0, -1.0}.normalized();
Eigen::Quaterniond const turned{Eigen::AngleAxisd{0.9, axis}};

// The steps the inverse tests take: zero, below the series threshold, moderate, large, and 0.04 rad
// short of π, where the logarithm is least well conditioned.
std::array<Eigen::Vector3d, 5> const steps{
    Eigen::Vector3d::Zero(),
    Eigen::Vector3d{1e-7, -2e-7, 0.5e-7},
    Eigen::Vector3d{0.1, -0.2, 0.3},
    Eigen::Vector3d{1.0, 0.5, -0.7},
    (std::numbers::pi - 0.04) * Eigen::Vector3d{0.0, 0.6, 0.8}
};

// The coordinates are Eigen's coeffs(), (x, y, z, w): the exponential of a rotation vector is
// Eigen's AngleAxis quaternion coefficient for coefficient, and rotating through map() of the
// coordinates is Eigen's rotation matrix. 1e-15 is rounding in sin and cos.
TEST(UnitQuaternion, StoresEigensCoefficientOrder)
{
    Eigen::Vector4d const q = UnitQuaternion::exponential(0.9 * axis).coeffs();
    EXPECT_LT((q - turned.coeffs()).norm(), 1e-15);
    EXPECT_GT(q(3), 0.0);
    EXPECT_DOUBLE_EQ(q(3), std::cos(0.45));
    auto const u = Eigen::Vector3d{0.3, -1.0, 2.0};
    EXPECT_LT((UnitQuaternion::map(q) * u - turned.toRotationMatrix() * u).norm(), 1e-15);
    EXPECT_EQ(Eigen::Quaterniond::FromCoeffsScalarFirst(q(3), q(0), q(1), q(2)).coeffs(), q);
}

// identity() and coefficientsOf() write a mean in the space's order without counting where w goes:
// the identity is w = 1 last, and a quaternion's coefficients are its coeffs().
TEST(UnitQuaternion, BuildsValuesInItsOwnOrder)
{
    EXPECT_EQ(UnitQuaternion::identity(), Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    EXPECT_EQ(UnitQuaternion::identity<float>(), Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_EQ(UnitQuaternion::coefficientsOf(turned), turned.coeffs());
    Eigen::Vector4d const identity = UnitQuaternion::identity();
    EXPECT_EQ(UnitQuaternion::map(identity).coeffs(), Eigen::Quaterniond::Identity().coeffs());
}

// normalize() scales to unit norm and picks w >= 0, so q and -q give one value; minus() agrees on
// either sign of either argument, since they are the same rotation.
TEST(UnitQuaternion, NormalizesOntoOneHemisphere)
{
    Eigen::Vector4d q = -3.0 * turned.coeffs();
    Emrys::normalizeValues(UnitQuaternion{}, q);
    EXPECT_LT((q - turned.coeffs()).norm(), 1e-15);

    Eigen::Vector4d const p = Eigen::Quaterniond{Eigen::AngleAxisd{0.4, Eigen::Vector3d::UnitZ()}}.coeffs();
    Eigen::Vector3d const d = minus(UnitQuaternion{}, q, p);
    EXPECT_LT((minus(UnitQuaternion{}, Eigen::Vector4d{-q}, p) - d).norm(), 1e-15);
    EXPECT_LT((minus(UnitQuaternion{}, q, Eigen::Vector4d{-p}) - d).norm(), 1e-15);
}

// plus() applies a step in the body frame, x ⊗ Exp(d), which is Eigen's product with the AngleAxis
// of the step on the right; minus() inverts it at every step, near π included, where 1e-12 is the
// conditioning of atan2 there.
TEST(UnitQuaternion, PlusIsBodyFrameAndMinusInvertsIt)
{
    for (auto const & d : steps)
    {
        Eigen::Vector4d const moved = plus(UnitQuaternion{}, turned.coeffs(), d);
        Eigen::Quaterniond step = Eigen::Quaterniond::Identity();
        if (d.norm() > 0)
        {
            step = Eigen::AngleAxisd{d.norm(), d.normalized()};
        }
        Eigen::Quaterniond const expected = turned * step;
        EXPECT_LT(std::abs(std::abs(expected.coeffs().dot(moved)) - 1.0), 1e-15) << d.transpose();
        EXPECT_NEAR(moved.norm(), 1.0, 1e-15);
        EXPECT_GE(moved(3), 0.0);
        EXPECT_LT((minus(UnitQuaternion{}, moved, turned.coeffs()) - d).norm(), 1e-12) << d.transpose();
    }
}

// The logarithm takes the short way round: a rotation of 3π/2 about an axis comes back as π/2
// about the opposite one.
TEST(UnitQuaternion, LogarithmTakesTheShortWay)
{
    auto const far = Eigen::Quaterniond{Eigen::AngleAxisd{1.5 * std::numbers::pi, Eigen::Vector3d::UnitX()}};
    Eigen::Vector3d const d = UnitQuaternion::logarithm(far);
    EXPECT_LT((d - Eigen::Vector3d{-0.5 * std::numbers::pi, 0.0, 0.0}).norm(), 1e-15);
    EXPECT_LT((UnitQuaternion::logarithm(UnitQuaternion::exponential(d)) - d).norm(), 1e-15);
}

// The space's reset Jacobian is its closed-form right Jacobian, which the reset computed by
// automatic differentiation of the same plus() and minus() equals to rounding [covariance-reset], and which is
// also the derivative of plus() in its step: the change of x ⊞ (d + ε) seen from x ⊞ d, by central
// differences, to 1e-9 for a step h of 1e-6.
TEST(UnitQuaternion, ResetJacobianIsTheRightJacobian)
{
    for (auto const & d : steps)
    {
        Eigen::Matrix3d const closed = Emrys::resetJacobian(UnitQuaternion{}, turned.coeffs(), d);
        EXPECT_EQ(closed, UnitQuaternion::rightJacobian(d));
        Eigen::Matrix3d const automatic = Emrys::resetJacobian(EmrysTest::AutoDiffQuaternion{}, turned.coeffs(), d);
        EXPECT_LT((automatic - closed).norm(), 1e-12) << d.transpose();
    }
    Eigen::Vector3d const d = steps[3];
    Eigen::Vector4d const moved = plus(UnitQuaternion{}, turned.coeffs(), d);
    double const h = 1e-6;
    for (int j = 0; j < 3; ++j)
    {
        Eigen::Vector3d const e = h * Eigen::Vector3d::Unit(j);
        Eigen::Vector4d const ahead = plus(UnitQuaternion{}, turned.coeffs(), Eigen::Vector3d{d + e});
        Eigen::Vector4d const behind = plus(UnitQuaternion{}, turned.coeffs(), Eigen::Vector3d{d - e});
        Eigen::Vector3d const column =
            (minus(UnitQuaternion{}, ahead, moved) - minus(UnitQuaternion{}, behind, moved)) / (2 * h);
        EXPECT_LT((column - UnitQuaternion::rightJacobian(d).col(j)).norm(), 1e-9);
    }
}

// map() of a vector, a const vector, and a column of a matrix is a map over the same memory:
// writable where the argument is, const where it is not, and writing through it writes the vector.
TEST(UnitQuaternion, MapViewsTheBlockWithoutCopying)
{
    Eigen::Vector4d q = turned.coeffs();
    auto writable = UnitQuaternion::map(q);
    static_assert(std::is_same_v<decltype(writable), Eigen::Map<Eigen::Quaterniond>>);
    EXPECT_EQ(writable.coeffs().data(), q.data());
    writable = Eigen::Quaterniond::Identity();
    EXPECT_EQ(q, Eigen::Vector4d::UnitW());

    Eigen::Vector4d const fixed = turned.coeffs();
    auto const view = UnitQuaternion::map(fixed);
    static_assert(std::is_same_v<std::remove_const_t<decltype(view)>, Eigen::Map<Eigen::Quaterniond const>>);
    EXPECT_EQ(view.coeffs().data(), fixed.data());

    Eigen::Matrix<double, 4, 3> states = Eigen::Matrix<double, 4, 3>::Zero();
    UnitQuaternion::map(states.col(1)) = turned;
    EXPECT_EQ(Eigen::Vector4d{states.col(1)}, turned.coeffs());
    EXPECT_EQ(states.col(0), Eigen::Vector4d::Zero());

    Eigen::Matrix<double, 4, 3> const constStates = states;
    static_assert(
        std::is_same_v<decltype(UnitQuaternion::map(constStates.col(1))), Eigen::Map<Eigen::Quaterniond const>>
    );
    EXPECT_EQ(UnitQuaternion::map(constStates.col(1)).coeffs(), turned.coeffs());
}

// map(block<Path>(space, x)) views the block found by tag path in a whole state, after a position's
// three coordinates, and map(block<Path>(space, x, element)) one repetition of a Repeated block
// after a position, the repeated block itself or the Attitude of a repeated Composite; each is a
// view at the block's own address, writable where the state is.
TEST(UnitQuaternion, MapViewsTheBlockFoundByTagPath)
{
    auto x = Eigen::Matrix<double, 10, 1>{};
    x << 1.0, 2.0, 3.0, turned.coeffs(), 0.1, 0.2, 0.3;
    auto const attitude = UnitQuaternion::map(block<Attitude>(Body{}, x));
    EXPECT_EQ(attitude.coeffs().data(), x.data() + 3);
    EXPECT_TRUE(attitude.isApprox(turned));
    UnitQuaternion::map(block<Body, Attitude>(x)) = Eigen::Quaterniond::Identity();
    EXPECT_EQ(Eigen::Vector4d{x.segment<4>(3)}, Eigen::Vector4d::UnitW());
    static_assert(std::is_same_v<
                  decltype(UnitQuaternion::map(block<Attitude>(Body{}, std::as_const(x)))),
                  Eigen::Map<Eigen::Quaterniond const>>);

    using Formation = Composite<Position, Attitudes>;
    auto const formation = Formation{Position{}, Attitudes{Attitude{}, 3}};
    Eigen::Matrix<double, 15, 1> const values = Eigen::Matrix<double, 15, 1>::Zero();
    EXPECT_EQ(UnitQuaternion::map(block<Attitudes>(formation, values, 2)).coeffs().data(), values.data() + 11);

    using Convoy = Composite<Position, Fleet>;
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 3}};
    Eigen::Matrix<double, 24, 1> vehicles = Eigen::Matrix<double, 24, 1>::Zero();
    auto const second = UnitQuaternion::map(block<TagPath<Fleet, Attitude>>(convoy, vehicles, 1));
    EXPECT_EQ(second.coeffs().data(), vehicles.data() + 13);
}

// A process model that turns the attitude at Path by a constant body rate, writing through map(),
// in the scalar type of its input so that it can be differentiated.
template <typename Path>
auto
bodyRate(Eigen::Vector3d const & rate, double spectralDensity)
{
    using Tag = Emrys::internal::PathTag<Path>;
    return Emrys::processModel(
        [rate]<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Eigen::Matrix<T, 3, 1> const turn = (dt * rate).template cast<T>();
            UnitQuaternion::map(block<Tag>(out, y)) =
                UnitQuaternion::map(block<Tag>(in, x)) * UnitQuaternion::exponential(turn);
            return y;
        },
        Emrys::reads(Emrys::path<Path>),
        Emrys::writes(Emrys::path<Path>),
        Emrys::continuousNoise(Emrys::on<Tag>(spectralDensity))
    );
}

// A measured direction in the world frame of a body-frame vector, predicted through map() of the
// attitude block the measurement is handed.
template <typename Method>
auto
direction(Method method, Eigen::Vector3d const & observed, Eigen::Vector3d const & body)
{
    auto const model = Emrys::measurementModel(
        [body]<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, 3, 1> const v = body.cast<T>();
            return Eigen::Matrix<T, 3, 1>{UnitQuaternion::map(Emrys::block<Attitude>(in, x)) * v};
        },
        Emrys::reads(Emrys::path<Attitude>),
        Emrys::additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 0.001 * Eigen::Matrix3d::Identity()})
    );
    return observe(method, model, observed);
}

// The same step with the tests' copying helpers, as the reference: a measurement that rotates a
// copy of its block.
auto
copiedDirection(Eigen::Vector3d const & observed, Eigen::Vector3d const & body)
{
    auto const model = Emrys::measurementModel(
        [body]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 3, 1>{
                EmrysTest::rotate(Eigen::Matrix<T, 4, 1>{Emrys::block<Attitude>(in, x)}, body)
            };
        },
        Emrys::reads(Emrys::path<Attitude>),
        Emrys::additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 0.001 * Eigen::Matrix3d::Identity()})
    );
    return observe(methods::linearized(), model, observed);
}

// A Kalman filter whose process block and measurement work through map(), under Linearized (whose
// automatic differentiation runs map() over dual scalars) and Unscented (over sigma points): the
// prediction turns the attitude by Exp(ω dt) exactly, and the update matches the same filter
// written with copies to 1e-12, rounding in two orders of the same arithmetic.
TEST(UnitQuaternion, KalmanFilterRunsModelsThroughMap)
{
    auto mean = Eigen::Matrix<double, 10, 1>{};
    mean << 1.0, 2.0, 3.0, turned.coeffs(), 0.1, 0.2, 0.3;
    Eigen::Matrix<double, 9, 9> const cov = 0.05 * Eigen::Matrix<double, 9, 9>::Identity();
    auto const rate = Eigen::Vector3d{0.2, -0.1, 0.4};
    auto const model = compose(bodyRate<Attitude>(rate, 0.01), Emrys::ConstantVelocity<Position, Velocity>{0.1});

    Eigen::Vector3d const body = Eigen::Vector3d::UnitX();
    Eigen::Quaterniond const truth = turned * Eigen::Quaterniond{Eigen::AngleAxisd{0.05, Eigen::Vector3d::UnitY()}};
    Eigen::Vector3d const observed = truth * body;

    auto linearized = KalmanFilter{model, GaussianParams<Body>{mean, cov}};
    auto unscented = KalmanFilter{model, GaussianParams<Body>{mean, cov}, methods::unscented()};
    auto copied = KalmanFilter{model, GaussianParams<Body>{mean, cov}};
    linearized.predict(0.5);
    copied.predict(0.5);
    unscented.predict(0.5);

    Eigen::Quaterniond const turnedOn = turned * UnitQuaternion::exponential(Eigen::Vector3d{0.5 * rate});
    EXPECT_TRUE(UnitQuaternion::map(block<Attitude>(Body{}, linearized.state().mean())).isApprox(turnedOn, 1e-14));
    EXPECT_TRUE(UnitQuaternion::map(block<Attitude>(Body{}, unscented.state().mean())).isApprox(turnedOn, 1e-12));

    ASSERT_TRUE(linearized.update(direction(methods::linearized(), observed, body)).value().accepted);
    ASSERT_TRUE(copied.update(copiedDirection(observed, body)).value().accepted);
    ASSERT_TRUE(unscented.update(direction(methods::unscented(), observed, body)).value().accepted);
    EXPECT_TRUE(linearized.state().mean().isApprox(copied.state().mean(), 1e-12));
    EXPECT_TRUE(linearized.state().covarianceLower().isApprox(copied.state().covarianceLower(), 1e-12));
    EXPECT_NEAR(unscented.state().mean().segment<4>(3).norm(), 1.0, 1e-14);
}

}  // namespace

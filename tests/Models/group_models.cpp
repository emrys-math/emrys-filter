// The models on a Lie group whose Jacobians are formed in closed form (docs/guide/groups.md, open question 8; a
// prototype; Models/GroupModels.h): bodyVelocity(), observedInWorld(), and observedInBody(). Each is checked against
// the same model written as a plain function and differentiated automatically, and each measurement's carry against
// the one named by hand. The error Jacobian of bodyVelocity() is the same at every estimate.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/GroupModels.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/ExtendedPose.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

using Emrys::block;
using Emrys::Composite;
using Emrys::ExtendedPose;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::path;
using Emrys::UnitQuaternion;

namespace carries = Emrys::carries;
namespace methods = Emrys::methods;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Velocity : Linear<3>
{};

struct Position : Linear<3>
{};

struct Bias : Linear<3>
{};

struct Twist : Linear<9>
{};

struct Fix : Linear<3>
{};

struct Body : ExtendedPose<Attitude, Velocity, Position>
{};

using State = Composite<Body, Bias>;
using StateVector = Emrys::ValueVector<State>;
using StateMatrix = Emrys::DifferenceMatrix<State>;

StateVector
priorMean(double scale = 1.0)
{
    StateVector x;
    x << UnitQuaternion::exp(scale * Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 0.5, 1.0, -0.2, 1.5 * scale, -2.0, 0.5,
        0.01, -0.02, 0.03;
    return x;
}

StateMatrix
priorCovariance()
{
    StateMatrix a = StateMatrix::Identity();
    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            a(i, j) += 0.05 * std::sin(1.0 + 3.0 * i + 7.0 * j);
        }
    }
    return 0.3 * a * a.transpose();
}

StateMatrix
full(GaussianParams<State> const & g)
{
    return g.covarianceLower().selfadjointView<Eigen::Lower>();
}

Eigen::Matrix3d
noise()
{
    return Eigen::Vector3d{0.04, 0.09, 0.02}.asDiagonal();
}

TEST(GroupModels, BodyVelocityHasTheSameJacobianAtEveryEstimate)
{
    Eigen::Matrix<double, 9, 1> twist;
    twist << 0.3, -0.2, 0.5, 1.0, 0.5, -0.4, 0.2, 0.1, -0.3;
    Eigen::Matrix<double, 9, 9> const twistNoise = 0.01 * Eigen::Matrix<double, 9, 9>::Identity();
    auto const motion = Emrys::blockwise(Emrys::bodyVelocity<Body, Twist>(), Emrys::RandomWalk<Bias>{0.001});
    double const dt = 0.5;
    Eigen::Matrix<double, 9, 9> const f = Emrys::adjoint<Body>(Emrys::exp<Body>((-dt * twist).eval()));
    Eigen::Matrix<double, 9, 9> const g = dt * Emrys::rightJacobian<Body>((dt * twist).eval());
    for (double const scale : {1.0, -2.0})
    {
        auto const prior = GaussianParams<State>{priorMean(scale), priorCovariance()};
        auto const predicted =
            Emrys::predict(methods::linearized(), prior, motion, dt, GaussianParams<Twist>{twist, twistNoise});
        StateMatrix expected = priorCovariance();
        expected.topLeftCorner<9, 9>() = f * priorCovariance().topLeftCorner<9, 9>() * f.transpose();
        expected.topRightCorner<9, 3>() = f * priorCovariance().topRightCorner<9, 3>();
        expected.bottomLeftCorner<3, 9>() = expected.topRightCorner<9, 3>().transpose();
        expected.topLeftCorner<9, 9>() += g * twistNoise * g.transpose();
        expected.bottomRightCorner<3, 3>() += 0.001 * dt * Eigen::Matrix3d::Identity();
        EXPECT_LT((full(predicted) - expected).norm(), 1e-12) << scale;
    }
}

TEST(GroupModels, ObservedInBodyMatchesTheFunctionAndCarriesByTheAdjoint)
{
    Eigen::Vector3d const landmark{4.0, 1.0, -2.0};
    auto const sight = Emrys::observedInBody<Body>(
        landmark,
        std::tuple{Emrys::on<Position>(1.0)},
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(noise())
    );
    auto const byFunction = Emrys::measurementModel(
        [landmark]<typename T>(auto const & in, auto const & x) {
            Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
            return Eigen::Matrix<T, 3, 1>{q.conjugate() * (landmark - block<Position>(in, x)).eval()};
        },
        Emrys::reads(path<Attitude>, path<Position>),
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(noise())
    );
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    Eigen::Vector3d const z{1.0, 2.5, -3.0};
    // No method and no carry named: linearized, with the adjoint carry.
    auto const updated = Emrys::update(prior, Emrys::observe(sight, z));
    auto const expected =
        Emrys::update(prior, Emrys::observe(methods::linearized(), byFunction, z, carries::adjoint()));
    auto const reset = Emrys::update(prior, Emrys::observe(sight, z, carries::reset()));
    ASSERT_TRUE(updated.has_value() && expected.has_value() && reset.has_value());
    EXPECT_LT(
        Emrys::minus<State>(updated->branches[0].posterior.mean(), expected->branches[0].posterior.mean()).norm(),
        1e-11
    );
    EXPECT_LT((full(updated->branches[0].posterior) - full(expected->branches[0].posterior)).norm(), 1e-11);
    // A carry named is taken instead.
    EXPECT_GT((full(updated->branches[0].posterior) - full(reset->branches[0].posterior)).norm(), 1e-4);
}

TEST(GroupModels, ObservedInWorldMatchesTheFunctionAndCarriesNothing)
{
    Eigen::Vector3d const arm{0.3, -0.1, 0.2};
    auto const fix = Emrys::observedInWorld<Body>(
        arm,
        std::tuple{Emrys::on<Position>(1.0)},
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(noise())
    );
    auto const byFunction = Emrys::measurementModel(
        [arm]<typename T>(auto const & in, auto const & x) {
            Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
            return Eigen::Matrix<T, 3, 1>{block<Position>(in, x) + q * arm};
        },
        Emrys::reads(path<Attitude>, path<Position>),
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(noise())
    );
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    Eigen::Vector3d const z{2.5, -1.0, 1.2};
    auto const updated = Emrys::update(prior, Emrys::observe(fix, z));
    auto const expected = Emrys::update(prior, Emrys::observe(methods::linearized(), byFunction, z, carries::none()));
    ASSERT_TRUE(updated.has_value() && expected.has_value());
    EXPECT_LT(
        Emrys::minus<State>(updated->branches[0].posterior.mean(), expected->branches[0].posterior.mean()).norm(),
        1e-11
    );
    EXPECT_LT((full(updated->branches[0].posterior) - full(expected->branches[0].posterior)).norm(), 1e-11);

    // A velocity fix: the velocity alone, with no point.
    auto const speed = Emrys::observedInWorld<Body>(
        Eigen::Vector3d{Eigen::Vector3d::Zero()},
        std::tuple{Emrys::on<Velocity>(1.0)},
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(noise())
    );
    auto const velocityByMatrix = Emrys::measurementModel(
        Eigen::Matrix3d::Identity(),
        Emrys::reads(path<Velocity>),
        Emrys::additiveNoise(noise())
    );
    auto const bySpeed = Emrys::update(prior, Emrys::observe(speed, z));
    auto const byMatrix = Emrys::update(prior, Emrys::observe(velocityByMatrix, z, carries::none()));
    ASSERT_TRUE(bySpeed.has_value() && byMatrix.has_value());
    EXPECT_LT((full(bySpeed->branches[0].posterior) - full(byMatrix->branches[0].posterior)).norm(), 1e-11);
}

}  // namespace

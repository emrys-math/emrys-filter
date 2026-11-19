// An ExtendedPose whose size is known at run time (docs/guide/groups.md, section 9 and open
// question 3; a prototype): a body with a Repeated block of feet among its vectors. A foot is inserted as a function
// of the pose and a leg's kinematics by insert<Path>(state, element, method, model, inputs...) (Models/Insertion.h),
// a model placed on each foot predicts beside the body's, a measurement reads the feet, and a foot is erased.
// Expected values are written out in plain Eigen from the pose's rows: a foot's rows are its step in the body frame.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Insertion.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
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
using Emrys::Repeated;
using Emrys::UnitQuaternion;

namespace methods = Emrys::methods;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Position : Linear<3>
{};

struct Foot : Linear<3>
{};

struct Feet : Repeated<Foot>
{
    using Repeated::Repeated;
};

struct Bias : Linear<3>
{};

struct Leg : Linear<3>
{};

struct Body : ExtendedPose<Attitude, Position, Feet>
{
    using ExtendedPose::ExtendedPose;
};

struct State : Composite<Body, Bias>
{
    using Composite::Composite;
};

Eigen::Matrix3d
skew(Eigen::Vector3d const & v)
{
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return m;
}

State
spaceWith(Eigen::Index feet)
{
    return State{Body{Attitude{}, Position{}, Feet{Foot{}, feet}}, Bias{}};
}

GaussianParams<State>
prior()
{
    Eigen::VectorXd mean(10);
    mean << UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 1.5, -2.0, 0.5, 0.01, -0.02, 0.03;
    Eigen::MatrixXd a = Eigen::MatrixXd::Identity(9, 9);
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            a(i, j) += 0.05 * std::sin(1.0 + 3.0 * i + 7.0 * j);
        }
    }
    return GaussianParams<State>{spaceWith(0), mean, Eigen::MatrixXd{0.3 * a * a.transpose()}};
}

// A foot at the end of a leg: the body's position plus the leg's vector, given in the body frame, turned to the world.
auto
placedFoot()
{
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
            Eigen::Matrix<T, 3, 1> const leg = block<Leg>(in, x);
            return Eigen::Matrix<T, 3, 1>{block<Position>(in, x) + q * leg};
        },
        Emrys::reads(path<Attitude>, path<Position>, path<Leg>),
        Emrys::outputSpace(Foot{})
    );
}

TEST(GroupRunTime, InsertAFootFromThePoseAndErase)
{
    auto state = prior();
    Eigen::Vector3d const leg{0.2, -0.1, -0.6};
    Eigen::Matrix3d const legNoise = Eigen::Vector3d{0.001, 0.002, 0.004}.asDiagonal();
    Eigen::MatrixXd const p = state.covarianceLower().selfadjointView<Eigen::Lower>();
    Eigen::Quaterniond const q{Eigen::Vector4d{state.mean().head<4>()}};
    Eigen::Vector3d const position = state.mean().segment<3>(4);

    Emrys::insert<Emrys::TagPath<Body, Feet>>(
        state,
        0,
        methods::linearized(),
        placedFoot(),
        GaussianParams<Leg>{leg, legNoise}
    );
    ASSERT_EQ(state.space().template block<0>().vectorCount(), 2);
    ASSERT_EQ(state.valueDimension(), 13);
    ASSERT_EQ(state.differenceDimension(), 12);
    EXPECT_TRUE(block<Feet>(state.space(), state.mean(), 0).isApprox(position + q * leg));

    // In the pose's rows the foot's step is -leg^ phi + rho_p + l with l the leg's own error: no rotation in it.
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, 9);
    h.leftCols<3>() = -skew(leg);
    h.middleCols<3>(3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd const cov = state.covarianceLower().selfadjointView<Eigen::Lower>();
    // The foot sits after the position: rows 6 to 8, before the bias.
    EXPECT_LT((cov.block<3, 3>(6, 6) - (h * p * h.transpose() + legNoise)).norm(), 1e-12);
    Eigen::MatrixXd expectedCross(3, 9);
    expectedCross << (h * p).leftCols<6>(), (h * p).rightCols<3>();
    Eigen::MatrixXd cross(3, 9);
    cross << cov.block<3, 6>(6, 0), cov.block<3, 3>(6, 9);
    EXPECT_LT((cross - expectedCross).norm(), 1e-12);

    // The group's members follow the count.
    Eigen::VectorXd const moved = Emrys::plus(state.space(), state.mean(), Eigen::VectorXd::Constant(12, 0.1));
    EXPECT_EQ(moved.size(), 13);
    EXPECT_EQ(Emrys::adjoint(state.space(), state.mean()).rows(), 12);

    state.erase<Emrys::TagPath<Body, Feet>>(0);
    EXPECT_EQ(state.valueDimension(), 10);
    Eigen::MatrixXd const erased = state.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((erased - p).norm(), 1e-12);
}

TEST(GroupRunTime, PredictAndUpdateWithFeet)
{
    auto state = prior();
    Eigen::Matrix3d const legNoise = 0.002 * Eigen::Matrix3d::Identity();
    for (Eigen::Vector3d const & leg : {Eigen::Vector3d{0.2, -0.1, -0.6}, Eigen::Vector3d{-0.2, 0.1, -0.6}})
    {
        Emrys::insert<Emrys::TagPath<Body, Feet>>(
            state,
            state.space().template block<0>().vectorCount() - 1,
            methods::linearized(),
            placedFoot(),
            GaussianParams<Leg>{leg, legNoise}
        );
    }
    ASSERT_EQ(state.differenceDimension(), 15);

    // The body turns at a known rate and holds its position; each foot and the bias walk at random.
    Eigen::Vector3d const rate{0.3, -0.5, 0.8};
    auto const turning = Emrys::processModel(
        [rate]<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
            auto y = Emrys::zeros<T>(out);
            block<Attitude>(out, y) = (q * UnitQuaternion::exp((dt * rate).eval()).template cast<T>()).coeffs();
            block<Position>(out, y) = block<Position>(in, x);
            return y;
        },
        Emrys::reads(path<Attitude>, path<Position>),
        Emrys::writes(path<Attitude>, path<Position>),
        Emrys::continuousNoise(Emrys::on<Attitude>(0.01 * Eigen::Matrix3d::Identity()))
    );
    auto const motion = Emrys::blockwise(
        turning,
        Emrys::onEach<Emrys::TagPath<Body, Feet>>(Emrys::RandomWalk<Foot>{0.05}),
        Emrys::RandomWalk<Bias>{0.001}
    );
    double const dt = 0.4;
    Eigen::MatrixXd const before = state.covarianceLower().selfadjointView<Eigen::Lower>();
    auto const predicted = Emrys::predict(methods::linearized(), state, motion, dt);
    Eigen::MatrixXd const after = predicted.covarianceLower().template selfadjointView<Eigen::Lower>();
    // Each foot stays where it was in the world, so its rows turn from the old body frame to the new.
    Eigen::Matrix3d const turn = UnitQuaternion::exp((dt * rate).eval()).toRotationMatrix();
    for (int foot = 0; foot < 2; ++foot)
    {
        int const r = 6 + 3 * foot;
        Eigen::Matrix3d const expected =
            turn.transpose() * before.block<3, 3>(r, r) * turn + 0.05 * dt * Eigen::Matrix3d::Identity();
        EXPECT_LT((after.block<3, 3>(r, r) - expected).norm(), 1e-12);
    }

    // The feet seen from the body through the legs: y_i = R^T (f_i - p), which reads the whole run of feet.
    auto const kinematics = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            Eigen::Quaternion<T> const q{UnitQuaternion::map(block<Attitude>(in, x))};
            auto const feet = block<Feet>(in, x);
            Eigen::Matrix<T, Eigen::Dynamic, 1> y(feet.size());
            for (Eigen::Index i = 0; i < feet.size() / 3; ++i)
            {
                y.template segment<3>(3 * i) =
                    q.conjugate() * (feet.template segment<3>(3 * i) - block<Position>(in, x)).eval();
            }
            return y;
        },
        Emrys::reads(path<Attitude>, path<Position>, path<Body, Feet>),
        Emrys::outputSpace(Linear<Eigen::Dynamic>{6}),
        Emrys::additiveNoise(Eigen::MatrixXd{0.01 * Eigen::MatrixXd::Identity(6, 6)})
    );
    Eigen::VectorXd z(6);
    z << 0.25, -0.1, -0.55, -0.2, 0.15, -0.62;
    auto const updated =
        Emrys::update(predicted, Emrys::observe(methods::linearized(), kinematics, z, Emrys::carries::adjoint()));
    ASSERT_TRUE(updated.has_value());
    Eigen::MatrixXd const posterior = updated->branches[0].posterior.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT(posterior.trace(), after.trace());
    EXPECT_TRUE(updated->branches[0].posterior.mean().allFinite());
}

}  // namespace

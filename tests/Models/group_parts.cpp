// Models that read or write a part of an ExtendedPose (docs/design/groups.md, open question 1; a prototype): a
// measurement that reads the position alone, as a function and as a matrix; and process models side by side that
// each write some parts of one pose, one of them returning its Jacobian. A part's rows of the covariance are steps
// in the pose's body frame, so a model that names a part sees the part's own values and differences, and the
// library turns between the two. Expected values come from the same models written over the whole state, which
// reads and writes the pose as one block and is differentiated through the pose's own plus() and minus().

#include <EmrysFilter/Filters/KalmanFilter.h>
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

namespace methods = Emrys::methods;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Velocity : Linear<3>
{};

struct Position : Linear<3>
{};

struct Foot : Linear<3>
{};

struct Bias : Linear<3>
{};

struct Fix : Linear<3>
{};

struct Body : ExtendedPose<Attitude, Velocity, Position, Foot>
{};

using State = Composite<Body, Bias>;

using StateVector = Emrys::ValueVector<State>;
using StateMatrix = Emrys::DifferenceMatrix<State>;

Eigen::Vector3d const rate{0.3, -0.5, 0.8};  // the body's turn rate, known

StateVector
priorMean()
{
    StateVector x;
    x << UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 0.5, 1.0, -0.2, 1.5, -2.0, 0.5, 2.0, 0.3, -1.0,
        0.01, -0.02, 0.03;
    return x;
}

StateMatrix
priorCovariance()
{
    StateMatrix a = StateMatrix::Identity();
    for (int i = 0; i < 15; ++i)
    {
        for (int j = 0; j < 15; ++j)
        {
            a(i, j) += 0.05 * std::sin(1.0 + 3.0 * i + 7.0 * j);
        }
    }
    return 0.3 * a * a.transpose();
}

Eigen::Matrix3d
fixNoise()
{
    return Eigen::Vector3d{0.04, 0.09, 0.02}.asDiagonal();
}

template <typename A, typename B>
void
expectSame(A const & a, B const & b, double tolerance)
{
    EXPECT_LT(Emrys::minus<State>(a.mean(), b.mean()).norm(), tolerance);
    StateMatrix const ca = a.covarianceLower().template selfadjointView<Eigen::Lower>();
    StateMatrix const cb = b.covarianceLower().template selfadjointView<Eigen::Lower>();
    EXPECT_LT((ca - cb).norm(), tolerance);
}

static_assert(Emrys::internal::posePartOf<State, Position> == Emrys::internal::PosePart::vectors);
static_assert(
    Emrys::internal::posePartOf<State, Emrys::TagPath<Body, Attitude>> == Emrys::internal::PosePart::rotation
);
static_assert(Emrys::internal::posePartOf<State, Body> == Emrys::internal::PosePart::none);
static_assert(Emrys::internal::posePartOf<State, Bias> == Emrys::internal::PosePart::none);
static_assert(Emrys::internal::posePartOf<Body, Foot> == Emrys::internal::PosePart::vectors);

// A position fix that reads the position alone, against one that reads the whole state.
TEST(GroupParts, MeasurementOfOnePartMatchesTheWholeStateModel)
{
    auto const byPart = Emrys::measurementModel(
        [](auto const & in, auto const & x) {
            return block<Position>(in, x).eval();
        },
        Emrys::reads(path<Position>),
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(fixNoise())
    );
    auto const whole = Emrys::measurementModel(
        [](auto const & x) {
            return x.template segment<3>(7).eval();
        },
        Emrys::outputSpace(Fix{}),
        Emrys::additiveNoise(fixNoise())
    );
    auto const byMatrix = Emrys::measurementModel(
        Eigen::Matrix3d::Identity(),
        Emrys::reads(path<Body, Position>),
        Emrys::additiveNoise(fixNoise())
    );
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    Eigen::Vector3d const z{2.5, -1.0, 1.2};

    auto const expected = Emrys::update(prior, Emrys::observe(methods::linearized(), whole, z));
    auto const part = Emrys::update(prior, Emrys::observe(methods::linearized(), byPart, z));
    auto const matrix = Emrys::update(prior, Emrys::observe(byMatrix, z));
    ASSERT_TRUE(expected.has_value() && part.has_value() && matrix.has_value());
    expectSame(part->branches[0].posterior, expected->branches[0].posterior, 1e-11);
    expectSame(matrix->branches[0].posterior, expected->branches[0].posterior, 1e-11);
    // The posterior moved, and not as a Composite of the same blocks would move it: the fix corrects the attitude
    // through the position's body-frame rows.
    EXPECT_GT(Emrys::minus<State>(part->branches[0].posterior.mean(), priorMean()).norm(), 0.1);

    // Under sigma points the part's read spreads over the pose, as the whole-state model's does over the state.
    auto const expectedPoints = Emrys::update(prior, Emrys::observe(methods::unscented(), whole, z));
    auto const partPoints = Emrys::update(prior, Emrys::observe(methods::unscented(), byPart, z));
    ASSERT_TRUE(expectedPoints.has_value() && partPoints.has_value());
    EXPECT_LT(
        Emrys::minus<State>(partPoints->branches[0].posterior.mean(), expectedPoints->branches[0].posterior.mean())
            .norm(),
        0.05
    );
}

// The attitude turned by a known rate, returning its Jacobian over the attitude's own differences.
auto
turning()
{
    return Emrys::processModel(
        [](auto const & in, auto const & out, auto const & x, double dt) {
            Eigen::Quaterniond const step = UnitQuaternion::exp((dt * rate).eval());
            Eigen::Quaterniond const q{UnitQuaternion::map(block<Attitude>(in, x))};
            auto y = Emrys::zeros<double>(out);
            block<Attitude>(out, y) = (q * step).coeffs();
            Eigen::Matrix3d const jacobian = step.toRotationMatrix().transpose();
            return Emrys::withJacobian(y, jacobian);
        },
        Emrys::reads(path<Attitude>),
        Emrys::writes(path<Attitude>),
        Emrys::continuousNoise(0.01 * Eigen::Matrix3d::Identity())
    );
}

// The same steps over the whole state: the attitude turns, the position moves with the velocity, the foot and the
// bias stay.
auto
wholeStateMotion()
{
    StateMatrix density = StateMatrix::Zero();
    density.diagonal() << 0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.001, 0.001, 0.001;
    return Emrys::processModel(
        []<typename T>(auto const & x, double dt) {
            Eigen::Matrix<T, 16, 1> y = x;
            Eigen::Quaternion<T> const q{x(3), x(0), x(1), x(2)};
            Eigen::Quaternion<T> const step = UnitQuaternion::exp((dt * rate).eval()).template cast<T>();
            y.template head<4>() = (q * step).coeffs();
            y.template segment<3>(7) = x.template segment<3>(7) + dt * x.template segment<3>(4);
            return y;
        },
        Emrys::continuousNoise(density)
    );
}

// Separate models write the attitude, the velocity and position, the foot, and the bias. Each sees its own blocks:
// the foot's random walk has the identity as its Jacobian, and in the pose's rows it is R'^T R.
TEST(GroupParts, ModelsSideBySideWriteThePartsOfOnePose)
{
    auto const sideBySide = Emrys::blockwise(
        turning(),
        Emrys::ConstantVelocity<Position, Velocity>{0.2},
        Emrys::RandomWalk<Foot>{0.05},
        Emrys::RandomWalk<Bias>{0.001}
    );
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    double const dt = 0.4;
    auto const expected = Emrys::predict(methods::linearized(), prior, wholeStateMotion(), dt);
    auto const predicted = Emrys::predict(methods::linearized(), prior, sideBySide, dt);
    // The velocity's noise drives the position within the step, which the whole-state model's discretization
    // of a diagonal density leaves out: compare the means and the covariance apart from that block.
    EXPECT_LT(Emrys::minus<State>(predicted.mean(), expected.mean()).norm(), 1e-12);
    StateMatrix const a = predicted.covarianceLower().selfadjointView<Eigen::Lower>();
    StateMatrix b = expected.covarianceLower().selfadjointView<Eigen::Lower>();
    // Q of a constant-velocity model over dt, in the pose's rows: the same in the body frame, being isotropic.
    b.block<3, 3>(3, 3) += Eigen::Matrix3d::Zero();
    b.block<3, 3>(6, 6) += 0.2 * dt * dt * dt / 3 * Eigen::Matrix3d::Identity();
    b.block<3, 3>(3, 6) += 0.2 * dt * dt / 2 * Eigen::Matrix3d::Identity();
    b.block<3, 3>(6, 3) += 0.2 * dt * dt / 2 * Eigen::Matrix3d::Identity();
    EXPECT_LT((a - b).norm(), 1e-11);

    // The foot's rows against the attitude's: the turn from the old body frame to the new.
    StateMatrix const p = priorCovariance();
    Eigen::Matrix3d const turn = UnitQuaternion::exp((dt * rate).eval()).toRotationMatrix();
    Eigen::Matrix3d const footCov = a.block<3, 3>(9, 9);
    Eigen::Matrix3d const footExpected =
        turn.transpose() * p.block<3, 3>(9, 9) * turn + 0.05 * dt * Eigen::Matrix3d::Identity();
    EXPECT_LT((footCov - footExpected).norm(), 1e-12);
}

// The marginal of a vector of a pose is over the smaller pose of the rotation and that vector, with the rotation's
// rows and the vector's (docs/design/groups.md, open question 2). The rotation's marginal is its own.
TEST(GroupParts, MarginalOfAVectorIsThePoseOfTheRotationAndTheVector)
{
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    auto const marginal = prior.marginal<Emrys::TagPath<Body, Position>>();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(marginal.space())>, ExtendedPose<Attitude, Position>>);
    Eigen::Matrix<double, 7, 1> const mean = marginal.mean();
    EXPECT_EQ(mean.head<4>(), prior.mean().head<4>());
    EXPECT_EQ(mean.tail<3>(), prior.mean().segment<3>(7));
    Eigen::Matrix<double, 6, 6> const cov = marginal.covarianceLower().template selfadjointView<Eigen::Lower>();
    StateMatrix const p = priorCovariance();
    EXPECT_LT((cov.topLeftCorner<3, 3>() - p.topLeftCorner<3, 3>()).norm(), 1e-15);
    EXPECT_LT((cov.bottomRightCorner<3, 3>() - p.block<3, 3>(6, 6)).norm(), 1e-15);
    EXPECT_LT((cov.bottomLeftCorner<3, 3>() - p.block<3, 3>(6, 0)).norm(), 1e-15);

    // Exact, not first order: a state moved by a difference has the smaller pose moved by that difference's rows.
    Emrys::DifferenceVector<State> step = Emrys::DifferenceVector<State>::LinSpaced(15, -0.8, 0.9);
    StateVector const moved = Emrys::plus<State>(prior.mean(), step);
    Eigen::Matrix<double, 6, 1> subStep;
    subStep << step.head<3>(), step.segment<3>(6);
    Eigen::Matrix<double, 7, 1> const subMoved = Emrys::plus(marginal.space(), mean, subStep);
    EXPECT_LT((subMoved.head<4>() - moved.head<4>()).norm(), 1e-14);
    EXPECT_LT((subMoved.tail<3>() - moved.segment<3>(7)).norm(), 1e-14);

    auto const attitude = prior.marginal<Attitude>();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(attitude.space())>, Attitude>);
}

// additiveNoise() placed by path on an output whose differences have other sizes than its values: the pose's 12
// rows and one of its parts' 3, each given as a covariance over the block's differences. A part's is in the pose's
// body frame, as its rows are.
TEST(GroupParts, AdditiveNoisePlacedOnAPoseAndOnAPart)
{
    Eigen::Matrix<double, 12, 12> const bodyNoise =
        Eigen::Matrix<double, 12, 1>::LinSpaced(12, 0.01, 0.12).asDiagonal();
    Eigen::Matrix3d const biasNoise = 0.002 * Eigen::Matrix3d::Identity();
    auto const hold = Emrys::processModel(
        [](auto const & x, double) {
            return x.eval();
        },
        Emrys::reads(path<Body>, path<Bias>),
        Emrys::writes(path<Body>, path<Bias>),
        Emrys::additiveNoise(Emrys::on<Body>(bodyNoise), Emrys::on<Bias>(biasNoise))
    );
    auto const prior = GaussianParams<State>{priorMean(), priorCovariance()};
    auto const predicted = Emrys::predict(methods::linearized(), prior, hold, 1.0);
    StateMatrix expected = priorCovariance();
    expected.topLeftCorner<12, 12>() += bodyNoise;
    expected.bottomRightCorner<3, 3>() += biasNoise;
    StateMatrix const cov = predicted.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((cov - expected).norm(), 1e-12);

    Eigen::Matrix3d const footNoise = Eigen::Vector3d{0.1, 0.2, 0.3}.asDiagonal();
    auto const holdFoot = Emrys::processModel(
        [](auto const & x, double) {
            return x.eval();
        },
        Emrys::reads(path<Body>, path<Bias>),
        Emrys::writes(path<Body>, path<Bias>),
        Emrys::additiveNoise(Emrys::on<Emrys::TagPath<Body, Foot>>(footNoise))
    );
    StateMatrix const withFoot =
        Emrys::predict(methods::linearized(), prior, holdFoot, 1.0).covarianceLower().selfadjointView<Eigen::Lower>();
    StateMatrix expectedFoot = priorCovariance();
    expectedFoot.block<3, 3>(9, 9) += footNoise;
    EXPECT_LT((withFoot - expectedFoot).norm(), 1e-12);
}

}  // namespace

// The covariance reset (roadmap step 13e, decisions.md [covariance-reset]): hasIdentityReset<>, resetJacobian(),
// and resetCovariance() in Spaces/Reset.h, and their use by the Kalman update (Models/Transform.h),
// the Gaussian conditional (Spaces/Indexing.h), and the smoother's step (Smoothers/GaussianHistory.h). The
// unit quaternion of tests/Spaces/unit_quaternion.h is the space with a reset, by automatic
// differentiation (AutoDiffQuaternion) and in closed form (UnitQuaternion).

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <array>
#include <cmath>
#include <numbers>
#include <utility>

#include "Spaces/unit_quaternion.h"

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::hasIdentityReset;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::resetCovariance;
using Emrys::resetJacobian;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;
using EmrysTest::AutoDiffQuaternion;
using EmrysTest::conjugate;
using EmrysTest::exponential;
using EmrysTest::logarithm;
using EmrysTest::multiply;
using EmrysTest::Quaternion;
using EmrysTest::rightJacobian;
using EmrysTest::rotate;
using EmrysTest::UnitQuaternion;

namespace
{

/** A positive gain, x ⊞ δ = x e^δ and a ⊟ b = log(a / b), whose reset Jacobian is exactly one and
 * which says so.
 */
struct Gain
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;
    static constexpr bool identityReset = true;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    [[nodiscard]] Linear<1>
    difference() const
    {
        return {};
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        Emrys::writable(out) = x.array() * d.array().exp();
    }

    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        Emrys::writable(out) = (a.array() / b.array()).log();
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(Gain const &, Gain const &) = default;
};

/** The same gain without the declaration, so its reset is differentiated. */
struct UndeclaredGain : Gain
{
    static constexpr bool identityReset = false;

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(UndeclaredGain const &, UndeclaredGain const &) = default;
};

struct Position : Linear<3>
{};

struct Attitude : AutoDiffQuaternion
{};

struct ClosedFormAttitude : UnitQuaternion
{};

struct Velocity : Linear<3>
{};

struct Heading : RadiansPi<>
{};

struct Range : Linear<1>
{};

using Body = Composite<Position, Attitude, Velocity>;
using ClosedFormBody = Composite<Position, ClosedFormAttitude, Velocity>;

struct Attitudes : Repeated<Attitude, 3>
{
    using Repeated::Repeated;
};

using Formation = Composite<Position, Attitudes>;

static_assert(hasIdentityReset<Linear<3>>);
static_assert(hasIdentityReset<Heading>);
static_assert(hasIdentityReset<Range>);
static_assert(hasIdentityReset<Composite<Position, Heading, Range>>);
static_assert(hasIdentityReset<Repeated<Composite<Heading, Range>>>);
static_assert(hasIdentityReset<Gain>);
static_assert(hasIdentityReset<Composite<Position, Gain>>);
static_assert(!hasIdentityReset<UndeclaredGain>);
static_assert(!hasIdentityReset<AutoDiffQuaternion>);
static_assert(!hasIdentityReset<UnitQuaternion>);
static_assert(!hasIdentityReset<Body>);
static_assert(!hasIdentityReset<Attitudes>);
static_assert(!hasIdentityReset<Formation>);

Quaternion<double> const turned = exponential(Eigen::Vector3d{0.3, -0.2, 0.5});

/** A positive definite matrix of size N with every entry nonzero, from a fixed square, exactly
 * symmetric: a product \f$ A A^T \f$ need not be, where the compiler fuses multiply-adds, as GCC
 * does on arm64.
 */
template <int N>
Eigen::Matrix<double, N, N>
spreadCov()
{
    auto a = Eigen::Matrix<double, N, N>{};
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            a(i, j) = std::sin(1.0 + 3.0 * i + 7.0 * j);
        }
    }
    Eigen::Matrix<double, N, N> const square = a * a.transpose();
    return 0.5 * (square + square.transpose()) + 0.5 * Eigen::Matrix<double, N, N>::Identity();
}

/** The mean of a Body at turned, with the position and velocity given. */
Eigen::Matrix<double, 10, 1>
bodyMean()
{
    auto mean = Eigen::Matrix<double, 10, 1>{};
    mean << 1.0, -2.0, 0.5, turned, 0.3, 0.1, -0.2;
    return mean;
}

}  // namespace

// The automatically differentiated reset Jacobian of the quaternion equals the closed-form right
// Jacobian J_r(d) of Solà, Deray, and Atchuthan (2018), eq. (143), at a zero step, a small one, a
// large one, and one 0.04 rad short of π, from two starting attitudes; the closed form the space
// writes is used as it is. 1e-12 is rounding in the trigonometric functions near π.
TEST(Reset, QuaternionJacobianIsTheRightJacobian)
{
    Eigen::Vector3d const axis = Eigen::Vector3d{1.0, 2.0, -1.0}.normalized();
    auto const steps = std::array<Eigen::Vector3d, 5>{
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d{1e-7, -2e-7, 0.5e-7},
        Eigen::Vector3d{0.1, -0.2, 0.3},
        Eigen::Vector3d{1.0, 0.5, -0.7},
        (std::numbers::pi - 0.04) * axis
    };
    auto const starts = std::array<Quaternion<double>, 2>{Quaternion<double>::UnitW(), turned};
    for (auto const & x : starts)
    {
        for (auto const & d : steps)
        {
            Eigen::Matrix3d const expected = rightJacobian(d);
            Eigen::Matrix3d const automatic = resetJacobian(AutoDiffQuaternion{}, x, d);
            EXPECT_LT((automatic - expected).norm(), 1e-12) << d.transpose();
            Eigen::Matrix3d const closed = resetJacobian(UnitQuaternion{}, x, d);
            EXPECT_EQ(closed, expected);
        }
    }
    // At a zero step the reset is the identity, as it must be: nothing moved.
    EXPECT_LT(
        (resetJacobian(AutoDiffQuaternion{}, turned, Eigen::Vector3d::Zero()) - Eigen::Matrix3d::Identity()).norm(),
        1e-15
    );
}

// The Jacobian is the derivative [covariance-reset] defines: a small error ε at x, carried to x ⊞ d, is
// (x ⊞ (d + ε)) ⊟ (x ⊞ d) ≈ G ε. Checked by central differences of plus() and minus() themselves,
// whose truncation error at a step of 1e-5 is about 1e-10.
TEST(Reset, JacobianCarriesAnErrorToTheMovedValue)
{
    auto const d = Eigen::Vector3d{0.4, -0.9, 0.6};
    Quaternion<double> const moved = Emrys::plus(AutoDiffQuaternion{}, turned, d);
    Eigen::Matrix3d const g = resetJacobian(AutoDiffQuaternion{}, turned, d);
    double const h = 1e-5;
    for (int j = 0; j < 3; ++j)
    {
        Eigen::Vector3d const e = h * Eigen::Vector3d::Unit(j);
        Quaternion<double> const ahead = Emrys::plus(AutoDiffQuaternion{}, turned, Eigen::Vector3d{d + e});
        Quaternion<double> const behind = Emrys::plus(AutoDiffQuaternion{}, turned, Eigen::Vector3d{d - e});
        Eigen::Vector3d const column =
            (Emrys::minus(AutoDiffQuaternion{}, ahead, moved) - Emrys::minus(AutoDiffQuaternion{}, behind, moved))
            / (2 * h);
        EXPECT_LT((column - g.col(j)).norm(), 1e-9);
    }
}

// On position, attitude, and velocity, resetJacobian() is the identity but for J_r on the attitude,
// resetCovariance() equals the dense G P G^T to rounding, leaves the position and velocity rows and
// columns where they meet bit for bit as they were, and returns an exactly symmetric matrix; the
// closed form gives the same covariance to rounding.
TEST(Reset, CovarianceMatchesTheDenseCongruence)
{
    auto const space = Body{};
    Eigen::Matrix<double, 10, 1> const x = bodyMean();
    auto d = Eigen::Matrix<double, 9, 1>{};
    d << 0.2, 0.1, -0.3, 0.5, -0.4, 0.8, 0.05, 0.0, -0.1;
    Eigen::Matrix<double, 9, 9> const p = spreadCov<9>();

    Eigen::Matrix<double, 9, 9> expectedG = Eigen::Matrix<double, 9, 9>::Identity();
    expectedG.block<3, 3>(3, 3) = rightJacobian(Eigen::Vector3d{d.segment<3>(3)});
    Eigen::Matrix<double, 9, 9> const g = resetJacobian(space, x, d);
    EXPECT_LT((g - expectedG).norm(), 1e-12);

    Eigen::Matrix<double, 9, 9> reset = p;
    resetCovariance(space, x, d, reset);
    Eigen::Matrix<double, 9, 9> const dense = g * p * g.transpose();
    EXPECT_LT((reset - dense).norm(), 1e-13 * p.norm());
    EXPECT_EQ(reset, reset.transpose());
    auto const plain = std::array<int, 6>{0, 1, 2, 6, 7, 8};
    for (int i : plain)
    {
        for (int j : plain)
        {
            EXPECT_EQ(reset(i, j), p(i, j)) << i << ", " << j;
        }
    }

    Eigen::Matrix<double, 9, 9> closed = p;
    resetCovariance(ClosedFormBody{}, x, d, closed);
    EXPECT_LT((closed - reset).norm(), 1e-12 * p.norm());
}

// In a Repeated block of quaternions after a position, each repetition is reset by its own J_r at
// its own value and difference offsets, which differ (four value coordinates, three difference
// ones), and the result equals the dense congruence by blockdiag(I, J_r(d_1), J_r(d_2), J_r(d_3)).
// minus() of the moved formation and the old one is the step, each repetition by its own minus().
TEST(Reset, RepeatedBlocksResetEachRepetition)
{
    auto const space = Formation{Position{}, Attitudes{Attitude{}, 3}};
    auto x = Eigen::Matrix<double, 15, 1>{};
    x << 1.0, 2.0, 3.0, turned, Quaternion<double>::UnitW(), exponential(Eigen::Vector3d{-0.4, 0.1, 0.2});
    auto d = Eigen::Matrix<double, 12, 1>{};
    d << 0.3, 0.3, 0.3, 0.2, -0.1, 0.4, 0.0, 0.0, 0.0, -0.6, 0.5, 0.2;
    Eigen::MatrixXd const p = spreadCov<12>();

    Eigen::Matrix<double, 12, 12> g = Eigen::Matrix<double, 12, 12>::Identity();
    for (int k = 0; k < 3; ++k)
    {
        g.block<3, 3>(3 + 3 * k, 3 + 3 * k) = rightJacobian(Eigen::Vector3d{d.segment<3>(3 + 3 * k)});
    }
    EXPECT_LT((resetJacobian(space, x, d) - g).norm(), 1e-12);
    Eigen::MatrixXd reset = p;
    resetCovariance(space, x, d, reset);
    EXPECT_LT((reset - g * p * g.transpose()).norm(), 1e-13 * p.norm());
    EXPECT_EQ(reset, reset.transpose());

    Eigen::Matrix<double, 15, 1> const moved = Emrys::plus(space, x, d);
    EXPECT_LT((Emrys::minus(space, moved, x) - d).norm(), 1e-12);
}

// The Kalman update on a space with an identity reset is bit for bit the update on Linear<4> given
// the same predicted output and cross-covariance, which is the update as it was before the reset
// existed: on a composite of plain blocks (linear, an angle, a magnitude), its mean too, and on a
// gain that declares its reset the identity, its covariance. The same gain without the declaration
// differentiates a Jacobian of one, to rounding, so its covariance agrees to rounding.
TEST(Reset, IdentityResetsLeaveTheUpdateAsItWas)
{
    auto const mean = Eigen::Vector4d{0.5, -1.0, 0.3, 2.0};
    Eigen::Matrix4d const cov = spreadCov<4>();
    auto const h = Eigen::Matrix<double, 2, 4>{{1.0, 0.5, 0.0, -0.2}, {0.0, 1.0, 0.3, 0.4}};
    auto const observed = Eigen::Vector2d{0.4, -0.6};
    auto const f = [&]<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 1>{h.cast<T>() * x};
    };
    // The posterior on space, and the posterior on Linear<4> given the same prediction.
    auto const updateOn = [&](auto const & space) {
        using S = std::remove_cvref_t<decltype(space)>;
        auto const prior = GaussianParams<S>{space, mean, cov};
        auto const predicted = Emrys::transform(methods::linearized(), f, Emrys::withCross(prior));
        auto const plainPrior = GaussianParams<Linear<4>>{mean, cov};
        auto const plainPredicted = Emrys::OutputWithCross<Linear<4>, Linear<2>, Emrys::JacobianForm, double>{
            Linear<2>{},
            predicted.output().mean(),
            predicted.jacobian(),
            predicted.crossCovariance(),
            predicted.explainedCovariance(),
            predicted.inputsCovariance(),
            predicted.unexplainedCovariance()
        };
        return std::pair{
            Emrys::condition(prior, predicted, observed).value().posterior,
            Emrys::condition(plainPrior, plainPredicted, observed).value().posterior
        };
    };

    auto const [blocks, blocksReference] = updateOn(Composite<Linear<2>, RadiansPi<>, Linear<1>>{});
    EXPECT_EQ(blocks.mean(), blocksReference.mean());
    EXPECT_EQ(blocks.covarianceLower(), blocksReference.covarianceLower());

    auto const [declared, declaredReference] = updateOn(Composite<Linear<3>, Gain>{});
    EXPECT_EQ(declared.covarianceLower(), declaredReference.covarianceLower());
    auto const [undeclared, undeclaredReference] = updateOn(Composite<Linear<3>, UndeclaredGain>{});
    EXPECT_EQ(undeclaredReference.covarianceLower(), declaredReference.covarianceLower());
    EXPECT_LT((undeclared.covarianceLower() - undeclaredReference.covarianceLower()).norm(), 1e-14 * cov.norm());
}

// A precise direction measurement in the Joseph form, with a large attitude correction (the
// measurement is 0.6 rad from the prior, and the step over 0.3 rad, so the reset is far from the
// identity), gives an exactly symmetric posterior with no negative eigenvalue beyond rounding,
// since the reset is a congruence.
TEST(Reset, JosephUpdateStaysPositiveSemidefinite)
{
    using State = Composite<Attitude, Velocity>;
    auto mean = Eigen::Matrix<double, 7, 1>{};
    mean << turned, 1.0, 0.0, -1.0;
    Eigen::Matrix<double, 6, 6> const cov = spreadCov<6>();
    auto const prior = GaussianParams<State>{State{}, mean, cov};
    auto const forward = Eigen::Vector3d{1.0, 0.0, 0.0};
    Eigen::Vector3d const observed = rotate(multiply(turned, exponential(Eigen::Vector3d{0.0, 0.4, 0.45})), forward);
    auto const direction = Emrys::measurementModel(
        [&]<typename T>(auto const & in, auto const &, auto const & x) {
            return rotate(Eigen::Matrix<T, 4, 1>{Emrys::block<Attitude>(in, x)}, forward).eval();
        },
        Emrys::reads(Emrys::path<Attitude>),
        Emrys::additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 1e-10 * Eigen::Matrix3d::Identity()})
    );

    auto const updated = Emrys::update(prior, observe(methods::linearized(), direction, observed, updates::joseph()));
    ASSERT_TRUE(updated);
    Eigen::Matrix<double, 6, 6> const posterior = updated->branches[0].posterior.covarianceLower();
    EXPECT_EQ(posterior, posterior.transpose());
    auto const eigen = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>{posterior};
    EXPECT_GE(eigen.eigenvalues().minCoeff(), -1e-15 * eigen.eigenvalues().maxCoeff());
    Eigen::Vector3d const step =
        Emrys::minus(Attitude{}, Quaternion<double>{updated->branches[0].posterior.mean().head<4>()}, turned);
    EXPECT_GT(step.norm(), 0.3);
}

// One smoother step on a quaternion against a plain-Eigen transcription with the resets: the
// smoothed next covariance carried back to the predicted mean by J_r(δ)^-1, the RTS step at the
// filtered mean, and the result carried to the smoothed mean by J_r(Kδ), δ the correction
// log(q_p^-1 q_s). The tolerance is rounding in the solves.
TEST(Reset, SmootherStepMovesItsCovariancesBetweenTangentSpaces)
{
    using Joint = Emrys::PredictedJoint<AutoDiffQuaternion>;
    Quaternion<double> const filteredMean = turned;
    Quaternion<double> const predictedMean = multiply(turned, exponential(Eigen::Vector3d{0.1, 0.2, -0.1}));
    Quaternion<double> const smoothedMean = multiply(predictedMean, exponential(Eigen::Vector3d{-0.5, 0.3, 0.4}));
    Eigen::Matrix3d const filteredCov = spreadCov<3>();
    Eigen::Matrix3d const predictedCov = filteredCov + 0.2 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d const smoothedCov = 0.6 * filteredCov;
    Eigen::Matrix3d const cross = 0.9 * filteredCov;
    auto const joint = Joint{
        GaussianParams<AutoDiffQuaternion>{filteredMean, filteredCov},
        GaussianParams<AutoDiffQuaternion>{predictedMean, predictedCov},
        cross
    };
    auto const next = GaussianParams<AutoDiffQuaternion>{smoothedMean, smoothedCov};
    auto const smoothed = Emrys::smoothStep(joint, next);

    Eigen::Vector3d const correction = logarithm(multiply(conjugate(predictedMean), smoothedMean));
    Eigen::Matrix3d const back = rightJacobian(correction).inverse();
    Eigen::Matrix3d const gain = cross * predictedCov.inverse();
    Eigen::Vector3d const step = gain * correction;
    Eigen::Matrix3d const forwardReset = rightJacobian(step);
    Eigen::Matrix3d const atFiltered =
        filteredCov + gain * (back * smoothedCov * back.transpose() - predictedCov) * gain.transpose();
    Eigen::Matrix3d const expected = forwardReset * atFiltered * forwardReset.transpose();
    EXPECT_TRUE(smoothed.mean().isApprox(multiply(filteredMean, exponential(step)), 1e-12));
    EXPECT_LT((smoothed.covarianceLower() - expected).norm(), 1e-12 * expected.norm());
    EXPECT_EQ(smoothed.covarianceLower(), smoothed.covarianceLower().transpose());
}

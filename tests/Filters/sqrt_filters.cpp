// The square-root filters (roadmap step 17; decisions.md [filters-and-recursions], [gaussian-forms]): SqrtKalmanFilter
// and SqrtInfoFilter deduce their types and run the Kalman recursion's steps on their forms; on an affine model and on
// a range-bearing measurement they equal KalmanFilter under Linearized, Unscented, and the sigma-point sets of step 17e
// (Cubature, SphericalSimplex, GaussHermite), with a fixed-step model too, in state, log-likelihood, and NIS; on a
// measurement far more precise than the prior they stay positive definite where the covariance
// form's posterior variance is exactly zero; the square-root information filter predicts from no
// information at all, where the information filter cannot, and reaches the least-squares estimate;
// an Unscented with a negative center weight predicts through the factorized covariance; and on a
// state with a unit quaternion both equal the Kalman filter's reset; and a model that returns its
// Jacobian with its step predicts by it in all three forms as differentiation would
// [returned-jacobians], and so does a fixed-step model of it with a sure input. A prediction through a
// transition with no inverse fails in each form that needs the inverse or the covariance it
// leaves, and an IMM and a mixture filter over the square-root form fail with it. Headers under
// test: Filters/SqrtKalmanFilter.h, Filters/SqrtInfoFilter.h, Models/FormSteps.h,
// Models/ProcessModel.h, Filters/ImmFilter.h, Filters/MixtureFilter.h.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include "Filters/form_filters.h"

#include <Eigen/LU>

#include <cmath>
#include <type_traits>

using namespace EmrysTest::FormFilters;

using Emrys::FixedStepModel;
using Emrys::GaussianError;
using Emrys::InfoFilter;
using Emrys::KalmanFilter;
using Emrys::SqrtGaussianParams;
using Emrys::SqrtInfoFilter;
using Emrys::SqrtInfoGaussianParams;
using Emrys::SqrtKalmanFilter;
using Emrys::toGaussian;
using Emrys::toInfo;
using Emrys::toSqrt;
using Emrys::toSqrtInfo;

namespace methods = Emrys::methods;

namespace
{

// Whether a prediction of Model in GyroState by Method with the inputs Inputs goes by the Jacobian
// the function returns, with no differentiation, as a prediction over a step given at each call or
// over a fixed step of Model decides.
template <typename Method, typename Model, typename... Inputs>
constexpr bool predictsByReturnedJacobian = Emrys::internal::predictsByJacobian<
    Method,
    Emrys::internal::ResolvedProcess<
        decltype(Emrys::internal::
                     modelWith(std::cref(std::declval<Model const &>()), std::declval<Inputs const &>()...)),
        GyroState>>;

}  // namespace

namespace
{

template <typename Method>
class SqrtFilterMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<
    methods::Linearized,
    methods::Unscented<>,
    Emrys::methods::Cubature,
    Emrys::methods::SphericalSimplex<>,
    Emrys::methods::GaussHermite<3>>;

// Runs steps of a position fix by method on the Kalman filter and on both square-root filters,
// predicting each with predict, and checks their states, log-likelihoods, and NIS against the
// Kalman filter's at each step.
template <typename Kalman, typename Root, typename Info, typename Method, typename Predict>
void
expectBothMatchTheKalmanFilter(
    Kalman & kalman,
    Root & root,
    Info & info,
    Method const & method,
    Predict predict,
    int steps,
    double tolerance
)
{
    for (int k = 1; k <= steps; ++k)
    {
        predict(kalman);
        ASSERT_TRUE(predict(root));
        ASSERT_TRUE(predict(info));
        auto const expected = kalman.update(makeFix(fixAt(k), method));
        auto const fromRoot = root.update(makeFix(fixAt(k), method));
        auto const fromInfo = info.update(makeFix(fixAt(k), method));
        ASSERT_TRUE(expected && fromRoot && fromInfo);
        EXPECT_TRUE(fromRoot->accepted && fromInfo->accepted);
        EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), tolerance);
        EXPECT_NEAR(fromInfo->logLikelihood(), expected->logLikelihood(), tolerance);
        EXPECT_NEAR(fromRoot->branches[0].nis(), expected->branches[0].nis(), tolerance);
        EXPECT_NEAR(fromInfo->branches[0].nis(), expected->branches[0].nis(), tolerance);
        expectSameState(root.state(), kalman.state(), tolerance);
        expectSameState(info.state(), kalman.state(), tolerance);
    }
}

// expectBothMatchTheKalmanFilter() over a model that takes a step, on filters predicting by method.
template <typename Model, typename Method>
void
expectBothMatchOverTheStep(Model const & model, Method const & method, int steps, double tolerance)
{
    auto kalman = KalmanFilter{model, makeInitial(), method};
    auto root = SqrtKalmanFilter{model, toSqrt(makeInitial()).value(), method};
    auto info = SqrtInfoFilter{model, toSqrtInfo(makeInitial()).value(), method};
    expectBothMatchTheKalmanFilter(
        kalman,
        root,
        info,
        method,
        [](auto & filter) {
            return filter.predict(timeStep);
        },
        steps,
        tolerance
    );
}

}  // namespace

TYPED_TEST_SUITE(SqrtFilterMethodTest, Methods);

// The input a measurement reads from a square-root form holds the root and forms the covariance
// S S^T only on request, which reads its lower triangle.
TEST(SqrtFilters, ARootInputFormsItsCovarianceOnRequest)
{
    Eigen::Matrix2d const root{{2.0, 0.0}, {0.5, 1.0}};
    auto const input =
        Emrys::internal::RootInput<Emrys::Linear<2>, double>{Emrys::Linear<2>{}, Eigen::Vector2d{1.0, -1.0}, root};
    EXPECT_EQ(input.root(), root);
    Eigen::Matrix2d const covariance = input.covarianceLower();
    Eigen::Matrix2d const expected = root * root.transpose();
    EXPECT_TRUE(covariance.triangularView<Eigen::Lower>().toDenseMatrix().isApprox(
        expected.triangularView<Eigen::Lower>().toDenseMatrix(),
        1e-15
    ));
}

// The deduction guides pick the space, model, and method for both filters, with a fixed-step
// model's own; the states are the forms; and the initial factor is the Gaussian's Cholesky
// factor.
TEST(SqrtFilters, DeduceTheirTypesAndHoldTheirParts)
{
    auto root = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(root), SqrtKalmanFilter<State, decltype(makeModel()), methods::Linearized>>);
    static_assert(std::is_same_v<decltype(root)::StateType, SqrtGaussianParams<State>>);
    auto const unscented = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value(), methods::unscented<0.5>()};
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(unscented.method())>, methods::Unscented<0.5>>);
    EXPECT_EQ(root.state().root(), Eigen::Matrix4d{makeInitial().covarianceLower().llt().matrixL()});

    auto info = SqrtInfoFilter{makeModel(), toSqrtInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(info), SqrtInfoFilter<State, decltype(makeModel()), methods::Linearized>>);
    static_assert(std::is_same_v<decltype(info)::StateType, SqrtInfoGaussianParams<State>>);
    auto const fixed = FixedStepModel{makeModel(), State{}, timeStep, methods::unscented()};
    auto fixedFilter = SqrtInfoFilter{fixed, toSqrtInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(fixedFilter)::MethodType, methods::Unscented<>>);
    EXPECT_TRUE(info.state().precision()->isApprox(makeInitial().covarianceLower().inverse(), 1e-12));
}

// Over five fixes, each method's predict() and update() in both square-root filters equal the
// Kalman filter's state, log-likelihood, and NIS to rounding: the triangularizations factor the
// same covariances the Kalman filter forms.
TYPED_TEST(SqrtFilterMethodTest, MatchTheKalmanFilterOnAnAffineModel)
{
    expectBothMatchOverTheStep(makeModel(), TypeParam{}, 5, 1e-10);
}

// Alternating a fix with a range-bearing measurement, whose innovation has a periodic bearing,
// the three form filters and the Kalman filter agree to rounding under each method: every form
// spreads its sigma points from the Cholesky factor of the same covariance, the default axes
// ([sigma-point-axes]), which the covariance form factorizes and the others triangularize from the
// factor they hold. The spherical simplex, whose result turns with the root more than the symmetric
// sets' do, shows it most.
TYPED_TEST(SqrtFilterMethodTest, MatchTheKalmanFilterOnRangeBearing)
{
    double const kalmanTolerance = 1e-9;
    auto kalman = KalmanFilter{makeModel(), makeInitial(), TypeParam{}};
    auto root = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value(), TypeParam{}};
    auto rootInfo = SqrtInfoFilter{makeModel(), toSqrtInfo(makeInitial()).value(), TypeParam{}};
    auto info = InfoFilter{makeModel(), toInfo(makeInitial()).value(), TypeParam{}};
    for (int k = 1; k <= 6; ++k)
    {
        kalman.predict(timeStep);
        ASSERT_TRUE(root.predict(timeStep));
        ASSERT_TRUE(rootInfo.predict(timeStep));
        ASSERT_TRUE(info.predict(timeStep));
        if (k % 2 == 0)
        {
            auto const expected = kalman.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            auto const fromRoot = root.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            auto const fromRootInfo = rootInfo.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            auto const fromInfo = info.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            ASSERT_TRUE(expected && fromRoot && fromRootInfo && fromInfo);
            EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), kalmanTolerance);
            EXPECT_NEAR(fromRootInfo->logLikelihood(), fromRoot->logLikelihood(), 1e-9);
            EXPECT_NEAR(fromInfo->logLikelihood(), fromRoot->logLikelihood(), 1e-9);
        }
        else
        {
            ASSERT_TRUE(kalman.update(makeFix(fixAt(k), TypeParam{})));
            ASSERT_TRUE(root.update(makeFix(fixAt(k), TypeParam{})));
            ASSERT_TRUE(rootInfo.update(makeFix(fixAt(k), TypeParam{})));
            ASSERT_TRUE(info.update(makeFix(fixAt(k), TypeParam{})));
        }
        expectSameState(root.state(), kalman.state(), kalmanTolerance);
        auto const reference = toGaussian(root.state());
        ASSERT_TRUE(reference);
        expectSameState(rootInfo.state(), *reference, 1e-9);
        expectSameState(info.state(), *reference, 1e-9);
    }
}

// With a FixedStepModel both filters take the model's method and predict over its step, matching
// the Kalman filter under the cached transition, whose factor is triangularized directly, and
// under Unscented.
TEST(SqrtFilters, AFixedStepModelMatchesTheKalmanFilter)
{
    auto const check = [](auto const & model, auto const & method) {
        auto kalman = KalmanFilter{model, makeInitial()};
        auto root = SqrtKalmanFilter{model, toSqrt(makeInitial()).value()};
        auto info = SqrtInfoFilter{model, toSqrtInfo(makeInitial()).value()};
        expectBothMatchTheKalmanFilter(
            kalman,
            root,
            info,
            method,
            [](auto & filter) {
                return filter.predict();
            },
            4,
            1e-10
        );
    };
    check(FixedStepModel{makeModel(), State{}, timeStep}, methods::linearized());
    check(FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()}, methods::unscented<0.5>());
}

// A fix with variance 1e-12 on a prior with variance 1e6, the example of [covariance-update-forms]: the covariance
// form's standard update subtracts to a posterior variance of exactly zero, so the Gaussian has no density, while the
// square-root forms keep the posterior variance at 1e-12 to the digits the factor carries, and the information form
// adds precisions and does too.
TEST(SqrtFilters, StayPositiveDefiniteWhereTheCovarianceFormCannot)
{
    Matrix4 cov = Matrix4::Identity();
    cov(0, 0) = cov(1, 1) = 1e6;
    auto const prior = GaussianParams<State>{State{}, Vector4::Zero(), cov};
    Eigen::Matrix2d const precise = 1e-12 * Eigen::Matrix2d::Identity();
    auto const z = Eigen::Vector2d{3.0, -2.0};

    auto kalman = KalmanFilter{makeModel(), prior};
    ASSERT_TRUE(kalman.update(makeFix<Position>(z, methods::linearized(), precise)));
    EXPECT_EQ(kalman.state().covarianceLower()(0, 0), 0.0);
    EXPECT_EQ(kalman.state().toDensity().error(), GaussianError::NotPositiveDefinite);

    auto root = SqrtKalmanFilter{makeModel(), toSqrt(prior).value()};
    ASSERT_TRUE(root.update(makeFix<Position>(z, methods::linearized(), precise)));
    EXPECT_NEAR(root.state().covariance().value()(0, 0), 1e-12, 1e-15);
    EXPECT_TRUE(root.state().toDensity().has_value());
    EXPECT_TRUE(root.state().mean().head<2>().isApprox(z, 1e-12));

    auto rootInfo = SqrtInfoFilter{makeModel(), toSqrtInfo(prior).value()};
    ASSERT_TRUE(rootInfo.update(makeFix<Position>(z, methods::linearized(), precise)));
    EXPECT_NEAR(rootInfo.state().covariance().value()(0, 0), 1e-12, 1e-15);
    EXPECT_TRUE(rootInfo.state().mean()->head<2>().isApprox(z, 1e-12));

    auto info = InfoFilter{makeModel(), toInfo(prior).value()};
    ASSERT_TRUE(info.update(makeFix<Position>(z, methods::linearized(), precise)));
    EXPECT_NEAR(info.state().covariance().value()(0, 0), 1e-12, 1e-15);
}

// From no information at all (R = 0) and with no process noise, the square-root information
// filter predicts through the inverse transition, takes a fix, predicts, and takes another, and
// then holds the least-squares estimate of the two fixes, x_2 = (z_2, (z_2 - z_1) / dt) with
// covariance (H^T R^{-1} H)^{-1} for H = [[I, -dt I], [I, 0]], computed here in plain Eigen. The
// first innovation has infinite covariance and a zero NIS; the information filter cannot predict
// from the same start.
TEST(SqrtFilters, TheSqrtInfoFilterPredictsFromNoInfo)
{
    auto const first = Eigen::Vector2d{1.0, 2.0};
    auto const second = Eigen::Vector2d{2.0, 2.5};
    auto filter = SqrtInfoFilter{
        makeModel(0.0),
        SqrtInfoGaussianParams<State>{Vector4::Zero(), Matrix4::Zero(), Vector4::Zero()}
    };
    auto const flat = filter.update(makeFix<Position>(first));
    ASSERT_TRUE(flat);
    EXPECT_TRUE(std::isinf(flat->logLikelihood()) && flat->logLikelihood() < 0);
    EXPECT_NEAR(flat->branches[0].nis(), 0.0, 1e-20);
    EXPECT_EQ(filter.state().mean().error(), GaussianError::Singular);
    ASSERT_TRUE(filter.predict(timeStep));
    EXPECT_EQ(filter.state().mean().error(), GaussianError::Singular);
    auto const determined = filter.update(makeFix<Position>(second));
    ASSERT_TRUE(determined);
    EXPECT_NEAR(determined->branches[0].nis(), 0.0, 1e-20);

    auto expected = Vector4{};
    expected << second, (second - first) / timeStep;
    Eigen::Matrix4d h = Eigen::Matrix4d::Zero();
    h.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity();
    h.topRightCorner<2, 2>() = -timeStep * Eigen::Matrix2d::Identity();
    h.bottomLeftCorner<2, 2>() = Eigen::Matrix2d::Identity();
    Matrix4 r = Matrix4::Zero();
    r.topLeftCorner<2, 2>() = fixNoise;
    r.bottomRightCorner<2, 2>() = fixNoise;
    Matrix4 const expectedCov = (h.transpose() * r.inverse() * h).inverse();
    EXPECT_TRUE(filter.state().mean()->isApprox(expected, 1e-12));
    EXPECT_TRUE(filter.state().covariance()->isApprox(expectedCov, 1e-10));

    auto info =
        InfoFilter{makeModel(0.0), Emrys::InfoGaussianParams<State>{Vector4::Zero(), Matrix4::Zero(), Vector4::Zero()}};
    ASSERT_TRUE(info.update(makeFix<Position>(first)));
    EXPECT_EQ(info.predict(timeStep).error(), GaussianError::Singular);
}

namespace
{

struct Level : Emrys::Linear<1>
{};

using Held = Emrys::Composite<Level>;

struct Only
{};

struct Single : Emrys::Named<Only>
{};

using Vector1 = Eigen::Matrix<double, 1, 1>;

// A model that forgets its value at every step, x' = 0, with noise of spectral density
// `density`: affine in its block and returning its Jacobian, zero, so that a fixed step caches a
// transition that has no inverse.
auto
forget(double density)
{
    return Emrys::processModel(
        []<typename T>(auto const &, auto const &, auto const &, auto) {
            return Emrys::withJacobian(Eigen::Matrix<T, 1, 1>{T(0)}, Vector1{0.0});
        },
        reads(path<Level>),
        Emrys::writes(path<Level>),
        Emrys::affineIn(path<Level>),
        Emrys::continuousNoise(Emrys::on<Level>(density))
    );
}

}  // namespace

// A prediction through a transition with no inverse fails where a form needs one or needs the
// covariance it leaves: the square-root information form inverts the transition, and a fixed step
// caches no inverse; without noise the prediction has no covariance, which the information forms
// cannot hold and the square-root form cannot factor under sigma points with a negative center
// weight, which form it, nor can an IMM or a mixture filter over it; and an information form
// with no information has no Gaussian for sigma points to move.
TEST(SqrtFilters, PredictionsThroughASingularTransitionFail)
{
    auto const prior = Emrys::GaussianParams<Held>{Held{}, Vector1{1.0}, Vector1{2.0}};
    auto const noisy = forget(0.5);
    auto const exact = forget(0.0);

    auto rootInfo = SqrtInfoFilter{noisy, toSqrtInfo(prior).value()};
    EXPECT_EQ(rootInfo.predict(1.0).error(), GaussianError::Singular);
    auto const fixed = FixedStepModel{noisy, Held{}, 1.0};
    EXPECT_FALSE(fixed.transitionInverse());
    auto fixedRootInfo = SqrtInfoFilter{fixed, toSqrtInfo(prior).value()};
    EXPECT_EQ(fixedRootInfo.predict().error(), GaussianError::Singular);

    auto info = InfoFilter{exact, toInfo(prior).value(), methods::unscented()};
    EXPECT_FALSE(info.predict(1.0));
    auto fixedInfo = InfoFilter{FixedStepModel{exact, Held{}, 1.0, methods::unscented()}, toInfo(prior).value()};
    EXPECT_FALSE(fixedInfo.predict());
    auto unscentedRootInfo = SqrtInfoFilter{exact, toSqrtInfo(prior).value(), methods::unscented()};
    EXPECT_FALSE(unscentedRootInfo.predict(1.0));
    auto root = SqrtKalmanFilter{exact, toSqrt(prior).value(), methods::unscented<0.5, 0.0>()};
    EXPECT_EQ(root.predict(1.0).error(), GaussianError::NotPositiveDefinite);
    auto imm = Emrys::ImmFilter{
        Emrys::SwitchingModel{Emrys::MarkovChain<Single>{}, exact},
        Emrys::HybridParams<Emrys::SqrtGaussian<Held>, Single>{
            Emrys::CategoricalParams<Single>{},
            toSqrt(prior).value()
        },
        methods::unscented<0.5, 0.0>()
    };
    EXPECT_EQ(imm.predict(1.0).error(), GaussianError::NotPositiveDefinite);
    auto roots = Emrys::MixtureParams<Emrys::SqrtGaussian<Held>>{Held{}};
    roots.add(0.0, toSqrt(prior).value());
    auto mixture = Emrys::MixtureFilter{exact, roots, methods::unscented<0.5, 0.0>()};
    EXPECT_EQ(mixture.predict(1.0).error(), GaussianError::NotPositiveDefinite);

    auto const noInfo = Emrys::InfoGaussianParams<Held>{Vector1::Zero(), Vector1::Zero(), Vector1::Zero()};
    auto blind = InfoFilter{FixedStepModel{noisy, Held{}, 1.0, methods::unscented()}, noInfo};
    EXPECT_EQ(blind.predict().error(), GaussianError::Singular);
    auto rootBlind = SqrtInfoFilter{
        noisy,
        SqrtInfoGaussianParams<Held>{Vector1::Zero(), Vector1::Zero(), Vector1::Zero()},
        methods::unscented()
    };
    EXPECT_EQ(rootBlind.predict(1.0).error(), GaussianError::Singular);
}

// An Unscented with alpha 0.5 and beta 0 gives the center point a negative covariance weight, so
// the square-root prediction subtracts its term from the formed covariance and factorizes that;
// the result still equals the Kalman filter's under the same method.
TEST(SqrtFilters, ANegativeCenterWeightPredictsThroughTheFactorizedCovariance)
{
    auto const method = methods::unscented<0.5, 0.0>();
    auto const weights = Emrys::internal::sigmaWeights<double, 4>(method, 4);
    ASSERT_LT(weights.centerCov, 0.0);
    expectBothMatchOverTheStep(makeModel(), method, 3, 1e-9);
}

// On a state with a unit quaternion both square-root filters equal the Kalman filter, whose update
// resets the covariance to the tangent space at the moved mean [covariance-reset]: the square-root form resets
// its factor, and the square-root information form moves its reference by the inverse Jacobian.
TEST(SqrtFilters, ResetOnAnAttitude)
{
    auto kalman = KalmanFilter{makeAttitudeModel(), makeAttitudeInitial()};
    auto root = SqrtKalmanFilter{makeAttitudeModel(), toSqrt(makeAttitudeInitial()).value()};
    auto info = SqrtInfoFilter{makeAttitudeModel(), toSqrtInfo(makeAttitudeInitial()).value()};
    for (int k = 1; k <= 4; ++k)
    {
        kalman.predict(1.0);
        ASSERT_TRUE(root.predict(1.0));
        ASSERT_TRUE(info.predict(1.0));
        auto const expected = kalman.update(makeDirection(k));
        auto const fromRoot = root.update(makeDirection(k));
        auto const fromInfo = info.update(makeDirection(k));
        ASSERT_TRUE(expected && fromRoot && fromInfo);
        EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), 1e-9);
        EXPECT_NEAR(fromInfo->logLikelihood(), expected->logLikelihood(), 1e-9);
        expectSameState(root.state(), kalman.state(), 1e-9);
        expectSameState(info.state(), kalman.state(), 1e-9);
    }
    EXPECT_NEAR(root.state().mean().head<4>().norm(), 1.0, 1e-14);
}

// A model that returns its Jacobian with its step [returned-jacobians], a gyroscope turning an
// attitude by a sure rate less a bias, predicts by that Jacobian under Linearized and needs no
// differentiation: the Kalman filter equals the one that differentiates the same model without it,
// to rounding, and both square-root filters equal the Kalman filter, over four steps of large turns
// with a direction update after each. An uncertain rate goes through the transform, which with no
// uncertainty in the rate gives the same prediction, and so does any method other than Linearized.
TEST(SqrtFilters, AReturnedJacobianMatchesDifferentiation)
{
    using Rate = Eigen::Vector3d;
    using SureRate = decltype(Emrys::sure<GyroRate>(Rate{}));
    using UncertainRate = GaussianParams<GyroRate>;
    using Model = decltype(makeGyroModel<GyroTurn>());
    using Differentiated = decltype(makeGyroModel<DifferentiatedGyroTurn>());
    static_assert(predictsByReturnedJacobian<methods::Linearized, Model, SureRate>);
    static_assert(!predictsByReturnedJacobian<methods::Linearized, Differentiated, SureRate>);
    static_assert(!predictsByReturnedJacobian<methods::Linearized, Model, UncertainRate>);
    static_assert(!predictsByReturnedJacobian<methods::Unscented<>, Model, SureRate>);

    auto const expectSame = [](GaussianParams<GyroState> const & actual, GaussianParams<GyroState> const & expected) {
        EXPECT_LT(Emrys::minus(GyroState{}, actual.mean(), expected.mean()).norm(), 1e-12);
        EXPECT_LT((actual.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);
    };
    auto kalman = KalmanFilter{makeGyroModel<GyroTurn>(), makeGyroInitial()};
    auto differentiated = KalmanFilter{makeGyroModel<DifferentiatedGyroTurn>(), makeGyroInitial()};
    auto root = SqrtKalmanFilter{makeGyroModel<GyroTurn>(), toSqrt(makeGyroInitial()).value()};
    auto info = SqrtInfoFilter{makeGyroModel<GyroTurn>(), toSqrtInfo(makeGyroInitial()).value()};
    for (int k = 1; k <= 4; ++k)
    {
        Rate const turning = 0.4 * k * Rate{0.6, -0.3, 0.8};
        auto const rate = Emrys::sure<GyroRate>(turning);
        auto const uncertain = UncertainRate{turning, Eigen::Matrix3d::Zero()};
        auto const throughTheTransform =
            Emrys::predict(methods::linearized(), kalman.state(), kalman.model(), 0.5, uncertain);
        kalman.predict(0.5, rate);
        differentiated.predict(0.5, rate);
        ASSERT_TRUE(root.predict(0.5, rate));
        ASSERT_TRUE(info.predict(0.5, rate));
        expectSame(differentiated.state(), kalman.state());
        expectSame(throughTheTransform, kalman.state());
        expectSameState(root.state(), kalman.state(), 1e-9);
        expectSameState(info.state(), kalman.state(), 1e-9);

        auto const direction = makeDirection<methods::Linearized, Attitude>(k);
        auto const expected = kalman.update(direction);
        auto const fromDifferentiated = differentiated.update(direction);
        auto const fromRoot = root.update(direction);
        auto const fromInfo = info.update(direction);
        ASSERT_TRUE(expected && fromDifferentiated && fromRoot && fromInfo);
        EXPECT_NEAR(fromDifferentiated->logLikelihood(), expected->logLikelihood(), 1e-9);
        EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), 1e-9);
        EXPECT_NEAR(fromInfo->logLikelihood(), expected->logLikelihood(), 1e-9);
    }
}

// A fixed-step model whose function returns the Jacobian of its step [returned-jacobians] predicts
// by it with a sure input, as the same model given its step at each call does: the arithmetic is
// the same, the cached process noise being the model's own over the step, so the Kalman filter
// equals that filter exactly in place. predict() equals it to rounding, since the compiler may
// fuse its multiply-adds otherwise, as GCC does on arm64, and so does the information filter.
// The fixed step differentiates where the function returns no Jacobian, a rate is
// uncertain, or the method is not Linearized, and differentiation gives the Jacobian to rounding.
// Four steps of large turns, with a direction update after each.
TEST(SqrtFilters, AFixedStepModelPredictsByTheJacobianItsFunctionReturns)
{
    using Rate = Eigen::Vector3d;
    using SureRate = decltype(Emrys::sure<GyroRate>(Rate{}));
    using UncertainRate = GaussianParams<GyroRate>;
    using Model = decltype(makeGyroModel<GyroTurn>());
    using Fixed = FixedStepModel<GyroState, Model>;
    using Differentiated = FixedStepModel<GyroState, decltype(makeGyroModel<DifferentiatedGyroTurn>())>;
    using Unscented = FixedStepModel<GyroState, Model, methods::Unscented<>>;
    static_assert(!Fixed::ReadsOnlyState && !Fixed::CachesTransition);
    static_assert(predictsByReturnedJacobian<Fixed::MethodType, Fixed::ModelType, SureRate>);
    static_assert(!predictsByReturnedJacobian<Fixed::MethodType, Fixed::ModelType, UncertainRate>);
    static_assert(!predictsByReturnedJacobian<Differentiated::MethodType, Differentiated::ModelType, SureRate>);
    static_assert(!predictsByReturnedJacobian<Unscented::MethodType, Unscented::ModelType, SureRate>);

    constexpr double dt = 0.5;
    auto const fixed = Fixed{makeGyroModel<GyroTurn>(), GyroState{}, dt};
    auto const differentiatedModel = Differentiated{makeGyroModel<DifferentiatedGyroTurn>(), GyroState{}, dt};
    auto kalman = KalmanFilter{makeGyroModel<GyroTurn>(), makeGyroInitial()};
    auto fixedKalman = KalmanFilter{fixed, makeGyroInitial()};
    auto differentiated = KalmanFilter{differentiatedModel, makeGyroInitial()};
    auto info = InfoFilter{fixed, toInfo(makeGyroInitial()).value()};
    for (int k = 1; k <= 4; ++k)
    {
        auto const rate = Emrys::sure<GyroRate>(Rate{0.4 * k * Rate{0.6, -0.3, 0.8}});
        auto const returned = Emrys::predict(fixedKalman.state(), fixed, rate);
        kalman.predict(dt, rate);
        fixedKalman.predict(rate);
        differentiated.predict(rate);
        ASSERT_TRUE(info.predict(rate));
        EXPECT_EQ(fixedKalman.state().mean(), kalman.state().mean());
        EXPECT_EQ(fixedKalman.state().covarianceLower(), kalman.state().covarianceLower());
        expectSameState(returned, kalman.state(), 1e-12);
        expectSameState(differentiated.state(), kalman.state(), 1e-12);
        expectSameState(info.state(), kalman.state(), 1e-9);

        auto const direction = makeDirection<methods::Linearized, Attitude>(k);
        ASSERT_TRUE(kalman.update(direction) && fixedKalman.update(direction));
        ASSERT_TRUE(differentiated.update(direction) && info.update(direction));
    }
}

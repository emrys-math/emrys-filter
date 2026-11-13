// The information filter (roadmap step 16; decisions.md [filters-and-recursions], [gaussian-forms]): InfoFilter deduces
// its type and runs the Kalman recursion's steps on an InfoGaussianParams; on an affine model and on a range-bearing
// measurement it equals KalmanFilter under Linearized, Unscented, and the sigma-point sets of step 17e (Cubature,
// SphericalSimplex, GaussHermite), with a fixed-step model too, in state, log-likelihood, and NIS; several measurements
// add their information in any order; from no information it takes Linearized updates at its reference and refuses a
// prediction and a sigma-point update until every direction is determined; and on a state with a unit quaternion its
// reference moves by the reset Jacobian, so that it equals the Kalman filter's reset. Headers under test:
// Filters/InfoFilter.h, Models/FormSteps.h.

#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

#include <gtest/gtest.h>

#include "Filters/form_filters.h"

#include <Eigen/LU>

#include <cmath>
#include <type_traits>

using namespace EmrysTest::FormFilters;

using Emrys::FixedStepModel;
using Emrys::GaussianError;
using Emrys::InfoFilter;
using Emrys::InfoGaussianParams;
using Emrys::KalmanFilter;
using Emrys::toGaussian;
using Emrys::toInfo;

namespace methods = Emrys::methods;

namespace
{

template <typename Method>
class InfoFilterMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<
    methods::Linearized,
    methods::Unscented<>,
    Emrys::methods::Cubature,
    Emrys::methods::SphericalSimplex<>,
    Emrys::methods::GaussHermite<3>>;

}  // namespace

TYPED_TEST_SUITE(InfoFilterMethodTest, Methods);

// The deduction guide picks the space, model, and method, with a fixed-step model's own; state(),
// model(), and method() are accessible; and the state is the information form.
TEST(InfoFilter, DeducesItsTypeAndHoldsItsParts)
{
    auto filter = InfoFilter{makeModel(), toInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(filter), InfoFilter<State, decltype(makeModel()), methods::Linearized>>);
    static_assert(std::is_same_v<decltype(filter)::StateType, InfoGaussianParams<State>>);
    auto const unscented = InfoFilter{makeModel(), toInfo(makeInitial()).value(), methods::unscented<0.5>()};
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(unscented.method())>, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    auto const fixed = FixedStepModel{makeModel(), State{}, timeStep, methods::unscented()};
    auto fixedFilter = InfoFilter{fixed, toInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(fixedFilter)::MethodType, methods::Unscented<>>);
    EXPECT_TRUE(filter.state().reference().isApprox(makeInitial().mean()));
    EXPECT_TRUE(filter.state().precision()->isApprox(makeInitial().covarianceLower().inverse(), 1e-12));
    EXPECT_EQ(filter.model().spectralDensity(), accelDensity * Eigen::Matrix2d::Identity());
}

// Over five fixes, each method's predict() and update() equal the Kalman filter's state,
// log-likelihood, and NIS to rounding: the prediction goes through the same moments, and the
// update adds the information the Kalman gain conditions on.
TYPED_TEST(InfoFilterMethodTest, MatchesTheKalmanFilterOnAnAffineModel)
{
    auto kalman = KalmanFilter{makeModel(), makeInitial(), TypeParam{}};
    auto info = InfoFilter{makeModel(), toInfo(makeInitial()).value(), TypeParam{}};
    for (int k = 1; k <= 5; ++k)
    {
        kalman.predict(timeStep);
        ASSERT_TRUE(info.predict(timeStep));
        auto const expected = kalman.update(makeFix(fixAt(k), TypeParam{}));
        auto const actual = info.update(makeFix(fixAt(k), TypeParam{}));
        ASSERT_TRUE(expected && actual);
        EXPECT_TRUE(actual->accepted);
        EXPECT_NEAR(actual->logLikelihood(), expected->logLikelihood(), 1e-10);
        EXPECT_NEAR(actual->branches[0].nis(), expected->branches[0].nis(), 1e-10);
        expectSameState(info.state(), kalman.state(), 1e-10);
    }
}

// Alternating a fix with a range-bearing measurement, whose innovation has a periodic bearing,
// every method equals its Kalman counterpart to rounding: the statistical linearization the
// information update takes is the one the Kalman gain is made of, and both forms spread their sigma
// points from the Cholesky factor of the same covariance, the default axes ([sigma-point-axes]),
// the covariance form by factorizing it and the information form by triangularizing its factor.
TYPED_TEST(InfoFilterMethodTest, MatchesTheKalmanFilterOnRangeBearing)
{
    double const tolerance = 1e-9;
    auto kalman = KalmanFilter{makeModel(), makeInitial(), TypeParam{}};
    auto info = InfoFilter{makeModel(), toInfo(makeInitial()).value(), TypeParam{}};
    for (int k = 1; k <= 6; ++k)
    {
        kalman.predict(timeStep);
        ASSERT_TRUE(info.predict(timeStep));
        if (k % 2 == 0)
        {
            auto const expected = kalman.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            auto const actual = info.update(makeRangeBearing(rangeBearingAt(k), TypeParam{}));
            ASSERT_TRUE(expected && actual);
            EXPECT_NEAR(actual->logLikelihood(), expected->logLikelihood(), tolerance);
        }
        else
        {
            ASSERT_TRUE(kalman.update(makeFix(fixAt(k), TypeParam{})));
            ASSERT_TRUE(info.update(makeFix(fixAt(k), TypeParam{})));
        }
        expectSameState(info.state(), kalman.state(), tolerance);
    }
}

// With a FixedStepModel the filter takes the model's method and predicts over its step, matching
// the Kalman filter on the same model under the cached transition and under Unscented.
TEST(InfoFilter, AFixedStepModelMatchesTheKalmanFilter)
{
    auto const check = [](auto const & model) {
        auto kalman = KalmanFilter{model, makeInitial()};
        auto info = InfoFilter{model, toInfo(makeInitial()).value()};
        for (int k = 1; k <= 4; ++k)
        {
            kalman.predict();
            ASSERT_TRUE(info.predict());
            ASSERT_TRUE(kalman.update(makeFix(fixAt(k))));
            ASSERT_TRUE(info.update(makeFix(fixAt(k))));
            expectSameState(info.state(), kalman.state(), 1e-10);
        }
    };
    check(FixedStepModel{makeModel(), State{}, timeStep});
    check(FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()});
}

// Two measurements in one step add their information, so the state is the same in either order to
// rounding, and equals the Kalman filter's two sequential updates.
TEST(InfoFilter, AddsInfoInAnyOrder)
{
    Eigen::Vector2d const position = fixAt(1);
    auto const velocity = Eigen::Vector2d{1.1, 0.4};
    auto forward = InfoFilter{makeModel(), toInfo(makeInitial()).value()};
    auto backward = InfoFilter{makeModel(), toInfo(makeInitial()).value()};
    auto kalman = KalmanFilter{makeModel(), makeInitial()};
    ASSERT_TRUE(forward.update(makeFix<Position>(position)));
    ASSERT_TRUE(forward.update(makeFix<Velocity>(velocity, methods::linearized(), velocityNoise)));
    ASSERT_TRUE(backward.update(makeFix<Velocity>(velocity, methods::linearized(), velocityNoise)));
    ASSERT_TRUE(backward.update(makeFix<Position>(position)));
    ASSERT_TRUE(kalman.update(makeFix<Position>(position)));
    ASSERT_TRUE(kalman.update(makeFix<Velocity>(velocity, methods::linearized(), velocityNoise)));

    EXPECT_TRUE(forward.state().precision()->isApprox(backward.state().precision().value(), 1e-13));
    EXPECT_TRUE(forward.state().mean()->isApprox(backward.state().mean().value(), 1e-13));
    expectSameState(forward.state(), kalman.state(), 1e-10);
}

// From no information at all, a Linearized update at the reference is taken, with an innovation
// of infinite covariance (log-likelihood minus infinity, NIS zero), while a prediction and an
// Unscented update return Singular; once a velocity fix determines the rest the state is the two
// fixes' least-squares estimate, and the filter predicts as the Kalman filter does from there.
TEST(InfoFilter, StartsFromNoInfo)
{
    auto const position = Eigen::Vector2d{1.0, 2.0};
    auto const velocity = Eigen::Vector2d{0.5, -0.5};
    auto filter = InfoFilter{
        makeModel(),
        InfoGaussianParams<State>{Vector4{0.1, 0.2, 0.3, 0.4}, Matrix4::Zero(), Vector4::Zero()}
    };
    auto const first = filter.update(makeFix<Position>(position));
    ASSERT_TRUE(first);
    EXPECT_TRUE(first->accepted);
    EXPECT_TRUE(std::isinf(first->logLikelihood()) && first->logLikelihood() < 0);
    EXPECT_EQ(first->branches[0].nis(), 0.0);
    EXPECT_EQ(filter.state().mean().error(), GaussianError::Singular);
    EXPECT_EQ(filter.predict(timeStep).error(), GaussianError::Singular);
    EXPECT_EQ(filter.update(makeFix<Position>(position, methods::unscented())).error(), GaussianError::Singular);

    ASSERT_TRUE(filter.update(makeFix<Velocity>(velocity, methods::linearized(), velocityNoise)));
    auto expected = Vector4{};
    expected << position, velocity;
    Matrix4 expectedCov = Matrix4::Zero();
    expectedCov.topLeftCorner<2, 2>() = fixNoise;
    expectedCov.bottomRightCorner<2, 2>() = velocityNoise;
    EXPECT_TRUE(filter.state().mean()->isApprox(expected, 1e-12));
    EXPECT_TRUE(filter.state().covariance()->isApprox(expectedCov, 1e-12));

    auto kalman = KalmanFilter{makeModel(), GaussianParams<State>{State{}, expected, expectedCov}};
    kalman.predict(timeStep);
    ASSERT_TRUE(filter.predict(timeStep));
    expectSameState(filter.state(), kalman.state(), 1e-10);
}

// On a state with a unit quaternion the filter equals the Kalman filter, whose update resets the
// covariance to the tangent space at the moved mean [covariance-reset]: the information update is taken at the
// reference, which the next step moves to the mean through the inverse reset Jacobian, the same
// transport seen from the other side.
TEST(InfoFilter, MovesItsReferenceByTheResetJacobianOnAnAttitude)
{
    auto kalman = KalmanFilter{makeAttitudeModel(), makeAttitudeInitial()};
    auto info = InfoFilter{makeAttitudeModel(), toInfo(makeAttitudeInitial()).value()};
    for (int k = 1; k <= 4; ++k)
    {
        kalman.predict(1.0);
        ASSERT_TRUE(info.predict(1.0));
        auto const expected = kalman.update(makeDirection(k));
        auto const actual = info.update(makeDirection(k));
        ASSERT_TRUE(expected && actual);
        EXPECT_NEAR(actual->logLikelihood(), expected->logLikelihood(), 1e-9);
        expectSameState(info.state(), kalman.state(), 1e-9);
    }
    EXPECT_NEAR(toGaussian(info.state())->mean().head<4>().norm(), 1.0, 1e-14);
}

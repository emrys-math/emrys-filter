// The square-root, information, and square-root information filters under the Iterated method of
// Models/Transform.h, through the form steps of Models/FormSteps.h, over range and bearing on a
// constant-velocity state: with Linearized they equal the Kalman filter (roadmap step 17b;
// decisions.md [iterated-update]), with Cubature they agree with one another and with the Kalman filter to the
// transform's dependence on its root (roadmap step 17e; [sigma-point-sets]), and an information form with no
// information has nothing to spread its points by. The same comparisons on an attitude are in
// iterated_forms_attitude.cpp.

#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Models/Transform.h>

#include <gtest/gtest.h>

#include <limits>

#include "Models/iterated_problem.h"

using namespace EmrysTest::IteratedProblem;

using Emrys::InfoFilter;

namespace methods = Emrys::methods;

// The square-root, information, and square-root information filters under Iterated equal the
// Kalman filter under it, over several steps of range and bearing, in state and log-likelihood,
// and report the same iterations: the forms place the iterates from their covariance and fold the
// last linearization in their own way.
TEST(IteratedUpdate, TheFormsEqualTheKalmanFilter)
{
    auto const method = methods::iterated(methods::linearized(), {.tolerance = 1e-10, .maxIterations = 20});
    expectFormsEqualKalman(makeModel(), makeInitial(), [&](int k) {
        return makeRangeBearing(rangeBearingAt(k), method);
    });
}

// The square-root, information, and square-root information filters under Iterated<Cubature>
// equal the Kalman filter to rounding over several steps of range and bearing, as every form spreads
// its points by the same default axes ([sigma-point-axes]).
TEST(IteratedPosteriorLinearization, TheFormsAgree)
{
    auto const method = methods::iterated(methods::cubature(), {.tolerance = 1e-10, .maxIterations = 20});
    expectFormsEqualKalman(makeModel(), makeInitial(), [&](int k) {
        return makeRangeBearing(rangeBearingAt(k), method);
    });
}

// An information form with no information has no covariance to spread points by, so an iterated
// sigma-point update of it returns Singular, as a sigma-point update does, where
// Iterated<Linearized> linearizes once at the reference.
TEST(IteratedPosteriorLinearization, NeedsInfoToSpreadItsPoints)
{
    auto filter = InfoFilter{
        makeModel(),
        Emrys::InfoGaussianParams<State>{Vector4{0.1, 0.2, 0.3, 0.4}, Matrix4::Zero(), Vector4::Zero()}
    };
    auto const result = filter.update(makeRangeBearing(rangeAndBearing, methods::iterated(methods::cubature())));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), Emrys::GaussianError::Singular);
}

// A measurement noise so far from positive definite that no innovation covariance is ends each
// filter's iterated update with NotPositiveDefinite, and the square-root information form's
// single linearization too.
TEST(IteratedUpdate, AnIndefiniteInnovationEndsTheIteration)
{
    Eigen::Matrix2d const indefinite = -100.0 * Eigen::Matrix2d::Identity();
    auto const z = Eigen::Vector2d{0.5, 0.2};
    auto const fix = makeFix(z, methods::iterated(methods::linearized()), indefinite);
    auto const notPositiveDefinite = Emrys::GaussianError::NotPositiveDefinite;
    auto kalman = Emrys::KalmanFilter{makeModel(), makeInitial()};
    auto root = Emrys::SqrtKalmanFilter{makeModel(), Emrys::toSqrt(makeInitial()).value()};
    auto info = InfoFilter{makeModel(), Emrys::toInfo(makeInitial()).value()};
    auto rootInfo = Emrys::SqrtInfoFilter{makeModel(), Emrys::toSqrtInfo(makeInitial()).value()};
    EXPECT_EQ(kalman.update(fix).error(), notPositiveDefinite);
    EXPECT_EQ(root.update(fix).error(), notPositiveDefinite);
    EXPECT_EQ(info.update(fix).error(), notPositiveDefinite);
    EXPECT_EQ(rootInfo.update(fix).error(), notPositiveDefinite);
    EXPECT_EQ(rootInfo.update(makeFix(z, methods::linearized(), indefinite)).error(), notPositiveDefinite);
    EXPECT_EQ(
        kalman.update(makeFix(z, methods::iterated(methods::cubature()), indefinite)).error(),
        notPositiveDefinite
    );
}

// An information form with no information linearizes an Iterated<Linearized> update once at its
// reference, as it would without iterating, while the square-root information form returns
// Singular for a sigma-point update, iterated or not, which needs a covariance to spread.
TEST(IteratedUpdate, NoInformationLinearizesOnceAtTheReference)
{
    auto const reference = Vector4{0.1, 0.2, 0.3, 0.4};
    auto const noInfo = Emrys::InfoGaussianParams<State>{reference, Matrix4::Zero(), Vector4::Zero()};
    auto const noRootInfo = Emrys::SqrtInfoGaussianParams<State>{reference, Matrix4::Zero(), Vector4::Zero()};
    auto info = InfoFilter{makeModel(), noInfo};
    auto const once = info.update(makeRangeBearing(rangeAndBearing, methods::iterated(methods::linearized())));
    ASSERT_TRUE(once);
    EXPECT_EQ(once->branches.front().iterations, 1);
    auto rootInfo = Emrys::SqrtInfoFilter{makeModel(), noRootInfo};
    auto const rootOnce = rootInfo.update(makeRangeBearing(rangeAndBearing, methods::iterated(methods::linearized())));
    ASSERT_TRUE(rootOnce);
    EXPECT_EQ(rootOnce->branches.front().iterations, 1);

    auto iterated = Emrys::SqrtInfoFilter{makeModel(), noRootInfo};
    EXPECT_EQ(
        iterated.update(makeRangeBearing(rangeAndBearing, methods::iterated(methods::cubature()))).error(),
        Emrys::GaussianError::Singular
    );
    auto unscented = Emrys::SqrtInfoFilter{makeModel(), noRootInfo};
    EXPECT_EQ(
        unscented.update(makeRangeBearing(rangeAndBearing, Emrys::methods::unscented())).error(),
        Emrys::GaussianError::Singular
    );
}

// A NaN in the information ends an update or a prediction with NonFinite, not with the Singular
// that an information form without information in some direction may go on from.
TEST(IteratedUpdate, NonFiniteInformationFails)
{
    auto const reference = Vector4{0.1, 0.2, 0.3, 0.4};
    Matrix4 nan = Matrix4::Identity();
    nan(1, 1) = std::numeric_limits<double>::quiet_NaN();
    auto const nonFinite = Emrys::GaussianError::NonFinite;
    auto info = InfoFilter{makeModel(), Emrys::InfoGaussianParams<State>{reference, nan, Vector4::Zero()}};
    EXPECT_EQ(info.update(makeFix(Eigen::Vector2d{0.5, 0.2})).error(), nonFinite);
    auto const nanRoot = Emrys::SqrtInfoGaussianParams<State>{reference, nan, Vector4::Zero()};
    auto rootInfo = Emrys::SqrtInfoFilter{makeModel(), nanRoot};
    EXPECT_EQ(rootInfo.update(makeFix(Eigen::Vector2d{0.5, 0.2})).error(), nonFinite);
    EXPECT_EQ(rootInfo.predict(timeStep).error(), nonFinite);
    auto fixed = Emrys::SqrtInfoFilter{Emrys::FixedStepModel{makeModel(), State{}, timeStep}, nanRoot};
    EXPECT_EQ(fixed.predict().error(), nonFinite);
    EXPECT_EQ(fixed.predict(Emrys::Steps{2}).error(), nonFinite);
}

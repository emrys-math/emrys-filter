// linearizedAbout() of Models/Transform.h: a transform, a prediction (Models/ProcessModel.h), and an
// update (Models/Measurement.h, and the forms' steps of Models/FormSteps.h) linearized about a point or
// a distribution other than their input, and a KalmanFilter shadowing another (roadmap step 17g;
// decisions.md [linearization-reference]). About the input itself each step is the step without a
// reference; about a point, Linearized is the affine model there, by hand; about an update's posterior,
// the update is an Iterated update's next iteration, by Linearized and by Cubature, on a vector state
// and on a unit quaternion; mixture noise branches from the one linearization; an affine model predicts
// the same about any point; a model's own Jacobian and the differentiated one agree about a point on a
// unit quaternion; and the square-root and information filters shadowing a main filter agree with a
// Kalman filter shadowing it.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Transform.h>

#include <gtest/gtest.h>

#include "Models/iterated_problem.h"

#include <cmath>
#include <cstddef>

using namespace EmrysTest::IteratedProblem;

using Emrys::GaussianMixtureParams;
using Emrys::KalmanFilter;
using Emrys::linearizedAbout;

namespace methods = Emrys::methods;

namespace
{

// The range and bearing of a state's position, as a transform's function of the whole state.
auto const rangeBearingOfState = []<typename T>(auto const & x) {
    return Eigen::Matrix<T, 2, 1>{
        Eigen::numext::sqrt(T(x.template head<2>().squaredNorm())),
        Eigen::numext::atan2(T(x(1)), T(x(0)))
    };
};

// A point near the target at (4, 3), away from the poor prior's mean, to linearize about.
Vector4 const nearTarget{3.8, 2.9, 0.9, 0.6};

// Whether two Gaussians in the same space have the same mean and covariance, to tolerance.
template <typename A, typename B>
void
expectSameGaussian(A const & a, B const & b, double tolerance)
{
    EXPECT_LE((a.mean() - b.mean()).norm(), tolerance);
    EXPECT_LE((a.covarianceLower() - b.covarianceLower()).norm(), tolerance);
}

}  // namespace

// About the input itself, by its mean for Linearized and by the whole distribution for every method, a
// transform is the transform without a reference: its affine model at the input, applied to the input,
// gives back the moments it came from. The tolerance is rounding through the regression's solve.
TEST(LinearizedAbout, TransformAboutTheInputIsTheTransform)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&](auto const & method, auto const & about, double tolerance) {
        auto const plain =
            Emrys::transform(method, rangeBearingOfState, Emrys::outputSpace(RangeBearingSpace{}), withCross(prior));
        auto const moved = Emrys::transform(
            method,
            rangeBearingOfState,
            Emrys::outputSpace(RangeBearingSpace{}),
            about,
            withCross(prior)
        );
        expectSameGaussian(moved.output(), plain.output(), tolerance);
        EXPECT_LT((moved.crossCovariance() - plain.crossCovariance()).norm(), tolerance);
    };
    check(methods::linearized(), linearizedAbout(prior.mean()), 1e-12);
    check(methods::linearized(), linearizedAbout(prior), 1e-12);
    check(methods::cubature(), linearizedAbout(prior), 1e-10);
    check(methods::unscented(), linearizedAbout(prior), 1e-10);
}

// About a point, Linearized is the affine model there applied to the input, by hand from the Jacobian
// rangeBearingOf() writes out: mean h(p) + H (mu - p), covariance H P H^T, cross-covariance P H^T. With
// no withCross() mark the result is the Gaussian alone, the same one.
TEST(LinearizedAbout, TransformAboutAPointIsTheAffineModelThere)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const [h, jacobian] = rangeBearingOf(nearTarget);
    Eigen::Vector2d const mean = h + jacobian * (prior.mean() - nearTarget);
    Eigen::Matrix2d const cov = jacobian * prior.covarianceLower() * jacobian.transpose();
    Eigen::Matrix<double, 4, 2> const cross = prior.covarianceLower() * jacobian.transpose();
    auto const about = linearizedAbout(nearTarget);
    auto const moved = Emrys::transform(
        methods::linearized(),
        rangeBearingOfState,
        Emrys::outputSpace(RangeBearingSpace{}),
        about,
        withCross(prior)
    );
    EXPECT_LT((moved.output().mean() - mean).norm(), 1e-12);
    EXPECT_LT((moved.output().covarianceLower() - cov).norm(), 1e-12);
    EXPECT_LT((moved.crossCovariance() - cross).norm(), 1e-12);
    auto const unmarked = Emrys::transform(
        methods::linearized(),
        rangeBearingOfState,
        Emrys::outputSpace(RangeBearingSpace{}),
        about,
        prior
    );
    expectSameGaussian(unmarked, moved.output(), 0.0);
}

// About the prior, by its mean for Linearized and by the whole prior for Cubature, an update is the
// update without a reference, with additive noise and with noise passed through the function.
TEST(LinearizedAbout, UpdateAboutThePriorIsTheUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&](auto const & measurement, auto const & about, double tolerance) {
        auto const plain = Emrys::update(prior, measurement);
        auto const moved = Emrys::update(prior, measurement, about);
        ASSERT_TRUE(plain && moved);
        expectSameGaussian(moved->branches.front().posterior, plain->branches.front().posterior, tolerance);
        EXPECT_NEAR(moved->logLikelihood(), plain->logLikelihood(), tolerance);
    };
    check(makeRangeBearing(rangeAndBearing, methods::linearized()), linearizedAbout(prior.mean()), 1e-12);
    check(makeRangeBearing(rangeAndBearing, methods::cubature()), linearizedAbout(prior), 1e-10);
    check(makeRangeBearingWithInputNoise(rangeAndBearing, methods::linearized()), linearizedAbout(prior.mean()), 1e-12);
    check(makeRangeBearingWithInputNoise(rangeAndBearing, methods::cubature()), linearizedAbout(prior), 1e-10);
}

// About a point, the Linearized update is the Kalman update of the prior under the affine model at that
// point, the linearized Kalman filter's (Gelb 1974, chapter 6), in plain Eigen: z_hat = h(p) + H (mu - p),
// S = H P H^T + R, K = P H^T S^-1, and the posterior mu + K (z - z_hat), P - K S K^T.
TEST(LinearizedAbout, UpdateAboutAPointIsTheLinearizedKalmanUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const [h, jacobian] = rangeBearingOf(nearTarget);
    Eigen::Vector2d const predicted = h + jacobian * (prior.mean() - nearTarget);
    Matrix2 const s = jacobian * prior.covarianceLower() * jacobian.transpose() + rangeBearingNoise;
    Eigen::Matrix<double, 4, 2> const gain = prior.covarianceLower() * jacobian.transpose() * s.inverse();
    Vector4 const mean = prior.mean() + gain * (rangeAndBearing - predicted);
    Matrix4 const cov = prior.covarianceLower() - gain * s * gain.transpose();

    auto const updated =
        Emrys::update(prior, makeRangeBearing(rangeAndBearing, methods::linearized()), linearizedAbout(nearTarget));
    ASSERT_TRUE(updated);
    EXPECT_LT((updated->branches.front().posterior.mean() - mean).norm(), 1e-12);
    EXPECT_LT((updated->branches.front().posterior.covarianceLower() - cov).norm(), 1e-12);
}

// About the posterior of an update, the update is the second linearization of an Iterated one, whose
// iterate is that posterior: by Linearized at its mean, the iterated EKF's step (Bell and Cathey 1993),
// and by Cubature over it, the iterated posterior linearization filter's (García-Fernández et al. 2015).
// Both are the same arithmetic as Iterated's second pass, so the tolerance is rounding.
TEST(LinearizedAbout, UpdateAboutThePosteriorIsTheNextIteration)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&](auto const & method, double tolerance) {
        auto const once = Emrys::update(prior, makeRangeBearing(rangeAndBearing, method));
        ASSERT_TRUE(once);
        auto const & first = once->branches.front().posterior;
        auto const moved = Emrys::update(prior, makeRangeBearing(rangeAndBearing, method), linearizedAbout(first));
        auto const twice = Emrys::update(
            prior,
            makeRangeBearing(rangeAndBearing, methods::iterated(method, {.tolerance = 0.0, .maxIterations = 2}))
        );
        ASSERT_TRUE(moved && twice);
        expectSameGaussian(moved->branches.front().posterior, twice->branches.front().posterior, tolerance);
        EXPECT_NEAR(moved->logLikelihood(), twice->logLikelihood(), tolerance);
    };
    check(methods::linearized(), 1e-12);
    check(methods::cubature(), 1e-10);
}

// The iterated EKF written with transforms and no measurement object:
// each pass linearizes the range and bearing at the current mean, applies that to the prior, and
// conditions the prior on the observation through condition(); n passes are the Iterated
// update capped at n linearizations, to rounding.
TEST(LinearizedAbout, TransformLoopIsTheIteratedUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    GaussianParams<State> current = prior;
    // The noise as additive noise, which the prediction keeps apart as its N.
    auto const noise = GaussianParams<Emrys::Linear<2>>{Eigen::Vector2d::Zero(), rangeBearingNoise};
    for (int i = 0; i < 3; ++i)
    {
        auto const predicted = Emrys::transform(
            methods::linearized(),
            rangeBearingOfState,
            Emrys::outputSpace(RangeBearingSpace{}),
            Emrys::additiveNoise(noise),
            linearizedAbout(current.mean()),
            withCross(prior)
        );
        auto const updated = Emrys::condition(prior, predicted, rangeAndBearing);
        ASSERT_TRUE(updated);
        current = updated->posterior;
    }
    auto const iterated = Emrys::update(
        prior,
        makeRangeBearing(
            rangeAndBearing,
            methods::iterated(methods::linearized(), {.tolerance = 0.0, .maxIterations = 3})
        )
    );
    ASSERT_TRUE(iterated);
    expectSameGaussian(current, iterated->branches.front().posterior, 1e-12);
}

// On a unit quaternion the update about a point is linearized there and carried to the prior's chart,
// as an Iterated update's iterate is, so about the first posterior's mean it is again the second
// iteration; and the square-root and information forms, which carry it to their own reference, agree.
TEST(LinearizedAbout, UpdateAboutAPointOnAUnitQuaternion)
{
    GaussianParams<Motion> const prior = makeAttitudeInitial();
    auto const measurement = makeDirection(3);
    auto const once = Emrys::update(prior, measurement);
    ASSERT_TRUE(once);
    auto const & first = once->branches.front().posterior;
    auto const about = linearizedAbout(first.mean());
    auto const moved = Emrys::update(prior, measurement, about);
    auto const twice = Emrys::update(
        prior,
        makeDirection(3, methods::iterated(methods::linearized(), {.tolerance = 0.0, .maxIterations = 2}))
    );
    ASSERT_TRUE(moved && twice);
    auto const & posterior = moved->branches.front().posterior;
    EXPECT_LT(Emrys::minus(prior.space(), posterior.mean(), twice->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT((posterior.covarianceLower() - twice->branches.front().posterior.covarianceLower()).norm(), 1e-12);
    // The reference moved the result: one linearization at the prior is further from it.
    EXPECT_GT(Emrys::minus(prior.space(), posterior.mean(), first.mean()).norm(), 1e-6);

    auto root = Emrys::SqrtKalmanFilter{makeAttitudeModel(), Emrys::toSqrt(prior).value()};
    auto info = Emrys::InfoFilter{makeAttitudeModel(), Emrys::toInfo(prior).value()};
    auto rootInfo = Emrys::SqrtInfoFilter{makeAttitudeModel(), Emrys::toSqrtInfo(prior).value()};
    ASSERT_TRUE(root.update(measurement, about));
    ASSERT_TRUE(info.update(measurement, about));
    ASSERT_TRUE(rootInfo.update(measurement, about));
    expectSameState(root.state(), posterior, 1e-9);
    expectSameState(info.state(), posterior, 1e-9);
    expectSameState(rootInfo.state(), posterior, 1e-9);
}

// With mixture noise the one linearization about the reference serves every component: each branch's
// posterior is the update about the reference with that component alone.
TEST(LinearizedAbout, MixtureNoiseBranchesFromOneLinearization)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto noise = GaussianMixtureParams<RangeBearingSpace>{RangeBearingSpace{}};
    noise.add(std::log(0.8), Eigen::Vector2d::Zero(), rangeBearingNoise);
    noise.add(std::log(0.2), Eigen::Vector2d{0.5, 0.0}, 25.0 * rangeBearingNoise);
    auto const about = linearizedAbout(nearTarget);
    auto const updated =
        Emrys::update(prior, observe(methods::linearized(), makeRangeBearingModel(noise), rangeAndBearing), about);
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 2u);
    std::size_t k = 0;
    for (auto const & entry : noise)
    {
        auto const alone = Emrys::update(
            prior,
            observe(methods::linearized(), makeRangeBearingModel(entry.component), rangeAndBearing),
            about
        );
        ASSERT_TRUE(alone);
        expectSameGaussian(updated->branches[k].posterior, alone->branches.front().posterior, 1e-14);
        ++k;
    }
}

// An affine model predicts the same about any point and about any distribution: its affine model is
// itself, so f(p) + F (mu - p) = F mu. The prediction and its cross-covariance with the prior match the
// prediction without a reference to rounding.
TEST(LinearizedAbout, AnAffineModelPredictsTheSameAboutAnyPoint)
{
    GaussianParams<State> const prior = makeInitial();
    auto const model = makeModel();
    auto const plain = Emrys::predictJoint(methods::linearized(), prior, model, timeStep);
    auto const elsewhere = GaussianParams<State>{State{}, nearTarget, 0.5 * Matrix4::Identity()};
    auto const check = [&](auto const & method, auto const & about) {
        auto const moved = Emrys::predictJoint(method, prior, model, timeStep, about);
        expectSameGaussian(moved.predicted, plain.predicted, 1e-12);
        EXPECT_LT((moved.cross - plain.cross).norm(), 1e-12);
        expectSameGaussian(Emrys::predict(method, prior, model, timeStep, about), plain.predicted, 1e-12);
    };
    check(methods::linearized(), linearizedAbout(nearTarget));
    check(methods::linearized(), linearizedAbout(elsewhere));
    check(methods::cubature(), linearizedAbout(elsewhere));
    check(methods::unscented(), linearizedAbout(prior));
}

// On a unit quaternion a gyroscope-driven step about a point away from the mean takes its Jacobian
// there: a model that returns it [returned-jacobians] and the same model differentiated agree to
// rounding, and about the mean itself the prediction is the prediction without a reference.
TEST(LinearizedAbout, PredictAboutAPointOnAUnitQuaternion)
{
    GaussianParams<GyroState> const prior = makeGyroInitial();
    auto const rate = Emrys::sure<GyroRate>(Eigen::Vector3d{0.3, -0.2, 0.5});
    double const dt = 0.1;
    auto step = Eigen::Matrix<double, 6, 1>{};
    step << 0.05, -0.04, 0.03, 0.002, 0.001, -0.003;
    Eigen::Matrix<double, 7, 1> const point = Emrys::plus(prior.space(), prior.mean(), step);
    auto const about = linearizedAbout(point);

    auto const given = Emrys::predict(methods::linearized(), prior, makeGyroModel<GyroTurn>(), dt, about, rate);
    auto const differentiated =
        Emrys::predict(methods::linearized(), prior, makeGyroModel<DifferentiatedGyroTurn>(), dt, about, rate);
    EXPECT_LT(Emrys::minus(prior.space(), given.mean(), differentiated.mean()).norm(), 1e-12);
    EXPECT_LT((given.covarianceLower() - differentiated.covarianceLower()).norm(), 1e-12);

    auto const plain = Emrys::predict(methods::linearized(), prior, makeGyroModel<GyroTurn>(), dt, rate);
    auto const atMean = Emrys::predict(
        methods::linearized(),
        prior,
        makeGyroModel<GyroTurn>(),
        dt,
        linearizedAbout(prior.mean()),
        rate
    );
    EXPECT_LT(Emrys::minus(prior.space(), atMean.mean(), plain.mean()).norm(), 1e-12);
    EXPECT_LT((atMean.covarianceLower() - plain.covarianceLower()).norm(), 1e-12);
    // The point moved the prediction: the step is not affine in the bias.
    EXPECT_GT((given.covarianceLower() - plain.covarianceLower()).norm(), 1e-9);
}

// A shadow filter in each form, predicting and updating about a main filter's state as a tracker's
// shadow does: by Linearized, the square-root and information filters equal the Kalman filter, as their
// own steps do (form_filters.h); by Cubature over the main state, every form spreads from the Cholesky
// factor of the main state's covariance, the default axes, and they agree to rounding.
TEST(LinearizedAbout, ShadowFiltersInEveryForm)
{
    auto const check = [&](auto const & method, double formTolerance, double kalmanTolerance) {
        auto const model = makeModel();
        GaussianParams<State> const initial = makeInitial();
        auto main = KalmanFilter{model, initial};
        auto kalman = KalmanFilter{model, initial};
        auto root = Emrys::SqrtKalmanFilter{model, Emrys::toSqrt(initial).value()};
        auto info = Emrys::InfoFilter{model, Emrys::toInfo(initial).value()};
        auto rootInfo = Emrys::SqrtInfoFilter{model, Emrys::toSqrtInfo(initial).value()};
        for (int k = 1; k <= 4; ++k)
        {
            GaussianParams<State> const filtered = main.state();
            auto const before = linearizedAbout(filtered);
            kalman.predict(timeStep, before);
            ASSERT_TRUE(root.predict(timeStep, before));
            ASSERT_TRUE(info.predict(timeStep, before));
            ASSERT_TRUE(rootInfo.predict(timeStep, before));
            main.predict(timeStep);

            GaussianParams<State> const predicted = main.state();
            auto const about = linearizedAbout(predicted);
            auto const measurement = makeRangeBearing(rangeBearingAt(k), method);
            auto const expected = kalman.update(measurement, about);
            auto const fromRoot = root.update(measurement, about);
            auto const fromInfo = info.update(measurement, about);
            auto const fromRootInfo = rootInfo.update(measurement, about);
            ASSERT_TRUE(expected && fromRoot && fromInfo && fromRootInfo);
            auto const rootState = Emrys::toGaussian(root.state()).value();
            expectSameState(info.state(), rootState, formTolerance);
            expectSameState(rootInfo.state(), rootState, formTolerance);
            EXPECT_NEAR(fromInfo->logLikelihood(), fromRoot->logLikelihood(), formTolerance);
            EXPECT_NEAR(fromRootInfo->logLikelihood(), fromRoot->logLikelihood(), formTolerance);
            expectSameState(root.state(), kalman.state(), kalmanTolerance);
            EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), kalmanTolerance);
            ASSERT_TRUE(main.update(makeRangeBearing(rangeBearingAt(k), methods::iterated(methods::linearized()))));
        }
    };
    check(methods::linearized(), 1e-9, 1e-9);
    check(methods::cubature(), 1e-8, 1e-8);
}

// In every form, a cubature update about the prior itself is the update without a reference (roadmap
// step 17u): about a distribution the reference's root form gives H, N, and Omega, and each form applies
// them to its own prior, which is the same distribution, so the posteriors and log-likelihoods agree to
// the rounding of the regression's solve, as UpdateAboutThePriorIsTheUpdate's do.
TEST(LinearizedAbout, CubatureAboutThePriorIsTheUpdateInEveryForm)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const about = linearizedAbout(prior);
    auto const check = [&](auto filter, auto alsoFilter, char const * name) {
        auto const measurement = makeRangeBearing(rangeAndBearing, methods::cubature());
        auto const plain = filter.update(measurement);
        auto const moved = alsoFilter.update(measurement, about);
        ASSERT_TRUE(plain && moved) << name;
        auto const plainState = Emrys::toGaussian(filter.state()).value();
        expectSameState(alsoFilter.state(), plainState, 1e-10);
        EXPECT_NEAR(moved->logLikelihood(), plain->logLikelihood(), 1e-10) << name;
    };
    auto const model = makeModel();
    check(KalmanFilter{model, prior}, KalmanFilter{model, prior}, "covariance");
    auto const root = Emrys::toSqrt(prior).value();
    check(Emrys::SqrtKalmanFilter{model, root}, Emrys::SqrtKalmanFilter{model, root}, "square root");
    auto const info = Emrys::toInfo(prior).value();
    check(Emrys::InfoFilter{model, info}, Emrys::InfoFilter{model, info}, "information");
    auto const rootInfo = Emrys::toSqrtInfo(prior).value();
    check(Emrys::SqrtInfoFilter{model, rootInfo}, Emrys::SqrtInfoFilter{model, rootInfo}, "square-root information");
}

// A KalmanFilter's update about a reference is the free update about it, whose one branch it keeps, and
// the update into a reused result fills the same.
TEST(LinearizedAbout, KalmanFilterUpdateIsTheFreeUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const measurement = makeRangeBearing(rangeAndBearing, methods::linearized());
    auto const about = linearizedAbout(nearTarget);
    auto const expected = Emrys::update(prior, measurement, about);
    ASSERT_TRUE(expected);
    auto filter = KalmanFilter{makeModel(), prior};
    auto const report = filter.update(measurement, about);
    ASSERT_TRUE(report);
    expectSameGaussian(filter.state(), expected->branches.front().posterior, 1e-14);
    EXPECT_NEAR(report->logLikelihood(), expected->logLikelihood(), 1e-14);

    auto reused = KalmanFilter{makeModel(), prior};
    auto result = *report;
    result.branches.front().logWeight = 1.0;
    ASSERT_TRUE(reused.update(measurement, about, Emrys::reductions::collapse(), result));
    expectSameGaussian(reused.state(), filter.state(), 0.0);
    EXPECT_EQ(result.branches.front().logWeight, report->branches.front().logWeight);
}

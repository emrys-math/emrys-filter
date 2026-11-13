// The Iterated method of Models/Transform.h with Linearized, the iterated extended Kalman filter's
// update, through update() in Models/Measurement.h (roadmap step 17b; decisions.md [iterated-update]): on a
// range-bearing update with a poor prior it reaches the Gauss-Newton solution computed in plain
// Eigen, where one linearization falls well short, and after n linearizations it is n Gauss-Newton
// iterations; one iteration is Linearized bitwise, in the standard and Joseph forms, with input
// noise, and in place in a KalmanFilter; each branch reports its iterations and whether the
// tolerance stopped them; mixture noise iterates each branch on its own; input noise that adds
// gives the additive update; and on a unit quaternion it reaches the Gauss-Newton solution in the
// prior's chart. The forms are in iterated_forms.cpp and the sigma-point methods in
// iterated_posterior_linearization.cpp.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include "Models/iterated_problem.h"

#include <array>
#include <cmath>
#include <type_traits>
#include <cstddef>
#include <utility>

using namespace EmrysTest::IteratedProblem;

using Emrys::GaussianMixtureParams;
using Emrys::KalmanFilter;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;

namespace
{

// The Gauss-Newton iterates of the maximum a posteriori estimate, from the prior mean: each solves
// the normal equations (P^-1 + H^T R^-1 H) dx = H^T R^-1 (z - h(x)) - P^-1 (x - mu), a different
// algebra from the Kalman gain's. Returns the iterate after iterations steps and its posterior
// covariance (P^-1 + H^T R^-1 H)^-1.
std::pair<Vector4, Matrix4>
gaussNewton(GaussianParams<State> const & prior, int iterations)
{
    Matrix4 const info = prior.covarianceLower().inverse();
    Matrix2 const noiseInfo = rangeBearingNoise.inverse();
    Vector4 x = prior.mean();
    for (int i = 0; i < iterations; ++i)
    {
        auto const [h, jacobian] = rangeBearingOf(x);
        Matrix4 const normal = info + jacobian.transpose() * noiseInfo * jacobian;
        Vector4 const gradient = jacobian.transpose() * noiseInfo * (rangeAndBearing - h) - info * (x - prior.mean());
        x += normal.ldlt().solve(gradient);
    }
    Jacobian const jacobian = rangeBearingOf(x).second;
    Matrix4 const posterior = (info + jacobian.transpose() * noiseInfo * jacobian).inverse();
    return {x, posterior};
}

// Whether two branches are the same to the bit: posterior, residual, and innovation covariance.
template <typename A, typename B>
void
expectSameBranch(A const & a, B const & b)
{
    EXPECT_TRUE(a.posterior.mean() == b.posterior.mean());
    EXPECT_TRUE(a.posterior.covarianceLower() == b.posterior.covarianceLower());
    EXPECT_TRUE(a.innovation.residual() == b.innovation.residual());
    EXPECT_TRUE(a.innovation.covarianceLower() == b.innovation.covarianceLower());
    EXPECT_EQ(a.logLikelihood(), b.logLikelihood());
}

}  // namespace

// On a poor prior and a precise range and bearing, the iterated update reaches the Gauss-Newton
// solution of the maximum a posteriori problem, mean and covariance, to rounding: the tolerance
// is far below the prior's spread, so the last linearization is at the solution. One linearization
// at the prior mean, the EKF's, misses it by a hundredth or more.
TEST(IteratedUpdate, ReachesTheGaussNewtonSolution)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const iterated = Emrys::update(
        prior,
        makeRangeBearing(
            rangeAndBearing,
            methods::iterated(methods::linearized(), {.tolerance = 1e-12, .maxIterations = 50})
        )
    );
    ASSERT_TRUE(iterated);
    auto const [expectedMean, expectedCov] = gaussNewton(prior, 100);
    auto const & branch = iterated->branches.front();
    EXPECT_LT((branch.posterior.mean() - expectedMean).norm(), 1e-10);
    EXPECT_LT((branch.posterior.covarianceLower() - expectedCov).norm(), 1e-10);
    EXPECT_TRUE(branch.converged);
    EXPECT_GT(branch.iterations, 2u);
    EXPECT_LT(branch.iterations, 50u);

    auto const once = Emrys::update(prior, makeRangeBearing(rangeAndBearing, methods::linearized()));
    ASSERT_TRUE(once);
    EXPECT_GT((once->branches.front().posterior.mean() - expectedMean).norm(), 1e-2);
}

// With no tolerance to stop it, the update makes exactly maxIterations linearizations, reports
// that the cap stopped it, and its mean is that many Gauss-Newton iterates from the prior mean,
// as Bell and Cathey (1993) show the iterated update to be. A KalmanFilter conditioning in place
// reaches the same state and reports the same.
TEST(IteratedUpdate, NIterationsAreNGaussNewtonSteps)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&]<std::size_t n>(std::integral_constant<std::size_t, n>) {
        auto const capped = methods::iterated(methods::linearized(), {.tolerance = 0.0, .maxIterations = n});
        auto const updated = Emrys::update(prior, makeRangeBearing(rangeAndBearing, capped));
        ASSERT_TRUE(updated);
        auto const & branch = updated->branches.front();
        EXPECT_EQ(branch.iterations, n);
        EXPECT_FALSE(branch.converged);
        Vector4 const expected = gaussNewton(prior, int(n)).first;
        EXPECT_LT((branch.posterior.mean() - expected).norm(), 1e-12);

        auto filter = KalmanFilter{makeModel(), prior};
        auto const report = filter.update(makeRangeBearing(rangeAndBearing, capped));
        ASSERT_TRUE(report);
        EXPECT_EQ(report->branches.front().iterations, n);
        EXPECT_FALSE(report->branches.front().converged);
        EXPECT_LT((filter.state().mean() - branch.posterior.mean()).norm(), 1e-14);
        EXPECT_LT((filter.state().covarianceLower() - branch.posterior.covarianceLower()).norm(), 1e-14);
    };
    check(std::integral_constant<std::size_t, 1>{});
    check(std::integral_constant<std::size_t, 2>{});
    check(std::integral_constant<std::size_t, 3>{});
    check(std::integral_constant<std::size_t, 5>{});
}

// One iteration is the Linearized update, to the bit: in the standard form, in the Joseph form,
// with input noise, and in place in a KalmanFilter. The Linearized update reports one
// linearization that nothing stopped early; the capped Iterated one that its cap stopped it.
TEST(IteratedUpdate, OneIterationIsLinearizedBitwise)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const once = methods::iterated(methods::linearized(), {.tolerance = 1e-3, .maxIterations = 1});
    auto const check = [&](auto const & iterated, auto const & linearized) {
        auto const a = Emrys::update(prior, iterated);
        auto const b = Emrys::update(prior, linearized);
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        expectSameBranch(a->branches.front(), b->branches.front());
        EXPECT_EQ(a->branches.front().iterations, 1u);
        EXPECT_FALSE(a->branches.front().converged);
        EXPECT_EQ(b->branches.front().iterations, 1u);
        EXPECT_TRUE(b->branches.front().converged);

        auto first = KalmanFilter{makeModel(), prior};
        auto second = KalmanFilter{makeModel(), prior};
        auto const reportA = first.update(iterated);
        auto const reportB = second.update(linearized);
        ASSERT_TRUE(reportA);
        ASSERT_TRUE(reportB);
        EXPECT_TRUE(first.state().mean() == second.state().mean());
        EXPECT_TRUE(first.state().covarianceLower() == second.state().covarianceLower());
        EXPECT_EQ(reportA->logLikelihood(), reportB->logLikelihood());
        EXPECT_EQ(reportA->branches.front().nis(), reportB->branches.front().nis());
    };
    check(makeRangeBearing(rangeAndBearing, once), makeRangeBearing(rangeAndBearing, methods::linearized()));
    check(
        makeRangeBearing(rangeAndBearing, once, updates::joseph()),
        makeRangeBearing(rangeAndBearing, methods::linearized(), updates::joseph())
    );
    check(
        makeRangeBearingWithInputNoise(rangeAndBearing, once),
        makeRangeBearingWithInputNoise(rangeAndBearing, methods::linearized())
    );
}

// Noise passed through the measurement function where it only adds gives the additive update at every
// iterate, so the two converge to the same posterior, to rounding: the input-noise path carries
// the prior's spread of the blocks read through the joint covariance rather than the
// cross-covariance's rows.
TEST(IteratedUpdate, InputNoiseThatAddsGivesTheAdditiveUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const method = methods::iterated(methods::linearized(), {.tolerance = 1e-12, .maxIterations = 50});
    auto const additive = Emrys::update(prior, makeRangeBearing(rangeAndBearing, method));
    auto const input = Emrys::update(prior, makeRangeBearingWithInputNoise(rangeAndBearing, method));
    ASSERT_TRUE(additive);
    ASSERT_TRUE(input);
    auto const & a = additive->branches.front();
    auto const & b = input->branches.front();
    EXPECT_LT((a.posterior.mean() - b.posterior.mean()).norm(), 1e-12);
    EXPECT_LT((a.posterior.covarianceLower() - b.posterior.covarianceLower()).norm(), 1e-12);
    EXPECT_NEAR(a.logLikelihood(), b.logLikelihood(), 1e-10);
    EXPECT_EQ(a.iterations, b.iterations);
}

// With mixture noise each branch is the iterated update under its own component, the same as a
// measurement with that component alone, and the branches are weighted by w_k p(z | k) / p(z)
// with each p(z | k) from its own last linearization.
TEST(IteratedUpdate, MixtureNoiseIteratesEachBranch)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const method = methods::iterated(methods::linearized(), {.tolerance = 1e-10, .maxIterations = 50});
    auto noise = GaussianMixtureParams<RangeBearingSpace>{RangeBearingSpace{}};
    noise.add(std::log(0.8), Eigen::Vector2d::Zero(), rangeBearingNoise);
    noise.add(std::log(0.2), Eigen::Vector2d{0.5, 0.0}, 25.0 * rangeBearingNoise);
    auto const updated = Emrys::update(prior, observe(method, makeRangeBearingModel(noise), rangeAndBearing));
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 2u);

    auto const weights = std::array<double, 2>{0.8, 0.2};
    auto logLikelihoods = std::array<double, 2>{};
    std::size_t k = 0;
    for (auto const & entry : noise)
    {
        auto const alone =
            Emrys::update(prior, observe(method, makeRangeBearingModel(entry.component), rangeAndBearing));
        ASSERT_TRUE(alone);
        auto const & branch = updated->branches[k];
        EXPECT_EQ(branch.component, Eigen::Index(k));
        EXPECT_LT((branch.posterior.mean() - alone->branches.front().posterior.mean()).norm(), 1e-14);
        EXPECT_LT(
            (branch.posterior.covarianceLower() - alone->branches.front().posterior.covarianceLower()).norm(),
            1e-14
        );
        EXPECT_EQ(branch.iterations, alone->branches.front().iterations);
        EXPECT_TRUE(branch.converged);
        logLikelihoods[k] = alone->logLikelihood();
        ++k;
    }
    // The two components' posteriors differ, as each iterated to its own mode.
    EXPECT_GT((updated->branches[0].posterior.mean() - updated->branches[1].posterior.mean()).norm(), 1e-3);
    double const evidence =
        std::log(weights[0] * std::exp(logLikelihoods[0]) + weights[1] * std::exp(logLikelihoods[1]));
    EXPECT_NEAR(updated->logLikelihood(), evidence, 1e-12);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(updated->branches[i].logWeight, std::log(weights[i]) + logLikelihoods[i] - evidence, 1e-12);
    }
}

// On a unit quaternion the iterated update is Gauss-Newton in the prior's chart, mu [+] delta: the
// posterior mean is mu [+] delta* with delta* the Gauss-Newton solution of
// |delta|^2_P^-1 + |g(mu [+] delta)|^2_R^-1, with g = z - h the residual, here with its Jacobian in delta by central
// differences in plain Eigen, and the covariance is (P^-1 + J^T R^-1 J)^-1 carried to the tangent
// space at the posterior mean by the reset Jacobian [covariance-reset]. The tolerance reflects the differences'
// truncation error. One linearization lands well away from it.
TEST(IteratedUpdate, ReachesGaussNewtonInThePriorsChartOnAnAttitude)
{
    using Vector7 = Eigen::Matrix<double, 7, 1>;
    using Matrix7 = Eigen::Matrix<double, 7, 7>;
    using Jacobian3 = Eigen::Matrix<double, 3, 7>;
    auto const space = Motion{};
    GaussianParams<Motion> const prior = makeAttitudeInitial();
    auto const method = methods::iterated(methods::linearized(), {.tolerance = 1e-12, .maxIterations = 50});
    auto const direction = makeDirection(6, method);
    auto const updated = Emrys::update(prior, direction);
    ASSERT_TRUE(updated);
    auto const & branch = updated->branches.front();
    EXPECT_TRUE(branch.converged);

    // The residual at mu [+] delta, from the measurement's own function.
    auto const residual = [&](Vector7 const & delta) {
        Eigen::Matrix<double, 8, 1> const x = Emrys::plus(space, prior.mean(), delta);
        Eigen::Vector4d const q = x.head<4>();
        return Eigen::Vector3d{
            direction.observation() - direction.model().function()(Emrys::Composite<Orientation>{}, Linear<3>{}, q)
        };
    };
    auto const jacobianAt = [&](Vector7 const & delta) {
        auto result = Jacobian3{};
        double const h = 1e-6;
        for (int j = 0; j < 7; ++j)
        {
            Vector7 const step = h * Vector7::Unit(j);
            result.col(j) = (residual(delta + step) - residual(delta - step)) / (2 * h);
        }
        return result;
    };
    Matrix7 const info = prior.covarianceLower().inverse();
    Eigen::Matrix3d const noiseInfo = 1000.0 * Eigen::Matrix3d::Identity();
    Vector7 delta = Vector7::Zero();
    for (int i = 0; i < 50; ++i)
    {
        Jacobian3 const j = jacobianAt(delta);
        Matrix7 const normal = info + j.transpose() * noiseInfo * j;
        delta -= normal.ldlt().solve(j.transpose() * noiseInfo * residual(delta) + info * delta);
    }
    Jacobian3 const j = jacobianAt(delta);
    Matrix7 const inChart = (info + j.transpose() * noiseInfo * j).inverse();
    Matrix7 const reset = Emrys::resetJacobian(space, prior.mean(), delta);
    Matrix7 const expectedCov = reset * inChart * reset.transpose();
    Eigen::Matrix<double, 8, 1> const expectedMean = Emrys::plus(space, prior.mean(), delta);

    EXPECT_LT(Emrys::minus(space, branch.posterior.mean(), expectedMean).norm(), 1e-7);
    EXPECT_LT((branch.posterior.covarianceLower() - expectedCov).norm(), 1e-7);
    auto const once = Emrys::update(prior, makeDirection(6));
    ASSERT_TRUE(once);
    EXPECT_GT(Emrys::minus(space, once->branches.front().posterior.mean(), expectedMean).norm(), 1e-3);
}

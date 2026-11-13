// The Iterated method of Models/Transform.h with a sigma-point method, the iterated posterior
// linearization filter's update, through update() in Models/Measurement.h (roadmap step 17e;
// decisions.md [sigma-point-sets]): n iterations are n steps of algorithm 1 of García-Fernández et al. (2015) in
// plain Eigen; one iteration is the sigma-point update; and on an affine measurement it stops at
// the second linearization with the Kalman update. The forms are in iterated_forms.cpp.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include "Models/iterated_problem.h"

#include <cmath>
#include <type_traits>
#include <cstddef>
#include <utility>

using namespace EmrysTest::IteratedProblem;

using Emrys::KalmanFilter;

namespace methods = Emrys::methods;

namespace
{

// The iterated posterior linearization filter's update, algorithm 1 of García-Fernández et al.
// (2015), in plain Eigen with the cubature rule over the position: at each iteration the
// statistical linearization of h over N(u_i, W_i), A = Psi^T W^-1, b = z_bar - A u, and
// Omega = Phi - A W A^T, then the Kalman update of the prior under it. The root of W is the
// Cholesky factor, the root the library's default axes take ([sigma-point-axes]), since the
// cubature rule's result depends on the root. Returns the posterior after iterations linearizations.
std::pair<Vector4, Matrix4>
posteriorLinearization(GaussianParams<State> const & prior, int iterations)
{
    Vector4 const mu = prior.mean();
    Matrix4 const p = prior.covarianceLower();
    Vector4 u = mu;
    Matrix4 w = p;
    for (int i = 0; i < iterations; ++i)
    {
        Eigen::Vector2d const center = u.head<2>();
        Matrix2 const spread = w.topLeftCorner<2, 2>();
        Matrix2 const root = Eigen::LLT<Matrix2>{spread}.matrixL();
        auto points = Eigen::Matrix<double, 2, 4>{};
        points << std::sqrt(2.0) * root, -std::sqrt(2.0) * root;
        points.colwise() += center;
        auto outputs = Eigen::Matrix<double, 2, 4>{};
        for (int j = 0; j < 4; ++j)
        {
            Vector4 x = Vector4::Zero();
            x.head<2>() = points.col(j);
            outputs.col(j) = rangeBearingOf(x).first;
        }
        Eigen::Vector2d const zBar = outputs.rowwise().mean();
        Matrix2 const psi = (points.colwise() - center) * (outputs.colwise() - zBar).transpose() / 4.0;
        Matrix2 const phi = (outputs.colwise() - zBar) * (outputs.colwise() - zBar).transpose() / 4.0;
        Matrix2 const a = spread.ldlt().solve(psi).transpose();
        Eigen::Vector2d const b = zBar - a * center;
        Matrix2 const omega = phi - a * spread * a.transpose();

        Jacobian h = Jacobian::Zero();
        h.leftCols<2>() = a;
        Matrix2 const s = h * p * h.transpose() + omega + rangeBearingNoise;
        Eigen::Matrix<double, 4, 2> const gain = p * h.transpose() * s.inverse();
        u = mu + gain * (rangeAndBearing - a * mu.head<2>() - b);
        w = p - gain * s * gain.transpose();
    }
    return {u, w};
}

}  // namespace

// With a sigma-point method, n iterations are n steps of the iterated posterior linearization
// filter computed in plain Eigen, mean and covariance, to rounding: each relinearizes over the
// last posterior, its mean and its covariance, and updates the prior under it. The tolerance
// allows for the two computing the same moments in different orders.
TEST(IteratedPosteriorLinearization, NIterationsAreNStepsOfAlgorithm1)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&]<std::size_t n>(std::integral_constant<std::size_t, n>) {
        auto const capped = methods::iterated(methods::cubature(), {.tolerance = 0.0, .maxIterations = n});
        auto const updated = Emrys::update(prior, makeRangeBearing(rangeAndBearing, capped));
        ASSERT_TRUE(updated);
        auto const & branch = updated->branches.front();
        EXPECT_EQ(branch.iterations, n);
        auto const [expectedMean, expectedCov] = posteriorLinearization(prior, int(n));
        EXPECT_LT((branch.posterior.mean() - expectedMean).norm(), 1e-10) << n;
        EXPECT_LT((branch.posterior.covarianceLower() - expectedCov).norm(), 1e-10) << n;
    };
    check(std::integral_constant<std::size_t, 1>{});
    check(std::integral_constant<std::size_t, 2>{});
    check(std::integral_constant<std::size_t, 3>{});
    check(std::integral_constant<std::size_t, 5>{});
    // Iterating moves the posterior well away from the one-step cubature update.
    auto const once = Emrys::update(prior, makeRangeBearing(rangeAndBearing, methods::cubature()));
    ASSERT_TRUE(once);
    EXPECT_GT((once->branches.front().posterior.mean() - posteriorLinearization(prior, 5).first).norm(), 1e-2);
}

// One iteration of a sigma-point method is that method's own update, to rounding rather than to
// the bit, since it takes the statistical linearization A = Sigma_Sy^T P_SS^-1 and rebuilds
// A P_SS A^T + Omega where the update reads Sigma_yy: with Cubature and Unscented, in a
// KalmanFilter too, and with input noise.
TEST(IteratedPosteriorLinearization, OneIterationIsTheSigmaPointUpdate)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const check = [&](auto const & iterated, auto const & plain) {
        auto const a = Emrys::update(prior, iterated);
        auto const b = Emrys::update(prior, plain);
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        EXPECT_LT((a->branches.front().posterior.mean() - b->branches.front().posterior.mean()).norm(), 1e-12);
        EXPECT_LT(
            (a->branches.front().posterior.covarianceLower() - b->branches.front().posterior.covarianceLower()).norm(),
            1e-12
        );
        EXPECT_NEAR(a->logLikelihood(), b->logLikelihood(), 1e-12);
        EXPECT_EQ(a->branches.front().iterations, 1u);

        auto first = KalmanFilter{makeModel(), prior};
        auto second = KalmanFilter{makeModel(), prior};
        ASSERT_TRUE(first.update(iterated));
        ASSERT_TRUE(second.update(plain));
        EXPECT_LT((first.state().mean() - second.state().mean()).norm(), 1e-12);
        EXPECT_LT((first.state().covarianceLower() - second.state().covarianceLower()).norm(), 1e-12);
    };
    check(
        makeRangeBearing(
            rangeAndBearing,
            methods::iterated(methods::cubature(), {.tolerance = 1e-3, .maxIterations = 1})
        ),
        makeRangeBearing(rangeAndBearing, methods::cubature())
    );
    check(
        makeRangeBearing(
            rangeAndBearing,
            methods::iterated(methods::unscented(), {.tolerance = 1e-3, .maxIterations = 1})
        ),
        makeRangeBearing(rangeAndBearing, methods::unscented())
    );
    check(
        makeRangeBearingWithInputNoise(
            rangeAndBearing,
            methods::iterated(methods::cubature(), {.tolerance = 1e-3, .maxIterations = 1})
        ),
        makeRangeBearingWithInputNoise(rangeAndBearing, methods::cubature())
    );
}

// On an affine measurement the statistical linearization is the same over every posterior, so
// the first step is the Kalman update's and the second moves nothing, as García-Fernández et al.
// (2015), section IV-E, note: a position fix under Iterated<GaussHermite<3>> stops at the
// linearization after that second step, the third, equals the fix under Linearized to rounding,
// and reports that the tolerance stopped it.
TEST(IteratedPosteriorLinearization, StopsAtOnceOnAnAffineMeasurement)
{
    GaussianParams<State> const prior = makePoorPrior();
    auto const iterated =
        Emrys::update(prior, makeFix(Eigen::Vector2d{4.0, 3.0}, methods::iterated(methods::gaussHermite<3>())));
    auto const linear = Emrys::update(prior, makeFix(Eigen::Vector2d{4.0, 3.0}, methods::linearized()));
    ASSERT_TRUE(iterated);
    ASSERT_TRUE(linear);
    auto const & branch = iterated->branches.front();
    EXPECT_EQ(branch.iterations, 3u);
    EXPECT_TRUE(branch.converged);
    EXPECT_LT((branch.posterior.mean() - linear->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (branch.posterior.covarianceLower() - linear->branches.front().posterior.covarianceLower()).norm(),
        1e-12
    );
}

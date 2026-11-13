// methods::spread() of Models/Methods.h (roadmap 17t, [sigma-point-axes]): a rule spread by the default axes is the
// rule; along(d) puts the first pair of points on the line through the mean along d; the axes leave a linear
// function's moments alone and move a nonlinear one's; the axes of the user's own in tests/user_axes.h spread a
// transform's points; and a spread rule is a method wherever a rule is, in an update and as the method an Iterated
// relinearizes. That every form now spreads the same points is in sqrt_filters.cpp, info_filter.cpp, and
// iterated_forms.cpp, whose Kalman tolerances are rounding.

#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "Models/iterated_problem.h"
#include "user_axes.h"

namespace methods = Emrys::methods;
namespace roots = Emrys::roots;

using Emrys::withCross;

namespace
{

using Three = Emrys::Linear<3>;

[[nodiscard]] Emrys::GaussianParams<Three>
makeInput()
{
    Eigen::Matrix3d a;
    a << 1.0, 0.0, 0.0, 0.6, 0.8, 0.0, -0.3, 0.5, 0.7;
    return {Eigen::Vector3d{1.0, -0.5, 2.0}, (a * a.transpose()).eval()};
}

// A function bent enough that the moments of a sigma-point rule depend on its root.
Eigen::Vector2d
bent(Eigen::Vector3d const & x)
{
    return Eigen::Vector2d{std::sin(x(0)) * x(1), std::exp(0.3 * x(2)) + x(0) * x(0) * x(0)};
}

// An affine function, whose moments every root gives exactly.
Eigen::Vector2d
affine(Eigen::Vector3d const & x)
{
    return Eigen::Vector2d{x(0) - 2.0 * x(1) + 0.5, 3.0 * x(2) + x(1)};
}

template <typename A, typename B>
[[nodiscard]] double
distance(A const & a, B const & b)
{
    return (a.output().mean() - b.output().mean()).norm()
         + (a.output().covarianceLower() - b.output().covarianceLower()).norm()
         + (a.crossCovariance() - b.crossCovariance()).norm();
}

}  // namespace

TEST(Spread, ByTheDefaultAxesIsTheRule)
{
    auto const x = makeInput();
    auto const check = [&](auto const & rule) {
        auto const plain = Emrys::transform(rule, bent, withCross(x));
        auto const spread =
            Emrys::transform(methods::spread(rule, roots::cholesky(roots::pivoted())), bent, withCross(x));
        EXPECT_EQ(distance(plain, spread), 0.0);
    };
    check(methods::cubature());
    check(methods::unscented());
    check(methods::sphericalSimplex());
    check(methods::gaussHermite<3>());
}

TEST(Spread, AlongPutsTheFirstPairOnTheLineThroughTheMean)
{
    auto const x = makeInput();
    Eigen::Vector3d const d{1.0, 1.0, -0.5};
    std::vector<Eigen::Vector3d> seen;
    auto const record = [&](Eigen::Vector3d const & v) {
        seen.push_back(v);
        return bent(v);
    };
    [[maybe_unused]] auto const moments =
        Emrys::transform(methods::spread(methods::cubature(), roots::along(d, roots::pivoted())), record, withCross(x));
    ASSERT_EQ(seen.size(), 6U);
    // The cubature rule's plus points come first, its first at mu + sqrt(3) s_1.
    double const length = std::sqrt(3.0) * d.norm() / std::sqrt(d.dot(x.covarianceLower().llt().solve(d)));
    Eigen::Vector3d const first = seen[0] - x.mean();
    Eigen::Vector3d const opposite = seen[3] - x.mean();
    EXPECT_LE((first - length * d.normalized()).norm(), 1e-12);
    EXPECT_LE((opposite + length * d.normalized()).norm(), 1e-12);
}

TEST(Spread, TheAxesMoveANonlinearFunctionsMomentsAndNotAnAffineOnes)
{
    auto const x = makeInput();
    auto const check = [&](auto const & axes) {
        auto const spread = methods::spread(methods::cubature(), axes);
        EXPECT_LE(
            distance(
                Emrys::transform(spread, affine, withCross(x)),
                Emrys::transform(methods::cubature(), affine, withCross(x))
            ),
            1e-12
        );
        EXPECT_GT(
            distance(
                Emrys::transform(spread, bent, withCross(x)),
                Emrys::transform(methods::cubature(), bent, withCross(x))
            ),
            1e-6
        );
    };
    check(roots::principal());
    check(roots::symmetric());
    check(roots::along(Eigen::Vector3d{0.0, 1.0, 1.0}, roots::pivoted()));
    check(roots::withFallbacks(EmrysTest::ReversedCholesky{}, roots::pivoted()));
}

TEST(Spread, ASpreadRuleIsAnUpdatesMethodAndIteratedRelinearizesIt)
{
    using namespace EmrysTest::IteratedProblem;
    // The poor prior with its position correlated, so that its principal axes are not its Cholesky factor's.
    Matrix4 cov = makePoorPrior().covarianceLower();
    cov(1, 0) = 1.5;
    auto const prior = GaussianParams<State>{State{}, makePoorPrior().mean(), cov};
    auto const spread = methods::spread(methods::cubature(), roots::principal());
    auto const once = Emrys::update(prior, makeRangeBearing(rangeAndBearing, spread));
    auto const iteratedOnce = Emrys::update(
        prior,
        makeRangeBearing(rangeAndBearing, methods::iterated(spread, {.tolerance = 0.0, .maxIterations = 1}))
    );
    auto const plain = Emrys::update(prior, makeRangeBearing(rangeAndBearing, methods::cubature()));
    ASSERT_TRUE(once && iteratedOnce && plain);
    auto const & posterior = once->branches.front().posterior;
    EXPECT_LE((iteratedOnce->branches.front().posterior.mean() - posterior.mean()).norm(), 1e-12);
    EXPECT_LE((iteratedOnce->branches.front().posterior.covarianceLower() - posterior.covarianceLower()).norm(), 1e-12);
    EXPECT_GT((plain->branches.front().posterior.mean() - posterior.mean()).norm(), 1e-9);
    // Iterated to convergence under the principal axes too.
    auto const converged = Emrys::update(prior, makeRangeBearing(rangeAndBearing, methods::iterated(spread)));
    ASSERT_TRUE(converged);
    EXPECT_TRUE(converged->branches.front().converged);
}

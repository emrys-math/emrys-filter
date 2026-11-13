// The sigma-point sets of Models/Methods.h, Cubature, SphericalSimplex, and GaussHermite, as
// methods beside Unscented (roadmap step 17e; decisions.md [sigma-point-sets]): each set's points have zero mean
// and identity covariance with weights summing to one; each reproduces the mean, covariance, and
// cross-covariance of an affine function exactly; the cubature rule is Unscented with alpha = 1,
// beta = 0, kappa = 0; the spherical simplex's points other than the center lie on one sphere;
// the one-dimensional Gauss-Hermite rule is the tabulated one and the product rule integrates the
// polynomials its degree promises exactly; both third-degree rules give a quadratic's mean
// exactly and Gauss-Hermite of order 3 its variance too; and each rule's parameters are compile-time constants while an
// Iterated's tolerance and cap are the run-time values given to iterated() ([policy-factories]), and the Unscented and
// cubature weights are constants for a float or double scalar over a fixed number of coordinates ([embedded-builds]).
// A set of the user's own (tests/user_point_set.h) meets PointSet by its interface, runs through
// every check the library's sets do, and equals the Unscented it is ([user-extension-points]).
// An UnscentedRule chooses the parameters by the number of coordinates: FourthMomentUnscented
// gives the variance of a square exactly over every number, at compile time and at run time alike,
// where Unscented<> does over one alone, and a rule of the user's own (tests/user_unscented_rule.h)
// meets UnscentedRule by its interface and equals the fixed parameters it gives at each number.
// Headers under test: Models/Methods.h, Models/Transform.h.

#include <EmrysFilter/Models/Transform.h>

#include "user_point_set.h"
#include "user_unscented_rule.h"

#include <unsupported/Eigen/AutoDiff>

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <type_traits>

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ScalarOf;
using Emrys::withCross;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

GaussianParams<State>
makeState()
{
    auto const cov =
        Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.3}, {0.1, -0.1, 0.5, 0.05}, {0.0, 0.3, 0.05, 0.8}};
    return {State{}, Eigen::Vector4d{1.0, -2.0, 0.5, 0.25}, cov};
}

// The sets under test, with a spherical simplex of zero and of nonzero center weight, and the
// user's own symmetric set.
auto const sets = std::tuple{
    methods::cubature(),
    methods::sphericalSimplex(),
    methods::sphericalSimplex<0.3>(),
    methods::gaussHermite<2>(),
    methods::gaussHermite<3>(),
    EmrysTest::SymmetricSet<0.5>{}
};

// Calls check with each set in sets.
template <typename Check>
void
forEachSet(Check const & check)
{
    std::apply(
        [&](auto const &... set) {
            (check(set), ...);
        },
        sets
    );
}

}  // namespace

// Each set's points, over 1 to 4 coordinates and over none, have weights summing to one, a
// weighted mean of zero, and a weighted second moment of the identity, which is what makes the
// points of a root S of P have mean mu and covariance P.
TEST(SigmaPointSets, MatchTheFirstTwoMomentsOfAStandardNormal)
{
    forEachSet([](auto const & set) {
        for (Eigen::Index n = 0; n <= 4; ++n)
        {
            auto const unit = Emrys::internal::unitPoints<double, Eigen::Dynamic, Eigen::Dynamic>(set, n);
            EXPECT_NEAR(unit.weights.sum(), 1.0, 1e-14);
            Eigen::VectorXd const mean = unit.points * unit.weights;
            Eigen::MatrixXd const second = unit.points * unit.weights.asDiagonal() * unit.points.transpose();
            EXPECT_LT(mean.norm(), 1e-14);
            EXPECT_LT((second - Eigen::MatrixXd::Identity(n, n)).norm(), 1e-13);
        }
    });
}

// Through an affine function with a marked input, every set gives the closed form to rounding:
// mean A mu + b, covariance A P A^T, and cross-covariance P A^T.
TEST(SigmaPointSets, AreExactForAnAffineFunction)
{
    GaussianParams<State> const state = makeState();
    auto const a = Eigen::Matrix<double, 3, 4>{{1.0, 0.5, 0.0, -1.0}, {0.0, 2.0, 1.0, 0.0}, {0.3, 0.0, 0.0, 1.0}};
    auto const b = Eigen::Vector3d{0.5, -1.0, 2.0};
    auto const f = [&]<typename T>(auto const & x) {
        return Eigen::Matrix<T, 3, 1>{a * x + b};
    };
    forEachSet([&](auto const & set) {
        auto const result = Emrys::transform(set, f, withCross(state));
        EXPECT_LT((result.output().mean() - (a * state.mean() + b)).norm(), 1e-12);
        EXPECT_LT((result.output().covarianceLower() - a * state.covarianceLower() * a.transpose()).norm(), 1e-12);
        EXPECT_LT((result.crossCovariance() - state.covarianceLower() * a.transpose()).norm(), 1e-12);
    });
}

// The cubature rule is the unscented transform with alpha = 1, beta = 0, and kappa = 0, whose
// center point weighs nothing (Arasaratnam and Haykin): the two agree to rounding on a nonlinear
// function, though only the unscented one evaluates the center.
TEST(SigmaPointSets, CubatureIsUnscentedWithAZeroCenterWeight)
{
    GaussianParams<State> const state = makeState();
    auto const f = []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 1>{
            Eigen::numext::sqrt(T(x(0) * x(0) + x(1) * x(1))),
            Eigen::numext::sin(x(2)) * x(3)
        };
    };
    auto const cubature = Emrys::transform(methods::cubature(), f, withCross(state));
    auto const unscented = Emrys::transform(methods::unscented<1.0, 0.0, 0.0>(), f, withCross(state));
    EXPECT_LT((cubature.output().mean() - unscented.output().mean()).norm(), 1e-12);
    EXPECT_LT((cubature.output().covarianceLower() - unscented.output().covarianceLower()).norm(), 1e-12);
    EXPECT_LT((cubature.crossCovariance() - unscented.crossCovariance()).norm(), 1e-12);
}

// A rule's parameters are compile-time constants, as a periodic coordinate's interval is, so a
// sigma-point method is an empty tag whose parameters read back as static members, while an
// Iterated's tolerance and cap are the run-time values given to iterated(), 1e-3 and 10 unless
// given. Unscented<> computes its weights in the input's scalar: a float Gaussian transforms to a
// float result, within float rounding (1e-4 relative) of the double one.
TEST(SigmaPointSets, RuleParametersAreCompileTimeConstants)
{
    static_assert(std::is_same_v<decltype(methods::unscented()), methods::Unscented<1.0, 2.0, 0.0>>);
    static_assert(std::is_empty_v<methods::Unscented<0.5>> && std::is_empty_v<methods::SphericalSimplex<0.3>>);
    static_assert(
        methods::Unscented<0.5>::alpha == 0.5 && methods::Unscented<0.5>::beta == 2.0
        && methods::Unscented<0.5>::kappa == 0.0
    );
    static_assert(methods::SphericalSimplex<0.3>::centerWeight == 0.3);
    auto const iterate = methods::iterated(methods::unscented<0.5>(), {.tolerance = 1e-6, .maxIterations = 4});
    static_assert(std::is_same_v<decltype(iterate), methods::Iterated<methods::Unscented<0.5>> const>);
    EXPECT_EQ(iterate.tolerance, 1e-6);
    EXPECT_EQ(iterate.maxIterations, 4u);
    EXPECT_EQ(methods::iterated().tolerance, 1e-3);
    EXPECT_EQ(methods::iterated().maxIterations, 10u);

    GaussianParams<State> const state = makeState();
    auto const f = []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 1>{
            Eigen::numext::sqrt(T(x(0) * x(0) + x(1) * x(1))),
            Eigen::numext::sin(x(2)) * x(3)
        };
    };
    auto const floatState = GaussianParams<State, float>{
        State{},
        Eigen::Vector4f{state.mean().cast<float>()},
        Eigen::Matrix4f{state.covarianceLower().cast<float>()}
    };
    auto const single = Emrys::transform(methods::unscented(), f, withCross(floatState));
    auto const twice = Emrys::transform(methods::unscented(), f, withCross(state));
    static_assert(std::is_same_v<typename std::remove_cvref_t<decltype(single.output())>::Scalar, float>);
    EXPECT_LT(
        (single.output().mean().cast<double>() - twice.output().mean()).norm(),
        1e-4 * twice.output().mean().norm()
    );
    EXPECT_LT(
        (single.output().covarianceLower().cast<double>() - twice.output().covarianceLower()).norm(),
        1e-4 * twice.output().covarianceLower().norm()
    );
}

// The spherical simplex's points other than the center lie at sqrt(n / (1 - W0)) from it, the
// radius the construction of Julier and Uhlmann (2004), appendix III, gives, and there are n + 2.
TEST(SigmaPointSets, SphericalSimplexPointsLieOnOneSphere)
{
    auto const check = [&]<double centerWeight>(methods::SphericalSimplex<centerWeight> const & set) {
        for (Eigen::Index n = 1; n <= 5; ++n)
        {
            auto const unit = Emrys::internal::unitPoints<double, Eigen::Dynamic, Eigen::Dynamic>(set, n);
            ASSERT_EQ(unit.points.cols(), n + 2);
            EXPECT_EQ(unit.weights(0), centerWeight);
            EXPECT_LT(unit.points.col(0).norm(), 1e-15);
            for (Eigen::Index j = 1; j < n + 2; ++j)
            {
                EXPECT_NEAR(unit.points.col(j).norm(), std::sqrt(double(n) / (1.0 - centerWeight)), 1e-13);
            }
        }
    };
    check(methods::sphericalSimplex());
    check(methods::sphericalSimplex<0.3>());
}

// Over a fixed number of coordinates a float or double scalar takes the Unscented weights and the
// cubature radius and weight as constants the compiler computed (internal::takesDoubleConstants),
// and those match the run-time computation over the same number given at run time to within a few
// ulps, the compile-time square root being within one of std::sqrt. The simplex and Gauss-Hermite
// points, computed at run time either way, are compared too.
TEST(SigmaPointSets, CompileTimeWeightsMatchTheRunTimeOnes)
{
    using Emrys::internal::sigmaWeights;
    using Emrys::internal::unitPoints;
    constexpr int Dynamic = Eigen::Dynamic;
    static_assert(Emrys::internal::unscentedWeights<float, methods::Unscented<0.5>, 3>().point > 0.0f);
    // Over four coordinates the fourth-moment rule's center weight, 1 - 4 / 3, is negative.
    static_assert(Emrys::internal::unscentedWeights<float, methods::FourthMomentUnscented, 4>().center < 0.0f);
    static_assert(!Emrys::internal::takesDoubleConstants<Eigen::AutoDiffScalar<Eigen::Vector2d>>);

    auto const fixed = sigmaWeights<double, 3>(methods::unscented<0.5>(), 3);
    auto const dynamic = sigmaWeights<double, Dynamic>(methods::unscented<0.5>(), 3);
    EXPECT_NEAR(fixed.scale, dynamic.scale, 1e-15);
    EXPECT_EQ(fixed.center, dynamic.center);
    EXPECT_EQ(fixed.centerCov, dynamic.centerCov);
    EXPECT_EQ(fixed.point, dynamic.point);
    auto const fixedRule = sigmaWeights<double, 4>(methods::fourthMomentUnscented(), 4);
    auto const dynamicRule = sigmaWeights<double, Dynamic>(methods::fourthMomentUnscented(), 4);
    EXPECT_NEAR(fixedRule.scale, dynamicRule.scale, 1e-15);
    EXPECT_NEAR(fixedRule.center, dynamicRule.center, 1e-15);
    EXPECT_NEAR(fixedRule.centerCov, dynamicRule.centerCov, 1e-15);
    EXPECT_NEAR(fixedRule.point, dynamicRule.point, 1e-15);

    auto const compare = [](auto const & set) {
        auto const constant = unitPoints<double, 4, 4>(set, 4);
        auto const computed = unitPoints<double, Dynamic, Dynamic>(set, 4);
        EXPECT_LT((constant.points - computed.points).cwiseAbs().maxCoeff(), 1e-14);
        EXPECT_LT((constant.weights - computed.weights).cwiseAbs().maxCoeff(), 1e-15);
        EXPECT_EQ(constant.reference, computed.reference);
    };
    compare(methods::cubature());
    compare(methods::sphericalSimplex<0.3>());
    compare(methods::gaussHermite<3>());
    compare(EmrysTest::SymmetricSet<0.5>{});
}

// The one-dimensional rules match the tabulated Gauss-Hermite rules for the weight N(0, 1): one
// point at 0; points +-1 with weights 1/2; and points 0 and +-sqrt(3) with weights 2/3 and 1/6.
TEST(SigmaPointSets, GaussHermiteMatchesTheTabulatedRules)
{
    auto const one = Emrys::internal::unitPoints<double, 1, 1>(methods::gaussHermite<1>(), 1);
    EXPECT_NEAR(one.points(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(one.weights(0), 1.0, 1e-15);

    auto const two = Emrys::internal::unitPoints<double, 1, 1>(methods::gaussHermite<2>(), 1);
    EXPECT_NEAR(two.points(0, 0), -1.0, 1e-14);
    EXPECT_NEAR(two.points(0, 1), 1.0, 1e-14);
    EXPECT_NEAR(two.weights(0), 0.5, 1e-14);
    EXPECT_NEAR(two.weights(1), 0.5, 1e-14);

    auto const three = Emrys::internal::unitPoints<double, 1, 1>(methods::gaussHermite<3>(), 1);
    EXPECT_NEAR(three.points(0, 0), -std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(three.points(0, 1), 0.0, 1e-14);
    EXPECT_NEAR(three.points(0, 2), std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(three.weights(0), 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(three.weights(1), 2.0 / 3.0, 1e-14);
    EXPECT_NEAR(three.weights(2), 1.0 / 6.0, 1e-14);
    EXPECT_EQ(three.reference, 1);
}

// The product rule of order p integrates x1^a x2^b against N(0, I) exactly for a, b <= 2p - 1:
// the moment is (a - 1)!! (b - 1)!! for even a and b and zero otherwise.
TEST(SigmaPointSets, GaussHermiteIntegratesPolynomialsOfItsDegree)
{
    auto const doubleFactorial = [](int k) {
        double result = 1;
        for (int i = k; i > 1; i -= 2)
        {
            result *= i;
        }
        return result;
    };
    auto const moment = [&](int a) {
        return a % 2 == 1 ? 0.0 : doubleFactorial(a - 1);
    };
    auto const check = [&]<int Order>(methods::GaussHermite<Order> const & set) {
        auto const unit = Emrys::internal::unitPoints<double, 2, 2>(set, 2);
        ASSERT_EQ(unit.points.cols(), Order * Order);
        for (int a = 0; a <= 2 * Order - 1; ++a)
        {
            for (int b = 0; b <= 2 * Order - 1; ++b)
            {
                double sum = 0;
                for (Eigen::Index j = 0; j < unit.points.cols(); ++j)
                {
                    sum += unit.weights(j) * std::pow(unit.points(0, j), a) * std::pow(unit.points(1, j), b);
                }
                EXPECT_NEAR(sum, moment(a) * moment(b), 1e-10 * (1.0 + moment(a) * moment(b)));
            }
        }
    };
    check(methods::gaussHermite<1>());
    check(methods::gaussHermite<2>());
    check(methods::gaussHermite<3>());
    check(methods::gaussHermite<5>());
}

// For a quadratic x^T A x of a Gaussian x, every third-degree rule gives the mean tr(A P) +
// mu^T A mu exactly, as the cubature and unscented rules both integrate degree three; the variance,
// 2 tr(A P A P) + 4 mu^T A P A mu, needs degree four, which Gauss-Hermite of order 3 has and the
// cubature rule has not.
TEST(SigmaPointSets, AgreeOnAQuadraticsMeanAndGaussHermiteOnItsVariance)
{
    GaussianParams<State> const state = makeState();
    auto const a =
        Eigen::Matrix4d{{1.0, 0.2, 0.0, 0.0}, {0.2, 0.5, 0.1, 0.0}, {0.0, 0.1, 2.0, 0.3}, {0.0, 0.0, 0.3, 1.0}};
    auto const f = [&]<typename T>(auto const & x) {
        return Eigen::Matrix<T, 1, 1>{x.dot(a * x)};
    };
    Eigen::Matrix4d const p = state.covarianceLower();
    Eigen::Vector4d const mu = state.mean();
    double const mean = (a * p).trace() + mu.dot(a * mu);
    double const variance = 2.0 * (a * p * a * p).trace() + 4.0 * mu.dot(a * p * a * mu);
    forEachSet([&](auto const & set) {
        EXPECT_NEAR(Emrys::transform(set, f, state).mean()(0), mean, 1e-11);
    });
    EXPECT_NEAR(Emrys::transform(methods::unscented(), f, state).mean()(0), mean, 1e-11);
    EXPECT_NEAR(Emrys::transform(methods::gaussHermite<3>(), f, state).covarianceLower()(0, 0), variance, 1e-9);
    EXPECT_GT(std::abs(Emrys::transform(methods::cubature(), f, state).covarianceLower()(0, 0) - variance), 1e-3);
}

// A set of the user's own (tests/user_point_set.h) meets PointSet by its two static members alone, and so is a
// TransformMethod; the library's sets meet it and Unscented and Linearized don't. Its transform is what it claims:
// the symmetric set with center weight W0 over n coordinates is Unscented<1, 0, kappa> with kappa = n W0 / (1 - W0),
// here 4 for W0 = 0.5 over the four-coordinate state, to rounding on a nonlinear function.
TEST(SigmaPointSets, UserSetIsATransformMethod)
{
    using Set = EmrysTest::SymmetricSet<0.5>;
    static_assert(Emrys::PointSet<Set>);
    static_assert(Emrys::TransformMethod<Set>);
    static_assert(
        Emrys::PointSet<methods::Cubature> && Emrys::PointSet<methods::SphericalSimplex<>>
        && Emrys::PointSet<methods::GaussHermite<>>
    );
    static_assert(!Emrys::PointSet<methods::Unscented<>>);
    static_assert(!Emrys::PointSet<Emrys::methods::Linearized>);
    static_assert(!Emrys::PointSet<int>);
    static_assert(Emrys::internal::UnitPoints<double, Set, 4, 4>::Count == 9);

    GaussianParams<State> const state = makeState();
    auto const f = []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 1>{
            Eigen::numext::sqrt(T(x(0) * x(0) + x(1) * x(1))),
            Eigen::numext::sin(x(2)) * x(3)
        };
    };
    auto const symmetric = Emrys::transform(Set{}, f, withCross(state));
    auto const unscented = Emrys::transform(methods::unscented<1.0, 0.0, 4.0>(), f, withCross(state));
    EXPECT_LT((symmetric.output().mean() - unscented.output().mean()).norm(), 1e-12);
    EXPECT_LT((symmetric.output().covarianceLower() - unscented.output().covarianceLower()).norm(), 1e-12);
    EXPECT_LT((symmetric.crossCovariance() - unscented.crossCovariance()).norm(), 1e-12);
}

// For y = x0^2 of a Gaussian x with a diagonal covariance, the variance is 4 mu0^2 P00 + 2 P00^2, which needs the
// fourth moment along x0. FourthMomentUnscented puts the points at sqrt(3) standard deviations whatever the number
// of coordinates, so it gives that exactly over every number: over four fixed at compile time, and over one to six
// given at run time, where its center weight 1 - n / 3 is negative from four on. Unscented<> spreads its points by
// sqrt(n) and gives P00^2 (n + 1) for the quadratic term, a hand computation, so 4 mu0^2 P00 + 5 P00^2 over four.
TEST(UnscentedRules, FourthMomentGivesTheVarianceOfASquareOverEveryDimension)
{
    using Fixed = Linear<4>;
    using Resizable = Linear<Eigen::Dynamic>;
    static_assert(Emrys::UnscentedRule<methods::FourthMomentUnscented> && Emrys::UnscentedRule<methods::Unscented<>>);
    static_assert(Emrys::TransformMethod<methods::FourthMomentUnscented>);
    static_assert(!Emrys::PointSet<methods::FourthMomentUnscented> && !Emrys::UnscentedRule<methods::Cubature>);

    auto const f = []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 1, 1>{x(0) * x(0)};
    };
    double const mu0 = 0.5;
    double const p00 = 2.0;
    double const mean = mu0 * mu0 + p00;
    double const variance = 4.0 * mu0 * mu0 * p00 + 2.0 * p00 * p00;

    auto const mu = Eigen::Vector4d{mu0, -1.0, 2.0, 0.3};
    auto const p = Eigen::Matrix4d{Eigen::Vector4d{p00, 1.0, 0.5, 3.0}.asDiagonal()};
    auto const fixed = GaussianParams<Fixed>{Fixed{}, mu, p};
    auto const rule = Emrys::transform(methods::fourthMomentUnscented(), f, fixed);
    EXPECT_NEAR(rule.mean()(0), mean, 1e-12);
    EXPECT_NEAR(rule.covarianceLower()(0, 0), variance, 1e-11);
    auto const constant = Emrys::transform(methods::unscented(), f, fixed);
    EXPECT_NEAR(constant.mean()(0), mean, 1e-12);
    EXPECT_NEAR(constant.covarianceLower()(0, 0), 4.0 * mu0 * mu0 * p00 + 5.0 * p00 * p00, 1e-11);

    for (Eigen::Index n = 1; n <= 6; ++n)
    {
        Eigen::VectorXd mean0 = Eigen::VectorXd::Constant(n, 0.25);
        mean0(0) = mu0;
        Eigen::VectorXd diagonal = Eigen::VectorXd::LinSpaced(n, 0.5, 1.5);
        diagonal(0) = p00;
        auto const resizable = GaussianParams<Resizable>{mean0, Eigen::MatrixXd{diagonal.asDiagonal()}};
        auto const result = Emrys::transform(methods::fourthMomentUnscented(), f, resizable);
        EXPECT_NEAR(result.mean()(0), mean, 1e-12) << "over " << n << " coordinates";
        EXPECT_NEAR(result.covarianceLower()(0, 0), variance, 1e-11) << "over " << n << " coordinates";
    }
}

// A rule of the user's own (tests/user_unscented_rule.h) meets UnscentedRule by its static member alone, and so is a
// TransformMethod an Iterated holds. Its transform is what it claims, to rounding on a nonlinear function:
// FourthMomentUnscented over two coordinates and Unscented<1, 0, 0> over the four-coordinate state, in double and in
// float, where the compiler computes its weights.
TEST(UnscentedRules, UserRuleIsATransformMethod)
{
    using Rule = EmrysTest::ClampedFourthMoment;
    static_assert(Emrys::UnscentedRule<Rule> && Emrys::TransformMethod<Rule>);
    static_assert(Emrys::TransformMethod<methods::Iterated<Rule>>);
    static_assert(!Emrys::PointSet<Rule>);
    static_assert(Emrys::internal::unscentedWeights<float, Rule, 2>().center > 0.0f);

    auto const f = []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 1>{
            Eigen::numext::sqrt(T(x(0) * x(0) + x(1) * x(1))),
            Eigen::numext::sin(x(0)) * x(1)
        };
    };
    auto const expectSame = [](auto const & a, auto const & b, double tolerance) {
        EXPECT_LT((a.output().mean() - b.output().mean()).norm(), tolerance);
        EXPECT_LT((a.output().covarianceLower() - b.output().covarianceLower()).norm(), tolerance);
        EXPECT_LT((a.crossCovariance() - b.crossCovariance()).norm(), tolerance);
    };
    auto const position =
        GaussianParams<Position>{Position{}, Eigen::Vector2d{1.0, -2.0}, Eigen::Matrix2d{{1.0, 0.2}, {0.2, 2.0}}};
    expectSame(
        Emrys::transform(Rule{}, f, withCross(position)),
        Emrys::transform(methods::fourthMomentUnscented(), f, withCross(position)),
        1e-12
    );
    GaussianParams<State> const state = makeState();
    expectSame(
        Emrys::transform(Rule{}, f, withCross(state)),
        Emrys::transform(methods::cubature(), f, withCross(state)),
        1e-12
    );
    auto const floatState = GaussianParams<State, float>{
        State{},
        Eigen::Vector4f{state.mean().cast<float>()},
        Eigen::Matrix4f{state.covarianceLower().cast<float>()}
    };
    expectSame(
        Emrys::transform(Rule{}, f, withCross(floatState)),
        Emrys::transform(methods::unscented<1.0, 0.0, 0.0>(), f, withCross(floatState)),
        1e-6
    );
}

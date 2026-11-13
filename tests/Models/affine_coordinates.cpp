// Affine coordinates declared with a model ([affine-coordinates]): a sigma-point method spreads its
// points over the coordinates the model's affineIn() does not name and carries the named ones exactly by their
// Gaussian given the rest, with the function's Jacobian in them from what it returns, from a composition's parts, or
// by automatic differentiation, which agree at the same number of calls. A bilinear model and noise scaled by the
// state get Isserlis' moments from every sigma-point method; a sure input is not factored; Linearized ignores the
// declaration.

#include <cmath>
#include <tuple>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <EmrysFilter/Distributions.h>
#include <EmrysFilter/Models.h>
#include <EmrysFilter/Spaces.h>

using Emrys::additiveNoise;
using Emrys::affineIn;
using Emrys::block;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::path;
using Emrys::processModel;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::sure;
using Emrys::update;
using Emrys::withJacobian;
using Emrys::writes;
using Emrys::zeros;

namespace methods = Emrys::methods;

namespace
{

struct Scale : Linear<1>
{};

struct Value : Linear<1>
{};

struct Product : Linear<1>
{};

struct Noise : Linear<1>
{};

using Bilinear = Composite<Scale, Value, Product>;

// p' = s v, affine in v given s, with s and v kept: Isserlis' theorem gives its moments. The model
// declaring v affine, and the same model declaring nothing.
[[nodiscard]] auto
productModels()
{
    auto const f = []<typename T>(auto const & in, auto const & out, auto const & x, double) {
        auto y = zeros<T>(out);
        block<Product>(out, y) = block<Scale>(in, x)(0) * block<Value>(in, x);
        return y;
    };
    return std::tuple{
        compose(
            processModel(f, reads(path<Scale>, path<Value>), writes(path<Product>), affineIn(path<Value>)),
            Constant<Scale>{},
            Constant<Value>{}
        ),
        compose(
            processModel(f, reads(path<Scale>, path<Value>), writes(path<Product>)),
            Constant<Scale>{},
            Constant<Value>{}
        )
    };
}

[[nodiscard]] GaussianParams<Bilinear>
bilinearPrior()
{
    Eigen::Vector3d const mean{2.0, -1.0, 0.0};
    Eigen::Vector3d const variances{0.5, 0.3, 1.0};
    return {Bilinear{}, mean, Eigen::Matrix3d{variances.asDiagonal()}};
}

// The moments of s v for independent Gaussian s and v (Isserlis): mean mu_s mu_v, variance
// mu_s^2 s_v + mu_v^2 s_s + s_s s_v, and covariance mu_v s_s with s and mu_s s_v with v.
void
expectIsserlis(GaussianParams<Bilinear> const & predicted, double tolerance)
{
    double const ms = 2.0, mv = -1.0, ss = 0.5, sv = 0.3;
    Eigen::Matrix3d const cov = predicted.covarianceLower();
    EXPECT_NEAR(predicted.mean()(2), ms * mv, tolerance);
    EXPECT_NEAR(cov(2, 2), ms * ms * sv + mv * mv * ss + ss * sv, tolerance);
    EXPECT_NEAR(cov(2, 0), mv * ss, tolerance);
    EXPECT_NEAR(cov(2, 1), ms * sv, tolerance);
}

}  // namespace

// Every sigma-point method factors the declared coordinate out of the bilinear model and gets Isserlis' moments,
// where the full transform over the same points does not.
TEST(AffineCoordinates, EverySigmaPointMethodGivesIsserlisMomentsOnABilinearModel)
{
    auto const [declared, undeclared] = productModels();
    auto const prior = bilinearPrior();
    auto const check = [&](auto const & method, bool fullMisses) {
        expectIsserlis(Emrys::predict(method, prior, declared, 1.0), 1e-12);
        auto const full = Emrys::predict(method, prior, undeclared, 1.0);
        if (fullMisses)
        {
            double const variance = full.covarianceLower()(2, 2);
            EXPECT_GT(std::abs(variance - (4.0 * 0.3 + 1.0 * 0.5 + 0.5 * 0.3)), 1e-6);
        }
    };
    check(methods::unscented(), true);
    check(methods::fourthMomentUnscented(), true);
    check(methods::cubature(), true);
    check(methods::sphericalSimplex(), true);
    check(methods::gaussHermite<3>(), false);
}

namespace
{

struct Position : Linear<1>
{};

using Scaled = Composite<Position, Scale>;

// x' = x + s w, noise scaled by the state, affine in x and w given s.
[[nodiscard]] auto
scaledNoiseModel()
{
    return compose(
        processModel(
            []<typename T>(auto const & in, auto const & out, auto const & x, double) {
                auto y = zeros<T>(out);
                block<Position>(out, y) = block<Position>(in, x) + block<Scale>(in, x)(0) * block<Noise>(in, x);
                return y;
            },
            reads(path<Position>, path<Scale>, path<Noise>),
            writes(path<Position>),
            affineIn(path<Position>, path<Noise>)
        ),
        Constant<Scale>{}
    );
}

}  // namespace

// Noise scaled by the state: Var(x') = s_x + (mu_s^2 + s_s) s_w for every sigma-point method.
TEST(AffineCoordinates, NoiseScaledByTheStateIsExactForEverySigmaPointMethod)
{
    auto const model = scaledNoiseModel();
    Eigen::Vector2d const mean{1.0, 2.0};
    Eigen::Matrix2d cov;
    cov << 0.4, 0.1, 0.1, 0.5;
    auto const prior = GaussianParams<Scaled>{Scaled{}, mean, cov};
    auto const noise = GaussianParams<Noise>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.2}};
    double const expected = 0.4 + (4.0 + 0.5) * 0.2;
    auto const check = [&](auto const & method) {
        auto const predicted = Emrys::predict(method, prior, model, 1.0, noise);
        EXPECT_NEAR(predicted.mean()(0), 1.0, 1e-12);
        EXPECT_NEAR(predicted.covarianceLower()(0, 0), expected, 1e-12);
        EXPECT_NEAR(predicted.covarianceLower()(1, 0), 0.1, 1e-12);
    };
    check(methods::unscented());
    check(methods::fourthMomentUnscented());
    check(methods::cubature());
    check(methods::sphericalSimplex());
    check(methods::gaussHermite<3>());
}

// A name the call gives as sure has no spread to factor: the prediction equals the model's without that name in its
// affineIn(), bitwise.
TEST(AffineCoordinates, ASureInputIsNotFactored)
{
    auto const f = []<typename T>(auto const & in, auto const & out, auto const & x, double) {
        auto y = zeros<T>(out);
        block<Position>(out, y) = block<Position>(in, x) + block<Scale>(in, x)(0) * block<Noise>(in, x);
        return y;
    };
    auto const withName = compose(
        processModel(
            f,
            reads(path<Position>, path<Scale>, path<Noise>),
            writes(path<Position>),
            affineIn(path<Position>, path<Noise>)
        ),
        Constant<Scale>{}
    );
    auto const withoutName = compose(
        processModel(
            f,
            reads(path<Position>, path<Scale>, path<Noise>),
            writes(path<Position>),
            affineIn(path<Position>)
        ),
        Constant<Scale>{}
    );
    Eigen::Matrix2d cov;
    cov << 0.4, 0.1, 0.1, 0.5;
    auto const prior = GaussianParams<Scaled>{Scaled{}, Eigen::Vector2d{1.0, 2.0}, cov};
    auto const w = sure<Noise>(Eigen::Matrix<double, 1, 1>{0.3});
    auto const a = Emrys::predict(methods::unscented(), prior, withName, 1.0, w);
    auto const b = Emrys::predict(methods::unscented(), prior, withoutName, 1.0, w);
    EXPECT_EQ(a.mean(), b.mean());
    EXPECT_EQ(a.covarianceLower(), b.covarianceLower());
}

// Linearized ignores the declaration: the model with and without it predicts the same, bitwise.
TEST(AffineCoordinates, LinearizedIgnoresTheDeclaration)
{
    auto const [declared, undeclared] = productModels();
    auto const prior = bilinearPrior();
    auto const a = Emrys::predict(methods::linearized(), prior, declared, 1.0);
    auto const b = Emrys::predict(methods::linearized(), prior, undeclared, 1.0);
    EXPECT_EQ(a.mean(), b.mean());
    EXPECT_EQ(a.covarianceLower(), b.covarianceLower());
}

namespace
{

struct Planar : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Heading : Linear<1>
{};

using Turning = Composite<Planar, Velocity, Heading>;

constexpr double timeStep = 0.5;

// R(theta) v, the velocity turned by the heading.
template <typename V, typename T>
[[nodiscard]] auto
turned(V const & v, T const & theta)
{
    using std::cos;
    using std::sin;
    using S = std::remove_cvref_t<T>;
    Eigen::Matrix<S, 2, 1> r;
    r << cos(theta) * v(0) - sin(theta) * v(1), sin(theta) * v(0) + cos(theta) * v(1);
    return r;
}

// The position's step and its Jacobian over (p, v, theta): [I, dt R, dt R' v].
[[nodiscard]] Eigen::Matrix<double, 2, 5>
positionJacobian(Eigen::Vector2d const & v, double theta, double dt)
{
    Eigen::Matrix<double, 2, 5> j = Eigen::Matrix<double, 2, 5>::Zero();
    j.block<2, 2>(0, 0).setIdentity();
    j.block<2, 2>(0, 2) << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    j.block<2, 2>(0, 2) *= dt;
    j.block<2, 1>(0, 4) << dt * (-std::sin(theta) * v(0) - std::cos(theta) * v(1)),
        dt * (std::cos(theta) * v(0) - std::sin(theta) * v(1));
    return j;
}

}  // namespace

// The function's Jacobian in the affine coordinates comes from what it returns, from a composition's parts' blocks,
// or by automatic differentiation, and the three give one prediction to 1e-12 with the function called once per
// point: three points over the heading, the one coordinate not declared affine.
TEST(AffineCoordinates, TheThreeRoutesToTheJacobianAgreeAtTheSameCalls)
{
    int calls = 0;
    auto const byAutoDiff = processModel(
        [&calls]<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            ++calls;
            auto y = zeros<T>(out);
            auto const theta = block<Heading>(in, x)(0);
            block<Planar>(out, y) = block<Planar>(in, x) + dt * turned(block<Velocity>(in, x), theta);
            block<Velocity>(out, y) = block<Velocity>(in, x);
            block<Heading>(out, y) = block<Heading>(in, x);
            return y;
        },
        reads(path<Planar>, path<Velocity>, path<Heading>),
        writes(path<Planar>, path<Velocity>, path<Heading>),
        affineIn(path<Planar>, path<Velocity>)
    );
    auto const byReturned = processModel(
        [&calls](auto const & in, auto const & out, auto const & x, double dt) {
            ++calls;
            Eigen::Matrix<double, 5, 1> y;
            double const theta = block<Heading>(in, x)(0);
            Eigen::Vector2d const v = block<Velocity>(in, x);
            y.head<2>() = block<Planar>(in, x) + dt * turned(v, theta);
            y.segment<2>(2) = v;
            y(4) = theta;
            Eigen::Matrix<double, 5, 5> j = Eigen::Matrix<double, 5, 5>::Identity();
            j.topRows<2>() = positionJacobian(v, theta, dt);
            static_cast<void>(out);
            return withJacobian(y, j);
        },
        reads(path<Planar>, path<Velocity>, path<Heading>),
        writes(path<Planar>, path<Velocity>, path<Heading>),
        affineIn(path<Planar>, path<Velocity>)
    );
    auto const byParts = compose(
        processModel(
            [&calls](auto const & in, auto const &, auto const & x, double dt) {
                ++calls;
                double const theta = block<Heading>(in, x)(0);
                Eigen::Vector2d const v = block<Velocity>(in, x);
                Eigen::Vector2d const y = block<Planar>(in, x) + dt * turned(v, theta);
                return withJacobian(y, positionJacobian(v, theta, dt));
            },
            reads(path<Planar>, path<Velocity>, path<Heading>),
            writes(path<Planar>),
            affineIn(path<Planar>, path<Velocity>)
        ),
        Constant<Velocity>{},
        Constant<Heading>{}
    );
    Eigen::Matrix<double, 5, 1> mean;
    mean << 1.0, -2.0, 3.0, 0.5, 0.7;
    Eigen::Matrix<double, 5, 5> root = Eigen::Matrix<double, 5, 5>::Identity();
    root(4, 2) = 0.3;
    root(3, 1) = 0.2;
    auto const prior = GaussianParams<Turning>{Turning{}, mean, root * root.transpose()};

    calls = 0;
    auto const a = Emrys::predict(methods::unscented(), prior, byAutoDiff, timeStep);
    EXPECT_EQ(calls, 3);
    calls = 0;
    auto const b = Emrys::predict(methods::unscented(), prior, byReturned, timeStep);
    EXPECT_EQ(calls, 3);
    calls = 0;
    auto const c = Emrys::predict(methods::unscented(), prior, byParts, timeStep);
    EXPECT_EQ(calls, 3);
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    EXPECT_LT((a.covarianceLower() - b.covarianceLower()).norm(), 1e-12);
    EXPECT_LT((a.mean() - c.mean()).norm(), 1e-12);
    EXPECT_LT((a.covarianceLower() - c.covarianceLower()).norm(), 1e-12);
}

namespace
{

struct Range : Linear<1>
{};

struct RangeNoise : Linear<1>
{};

using Plane = Composite<Planar, Velocity>;

}  // namespace

// A measurement model whose noise is an input it declares affine updates by the unscented transform as the model
// with that noise additive does: the points spread over the state alone.
TEST(AffineCoordinates, AnAffineNoiseInputMatchesAdditiveNoise)
{
    double const r = 0.04;
    auto const throughFunction = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Planar>(in, x).norm() + block<RangeNoise>(in, x)(0)};
        },
        reads(path<Planar>, path<RangeNoise>),
        writes(path<Range>),
        affineIn(path<RangeNoise>)
    );
    auto const additive = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Planar>(in, x).norm()};
        },
        reads(path<Planar>),
        writes(path<Range>),
        additiveNoise(GaussianParams<Range>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{r}})
    );
    Eigen::Vector4d const mean{3.0, 4.0, 0.5, -0.2};
    Eigen::Matrix4d root = Eigen::Matrix4d::Identity();
    root(1, 0) = 0.3;
    auto const prior = GaussianParams<Plane>{Plane{}, mean, root * root.transpose()};
    Eigen::Matrix<double, 1, 1> const z{5.2};
    auto const noise = GaussianParams<RangeNoise>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{r}};
    auto const a = update(prior, observe(methods::unscented(), throughFunction, z, noise));
    auto const b = update(prior, observe(methods::unscented(), additive, z));
    ASSERT_TRUE(a && b);
    EXPECT_LT((a->branches.front().posterior.mean() - b->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (a->branches.front().posterior.covarianceLower() - b->branches.front().posterior.covarianceLower()).norm(),
        1e-12
    );
}

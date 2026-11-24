// Affine coordinates declared with a model ([affine-coordinates]): a sigma-point method spreads its
// points over the coordinates the model's affineIn() does not name and carries the named ones exactly by their
// Gaussian given the rest, with the function's Jacobian in them from what it returns, from a composition's parts, or
// by automatic differentiation, which agree at the same number of calls. A bilinear model and noise scaled by the
// state get Isserlis' moments from every sigma-point method; a sure input is not factored; Linearized ignores the
// declaration. Constant items of affineIn(), on<Name>(H) and offset(), give the same result under every method as the
// model written as one function, in a measurement model, a process model, and transform(), with the function called as
// often and never given the constant input. A model without a function, affineIn() alone, is its exact affine map under
// every method, and a process model of it caches its transition.

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
using Emrys::offset;
using Emrys::on;
using Emrys::outputSpace;
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

namespace
{

// A place, read by a nonlinear function, a lean, which enters only by a constant matrix, and a drift,
// which moves the place.
struct Place : Linear<2>
{};

struct Lean : Linear<2>
{};

struct Drift : Linear<2>
{};

// A reading of the place: a sighting and a level, each with a constant offset.
struct Seen : Linear<2>
{};

struct Level : Linear<1>
{};

using Reading = Composite<Seen, Level>;

using Surveyed = Composite<Place, Lean, Drift>;

// The constant part: the lean's columns over the reading's three rows, and the offsets of its blocks.
Eigen::Matrix<double, 3, 2>
leanColumns()
{
    return Eigen::Matrix<double, 3, 2>{{0.7, -0.2}, {0.1, 0.5}, {-0.3, 0.4}};
}

Eigen::Vector2d const seenOffset{0.25, -0.5};
Eigen::Matrix<double, 1, 1> const levelOffset{0.75};

// g(place), the part that depends on the place alone, in T.
template <typename T, typename X>
Eigen::Vector<T, 3>
reading(X const & place)
{
    using std::cos;
    using std::sin;
    return Eigen::Vector<T, 3>{
        T(0.1) * place(0) * place(0) + sin(place(1)),
        T(0.2) * place(0) * place(1),
        cos(place(0))
    };
}

// Its Jacobian over the place.
template <typename T, typename X>
Eigen::Matrix<T, 3, 2>
readingJacobian(X const & place)
{
    using std::cos;
    using std::sin;
    return Eigen::Matrix<T, 3, 2>{
        {T(0.2) * place(0), cos(place(1))},
        {T(0.2) * place(1), T(0.2) * place(0)},
        {-sin(place(0)), T(0)}
    };
}

GaussianParams<Surveyed>
surveyed()
{
    Eigen::Matrix<double, 6, 6> a;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            a(i, j) = 0.15 * std::sin(1.0 + 2.0 * i + 3.0 * j);
        }
    }
    Eigen::Matrix<double, 6, 1> mean;
    mean << 0.8, -0.3, 0.4, 0.6, -0.2, 0.1;
    return {
        Surveyed{},
        mean,
        Eigen::Matrix<double, 6, 6>{a * a.transpose() + 0.1 * Eigen::Matrix<double, 6, 6>::Identity()}
    };
}

GaussianParams<Reading>
readingNoise()
{
    return {Reading{}, Eigen::Vector3d::Zero(), Eigen::Vector3d{0.04, 0.09, 0.01}.asDiagonal()};
}

template <typename Method>
class ConstantItemsTest : public ::testing::Test
{};

using ConstantMethods =
    ::testing::Types<methods::Linearized, methods::Unscented<>, methods::Cubature, methods::GaussHermite<3>>;

}  // namespace

TYPED_TEST_SUITE(ConstantItemsTest, ConstantMethods);

// A measurement model with on<Lean>(H) and offset() in affineIn() updates as the model whose function
// adds H lean + c itself, reading the lean with path<Lean> affine, under every method, with the
// function called as often and given the place alone; with a function that returns its Jacobian and
// with one that a method differentiates.
TYPED_TEST(ConstantItemsTest, AMeasurementModelUpdatesAsTheWholeFunction)
{
    auto const prior = surveyed();
    Eigen::Vector3d const z{0.9, -0.4, 1.3};
    int constantCalls = 0;
    int wholeCalls = 0;
    auto const constant = measurementModel(
        [&]<typename T>(auto const & in, auto const &, auto const & x) {
            static_assert(std::is_same_v<std::remove_cvref_t<decltype(in)>, Composite<Place>>);
            ++constantCalls;
            return reading<T>(block<Place>(in, x));
        },
        reads(path<Place>),
        affineIn(on<Lean>(leanColumns()), offset(on<Seen>(seenOffset), on<Level>(levelOffset))),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const whole = measurementModel(
        [&]<typename T>(auto const & in, auto const &, auto const & x) {
            ++wholeCalls;
            Eigen::Vector<T, 3> y = reading<T>(block<Place>(in, x)) + leanColumns().cast<T>() * block<Lean>(in, x);
            y.template head<2>() += seenOffset.cast<T>();
            y.template tail<1>() += levelOffset.cast<T>();
            return y;
        },
        reads(path<Place>, path<Lean>),
        affineIn(path<Lean>),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const expectSame = [&](auto const & a, auto const & b) {
        auto const x = update(prior, observe(TypeParam{}, a, z));
        auto const y = update(prior, observe(TypeParam{}, b, z));
        ASSERT_TRUE(x && y);
        auto const & p = x->branches.front().posterior;
        auto const & q = y->branches.front().posterior;
        EXPECT_LT((p.mean() - q.mean()).norm(), 1e-10);
        EXPECT_LT((p.covarianceLower() - q.covarianceLower()).norm(), 1e-10);
        EXPECT_GT((p.mean() - prior.mean()).norm(), 1e-3);
    };
    expectSame(constant, whole);
    EXPECT_EQ(constantCalls, wholeCalls);
    EXPECT_GT(constantCalls, 0);

    // The function returns its Jacobian over the place; the model adds H in the lean's columns.
    auto const returning = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return withJacobian(reading<T>(block<Place>(in, x)), readingJacobian<double>(block<Place>(in, x)));
        },
        reads(path<Place>),
        affineIn(on<Lean>(leanColumns()), offset(on<Seen>(seenOffset), on<Level>(levelOffset))),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    expectSame(returning, whole);
}

// offset(c) of the whole output gives the same model as the offset by block.
TEST(ConstantItems, AWholeOffsetIsTheOffsetByBlock)
{
    auto const f = []<typename T>(auto const & in, auto const &, auto const & x) {
        return reading<T>(block<Place>(in, x));
    };
    Eigen::Vector3d const c{seenOffset(0), seenOffset(1), levelOffset(0)};
    auto const byBlock = measurementModel(
        f,
        reads(path<Place>),
        affineIn(on<Lean>(leanColumns()), offset(on<Seen>(seenOffset), on<Level>(levelOffset))),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const whole = measurementModel(
        f,
        reads(path<Place>),
        affineIn(offset(c), on<Lean>(leanColumns())),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const z = Eigen::Vector3d{0.9, -0.4, 1.3};
    auto const x = update(surveyed(), observe(methods::unscented(), byBlock, z));
    auto const y = update(surveyed(), observe(methods::unscented(), whole, z));
    ASSERT_TRUE(x && y);
    EXPECT_LT((x->branches.front().posterior.mean() - y->branches.front().posterior.mean()).norm(), 1e-14);
}

// A process model with on<Drift>(H) in affineIn() holds one step, so it predicts over fixedStep(), and
// does as the model whose function adds H drift itself, under every method, with the function called
// as often.
TYPED_TEST(ConstantItemsTest, AProcessModelPredictsAsTheWholeFunctionOverItsStep)
{
    constexpr double dt = 0.5;
    Eigen::Matrix2d const driftColumns{{dt, 0.1}, {0.0, dt}};
    int constantCalls = 0;
    int wholeCalls = 0;
    auto const constant = Emrys::compose(
        processModel(
            [&]<typename T>(auto const & in, auto const & out, auto const & x, double) {
                ++constantCalls;
                auto y = zeros<T>(out);
                auto const place = block<Place>(in, x);
                block<Place>(out, y) = place + T(0.05) * place.cwiseProduct(place);
                return y;
            },
            reads(path<Place>),
            writes(path<Place>),
            affineIn(on<Drift>(driftColumns)),
            Emrys::continuousNoise(on<Place>(0.2))
        ),
        Constant<Lean>{},
        Constant<Drift>{}
    );
    auto const whole = Emrys::compose(
        processModel(
            [&]<typename T>(auto const & in, auto const & out, auto const & x, double) {
                ++wholeCalls;
                auto y = zeros<T>(out);
                auto const place = block<Place>(in, x);
                block<Place>(out, y) =
                    place + T(0.05) * place.cwiseProduct(place) + driftColumns.cast<T>() * block<Drift>(in, x);
                return y;
            },
            reads(path<Place>, path<Drift>),
            writes(path<Place>),
            affineIn(path<Drift>),
            Emrys::continuousNoise(on<Place>(0.2))
        ),
        Constant<Lean>{},
        Constant<Drift>{}
    );
    static_assert(Emrys::internal::holdsOneStep<std::remove_cvref_t<decltype(constant)>>);
    static_assert(!Emrys::internal::holdsOneStep<std::remove_cvref_t<decltype(whole)>>);
    auto const prior = surveyed();
    auto const p = Emrys::predict(prior, Emrys::fixedStep(constant, dt, TypeParam{}));
    auto const q = Emrys::predict(prior, Emrys::fixedStep(whole, dt, TypeParam{}));
    EXPECT_LT((p.mean() - q.mean()).norm(), 1e-10);
    EXPECT_LT((p.covarianceLower() - q.covarianceLower()).norm(), 1e-10);
    EXPECT_EQ(constantCalls, wholeCalls);
    EXPECT_GT((p.mean() - prior.mean()).norm(), 1e-3);
}

// transform() with on<Lean>(H) and offset(c) in affineIn() gives the output's moments as the function
// written whole, under every method, with the lean an input of the call.
TYPED_TEST(ConstantItemsTest, TransformGivesTheMomentsOfTheWholeFunction)
{
    auto const place = GaussianParams<Place>{Eigen::Vector2d{0.8, -0.3}, Eigen::Matrix2d{{0.2, 0.03}, {0.03, 0.1}}};
    auto const lean = GaussianParams<Lean>{Eigen::Vector2d{0.4, 0.6}, Eigen::Matrix2d{{0.3, -0.05}, {-0.05, 0.2}}};
    Eigen::Vector3d const c{seenOffset(0), seenOffset(1), levelOffset(0)};
    auto const constant = Emrys::transform(
        TypeParam{},
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return reading<T>(block<Place>(in, x));
        },
        reads(path<Place>),
        affineIn(on<Lean>(leanColumns()), offset(c)),
        outputSpace(Reading{}),
        place,
        lean
    );
    auto const whole = Emrys::transform(
        TypeParam{},
        [c]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Vector<T, 3>{
                reading<T>(block<Place>(in, x)) + leanColumns().cast<T>() * block<Lean>(in, x) + c.cast<T>()
            };
        },
        reads(path<Place>, path<Lean>),
        affineIn(path<Lean>),
        outputSpace(Reading{}),
        place,
        lean
    );
    EXPECT_LT((constant.mean() - whole.mean()).norm(), 1e-10);
    EXPECT_LT((constant.covarianceLower() - whole.covarianceLower()).norm(), 1e-10);
}

namespace
{

// The place's columns of a reading without a function.
Eigen::Matrix<double, 3, 2>
placeColumns()
{
    return Eigen::Matrix<double, 3, 2>{{1.0, 0.2}, {-0.4, 0.8}, {0.3, 0.3}};
}

}  // namespace

// A measurement model without a function, affineIn() of on<Place>(Hp), on<Lean>(Hl), and offset(c),
// updates as the function that returns Hp place + Hl lean + c with its Jacobian, under every method,
// and observe() without a method takes its exact update.
TYPED_TEST(ConstantItemsTest, AModelWithoutAFunctionIsItsAffineMap)
{
    auto const prior = surveyed();
    Eigen::Vector3d const z{0.9, -0.4, 1.3};
    auto const map = measurementModel(
        affineIn(
            on<Place>(placeColumns()),
            on<Lean>(leanColumns()),
            offset(on<Seen>(seenOffset), on<Level>(levelOffset))
        ),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const function = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Vector<T, 3> y =
                placeColumns().cast<T>() * block<Place>(in, x) + leanColumns().cast<T>() * block<Lean>(in, x);
            y.template head<2>() += seenOffset.cast<T>();
            y.template tail<1>() += levelOffset.cast<T>();
            Eigen::Matrix<double, 3, 4> jacobian;
            jacobian << placeColumns(), leanColumns();
            return withJacobian(y, jacobian);
        },
        reads(path<Place>, path<Lean>),
        outputSpace(Reading{}),
        additiveNoise(readingNoise())
    );
    auto const x = update(prior, observe(TypeParam{}, map, z));
    auto const y = update(prior, observe(methods::linearized(), function, z));
    auto const w = update(prior, observe(map, z));
    ASSERT_TRUE(x && y && w);
    EXPECT_LT((x->branches.front().posterior.mean() - y->branches.front().posterior.mean()).norm(), 1e-10);
    EXPECT_LT(
        (x->branches.front().posterior.covarianceLower() - y->branches.front().posterior.covarianceLower()).norm(),
        1e-10
    );
    EXPECT_LT((w->branches.front().posterior.mean() - y->branches.front().posterior.mean()).norm(), 1e-12);
}

// A process model without a function, affineIn() of on<Place>(I) and on<Drift>(dt I) writing the
// place, holds one step; over fixedStep() it caches its transition, and predicts F P F' + Q.
TEST(ConstantItems, AProcessModelWithoutAFunctionCachesItsTransition)
{
    constexpr double dt = 0.5;
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto const model = Emrys::compose(
        processModel(
            affineIn(on<Place>(i2), on<Drift>(Eigen::Matrix2d{dt * i2})),
            writes(path<Place>),
            Emrys::continuousNoise(on<Place>(0.2))
        ),
        Constant<Lean>{},
        Constant<Drift>{}
    );
    static_assert(Emrys::internal::holdsOneStep<std::remove_cvref_t<decltype(model)>>);
    auto const fixed = Emrys::FixedStepModel{model, Surveyed{}, dt};
    static_assert(decltype(fixed)::CachesTransition);
    Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Identity();
    f.block<2, 2>(0, 4) = dt * i2;
    EXPECT_LT((fixed.transition() - f).norm(), 1e-15);
    auto const prior = surveyed();
    auto const predicted = Emrys::predict(prior, fixed);
    Eigen::Matrix<double, 6, 6> q = Eigen::Matrix<double, 6, 6>::Zero();
    q.block<2, 2>(0, 0) = 0.2 * dt * i2;
    Eigen::Matrix<double, 6, 6> const p = prior.covarianceLower().selfadjointView<Eigen::Lower>();
    Eigen::Matrix<double, 6, 6> const cov = predicted.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((predicted.mean() - f * prior.mean()).norm(), 1e-12);
    EXPECT_LT((cov - (f * p * f.transpose() + q)).norm(), 1e-12);
}

// affineIn() of one matrix is the whole state's map without a function, as the matrix forms are:
// processModel(affineIn(F), additiveNoise(Q)) over its step, and measurementModel(affineIn(H), noise).
TEST(ConstantItems, AMatrixIsTheWholeStatesMap)
{
    Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Identity();
    f.block<2, 2>(0, 4) = 0.5 * Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 6, 6> const q = 0.01 * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 3, 6> h = Eigen::Matrix<double, 3, 6>::Zero();
    h.leftCols<2>() = placeColumns();
    h.middleCols<2>(2) = leanColumns();
    auto const prior = surveyed();
    auto const noise = GaussianParams<Linear<6>>{Eigen::Matrix<double, 6, 1>::Zero(), q};
    auto const p = Emrys::predict(prior, Emrys::fixedStep(processModel(affineIn(f), additiveNoise(noise)), 0.5));
    auto const r = Emrys::predict(prior, Emrys::fixedStep(processModel(affineIn(f), additiveNoise(noise)), 0.5));
    EXPECT_LT((p.mean() - r.mean()).norm(), 1e-14);
    EXPECT_LT((p.covarianceLower() - r.covarianceLower()).norm(), 1e-14);
    auto const fixNoise =
        GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), Eigen::Matrix3d{0.05 * Eigen::Matrix3d::Identity()}};
    auto const z = Eigen::Vector3d{0.9, -0.4, 1.3};
    auto const x = update(prior, observe(measurementModel(affineIn(h), additiveNoise(fixNoise)), z));
    auto const y = update(prior, observe(measurementModel(affineIn(h), additiveNoise(fixNoise)), z));
    ASSERT_TRUE(x && y);
    EXPECT_LT((x->branches.front().posterior.mean() - y->branches.front().posterior.mean()).norm(), 1e-14);
}

// transform() of affineIn() alone gives the affine map's moments, Hp mu + c and Hp P Hp', under every method.
TYPED_TEST(ConstantItemsTest, TransformOfAMapWithoutAFunction)
{
    auto const place = GaussianParams<Place>{Eigen::Vector2d{0.8, -0.3}, Eigen::Matrix2d{{0.2, 0.03}, {0.03, 0.1}}};
    Eigen::Vector3d const c{seenOffset(0), seenOffset(1), levelOffset(0)};
    auto const y =
        Emrys::transform(TypeParam{}, affineIn(on<Place>(placeColumns()), offset(c)), outputSpace(Reading{}), place);
    Eigen::Matrix2d const p = place.covarianceLower().selfadjointView<Eigen::Lower>();
    Eigen::Matrix3d const cov = y.covarianceLower().template selfadjointView<Eigen::Lower>();
    EXPECT_LT((y.mean() - (placeColumns() * place.mean() + c)).norm(), 1e-12);
    EXPECT_LT((cov - placeColumns() * p * placeColumns().transpose()).norm(), 1e-12);
}

// Linear measurement models, measurementModel() of a matrix or of Columns, of Models/MeasurementModel.h (roadmap
// step 16d, 17z; [linear-sensors]): a model made from its matrix H gives the hand Kalman update on blocks read out
// of the state's order, the same posterior as the function H x under every Gaussian form, covariance update, and
// mixture noise, wraps a periodic residual, weights particles and simulates as the function does, and takes its
// Jacobian from the matrix rather than by differentiating. Its compile errors are in
// tests/Models/compile_errors.cpp, and its updates' allocation in tests/allocation.cpp.

#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Columns;
using Emrys::Composite;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::onEach;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Bias, Velocity>;

struct Heading : RadiansPi<1>
{};

using Pose = Composite<Position, Heading>;

using Vector1 = Eigen::Matrix<double, 1, 1>;
using Vector5 = Eigen::Matrix<double, 5, 1>;
using Matrix5 = Eigen::Matrix<double, 5, 5>;

GaussianParams<State>
makePrior()
{
    Matrix5 cov = Matrix5::Identity();
    cov.diagonal() << 4.0, 3.0, 0.5, 1.0, 2.0;
    cov(0, 3) = cov(3, 0) = 0.8;
    cov(1, 4) = cov(4, 1) = -0.6;
    cov(2, 0) = cov(0, 2) = 0.3;
    return {State{}, Vector5{1.0, -2.0, 0.4, 0.5, -0.25}, cov};
}

// H over (velocity, position), the order the models read them in, which is not the state's.
Eigen::Matrix<double, 3, 4> const h{{1.0, 0.0, 2.0, 0.0}, {0.0, -1.0, 0.0, 1.0}, {0.5, 0.5, 0.0, 3.0}};

auto
makeNoise()
{
    return GaussianParams<Linear<3>>{
        Eigen::Vector3d::Zero(),
        Eigen::Matrix3d{{0.4, 0.1, 0.0}, {0.1, 0.3, 0.05}, {0.0, 0.05, 0.6}}
    };
}

// The two spellings of z = H (v, p) + w: the matrix, and the function.
template <typename Noise>
auto
makeLinear(Noise noise)
{
    return measurementModel(h, reads(path<Velocity>, path<Position>), additiveNoise(std::move(noise)));
}

template <typename Noise>
auto
makeFunction(Noise noise)
{
    return measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 3, 1>{
                h.leftCols<2>().cast<T>() * block<Velocity>(in, x) + h.rightCols<2>().cast<T>() * block<Position>(in, x)
            };
        },
        reads(path<Velocity>, path<Position>),
        additiveNoise(std::move(noise))
    );
}

Eigen::Vector3d const reading{1.5, -0.5, 2.0};

// Posteriors equal to rounding: the same mean and covariance.
template <typename A, typename B>
void
expectSamePosterior(A const & a, B const & b)
{
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    EXPECT_LT((a.covarianceLower() - b.covarianceLower()).norm(), 1e-12);
}

}  // namespace

// The model's function is H x with its Jacobian H, the model's matrix itself, so no differentiation
// happens whatever method the call names.
TEST(LinearMeasurement, ReturnsItsMatrixAsTheJacobian)
{
    auto const linear = makeLinear(makeNoise());
    auto const in = Composite<Velocity, Position>{};
    Eigen::Vector4d x;
    x << 0.5, -0.25, 1.0, -2.0;
    auto const returned = linear.function()(in, Linear<3>{}, x);
    EXPECT_TRUE(returned.value.isApprox(h * x, 1e-15));
    EXPECT_EQ(returned.jacobian, h);
    EXPECT_EQ(returned.jacobian.data(), linear.function().matrix().data());
}

// Reading the velocity and then the position, the update is the hand Kalman update with H placed
// in the state's columns of those blocks: K = P H^T S^{-1}, S = H P H^T + R.
TEST(LinearMeasurement, MatchesTheHandKalmanUpdateOnBlocksOutOfOrder)
{
    auto const prior = makePrior();
    Eigen::Matrix<double, 3, 5> full = Eigen::Matrix<double, 3, 5>::Zero();
    full.rightCols<2>() = h.leftCols<2>();
    full.leftCols<2>() = h.rightCols<2>();
    Matrix5 const p = prior.covarianceLower();
    Eigen::Matrix3d const s = full * p * full.transpose() + makeNoise().covarianceLower();
    Eigen::Matrix<double, 5, 3> const k = p * full.transpose() * s.inverse();

    auto const updated = Emrys::update(prior, observe(methods::linearized(), makeLinear(makeNoise()), reading));
    ASSERT_TRUE(updated);
    EXPECT_LT(
        (updated->branches[0].posterior.mean() - (prior.mean() + k * (reading - full * prior.mean()))).norm(),
        1e-12
    );
    EXPECT_LT((updated->branches[0].posterior.covarianceLower() - (p - k * full * p)).norm(), 1e-12);
}

// The matrix and the function under Linearized give the same posterior and likelihood, under each
// covariance update and under noise of two components.
TEST(LinearMeasurement, GivesTheFunctionsPosterior)
{
    auto const prior = makePrior();
    auto const check = [&](auto const & linear, auto const & function, auto const &... form) {
        auto const a = Emrys::update(prior, observe(methods::linearized(), linear, reading, form...));
        auto const b = Emrys::update(prior, observe(methods::linearized(), function, reading, form...));
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        ASSERT_EQ(a->branches.size(), b->branches.size());
        for (std::size_t i = 0; i < a->branches.size(); ++i)
        {
            expectSamePosterior(a->branches[i].posterior, b->branches[i].posterior);
            EXPECT_NEAR(a->branches[i].logWeight, b->branches[i].logWeight, 1e-12);
        }
        EXPECT_NEAR(a->logLikelihood(), b->logLikelihood(), 1e-12);
    };
    check(makeLinear(makeNoise()), makeFunction(makeNoise()));
    check(makeLinear(makeNoise()), makeFunction(makeNoise()), Emrys::updates::joseph());
    check(makeLinear(makeNoise()), makeFunction(makeNoise()), Emrys::updates::symmetric());

    auto mixture = GaussianMixtureParams<Linear<3>>{Linear<3>{}};
    mixture.add(std::log(0.9), makeNoise());
    mixture.add(std::log(0.1), Eigen::Vector3d{1.0, 0.0, -1.0}, 25.0 * Eigen::Matrix3d::Identity());
    check(makeLinear(mixture), makeFunction(mixture));
}

// The square-root and information forms take the linear model as they take the function, and all
// four filters end where the Kalman filter with the function does.
TEST(LinearMeasurement, EveryFormGivesTheSamePosterior)
{
    auto const model = Emrys::compose(Emrys::ConstantVelocity<Position, Velocity>{0.1}, Emrys::RandomWalk<Bias>{0.01});
    auto reference = Emrys::KalmanFilter{model, makePrior()};
    auto kalman = Emrys::KalmanFilter{model, makePrior()};
    auto root = Emrys::SqrtKalmanFilter{model, Emrys::toSqrt(makePrior()).value()};
    auto info = Emrys::InfoFilter{model, Emrys::toInfo(makePrior()).value()};
    auto rootInfo = Emrys::SqrtInfoFilter{model, Emrys::toSqrtInfo(makePrior()).value()};
    auto const linear = makeLinear(makeNoise());
    for (int k = 0; k < 5; ++k)
    {
        Eigen::Vector3d const observed = reading + 0.3 * k * Eigen::Vector3d::Ones();
        reference.predict(1.0);
        kalman.predict(1.0);
        ASSERT_TRUE(root.predict(1.0));
        ASSERT_TRUE(info.predict(1.0));
        ASSERT_TRUE(rootInfo.predict(1.0));
        ASSERT_TRUE(reference.update(observe(methods::linearized(), makeFunction(makeNoise()), observed)));
        ASSERT_TRUE(kalman.update(observe(methods::linearized(), linear, observed)));
        ASSERT_TRUE(root.update(observe(methods::linearized(), linear, observed)));
        ASSERT_TRUE(info.update(observe(methods::linearized(), linear, observed)));
        ASSERT_TRUE(rootInfo.update(observe(methods::linearized(), linear, observed)));
    }
    auto const expectNear = [&](auto const & form) {
        auto const gaussian = Emrys::toGaussian(form);
        ASSERT_TRUE(gaussian);
        EXPECT_LT((gaussian->mean() - reference.state().mean()).norm(), 1e-9);
        EXPECT_LT((gaussian->covarianceLower() - reference.state().covarianceLower()).norm(), 1e-9);
    };
    expectSamePosterior(kalman.state(), reference.state());
    expectNear(root.state());
    expectNear(info.state());
    expectNear(rootInfo.state());
}

// A heading read through a periodic measurement space: with the prior at 3.1 rad and the
// observation at -3.1 rad, the residual wraps to 2 pi - 6.2, as the function's and Columns' given
// the space do, and the posterior heading moves past pi rather than back across the circle.
TEST(LinearMeasurement, WrapsAPeriodicResidual)
{
    auto const prior = GaussianParams<Pose>{
        Pose{},
        Eigen::Vector3d{1.0, 2.0, 3.1},
        Eigen::Vector3d{1.0, 1.0, 0.04}.asDiagonal().toDenseMatrix()
    };
    auto const noise = GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{0.01}};
    auto const compass =
        measurementModel(Vector1{1.0}, reads(path<Heading>), outputSpace(RadiansPi<1>{}), additiveNoise(noise));
    auto const function = measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return block<Heading>(in, x);
        },
        reads(path<Heading>),
        outputSpace(RadiansPi<1>{}),
        additiveNoise(noise)
    );
    auto const columns =
        measurementModel(Columns{on<Heading>(Vector1{1.0})}, outputSpace(RadiansPi<1>{}), additiveNoise(noise));
    using Resolved =
        decltype(Emrys::internal::resolveMeasurement(observe(methods::linearized(), columns, Vector1{}), Pose{}));
    static_assert(std::is_same_v<typename Resolved::Out, RadiansPi<1>>);
    auto const a = Emrys::update(prior, observe(methods::linearized(), compass, Vector1{-3.1}));
    auto const b = Emrys::update(prior, observe(methods::linearized(), function, Vector1{-3.1}));
    auto const c = Emrys::update(prior, observe(methods::linearized(), columns, Vector1{-3.1}));
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    ASSERT_TRUE(c);
    expectSamePosterior(a->branches[0].posterior, b->branches[0].posterior);
    expectSamePosterior(a->branches[0].posterior, c->branches[0].posterior);
    double const residual = 2 * std::numbers::pi - 6.2;
    EXPECT_NEAR(a->branches[0].innovation.residual()(0), residual, 1e-12);
    double const moved = 3.1 + 0.04 / 0.05 * residual - 2 * std::numbers::pi;
    EXPECT_NEAR(a->branches[0].posterior.mean()(2), moved, 1e-12);
}

// The filters that weight points weigh by the matrix's prediction as by the function's, and a
// simulation from one seed draws the same measurements from either.
TEST(LinearMeasurement, WeighsAndSimulatesAsTheFunctionDoes)
{
    auto const linear = makeLinear(makeNoise());
    auto const function = makeFunction(makeNoise());
    auto rng = std::mt19937_64{3};
    auto const prior = makePrior();
    // The prior's covariance is positive definite by construction, so toDensity() can't fail here.
    Eigen::Matrix<double, 5, 20> const states = Emrys::drawSamples(prior.toDensity().value(), 20, rng);
    auto const a = Emrys::logLikelihoods(observe(linear, reading), State{}, states);
    auto const b = Emrys::logLikelihoods(observe(function, reading), State{}, states);
    EXPECT_LT((a - b).matrix().norm(), 1e-12);

    auto linearDraws = std::mt19937_64{4};
    auto functionDraws = std::mt19937_64{4};
    for (Eigen::Index i = 0; i < states.cols(); ++i)
    {
        Eigen::Vector3d const fromLinear = Emrys::simulate(State{}, states.col(i), linear, linearDraws);
        Eigen::Vector3d const fromFunction = Emrys::simulate(State{}, states.col(i), function, functionDraws);
        EXPECT_LT((fromLinear - fromFunction).norm(), 1e-12);
    }
}

namespace
{

struct Landmark : Linear<2>
{};

struct Landmarks : Emrys::Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Landmarks>;

struct Offset : Linear<2>
{};

struct Offsets : Emrys::Repeated<Offset, 8>
{
    using Repeated::Repeated;
};

// A vehicle at a position and count landmarks, all correlated.
GaussianParams<Map>
makeMap(Eigen::Index count)
{
    Eigen::Index const n = 2 + 2 * count;
    Eigen::MatrixXd const a = Eigen::MatrixXd::NullaryExpr(n, n, [](Eigen::Index i, Eigen::Index j) {
        return std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
    });
    Eigen::MatrixXd const cov = a * a.transpose() + Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd const mean = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);
    return {Map{Position{}, Landmarks{Landmark{}, count}}, mean, cov};
}

Eigen::Matrix2d const offsetNoise{{0.3, 0.05}, {0.05, 0.2}};

// Each landmark's position relative to the vehicle's, z_i = l_i - p, by its columns: the identity
// on each landmark and minus the identity on the position, with the noise on each offset.
auto
makeRelative()
{
    return measurementModel(
        Columns{onEach<Landmarks>(Eigen::Matrix2d::Identity()), on<Position>(-Eigen::Matrix2d::Identity())},
        additiveNoise(onEach<>(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), offsetNoise}))
    );
}

// The same measurement as a function, in a space of offsets counted from the landmarks.
auto
makeRelativeFunction()
{
    return measurementModel(
        []<typename T>(auto const & in, auto const & out, auto const & x) {
            auto const landmarks = block<Landmarks>(in, x);
            auto const position = block<Position>(in, x);
            auto result = Emrys::zeros<T>(out);
            for (Eigen::Index i = 0; i < out.count(); ++i)
            {
                result.template segment<2>(2 * i) = landmarks.template segment<2>(2 * i) - position;
            }
            return result;
        },
        reads(path<Landmarks>, path<Position>),
        outputSpace(Emrys::CountFrom<Landmarks, Offsets>{}),
        additiveNoise(onEach<Offsets>(GaussianParams<Offset>{Eigen::Vector2d::Zero(), offsetNoise}))
    );
}

}  // namespace

// Columns that don't repeat are the matrix split by block: the same Jacobian over the blocks read,
// and the same posterior.
TEST(LinearMeasurement, ColumnsWithoutRepetitionAreTheMatrix)
{
    auto const columns = measurementModel(
        Columns{on<Velocity>(h.leftCols<2>()), on<Position>(h.rightCols<2>())},
        additiveNoise(makeNoise())
    );
    auto const in = Composite<Velocity, Position>{};
    Eigen::Vector4d x;
    x << 0.5, -0.25, 1.0, -2.0;
    auto const returned = columns.function()(in, Linear<3>{}, x);
    EXPECT_TRUE(returned.value.isApprox(h * x, 1e-15));
    EXPECT_EQ(returned.jacobian, h);

    auto const prior = makePrior();
    auto const a = Emrys::update(prior, observe(methods::linearized(), columns, reading));
    auto const b = Emrys::update(prior, observe(methods::linearized(), makeLinear(makeNoise()), reading));
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    expectSamePosterior(a->branches[0].posterior, b->branches[0].posterior);
}

// A model given its matrix, or its matrix by block, needs no method: observe(model, z) pushes the
// state through by Linearized, which is exact for it, and gives the posterior that
// observe(methods::linearized(), model, z) does. A function's measurement given no method has none.
TEST(LinearMeasurement, NeedsNoMethod)
{
    auto const linear = makeLinear(makeNoise());
    auto const columns = measurementModel(
        Columns{on<Velocity>(h.leftCols<2>()), on<Position>(h.rightCols<2>())},
        additiveNoise(makeNoise())
    );
    static_assert(std::is_same_v<decltype(observe(linear, reading))::MethodType, methods::Linearized>);
    static_assert(std::is_same_v<decltype(observe(columns, reading))::MethodType, methods::Linearized>);
    static_assert(
        std::is_same_v<decltype(observe(makeFunction(makeNoise()), reading))::MethodType, Emrys::internal::NoMethod>
    );

    auto const prior = makePrior();
    auto const check = [&](auto const & model) {
        auto const a = Emrys::update(prior, observe(model, reading));
        auto const b = Emrys::update(prior, observe(methods::linearized(), model, reading));
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        expectSamePosterior(a->branches[0].posterior, b->branches[0].posterior);
    };
    check(linear);
    check(columns);
}

// Columns that repeat give the hand Kalman update with H = [-1 (x) I, I (x) I] in the state's
// order and R = I (x) R_1, and one model follows the count of landmarks without being rebuilt.
TEST(LinearMeasurement, RepeatsWithTheBlockItReadsEachOf)
{
    auto const relative = makeRelative();
    for (Eigen::Index const count : {Eigen::Index{2}, Eigen::Index{4}})
    {
        auto const prior = makeMap(count);
        Eigen::Index const n = 2 + 2 * count;
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(2 * count, n);
        Eigen::MatrixXd r = Eigen::MatrixXd::Zero(2 * count, 2 * count);
        for (Eigen::Index i = 0; i < count; ++i)
        {
            full.block<2, 2>(2 * i, 0) = -Eigen::Matrix2d::Identity();
            full.block<2, 2>(2 * i, 2 + 2 * i) = Eigen::Matrix2d::Identity();
            r.block<2, 2>(2 * i, 2 * i) = offsetNoise;
        }
        Eigen::VectorXd const observed = Eigen::VectorXd::LinSpaced(2 * count, 0.5, -1.5);
        Eigen::MatrixXd const p = prior.covarianceLower().selfadjointView<Eigen::Lower>();
        Eigen::MatrixXd const s = full * p * full.transpose() + r;
        Eigen::MatrixXd const k = p * full.transpose() * s.inverse();

        auto const updated = Emrys::update(prior, observe(methods::linearized(), relative, observed));
        ASSERT_TRUE(updated);
        auto const & posterior = updated->branches[0].posterior;
        EXPECT_LT((posterior.mean() - (prior.mean() + k * (observed - full * prior.mean()))).norm(), 1e-12);
        Eigen::MatrixXd const expected = p - k * full * p;
        Eigen::MatrixXd const lower = posterior.covarianceLower().template triangularView<Eigen::Lower>();
        EXPECT_LT((lower - Eigen::MatrixXd{expected.triangularView<Eigen::Lower>()}).norm(), 1e-12);
    }
}

// Columns that repeat update, weigh points, and simulate as the function does in the space that
// CountFrom counts, with the noise placed on each offset in both.
TEST(LinearMeasurement, RepeatsAsTheFunctionDoes)
{
    auto const relative = makeRelative();
    auto const function = makeRelativeFunction();
    auto const prior = makeMap(3);
    Eigen::VectorXd const observed = Eigen::VectorXd::LinSpaced(6, 1.0, -1.0);
    auto const a = Emrys::update(prior, observe(methods::linearized(), relative, observed));
    auto const b = Emrys::update(prior, observe(methods::linearized(), function, observed));
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    expectSamePosterior(a->branches[0].posterior, b->branches[0].posterior);
    EXPECT_NEAR(a->logLikelihood(), b->logLikelihood(), 1e-12);

    auto rng = std::mt19937_64{3};
    // The prior's covariance is positive definite by construction, so toDensity() can't fail here.
    Eigen::MatrixXd const states = Emrys::drawSamples(prior.toDensity().value(), 10, rng);
    auto const & space = prior.space();
    auto const weights = Emrys::logLikelihoods(observe(relative, observed), space, states);
    auto const expected = Emrys::logLikelihoods(observe(function, observed), space, states);
    EXPECT_LT((weights - expected).matrix().norm(), 1e-12);

    auto linearDraws = std::mt19937_64{4};
    auto functionDraws = std::mt19937_64{4};
    for (Eigen::Index i = 0; i < states.cols(); ++i)
    {
        Eigen::VectorXd const fromLinear = Emrys::simulate(space, states.col(i), relative, linearDraws);
        Eigen::VectorXd const fromFunction = Emrys::simulate(space, states.col(i), function, functionDraws);
        EXPECT_LT((fromLinear - fromFunction).norm(), 1e-12);
    }
}

namespace
{

struct Course : RadiansPi<1>
{};

struct Courses : Emrys::Repeated<Course, 4>
{
    using Repeated::Repeated;
};

using Fleet = Composite<Position, Courses>;

struct Bearing : RadiansPi<1>
{};

struct Bearings : Emrys::Repeated<Bearing, 4>
{
    using Repeated::Repeated;
};

}  // namespace

// Columns that repeat into a periodic space that CountFrom counts: each course's residual wraps on
// its own, so the course at 3.1 rad observed at -3.1 rad moves past pi, and the one at 0.5 rad
// observed at 0.6 rad moves toward it.
TEST(LinearMeasurement, RepeatsIntoAPeriodicSpace)
{
    auto const compasses = measurementModel(
        Columns{onEach<Courses>(Vector1{1.0})},
        outputSpace(Emrys::CountFrom<Courses, Bearings>{}),
        additiveNoise(onEach<Bearings>(GaussianParams<Bearing>{Vector1{0.0}, Vector1{0.01}}))
    );
    auto const prior = GaussianParams<Fleet>{
        Fleet{Position{}, Courses{Course{}, 2}},
        Eigen::Vector4d{1.0, 2.0, 3.1, 0.5},
        Eigen::Vector4d{1.0, 1.0, 0.04, 0.04}.asDiagonal().toDenseMatrix()
    };
    auto const updated = Emrys::update(prior, observe(methods::linearized(), compasses, Eigen::Vector2d{-3.1, 0.6}));
    ASSERT_TRUE(updated);
    double const residual = 2 * std::numbers::pi - 6.2;
    EXPECT_NEAR(updated->branches[0].innovation.residual()(0), residual, 1e-12);
    EXPECT_NEAR(updated->branches[0].posterior.mean()(2), 3.1 + 0.8 * residual - 2 * std::numbers::pi, 1e-12);
    EXPECT_NEAR(updated->branches[0].posterior.mean()(3), 0.5 + 0.8 * 0.1, 1e-12);
}

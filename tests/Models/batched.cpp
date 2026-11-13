// Batched functions (step 18c; [batched-functions]), in Models/Batched.h, Models/Likelihood.h, and the
// particle and point-mass filters that weight by a likelihood: a batched logLikelihood() is called once
// per particle set under Sequential and once per chunk under another policy, and gives the per-point
// function's weights; a point-mass filter calls it once over its cells; a single point is a batch of one;
// BatchedOutput lays out the Jacobians it is given storage for by block, only the blocks wanted; and the
// sigma-point transforms in Models/Transform.h, the factored one and the iterated posterior linearization
// filter's included, call a batched model once over their points.

#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Likelihood.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/SwitchingModel.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

#include "../user_policies.h"

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ParticleSet;

namespace jacobians = Emrys::jacobians;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Heading : Linear<1>
{};

using State = Composite<Position, Velocity>;

constexpr double rangeDeviation = 0.5;

// The log-density of a range residual r under Gaussian noise of rangeDeviation.
double
rangeLogDensity(double r)
{
    double const s = r / rangeDeviation;
    return -0.5 * s * s - std::log(rangeDeviation * std::sqrt(2 * std::numbers::pi));
}

// A range from the origin, known by its likelihood at one point, which takes the scalar of x as its
// template parameter (decisions.md [function-scalars]).
auto
pointRange(double z)
{
    return observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Position>),
            Emrys::logLikelihood([]<typename T>(auto const & in, auto const & x, auto const & observed) {
                return T(rangeLogDensity(observed(0) - Emrys::block<Position>(in, x).norm()));
            })
        ),
        Eigen::Matrix<double, 1, 1>{z}
    );
}

// The same range as a batched likelihood, every point a column of X, counting its calls and the
// columns it was given.
auto
batchedRange(double z, std::atomic<int> * calls, std::atomic<int> * columns)
{
    return observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Position>),
            Emrys::logLikelihood(
                Emrys::batched([calls, columns](auto const & in, auto const & x, auto const & observed, auto & y) {
                    ++*calls;
                    *columns += int(x.cols());
                    auto const r =
                        (observed(0) - Emrys::block<Position>(in, x).colwise().norm().array()) / rangeDeviation;
                    y.values() =
                        (-0.5 * r.square() - std::log(rangeDeviation * std::sqrt(2 * std::numbers::pi))).matrix();
                })
            )
        ),
        Eigen::Matrix<double, 1, 1>{z}
    );
}

ParticleSet<State>
makeParticles(Eigen::Index count, unsigned seed)
{
    auto rng = std::mt19937_64{seed};
    Eigen::Vector4d const variances{9.0, 9.0, 1.0, 1.0};
    auto const prior =
        GaussianParams<State>{State{}, Eigen::Vector4d{2.0, 1.0, 0.0, 0.0}, Eigen::Matrix4d{variances.asDiagonal()}};
    return ParticleSet<State>{State{}, Emrys::drawSamples(prior.toDensity().value(), count, rng)};
}

struct Range : Linear<1>
{};

// A specific force measured by an accelerometer, an uncertain input of a prediction.
struct SpecificForce : Linear<2>
{};

// The calls of the batched range and step functions, and the columns they were given.
std::atomic<int> rangeCalls{0};
std::atomic<int> rangeColumns{0};
std::atomic<int> stepCalls{0};
std::atomic<int> stepColumns{0};

// The additive noise of a range.
GaussianParams<Linear<1>>
rangeNoise()
{
    return {Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.25}};
}

// A range from the origin to the position, per point.
auto
rangeModel()
{
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const & out, auto const & x) {
            auto y = Emrys::zeros<T>(out);
            y(0) = Eigen::numext::sqrt(T(Emrys::block<Position>(in, x).squaredNorm()));
            return y;
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Range>),
        Emrys::additiveNoise(rangeNoise())
    );
}

// The same range batched, without Jacobians or, given jacobians::perColumn(), with each column's.
template <typename... Mark>
auto
batchedRangeModel(Mark... mark)
{
    return Emrys::measurementModel(
        Emrys::batched(
            [](auto const & in, auto const &, auto const & x, auto & y) {
                using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
                ++rangeCalls;
                rangeColumns += int(x.cols());
                auto const p = Emrys::block<Position>(in, x);
                for (Eigen::Index j = 0; j < x.cols(); ++j)
                {
                    T const r = Eigen::numext::sqrt(T(p.col(j).squaredNorm()));
                    y.values()(0, j) = r;
                    if constexpr (sizeof...(Mark) > 0)
                    {
                        if (y.template wantsJacobian<Position>())
                        {
                            y.template jacobian<Position>(j) = p.col(j).transpose() / r;
                        }
                    }
                }
            },
            mark...
        ),
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Range>),
        Emrys::additiveNoise(rangeNoise())
    );
}

// Constant velocity written by hand, per point, with continuous noise on the velocity.
auto
velocityModel()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(out, y) = Emrys::block<Velocity>(in, x);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.05))
    );
}

// The same model batched, without Jacobians or, given jacobians::shared(), with the one Jacobian
// of the linear step.
template <typename... Mark>
auto
batchedVelocityModel(Mark... mark)
{
    return Emrys::processModel(
        Emrys::batched(
            [](auto const & in, auto const & out, auto const & x, double dt, auto & y) {
                using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
                ++stepCalls;
                stepColumns += int(x.cols());
                Emrys::block<Position>(out, y.values()) =
                    Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
                Emrys::block<Velocity>(out, y.values()) = Emrys::block<Velocity>(in, x);
                if constexpr (sizeof...(Mark) > 0)
                {
                    Eigen::Matrix<double, 4, 2> position = Eigen::Matrix<double, 4, 2>::Zero();
                    position.topRows<2>().setIdentity();
                    Eigen::Matrix<double, 4, 2> velocity = Eigen::Matrix<double, 4, 2>::Zero();
                    velocity.topRows<2>() = dt * Eigen::Matrix2d::Identity();
                    velocity.bottomRows<2>().setIdentity();
                    if (y.template wantsJacobian<Position>())
                    {
                        y.template jacobian<Position>() = position.cast<T>();
                    }
                    if (y.template wantsJacobian<Velocity>())
                    {
                        y.template jacobian<Velocity>() = velocity.cast<T>();
                    }
                }
            },
            mark...
        ),
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.05))
    );
}

// Velocity driven by a measured specific force, an uncertain input, per point and batched, with any
// further policies, such as affineIn().
template <typename... Policies>
auto
forcedModel(Policies... policies)
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Eigen::Matrix<T, 2, 1> const a = Emrys::block<SpecificForce>(in, x);
            Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(out, y) = Emrys::block<Velocity>(in, x) + T(dt) * a;
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>, Emrys::path<SpecificForce>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.05)),
        policies...
    );
}

template <typename... Policies>
auto
batchedForcedModel(Policies... policies)
{
    return Emrys::processModel(
        Emrys::batched([](auto const & in, auto const & out, auto const & x, double dt, auto & y) {
            using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
            ++stepCalls;
            stepColumns += int(x.cols());
            Emrys::block<Position>(out, y.values()) =
                Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(out, y.values()) =
                Emrys::block<Velocity>(in, x) + T(dt) * Emrys::block<SpecificForce>(in, x);
        }),
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>, Emrys::path<SpecificForce>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.05)),
        policies...
    );
}

// The forced model batched with the one Jacobian of its linear step, given jacobians::shared(), each
// block's written where it is wanted.
template <typename... Policies>
auto
batchedForcedModelWithJacobians(Policies... policies)
{
    return Emrys::processModel(
        Emrys::batched(
            [](auto const & in, auto const & out, auto const & x, double dt, auto & y) {
                using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
                ++stepCalls;
                stepColumns += int(x.cols());
                Emrys::block<Position>(out, y.values()) =
                    Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
                Emrys::block<Velocity>(out, y.values()) =
                    Emrys::block<Velocity>(in, x) + T(dt) * Emrys::block<SpecificForce>(in, x);
                Eigen::Matrix<double, 4, 2> top = Eigen::Matrix<double, 4, 2>::Zero();
                top.topRows<2>().setIdentity();
                Eigen::Matrix<double, 4, 2> bottom = Eigen::Matrix<double, 4, 2>::Zero();
                bottom.bottomRows<2>().setIdentity();
                if (y.template wantsJacobian<Position>())
                {
                    y.template jacobian<Position>() = top.cast<T>();
                }
                if (y.template wantsJacobian<Velocity>())
                {
                    y.template jacobian<Velocity>() = (dt * top + bottom).cast<T>();
                }
                if (y.template wantsJacobian<SpecificForce>())
                {
                    y.template jacobian<SpecificForce>() = (dt * bottom).cast<T>();
                }
            },
            jacobians::shared()
        ),
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>, Emrys::path<SpecificForce>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.05)),
        policies...
    );
}

// A bias added to a range, an uncertain input the range is affine in.
struct RangeBias : Linear<1>
{};

// The range plus its bias, per point.
auto
biasedRangeModel()
{
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const & out, auto const & x) {
            auto y = Emrys::zeros<T>(out);
            y(0) =
                Eigen::numext::sqrt(T(Emrys::block<Position>(in, x).squaredNorm())) + Emrys::block<RangeBias>(in, x)(0);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<RangeBias>),
        Emrys::writes(Emrys::path<Range>),
        Emrys::additiveNoise(rangeNoise()),
        Emrys::affineIn(Emrys::path<RangeBias>)
    );
}

// The same batched, without Jacobians or, given jacobians::perColumn(), with each column's.
template <typename... Mark>
auto
batchedBiasedRangeModel(Mark... mark)
{
    return Emrys::measurementModel(
        Emrys::batched(
            [](auto const & in, auto const &, auto const & x, auto & y) {
                using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
                ++rangeCalls;
                rangeColumns += int(x.cols());
                auto const p = Emrys::block<Position>(in, x);
                for (Eigen::Index j = 0; j < x.cols(); ++j)
                {
                    T const r = Eigen::numext::sqrt(T(p.col(j).squaredNorm()));
                    y.values()(0, j) = r + Emrys::block<RangeBias>(in, x)(0, j);
                    if constexpr (sizeof...(Mark) > 0)
                    {
                        if (y.template wantsJacobian<Position>())
                        {
                            y.template jacobian<Position>(j) = p.col(j).transpose() / r;
                        }
                        if (y.template wantsJacobian<RangeBias>())
                        {
                            y.template jacobian<RangeBias>(j).setOnes();
                        }
                    }
                }
            },
            mark...
        ),
        Emrys::reads(Emrys::path<Position>, Emrys::path<RangeBias>),
        Emrys::writes(Emrys::path<Range>),
        Emrys::additiveNoise(rangeNoise()),
        Emrys::affineIn(Emrys::path<RangeBias>)
    );
}

// The accelerometer's reading of the specific force, an uncertain input.
GaussianParams<SpecificForce>
accelerometerReading()
{
    return {Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
}

GaussianParams<State>
makePrior()
{
    Eigen::Matrix4d cov = Eigen::Vector4d{4.0, 3.0, 1.0, 0.5}.asDiagonal();
    cov(0, 1) = 0.8;
    cov(1, 0) = 0.8;
    return GaussianParams<State>{State{}, Eigen::Vector4d{3.0, 2.0, 0.5, -0.2}, cov};
}

// Expects two Gaussians to agree to 1e-12 in mean and covariance.
template <typename A, typename B>
void
expectSameGaussian(A const & a, B const & b)
{
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    EXPECT_LT((a.covarianceLower() - b.covarianceLower()).norm(), 1e-12);
}

}  // namespace

static_assert(Emrys::JacobianMark<jacobians::PerColumn>);
static_assert(Emrys::JacobianMark<jacobians::Shared>);
static_assert(!Emrys::JacobianMark<Emrys::internal::NoJacobians>);
static_assert(!Emrys::Batched<int>::GivesJacobians);
static_assert(decltype(Emrys::batched([] {}, jacobians::perColumn()))::PerColumn);

// The particle filter's update by a batched likelihood gives the per-point likelihood's log-likelihood and
// weights, to rounding (the batch computes the norms column by column as the point does, in a different
// expression), calling the function once over every particle under Sequential.
TEST(Batched, ParticleFilterCallsALikelihoodOncePerSet)
{
    auto calls = std::atomic<int>{0};
    auto columns = std::atomic<int>{0};
    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto byPoint = Emrys::ParticleFilter{model, makeParticles(3000, 7)};
    auto byBatch = Emrys::ParticleFilter{model, makeParticles(3000, 7)};
    auto const expected = byPoint.update(pointRange(3.0));
    auto const update = byBatch.update(batchedRange(3.0, &calls, &columns));
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(columns, 3000);
    EXPECT_TRUE(update->accepted);
    EXPECT_NEAR(update->logLikelihood(), expected->logLikelihood(), 1e-10);
    EXPECT_LT((byBatch.state().logWeights() - byPoint.state().logWeights()).abs().maxCoeff(), 1e-10);
}

// Under a policy other than Sequential the function is called once per chunk of 1024 particles,
// three for 3000, each over its own columns, and the weights are Sequential's: each entry is
// written by one call over the same column, so the order the chunks run in changes nothing.
TEST(Batched, ParticleFilterCallsALikelihoodOncePerChunkUnderAPolicy)
{
    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto sequentialCalls = std::atomic<int>{0};
    auto sequentialColumns = std::atomic<int>{0};
    auto sequential = Emrys::ParticleFilter{model, makeParticles(3000, 11)};
    auto const expected = sequential.update(batchedRange(2.5, &sequentialCalls, &sequentialColumns));
    auto const check = [&](auto const & policy) {
        auto calls = std::atomic<int>{0};
        auto columns = std::atomic<int>{0};
        auto filter = Emrys::ParticleFilter{model, makeParticles(3000, 11)};
        auto const update = filter.update(policy, batchedRange(2.5, &calls, &columns));
        EXPECT_EQ(calls, Emrys::internal::chunkCount(3000));
        EXPECT_EQ(columns, 3000);
        EXPECT_NEAR(update->logLikelihood(), expected->logLikelihood(), 1e-12);
        EXPECT_LT((filter.state().logWeights() - sequential.state().logWeights()).abs().maxCoeff(), 1e-12);
    };
    check(EmrysTest::ReversedPolicy{});
    check(EmrysTest::ThreadsPolicy{});
}

// logLikelihoods() over a matrix of states calls the batched function once and gives the per-point
// values; logLikelihood() at one point is a batch of one column.
TEST(Batched, LikelihoodsAtStatesAndAtAPoint)
{
    auto calls = std::atomic<int>{0};
    auto columns = std::atomic<int>{0};
    auto const particles = makeParticles(40, 3);
    auto const batched = Emrys::logLikelihoods<State>(batchedRange(1.5, &calls, &columns), particles.states());
    auto const pointwise = Emrys::logLikelihoods<State>(pointRange(1.5), particles.states());
    EXPECT_EQ(calls, 1);
    EXPECT_LT((batched - pointwise).abs().maxCoeff(), 1e-12);
    Eigen::Vector4d const x = particles.states().col(5);
    EXPECT_NEAR(Emrys::logLikelihood<State>(batchedRange(1.5, &calls, &columns), x), pointwise(5), 1e-12);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(columns, 41);
}

// A point-mass filter weights its cells by a batched likelihood in one call over every cell, with the
// per-point likelihood's evidence and weights.
TEST(Batched, PointMassFilterCallsALikelihoodOnceOverItsCells)
{
    auto const grid =
        Emrys::regularGrid<Composite<Position>, Position>(Eigen::Vector2d{-10, -10}, Eigen::Vector2d{1, 1}, {21, 21});
    auto calls = std::atomic<int>{0};
    auto columns = std::atomic<int>{0};
    auto byPoint = Emrys::PointMassFilter{grid, Emrys::SharedCovariance{}};
    auto byBatch = Emrys::PointMassFilter{grid, Emrys::SharedCovariance{}};
    auto const expected = byPoint.update(pointRange(4.0));
    auto const update = byBatch.update(batchedRange(4.0, &calls, &columns));
    ASSERT_TRUE(expected);
    ASSERT_TRUE(update);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(columns, 21 * 21);
    EXPECT_NEAR(update->logLikelihood(), expected->logLikelihood(), 1e-10);
    ASSERT_EQ(byBatch.state().size(), byPoint.state().size());
    EXPECT_LT((byBatch.state().logWeights() - byPoint.state().logWeights()).abs().maxCoeff(), 1e-10);
}

// BatchedOutput gives each wanted block of the input its columns of the storage, side by side, once per
// column under perColumn() and once under shared(), and a block not wanted none: the input here is
// Position (2), Heading (1), Velocity (2), with Position and Velocity wanted.
TEST(Batched, OutputLaysOutTheWantedJacobiansByBlock)
{
    using In = Composite<Position, Heading, Velocity>;
    auto const in = In{};
    auto values = Eigen::Matrix<double, 3, 2>{};
    auto storage = Eigen::Matrix<double, 3, 8>{Eigen::Matrix<double, 3, 8>::Zero()};
    using PerColumn = Emrys::BatchedOutput<In, Linear<3>, double, jacobians::PerColumn>;
    auto y = PerColumn{
        PerColumn::ValuesType{values.data(), 3, 2},
        PerColumn::JacobiansType{storage.data(), 3, 8},
        in,
        {true, false, true}
    };
    EXPECT_TRUE(y.wantsJacobian<Position>());
    EXPECT_FALSE(y.wantsJacobian<Heading>());
    EXPECT_TRUE(y.wantsJacobian<Velocity>());
    static_assert(decltype(y.jacobian<Position>(0))::ColsAtCompileTime == 2);
    y.jacobian<Position>(1).setConstant(1.0);
    y.jacobian<Velocity>(1).setConstant(2.0);
    y.jacobian<Velocity>(0).setConstant(3.0);
    y.values().setConstant(4.0);
    auto expected = Eigen::Matrix<double, 3, 8>{};
    expected << Eigen::Matrix<double, 3, 2>::Zero(), Eigen::Matrix<double, 3, 2>::Constant(3.0),
        Eigen::Matrix<double, 3, 2>::Constant(1.0), Eigen::Matrix<double, 3, 2>::Constant(2.0);
    EXPECT_EQ(storage, expected);
    EXPECT_EQ(values, (Eigen::Matrix<double, 3, 2>::Constant(4.0)));

    auto shared = Eigen::Matrix<double, 3, 1>{Eigen::Matrix<double, 3, 1>::Zero()};
    using Shared = Emrys::BatchedOutput<In, Linear<3>, double, jacobians::Shared>;
    auto z = Shared{
        Shared::ValuesType{values.data(), 3, 2},
        Shared::JacobiansType{shared.data(), 3, 1},
        in,
        {false, true, false}
    };
    EXPECT_FALSE(z.wantsJacobian<Position>());
    z.jacobian<Heading>().setConstant(5.0);
    EXPECT_EQ(shared, (Eigen::Matrix<double, 3, 1>::Constant(5.0)));
}

// A batched measurement model updates as its per-point form does under every method: Linearized and
// the iterated update by autodiff over a batch of one column, or with the Jacobian it writes per
// column; the sigma-point methods and the iterated posterior linearization filter at their points.
TEST(Batched, MeasurementModelMatchesItsPerPointFormUnderEveryMethod)
{
    auto const prior = makePrior();
    auto const z = Eigen::Matrix<double, 1, 1>{3.9};
    auto const check = [&](auto const & method) {
        auto const expected = Emrys::update(prior, observe(method, rangeModel(), z));
        ASSERT_TRUE(expected);
        auto const compare = [&](auto const & model) {
            auto const updated = Emrys::update(prior, observe(method, model, z));
            ASSERT_TRUE(updated);
            expectSameGaussian(updated->branches.front().posterior, expected->branches.front().posterior);
            EXPECT_NEAR(updated->logLikelihood(), expected->logLikelihood(), 1e-12);
        };
        compare(batchedRangeModel());
        compare(batchedRangeModel(jacobians::perColumn()));
    };
    check(Emrys::methods::linearized());
    check(Emrys::methods::iterated());
    check(Emrys::methods::unscented());
    check(Emrys::methods::cubature());
    check(Emrys::methods::iterated(Emrys::methods::cubature(), {.tolerance = 1e-9, .maxIterations = 4}));
}

// A sigma-point method calls a batched measurement model once over its points: the range reads the
// two coordinates of Position, so the unscented transform has 2 * 2 + 1 points, cubature 2 * 2, and a
// third-order Gauss-Hermite rule 3^2; transform() of the model calls it once too.
TEST(Batched, SigmaPointMethodsCallAMeasurementModelOncePerTransform)
{
    auto const prior = makePrior();
    auto const z = Eigen::Matrix<double, 1, 1>{3.9};
    auto const check = [&](auto const & method, int points) {
        rangeCalls = 0;
        rangeColumns = 0;
        ASSERT_TRUE(Emrys::update(prior, observe(method, batchedRangeModel(), z)));
        EXPECT_EQ(rangeCalls, 1);
        EXPECT_EQ(rangeColumns, points);
    };
    check(Emrys::methods::unscented(), 5);
    check(Emrys::methods::cubature(), 4);
    check(Emrys::methods::gaussHermite<3>(), 9);
    rangeCalls = 0;
    rangeColumns = 0;
    auto const position = prior.marginal<Position>();
    auto const predicted = Emrys::transform(Emrys::methods::cubature(), batchedRangeModel(), position);
    EXPECT_EQ(rangeCalls, 1);
    EXPECT_EQ(rangeColumns, 4);
    expectSameGaussian(predicted, Emrys::transform(Emrys::methods::cubature(), rangeModel(), position));
}

// The iterated posterior linearization filter calls a batched measurement model once per iteration,
// over the four cubature points of each.
TEST(Batched, IteratedPosteriorLinearizationCallsAMeasurementModelOncePerIteration)
{
    rangeCalls = 0;
    rangeColumns = 0;
    auto const method = Emrys::methods::iterated(Emrys::methods::cubature(), {.tolerance = 1e-9, .maxIterations = 4});
    ASSERT_TRUE(Emrys::update(makePrior(), observe(method, batchedRangeModel(), Eigen::Matrix<double, 1, 1>{3.9})));
    EXPECT_GE(rangeCalls, 1);
    EXPECT_LE(rangeCalls, 4);
    EXPECT_EQ(rangeColumns, 4 * rangeCalls);
}

// A sigma-point method calls a batched process model once over its points: the state has four
// coordinates, so the unscented transform has 2 * 4 + 1 points and cubature 2 * 4.
TEST(Batched, SigmaPointMethodsCallAProcessModelOncePerPrediction)
{
    auto const prior = makePrior();
    auto const check = [&](auto const & method, int points) {
        stepCalls = 0;
        stepColumns = 0;
        auto const predicted = Emrys::predict(method, prior, batchedVelocityModel(), 0.5);
        EXPECT_EQ(stepCalls, 1);
        EXPECT_EQ(stepColumns, points);
        expectSameGaussian(predicted, Emrys::predict(method, prior, velocityModel(), 0.5));
    };
    check(Emrys::methods::unscented(), 9);
    check(Emrys::methods::cubature(), 8);
}

// The factored transform calls a batched function once over its points, differentiated in dual numbers
// seeded on the coordinates it factors out, and predicts as the per-point form does: with Velocity and
// the specific force factored, the points spread the two coordinates of Position, 2 * 2 + 1 for the
// unscented transform and 2 * 2 for cubature.
TEST(Batched, FactoredTransformCallsAProcessModelOncePerPrediction)
{
    auto const prior = makePrior();
    auto const affine = Emrys::affineIn(Emrys::path<Velocity>, Emrys::path<SpecificForce>);
    auto const check = [&](auto const & method, int points) {
        stepCalls = 0;
        stepColumns = 0;
        auto const predicted = Emrys::predict(method, prior, batchedForcedModel(affine), 0.5, accelerometerReading());
        EXPECT_EQ(stepCalls, 1);
        EXPECT_EQ(stepColumns, points);
        expectSameGaussian(predicted, Emrys::predict(method, prior, forcedModel(affine), 0.5, accelerometerReading()));
    };
    check(Emrys::methods::unscented(), 5);
    check(Emrys::methods::cubature(), 4);
}

// The factored transform takes the Jacobians a batched function marked jacobians::shared() writes, in
// one call with the factored blocks wanted, and predicts as the per-point form does: the identity on
// Velocity, which the step reads, factors out, and writes, comes from the function's rows.
TEST(Batched, FactoredTransformTakesTheJacobiansAFunctionWrites)
{
    auto const prior = makePrior();
    auto const affine = Emrys::affineIn(Emrys::path<Velocity>, Emrys::path<SpecificForce>);
    auto const check = [&](auto const & method) {
        stepCalls = 0;
        auto const predicted =
            Emrys::predict(method, prior, batchedForcedModelWithJacobians(affine), 0.5, accelerometerReading());
        EXPECT_EQ(stepCalls, 1);
        expectSameGaussian(predicted, Emrys::predict(method, prior, forcedModel(affine), 0.5, accelerometerReading()));
    };
    check(Emrys::methods::unscented());
    check(Emrys::methods::cubature());
}

// The likelihood at points of a batched function with an uncertain input it is affine in calls it once
// over the points, with its Jacobian in the input by autodiff or as it writes it, and widens the noise
// by the input's covariance as the per-point linearization does.
TEST(Batched, LikelihoodsWithAnUncertainInputCallTheFunctionOnce)
{
    using Biased = Composite<Position, Velocity>;
    auto const particles = makeParticles(200, 29);
    auto const z = Eigen::Matrix<double, 1, 1>{3.2};
    auto const bias = GaussianParams<RangeBias>{Eigen::Matrix<double, 1, 1>{0.1}, Eigen::Matrix<double, 1, 1>{0.04}};
    Eigen::ArrayXd const expected =
        Emrys::logLikelihoods(observe(biasedRangeModel(), z, bias), Biased{}, particles.states());
    auto const check = [&](auto const & model) {
        rangeCalls = 0;
        rangeColumns = 0;
        Eigen::ArrayXd const batched = Emrys::logLikelihoods(observe(model, z, bias), Biased{}, particles.states());
        EXPECT_EQ(rangeCalls, 1);
        EXPECT_EQ(rangeColumns, 200);
        EXPECT_LT((batched - expected).abs().maxCoeff(), 1e-12);
    };
    check(batchedBiasedRangeModel());
    check(batchedBiasedRangeModel(jacobians::perColumn()));
}

// A Kalman update by a sigma-point method factors the bias out of a batched function in one call,
// with the Jacobian in it by autodiff or as the function writes it, as the per-point form does.
TEST(Batched, FactoredMeasurementUpdateCallsTheFunctionOnce)
{
    auto const prior = makePrior();
    auto const z = Eigen::Matrix<double, 1, 1>{3.9};
    auto const bias = GaussianParams<RangeBias>{Eigen::Matrix<double, 1, 1>{0.1}, Eigen::Matrix<double, 1, 1>{0.04}};
    auto const expected = Emrys::update(prior, observe(Emrys::methods::unscented(), biasedRangeModel(), z, bias));
    ASSERT_TRUE(expected);
    auto const check = [&](auto const & model) {
        rangeCalls = 0;
        rangeColumns = 0;
        auto const updated = Emrys::update(prior, observe(Emrys::methods::unscented(), model, z, bias));
        ASSERT_TRUE(updated);
        EXPECT_EQ(rangeCalls, 1);
        EXPECT_EQ(rangeColumns, 5);
        expectSameGaussian(updated->branches.front().posterior, expected->branches.front().posterior);
    };
    check(batchedBiasedRangeModel());
    check(batchedBiasedRangeModel(jacobians::perColumn()));
}

// An update given a block, as a Rao-Blackwellized filter's conditional Gaussian takes it, calls a
// batched measurement model once over the transform's points: the rest is Velocity, two coordinates,
// so the unscented transform has 2 * 2 + 1 points.
TEST(Batched, AnUpdateGivenABlockCallsAMeasurementModelOnce)
{
    using Rest = Emrys::ParticleGaussianParams<State, Position>::RestSpace;
    auto const prior =
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, -0.2}, Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}};
    auto const z = Eigen::Matrix<double, 1, 1>{3.1};
    auto const known = Eigen::Vector2d{3.0, 1.0};
    auto const position = Emrys::given<Position>(State{}, known);
    auto const expected = Emrys::update(prior, observe(Emrys::methods::unscented(), rangeModel(), z), position);
    ASSERT_TRUE(expected);
    rangeCalls = 0;
    rangeColumns = 0;
    auto const updated = Emrys::update(prior, observe(Emrys::methods::unscented(), batchedRangeModel(), z), position);
    ASSERT_TRUE(updated);
    EXPECT_EQ(rangeCalls, 1);
    EXPECT_EQ(rangeColumns, 5);
    expectSameGaussian(updated->branches.front().posterior, expected->branches.front().posterior);
    EXPECT_NEAR(updated->logLikelihood(), expected->logLikelihood(), 1e-12);
}

namespace
{

struct Steady
{};

struct Drifting
{};

// Two modes of one dynamics, so that a batched switching model can be matched against a per-point one.
struct Mode : Emrys::Named<Steady, Drifting>
{};

}  // namespace

// Under a switching model of batched modes, a Rao-Blackwellized particle filter predicts each
// particle's hybrid by one call per mode and particle over the unscented transform's points, the rest
// being Velocity's two coordinates, and moves the particles and their hybrids as the per-point modes
// do from the same generator.
TEST(Batched, HybridParticlesCallABatchedModeOncePerTransform)
{
    using Rest = Emrys::ParticleGaussianParams<State, Position>::RestSpace;
    auto const hybrid = Emrys::HybridGaussianParams<Rest, Mode>{
        Emrys::CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.6, 0.4}),
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, -0.2}, Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}}
    };
    auto const positions = Eigen::Matrix<double, 2, 5>{{-3, 0, 2, 5, -1}, {1, -2, 0, 3, 4}};
    auto const state = Emrys::particleGaussian<Position>(State{}, ParticleSet<Position>{positions}, hybrid);
    auto const chain = Emrys::MarkovChain<Mode>::persistent(0.9);
    auto byPoint = Emrys::RbParticleFilter{
        Emrys::SwitchingModel{chain, velocityModel(), velocityModel()},
        state,
        Emrys::methods::unscented()
    };
    auto byBatch = Emrys::RbParticleFilter{
        Emrys::SwitchingModel{chain, batchedVelocityModel(), batchedVelocityModel()},
        state,
        Emrys::methods::unscented()
    };
    auto first = std::mt19937_64{31};
    auto second = std::mt19937_64{31};
    ASSERT_TRUE(byPoint.predict(first, 0.5));
    stepCalls = 0;
    stepColumns = 0;
    ASSERT_TRUE(byBatch.predict(second, 0.5));
    EXPECT_EQ(stepCalls, 2 * 5);
    EXPECT_EQ(stepColumns, 2 * 5 * 5);
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(byBatch.state().particle(i).isApprox(byPoint.state().particle(i), 1e-12)) << i;
        auto const batched = byBatch.state().conditional(i).toDensity()->collapsed();
        auto const pointwise = byPoint.state().conditional(i).toDensity()->collapsed();
        expectSameGaussian(batched, pointwise);
    }
}

namespace
{

// One craft's position and velocity, placed as a lead, a wingman, and a fleet of escorts.
struct Craft : Composite<Position, Velocity>
{};

struct Lead : Craft
{};

struct Wing : Craft
{};

struct Escorts : Emrys::Repeated<Craft, 4>
{
    using Repeated::Repeated;
};

using Formation = Composite<Lead, Wing>;
using Flight = Composite<Lead, Escorts>;

// A Gaussian over a state of n coordinates, its covariance with the correlations of makePrior()'s.
template <typename S>
GaussianParams<S>
spreadPrior(S const & space)
{
    Eigen::Index const n = Emrys::valueDimension(space);
    Eigen::VectorXd const mean = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);
    Eigen::MatrixXd cov = 0.5 * Eigen::MatrixXd::Identity(n, n);
    for (Eigen::Index i = 0; i + 1 < n; ++i)
    {
        cov(i, i + 1) = cov(i + 1, i) = 0.1;
    }
    return GaussianParams<S>{space, mean, cov};
}

}  // namespace

// A composition calls a batched part once over its rows of the unscented transform's points and its
// other part at each point: the formation has eight coordinates, so 2 * 8 + 1 points, and predicts as
// the composition of per-point parts does.
TEST(Batched, ACompositionCallsItsBatchedPartOncePerTransform)
{
    auto const prior = spreadPrior(Formation{});
    auto const byPoint = Emrys::compose(Emrys::on<Lead>(velocityModel()), Emrys::on<Wing>(velocityModel()));
    auto const byBatch = Emrys::compose(Emrys::on<Lead>(batchedVelocityModel()), Emrys::on<Wing>(velocityModel()));
    auto const expected = Emrys::predict(Emrys::methods::unscented(), prior, byPoint, 0.5);
    stepCalls = 0;
    stepColumns = 0;
    auto const predicted = Emrys::predict(Emrys::methods::unscented(), prior, byBatch, 0.5);
    EXPECT_EQ(stepCalls, 1);
    EXPECT_EQ(stepColumns, 17);
    expectSameGaussian(predicted, expected);
}

// onEach() calls a batched model once per repetition over the transform's points: three escorts and a
// lead, sixteen coordinates, so three calls of 2 * 16 + 1 columns each. The flight's size is a run-time
// count, so the transform first sizes its output by the function at the mean, a call of one column per
// repetition.
TEST(Batched, OnEachCallsABatchedModelOncePerRepetition)
{
    auto const flight = Flight{Lead{}, Escorts{Craft{}, 3}};
    auto const prior = spreadPrior(flight);
    auto const byPoint = Emrys::compose(Emrys::on<Lead>(velocityModel()), Emrys::onEach<Escorts>(velocityModel()));
    auto const byBatch =
        Emrys::compose(Emrys::on<Lead>(velocityModel()), Emrys::onEach<Escorts>(batchedVelocityModel()));
    auto const expected = Emrys::predict(Emrys::methods::unscented(), prior, byPoint, 0.5);
    stepCalls = 0;
    stepColumns = 0;
    auto const predicted = Emrys::predict(Emrys::methods::unscented(), prior, byBatch, 0.5);
    EXPECT_EQ(stepCalls, 3 + 3);
    EXPECT_EQ(stepColumns, 3 * 33 + 3);
    expectSameGaussian(predicted, expected);
}

// A measurement model placed on a block calls its batched function once over the transform's points,
// reading the lead's position, two coordinates, so 2 * 2 + 1 points.
TEST(Batched, APlacedMeasurementModelCallsItsFunctionOnce)
{
    auto const prior = spreadPrior(Formation{});
    auto const z = Eigen::Matrix<double, 1, 1>{2.4};
    auto const expected = Emrys::update(prior, observe(Emrys::methods::unscented(), Emrys::on<Lead>(rangeModel()), z));
    ASSERT_TRUE(expected);
    rangeCalls = 0;
    rangeColumns = 0;
    auto const updated =
        Emrys::update(prior, observe(Emrys::methods::unscented(), Emrys::on<Lead>(batchedRangeModel()), z));
    ASSERT_TRUE(updated);
    EXPECT_EQ(rangeCalls, 1);
    EXPECT_EQ(rangeColumns, 5);
    expectSameGaussian(updated->branches.front().posterior, expected->branches.front().posterior);
}

// A batched process model predicts as its per-point form does under every method, by autodiff and
// with the shared Jacobian of a linear function.
TEST(Batched, ProcessModelMatchesItsPerPointFormUnderEveryMethod)
{
    auto const prior = makePrior();
    auto const check = [&](auto const & method) {
        auto const expected = Emrys::predict(method, prior, velocityModel(), 0.5);
        expectSameGaussian(Emrys::predict(method, prior, batchedVelocityModel(), 0.5), expected);
        expectSameGaussian(Emrys::predict(method, prior, batchedVelocityModel(jacobians::shared()), 0.5), expected);
    };
    check(Emrys::methods::linearized());
    check(Emrys::methods::unscented());
    check(Emrys::methods::cubature());
}

// A particle filter moves its particles and weights them by batched models as by their per-point
// forms: the same draws give the same particles, and the additive noise's density the same weights.
TEST(Batched, ParticleFilterMatchesThePerPointModels)
{
    auto byPoint = Emrys::ParticleFilter{velocityModel(), makeParticles(500, 5)};
    auto byBatch = Emrys::ParticleFilter{batchedVelocityModel(), makeParticles(500, 5)};
    auto first = std::mt19937_64{9};
    auto second = std::mt19937_64{9};
    byPoint.predict(first, 0.5);
    byBatch.predict(second, 0.5);
    EXPECT_LT((byBatch.state().states() - byPoint.state().states()).cwiseAbs().maxCoeff(), 1e-12);
    auto const z = Eigen::Matrix<double, 1, 1>{3.0};
    auto const expected = byPoint.update(observe(rangeModel(), z));
    auto const update = byBatch.update(observe(batchedRangeModel(), z));
    EXPECT_NEAR(update->logLikelihood(), expected->logLikelihood(), 1e-10);
    EXPECT_LT((byBatch.state().logWeights() - byPoint.state().logWeights()).abs().maxCoeff(), 1e-10);
}

// The particle filter calls a batched process model once per prediction and a batched measurement
// model under additive noise once per update under Sequential, and once per chunk under another
// policy, with every particle a column of one call; the particles and weights are Sequential's.
TEST(Batched, ParticleFilterCallsBatchedModelsOncePerSetOrChunk)
{
    auto const z = Eigen::Matrix<double, 1, 1>{3.0};
    auto sequential = Emrys::ParticleFilter{batchedVelocityModel(), makeParticles(3000, 13)};
    auto rng = std::mt19937_64{17};
    stepCalls = 0;
    stepColumns = 0;
    rangeCalls = 0;
    rangeColumns = 0;
    sequential.predict(rng, 0.5);
    auto const expected = sequential.update(observe(batchedRangeModel(), z));
    EXPECT_EQ(stepCalls, 1);
    EXPECT_EQ(stepColumns, 3000);
    EXPECT_EQ(rangeCalls, 1);
    EXPECT_EQ(rangeColumns, 3000);
    auto const check = [&](auto const & policy) {
        auto filter = Emrys::ParticleFilter{batchedVelocityModel(), makeParticles(3000, 13)};
        auto same = std::mt19937_64{17};
        stepCalls = 0;
        stepColumns = 0;
        rangeCalls = 0;
        filter.predict(policy, same, 0.5);
        auto const update = filter.update(policy, observe(batchedRangeModel(), z));
        EXPECT_EQ(stepCalls, Emrys::internal::chunkCount(3000));
        EXPECT_EQ(stepColumns, 3000);
        EXPECT_EQ(rangeCalls, Emrys::internal::chunkCount(3000));
        EXPECT_EQ(filter.state().states(), sequential.state().states());
        EXPECT_NEAR(update->logLikelihood(), expected->logLikelihood(), 1e-12);
    };
    check(EmrysTest::ReversedPolicy{});
    check(EmrysTest::ThreadsPolicy{});
}

// A batched process model with an uncertain input moves the particles as its per-point form does
// from the same generator: each particle's draws, its noise and then its input, are taken in the
// same order before the one call.
TEST(Batched, ParticleFilterDrawsInputsForABatchedModelAsPerPoint)
{
    auto const accelerometer = accelerometerReading();
    auto byPoint = Emrys::ParticleFilter{forcedModel(), makeParticles(200, 19)};
    auto byBatch = Emrys::ParticleFilter{batchedForcedModel(), makeParticles(200, 19)};
    auto first = std::mt19937_64{23};
    auto second = std::mt19937_64{23};
    stepCalls = 0;
    byPoint.predict(first, 0.5, accelerometer);
    byBatch.predict(second, 0.5, accelerometer);
    EXPECT_EQ(stepCalls, 1);
    EXPECT_LT((byBatch.state().states() - byPoint.state().states()).cwiseAbs().maxCoeff(), 1e-12);
}

// GaussianMixtureFilter (steps 4, 13d; [mixture-filter], [fixed-step-models]): deduction and accessors, a fixed-step
// model fixes the step and the method, one component matches KalmanFilter under every method, components are reweighted
// by their likelihoods, Keep matches the exact posterior under mixture noise, an ambiguity is carried until a
// measurement resolves it, Gate drops components and rejects when none survive, and Keep with truncation rides out
// injected outliers.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>

#include "user_recursion.h"

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double accelDensity = 0.1;
constexpr double timeStep = 1.0;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

Vector4 const initialVariances{4.0, 4.0, 1.0, 1.0};

GaussianParams<State>
makeGaussian(Vector4 const & mean)
{
    return {State{}, mean, Matrix4{initialVariances.asDiagonal()}};
}

GaussianMixtureParams<State>
makeMixture(std::vector<std::pair<double, Vector4>> const & components)
{
    auto mixture = GaussianMixtureParams<State>{State{}};
    for (auto const & [weight, mean] : components)
    {
        mixture.add(std::log(weight), mean, Matrix4{initialVariances.asDiagonal()});
    }
    return mixture;
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

Matrix4
processNoise()
{
    Eigen::Matrix2d const i2 = Eigen::Matrix2d::Identity();
    auto q = Matrix4{};
    q << std::pow(timeStep, 3) / 3 * i2, timeStep * timeStep / 2 * i2, timeStep * timeStep / 2 * i2, timeStep * i2;
    return accelDensity * q;
}

Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};
Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

template <typename Method, typename Noise>
auto
makeFix(Eigen::Vector2d const & z, Method const & method, Noise noise)
{
    auto const position = [](auto const & in, auto const &, auto const & x) {
        return Emrys::block<Position>(in, x);
    };
    return observe(
        method,
        Emrys::measurementModel(position, Emrys::reads(Emrys::path<Position>), additiveNoise(std::move(noise))),
        z
    );
}

template <typename Method>
auto
makeFix(Eigen::Vector2d const & z, Method const & method)
{
    return makeFix(z, method, GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R});
}

// Range from the origin, which can't tell a position from its mirror image.
template <typename Method>
auto
makeRange(double z, Method const & method)
{
    auto const range = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(Emrys::block<Position>(in, x).squaredNorm()))};
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.04}})
    );
    return observe(method, range, Eigen::Matrix<double, 1, 1>{z});
}

struct Moments4
{
    Vector4 mean;
    Matrix4 cov;
};

Moments4
predictByHand(Moments4 const & x)
{
    Matrix4 const f = transitionMatrix();
    return {f * x.mean, f * x.cov * f.transpose() + processNoise()};
}

struct UpdatedByHand
{
    Moments4 posterior;
    double logLikelihood;
};

// The Kalman update of a position fix whose noise has mean `offset` and covariance `noise`.
UpdatedByHand
updateByHand(
    Moments4 const & x,
    Eigen::Vector2d const & z,
    Eigen::Vector2d const & offset,
    Eigen::Matrix2d const & noise
)
{
    Eigen::Matrix2d const s = H * x.cov * H.transpose() + noise;
    Eigen::Vector2d const innovation = z - H * x.mean - offset;
    Eigen::Matrix<double, 4, 2> const k = x.cov * H.transpose() * s.inverse();
    return {
        {x.mean + k * innovation, (Matrix4::Identity() - k * H) * x.cov},
        -0.5 * (2 * std::log(2 * pi) + std::log(s.determinant()) + innovation.dot(s.inverse() * innovation))
    };
}

double
logSumExp(std::vector<double> const & values)
{
    double const top = std::ranges::max(values);
    double sum = 0;
    for (double const v : values)
    {
        sum += std::exp(v - top);
    }
    return top + std::log(sum);
}

template <typename MixtureParams>
std::vector<double>
normalizedWeights(MixtureParams const & mixture)
{
    auto logWeights = std::vector<double>{};
    for (auto const & component : mixture)
    {
        logWeights.push_back(component.logWeight);
    }
    double const total = logSumExp(logWeights);
    auto weights = std::vector<double>{};
    for (double const w : logWeights)
    {
        weights.push_back(std::exp(w - total));
    }
    return weights;
}

template <typename Method>
class GaussianMixtureFilterMethodTest : public ::testing::Test
{};

using Methods = ::testing::Types<methods::Linearized, methods::Unscented<>>;

}  // namespace

TYPED_TEST_SUITE(GaussianMixtureFilterMethodTest, Methods);

// The deduction guide picks the space and model, the accessors work, prediction moves each
// component by the hand prediction and leaves the weights alone, and the state can be pruned in
// place between steps.
TEST(GaussianMixtureFilter, DeducesItsTypeAndHoldsItsParts)
{
    auto const prior = makeMixture({{0.5, Vector4{0, 0, 1, 0}}, {0.5, Vector4{10, 0, -1, 0}}});
    auto filter = GaussianMixtureFilter{makeModel(), prior};
    static_assert(std::is_same_v<decltype(filter), GaussianMixtureFilter<State, decltype(makeModel())>>);
    auto const unscented = GaussianMixtureFilter{makeModel(), prior, methods::unscented()};
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(unscented.method())>, methods::Unscented<>>);
    EXPECT_EQ(filter.state().size(), 2);
    EXPECT_EQ(filter.model().spectralDensity(), accelDensity * Eigen::Matrix2d::Identity());
    EXPECT_EQ(&std::as_const(filter).model(), &filter.model());
    EXPECT_EQ(&filter.recursion().model(), &filter.model());

    // Prediction moves each component and leaves the weights alone.
    filter.predict(timeStep);
    auto it = filter.state().begin();
    for (auto const & component : prior)
    {
        Moments4 const expected = predictByHand({component.component.mean(), component.component.covarianceLower()});
        EXPECT_LT((it->component.mean() - expected.mean).norm(), 1e-12);
        EXPECT_LT((it->component.covarianceLower() - expected.cov).norm(), 1e-12);
        EXPECT_EQ(it->logWeight, component.logWeight);
        ++it;
    }

    // The state can be reduced in place between steps.
    ASSERT_TRUE(filter.state().pruneToSize(1));
    EXPECT_EQ(filter.state().size(), 1);
}

// setState() replaces the state with a mixture of another size, after an update under Keep has
// exchanged the state with the filter's scratch: the steps that follow equal those of a filter made
// with the new state.
TEST(GaussianMixtureFilter, SetStateReplacesTheState)
{
    auto const first = makeMixture({{0.5, Vector4{0, 0, 1, 0}}, {0.5, Vector4{10, 0, -1, 0}}});
    auto const second =
        makeMixture({{0.2, Vector4{1, 1, 0, 0}}, {0.3, Vector4{4, 0, 1, 0}}, {0.5, Vector4{-3, 2, 0, 1}}});
    auto const fix = makeFix(Eigen::Vector2d{1.2, 0.4}, methods::linearized());
    auto filter = GaussianMixtureFilter{makeModel(), first};
    auto reference = GaussianMixtureFilter{makeModel(), second};
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(fix, reductions::keep()));

    filter.setState(second);
    ASSERT_EQ(filter.state().size(), 3);
    for (auto * const each : {&filter, &reference})
    {
        each->predict(timeStep);
        ASSERT_TRUE(each->update(fix, reductions::keep()));
    }
    ASSERT_EQ(filter.state().size(), reference.state().size());
    for (Eigen::Index i = 0; i < filter.state().size(); ++i)
    {
        auto const & replaced = filter.state()[i];
        auto const & expected = reference.state()[i];
        EXPECT_EQ(replaced.logWeight, expected.logWeight);
        EXPECT_EQ(replaced.component.mean(), expected.component.mean());
        EXPECT_EQ(replaced.component.covarianceLower(), expected.component.covarianceLower());
    }
}

// With a FixedStepModel [fixed-step-models] the filter deduces the model's method, with its parameters, and
// predict() takes no step. Over three fixes it matches a filter predicting over the step: exactly
// under Unscented, which takes the same path with the same noise, and to rounding under Linearized,
// which predicts by the cached transition.
TEST(GaussianMixtureFilter, AFixedStepModelFixesTheStepAndTheMethod)
{
    using Emrys::FixedStepModel;

    auto const prior = makeMixture({{0.5, Vector4{0, 0, 1, 0}}, {0.5, Vector4{10, 0, -1, 0}}});
    auto cached = GaussianMixtureFilter{FixedStepModel{makeModel(), State{}, timeStep}, prior};
    static_assert(std::is_same_v<decltype(cached)::MethodType, methods::Linearized>);
    auto unscented =
        GaussianMixtureFilter{FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()}, prior};
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    auto continuousCached = GaussianMixtureFilter{makeModel(), prior};
    auto continuousUnscented = GaussianMixtureFilter{makeModel(), prior, methods::unscented<0.5>()};

    auto const expectSame = [](auto const & filter, auto const & expected, double tolerance) {
        ASSERT_EQ(filter.state().size(), expected.state().size());
        auto it = expected.state().begin();
        for (auto const & component : filter.state())
        {
            EXPECT_LE((component.component.mean() - it->component.mean()).norm(), tolerance);
            EXPECT_LE((component.component.covarianceLower() - it->component.covarianceLower()).norm(), tolerance);
            EXPECT_NEAR(component.logWeight, it->logWeight, tolerance);
            ++it;
        }
    };
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {1.9, 1.1}, {3.2, 1.4}};
    for (Eigen::Vector2d const & z : fixes)
    {
        cached.predict();
        continuousCached.predict(timeStep);
        unscented.predict();
        continuousUnscented.predict(timeStep);
        ASSERT_TRUE(cached.update(makeFix(z, methods::linearized())));
        ASSERT_TRUE(continuousCached.update(makeFix(z, methods::linearized())));
        ASSERT_TRUE(unscented.update(makeFix(z, methods::unscented())));
        ASSERT_TRUE(continuousUnscented.update(makeFix(z, methods::unscented())));
        expectSame(cached, continuousCached, 1e-12);
        expectSame(unscented, continuousUnscented, 0.0);
    }
}

// A one-component mixture filter and a KalmanFilter with the same method agree exactly over three
// fixes in state, log-likelihood, and NIS.
TYPED_TEST(GaussianMixtureFilterMethodTest, OneComponentMatchesTheKalmanFilter)
{
    auto const mean = Vector4{0.0, 0.0, 1.0, 0.5};
    auto mixture = GaussianMixtureFilter{makeModel(), makeMixture({{1.0, mean}}), TypeParam{}};
    auto single = KalmanFilter{makeModel(), makeGaussian(mean), TypeParam{}};
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {1.9, 1.1}, {3.2, 1.4}};
    for (Eigen::Vector2d const & z : fixes)
    {
        mixture.predict(timeStep);
        single.predict(timeStep);
        auto const mixed = mixture.update(makeFix(z, TypeParam{}));
        auto const expected = single.update(makeFix(z, TypeParam{}));
        ASSERT_TRUE(mixed);
        ASSERT_TRUE(expected);
        ASSERT_EQ(mixture.state().size(), 1);
        auto const first = mixture.state().begin();
        auto const & component = *first;
        EXPECT_LT((component.component.mean() - single.state().mean()).norm(), 1e-12);
        EXPECT_LT((component.component.covarianceLower() - single.state().covarianceLower()).norm(), 1e-12);
        EXPECT_NEAR(component.logWeight, 0.0, 1e-15);
        EXPECT_NEAR(mixed->logLikelihood(), expected->logLikelihood(), 1e-12);
        ASSERT_EQ(mixed->components.size(), 1u);
        EXPECT_EQ(mixed->components[0].branches[0].nis(), expected->branches[0].nis());
        EXPECT_TRUE(mixed->accepted);
    }
}

// Two components updated by one fix: each component's posterior is its own Kalman update, its
// weight is prior times likelihood normalized, the result's log-likelihood is the log-sum-exp over
// components, and the nearer component gains weight.
TEST(GaussianMixtureFilter, ReweightsEachComponentByItsLikelihood)
{
    // Two components for where the target started; a fix between them, nearer the first.
    auto const priorWeights = std::array<double, 2>{0.3, 0.7};
    auto const means = std::array<Vector4, 2>{Vector4{0, 0, 1, 0}, Vector4{6, 2, 0, 1}};
    auto filter =
        GaussianMixtureFilter{makeModel(), makeMixture({{priorWeights[0], means[0]}, {priorWeights[1], means[1]}})};
    auto const z = Eigen::Vector2d{2.0, 0.5};
    filter.predict(timeStep);
    auto const result = filter.update(makeFix(z, methods::linearized()));
    ASSERT_TRUE(result);

    auto expected = std::array<UpdatedByHand, 2>{};
    auto logJoint = std::vector<double>{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        Moments4 const predicted = predictByHand({means[i], Matrix4{initialVariances.asDiagonal()}});
        expected[i] = updateByHand(predicted, z, Eigen::Vector2d::Zero(), R);
        logJoint.push_back(std::log(priorWeights[i]) + expected[i].logLikelihood);
        EXPECT_NEAR(result->components[i].logLikelihood(), expected[i].logLikelihood, 1e-12);
    }
    EXPECT_NEAR(result->logLikelihood(), logSumExp(logJoint), 1e-12);

    ASSERT_EQ(filter.state().size(), 2);
    auto const weights = normalizedWeights(filter.state());
    std::size_t i = 0;
    for (auto const & component : filter.state())
    {
        EXPECT_NEAR(weights[i], std::exp(logJoint[i] - logSumExp(logJoint)), 1e-12);
        EXPECT_LT((component.component.mean() - expected[i].posterior.mean).norm(), 1e-12);
        EXPECT_LT((component.component.covarianceLower() - expected[i].posterior.cov).norm(), 1e-12);
        ++i;
    }
    EXPECT_GT(weights[0], weights[1]);
}

// With affine models and two-component noise, three updates under Keep produce the exact
// 16-component posterior enumerated by hand over every sequence of noise components, with the
// summed log-likelihood equal to the log-evidence.
TEST(GaussianMixtureFilter, KeepMatchesTheExactPosteriorUnderMixtureNoise)
{
    // With affine models and mixture noise, the exact posterior is a mixture over every
    // sequence of noise components. Keep builds it; here it is enumerated by hand.
    auto const noiseWeights = std::array<double, 2>{0.8, 0.2};
    auto const offsets = std::array<Eigen::Vector2d, 2>{Eigen::Vector2d::Zero(), Eigen::Vector2d{1.5, -1.0}};
    auto const noiseCovs = std::array<Eigen::Matrix2d, 2>{R, 25 * R};
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    for (std::size_t k = 0; k < 2; ++k)
    {
        noise.add(std::log(noiseWeights[k]), offsets[k], noiseCovs[k]);
    }

    auto const prior = makeMixture({{0.4, Vector4{0, 0, 1, 0.5}}, {0.6, Vector4{1, -1, 0.5, 1}}});
    auto filter = GaussianMixtureFilter{makeModel(), prior};

    struct Branch
    {
        double logWeight;
        Moments4 moments;
    };

    auto exact = std::vector<Branch>{};
    for (auto const & component : prior)
    {
        exact.push_back({component.logWeight, {component.component.mean(), component.component.covarianceLower()}});
    }

    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {4.0, 1.1}, {3.2, 1.4}};
    double totalLogLikelihood = 0;
    for (Eigen::Vector2d const & z : fixes)
    {
        filter.predict(timeStep);
        auto const result = filter.update(makeFix(z, methods::linearized(), noise), reductions::keep());
        ASSERT_TRUE(result);
        totalLogLikelihood += result->logLikelihood();

        auto next = std::vector<Branch>{};
        for (Branch const & branch : exact)
        {
            Moments4 const predicted = predictByHand(branch.moments);
            for (std::size_t k = 0; k < 2; ++k)
            {
                auto const updated = updateByHand(predicted, z, offsets[k], noiseCovs[k]);
                next.push_back(
                    {branch.logWeight + std::log(noiseWeights[k]) + updated.logLikelihood, updated.posterior}
                );
            }
        }
        exact = std::move(next);
    }

    ASSERT_EQ(filter.state().size(), Eigen::Index(exact.size()));
    ASSERT_EQ(exact.size(), 16u);
    auto logWeights = std::vector<double>{};
    for (Branch const & branch : exact)
    {
        logWeights.push_back(branch.logWeight);
    }
    double const logEvidence = logSumExp(logWeights);
    EXPECT_NEAR(totalLogLikelihood, logEvidence - logSumExp({std::log(0.4), std::log(0.6)}), 1e-10);
    auto const weights = normalizedWeights(filter.state());
    std::size_t i = 0;
    for (auto const & component : filter.state())
    {
        EXPECT_NEAR(weights[i], std::exp(exact[i].logWeight - logEvidence), 1e-12) << "component " << i;
        EXPECT_LT((component.component.mean() - exact[i].moments.mean).norm(), 1e-10) << "component " << i;
        EXPECT_LT((component.component.covarianceLower() - exact[i].moments.cov).norm(), 1e-10) << "component " << i;
        ++i;
    }
}

// Two mirror-image components keep equal weight under five range measurements, where a single
// Gaussian would sit between them, and one position fix resolves the ambiguity.
TEST(GaussianMixtureFilter, CarriesAnAmbiguityUntilAMeasurementResolvesIt)
{
    // The target starts at one of two mirror-image positions. Ranges from the origin can't tell
    // them apart, so both components keep their weight; one position fix settles it.
    auto filter = GaussianMixtureFilter{
        makeModel(),
        makeMixture({{0.5, Vector4{10, 0, 0, 1}}, {0.5, Vector4{-10, 0, 0, 1}}}),
        methods::unscented()
    };
    auto truth = Vector4{10, 0, 0, 1};
    for (int step = 0; step < 5; ++step)
    {
        truth = transitionMatrix() * truth;
        filter.predict(timeStep);
        ASSERT_TRUE(filter.update(makeRange(truth.head<2>().norm(), methods::unscented())));
        auto const weights = normalizedWeights(filter.state());
        EXPECT_NEAR(weights[0], 0.5, 1e-9) << "step " << step;
        EXPECT_NEAR(weights[1], 0.5, 1e-9) << "step " << step;
    }

    // A single moment-matched Gaussian would sit near the origin, between the components.
    auto const blended = filter.state().toDensity()->moments();
    EXPECT_LT(std::abs(blended.mean()(0)), 1e-6);

    truth = transitionMatrix() * truth;
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeFix(truth.head<2>(), methods::unscented())));
    auto const weights = normalizedWeights(filter.state());
    EXPECT_GT(weights[0], 1 - 1e-12);
    auto const resolved = filter.state().toDensity()->moments();
    EXPECT_LT((resolved.mean().head<2>() - truth.head<2>()).norm(), 1.0);
}

// Under a Gate, a fix near one component drops the other and leaves a one-component state equal to
// the Kalman update; a fix far from both is rejected and leaves the prior; and a gate around Keep
// keeps each component's passing noise branches.
TEST(GaussianMixtureFilter, GateDropsComponentsAndRejectsWhenNoneSurvive)
{
    auto const means = std::array<Vector4, 2>{Vector4{0, 0, 0, 0}, Vector4{40, 0, 0, 0}};
    auto const initial = GaussianMixtureFilter{makeModel(), makeMixture({{0.5, means[0]}, {0.5, means[1]}})};

    // Near the first component only: the second fails the gate and is dropped.
    auto near = initial;
    auto const gated = near.update(makeFix(Eigen::Vector2d{0.5, -0.5}, methods::linearized()), reductions::gate(0.997));
    ASSERT_TRUE(gated);
    EXPECT_TRUE(gated->accepted);
    EXPECT_TRUE(gated->components[0].accepted);
    EXPECT_FALSE(gated->components[1].accepted);
    ASSERT_EQ(near.state().size(), 1);
    auto single = KalmanFilter{makeModel(), makeGaussian(means[0])};
    ASSERT_TRUE(single.update(makeFix(Eigen::Vector2d{0.5, -0.5}, methods::linearized())));
    EXPECT_LT((near.state().begin()->component.mean() - single.state().mean()).norm(), 1e-12);
    EXPECT_NEAR(near.state().begin()->logWeight, 0.0, 1e-15);

    // Far from both: rejected, and the state keeps its prior.
    auto far = initial;
    auto const rejected =
        far.update(makeFix(Eigen::Vector2d{20.0, 30.0}, methods::linearized()), reductions::gate(0.997));
    ASSERT_TRUE(rejected);
    EXPECT_FALSE(rejected->accepted);
    ASSERT_EQ(far.state().size(), 2);
    auto it = far.state().begin();
    for (auto const & component : initial.state())
    {
        EXPECT_EQ(it->component.mean(), component.component.mean());
        EXPECT_EQ(it->logWeight, component.logWeight);
        ++it;
    }

    // A gate around Keep keeps each component's passing noise branches.
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), 400 * R);
    auto kept = initial;
    auto const both = kept.update(
        makeFix(Eigen::Vector2d{0.5, -0.5}, methods::linearized(), noise),
        reductions::Gate(0.997, reductions::keep())
    );
    ASSERT_TRUE(both);
    EXPECT_TRUE(both->accepted);
    // The first component keeps both branches; the second keeps only its wide one.
    ASSERT_EQ(kept.state().size(), 3);

    // Each kept branch has the weight Keep alone gives it, w_i w_k p(z | i, k), renormalized over
    // the three: the second component is weighted by its wide branch's likelihood, not by the
    // whole of p(z | 2), which its rejected narrow branch dominates.
    auto all = initial;
    ASSERT_TRUE(all.update(makeFix(Eigen::Vector2d{0.5, -0.5}, methods::linearized(), noise), reductions::keep()));
    ASSERT_EQ(all.state().size(), 4);
    auto const keptOfAll = std::array<std::ptrdiff_t, 3>{0, 1, 3};
    auto total = Emrys::LogSumExp<double>{};
    for (auto const k : keptOfAll)
    {
        total.add(all.state()[k].logWeight);
    }
    for (std::ptrdiff_t j = 0; j < 3; ++j)
    {
        auto const & expected = all.state()[keptOfAll[std::size_t(j)]];
        auto const & actual = kept.state()[j];
        EXPECT_TRUE(actual.component.mean().isApprox(expected.component.mean(), 1e-12));
        EXPECT_NEAR(actual.logWeight, expected.logWeight - total.value(), 1e-12);
    }

    // A gate around Collapse weights each component by its kept branches' likelihood the same way.
    auto collapsed = initial;
    ASSERT_TRUE(
        collapsed.update(makeFix(Eigen::Vector2d{0.5, -0.5}, methods::linearized(), noise), reductions::Gate(0.997))
    );
    ASSERT_EQ(collapsed.state().size(), 2);
    auto first = Emrys::LogSumExp<double>{};
    first.add(all.state()[0].logWeight);
    first.add(all.state()[1].logWeight);
    EXPECT_NEAR(collapsed.state().begin()->logWeight, first.value() - total.value(), 1e-12);
    EXPECT_NEAR(collapsed.state()[1].logWeight, all.state()[3].logWeight - total.value(), 1e-12);
}

// posteriorIndices(i) gives the state's components that prior component i became, in order: under
// a gate over Collapse a dropped middle component has none and the one after it moves down; under a
// gate around Keep the first prior keeps both its noise branches and the second its wide one; and a
// rejected fix, through either path, leaves each prior where it was.
TEST(GaussianMixtureFilter, PosteriorIndicesSayWhereEachPriorWent)
{
    using Indices = std::vector<Eigen::Index>;
    // Not Indices(begin, end): libstdc++'s iota iterator over a long is not a C++17 iterator.
    auto const indices = [](auto const & result, Eigen::Index i) {
        return std::ranges::to<Indices>(result.posteriorIndices(i));
    };
    auto const near = Eigen::Vector2d{0.5, -0.5};
    auto const farAway = Eigen::Vector2d{2000.0, 3000.0};

    auto three = GaussianMixtureFilter{
        makeModel(),
        makeMixture({{0.4, Vector4{0, 0, 0, 0}}, {0.3, Vector4{40, 0, 0, 0}}, {0.3, Vector4{1, 0, 0, 0}}})
    };
    auto const gated = three.update(makeFix(near, methods::linearized()), reductions::gate(0.997));
    ASSERT_TRUE(gated);
    EXPECT_FALSE(gated->components[1].accepted);
    ASSERT_EQ(three.state().size(), 2);
    EXPECT_EQ(indices(*gated, 0), Indices{0});
    EXPECT_EQ(indices(*gated, 1), Indices{});
    EXPECT_EQ(indices(*gated, 2), Indices{1});
    // The fix pulls the components from 0 and 1 toward 0.5 from either side.
    EXPECT_LT(three.state()[0].component.mean()(0), 0.5);
    EXPECT_GT(three.state()[1].component.mean()(0), 0.5);

    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), 400 * R);
    auto const two = makeMixture({{0.5, Vector4{0, 0, 0, 0}}, {0.5, Vector4{40, 0, 0, 0}}});
    auto kept = GaussianMixtureFilter{makeModel(), two};
    auto const gateKeep = reductions::Gate(0.997, reductions::keep());
    auto const branches = kept.update(makeFix(near, methods::linearized(), noise), gateKeep);
    ASSERT_TRUE(branches);
    ASSERT_EQ(kept.state().size(), 3);
    EXPECT_EQ(indices(*branches, 0), (Indices{0, 1}));
    EXPECT_EQ(indices(*branches, 1), Indices{2});

    auto far = GaussianMixtureFilter{makeModel(), two};
    auto const rejected = far.update(makeFix(farAway, methods::linearized()), reductions::gate(0.997));
    ASSERT_TRUE(rejected);
    EXPECT_FALSE(rejected->accepted);
    EXPECT_EQ(indices(*rejected, 0), Indices{0});
    EXPECT_EQ(indices(*rejected, 1), Indices{1});
    auto const rejectedBranches = far.update(makeFix(farAway, methods::linearized(), noise), gateKeep);
    ASSERT_TRUE(rejectedBranches);
    EXPECT_FALSE(rejectedBranches->accepted);
    EXPECT_EQ(indices(*rejectedBranches, 0), Indices{0});
    EXPECT_EQ(indices(*rejectedBranches, 1), Indices{1});
}

// The Kalman filter's outlier test with Keep and truncation to eight components after each update
// stays within 5 units of the truth through 70 steps with an outlier every seventh.
TEST(GaussianMixtureFilter, KeepWithTruncationRejectsInjectedOutliers)
{
    // The outlier test of the Kalman filter, with every inlier/outlier branch kept and the
    // mixture pruned to its heaviest components after each update.
    constexpr int steps = 70;
    constexpr std::ptrdiff_t maxComponents = 8;
    Eigen::Matrix2d const inlierCov = Eigen::Matrix2d::Identity();
    auto robustNoise = GaussianMixtureParams<Linear<2>>{2};
    robustNoise.add(std::log(0.95), Eigen::Vector2d::Zero(), inlierCov);
    robustNoise.add(std::log(0.05), Eigen::Vector2d::Zero(), 1e4 * inlierCov);

    auto const start = Vector4{0.0, 0.0, 1.0, 0.5};
    auto filter = GaussianMixtureFilter{makeModel(), makeMixture({{1.0, start}})};
    auto rng = std::mt19937_64{17};
    auto gaussian = std::normal_distribution<double>{0.0, 1.0};
    Vector4 truth = start;
    double maxError = 0;
    for (int step = 1; step <= steps; ++step)
    {
        truth = transitionMatrix() * truth;
        bool const outlier = step % 7 == 0;
        Eigen::Vector2d const z = truth.head<2>() + Eigen::Vector2d{gaussian(rng), gaussian(rng)}
                                + (outlier ? Eigen::Vector2d{60.0, 0.0} : Eigen::Vector2d::Zero());
        filter.predict(timeStep);
        ASSERT_TRUE(filter.update(makeFix(z, methods::linearized(), robustNoise), reductions::keep()));
        ASSERT_TRUE(filter.state().pruneToSize(maxComponents));
        EXPECT_LE(filter.state().size(), maxComponents);
        if (step > 5)
        {
            auto const estimate = filter.state().toDensity()->moments();
            maxError = std::max(maxError, (estimate.mean().head<2>() - truth.head<2>()).norm());
        }
    }
    EXPECT_LT(maxError, 5.0);
}

namespace
{

// A reduction output that holds several components, as a mixture filter's does: what a policy
// kept of some branches, each with its log-weight among them.
struct KeptAll
{
    using ComponentType = GaussianParams<State>;
    static constexpr bool HoldsSeveral = true;

    std::vector<std::pair<double, ComponentType>> kept;

    void
    add(double weight, ComponentType component)
    {
        kept.emplace_back(weight, std::move(component));
    }
};

}  // namespace

// Keep adds every branch to the output with its log-weight and posterior, and a Gate over it adds
// the branch that passes with its own log-weight [reduction-policies].
TEST(Reduction, KeepAddsEveryBranch)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.8), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.2), Eigen::Vector2d{2.0, -1.0}, 16 * R);
    auto const updated =
        Emrys::update(makeGaussian(Vector4{0, 0, 1, 0.5}), makeFix({2.5, -0.5}, methods::linearized(), noise));
    ASSERT_TRUE(updated);
    auto kept = KeptAll{};
    reductions::keep()(updated->branches, kept);
    ASSERT_EQ(kept.kept.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_EQ(kept.kept[i].first, updated->branches[i].logWeight);
        EXPECT_EQ(kept.kept[i].second.mean(), updated->branches[i].posterior.mean());
    }

    // A threshold between the two NIS passes the branch of the smaller alone.
    double const nis0 = updated->branches[0].nis();
    double const nis1 = updated->branches[1].nis();
    ASSERT_NE(nis0, nis1);
    std::size_t const passing = nis0 < nis1 ? 0 : 1;
    auto gated = KeptAll{};
    reductions::gate(Emrys::NisThreshold{(nis0 + nis1) / 2}, reductions::keep())(updated->branches, gated);
    ASSERT_EQ(gated.kept.size(), 1u);
    EXPECT_EQ(gated.kept[0].first, updated->branches[passing].logWeight);
    EXPECT_EQ(gated.kept[0].second.mean(), updated->branches[passing].posterior.mean());
}

// A policy of the user's own that adds some of each component's branches to the output, here the
// inlier ones, leaves the components Keep gives those branches, with the weights Keep gives them
// renormalized over the ones kept: a branch it drops is not credited to the rest [reduction-policies].
TEST(GaussianMixtureFilter, APolicyOfTheUsersOwnAddsTheBranchesItKeeps)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.8), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.2), Eigen::Vector2d{2.0, -1.0}, 16 * R);
    auto const initial =
        GaussianMixtureFilter{makeModel(), makeMixture({{0.4, Vector4{0, 0, 1, 0.5}}, {0.6, Vector4{1, -1, 0.5, 1}}})};
    auto const fix = makeFix(Eigen::Vector2d{1.2, 0.4}, methods::linearized(), noise);

    auto all = initial;
    auto const keptAll = all.update(fix, reductions::keep());
    ASSERT_TRUE(keptAll);
    ASSERT_EQ(all.state().size(), 4);

    auto const keepInliers = [](auto && branches, auto & out) {
        for (auto & branch : branches)
        {
            if (branch.component == 0)
            {
                out.add(branch.logWeight, std::move(branch.posterior));
            }
        }
    };
    auto inliers = initial;
    auto const result = inliers.update(fix, keepInliers);
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->accepted);
    ASSERT_EQ(inliers.state().size(), 2);
    EXPECT_EQ(result->posteriorIndices(0).size(), 1u);
    EXPECT_EQ(result->posteriorIndices(1).size(), 1u);

    // Keep's components 0 and 2 are the two priors' inlier branches.
    auto const keptOfAll = std::array<std::ptrdiff_t, 2>{0, 2};
    auto total = Emrys::LogSumExp<double>{};
    for (auto const k : keptOfAll)
    {
        total.add((all.state().begin() + k)->logWeight);
    }
    for (std::ptrdiff_t j = 0; j < 2; ++j)
    {
        auto const & expected = *(all.state().begin() + keptOfAll[std::size_t(j)]);
        auto const & actual = *(inliers.state().begin() + j);
        EXPECT_EQ(actual.component.mean(), expected.component.mean());
        EXPECT_NEAR(actual.logWeight, expected.logWeight - total.value(), 1e-12);
    }
    // The evidence counts every branch, so it is Keep's.
    EXPECT_NEAR(result->logLikelihood(), keptAll->logLikelihood(), 1e-12);
}

// A recursion of the user's own (tests/user_recursion.h) runs on each component: MixtureFilter deduces it from the
// constructor as its last argument, after a prediction every component's covariance is the plain mixture filter's
// times alpha squared with the means and weights unchanged, and an update on that prior matches the plain filter's
// [filters-and-recursions], [user-extension-points].
TEST(GaussianMixtureFilter, UserRecursionRunsOnEachComponent)
{
    using Model = decltype(makeModel());
    using Inner = Emrys::KalmanRecursion<Emrys::Gaussian<State>, Model>;
    using Fading = EmrysTest::FadingMemory<Inner>;
    constexpr double alpha = 1.1;
    auto const initial = makeMixture({{0.5, Vector4{0, 0, 1, 0}}, {0.5, Vector4{10, 0, -1, 0}}});
    auto filter = Emrys::MixtureFilter{EmrysTest::FadingMemory{Inner{makeModel()}, alpha}, initial};
    static_assert(
        std::is_same_v<
            decltype(filter),
            Emrys::MixtureFilter<Emrys::Gaussian<State>, Model, methods::Linearized, Emrys::DefaultAllocator, Fading>>
    );
    static_assert(std::is_same_v<decltype(filter)::RecursionType, Fading>);
    auto plain = GaussianMixtureFilter{makeModel(), initial};
    static_assert(std::is_same_v<decltype(plain)::RecursionType, Inner>);

    filter.predict(timeStep);
    plain.predict(timeStep);
    ASSERT_EQ(filter.state().size(), plain.state().size());
    for (Eigen::Index i = 0; i < filter.state().size(); ++i)
    {
        auto const & faded = filter.state()[i];
        auto const & reference = plain.state()[i];
        EXPECT_DOUBLE_EQ(faded.logWeight, reference.logWeight);
        EXPECT_LT((faded.component.mean() - reference.component.mean()).norm(), 1e-12);
        EXPECT_LT(
            (faded.component.covarianceLower() - alpha * alpha * reference.component.covarianceLower()).norm(),
            1e-12
        );
    }

    plain.state() = filter.state();
    auto const z = Eigen::Vector2d{1.2, 0.4};
    auto const updated = filter.update(makeFix(z, methods::linearized()));
    auto const reference = plain.update(makeFix(z, methods::linearized()));
    ASSERT_TRUE(updated && reference);
    EXPECT_TRUE(updated->accepted);
    EXPECT_DOUBLE_EQ(updated->logLikelihood(), reference->logLikelihood());
    ASSERT_EQ(filter.state().size(), plain.state().size());
    for (Eigen::Index i = 0; i < filter.state().size(); ++i)
    {
        auto const & faded = filter.state()[i];
        auto const & expected = plain.state()[i];
        EXPECT_DOUBLE_EQ(faded.logWeight, expected.logWeight);
        EXPECT_LT((faded.component.mean() - expected.component.mean()).norm(), 1e-12);
    }
}

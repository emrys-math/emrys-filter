// MixtureFilter over hybrid components, a mixture of interacting multiple model filters (step 15c;
// [component-first], [filters-and-recursions]): the recursion a hybrid component names, one component equals the IMM
// filter, two components equal two IMMs run apart with weights updated by their likelihoods, noise by mode axis reaches
// each IMM's Markov outlier axis, and a gate drops a component and rejects the measurement when none survives.
// Square-root components ([gaussian-forms]) run the same filter: a mixture of them, and a mixture of square-root
// hybrids, track the covariance form's to rounding.
// Expected values come from ImmFilter itself, which tests/Filters/imm_filter.cpp checks against a hand computation.
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::DiscreteProduct;
using Emrys::Gaussian;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::ImmFilter;
using Emrys::ImmRecursion;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::MixtureFilter;
using Emrys::MixtureParams;
using Emrys::Named;
using Emrys::ScalarOf;
using Emrys::SqrtGaussian;
using Emrys::SqrtGaussianParams;
using Emrys::SqrtKalmanFilter;
using Emrys::switchingModel;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Cruise
{};

struct Maneuvering
{};

struct Maneuver : Named<Cruise, Maneuvering>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Named<Inlier, Outlier>
{};

using Mode = DiscreteProduct<Maneuver, Sensor>;
using HybridDensity = HybridGaussian<State, Mode>;
using HybridState = HybridGaussianParams<State, Mode>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double timeStep = 1.0;
constexpr double quiet = 0.01;
constexpr double agile = 4.0;
Vector4 const variances{4.0, 4.0, 1.0, 1.0};

auto
makeModel(double density)
{
    return ConstantVelocity<Position, Velocity>{density};
}

auto
makeSwitching()
{
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    return switchingModel<Maneuver>(chain, makeModel(quiet), makeModel(agile));
}

using Model = decltype(makeSwitching());

HybridState
makeHybrid(Vector4 const & mean)
{
    return HybridState{
        CategoricalParams<Mode>{},
        GaussianParams<State>{State{}, mean, Matrix4{variances.asDiagonal()}}
    };
}

MixtureParams<HybridDensity>
makeMixture(std::vector<std::pair<double, Vector4>> const & components)
{
    auto mixture = MixtureParams<HybridDensity>{State{}};
    for (auto const & [weight, mean] : components)
    {
        mixture.add(std::log(weight), makeHybrid(mean));
    }
    return mixture;
}

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};
constexpr double outlierScale = 100;

// A position fix with the noise policy \a noise.
template <typename NoisePolicy>
auto
observeFix(Eigen::Vector2d const & z, NoisePolicy noise)
{
    return observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(Eigen::Matrix2d::Identity(), Emrys::reads(Emrys::path<Position>), std::move(noise)),
        z
    );
}

template <typename Noise>
auto
makeFix(Eigen::Vector2d const & z, Noise noise)
{
    return observeFix(z, additiveNoise(std::move(noise)));
}

auto
makeFix(Eigen::Vector2d const & z)
{
    return makeFix(z, GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R});
}

// Inlier noise R and outlier noise 100 R, one component per value of Sensor, by that mode axis.
auto
makeRobustFix(Eigen::Vector2d const & z)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), outlierScale * R);
    return observeFix(z, additiveNoise(std::move(noise)).byMode<Sensor>());
}

// The fixes a run takes, in order: five cycles that drift along x, the third an outlier.
std::vector<Eigen::Vector2d>
fixes()
{
    return {{1.0, 0.1}, {2.1, 0.0}, {30.0, -20.0}, {4.0, 0.2}, {5.1, 0.1}};
}

// Whether two hybrid states are equal to rounding: the mode probabilities and each mode's moments.
void
expectSameHybrid(HybridState const & actual, HybridState const & expected, double tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_LT(
        (actual.modes().toDensity().value().probabilities() - expected.modes().toDensity().value().probabilities())
            .matrix()
            .norm(),
        tolerance
    );
    for (Eigen::Index m = 0; m < actual.size(); ++m)
    {
        EXPECT_LT((actual.conditional(m).mean() - expected.conditional(m).mean()).norm(), tolerance) << "mode " << m;
        EXPECT_LT(
            (actual.conditional(m).covarianceLower() - expected.conditional(m).covarianceLower()).norm(),
            tolerance
        ) << "mode "
          << m;
    }
}

}  // namespace

// A mixture of hybrids deduces the filter over its component type and runs the IMM recursion on
// each component, which holds scratch and so is not reentrant.
TEST(MixtureFilter, HybridComponentsRunTheImmRecursion)
{
    auto filter = MixtureFilter{makeSwitching(), makeMixture({{1.0, Vector4::Zero()}})};
    static_assert(std::is_same_v<decltype(filter), MixtureFilter<HybridDensity, Model>>);
    using Filter = decltype(filter);
    static_assert(std::is_same_v<Filter::RecursionType, ImmRecursion<Gaussian<State>, Model, methods::Linearized>>);
    static_assert(!Filter::RecursionType::Reentrant);
    static_assert(std::is_same_v<Filter::StateType, MixtureParams<HybridDensity>>);
    EXPECT_EQ(filter.state().size(), 1);
    EXPECT_EQ(filter.model().modeSpace().cardinality(), 4);
}

// With one hybrid component the mixture filter is the IMM filter over that hybrid: the same
// predictions, the same updates with the fix's own noise and through the Sensor axis, and the
// same evidence, to rounding.
TEST(MixtureFilter, OneHybridComponentIsTheImmFilter)
{
    auto const start = Vector4{0.0, 0.0, 1.0, 0.0};
    auto mixture = MixtureFilter{makeSwitching(), makeMixture({{1.0, start}})};
    auto imm = ImmFilter{makeSwitching(), makeHybrid(start)};

    for (auto const & z : fixes())
    {
        mixture.predict(timeStep);
        imm.predict(timeStep);
        expectSameHybrid(mixture.state().begin()->component, imm.state(), 1e-12);

        auto const mixed = mixture.update(makeFix(z));
        auto const single = imm.update(makeFix(z));
        ASSERT_TRUE(mixed);
        ASSERT_TRUE(single);
        EXPECT_TRUE(mixed->accepted);
        EXPECT_NEAR(mixed->logLikelihood(), single->logLikelihood(), 1e-12);
        ASSERT_EQ(mixed->components.size(), 1u);
        EXPECT_NEAR(mixed->components.front().logLikelihood(), single->logLikelihood(), 1e-12);
        expectSameHybrid(mixture.state().begin()->component, imm.state(), 1e-12);

        auto const mixedRobust = mixture.update(makeRobustFix(z));
        auto const singleRobust = imm.update(makeRobustFix(z));
        ASSERT_TRUE(mixedRobust);
        ASSERT_TRUE(singleRobust);
        EXPECT_NEAR(mixedRobust->logLikelihood(), singleRobust->logLikelihood(), 1e-12);
        expectSameHybrid(mixture.state().begin()->component, imm.state(), 1e-12);
        EXPECT_NEAR(mixture.state().begin()->logWeight, 0.0, 1e-12);
    }
}

// Two hybrid components are two IMM filters run apart: each component's state is its IMM's, and
// the weights are the prior weights times each IMM's likelihoods, normalized, with the evidence
// their log-sum-exp.
TEST(MixtureFilter, TwoHybridComponentsAreTwoImmsWeightedByTheirLikelihoods)
{
    auto const priors = std::vector<std::pair<double, Vector4>>{
        {0.3, Vector4{0.0, 0.0, 1.0, 0.0}},
        {0.7, Vector4{0.5, -0.5, 0.8, 0.1}}
    };
    auto mixture = MixtureFilter{makeSwitching(), makeMixture(priors)};
    auto imms = std::vector<ImmFilter<Gaussian<State>, Model>>{};
    auto logWeights = std::vector<double>{};
    for (auto const & [weight, mean] : priors)
    {
        imms.emplace_back(makeSwitching(), makeHybrid(mean));
        logWeights.push_back(std::log(weight));
    }

    for (auto const & z : fixes())
    {
        mixture.predict(timeStep);
        for (auto & imm : imms)
        {
            imm.predict(timeStep);
        }
        auto const update = mixture.update(makeRobustFix(z));
        ASSERT_TRUE(update);
        ASSERT_EQ(update->components.size(), imms.size());

        double evidence = 0;
        for (std::size_t i = 0; i < imms.size(); ++i)
        {
            auto const single = imms[i].update(makeRobustFix(z));
            ASSERT_TRUE(single);
            EXPECT_NEAR(update->components[i].logLikelihood(), single->logLikelihood(), 1e-12);
            evidence += std::exp(logWeights[i] + single->logLikelihood());
            logWeights[i] += single->logLikelihood();
        }
        EXPECT_NEAR(update->logLikelihood(), std::log(evidence), 1e-12);
        double const logTotal = std::log(evidence);
        for (auto & logWeight : logWeights)
        {
            logWeight -= logTotal;
        }

        std::size_t i = 0;
        for (auto const & entry : mixture.state())
        {
            EXPECT_NEAR(entry.logWeight, logWeights[i], 1e-12) << "component " << i;
            expectSameHybrid(entry.component, imms[i].state(), 1e-12);
            ++i;
        }
    }
}

// A gate drops a hybrid component none of whose modes passes, as it drops a Gaussian one, and
// rejects the measurement, leaving the state alone, when no component survives; the result's
// posterior indices say so, as a Gaussian mixture's do.
TEST(MixtureFilter, GateDropsHybridComponentsAndRejectsWhenNoneSurvive)
{
    auto filter = MixtureFilter{
        makeSwitching(),
        makeMixture({{0.5, Vector4{0.0, 0.0, 0.0, 0.0}}, {0.5, Vector4{200.0, 200.0, 0.0, 0.0}}})
    };
    auto const gate = reductions::gate(0.997);

    auto const dropped = filter.update(makeFix(Eigen::Vector2d{0.5, 0.5}), gate);
    ASSERT_TRUE(dropped);
    EXPECT_TRUE(dropped->accepted);
    ASSERT_EQ(dropped->components.size(), 2u);
    EXPECT_TRUE(dropped->components[0].accepted);
    EXPECT_FALSE(dropped->components[1].accepted);
    ASSERT_EQ(filter.state().size(), 1);
    EXPECT_EQ(dropped->posteriorIndices(0).size(), 1);
    EXPECT_TRUE(dropped->posteriorIndices(1).empty());
    EXPECT_NEAR(filter.state().begin()->logWeight, 0.0, 1e-12);
    EXPECT_LT(
        (filter.state().begin()->component.toDensity()->collapsed().mean().head<2>() - Eigen::Vector2d{0.5, 0.5})
            .norm(),
        1.0
    );

    auto const before = filter.state().begin()->component;
    auto const rejected = filter.update(makeFix(Eigen::Vector2d{-500.0, 300.0}), gate);
    ASSERT_TRUE(rejected);
    EXPECT_FALSE(rejected->accepted);
    ASSERT_EQ(filter.state().size(), 1);
    EXPECT_EQ(rejected->posteriorIndices(0).front(), 0);
    expectSameHybrid(filter.state().begin()->component, before, 1e-15);
}

namespace
{

using SqrtHybridDensity = Hybrid<SqrtGaussian<State>, Mode>;
using SqrtHybridState = Emrys::HybridParams<SqrtGaussian<State>, Mode>;

// The square-root form of a Gaussian, by the Cholesky factor of its covariance.
SqrtGaussianParams<State>
toSqrtForm(GaussianParams<State> const & gaussian)
{
    Matrix4 const factor = gaussian.covarianceLower().selfadjointView<Eigen::Lower>().llt().matrixL();
    return SqrtGaussianParams<State>{State{}, gaussian.mean(), factor};
}

// Whether a square-root form has a Gaussian's mean and covariance, to rounding.
void
expectSameGaussian(SqrtGaussianParams<State> const & actual, GaussianParams<State> const & expected, double tolerance)
{
    EXPECT_LT((actual.mean() - expected.mean()).norm(), tolerance);
    Matrix4 const covariance = expected.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((actual.covariance().value() - covariance).norm(), tolerance);
}

}  // namespace

// A mixture of square-root forms runs the KalmanRecursion over the square-root form on each
// component; one component is the SqrtKalmanFilter, and two track the covariance form's mixture,
// weights and evidence included, so the factors never become covariances along the way.
TEST(MixtureFilter, SqrtComponentsTrackTheCovarianceForm)
{
    auto const starts =
        std::vector<std::pair<double, Vector4>>{{0.7, {0.0, 0.0, 1.0, 0.0}}, {0.3, {3.0, 1.0, 0.0, 1.0}}};
    auto covariance = GaussianMixtureParams<State>{State{}};
    auto squareRoot = MixtureParams<SqrtGaussian<State>>{State{}};
    for (auto const & [weight, mean] : starts)
    {
        auto const gaussian = GaussianParams<State>{State{}, mean, Matrix4{variances.asDiagonal()}};
        covariance.add(std::log(weight), gaussian);
        squareRoot.add(std::log(weight), toSqrtForm(gaussian));
    }
    auto expected = MixtureFilter{makeModel(agile), covariance};
    auto filter = MixtureFilter{makeModel(agile), squareRoot};
    using Filter = decltype(filter);
    static_assert(
        std::is_same_v<Filter::RecursionType, Emrys::KalmanRecursion<SqrtGaussian<State>, decltype(makeModel(agile))>>
    );
    auto single = SqrtKalmanFilter{makeModel(agile), toSqrtForm(covariance.begin()->component)};
    auto oneComponent = MixtureParams<SqrtGaussian<State>>{State{}};
    oneComponent.add(0.0, single.state());
    auto alone = MixtureFilter{makeModel(agile), oneComponent};

    for (auto const & z : fixes())
    {
        expected.predict(timeStep);
        ASSERT_TRUE(filter.predict(timeStep));
        ASSERT_TRUE(single.predict(timeStep));
        ASSERT_TRUE(alone.predict(timeStep));
        auto const updated = expected.update(makeFix(z));
        auto const sqrtUpdated = filter.update(makeFix(z));
        ASSERT_TRUE(updated);
        ASSERT_TRUE(sqrtUpdated);
        EXPECT_NEAR(sqrtUpdated->logLikelihood(), updated->logLikelihood(), 1e-10);
        ASSERT_EQ(filter.state().size(), expected.state().size());
        for (Eigen::Index i = 0; i < filter.state().size(); ++i)
        {
            auto const & entry = filter.state()[i];
            auto const & reference = expected.state()[i];
            EXPECT_NEAR(entry.logWeight, reference.logWeight, 1e-10) << "component " << i;
            expectSameGaussian(entry.component, reference.component, 1e-9);
        }

        auto const singleUpdated = single.update(makeFix(z));
        auto const aloneUpdated = alone.update(makeFix(z));
        ASSERT_TRUE(singleUpdated);
        ASSERT_TRUE(aloneUpdated);
        EXPECT_NEAR(aloneUpdated->logLikelihood(), singleUpdated->logLikelihood(), 1e-12);
        auto const & component = alone.state().begin()->component;
        EXPECT_LT((component.mean() - single.state().mean()).norm(), 1e-12);
        EXPECT_LT((component.root() - single.state().root()).norm(), 1e-12);
    }
}

// A mixture of square-root hybrids runs the IMM recursion over the square-root form, whose mixing
// collapses factors by triangularization: it tracks the covariance form's mixture of hybrids, mode
// probabilities, moments per mode, and weights, to rounding, refilling one result at every step.
TEST(MixtureFilter, SqrtHybridComponentsTrackTheCovarianceForm)
{
    auto const starts =
        std::vector<std::pair<double, Vector4>>{{0.6, {0.0, 0.0, 1.0, 0.0}}, {0.4, {2.0, -1.0, 0.5, 0.5}}};
    auto covariance = makeMixture(starts);
    auto squareRoot = MixtureParams<SqrtHybridDensity>{State{}};
    for (auto const & [weight, mean] : starts)
    {
        squareRoot.add(
            std::log(weight),
            SqrtHybridState{
                CategoricalParams<Mode>{},
                toSqrtForm(GaussianParams<State>{State{}, mean, Matrix4{variances.asDiagonal()}})
            }
        );
    }
    auto expected = MixtureFilter{makeSwitching(), covariance};
    auto filter = MixtureFilter{makeSwitching(), squareRoot};
    static_assert(
        std::is_same_v<decltype(filter)::RecursionType, ImmRecursion<SqrtGaussian<State>, Model, methods::Linearized>>
    );

    using Update = decltype(filter.update(makeFix(fixes().front())))::value_type;
    auto sqrtUpdated = Update{};
    for (auto const & z : fixes())
    {
        expected.predict(timeStep);
        ASSERT_TRUE(filter.predict(timeStep));
        auto const updated = expected.update(makeFix(z));
        ASSERT_TRUE(updated);
        ASSERT_TRUE(filter.update(makeFix(z), reductions::collapse(), sqrtUpdated));
        EXPECT_NEAR(sqrtUpdated.logLikelihood(), updated->logLikelihood(), 1e-10);
        ASSERT_EQ(filter.state().size(), expected.state().size());
        for (Eigen::Index i = 0; i < filter.state().size(); ++i)
        {
            auto const & entry = filter.state()[i];
            auto const & reference = expected.state()[i];
            EXPECT_NEAR(entry.logWeight, reference.logWeight, 1e-10) << "component " << i;
            EXPECT_LT(
                (entry.component.modes().toDensity().value().probabilities()
                 - reference.component.modes().toDensity().value().probabilities())
                    .matrix()
                    .norm(),
                1e-10
            );
            for (Eigen::Index m = 0; m < entry.component.size(); ++m)
            {
                expectSameGaussian(entry.component.conditional(m), reference.component.conditional(m), 1e-9);
            }
        }
    }
}

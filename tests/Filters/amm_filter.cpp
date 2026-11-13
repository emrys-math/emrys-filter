// AmmFilter (step 17c, [autonomous-multiple-model]): the IMM on the identity chain is a bank of
// independent Kalman filters with Bayes-weighted mode probabilities, the IMM's predict() skips the
// mixing on the identity chain and agrees with the full collapse, the filter deduces its arguments
// with a method and with fixed-step models, and FloorModes keeps a mode recoverable after a fix that
// rules it out. Headers: Filters/AmmFilter.h, Filters/ImmFilter.h, Distributions/Upkeep.h.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Filters/AmmFilter.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

using Emrys::additiveNoise;
using Emrys::AmmFilter;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::Gaussian;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::ImmFilter;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::ScalarOf;
using Emrys::SwitchingModel;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;
namespace upkeep = Emrys::upkeep;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Slow
{};

struct Fast
{};

struct Regime : Named<Slow, Fast>
{};

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double timeStep = 1.0;
constexpr double quiet = 0.01;
constexpr double agile = 4.0;

auto
makeModel(double density)
{
    return ConstantVelocity<Position, Velocity>{density};
}

GaussianParams<State>
makeGaussian(Vector4 const & mean, Vector4 const & variances)
{
    return {State{}, mean, Matrix4{variances.asDiagonal()}};
}

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

auto
makeFix(Eigen::Vector2d const & z)
{
    return observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R})
        ),
        z
    );
}

// The fixes of a target that turns: east, then north.
Eigen::Vector2d
fixAt(int k)
{
    return k <= 3 ? Eigen::Vector2d{1.1 * k, 0.2} : Eigen::Vector2d{3.3, 0.2 + 1.5 * (k - 3)};
}

HybridGaussianParams<State, Regime>
makeInitial()
{
    return {
        CategoricalParams<Regime>::fromWeights(Regime{}, Eigen::Array2d{0.7, 0.3}),
        makeGaussian(Vector4{0, 0, 1, 0.5}, Vector4{4, 4, 1, 1})
    };
}

}  // namespace

// Each mode's Gaussian equals a KalmanFilter's with that mode's process model, step for step, and
// the mode probabilities are the prior's times each filter's likelihood, normalized (Bayes' rule
// in plain arithmetic over the Kalman filters' log-likelihoods); the update's log-likelihood is
// the log of their sum. The tolerance is rounding: the two run the same steps.
TEST(AmmFilter, IsABankOfKalmanFilters)
{
    auto filter = AmmFilter{makeInitial(), makeModel(quiet), makeModel(agile)};
    static_assert(std::is_base_of_v<
                  ImmFilter<
                      Gaussian<State>,
                      SwitchingModel<Regime, double, Regime, decltype(makeModel(quiet)), decltype(makeModel(agile))>>,
                  decltype(filter)>);
    EXPECT_TRUE(filter.model().chain().matrix().isIdentity(0.0));

    auto const initial = makeInitial();
    auto bank = std::vector<KalmanFilter<Gaussian<State>, decltype(makeModel(quiet))>>{
        KalmanFilter{makeModel(quiet), initial.conditional(0)},
        KalmanFilter{makeModel(agile), initial.conditional(1)}
    };
    auto probabilities = Eigen::Array2d{0.7, 0.3};
    for (int k = 1; k <= 6; ++k)
    {
        filter.predict(timeStep);
        auto const update = filter.update(makeFix(fixAt(k)));
        ASSERT_TRUE(update);
        ASSERT_TRUE(update->accepted);

        double evidence = 0;
        for (std::size_t m = 0; m < bank.size(); ++m)
        {
            bank[m].predict(timeStep);
            auto const kalmanUpdate = bank[m].update(makeFix(fixAt(k)));
            ASSERT_TRUE(kalmanUpdate);
            probabilities(Eigen::Index(m)) *= std::exp(kalmanUpdate->logLikelihood());
            evidence += probabilities(Eigen::Index(m));
        }
        probabilities /= evidence;
        EXPECT_NEAR(update->logLikelihood(), std::log(evidence), 1e-12) << "step " << k;
        for (Eigen::Index m = 0; m < 2; ++m)
        {
            auto const & kalman = bank[std::size_t(m)].state();
            EXPECT_NEAR(filter.state().modes().toDensity().value().probability(m), probabilities(m), 1e-12)
                << "step " << k;
            EXPECT_TRUE(filter.state().conditional(m).mean().isApprox(kalman.mean(), 1e-12)) << "step " << k;
            EXPECT_TRUE(filter.state().conditional(m).covarianceLower().isApprox(kalman.covarianceLower(), 1e-12))
                << "step " << k;
        }
    }
}

// On the identity chain, predict() leaves the mode probabilities bitwise as they were, and agrees
// to rounding with an IMM whose chain is the identity but for off-diagonal probabilities of
// 1e-300, which takes the full collapse: their mixing weights differ from the identity's by about
// 1e-300 of the probabilities.
TEST(AmmFilter, SkipsTheMixingAndAgreesWithTheFullCollapse)
{
    auto filter = AmmFilter{makeInitial(), makeModel(quiet), makeModel(agile)};
    auto const nearlyIdentity = MarkovChain<Regime>{Eigen::Matrix2d{{1.0, 1e-300}, {1e-300, 1.0}}};
    auto imm = ImmFilter{SwitchingModel{nearlyIdentity, makeModel(quiet), makeModel(agile)}, makeInitial()};
    for (int k = 1; k <= 6; ++k)
    {
        auto const before = filter.state().modes().toDensity().value().logProbabilities();
        filter.predict(timeStep);
        imm.predict(timeStep);
        EXPECT_TRUE((filter.state().modes().toDensity().value().logProbabilities() == before).all()) << "step " << k;
        for (Eigen::Index m = 0; m < 2; ++m)
        {
            EXPECT_NEAR(
                filter.state().modes().toDensity().value().probability(m),
                imm.state().modes().toDensity().value().probability(m),
                1e-12
            );
            EXPECT_TRUE(filter.state().conditional(m).mean().isApprox(imm.state().conditional(m).mean(), 1e-12));
            EXPECT_TRUE(filter.state().conditional(m).covarianceLower().isApprox(
                imm.state().conditional(m).covarianceLower(),
                1e-12
            ));
        }
        ASSERT_TRUE(filter.update(makeFix(fixAt(k))));
        ASSERT_TRUE(imm.update(makeFix(fixAt(k))));
    }
}

// The filter deduces its type with the default method, with a method given first, and with
// fixed-step models, whose predict() takes no step and moves the modes as
// the same models' filter with a time step does (the fixed step is that step).
TEST(AmmFilter, DeducesItsArguments)
{
    auto const linearized = AmmFilter{makeInitial(), makeModel(quiet), makeModel(agile)};
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(linearized)>::MethodType, methods::Linearized>);

    auto unscented = AmmFilter{methods::unscented(), makeInitial(), makeModel(quiet), makeModel(agile)};
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<>>);
    unscented.predict(timeStep);
    EXPECT_TRUE(unscented.update(makeFix(fixAt(1))).value().accepted);

    // One model shared by every mode.
    auto const shared = AmmFilter{makeInitial(), makeModel(quiet)};
    static_assert(std::remove_cvref_t<decltype(shared)>::ModelType::Shared);

    auto fixedStep = AmmFilter{
        makeInitial(),
        FixedStepModel{makeModel(quiet), State{}, timeStep},
        FixedStepModel{makeModel(agile), State{}, timeStep}
    };
    auto timed = AmmFilter{makeInitial(), makeModel(quiet), makeModel(agile)};
    fixedStep.predict();
    timed.predict(timeStep);
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        EXPECT_TRUE(fixedStep.state().conditional(m).covarianceLower().isApprox(
            timed.state().conditional(m).covarianceLower(),
            1e-12
        ));
    }
}

// A confident slow mode at the origin and a vague fast one: one outlier 30 m away fails the gate
// in the slow mode only, which takes its probability to exactly zero. With no chain to bring it
// back, the plain filter keeps it at zero through ten fixes at the origin, which the slow mode
// explains far better; with FloorModes applied after each update it returns above 0.99.
TEST(AmmFilter, FloorKeepsARuledOutModeRecoverable)
{
    auto const conditionals = std::vector<GaussianParams<State>>{
        makeGaussian(Vector4::Zero(), Vector4{0.5, 0.5, 0.1, 0.1}),
        makeGaussian(Vector4::Zero(), Vector4{400, 400, 1, 1})
    };
    auto const initial = HybridGaussianParams<State, Regime>{CategoricalParams<Regime>{}, conditionals};
    auto plain = AmmFilter{initial, makeModel(quiet), makeModel(agile)};
    auto floored = AmmFilter{initial, makeModel(quiet), makeModel(agile)};
    auto const floor = upkeep::floorModes(1e-3);

    auto const outlier = plain.update(makeFix(Eigen::Vector2d{30, 0}), reductions::gate(0.997));
    ASSERT_TRUE(outlier);
    ASSERT_FALSE(outlier->modes[0].accepted);
    ASSERT_TRUE(floored.update(makeFix(Eigen::Vector2d{30, 0}), reductions::gate(0.997)).value().accepted);
    EXPECT_EQ(plain.state().modes().toDensity().value().probability<Slow>(), 0.0);
    ASSERT_TRUE(floor(floored.state()));
    EXPECT_NEAR(floored.state().modes().toDensity().value().probability<Slow>(), 1e-3, 1e-15);

    for (int k = 1; k <= 10; ++k)
    {
        plain.predict(timeStep);
        floored.predict(timeStep);
        ASSERT_TRUE(plain.update(makeFix(Eigen::Vector2d::Zero()), reductions::gate(0.997)));
        ASSERT_TRUE(floored.update(makeFix(Eigen::Vector2d::Zero()), reductions::gate(0.997)));
        ASSERT_TRUE(floor(floored.state()));
    }
    EXPECT_EQ(plain.state().modes().toDensity().value().probability<Slow>(), 0.0);
    EXPECT_GT(floored.state().modes().toDensity().value().probability<Slow>(), 0.99);
    // The floored filter's estimate is back at the origin; the plain one is still pulled by the outlier.
    double const flooredDistance = floored.state().toDensity()->collapsed().mean().head<2>().norm();
    EXPECT_LT(flooredDistance, 1.0);
    EXPECT_GT(plain.state().toDensity()->collapsed().mean().head<2>().norm(), flooredDistance);
}

// The upkeep a filter holds ([upkeep]): each filter takes an optional upkeep last to its constructor (the AMM's
// before its state, its models being a pack), deduces its type, holds upkeep::None when given none, and runs it
// on its state by maintain(), which takes no arguments, as calling it on state() would. Headers under test:
// Filters/FilterUpkeep.h, the filters' constructors and deduction guides in Filters/, and Sequence and None in
// Distributions/Upkeep.h.

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

#include <cmath>
#include <expected>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>

using Emrys::AmmFilter;
using Emrys::CategoricalParams;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::Gaussian;
using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::HiddenMarkovFilter;
using Emrys::HybridGaussianParams;
using Emrys::ImmFilter;
using Emrys::InfoFilter;
using Emrys::KalmanFilter;
using Emrys::KalmanRecursion;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::MixtureFilter;
using Emrys::Named;
using Emrys::NumericError;
using Emrys::ParticleFilter;
using Emrys::ParticleSet;
using Emrys::PointMassFilter;
using Emrys::RbParticleFilter;
using Emrys::RbPointMassFilter;
using Emrys::SharedCovariance;
using Emrys::SwitchingModel;

namespace methods = Emrys::methods;
namespace upkeep = Emrys::upkeep;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

struct Slow
{};

struct Fast
{};

struct Regime : Named<Slow, Fast>
{};

using State = Composite<Position, Velocity>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

auto
makeModel(double density = 0.1)
{
    return ConstantVelocity<Position, Velocity>{density};
}

using Model = decltype(makeModel());

GaussianParams<State>
makeInitial()
{
    return {State{}, Vector4{0, 0, 1, 0.5}, Matrix4{Vector4{4, 4, 1, 1}.asDiagonal()}};
}

HybridGaussianParams<State, Regime>
makeHybrid(double slow)
{
    return {CategoricalParams<Regime>::fromWeights(Regime{}, Eigen::Array2d{slow, 1 - slow}), makeInitial()};
}

// A caller's action on any state, counting its calls; it changes nothing.
struct Count
{
    int * calls;

    template <typename State_>
    [[nodiscard]] std::expected<bool, NumericError>
    operator()(State_ &) const
    {
        ++*calls;
        return false;
    }
};

// Whether a filter of type F has a maintain() to call.
template <typename F>
concept Maintainable = requires(F & filter) { filter.maintain(); };

}  // namespace

// A filter given no upkeep holds upkeep::None, whose maintain() changes nothing and returns false.
TEST(FilterUpkeep, AFilterGivenNoneHoldsNone)
{
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    static_assert(std::is_same_v<decltype(filter)::UpkeepType, upkeep::None>);
    static_assert(std::is_same_v<decltype(filter), KalmanFilter<Gaussian<State>, Model>>);
    auto const result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
}

// maintain() requires the upkeep to apply to the filter's state, and each upkeep operation says which
// states it applies to (decisions.md [upkeep]): a Kalman filter holding a Resample, which its Gaussian can't do, or
// a Sequence holding one, has no maintain(), and one holding RepairCovariance, alone or in a Sequence, has.
TEST(FilterUpkeep, MaintainRequiresAnUpkeepOfTheState)
{
    using Rng = std::mt19937_64;
    using Resampling = decltype(KalmanFilter{makeModel(), makeInitial(), upkeep::resample(std::declval<Rng &>())});
    using Sequenced = decltype(KalmanFilter{
        makeModel(),
        makeInitial(),
        upkeep::sequence(upkeep::repairCovariance(), upkeep::resample(std::declval<Rng &>()))
    });
    using Repairing = decltype(KalmanFilter{makeModel(), makeInitial(), upkeep::repairCovariance()});
    using Repairs = decltype(KalmanFilter{makeModel(), makeInitial(), upkeep::sequence(upkeep::repairCovariance())});
    static_assert(!Maintainable<Resampling>);
    static_assert(!Maintainable<Sequenced>);
    static_assert(Maintainable<Repairing>);
    static_assert(Maintainable<Repairs>);
}

// A Kalman filter given RepairCovariance last deduces it, after a method or without one, and over a
// fixed-step model, and maintain() repairs a covariance that is not positive definite as the action on
// state() does.
TEST(FilterUpkeep, AKalmanFilterRepairsByItsUpkeep)
{
    auto filter = KalmanFilter{makeModel(), makeInitial(), upkeep::repairCovariance()};
    static_assert(std::is_same_v<
                  decltype(filter),
                  KalmanFilter<
                      Gaussian<State>,
                      Model,
                      methods::Linearized,
                      KalmanRecursion<Gaussian<State>, Model, methods::Linearized>,
                      upkeep::RepairCovariance<>>>);
    auto withMethod = KalmanFilter{makeModel(), makeInitial(), methods::unscented(), upkeep::repairCovariance()};
    static_assert(
        std::is_same_v<decltype(withMethod)::MethodType, std::remove_cvref_t<decltype(methods::unscented())>>
    );
    static_assert(std::is_same_v<decltype(withMethod)::UpkeepType, upkeep::RepairCovariance<>>);
    auto fixed = KalmanFilter{Emrys::fixedStep(makeModel(), 0.1), makeInitial(), upkeep::repairCovariance()};
    static_assert(std::is_same_v<decltype(fixed)::UpkeepType, upkeep::RepairCovariance<>>);

    auto result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);
    Matrix4 indefinite = makeInitial().covarianceLower();
    indefinite(3, 3) = -1e-9;
    filter.state().setCovariance(indefinite);
    ASSERT_EQ(filter.state().toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
    result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_TRUE(filter.state().toDensity().has_value());
}

// A particle filter's held trigger draws from the generator its actions hold, under the policy they were
// made with; maintain() takes no arguments.
TEST(FilterUpkeep, AParticleFilterResamplesByItsUpkeep)
{
    auto rng = std::mt19937_64{2};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State>{Emrys::drawSamples(initial.toDensity().value(), 100, rng)};
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng, 0.25));
    auto filter = ParticleFilter{makeModel(), particles, resampling};
    static_assert(std::is_same_v<decltype(filter)::UpkeepType, std::remove_const_t<decltype(resampling)>>);

    filter.state().setLogWeights(Eigen::ArrayXd::LinSpaced(100, -8.0, 0.0));
    auto result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_NEAR(filter.state().ess().value(), 100.0, 1e-9);
    result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_FALSE(*result);

    auto underPolicy = ParticleFilter{
        makeModel(),
        particles,
        upkeep::whenEss(
            Emrys::execution::sequential,
            std::less{},
            0.5,
            upkeep::resample(Emrys::execution::sequential, rng)
        )
    };
    underPolicy.state().setLogWeights(Eigen::ArrayXd::LinSpaced(100, -8.0, 0.0));
    result = underPolicy.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
}

// The upkeep of review §6 as one value: a point-mass filter given a Sequence prunes to 100 cells and then
// subdivides whenever fewer than 200 are left, halving the spacing at each call.
TEST(FilterUpkeep, APointMassFilterRunsASequence)
{
    using Whole = Composite<Position>;
    double const spacing = 5;
    Eigen::Matrix2d const box = Eigen::Matrix2d::Identity() * spacing * spacing / 12;
    auto const grid = Emrys::regularGrid<Whole, Position>(
        Eigen::Vector2d{-50, -50},
        Eigen::Vector2d{spacing, spacing},
        {21, 21},
        box
    );
    auto const steps =
        upkeep::sequence(upkeep::pruneToSize(100), upkeep::whenSize(std::less{}, 200, upkeep::subdivide(2)));
    auto filter = PointMassFilter{grid, steps};
    static_assert(std::is_same_v<
                  decltype(filter),
                  PointMassFilter<
                      std::remove_const_t<decltype(grid)>,
                      SharedCovariance,
                      std::remove_const_t<decltype(steps)>>>);
    ASSERT_EQ(filter.state().size(), 441);

    for (double const expected : {2.5, 1.25})
    {
        auto const result = filter.maintain();
        ASSERT_TRUE(result);
        EXPECT_TRUE(*result);
        EXPECT_EQ(filter.state().size(), 400);
        EXPECT_EQ(filter.state().grid().spacing()(0), expected);
    }
}

// A Rao-Blackwellized point-mass filter takes its upkeep after the conditional, after a covariance form, or
// after both a covariance form and a method.
TEST(FilterUpkeep, ARbPointMassFilterTakesItsUpkeepLast)
{
    using Whole = Composite<Position, Bias, Drift>;
    using Rest = Composite<Bias, Drift>;
    auto const grid = Emrys::regularGrid<Whole, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto const rest = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Matrix2d{{4, 0}, {0, 0.01}}};
    auto const bias = ConstantVelocity<Bias, Drift>{0.01};

    auto plain = RbPointMassFilter{bias, grid, rest, upkeep::pruneToSize(5)};
    static_assert(std::is_same_v<decltype(plain)::UpkeepType, upkeep::PruneToSize>);
    auto perCell = RbPointMassFilter{bias, grid, rest, Emrys::PerPointCovariance{}, upkeep::pruneToSize(5)};
    static_assert(std::is_same_v<decltype(perCell)::UpkeepType, upkeep::PruneToSize>);
    auto unscented =
        RbPointMassFilter{bias, grid, rest, Emrys::PerPointCovariance{}, methods::unscented(), upkeep::pruneToSize(5)};
    static_assert(std::is_same_v<decltype(unscented)::UpkeepType, upkeep::PruneToSize>);
    static_assert(std::is_same_v<decltype(unscented)::MethodType, std::remove_cvref_t<decltype(methods::unscented())>>);

    auto const result = plain.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(plain.state().size(), 5);
}

// A mixture filter, under either spelling, prunes by its upkeep.
TEST(FilterUpkeep, AMixtureFilterPrunesByItsUpkeep)
{
    auto mixture = GaussianMixtureParams<State>{State{}};
    mixture.add(std::log(0.8), Vector4{0, 0, 1, 0.5}, Matrix4::Identity());
    mixture.add(std::log(0.2), Vector4{5, 5, 1, 0.5}, Matrix4::Identity());

    auto filter = GaussianMixtureFilter{makeModel(), mixture, upkeep::pruneToSize(1)};
    static_assert(std::is_same_v<decltype(filter)::UpkeepType, upkeep::PruneToSize>);
    auto general = MixtureFilter{makeModel(), mixture, methods::unscented(), upkeep::pruneToSize(1)};
    static_assert(std::is_same_v<decltype(general)::UpkeepType, upkeep::PruneToSize>);

    auto result = filter.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(filter.state().size(), 1);
    result = general.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_EQ(general.state().size(), 1);
}

// The IMM takes its upkeep last; the AMM, whose models are a pack, before its state and after a method.
// maintain() floors a mode as FloorModes on state() does.
TEST(FilterUpkeep, MultipleModelFiltersFloorByTheirUpkeep)
{
    auto imm = ImmFilter{
        SwitchingModel{MarkovChain<Regime>{}, makeModel(0.01), makeModel(4.0)},
        makeHybrid(1.0),
        upkeep::floorModes(0.01)
    };
    static_assert(std::is_same_v<decltype(imm)::UpkeepType, upkeep::FloorModes<double>>);
    auto amm = AmmFilter{upkeep::floorModes(0.01), makeHybrid(1.0), makeModel(0.01), makeModel(4.0)};
    static_assert(std::is_same_v<decltype(amm)::UpkeepType, upkeep::FloorModes<double>>);
    auto unscented =
        AmmFilter{methods::unscented(), upkeep::floorModes(0.01), makeHybrid(1.0), makeModel(0.01), makeModel(4.0)};
    static_assert(std::is_same_v<decltype(unscented)::UpkeepType, upkeep::FloorModes<double>>);
    static_assert(std::is_same_v<decltype(unscented)::MethodType, std::remove_cvref_t<decltype(methods::unscented())>>);

    auto const fast = [](auto const & filter) {
        return filter.state().modes().toDensity().value().probability(1);
    };
    EXPECT_EQ(fast(imm), 0.0);
    EXPECT_EQ(fast(amm), 0.0);
    auto result = imm.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    result = amm.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_NEAR(fast(imm), 0.01, 1e-15);
    EXPECT_NEAR(fast(amm), 0.01, 1e-15);
}

// The hidden Markov filter and a mixture of IMM filters hold FloorModes too: maintain() floors the
// categorical and each hybrid component, as FloorModes on state() does.
TEST(FilterUpkeep, HiddenMarkovAndMixtureOfImmFiltersFloorByTheirUpkeep)
{
    auto hmm = HiddenMarkovFilter{
        MarkovChain<Regime>::persistent(1.0),
        CategoricalParams<Regime>::fromWeights(1.0, 0.0),
        upkeep::floorModes(0.01)
    };
    static_assert(std::is_same_v<decltype(hmm)::UpkeepType, upkeep::FloorModes<double>>);
    auto result = hmm.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    EXPECT_NEAR(hmm.state().toDensity().value().probability(1), 0.01, 1e-15);

    auto prior = Emrys::MixtureParams<Emrys::HybridGaussian<State, Regime>>{State{}};
    prior.add(std::log(0.5), makeHybrid(1.0));
    prior.add(std::log(0.5), makeHybrid(0.995));
    auto mixture = MixtureFilter{
        SwitchingModel{MarkovChain<Regime>{}, makeModel(0.01), makeModel(4.0)},
        prior,
        upkeep::floorModes(0.01)
    };
    static_assert(std::is_same_v<decltype(mixture)::UpkeepType, upkeep::FloorModes<double>>);
    result = mixture.maintain();
    ASSERT_TRUE(result);
    EXPECT_TRUE(*result);
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        EXPECT_NEAR(mixture.state()[i].component.modes().toDensity().value().probability(1), 0.01, 1e-15);
    }
}

// The Rao-Blackwellized particle filter, the hidden Markov filter, and the form filters take an upkeep last
// too; the hidden Markov filter and the information filter run a caller's own action.
TEST(FilterUpkeep, EveryFilterTakesAnUpkeep)
{
    auto rng = std::mt19937_64{3};
    auto rbpf = RbParticleFilter{
        makeModel(),
        Emrys::drawParticleGaussian<Position>(makeInitial(), 10, rng),
        upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))
    };
    auto resampled = rbpf.maintain();
    ASSERT_TRUE(resampled);
    EXPECT_FALSE(*resampled);  // equal weights

    int calls = 0;
    auto hmm = HiddenMarkovFilter{MarkovChain<Regime>::persistent(0.9), CategoricalParams<Regime>{}, Count{&calls}};
    static_assert(std::is_same_v<decltype(hmm)::UpkeepType, Count>);
    auto info = InfoFilter{makeModel(), Emrys::toInfo(makeInitial()).value(), Count{&calls}};
    static_assert(std::is_same_v<decltype(info)::UpkeepType, Count>);
    auto root = Emrys::SqrtKalmanFilter{makeModel(), Emrys::toSqrt(makeInitial()).value(), Count{&calls}};
    static_assert(std::is_same_v<decltype(root)::UpkeepType, Count>);
    EXPECT_TRUE(hmm.maintain().has_value());
    EXPECT_TRUE(info.maintain().has_value());
    EXPECT_TRUE(root.maintain().has_value());
    EXPECT_EQ(calls, 3);
}

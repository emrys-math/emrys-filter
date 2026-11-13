// MixtureMarginal and the mixture conditional (Mixture.h, Hybrid.h; steps 4 and 17l): a marginal
// of a mixture or a hybrid is a range over the components that gives each one's log-weight and its
// own marginal, borrowing one resolved selection and copying nothing [element-marginals], also for a mixture of
// hybrids; a conditional conditions each component and reweights by its likelihood so that the
// total weight is the marginal likelihood, and fails on a block a component knows exactly [innovation-cholesky];
// and the views check their parameters as Gaussian's views do [view-stamp].

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>

#include <gtest/gtest.h>

#include <cmath>
#include <expected>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::CategoricalParams;
using Emrys::Composite;
using Emrys::Gaussian;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::Linear;
using Emrys::MixtureParams;
using Emrys::Named;

namespace
{

struct Position : Linear<2>
{};

struct Speed : Linear<1>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Speed, Bias>;  // coordinates 0 and 1, 2, and 3

using Mixture4 = GaussianMixtureParams<State>;

struct Still
{};

struct Moving
{};

struct Mode : Named<Still, Moving>
{};

using HybridDensity = HybridGaussian<State, Mode>;
using HybridState = HybridGaussianParams<State, Mode>;

Eigen::Matrix4d
makeSpd(double phase)
{
    Eigen::Matrix4d const a = Eigen::Matrix4d::NullaryExpr([phase](Eigen::Index i, Eigen::Index j) {
        return std::sin(7.0 * double(i) + 3.0 * double(j) + phase);
    });
    return a * a.transpose() + Eigen::Matrix4d::Identity();
}

Eigen::Vector4d
makeMean(double phase)
{
    return Eigen::Vector4d::NullaryExpr([phase](Eigen::Index i) {
        return 2.0 * std::sin(1.3 * double(i) + phase);
    });
}

// Three components with unnormalized weights 0.5, 2, 1.
Mixture4
make()
{
    auto mixture = Mixture4{};
    mixture.add(std::log(0.5), makeMean(0.0), makeSpd(1.0));
    mixture.add(std::log(2.0), makeMean(1.0), makeSpd(2.0));
    mixture.add(std::log(1.0), makeMean(2.0), makeSpd(3.0));
    return mixture;
}

// Mode probabilities (pStill, 1 - pStill), with a Gaussian per mode whose moments start at phase.
HybridState
makeHybrid(double pStill, double phase)
{
    auto conditionals = HybridState::ConditionalsType{};
    conditionals.push_back(GaussianParams<State>{State{}, makeMean(phase), makeSpd(phase + 1.0)});
    conditionals.push_back(GaussianParams<State>{State{}, makeMean(phase + 1.0), makeSpd(phase + 2.0)});
    return HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{pStill, 1.0 - pStill}),
        std::move(conditionals)
    };
}

}  // namespace

// A tag marginal of a mixture is a random-access range with one element per source component,
// holding the component's log-weight and its own marginal, which reads the component in place; it
// has the block's compile-time dimension and copies to a GaussianMixtureParams over the block space
// [element-marginals].
TEST(GaussianMixtureMarginal, MarginalizesEveryComponent)
{
    auto const mixture = make();
    auto const marginal = mixture.marginal<Position>();
    using Marginal = std::remove_const_t<decltype(marginal)>;
    static_assert(Marginal::ValueDimensionAtCompileTime == 2);
    static_assert(std::ranges::random_access_range<Marginal const>);
    static_assert(std::ranges::sized_range<Marginal const>);
    ASSERT_EQ(marginal.size(), 3);
    EXPECT_EQ(marginal.valueDimension(), 2);

    Eigen::Index i = 0;
    for (auto const & [logWeight, component] : marginal)
    {
        auto const & source = mixture[i];
        auto const expected = source.component.marginal<Position>().moments();
        EXPECT_EQ(logWeight, source.logWeight);
        EXPECT_EQ(component.mean(), expected.mean());
        EXPECT_EQ(Eigen::Matrix2d{component.covarianceLower()}, expected.covarianceLower());
        // The component's marginal borrows the selection the mixture's resolved, rather than its own.
        EXPECT_EQ(&component.indices(), &marginal.indices());
        EXPECT_EQ(&component.space(), &marginal.space());
        ++i;
    }
    EXPECT_EQ(i, 3);
    EXPECT_EQ(marginal[2].component.mean(), mixture[2].component.mean().head<2>());

    auto copy = marginal.copy();
    static_assert(std::is_same_v<decltype(copy), GaussianMixtureParams<Position>>);
    EXPECT_EQ(copy.size(), 3);
    EXPECT_TRUE(copy.toDensity().has_value());
}

// Changing a component's mean or scaling the weights while a marginal is alive is allowed, and
// the marginal sees the new values.
TEST(GaussianMixtureMarginal, ValuesMayChangeUnderAMarginal)
{
    auto mixture = make();
    auto const marginal = mixture.marginal<Speed>();
    mixture.setMean(0, Eigen::Vector4d::Zero());
    mixture.scaleWeights(1.0);
    EXPECT_EQ(marginal[0].component.mean()(0), 0.0);
    EXPECT_DOUBLE_EQ(marginal[0].logWeight, std::log(0.5) + 1.0);
}

// The marginal of a mixture's parameters and that of its factorized view are the same range: the
// same log-weights and component moments, and so are those of one repetition of a repeated block,
// which have the repetition's difference dimension. A hybrid gives the marginal of one
// repetition too, by mode, also when its components come from an allocator of their own.
TEST(GaussianMixtureMarginal, DensityMarginalMatchesParams)
{
    auto const mixture = make();
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density.has_value());
    auto const fromParams = mixture.marginal<Speed>();
    auto const fromDensity = density->marginal<Speed>();
    ASSERT_EQ(fromParams.size(), fromDensity.size());
    for (Eigen::Index i = 0; i < fromParams.size(); ++i)
    {
        EXPECT_EQ(fromParams[i].logWeight, fromDensity[i].logWeight);
        EXPECT_EQ(fromParams[i].component.moments().mean(), fromDensity[i].component.moments().mean());
    }

    // So is a marginal of one repetition of a repeated block.
    struct Fleet : Emrys::Repeated<Speed, 4>
    {};

    using Convoy = Emrys::Composite<Fleet>;
    auto convoy = GaussianMixtureParams<Convoy>{Convoy{}, 4};
    convoy.add(0.0, makeMean(0.0), makeSpd(1.0));
    convoy.add(std::log(2.0), makeMean(1.0), makeSpd(2.0));
    auto const convoyDensity = convoy.toDensity();
    ASSERT_TRUE(convoyDensity.has_value());
    auto const second = convoyDensity->marginal<Fleet>(1);
    auto const secondOfParams = convoy.marginal<Fleet>(1);
    ASSERT_EQ(second.size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        EXPECT_EQ(second[i].component.moments().mean(), secondOfParams[i].component.moments().mean());
    }
    EXPECT_EQ(second.differenceDimension(), 1);
    EXPECT_EQ(Eigen::Index(second.differenceIndices().size()), 1);

    using Pmr = std::pmr::polymorphic_allocator<>;
    auto resource = std::pmr::monotonic_buffer_resource{};
    auto const conditionals = std::vector{
        GaussianParams<Convoy>{Convoy{}, makeMean(0.0), makeSpd(1.0)},
        GaussianParams<Convoy>{Convoy{}, makeMean(1.0), makeSpd(2.0)}
    };
    auto const hybrid =
        HybridGaussianParams<Convoy, Mode, double, Pmr>{CategoricalParams<Mode>{}, conditionals, Pmr{&resource}};
    EXPECT_EQ(hybrid.get_allocator(), Pmr{&resource});
    auto const third = hybrid.marginal<Fleet>(2);
    ASSERT_EQ(third.size(), 2);
    EXPECT_EQ(third[1].component.mean()(0), conditionals[1].mean()(2));
}

// A hybrid's marginal ranges over its modes: each element holds the mode's log-probability and the
// marginal of its component, read in place, and copy() gives a hybrid of the marginals on the same
// modes.
TEST(HybridMarginal, MarginalizesEveryMode)
{
    auto const hybrid = makeHybrid(0.3, 0.5);
    auto const marginal = hybrid.marginal<Position>();
    static_assert(std::ranges::random_access_range<std::remove_const_t<decltype(marginal)> const>);
    ASSERT_EQ(marginal.size(), 2);
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        auto const element = marginal[m];
        EXPECT_EQ(element.logWeight, hybrid.modes().logWeight(m));
        EXPECT_EQ(element.component.mean(), hybrid.conditional(m).mean().head<2>());
        Eigen::Matrix2d const expected = hybrid.conditional(m).covarianceLower().topLeftCorner<2, 2>();
        EXPECT_EQ(Eigen::Matrix2d{element.component.covarianceLower()}, expected);
    }

    auto const copy = marginal.copy();
    static_assert(std::is_same_v<decltype(copy), HybridGaussianParams<Position, Mode> const>);
    EXPECT_TRUE(copy.modes().toDensity().value().logProbabilities().isApprox(
        hybrid.modes().toDensity().value().logProbabilities()
    ));
    EXPECT_EQ(copy.conditional(1).mean(), hybrid.conditional(1).mean().head<2>());
}

// A mixture of hybrids, which had no marginal while it took only Gaussian components, gives for
// each hybrid a range over its modes, all three levels borrowing one selection; copy() gives a
// mixture of hybrids of the marginals.
TEST(HybridMarginal, MixtureOfHybridsGivesARangeOfRanges)
{
    auto mixture = MixtureParams<HybridDensity>{State{}};
    mixture.add(std::log(0.25), makeHybrid(0.3, 0.0));
    mixture.add(std::log(0.75), makeHybrid(0.6, 1.0));
    auto const marginal = mixture.marginal<Speed>();
    ASSERT_EQ(marginal.size(), 2);
    Eigen::Index c = 0;
    for (auto const & [logWeight, hybrid] : marginal)
    {
        auto const & source = mixture[c].component;
        EXPECT_EQ(logWeight, mixture[c].logWeight);
        ASSERT_EQ(hybrid.size(), 2);
        for (auto const & [logProbability, speed] : hybrid)
        {
            EXPECT_EQ(&speed.indices(), &marginal.indices());
            static_cast<void>(logProbability);
        }
        EXPECT_EQ(hybrid[1].logWeight, source.modes().logWeight(1));
        EXPECT_EQ(hybrid[1].component.mean()(0), source.conditional(1).mean()(2));
        ++c;
    }

    auto const copy = marginal.copy();
    static_assert(std::is_same_v<decltype(copy), MixtureParams<Hybrid<Gaussian<Speed>, Mode>> const>);
    ASSERT_EQ(copy.size(), 2);
    EXPECT_EQ(copy[1].component.conditional(0).mean()(0), mixture[1].component.conditional(0).mean()(2));
}

// conditional<Path>(v) conditions each component as a Gaussian would and adds the log-density of
// its marginal at v to its log-weight; the result is over the remaining blocks.
TEST(GaussianMixtureConditional, ConditionsEachComponentAndReweights)
{
    auto const mixture = make();
    auto const v = Eigen::Vector2d{0.4, -1.1};
    auto const conditional = mixture.conditional<Position>(v);
    static_assert(std::is_same_v<
                  decltype(conditional),
                  std::expected<GaussianMixtureParams<Composite<Speed, Bias>>, Emrys::GaussianError> const>);
    ASSERT_TRUE(conditional);
    ASSERT_EQ(conditional->size(), 3);

    for (Eigen::Index i = 0; i < conditional->size(); ++i)
    {
        auto const & source = mixture[i];
        auto const & result = (*conditional)[i];
        auto const expected = source.component.conditional<Position>(v);
        auto const marginal = source.component.marginal<Position>().copy();
        double const logLikelihood = marginal.toDensity()->logDensity(v);
        EXPECT_LT((result.component.mean() - expected.mean()).norm(), 1e-12);
        EXPECT_LT((result.component.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);
        EXPECT_NEAR(result.logWeight, source.logWeight + logLikelihood, 1e-12);
    }
}

// The conditional's total weight is the source's total weight times the mixture marginal's density
// at the given value: the marginal likelihood of v.
TEST(GaussianMixtureConditional, TotalWeightIsMarginalLikelihood)
{
    auto const mixture = make();
    auto const v = Eigen::Vector2d{0.4, -1.1};
    auto const conditional = mixture.conditional<Position>(v);

    auto const marginalAtGiven = mixture.marginal<Position>().copy();
    double const expected = mixture.logTotalWeight().value() + marginalAtGiven.toDensity()->logDensity(v);
    EXPECT_NEAR(conditional.value().logTotalWeight().value(), expected, 1e-12);
}

// Conditioning through the parameters and through its factorized view give the same components.
TEST(GaussianMixtureConditional, DensityMatchesParams)
{
    auto const mixture = make();
    auto const v = Eigen::Matrix<double, 1, 1>{0.2};
    auto const fromMixture = mixture.conditional<Speed>(v).value();
    auto const fromDensity = mixture.toDensity()->conditional<Speed>(v).value();
    ASSERT_EQ(fromMixture.size(), fromDensity.size());
    for (Eigen::Index i = 0; i < fromMixture.size(); ++i)
    {
        EXPECT_EQ(fromMixture[i].logWeight, fromDensity[i].logWeight);
        EXPECT_EQ(fromMixture[i].component.mean(), fromDensity[i].component.mean());
    }
}

// A component that knows the given block exactly has no density there, so conditioning the mixture
// on it fails rather than give that component all the weight wherever the value lies [innovation-cholesky].
TEST(GaussianMixtureConditional, RefusesABlockKnownExactly)
{
    auto mixture = make();
    Eigen::Matrix4d exact = makeSpd(0.5);
    exact.row(0).setZero();
    exact.col(0).setZero();
    mixture.add(0.0, makeMean(0.5), exact);
    auto const conditional = mixture.conditional<Position>(Eigen::Vector2d{50.0, -1.1});
    ASSERT_FALSE(conditional);
    EXPECT_EQ(conditional.error(), Emrys::GaussianError::NotPositiveDefinite);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// Reading a marginal after its mixture gained, lost, or was moved from components dies [view-stamp].
TEST(GaussianMixtureMarginalDeathTest, ReadAfterStructuralChange)
{
    auto mixture = make();
    auto const marginal = mixture.marginal<Position>();
    std::string const message = "changed size or structure while a marginal";
    EXPECT_DEATH(
        {
            mixture.add(0.0, makeMean(3.0), makeSpd(4.0));
            std::ignore = marginal[0];
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.remove(0);
            std::ignore = marginal[0];
        },
        message
    );
    EXPECT_DEATH(
        {
            (void)mixture.pruneToSize(1);
            std::ignore = marginal[0];
        },
        message
    );
    EXPECT_DEATH(
        {
            mixture.clear();
            std::ignore = marginal.size();
        },
        message
    );
    EXPECT_DEATH(
        {
            auto stolen = std::move(mixture);
            std::ignore = marginal[0];
        },
        message
    );
}

// A marginal taken from the factorized view holds a copy of the view after the view itself is
// gone, so reading it after a value change dies as the view would.
TEST(GaussianMixtureMarginalDeathTest, MarginalOfViewChecks)
{
    auto mixture = make();
    auto marginal = std::optional<decltype(mixture.toDensity()->marginal<Speed>())>{};
    marginal.emplace(mixture.toDensity()->marginal<Speed>());
    mixture.scaleWeights(1.0);
    EXPECT_DEATH(std::ignore = (*marginal)[0], "changed while a view");
}

// A hybrid keeps a stamp too: reading its marginal after its components were swapped out dies,
// while a change of its mode probabilities does not [view-stamp].
TEST(HybridMarginalDeathTest, ReadAfterStructuralChange)
{
    auto hybrid = makeHybrid(0.3, 0.5);
    auto const marginal = hybrid.marginal<Position>();
    hybrid.setModes(CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.5, 0.5}));
    EXPECT_DOUBLE_EQ(marginal[0].logWeight, std::log(0.5));
    EXPECT_DEATH(
        {
            auto replacement = makeHybrid(0.3, 0.0).toMixture();
            auto conditionals = HybridState::ConditionalsType{};
            conditionals.push_back(replacement.begin()->component);
            conditionals.push_back(replacement[1].component);
            hybrid.swapConditionals(conditionals);
            std::ignore = marginal[0];
        },
        "changed size or structure while a marginal"
    );
}

#endif

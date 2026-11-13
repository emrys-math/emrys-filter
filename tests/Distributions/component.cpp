// The Component concept and ComponentTraits (step 15c; [component-first]): Gaussian and Hybrid satisfy
// it and a Mixture does not; a hybrid's log-density, read through its Hybrid, equals its
// toMixture()'s; the collapse of hybrids sums the mode probabilities and moment-matches each mode,
// which a Gaussian mixture built by hand confirms, as does merging two entries of a mixture; and a
// mixture of hybrids has the density and moments of the Gaussian mixture it flattens to. The square-root form is a
// component too, whose collapse, density, merge, and marginals are the covariance form's, and the information forms
// are not.
// Tolerances are rounding on sums of a few terms.
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::CategoricalParams;
using Emrys::Component;
using Emrys::Composite;
using Emrys::Gaussian;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianParams;
using Emrys::HybridParams;
using Emrys::InfoGaussian;
using Emrys::Linear;
using Emrys::MixtureParams;
using Emrys::Named;
using Emrys::RadiansPi;
using Emrys::SqrtGaussian;
using Emrys::SqrtGaussianParams;
using Emrys::toSqrt;

namespace
{

struct Still
{};

struct Moving
{};

struct Mode : Named<Still, Moving>
{};

using Space = Linear<2>;
using HybridDensity = HybridGaussian<Space, Mode>;
using HybridState = HybridGaussianParams<Space, Mode>;
using Traits = Emrys::ComponentTraits<HybridState>;

GaussianParams<Space>
makeGaussian(double x, double y, double spread)
{
    return GaussianParams<Space>{Eigen::Vector2d{x, y}, Eigen::Matrix2d{{spread, 0.1}, {0.1, spread + 0.5}}};
}

// Mode probabilities (pStill, 1 - pStill), with a Gaussian per mode.
HybridState
makeHybrid(double pStill, GaussianParams<Space> const & still, GaussianParams<Space> const & moving)
{
    return HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{pStill, 1.0 - pStill}),
        std::vector<GaussianParams<Space>>{still, moving}
    };
}

// The hybrids a mixture of two holds, with their log-weights.
struct Pair
{
    HybridState first = makeHybrid(0.3, makeGaussian(0.0, 0.0, 1.0), makeGaussian(2.0, 1.0, 2.0));
    HybridState second = makeHybrid(0.8, makeGaussian(-1.0, 0.5, 0.5), makeGaussian(3.0, -1.0, 1.5));
    double logFirst = std::log(0.4);
    double logSecond = std::log(0.6);

    // The same density as a mixture of Gaussians, one per hybrid and mode, weighted w_c P_c(m).
    [[nodiscard]] GaussianMixtureParams<Space>
    flattened() const
    {
        auto mixture = GaussianMixtureParams<Space>{Space{}};
        for (auto const & [logWeight, hybrid] : {std::pair{logFirst, &first}, std::pair{logSecond, &second}})
        {
            for (Eigen::Index m = 0; m < hybrid->size(); ++m)
            {
                mixture.add(
                    logWeight + hybrid->modes().toDensity().value().logProbability(m),
                    hybrid->conditional(m).moments()
                );
            }
        }
        return mixture;
    }
};

std::vector<Eigen::Vector2d> const points{{0.0, 0.0}, {1.5, -0.5}, {-2.0, 3.0}, {4.0, 4.0}};

}  // namespace

// What is and is not a component: a Gaussian, a hybrid of them, and a hybrid of hybrids are; a
// mixture holds components and is not one; and a component is named by the type that evaluates,
// so GaussianParams is not one [params-and-density].
TEST(Component, GaussiansAndHybridsSatisfyTheConcept)
{
    static_assert(Component<Gaussian<Space>>);
    static_assert(Component<Gaussian<Space, float>>);
    static_assert(Component<HybridDensity>);
    static_assert(Component<Hybrid<HybridDensity, Mode>>);
    static_assert(!Component<HybridState>);
    static_assert(std::is_same_v<Emrys::Params<HybridDensity>, HybridState>);
    static_assert(std::is_same_v<Emrys::Density<HybridState>, HybridDensity>);
    static_assert(!Component<GaussianMixtureParams<Space>>);
    static_assert(!Component<MixtureParams<HybridDensity>>);
    static_assert(!Component<GaussianParams<Space>>);
    static_assert(std::is_same_v<MixtureParams<Gaussian<Space>>::ComponentType, GaussianParams<Space>>);
    static_assert(Emrys::isGaussian<GaussianParams<Space>>);
    static_assert(!Emrys::isGaussian<HybridState>);

    // A hybrid deduces its component from the one given for every mode; MSVC does not deduce
    // through the HybridGaussianParams alias, so examples deduce through HybridParams (decisions.md
    // [filters-and-recursions]).
    auto const deduced = HybridParams{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.5, 0.5}),
        makeGaussian(0.0, 0.0, 1.0)
    };
    static_assert(std::is_same_v<decltype(deduced), HybridState const>);
    EXPECT_EQ(deduced.size(), 2);
}

// A hybrid's log-density, read from its factorized conditionals without a check, is the
// log-sum-exp over its modes: what its view as a mixture evaluates.
TEST(Component, AHybridsDensityIsItsMixtures)
{
    auto const pair = Pair{};
    ASSERT_TRUE(pair.first.factorize());
    auto const mixture = pair.first.toMixture();
    auto const asMixture = mixture.toDensity();
    ASSERT_TRUE(asMixture);
    for (auto const & x : points)
    {
        EXPECT_NEAR(Traits::logDensity(pair.first, x), asMixture->logDensity(x), 1e-12);
    }
}

// A hybrid's density is its checked view (decisions.md [params-and-density]): toDensity() fails as
// factorize() does, and the view evaluates the log-density the traits read and gives the moments
// and collapse of its mixture, which a density of the same components weighted by the modes gives.
TEST(Component, AHybridsDensityViewIsChecked)
{
    auto const pair = Pair{};
    auto const density = pair.first.toDensity();
    ASSERT_TRUE(density);
    static_assert(std::is_same_v<decltype(density)::value_type, HybridDensity>);
    auto const mixture = pair.first.toMixture();
    auto const asMixture = mixture.toDensity();
    ASSERT_TRUE(asMixture);
    for (auto const & x : points)
    {
        EXPECT_EQ(density->logDensity(x), Traits::logDensity(pair.first, x));
        EXPECT_NEAR(density->density(x), std::exp(asMixture->logDensity(x)), 1e-15);
    }
    auto const moments = density->moments();
    EXPECT_LT((moments.mean() - asMixture->moments().mean()).norm(), 1e-12);
    EXPECT_LT((moments.covarianceLower() - asMixture->moments().covarianceLower()).norm(), 1e-12);
    EXPECT_EQ(density->collapsed().mean(), moments.mean());
    EXPECT_EQ(density->size(), 2);
    EXPECT_EQ(&density->conditional(1), &pair.first.conditional(1));

    auto const zero = HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.0, 0.0}),
        std::vector<GaussianParams<Space>>{makeGaussian(0.0, 0.0, 1.0), makeGaussian(1.0, 0.0, 1.0)}
    };
    EXPECT_EQ(zero.toDensity().error(), zero.factorize().error());
    auto const broken = makeHybrid(
        0.3,
        makeGaussian(0.0, 0.0, 1.0),
        GaussianParams<Space>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}}}
    );
    EXPECT_EQ(broken.toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
    auto repaired = broken;
    EXPECT_TRUE(repaired.toDensity(Emrys::roots::ridge()));
}

// A hybrid's mode weights need not sum to one, as a mixture's need not: weights of 3 and 7 give
// the density of probabilities 0.3 and 0.7. Weights that fail to define a distribution are
// reported by factorize(), with a ridge repair too, and by a mixture's toDensity() as the hybrid's
// ComponentError. The repair factorizes a conditional whose covariance is singular.
TEST(Component, AHybridsModeWeightsAreCheckedByFactorize)
{
    auto const still = makeGaussian(0.0, 0.0, 1.0);
    auto const moving = makeGaussian(2.0, 1.0, 2.0);
    auto const normalized = makeHybrid(0.3, still, moving);
    auto const weighted = HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{3, 7}),
        std::vector<GaussianParams<Space>>{still, moving}
    };
    ASSERT_TRUE(normalized.factorize());
    ASSERT_TRUE(weighted.factorize());
    for (auto const & x : points)
    {
        EXPECT_NEAR(Traits::logDensity(weighted, x), Traits::logDensity(normalized, x), 1e-12);
    }

    auto const zero = HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0, 0}),
        std::vector<GaussianParams<Space>>{still, moving}
    };
    EXPECT_EQ(zero.factorize().error(), Emrys::WeightError::ZeroSum);
    auto repaired = zero;
    EXPECT_EQ(repaired.factorize(Emrys::roots::ridge()).error(), Emrys::WeightError::ZeroSum);
    auto singular = HybridState{
        CategoricalParams<Mode>{},
        std::vector<GaussianParams<Space>>{
            still,
            GaussianParams<Space>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Ones()}
        }
    };
    EXPECT_FALSE(singular.factorize());
    EXPECT_TRUE(singular.factorize(Emrys::roots::ridge()));
    // A zero covariance has no ridge to add.
    auto flat = HybridState{
        CategoricalParams<Mode>{},
        std::vector<GaussianParams<Space>>{
            still,
            GaussianParams<Space>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()}
        }
    };
    EXPECT_EQ(flat.factorize(Emrys::roots::ridge()).error(), Emrys::GaussianError::NotPositiveDefinite);
    auto mixture = MixtureParams<HybridDensity>{Space{}};
    mixture.add(0.0, normalized);
    mixture.add(0.0, zero);
    EXPECT_EQ(
        mixture.toDensity().error(),
        Emrys::MixtureError{(Emrys::ComponentError{1, Emrys::WeightError::ZeroSum})}
    );
}

// Collapsing two hybrids gives the mode probabilities 0.4 P_1(m) + 0.6 P_2(m) and, in each mode,
// the moment match of the two conditionals with the weights 0.4 P_1(m) and 0.6 P_2(m), which a
// two-component Gaussian mixture computes.
TEST(Component, CollapsingHybridsSumsModesAndMatchesMomentsPerMode)
{
    auto const pair = Pair{};
    HybridState const * hybrids[] = {&pair.first, &pair.second};
    double const logWeights[] = {pair.logFirst, pair.logSecond};
    HybridState const collapsed = Traits::collapse(
        2,
        [&](Eigen::Index c) -> HybridState const & {
            HybridState const * hybrid = hybrids[c];
            return *hybrid;
        },
        [&](Eigen::Index c) {
            return logWeights[c];
        }
    );

    ASSERT_EQ(collapsed.size(), 2);
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        double const expectedProbability = 0.4 * pair.first.modes().toDensity().value().probability(m)
                                         + 0.6 * pair.second.modes().toDensity().value().probability(m);
        EXPECT_NEAR(collapsed.modes().toDensity().value().probability(m), expectedProbability, 1e-12);

        auto byHand = GaussianMixtureParams<Space>{Space{}};
        byHand.add(
            pair.logFirst + pair.first.modes().toDensity().value().logProbability(m),
            pair.first.conditional(m).moments()
        );
        byHand.add(
            pair.logSecond + pair.second.modes().toDensity().value().logProbability(m),
            pair.second.conditional(m).moments()
        );
        auto const expected = byHand.toDensity();
        ASSERT_TRUE(expected);
        auto const moments = expected->moments();
        EXPECT_LT((collapsed.conditional(m).mean() - moments.mean()).norm(), 1e-12) << "mode " << m;
        EXPECT_LT((collapsed.conditional(m).covarianceLower() - moments.covarianceLower()).norm(), 1e-12)
            << "mode " << m;
    }
}

// A mixture of hybrids is a mixture: its density and moments are those of the Gaussian mixture
// with one component per hybrid and mode, and its collapse is the collapse of the hybrids.
TEST(Component, AMixtureOfHybridsHasTheFlattenedDensity)
{
    auto const pair = Pair{};
    auto mixture = MixtureParams<HybridDensity>{Space{}};
    mixture.add(pair.logFirst, pair.first);
    mixture.add(pair.logSecond, pair.second);
    EXPECT_EQ(mixture.size(), 2);
    EXPECT_EQ(mixture.valueDimension(), 2);

    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    auto const byHand = pair.flattened();
    auto const flattened = byHand.toDensity();
    ASSERT_TRUE(flattened);
    for (auto const & x : points)
    {
        EXPECT_NEAR(density->logDensity(x), flattened->logDensity(x), 1e-12);
    }
    auto const moments = density->moments();
    auto const expected = flattened->moments();
    EXPECT_LT((moments.mean() - expected.mean()).norm(), 1e-12);
    EXPECT_LT((moments.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);

    HybridState const collapsed = density->collapsed();
    EXPECT_NEAR(collapsed.modes().toDensity().value().probability(0), 0.4 * 0.3 + 0.6 * 0.8, 1e-12);
    EXPECT_LT((collapsed.toDensity()->moments().mean() - expected.mean()).norm(), 1e-12);
}

// merge(keep, absorb) collapses the pair for any component: the merged hybrid has the mode
// probabilities 0.4 P_1(m) + 0.6 P_2(m) and, in each mode, the moments of the two conditionals
// weighted by 0.4 P_1(m) and 0.6 P_2(m), as the flattened Gaussian mixture's per-mode pairs give
// them.
TEST(Component, MergingHybridsKeepsModesAndMomentsPerMode)
{
    auto const pair = Pair{};
    auto mixture = MixtureParams<HybridDensity>{Space{}};
    mixture.add(pair.logFirst, pair.first);
    mixture.add(pair.logSecond, pair.second);

    mixture.merge(0, 1);
    ASSERT_EQ(mixture.size(), 1);
    EXPECT_NEAR(mixture.begin()->logWeight, 0.0, 1e-12);

    HybridState const & merged = mixture.begin()->component;
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        double const expectedProbability = 0.4 * pair.first.modes().toDensity().value().probability(m)
                                         + 0.6 * pair.second.modes().toDensity().value().probability(m);
        EXPECT_NEAR(merged.modes().toDensity().value().probability(m), expectedProbability, 1e-12);

        auto byHand = GaussianMixtureParams<Space>{Space{}};
        byHand.add(
            pair.logFirst + pair.first.modes().toDensity().value().logProbability(m),
            pair.first.conditional(m).moments()
        );
        byHand.add(
            pair.logSecond + pair.second.modes().toDensity().value().logProbability(m),
            pair.second.conditional(m).moments()
        );
        auto const expected = byHand.toDensity();
        ASSERT_TRUE(expected);
        auto const moments = expected->moments();
        EXPECT_LT((merged.conditional(m).mean() - moments.mean()).norm(), 1e-12) << "mode " << m;
        EXPECT_LT((merged.conditional(m).covarianceLower() - moments.covarianceLower()).norm(), 1e-12) << "mode " << m;
    }
}

// A mode of weight zero takes no part: the density is the other mode's, and collapsing hybrids
// that all give a mode zero probability keeps the first one's component for it, unweighted.
TEST(Component, AModeOfWeightZeroTakesNoPart)
{
    auto const still = makeGaussian(0.0, 0.0, 1.0);
    auto const moving = makeGaussian(2.0, 1.0, 2.0);
    auto const neverStill = makeHybrid(0.0, still, moving);
    ASSERT_TRUE(neverStill.factorize());
    auto const movingDensity = moving.toDensity();
    ASSERT_TRUE(movingDensity);
    for (auto const & x : points)
    {
        EXPECT_NEAR(Traits::logDensity(neverStill, x), movingDensity->logDensity(x), 1e-12);
    }

    auto const alsoNeverStill = makeHybrid(0.0, makeGaussian(5.0, 5.0, 1.0), makeGaussian(-2.0, 0.0, 1.0));
    HybridState const * hybrids[] = {&neverStill, &alsoNeverStill};
    HybridState const collapsed = Traits::collapse(
        2,
        [&](Eigen::Index c) -> HybridState const & {
            return *hybrids[c];
        },
        [](Eigen::Index) {
            return std::log(0.5);
        }
    );
    EXPECT_EQ(collapsed.modes().toDensity().value().probability(0), 0.0);
    EXPECT_EQ(collapsed.conditional(0).mean(), still.mean());
}

// factorize() reports a component that fails, after the mode weights.
TEST(Component, AFailingComponentIsReported)
{
    auto const singular = GaussianParams<Space>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()};
    auto const broken = makeHybrid(0.3, makeGaussian(0.0, 0.0, 1.0), singular);
    EXPECT_EQ(broken.factorize().error(), Emrys::GaussianError::NotPositiveDefinite);
}

// A hybrid whose mode weights fail to define a distribution prints each mode's log-weight in
// place of its probability.
TEST(Component, AHybridWithInvalidWeightsPrintsLogWeights)
{
    auto const zero = HybridState{
        CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0, 0}),
        std::vector<GaussianParams<Space>>{makeGaussian(0.0, 0.0, 1.0), makeGaussian(2.0, 1.0, 2.0)}
    };
    auto stream = std::ostringstream{};
    stream << zero;
    EXPECT_NE(stream.str().find("every weight is zero"), std::string::npos) << stream.str();
    EXPECT_NE(stream.str().find("mode 0, log-weight -inf:"), std::string::npos) << stream.str();
}

// A hybrid hands out its components read-only: one is replaced by setConditional() or changed in
// place by updateConditional(), which returns what its function does, and either keeps the
// hybrid's space, which a component on another fails. A copy assigned over another hybrid takes
// its modes and components.
TEST(Component, AHybridsComponentsChangeOnlyInItsSpace)
{
    static_assert(
        std::is_same_v<decltype(std::declval<HybridState &>().conditional(0)), GaussianParams<Space> const &>
    );
    auto hybrid = makeHybrid(0.3, makeGaussian(0.0, 0.0, 1.0), makeGaussian(2.0, 1.0, 2.0));
    hybrid.setConditional(1, makeGaussian(5.0, 5.0, 1.0));
    EXPECT_EQ(hybrid.conditional(1).mean(), Eigen::Vector2d(5.0, 5.0));
    double const x = hybrid.updateConditional(0, [](GaussianParams<Space> & g) {
        g.setMean(Eigen::Vector2d{-1.0, 4.0});
        return g.mean().x();
    });
    EXPECT_EQ(x, -1.0);
    EXPECT_EQ(hybrid.conditional(0).mean(), Eigen::Vector2d(-1.0, 4.0));
    auto assigned = makeHybrid(0.5, makeGaussian(0.0, 0.0, 1.0), makeGaussian(0.0, 0.0, 1.0));
    assigned = hybrid;
    EXPECT_EQ(assigned.conditional(0).mean(), Eigen::Vector2d(-1.0, 4.0));
    EXPECT_EQ(assigned.conditional(1).mean(), Eigen::Vector2d(5.0, 5.0));

    auto mixture = GaussianMixtureParams<Space>{Space{}};
    mixture.add(0.0, makeGaussian(0.0, 0.0, 1.0).moments());
    mixture.forEach([](auto & entry) {
        entry.setComponent(makeGaussian(1.0, 2.0, 1.0));
        entry.updateComponent([](GaussianParams<Space> & g) {
            g.setMean(g.mean() * 2.0);
        });
    });
    EXPECT_EQ(mixture.begin()->component.mean(), Eigen::Vector2d(2.0, 4.0));
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
// A component on another space, set or left by an update, fails the hybrid's and the mixture's
// assertion.
TEST(ComponentDeathTest, AComponentOnAnotherSpaceFails)
{
    using Resizable = Linear<Eigen::Dynamic>;
    using G = GaussianParams<Resizable>;
    auto const two = G{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    auto const three = G{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    auto hybrid = HybridGaussianParams<Resizable, Mode>{CategoricalParams<Mode>{}, two};
    EXPECT_DEATH(hybrid.setConditional(0, three), "the component must be in the hybrid's space");
    EXPECT_DEATH(
        hybrid.updateConditional(
            0,
            [&](G & g) {
                g = three;
            }
        ),
        "the component must keep its space"
    );
    auto mixture = GaussianMixtureParams<Resizable>{Resizable{2}};
    mixture.add(0.0, two.moments());
    EXPECT_DEATH(
        mixture.forEach([&](auto & entry) {
            entry.setComponent(three);
        }),
        "the component must be in"
    );
}

// Reading a hybrid's density after any change to its parameters dies [view-stamp].
TEST(ComponentDeathTest, AHybridsDensityReadAfterChangeFails)
{
    auto hybrid = makeHybrid(0.3, makeGaussian(0.0, 0.0, 1.0), makeGaussian(2.0, 1.0, 2.0));
    auto const density = hybrid.toDensity();
    ASSERT_TRUE(density);
    EXPECT_DEATH(
        {
            hybrid.setConditional(1, makeGaussian(5.0, 5.0, 1.0));
            std::ignore = density->moments();
        },
        "changed while a view"
    );
    EXPECT_DEATH(
        {
            hybrid.setModes(CategoricalParams<Mode>{});
            std::ignore = density->logDensity(points.front());
        },
        "changed while a view"
    );
}
#endif

namespace
{

struct Along : Linear<1>
{};

struct Across : Linear<1>
{};

struct Heading : RadiansPi<1>
{};

using Plane = Composite<Along, Across>;
using SqrtTraits = Emrys::ComponentTraits<SqrtGaussianParams<Plane>>;

// Three components of a mixture in the plane, with their log-weights, in the covariance form.
std::vector<std::pair<double, GaussianParams<Plane>>>
planeComponents()
{
    return {
        {std::log(0.5), GaussianParams<Plane>{Eigen::Vector2d{0.0, 1.0}, Eigen::Matrix2d{{1.0, 0.3}, {0.3, 0.8}}}},
        {std::log(0.3), GaussianParams<Plane>{Eigen::Vector2d{2.0, -1.0}, Eigen::Matrix2d{{0.5, -0.1}, {-0.1, 2.0}}}},
        {std::log(0.2), GaussianParams<Plane>{Eigen::Vector2d{-1.0, 0.5}, Eigen::Matrix2d{{3.0, 0.0}, {0.0, 0.2}}}},
    };
}

Eigen::Matrix2d
covarianceOf(SqrtGaussianParams<Plane> const & gaussian)
{
    return gaussian.covariance().value();
}

}  // namespace

// The square-root form is a component and the information forms are not: they have no collapse,
// and a container given one says to convert it.
TEST(Component, TheSqrtFormIsAComponentAndTheInfoFormsAreNot)
{
    static_assert(Component<SqrtGaussian<Plane>>);
    static_assert(Component<Hybrid<SqrtGaussian<Plane>, Mode>>);
    static_assert(!Component<InfoGaussian<Plane>>);
    static_assert(Emrys::internal::InfoFormDensity<InfoGaussian<Plane>>);
    static_assert(!Emrys::internal::InfoFormDensity<SqrtGaussian<Plane>>);
    static_assert(!Emrys::isGaussian<SqrtGaussianParams<Plane>>);
}

// The square-root collapse is the covariance form's: the same mean, and a factor of the same
// covariance, made by triangularization; a component of zero weight is left out, and one alone
// with weight is the collapse.
TEST(Component, TheSqrtCollapseIsTheCovarianceCollapse)
{
    auto const components = planeComponents();
    auto factors = std::vector<SqrtGaussianParams<Plane>>{};
    for (auto const & [logWeight, gaussian] : components)
    {
        factors.push_back(toSqrt(gaussian).value());
    }
    auto const logWeightAt = [&](Eigen::Index i) {
        return components[std::size_t(i)].first;
    };
    auto const expected = Emrys::ComponentTraits<GaussianParams<Plane>>::collapse(
        3,
        [&](Eigen::Index i) -> GaussianParams<Plane> const & {
            return components[std::size_t(i)].second;
        },
        logWeightAt
    );
    auto const factorAt = [&](Eigen::Index i) -> SqrtGaussianParams<Plane> const & {
        return factors[std::size_t(i)];
    };
    auto const collapsed = SqrtTraits::collapse(3, factorAt, logWeightAt);
    EXPECT_LT((collapsed.mean() - expected.mean()).norm(), 1e-12);
    Eigen::Matrix2d const covariance = expected.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((covarianceOf(collapsed) - covariance).norm(), 1e-12);
    EXPECT_EQ(collapsed.root()(0, 1), 0.0);

    // Without the third component, the collapse of the first two.
    auto const firstTwo = [&](Eigen::Index i) {
        return i == 2 ? -std::numeric_limits<double>::infinity() : logWeightAt(i);
    };
    auto const expectedTwo = Emrys::ComponentTraits<GaussianParams<Plane>>::collapse(
        3,
        [&](Eigen::Index i) -> GaussianParams<Plane> const & {
            return components[std::size_t(i)].second;
        },
        firstTwo
    );
    auto const collapsedTwo = SqrtTraits::collapse(3, factorAt, firstTwo);
    EXPECT_LT((collapsedTwo.mean() - expectedTwo.mean()).norm(), 1e-12);
    Eigen::Matrix2d const covarianceTwo = expectedTwo.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((covarianceOf(collapsedTwo) - covarianceTwo).norm(), 1e-12);

    auto const alone = SqrtTraits::collapse(3, factorAt, [&](Eigen::Index i) {
        return i == 1 ? 0.0 : -std::numeric_limits<double>::infinity();
    });
    EXPECT_EQ(alone.mean(), factors[1].mean());
    EXPECT_EQ(alone.root(), factors[1].root());
}

// On a periodic coordinate the means straddling the seam collapse to a mean near it, with the
// spread of the means measured across it: the covariance form's collapse, to rounding.
TEST(Component, TheSqrtCollapseWrapsAPeriodicCoordinate)
{
    double const pi = std::numbers::pi;
    auto const gaussians = std::vector<GaussianParams<Heading>>{
        GaussianParams<Heading>{Eigen::Matrix<double, 1, 1>{pi - 0.1}, Eigen::Matrix<double, 1, 1>{0.01}},
        GaussianParams<Heading>{Eigen::Matrix<double, 1, 1>{-pi + 0.2}, Eigen::Matrix<double, 1, 1>{0.04}},
    };
    auto const factors =
        std::vector<SqrtGaussianParams<Heading>>{toSqrt(gaussians[0]).value(), toSqrt(gaussians[1]).value()};
    auto const logWeightAt = [](Eigen::Index i) {
        return std::log(i == 0 ? 0.6 : 0.4);
    };
    auto const expected = Emrys::ComponentTraits<GaussianParams<Heading>>::collapse(
        2,
        [&](Eigen::Index i) -> GaussianParams<Heading> const & {
            return gaussians[std::size_t(i)];
        },
        logWeightAt
    );
    auto const collapsed = Emrys::ComponentTraits<SqrtGaussianParams<Heading>>::collapse(
        2,
        [&](Eigen::Index i) -> SqrtGaussianParams<Heading> const & {
            return factors[std::size_t(i)];
        },
        logWeightAt
    );
    EXPECT_NEAR(collapsed.mean()(0), expected.mean()(0), 1e-12);
    EXPECT_GT(std::abs(collapsed.mean()(0)), 3.0);
    EXPECT_NEAR(collapsed.covariance().value()(0, 0), expected.covarianceLower()(0, 0), 1e-12);
}

// A mixture of square-root forms has the covariance form's density, collapse, merge, and
// marginals: factorize() caches each component's log-determinant, which the density reads.
TEST(Component, AMixtureOfSqrtFormsHasTheCovarianceFormsDensity)
{
    auto covariance = MixtureParams<Gaussian<Plane>>{Plane{}};
    auto squareRoot = MixtureParams<SqrtGaussian<Plane>>{Plane{}};
    for (auto const & [logWeight, gaussian] : planeComponents())
    {
        covariance.add(logWeight, gaussian);
        squareRoot.add(logWeight, toSqrt(gaussian).value());
    }
    {
        auto const expected = covariance.toDensity();
        auto const density = squareRoot.toDensity();
        ASSERT_TRUE(expected);
        ASSERT_TRUE(density);
        for (auto const & x : {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{1.5, -0.5}, Eigen::Vector2d{-2.0, 3.0}})
        {
            EXPECT_NEAR(density->logDensity(x), expected->logDensity(x), 1e-12);
        }
        auto const collapsed = density->collapsed();
        Eigen::Matrix2d const expectedCovariance =
            expected->collapsed().covarianceLower().selfadjointView<Eigen::Lower>();
        EXPECT_LT((collapsed.mean() - expected->collapsed().mean()).norm(), 1e-12);
        EXPECT_LT((covarianceOf(collapsed) - expectedCovariance).norm(), 1e-12);
    }

    auto const marginal = squareRoot.marginal<Across>();
    auto const expectedMarginal = covariance.marginal<Across>();
    ASSERT_EQ(marginal.size(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(marginal[i].component.mean()(0), expectedMarginal[i].component.mean()(0), 1e-12);
        EXPECT_NEAR(
            marginal[i].component.covariance().value()(0, 0),
            expectedMarginal[i].component.covarianceLower()(0, 0),
            1e-12
        );
    }
    auto const copied = marginal.copy();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(copied)>, MixtureParams<SqrtGaussian<Across>>>);
    EXPECT_EQ(copied.size(), 3);

    covariance.merge(0, 2);
    squareRoot.merge(0, 2);
    ASSERT_EQ(squareRoot.size(), 2);
    EXPECT_NEAR(squareRoot.begin()->logWeight, covariance.begin()->logWeight, 1e-12);
    Eigen::Matrix2d const merged = covariance.begin()->component.covarianceLower().selfadjointView<Eigen::Lower>();
    EXPECT_LT((covarianceOf(squareRoot.begin()->component) - merged).norm(), 1e-12);
}

// factorize() reports a factor with a zero on its diagonal, which a mixture names by component.
TEST(Component, ASingularSqrtComponentFailsToFactorize)
{
    auto mixture = MixtureParams<SqrtGaussian<Plane>>{Plane{}};
    mixture.add(0.0, SqrtGaussianParams<Plane>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.0}, {0.5, 0.0}}});
    EXPECT_EQ(mixture.begin()->component.factorize().error(), Emrys::GaussianError::NotPositiveDefinite);
    auto const density = mixture.toDensity();
    ASSERT_FALSE(density);
    auto const * failed = std::get_if<Emrys::ComponentError>(&density.error());
    ASSERT_NE(failed, nullptr);
    EXPECT_EQ(failed->index, 0);
}

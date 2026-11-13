// Discrete spaces and the distributions over them (step 5a, [discrete-spaces], [density-names], [params-and-density]):
// DiscreteSpace and DiscreteProduct cardinalities and named values; CategoricalParams, whose weights are written
// unchecked and checked once by toDensity(), and the Categorical view it returns; normalization,
// entropy, events, marginals, conditionals by path, Bayes updates, run-time cardinality, and
// sampling, a batch or one draw at a time [allocation]; MarkovChain prediction, with independent axes of
// a product, chains from mean dwell times and exit weights, and stationary distributions.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::Categorical;
using Emrys::CategoricalParams;
using Emrys::ChainError;
using Emrys::DiscreteProduct;
using Emrys::DiscreteSpace;
using Emrys::Finite;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::TagPath;
using Emrys::valueOf;
using Emrys::WeightError;

namespace
{

struct Cruise
{};

struct Turn
{};

struct Maneuver : Named<Cruise, Turn>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Named<Inlier, Outlier>
{};

using Mode = DiscreteProduct<Maneuver, Sensor>;

/// whether \a P has a fromWeights() that takes two weights as arguments
template <typename P>
concept TakesTwoWeights = requires { P::fromWeights(0.5, 0.5); };

struct SensorA : Named<Inlier, Outlier>
{};

struct SensorB : Named<Inlier, Outlier>
{};

struct Sensors : DiscreteProduct<SensorA, SensorB>
{};

using Nested = DiscreteProduct<Maneuver, Sensors>;

struct Cell : Finite<3>
{};

constexpr double inf = std::numeric_limits<double>::infinity();

/** The probabilities of \a params, which must define a distribution. */
template <typename Params>
Eigen::ArrayXd
probabilitiesOf(Params const & params)
{
    return params.toDensity().value().probabilities();
}

}  // namespace

// A product of fixed dimensions is empty, like a fixed-size Composite.
static_assert(std::is_empty_v<Mode> && std::is_empty_v<DiscreteProduct<Maneuver, Finite<3>>>);

// Named, Finite, and DiscreteProduct satisfy DiscreteSpace and Linear does not; fixed cardinalities
// and value indices are constexpr, a product with a dynamic axis has dynamic cardinality and a
// bound, and a run-time product reports cardinality, strides, flat values, and coordinates.
TEST(DiscreteSpace, Cardinalities)
{
    static_assert(DiscreteSpace<Maneuver> && DiscreteSpace<Mode> && DiscreteSpace<Finite<5>>);
    static_assert(DiscreteSpace<Finite<Eigen::Dynamic>> && DiscreteSpace<Nested>);
    static_assert(!DiscreteSpace<Linear<2>>);
    static_assert(std::is_empty_v<Maneuver> && std::is_empty_v<Finite<4>>);
    static_assert(Maneuver::Cardinality == 2 && Mode::Cardinality == 4 && Nested::Cardinality == 8);
    static_assert(Maneuver::index<Cruise>() == 0 && Maneuver::index<Turn>() == 1);
    static_assert(DiscreteProduct<Cell, Finite<Eigen::Dynamic>>::Cardinality == Eigen::Dynamic);
    static_assert(DiscreteProduct<Cell, Finite<Eigen::Dynamic, 4>>::MaxCardinality == 12);

    auto const grid = DiscreteProduct<Cell, Finite<Eigen::Dynamic>>{Cell{}, Finite<Eigen::Dynamic>{5}};
    EXPECT_EQ(grid.cardinality(), 15);
    EXPECT_EQ(grid.stride<0>(), 5);
    EXPECT_EQ(grid.stride<1>(), 1);
    EXPECT_EQ(grid.value({2, 3}), 13);
    EXPECT_EQ(grid.coordinates(13), (std::array<Eigen::Index, 2>{2, 3}));
    EXPECT_NE(grid, (DiscreteProduct<Cell, Finite<Eigen::Dynamic>>{Cell{}, Finite<Eigen::Dynamic>{4}}));
}

// valueOf<Names...>() flattens a product's named values with the last axis varying fastest, in any
// order of names, and a nested product flattens as if its axes were listed in place.
TEST(DiscreteSpace, ValuesByName)
{
    // The last dimension varies fastest.
    EXPECT_EQ((valueOf<Mode, Cruise, Inlier>()), 0);
    EXPECT_EQ((valueOf<Mode, Cruise, Outlier>()), 1);
    EXPECT_EQ((valueOf<Mode, Turn, Inlier>()), 2);
    EXPECT_EQ((valueOf<Mode, Outlier, Turn>()), 3);
    EXPECT_EQ((valueOf<Maneuver, Turn>()), 1);

    // A nested product flattens as if its dimensions were listed in place.
    EXPECT_EQ((valueOf<Nested, Turn, TagPath<SensorA, Inlier>, TagPath<SensorB, Outlier>>()), 5);
    EXPECT_EQ((valueOf<Nested, Turn, TagPath<Sensors, SensorA, Outlier>, TagPath<SensorB, Outlier>>()), 7);
    EXPECT_EQ((DiscreteProduct<Maneuver, SensorA, SensorB>{}.value({1, 0, 1})), 5);
}

namespace
{

struct Row : Finite<Eigen::Dynamic>
{
    using Finite::Finite;
};

/** \returns 1 for the mode (Turn, Outlier), whose flat index is a constant expression usable as a case label */
int
isTurnOutlier(Eigen::Index mode)
{
    switch (mode)
    {
        case valueOf<Mode, Turn, Outlier>():
            return 1;
        default:
            return 0;
    }
}

}  // namespace

// at<Axis>(i) gives an axis without names its value, alone or beside named values, in any order; a
// Named axis takes an index too, and a product takes one per leaf axis in order. valueOf() of a
// fixed-size space is a constant expression, and of a run-time one counts its run-time cardinality.
TEST(DiscreteSpace, ValuesByIndex)
{
    using Gridded = DiscreteProduct<Maneuver, Cell>;
    static_assert(valueOf<Gridded, Turn>(Emrys::at<Cell>(2)) == 5);
    static_assert(valueOf<Gridded>(Emrys::at<Cell>(2), Emrys::at<Maneuver>(1)) == 5);
    static_assert(valueOf<Gridded>(Emrys::at<Gridded>(1, 2)) == 5);
    static_assert(valueOf<Turn>(Gridded{}, Emrys::at<Cell>(2)) == 5);
    static_assert(valueOf<Mode, Turn, Outlier>() == 3);
    EXPECT_EQ(isTurnOutlier(3), 1);
    EXPECT_EQ(isTurnOutlier(2), 0);

    // A nested product takes one index per leaf axis, so Sensors's two flatten in place.
    static_assert(valueOf<Nested, Turn>(Emrys::at<Sensors>(1, 1)) == 7);
    static_assert(valueOf<Nested, Turn, TagPath<SensorA, Inlier>>(Emrys::at<TagPath<Sensors, SensorB>>(1)) == 5);

    auto const space = DiscreteProduct<Maneuver, Row>{Maneuver{}, Row{5}};
    EXPECT_EQ(valueOf<Turn>(space, Emrys::at<Row>(3)), 8);
    EXPECT_EQ(valueOf<Cruise>(space, Emrys::at<Row>(4)), 4);
}

// Parameters from a space are uniform, with log-weights that are already log-probabilities;
// fromWeights(), with or without its fixed-size space, and log-weights need not sum to one, and
// toDensity() normalizes them in logs (so -1000 is fine) and reports the probabilities,
// log-probabilities, and the most probable value.
TEST(Categorical, NormalizesWeights)
{
    auto const uniform = CategoricalParams{Mode{}};
    static_assert(std::is_same_v<decltype(uniform), CategoricalParams<Mode> const>);
    static_assert(decltype(uniform)::ArrayType::SizeAtCompileTime == 4);
    static_assert(std::is_same_v<Emrys::Params<Categorical<Mode>>, CategoricalParams<Mode>>);
    static_assert(std::is_same_v<Emrys::Density<CategoricalParams<Mode>>, Categorical<Mode>>);
    EXPECT_TRUE(uniform.logWeights().isApproxToConstant(std::log(0.25)));
    EXPECT_TRUE(probabilitiesOf(uniform).isApproxToConstant(0.25));

    auto const params = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{6, 1, 2.5, 0.5});
    EXPECT_EQ(
        CategoricalParams<Mode>::fromWeights(Eigen::Vector4d{6, 1, 2.5, 0.5}).logWeights().matrix(),
        params.logWeights().matrix()
    );
    auto const density = params.toDensity();
    ASSERT_TRUE(density);
    EXPECT_TRUE(density->probabilities().isApprox(Eigen::Array4d{0.6, 0.1, 0.25, 0.05}));
    EXPECT_NEAR(density->logProbabilities().exp().sum(), 1.0, 1e-15);
    EXPECT_DOUBLE_EQ(density->probability(2), 0.25);
    EXPECT_EQ(density->mostProbable(), 0);

    auto const fromLogs = CategoricalParams{Maneuver{}, Eigen::Array2d{-1000, -1000 + std::log(3.0)}};
    EXPECT_NEAR(fromLogs.toDensity().value().probability(1), 0.75, 1e-12);  // -1000 + log 3 is rounded to 1e-13
}

// fromWeights() of a fixed-size space with a fixed number of values takes the weights as
// arguments, one per value in flat-index order, as the Eigen vector would give them; a space
// whose number of values is known only at run time has no such form.
TEST(Categorical, FromWeightsAsArguments)
{
    auto const params = CategoricalParams<Mode>::fromWeights(6, 1, 2.5, 0.5);
    EXPECT_EQ(
        params.logWeights().matrix(),
        CategoricalParams<Mode>::fromWeights(Eigen::Vector4d{6, 1, 2.5, 0.5}).logWeights().matrix()
    );
    auto const single = CategoricalParams<Maneuver, float>::fromWeights(0.9, 0.1);
    static_assert(std::is_same_v<decltype(single), CategoricalParams<Maneuver, float> const>);
    EXPECT_NEAR(single.toDensity().value().probability(0), 0.9f, 1e-6f);

    struct Cells : Finite<Eigen::Dynamic>
    {
        using Finite::Finite;
    };

    static_assert(TakesTwoWeights<CategoricalParams<Maneuver>> && !TakesTwoWeights<CategoricalParams<Cells>>);
}

// entropy() is -sum p log p in nats: log n for the uniform distribution, zero for a certain value
// (whose impossible values add nothing rather than 0 * -inf = NaN), and the same in float.
TEST(Categorical, Entropy)
{
    auto const uniform = CategoricalParams{Mode{}};
    EXPECT_NEAR(uniform.toDensity()->entropy(), std::log(4.0), 1e-15);
    auto const params = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{6, 1, 2.5, 0.5});
    double const expected =
        -(0.6 * std::log(0.6) + 0.1 * std::log(0.1) + 0.25 * std::log(0.25) + 0.05 * std::log(0.05));
    EXPECT_NEAR(params.toDensity()->entropy(), expected, 1e-15);
    EXPECT_NEAR(params.toDensity()->entropy(), 1.0331141, 1e-7);

    auto const certain = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0, 0, 2, 0});
    EXPECT_EQ(certain.toDensity()->entropy(), 0.0);

    auto const single = CategoricalParams<Mode, float>::fromWeights(Mode{}, Eigen::Vector4f{6, 1, 2.5f, 0.5f});
    static_assert(std::is_same_v<decltype(single.toDensity()->entropy()), float>);
    EXPECT_NEAR(single.toDensity()->entropy(), 1.0331141f, 1e-6f);
}

// Weights that fail to define a distribution are accepted as written and reported by
// toDensity() and logTotalWeight(): all zero, NaN, +inf, and a negative weight (whose logarithm
// is NaN).
TEST(Categorical, ToDensityChecksTheWeights)
{
    auto const zero = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0, 0});
    EXPECT_EQ(zero.toDensity().error(), WeightError::ZeroSum);
    EXPECT_EQ(zero.logTotalWeight().error(), WeightError::ZeroSum);
    EXPECT_EQ((CategoricalParams{Maneuver{}, Eigen::Array2d{0, NAN}}.logTotalWeight().error()), WeightError::NonFinite);
    EXPECT_EQ((CategoricalParams{Maneuver{}, Eigen::Array2d{0, inf}}.logTotalWeight().error()), WeightError::NonFinite);
    auto const negative = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{1, -1});
    EXPECT_EQ(negative.toDensity().error(), WeightError::NonFinite);

    // normalize() reports the same error and leaves the weights as they were.
    auto unchanged = zero;
    EXPECT_EQ(unchanged.normalize().error(), WeightError::ZeroSum);
    EXPECT_TRUE((unchanged.logWeights() == zero.logWeights()).all());
}

// Weights are written one at a time, by index or by name, through states that fail to define a
// distribution: a histogram filled from zero, and a value ruled out. Only the finished set is
// checked, by toDensity().
TEST(Categorical, WritesWeightsUnchecked)
{
    auto histogram = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array4d::Zero());
    EXPECT_FALSE(histogram.toDensity());
    for (Eigen::Index const v : {0, 2, 2, 3, 2, 0})
    {
        histogram.setWeight(v, std::exp(histogram.logWeight(v)) + 1);
    }
    EXPECT_TRUE(probabilitiesOf(histogram).isApprox(Eigen::Array4d{2, 0, 3, 1} / 6));

    histogram.setWeight<Turn, Inlier>(0);
    EXPECT_TRUE(probabilitiesOf(histogram).isApprox(Eigen::Array4d{2, 0, 0, 1} / 3));
    histogram.setLogWeight<Cruise, Outlier>(std::log(3.0));
    EXPECT_TRUE(probabilitiesOf(histogram).isApprox(Eigen::Array4d{2, 3, 0, 1} / 6));
    histogram.setWeights(Eigen::Array4d{0, 0, 0, 0});
    EXPECT_EQ(histogram.toDensity().error(), WeightError::ZeroSum);
    histogram.setLogWeights(Eigen::Array4d{0, 0, 0, std::log(3.0)});
    EXPECT_TRUE(probabilitiesOf(histogram).isApprox(Eigen::Array4d{1, 1, 1, 3} / 6));
}

// probability<Names...>() sums an event over a product, marginal<Axis>() sums out the other axes,
// conditional<Value>() keeps the rest, and conditioning on an impossible value gives parameters
// whose toDensity() fails with ZeroSum.
TEST(Categorical, EventsMarginalsAndConditionals)
{
    auto const params = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0.6, 0.1, 0.25, 0.05});
    auto const p = params.toDensity().value();
    EXPECT_NEAR(p.probability<Turn>(), 0.3, 1e-15);
    EXPECT_NEAR(p.probability<Outlier>(), 0.15, 1e-15);
    EXPECT_NEAR((p.probability<Turn, Outlier>()), 0.05, 1e-15);
    EXPECT_NEAR((p.probability<TagPath<Sensor, Inlier>>()), 0.85, 1e-15);

    CategoricalParams<Sensor> const sensor = p.marginal<Sensor>();
    EXPECT_TRUE(sensor.logWeights().exp().isApprox(Eigen::Array2d{0.85, 0.15}));
    CategoricalParams<Maneuver> const maneuver = params.marginal<Maneuver>();
    EXPECT_TRUE(probabilitiesOf(maneuver).isApprox(Eigen::Array2d{0.7, 0.3}));

    auto const givenTurn = p.conditional<Turn>();
    static_assert(std::is_same_v<decltype(givenTurn), CategoricalParams<Sensor> const>);
    EXPECT_TRUE(probabilitiesOf(givenTurn).isApprox(Eigen::Array2d{0.25, 0.05} / 0.3));
    auto const givenOutlier = params.conditional<Sensor>(Sensor::index<Outlier>());
    static_assert(std::is_same_v<decltype(givenOutlier), CategoricalParams<Maneuver> const>);
    EXPECT_TRUE(probabilitiesOf(givenOutlier).isApprox(Eigen::Array2d{0.1, 0.05} / 0.15));
    auto const density = params.toDensity();
    ASSERT_TRUE(density);
    EXPECT_TRUE(probabilitiesOf(density->conditional<Sensor>(Sensor::index<Outlier>()))
                    .isApprox(Eigen::Array2d{0.1, 0.05} / 0.15));

    auto const impossible = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0.5, 0, 0.5, 0});
    EXPECT_EQ(impossible.conditional<Outlier>().logTotalWeight().error(), WeightError::ZeroSum);

    // The marginal of an impossible value is impossible, not a NaN from the log of a sum of none.
    Eigen::Array2d const sensorWeights = impossible.marginal<Sensor>().logWeights();
    EXPECT_EQ(sensorWeights(1), -inf);
    EXPECT_NEAR(sensorWeights(0), 0.0, 1e-15);
}

// Marginals, events, and conditionals address axes of a nested product by tag path, and a
// conditional's space is the product of the remaining axes, flattened.
TEST(Categorical, NestedDimensionsByPath)
{
    auto w = Eigen::Array<double, 8, 1>{};
    w << 1, 2, 3, 4, 5, 6, 7, 8;
    auto const params = CategoricalParams<Nested>::fromWeights(Nested{}, w);
    auto const p = params.toDensity().value();
    double const total = 36;

    CategoricalParams<Sensors> const sensors = p.marginal<Sensors>();
    EXPECT_TRUE(probabilitiesOf(sensors).isApprox(Eigen::Array4d{6, 8, 10, 12} / total));
    CategoricalParams<SensorB> const b = p.marginal<SensorB>();
    EXPECT_TRUE(probabilitiesOf(b).isApprox(Eigen::Array2d{16, 20} / total));
    EXPECT_NEAR((p.probability<TagPath<Sensors, SensorB, Outlier>>()), 20 / total, 1e-15);
    EXPECT_NEAR((p.probability<Turn, TagPath<SensorA, Outlier>>()), (7 + 8) / total, 1e-15);

    // Conditioning on a value keeps the remaining dimensions, with nested products flattened.
    auto const givenTurn = p.conditional<Turn>();
    static_assert(std::is_same_v<decltype(givenTurn), CategoricalParams<Sensors> const>);
    EXPECT_TRUE(probabilitiesOf(givenTurn).isApprox(Eigen::Array4d{5, 6, 7, 8} / 26));
    auto const givenAOutlier = p.conditional<TagPath<SensorA, Outlier>>();
    static_assert(std::is_same_v<decltype(givenAOutlier), CategoricalParams<DiscreteProduct<Maneuver, SensorB>> const>);
    EXPECT_TRUE(probabilitiesOf(givenAOutlier).isApprox(Eigen::Array4d{3, 4, 7, 8} / 22));
}

// update(logLikelihoods) multiplies in the likelihoods, normalizes, and returns the log-evidence,
// from weights that need not sum to one; it works with likelihoods far below the double range,
// and leaves the weights unchanged when every likelihood is zero or the prior fails to define a
// distribution.
TEST(Categorical, BayesUpdate)
{
    auto p = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{8, 2});
    auto const evidence = p.update(Eigen::Array2d{std::log(0.1), std::log(0.6)});
    ASSERT_TRUE(evidence);
    EXPECT_NEAR(*evidence, std::log(0.8 * 0.1 + 0.2 * 0.6), 1e-15);
    EXPECT_TRUE(p.logWeights().exp().isApprox(Eigen::Array2d{0.08, 0.12} / 0.2));

    // Likelihoods far below the double range still update, since everything stays in logs.
    ASSERT_TRUE(p.update(Eigen::Array2d{-2000, -2001}));
    EXPECT_NEAR(probabilitiesOf(p)(0), 0.4 / (0.4 + 0.6 / std::exp(1.0)), 1e-12);

    CategoricalParams const before = p;
    EXPECT_EQ(p.update(Eigen::Array2d{-inf, -inf}).error(), WeightError::ZeroSum);
    EXPECT_TRUE(p.logWeights().isApprox(before.logWeights()));

    auto invalid = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0, 0});
    EXPECT_EQ(invalid.update(Eigen::Array2d{0, 0}).error(), WeightError::ZeroSum);
}

// reweight() multiplies in several measurements' likelihoods without a check, and one
// normalization afterward gives the posterior that an update() per measurement gives.
TEST(Categorical, ReweightsBeforeOneNormalization)
{
    auto const prior = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0.6, 0.1, 0.25, 0.05});
    auto const first = Eigen::Array4d{std::log(0.2), std::log(0.9), std::log(0.4), std::log(0.1)};
    auto const second = Eigen::Array4d{-3000, -3001, -3002, -inf};

    auto accumulated = prior;
    accumulated.reweight(first);
    accumulated.reweight(second);
    ASSERT_TRUE(accumulated.normalize());

    auto sequential = prior;
    ASSERT_TRUE(sequential.update(first));
    ASSERT_TRUE(sequential.update(second));
    EXPECT_TRUE(accumulated.logWeights().head<3>().isApprox(sequential.logWeights().head<3>(), 1e-12));
    EXPECT_EQ(accumulated.logWeights()(3), -inf);
}

// next(i) is the distribution of the value after value i, row i of the matrix, and is predict() of
// the distribution certain of i; a transition of probability zero has log-weight minus infinity.
TEST(MarkovChain, NextIsARowOfTheMatrix)
{
    auto const chain = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.97, 0.03}, {1.0, 0.0}}};
    CategoricalParams<Maneuver> const fromCruise = chain.next(0);
    EXPECT_TRUE(probabilitiesOf(fromCruise).isApprox(Eigen::Array2d{0.97, 0.03}, 1e-15));
    auto const certain = CategoricalParams<Maneuver>::fromWeights(1.0, 0.0);
    EXPECT_TRUE(probabilitiesOf(fromCruise).isApprox(probabilitiesOf(predict(certain, chain)), 1e-15));
    CategoricalParams<Maneuver> const fromTurn = chain.next(valueOf<Maneuver, Turn>());
    EXPECT_EQ(fromTurn.logWeights()(1), -inf);
    EXPECT_EQ(fromTurn.toDensity().value().mostProbable(), 0);
}

// persistent(p) builds the chain that stays with probability p and spreads the rest evenly,
// predict() applies one step and keeps the total weight, and memoryless(q) forgets the start.
TEST(MarkovChain, PredictsOneStep)
{
    MarkovChain const chain = MarkovChain<Maneuver>::persistent(0.9);
    EXPECT_TRUE(chain.matrix().isApprox(Eigen::Matrix2d{{0.9, 0.1}, {0.1, 0.9}}));
    auto const p = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.7, 0.3});
    CategoricalParams<Maneuver> const next = predict(p, chain);
    EXPECT_TRUE(next.logWeights().exp().isApprox(Eigen::Array2d{0.7 * 0.9 + 0.3 * 0.1, 0.7 * 0.1 + 0.3 * 0.9}));

    // Weights that don't sum to one keep their total, far below the double range too.
    auto const scaled = CategoricalParams{Maneuver{}, Eigen::Array2d{std::log(0.7) - 5000, std::log(0.3) - 5000}};
    EXPECT_TRUE((predict(scaled, chain).logWeights() + 5000).isApprox(next.logWeights(), 1e-12));

    MarkovChain<Cell> const three = MarkovChain<Cell>::persistent(0.6);
    EXPECT_TRUE(three.matrix().isApprox(Eigen::Matrix3d{{0.6, 0.2, 0.2}, {0.2, 0.6, 0.2}, {0.2, 0.2, 0.6}}));

    // A memoryless chain forgets where it started.
    auto const target = CategoricalParams<Sensor>::fromWeights(Sensor{}, Eigen::Vector2d{0.9, 0.1});
    MarkovChain const memoryless = MarkovChain<Sensor>::memoryless(target.toDensity().value());
    auto const outlier = CategoricalParams<Sensor>::fromWeights(Sensor{}, Eigen::Vector2d{0, 1});
    EXPECT_TRUE(probabilitiesOf(predict(outlier, memoryless)).isApprox(Eigen::Array2d{0.9, 0.1}));
}

// An rvalue transition matrix of the stored type is moved in (same data pointer), another scalar is
// cast, and an lvalue is copied [forwarded-matrices].
TEST(MarkovChain, MovesRvalueTransitions)
{
    using Chain = MarkovChain<Finite<Eigen::Dynamic>>;
    Chain::MatrixType transitions = Eigen::Matrix2d{{0.9, 0.1}, {0.3, 0.7}};  // row-major, as stored
    double const * data = transitions.data();
    auto const moved = Chain{Finite<Eigen::Dynamic>{2}, std::move(transitions)};
    EXPECT_EQ(moved.matrix().data(), data);

    // Another scalar is cast, and an lvalue is copied.
    Eigen::MatrixXf const single = Eigen::Matrix2f{{0.5f, 0.5f}, {0.25f, 0.75f}};
    auto const cast = Chain{Finite<Eigen::Dynamic>{2}, single};
    EXPECT_DOUBLE_EQ(cast.transition(1, 1), 0.75);
    auto const copied = Chain{Finite<Eigen::Dynamic>{2}, moved.matrix()};
    EXPECT_NE(copied.matrix().data(), moved.matrix().data());
    EXPECT_EQ(copied.matrix(), moved.matrix());
}

// A chain over a product built from one chain per axis has the Kronecker-product transition matrix,
// and an independent prior stays independent with each marginal moved by its own chain.
TEST(MarkovChain, IndependentDimensionsOfAProduct)
{
    auto const maneuver = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.2, 0.8}}};
    auto const sensor = MarkovChain<Sensor>{Eigen::Matrix2d{{0.9, 0.1}, {0.5, 0.5}}};
    auto const mode = MarkovChain<Mode>{maneuver, sensor};
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        for (Eigen::Index j = 0; j < 4; ++j)
        {
            EXPECT_DOUBLE_EQ(
                mode.transition(i, j),
                maneuver.transition(i / 2, j / 2) * sensor.transition(i % 2, j % 2)
            );
        }
    }

    // An independent prior stays independent, and each marginal moves by its own chain.
    auto const pm = Eigen::Array2d{0.7, 0.3};
    auto const ps = Eigen::Array2d{0.6, 0.4};
    auto const prior = CategoricalParams<Mode>::fromWeights(
        Mode{},
        Eigen::Array4d{pm(0) * ps(0), pm(0) * ps(1), pm(1) * ps(0), pm(1) * ps(1)}
    );
    CategoricalParams<Mode> const next = predict(prior, mode);
    CategoricalParams<Maneuver> const nextManeuver =
        predict(CategoricalParams<Maneuver>::fromWeights(Maneuver{}, pm), maneuver);
    CategoricalParams<Sensor> const nextSensor = predict(CategoricalParams<Sensor>::fromWeights(Sensor{}, ps), sensor);
    EXPECT_TRUE(probabilitiesOf(next.marginal<Maneuver>()).isApprox(probabilitiesOf(nextManeuver)));
    EXPECT_TRUE(probabilitiesOf(next.marginal<Sensor>()).isApprox(probabilitiesOf(nextSensor)));
    EXPECT_NEAR(
        (next.toDensity().value().probability<Turn, Outlier>()),
        probabilitiesOf(nextManeuver)(1) * probabilitiesOf(nextSensor)(1),
        1e-15
    );
}

// fromDwellTimes() stays at value i with probability 1 - 1/tau_i, so that the geometric number of
// steps there has mean tau_i, and splits the rest by the exit weights: the matrices are computed
// by hand. Without exit weights it splits evenly, and an infinite dwell time never leaves.
TEST(MarkovChain, FromDwellTimes)
{
    MarkovChain const two = MarkovChain<Maneuver>::fromDwellTimes(Eigen::Vector2d{20, 5});
    EXPECT_TRUE(two.matrix().isApprox(Eigen::Matrix2d{{0.95, 0.05}, {0.2, 0.8}}, 1e-15));

    auto const exits = Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}};
    MarkovChain const flight = MarkovChain<Cell>::fromDwellTimes(Eigen::Vector3d{30, 8, 4}, exits);
    auto const expected =
        Eigen::Matrix3d{{1 - 1.0 / 30, 2.0 / 90, 1.0 / 90}, {1.0 / 8, 1 - 1.0 / 8, 0}, {1.0 / 4, 0, 1 - 1.0 / 4}};
    EXPECT_TRUE(flight.matrix().isApprox(expected, 1e-15));

    // Even exits are persistent() with each value's own stay probability.
    MarkovChain const even = MarkovChain<Cell>::fromDwellTimes(Eigen::Vector3d{10, 10, 10});
    EXPECT_TRUE(even.matrix().isApprox(MarkovChain<Cell>::persistent(0.9).matrix(), 1e-15));

    // An infinite dwell time is absorbing, and its row of exit weights may be zero.
    auto const oneWay = Eigen::Matrix2d{{0, 1}, {0, 0}};
    MarkovChain const absorbing = MarkovChain<Maneuver>::fromDwellTimes(Eigen::Vector2d{4, inf}, oneWay);
    EXPECT_TRUE(absorbing.matrix().isApprox(Eigen::Matrix2d{{0.75, 0.25}, {0, 1}}, 1e-15));
}

// stationary() gives the probabilities that one step leaves alone. For the three-value chain of
// FromDwellTimes the jump chain visits the values 1/2, 1/3, and 1/6 of the time and the chain stays
// 30, 8, and 4 steps per visit, so the stationary probabilities are 15 : 8/3 : 2/3, or 9/11, 8/55,
// and 2/55. The two-value chain's are (b, a) / (a + b) in closed form, kept to the last digits
// when a and b are 1e-12, where a direct solve of pi (P - I) = 0 loses about twelve.
TEST(MarkovChain, Stationary)
{
    auto const exits = Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}};
    MarkovChain const flight = MarkovChain<Cell>::fromDwellTimes(Eigen::Vector3d{30, 8, 4}, exits);
    auto const prior = stationary(flight);
    ASSERT_TRUE(prior);
    auto const p = probabilitiesOf(*prior);
    EXPECT_TRUE(p.isApprox(Eigen::Array3d{9.0 / 11, 8.0 / 55, 2.0 / 55}, 1e-14));
    EXPECT_TRUE(probabilitiesOf(predict(*prior, flight)).isApprox(p, 1e-14));

    double const a = 1e-12;
    double const b = 3e-12;
    auto const sticky = MarkovChain<Maneuver>{Eigen::Matrix2d{{1 - a, a}, {b, 1 - b}}};
    auto const settled = stationary(sticky);
    ASSERT_TRUE(settled);
    EXPECT_NEAR(probabilitiesOf(*settled)(0), 0.75, 1e-15);
    EXPECT_NEAR(probabilitiesOf(*settled)(1), 0.25, 1e-15);

    // A product of independent axes settles into the product of the axes' stationary distributions.
    auto const maneuver = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.2, 0.8}}};
    auto const sensor = MarkovChain<Sensor>{Eigen::Matrix2d{{0.9, 0.1}, {0.5, 0.5}}};
    auto const mode = stationary(MarkovChain<Mode>{maneuver, sensor});
    ASSERT_TRUE(mode);
    Eigen::Array2d const pm = probabilitiesOf(stationary(maneuver).value());
    Eigen::Array2d const ps = probabilitiesOf(stationary(sensor).value());
    EXPECT_TRUE(probabilitiesOf(*mode)
                    .isApprox(Eigen::Array4d{pm(0) * ps(0), pm(0) * ps(1), pm(1) * ps(0), pm(1) * ps(1)}, 1e-14));

    // One value is its own stationary distribution.
    auto const single = stationary(MarkovChain<Finite<1>>{});
    ASSERT_TRUE(single);
    EXPECT_EQ(probabilitiesOf(*single)(0), 1.0);
}

// A chain in which some value can't reach some other is Reducible: the identity on two values, for
// which every distribution is stationary, and the two chains with one absorbing value, whose
// stationary distribution is unique but zero at the other value, whichever of the two is absorbing.
TEST(MarkovChain, StationaryOfAReducibleChain)
{
    EXPECT_EQ(stationary(MarkovChain<Maneuver>{}).error(), ChainError::Reducible);
    auto const transient = MarkovChain<Maneuver>{Eigen::Matrix2d{{1, 0}, {0.5, 0.5}}};
    EXPECT_EQ(stationary(transient).error(), ChainError::Reducible);
    auto const leaving = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.5, 0.5}, {0, 1}}};
    EXPECT_EQ(stationary(leaving).error(), ChainError::Reducible);
}

// A product with a run-time Finite axis gives dynamic-size parameters, with marginals and a
// product chain that take the axis instance.
TEST(Categorical, RunTimeCardinality)
{
    struct Cells : Finite<Eigen::Dynamic>
    {
        using Finite::Finite;
    };

    using Grid = DiscreteProduct<Maneuver, Cells>;
    auto const grid = Grid{Maneuver{}, Cells{3}};
    auto const p = CategoricalParams{grid};
    static_assert(decltype(p)::ArrayType::SizeAtCompileTime == Eigen::Dynamic);
    EXPECT_EQ(p.cardinality(), 6);
    CategoricalParams<Cells> const cells = p.marginal<Cells>();
    EXPECT_EQ(cells.cardinality(), 3);
    EXPECT_TRUE(probabilitiesOf(cells).isApproxToConstant(1.0 / 3));
    auto const chain =
        MarkovChain<Grid>{grid, MarkovChain<Maneuver>::persistent(0.9), MarkovChain<Cells>::persistent(0.5, Cells{3})};
    EXPECT_TRUE(probabilitiesOf(predict(p, chain)).isApproxToConstant(1.0 / 6));
}

// 100,000 draws have empirical frequencies within 0.005 of the probabilities.
TEST(Categorical, DrawsSamplesInProportion)
{
    auto const params = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0.6, 0.1, 0.25, 0.05});
    auto const p = params.toDensity().value();
    auto rng = std::mt19937_64{42};
    std::vector<Eigen::Index> const samples = drawSamples(p, 100'000, rng);
    Eigen::Array4d counts = Eigen::Array4d::Zero();
    for (Eigen::Index const v : samples)
    {
        counts(v) += 1;
    }
    EXPECT_TRUE(((counts / 100'000 - p.probabilities()).abs() < 0.005).all()) << counts.transpose();
}

// 100,000 single draws, one drawSample() at a time, which scans the running sum of the
// probabilities rather than building a std::discrete_distribution, have the same frequencies to
// within 0.005, about five standard errors for the largest probability.
TEST(Categorical, StreamOfSingleDrawsIsInProportion)
{
    auto const params = CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{0.6, 0.1, 0.25, 0.05});
    auto const p = params.toDensity().value();
    auto rng = std::mt19937_64{43};
    Eigen::Array4d counts = Eigen::Array4d::Zero();
    for (int k = 0; k < 100'000; ++k)
    {
        counts(Emrys::drawSample(p, rng)) += 1;
    }
    EXPECT_TRUE(((counts / 100'000 - p.probabilities()).abs() < 0.005).all()) << counts.transpose();
}

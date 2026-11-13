// ParticleSetMarginal and the particle conditional (steps 7, 8a, 17l): a marginal selects state rows
// and keeps the weights and a stored bandwidth, and iterates the particles as a range [element-marginals], the
// bandwidth is stored or Silverman's, a KernelDensityConditional is the joint over the marginal,
// and reading a marginal after a structural change of its set dies [view-stamp].

#include <EmrysFilter/Distributions/ParticleSet.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

using Emrys::Composite;
using Emrys::GaussianError;
using Emrys::Linear;
using Emrys::NumericError;
using Emrys::ParticleSet;
using Emrys::TagPath;
using Emrys::WeightError;

namespace
{

struct X : Linear<1>
{};

struct Y : Linear<1>
{};

struct Z : Linear<1>
{};

struct Horizontal : Composite<X, Y>
{};

using State = Composite<Horizontal, Z>;  // X, Y, Z are coordinates 0, 1, 2

using Particles3 = ParticleSet<State>;

// Nine full-rank particles with unnormalized weights 1..9.
Particles3
make()
{
    Eigen::Matrix<double, 3, 9> const states =
        Eigen::Matrix<double, 3, 9>::NullaryExpr([](Eigen::Index i, Eigen::Index j) {
            return std::sin(1.3 * double(i * j) + 0.7 * double(j) + double(i));
        });
    Eigen::Array<double, 9, 1> const logWeights = Eigen::Array<double, 9, 1>::NullaryExpr([](Eigen::Index j) {
        return std::log(double(j + 1));
    });
    return Particles3{states, logWeights};
}

Eigen::Matrix3d const bandwidth3{{0.30, 0.05, 0.02}, {0.05, 0.20, -0.04}, {0.02, -0.04, 0.25}};

}  // namespace

// A tag marginal of a particle set views the block's rows of the states, keeps every log-weight,
// has the block's space, compile-time dimension, and rows, and has no bandwidth when the set
// stores none.
TEST(ParticleSetMarginal, SelectsStateRowsAndKeepsWeights)
{
    auto const set = make();
    auto const marginal = set.marginal<Horizontal>();
    static_assert(decltype(marginal)::ValueDimensionAtCompileTime == 2);
    EXPECT_EQ(marginal.size(), 9);
    EXPECT_EQ(marginal.valueDimension(), 2);
    EXPECT_EQ(marginal.differenceDimension(), 2);
    EXPECT_EQ(marginal.space(), Horizontal{});
    EXPECT_EQ(Eigen::VectorX<Eigen::Index>{marginal.indices()}, (Eigen::Vector2<Eigen::Index>{0, 1}));
    EXPECT_EQ(Eigen::VectorX<Eigen::Index>{marginal.differenceIndices()}, (Eigen::Vector2<Eigen::Index>{0, 1}));
    EXPECT_EQ(Eigen::MatrixXd{marginal.states()}, Eigen::MatrixXd{set.states().topRows<2>()});
    EXPECT_EQ(Eigen::Vector2d{marginal.state(4)}, Eigen::Vector2d(set.state(4)(0), set.state(4)(1)));
    EXPECT_TRUE((marginal.logWeights() == set.logWeights()).all());
    EXPECT_FALSE(marginal.bandwidth().has_value());
    EXPECT_FALSE(marginal.copy().bandwidth().has_value());
}

// A particle set's marginal is a random-access range over the particles, each element holding the
// particle's log-weight and the selected entries of its state, read in place [element-marginals].
TEST(ParticleSetMarginal, IteratesTheParticles)
{
    auto const set = make();
    auto const marginal = set.marginal<Horizontal>();
    using Marginal = std::remove_const_t<decltype(marginal)>;
    static_assert(std::ranges::random_access_range<Marginal const>);
    static_assert(std::ranges::sized_range<Marginal const>);
    EXPECT_EQ(std::ranges::distance(marginal), 9);
    Eigen::Index i = 0;
    for (auto const & [logWeight, state] : marginal)
    {
        EXPECT_EQ(logWeight, set.logWeight(i));
        EXPECT_EQ(Eigen::Vector2d{state}, Eigen::Vector2d(set.state(i)(0), set.state(i)(1)));
        ++i;
    }
    EXPECT_EQ(i, 9);
    EXPECT_EQ(marginal[8].logWeight, std::log(9.0));
}

// A stored bandwidth marginalizes to its block, in the view and in a copy of it.
TEST(ParticleSetMarginal, MarginalizesStoredBandwidth)
{
    auto set = make();
    set.setBandwidth(bandwidth3);
    auto const marginal = set.marginal<Horizontal>();
    auto const expected = Eigen::Matrix2d{{0.30, 0.05}, {0.05, 0.20}};
    ASSERT_TRUE(marginal.bandwidth().has_value());
    EXPECT_EQ(marginal.bandwidth().value(), expected);

    auto const copy = marginal.copy();
    static_assert(std::is_same_v<decltype(copy), ParticleSet<Horizontal> const>);
    ASSERT_TRUE(copy.bandwidth().has_value());
    EXPECT_EQ(copy.bandwidth().value(), expected);
}

// Setting the states or the bandwidth while a marginal is alive is allowed, and the marginal
// sees the new values.
TEST(ParticleSetMarginal, StatesMayChangeUnderAMarginal)
{
    auto set = make();
    auto const marginal = set.marginal<Z>();
    set.setStates(Eigen::Matrix<double, 3, 9>::Zero());
    set.setBandwidth(bandwidth3);
    EXPECT_EQ(Eigen::MatrixXd{marginal.states()}, Eigen::MatrixXd::Zero(1, 9));
    EXPECT_TRUE(marginal.bandwidth().has_value());
}

// kdeBandwidth() and the kernel density use a stored bandwidth when there is one and Silverman's
// rule otherwise; resampling and clearBandwidth() remove a stored bandwidth, and resizing
// the particle dimension drops it.
TEST(ParticleSetBandwidth, StoredOrSilverman)
{
    auto set = make();
    auto const silverman = Emrys::silvermanBandwidth(set).value();
    EXPECT_EQ(set.kdeBandwidth(), silverman);
    EXPECT_EQ(set.toDensity()->bandwidth(), silverman);

    set.setBandwidth(bandwidth3);
    EXPECT_EQ(set.kdeBandwidth(), bandwidth3);
    EXPECT_EQ(set.toDensity()->bandwidth(), bandwidth3);

    auto rng = std::mt19937_64{1};
    ASSERT_TRUE(set.resample(rng).has_value());
    EXPECT_FALSE(set.bandwidth().has_value());  // cleared by resampling

    set.setBandwidth(bandwidth3);
    set.clearBandwidth();
    EXPECT_FALSE(set.bandwidth().has_value());

    auto dynamic = ParticleSet<Linear<Eigen::Dynamic>>{Eigen::MatrixXd::Random(2, 5)};
    dynamic.setBandwidth(Eigen::Matrix2d::Identity());
    dynamic.resizeParticles(3);
    EXPECT_FALSE(dynamic.bandwidth().has_value());
}

// toDensity() reports an empty set as a weight error and coincident particles, whose Silverman
// bandwidth is singular, as a not-positive-definite Gaussian error.
TEST(ParticleSetBandwidth, ToNumericErrors)
{
    auto const empty = Particles3{};
    EXPECT_EQ(empty.toDensity().error(), WeightError::Empty);
    auto const collapsed = Particles3{Eigen::Matrix<double, 3, 4>::Ones()};
    EXPECT_EQ(collapsed.toDensity().error(), GaussianError::NotPositiveDefinite);
}

// conditional<Path>(v) gives a particle set over the remaining blocks whose kernel density is the
// joint kernel density divided by the marginal's at v, with the total weight scaled by that
// marginal density.
TEST(KernelDensityConditional, IsJointOverMarginal)
{
    auto set = make();
    set.setBandwidth(bandwidth3);
    auto const v = Eigen::Matrix<double, 1, 1>{0.35};
    using Given = TagPath<Horizontal, Y>;

    auto conditional = set.conditional<Given>(v);
    static_assert(std::is_same_v<decltype(conditional), std::expected<ParticleSet<Composite<X, Z>>, GaussianError>>);
    ASSERT_TRUE(conditional);
    ASSERT_TRUE(conditional->bandwidth().has_value());
    auto const conditionalKde = conditional->toDensity();
    ASSERT_TRUE(conditionalKde.has_value());

    auto const joint = set.toDensity();
    auto const marginalOfGiven = set.marginal<Given>().copy();
    auto const marginalKde = marginalOfGiven.toDensity();
    for (Eigen::Vector2d const & a :
         {Eigen::Vector2d{0.1, -0.2}, Eigen::Vector2d{-0.5, 0.6}, Eigen::Vector2d{0.9, 0.0}})
    {
        auto const x = Eigen::Vector3d{a(0), v(0), a(1)};
        EXPECT_NEAR(conditionalKde->logDensity(a), joint->logDensity(x) - marginalKde->logDensity(v), 1e-10);
    }
    EXPECT_NEAR(
        conditional->logTotalWeight().value(),
        set.logTotalWeight().value() + marginalKde->logDensity(v),
        1e-10
    );
}

// Without a stored bandwidth, conditioning uses the Silverman bandwidth, the same as storing it
// first, and fails when the weights cannot be normalized to give one.
TEST(KernelDensityConditional, SilvermanWhenNoBandwidthStored)
{
    auto const set = make();
    auto const v = Eigen::Matrix<double, 1, 1>{0.0};
    auto const conditional = set.conditional<TagPath<Horizontal, X>>(v);
    auto withSilverman = make();
    withSilverman.setBandwidth(set.kdeBandwidth().value());
    ASSERT_TRUE(conditional);
    EXPECT_EQ(conditional->states(), (withSilverman.conditional<TagPath<Horizontal, X>>(v).value().states()));

    auto unweighted = make();
    unweighted.scaleWeights(-std::numeric_limits<double>::infinity());
    EXPECT_EQ((unweighted.conditional<TagPath<Horizontal, X>>(v).error()), GaussianError::NonFinite);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// Reading a marginal after its set gained, lost, or resampled particles, or was moved from, dies
// [view-stamp].
TEST(ParticleSetMarginalDeathTest, ReadAfterStructuralChange)
{
    auto set = make();
    auto const marginal = set.marginal<Horizontal>();
    auto rng = std::mt19937_64{1};
    std::string const message = "changed size or structure while a marginal";
    EXPECT_DEATH(
        {
            set.add(0.0, Eigen::Vector3d::Zero());
            std::ignore = marginal.states();
        },
        message
    );
    EXPECT_DEATH(
        {
            set.remove(0);
            std::ignore = marginal.states();
        },
        message
    );
    EXPECT_DEATH(
        {
            std::ignore = set.resample(rng);
            std::ignore = marginal.logWeights();
        },
        message
    );
    EXPECT_DEATH(
        {
            set.clear();
            std::ignore = marginal.size();
        },
        message
    );
    EXPECT_DEATH(
        {
            auto stolen = std::move(set);
            std::ignore = marginal.states();
        },
        message
    );
}

#endif

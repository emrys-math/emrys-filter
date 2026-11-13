// A Gaussian over a non-linear space (step 5a; [discrete-spaces]): the mean is normalized, residuals take the
// short way around a periodic coordinate, samples stay in range, tag marginals and conditionals
// keep the kind of block and wrap the given residual, and resizable and capped spaces give dynamic
// and fixed-capacity storage.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

#include "Distributions/draw_stream.h"

using Emrys::allowsValueDimension;
using Emrys::Composite;
using Emrys::Degrees360;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::TagPath;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Heading : RadiansPi<>
{};

struct Range : Linear<1>
{};

using State = Composite<Position, Heading, Range>;
using StateGaussian = GaussianParams<State>;

static_assert(std::is_same_v<GaussianParams<Linear<4>>::SpaceType, Linear<4>>);
static_assert(StateGaussian::ValueDimensionAtCompileTime == 4 && StateGaussian::MaxValueDimensionAtCompileTime == 4);
static_assert(std::is_same_v<StateGaussian::VectorType, Eigen::Vector4d>);
static_assert(std::is_nothrow_move_constructible_v<StateGaussian>);

Eigen::Matrix4d
makeCov()
{
    return Eigen::Matrix4d{{1.0, 0.2, 0.1, 0.0}, {0.2, 2.0, -0.1, 0.1}, {0.1, -0.1, 0.04, 0.0}, {0.0, 0.1, 0.0, 0.5}};
}

}  // namespace

// Construction and setMean() wrap a periodic coordinate and leave a linear one as given.
TEST(GaussianSpace, MeanIsNormalized)
{
    auto g = StateGaussian{State{}, Eigen::Vector4d{1.0, 2.0, 3.5, -2.0}, makeCov()};
    EXPECT_NEAR(g.mean()(2), 3.5 - 2 * pi, 1e-15);
    EXPECT_EQ(g.mean()(3), -2.0);

    g.setMean(Eigen::Vector4d{0.0, 0.0, -4.0, -1.0});
    EXPECT_NEAR(g.mean()(2), -4.0 + 2 * pi, 1e-15);
    EXPECT_EQ(g.mean()(3), -1.0);
}

// The density at a point just across the seam equals a linear Gaussian's at the unwrapped point:
// residuals are the short way around.
TEST(GaussianSpace, ResidualsTakeTheShortWayAround)
{
    auto const g = StateGaussian{State{}, Eigen::Vector4d{0.0, 0.0, pi - 0.1, 3.0}, makeCov()};
    auto const linear = GaussianParams<Linear<4>>{Eigen::Vector4d{0.0, 0.0, pi - 0.1, 3.0}, makeCov()};
    // -π + 0.1 is 0.2 past the mean across the wrap, not 2π - 0.2 behind it.
    auto const across = Eigen::Vector4d{0.3, -0.2, -pi + 0.1, 3.4};
    auto const unwrapped = Eigen::Vector4d{0.3, -0.2, pi + 0.1, 3.4};
    EXPECT_NEAR(g.toDensity()->logDensity(across), linear.toDensity()->logDensity(unwrapped), 1e-10);
}

// Samples of a Gaussian near the seam stay in the period, some having wrapped, and carry the
// density's own log-density, in a batch and one drawSampleWithLogDensity() at a time, which off a
// linear space is exactly logDensity() of the stored draw.
TEST(GaussianSpace, SamplesStayInRange)
{
    auto const g = StateGaussian{State{}, Eigen::Vector4d{0.0, 0.0, pi - 0.05, 0.1}, makeCov()};
    auto const view = g.toDensity();
    auto rng = std::mt19937_64{3};
    auto const samples = Emrys::drawSamples(*view, 2000, rng);
    EXPECT_TRUE((samples.row(2).array() >= -pi).all() && (samples.row(2).array() < pi).all());
    EXPECT_GT((samples.row(2).array() < 0.0).count(), 100);  // some wrapped past π

    auto again = std::mt19937_64{3};
    auto const drawn = Emrys::drawSamplesWithLogDensity(*view, 50, again);
    for (Eigen::Index j = 0; j < 50; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), view->logDensity(drawn.samples.col(j)), 1e-12) << "sample " << j;
    }
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(*view, 50, 3, 0.0);
}

// A marginal of a Degrees360 block is over that tag's space, so a copy of it wraps at 360.
TEST(GaussianSpace, TagMarginalKeepsTheSubspace)
{
    struct Compass : Degrees360<>
    {};

    using Navigation = Composite<Position, Compass, Range>;
    auto const g = GaussianParams<Navigation>{Navigation{}, Eigen::Vector4d{1.0, 2.0, 359.0, 3.0}, makeCov()};
    auto const heading = g.marginal<Compass>();
    static_assert(std::is_same_v<decltype(heading)::SpaceType, Compass>);
    static_assert(Compass::period() == 360.0);
    EXPECT_EQ(heading.mean()(0), 359.0);

    auto copy = heading.copy();
    static_assert(std::is_same_v<decltype(copy), GaussianParams<Compass>>);
    copy.setMean(Eigen::Matrix<double, 1, 1>{361.0});
    EXPECT_EQ(copy.mean()(0), 1.0);

    auto const position = g.marginal<Position>().copy();
    static_assert(std::is_same_v<decltype(position), GaussianParams<Position> const>);
    EXPECT_EQ(position.mean(), Eigen::Vector2d(1.0, 2.0));
}

// A path through a capped Repeated block gives a Repeated of the leaf with the same cap and bound.
TEST(GaussianSpace, TagMarginalThroughACappedRepeatedBlockKeepsTheCap)
{
    struct Altitude : Linear<1>
    {};

    struct Vehicle : Composite<Linear<2>, Altitude>
    {};

    struct Fleet : Repeated<Vehicle, 3>
    {};

    using World = Composite<Linear<1>, Fleet>;
    auto const g =
        GaussianParams<World>{World{}, Eigen::VectorXd::LinSpaced(7, 0.0, 6.0), Eigen::MatrixXd::Identity(7, 7)};
    auto const altitudes = g.marginal<TagPath<Fleet, Altitude>>().copy();
    static_assert(std::is_same_v<decltype(altitudes)::SpaceType, Repeated<Altitude, 3>>);
    static_assert(decltype(altitudes)::MaxValueDimensionAtCompileTime == 3);
    EXPECT_EQ(altitudes.mean(), Eigen::Vector2d(3.0, 6.0));
}

// Two blocks of the same space type are told apart by tag, and the marginal's copy wraps as its
// block does.
TEST(GaussianSpace, TagMarginalKeepsTheKindOfBlock)
{
    struct Near : Degrees360<2>
    {};

    struct Far : Degrees360<2>
    {};

    using Bearings = Composite<Near, Far>;
    auto const bearings = GaussianParams<Bearings>{Bearings{}, Eigen::Vector4d{10.0, 20.0, 30.0, 40.0}, makeCov()};
    auto copy = bearings.marginal<Far>().copy();
    static_assert(std::is_same_v<decltype(copy), GaussianParams<Far>>);
    EXPECT_EQ(copy.mean(), Eigen::Vector2d(30.0, 40.0));
    copy.setMean(Eigen::Vector2d{-1.0, 361.0});
    EXPECT_EQ(copy.mean(), Eigen::Vector2d(359.0, 1.0));
}

// marginal<Fleet>(i) is a Gaussian over the element space with that repetition's values, wrapped,
// and a path into a repetition gives the leaf space, through the view too. mean<Fleet>(i) and
// covariance<Fleet>(i) read one repetition's block as stored, wrapped when the Gaussian was made,
// and the view reads the same, its cross-covariances too.
TEST(GaussianSpace, MarginalOfOneRepetitionIsInTheBlockSpace)
{
    struct Altitude : Linear<1>
    {};

    struct Vehicle : Composite<Linear<2>, Heading, Altitude>
    {};

    struct Fleet : Repeated<Vehicle, 3>
    {};

    using World = Composite<Linear<1>, Fleet>;
    auto const g =
        GaussianParams<World>{World{}, Eigen::VectorXd::LinSpaced(9, 0.0, 8.0), Eigen::MatrixXd::Identity(9, 9)};

    auto const second = g.marginal<Fleet>(1).copy();
    static_assert(std::is_same_v<decltype(second), GaussianParams<Vehicle> const>);
    EXPECT_EQ(second.mean(), Eigen::Vector4d(5.0, 6.0, 7.0 - 2 * pi, 8.0));

    auto const heading = g.marginal<TagPath<Fleet, Heading>>(0).copy();
    static_assert(std::is_same_v<decltype(heading), GaussianParams<Heading> const>);
    EXPECT_NEAR(heading.mean()(0), 3.0, 1e-15);

    auto const density = g.toDensity();
    ASSERT_TRUE(density.has_value());
    static_assert(std::is_same_v<decltype(density->marginal<Fleet>(0))::SpaceType, Vehicle>);
    EXPECT_EQ(density->marginal<Fleet>(1).copy().mean(), second.mean());

    Eigen::VectorXd const secondMean = g.mean<Fleet>(1);
    EXPECT_EQ(secondMean, Eigen::VectorXd{g.mean().segment<4>(5)});
    EXPECT_EQ(secondMean, second.mean());
    Eigen::MatrixXd const secondCov = g.covariance<Fleet>(1);
    EXPECT_EQ(secondCov, Eigen::Matrix4d::Identity());
    EXPECT_EQ(Eigen::VectorXd{density->mean<Fleet>(1)}, secondMean);
    EXPECT_EQ(Eigen::MatrixXd{density->covariance<Fleet>(1)}, secondCov);
    EXPECT_EQ(Eigen::VectorXd{density->mean<Fleet>()}, Eigen::VectorXd{g.mean().tail<8>()});
    EXPECT_EQ(Eigen::MatrixXd{density->covariance<Fleet>()}, Eigen::MatrixXd::Identity(8, 8));
    EXPECT_EQ(Eigen::MatrixXd{(density->covariance<Fleet, Fleet>())}, Eigen::MatrixXd::Identity(8, 8));
}

// mean<Path>() and covariance<Path>() select blocks with compile-time sizes, covariance<A, B>() reads the lower
// triangle whichever block comes first, and the view and the moments offer the same.
TEST(GaussianSpace, MeanAndCovTakeAPath)
{
    Eigen::Matrix4d lower = makeCov();
    lower.triangularView<Eigen::StrictlyUpper>().setConstant(99.0);  // only the lower triangle is read
    auto const g = StateGaussian{State{}, Eigen::Vector4d{1.0, 2.0, 0.5, 3.0}, lower};

    auto const position = g.mean<Position>();
    static_assert(decltype(position)::RowsAtCompileTime == 2);
    EXPECT_EQ(Eigen::Vector2d{position}, Eigen::Vector2d(1.0, 2.0));
    EXPECT_EQ(g.mean<Heading>(), 0.5);

    Eigen::Matrix2d const positionCov = g.covariance<Position>();
    Eigen::Matrix2d const expected = makeCov().topLeftCorner<2, 2>();
    EXPECT_EQ(positionCov, expected);
    EXPECT_EQ(g.covariance<Range>(), 0.5);

    // The cross-covariance reads the lower triangle whichever block comes first.
    auto const below = g.covariance<Range, Position>();
    static_assert(decltype(below)::RowsAtCompileTime == 1 && decltype(below)::ColsAtCompileTime == 2);
    EXPECT_EQ(Eigen::RowVector2d{below}, Eigen::RowVector2d(0.0, 0.1));
    auto const above = g.covariance<Position, Range>();
    static_assert(decltype(above)::RowsAtCompileTime == 2 && decltype(above)::ColsAtCompileTime == 1);
    EXPECT_EQ(Eigen::Vector2d{above}, Eigen::Vector2d(0.0, 0.1));
    EXPECT_EQ((g.covariance<Heading, Position>()), (makeCov().block<1, 2>(2, 0)));
    EXPECT_EQ((g.covariance<Range, Heading>()), 0.0);
    EXPECT_EQ((g.covariance<Heading, Range>()), 0.0);

    auto const density = g.toDensity();
    ASSERT_TRUE(density.has_value());
    EXPECT_EQ(Eigen::Matrix2d{density->covariance<Position>()}, positionCov);
    EXPECT_EQ(Eigen::Vector2d{g.moments().mean<Position>()}, Eigen::Vector2d(1.0, 2.0));
}

// Conditioning on a nested path leaves the sibling blocks and the other top-level blocks, and the
// given value is a residual in its own space, so 2 pi past the mean is the mean.
TEST(GaussianSpace, ConditionalOnANestedPathKeepsTheRest)
{
    struct Ownship : Composite<Position, Heading>
    {};

    struct Target : Composite<Linear<2>, Range>
    {};

    using Scene = Composite<Ownship, Target>;
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(2, 3) = cov(3, 2) = 0.5;  // the ownship heading and the target's first coordinate
    auto const g = GaussianParams<Scene>{Scene{}, Eigen::Vector<double, 6>{1.0, 2.0, 0.5, 4.0, 5.0, 6.0}, cov};

    // Ownship minus its position is just the heading, which sits beside the target.
    auto const rest = g.conditional<TagPath<Ownship, Position>>(Eigen::Vector2d{0.0, 0.0});
    static_assert(std::is_same_v<decltype(rest)::SpaceType, Composite<Heading, Target>>);
    EXPECT_EQ(rest.mean(), (Eigen::Vector4d{0.5, 4.0, 5.0, 6.0}));

    // The given heading is a residual in its own space: 2π past the mean is the mean.
    auto const target = g.conditional<TagPath<Ownship, Heading>>(Eigen::Matrix<double, 1, 1>{0.5 + 2 * pi});
    static_assert(std::is_same_v<decltype(target)::SpaceType, Composite<Position, Target>>);
    EXPECT_NEAR(target.mean()(2), 4.0, 1e-12);
}

// A given heading across the seam conditions as the linear Gaussian would with the unwrapped value,
// and a top-level given block leaves the remaining named blocks.
TEST(GaussianSpace, ConditionalWrapsTheGivenResidual)
{
    auto const mean = Eigen::Vector4d{1.0, 2.0, -pi + 0.1, 3.0};
    auto const g = StateGaussian{State{}, mean, makeCov()};

    struct FlatHeading : Linear<1>
    {};

    struct FlatRange : Linear<1>
    {};

    using Flat = Composite<Position, FlatHeading, FlatRange>;
    auto const linear = GaussianParams<Flat>{Flat{}, mean, makeCov()};
    // Given heading π - 0.1 is 0.2 behind the mean across the wrap.
    auto const given = Eigen::Matrix<double, 1, 1>{pi - 0.1};
    auto const unwrapped = Eigen::Matrix<double, 1, 1>{-pi - 0.1};
    auto const conditional = g.conditional<Heading>(given);
    auto const expected = linear.conditional<FlatHeading>(unwrapped);
    EXPECT_LT((conditional.mean() - expected.mean()).norm(), 1e-12);
    EXPECT_LT((conditional.covarianceLower() - expected.covarianceLower()).norm(), 1e-12);
    // Heading is a top-level block, so the rest of the state keeps its named blocks.
    static_assert(std::is_same_v<decltype(conditional)::SpaceType, Composite<Position, Range>>);
}

// A conditional mean pushed past pi by correlation is wrapped.
TEST(GaussianSpace, ConditionalNormalizesTheTargetMean)
{
    // Strong position-heading correlation pushes the conditional heading past π.
    struct Offset : Linear<1>
    {};

    struct Angle : RadiansPi<>
    {};

    auto const cov = Eigen::Matrix2d{{1.0, 0.9}, {0.9, 1.0}};
    auto const g = GaussianParams<Composite<Offset, Angle>>{{}, Eigen::Vector2d{0.0, 3.0}, cov};
    auto const conditional = g.conditional<Offset>(Eigen::Matrix<double, 1, 1>{1.0});
    static_assert(std::is_same_v<decltype(conditional)::SpaceType, Angle>);
    EXPECT_NEAR(conditional.mean()(0), 3.0 + 0.9 - 2 * pi, 1e-12);
}

// A Gaussian over a space with an uncapped Repeated block is constructed and resized by size and
// wraps every repetition's angle.
TEST(GaussianSpace, ResizableSpace)
{
    struct Vehicle : Composite<Linear<2>, RadiansPi<>>
    {};

    using World = Composite<Linear<1>, Repeated<Vehicle>>;
    auto g = GaussianParams<World>{World{}, Eigen::Index{7}};
    EXPECT_EQ(g.valueDimension(), 7);
    g.resize(10);
    EXPECT_EQ(g.valueDimension(), 10);
    g.setMean(Eigen::VectorXd::Constant(10, 4.0));
    EXPECT_NEAR(g.mean()(3), 4.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(g.mean()(9), 4.0 - 2 * pi, 1e-15);
    EXPECT_EQ(g.mean()(8), 4.0);
}

// A capped Repeated gives a dynamic Gaussian with a compile-time bound and fixed-capacity Eigen
// storage.
TEST(GaussianSpace, CappedSpaceGivesFixedCapacityStorage)
{
    struct Vehicle : Composite<Linear<2>, RadiansPi<>>
    {};

    using Fleet = Composite<Linear<1>, Repeated<Vehicle, 3>>;
    using G = GaussianParams<Fleet>;
    static_assert(G::ValueDimensionAtCompileTime == Eigen::Dynamic && G::MaxValueDimensionAtCompileTime == 10);
    static_assert(G::VectorType::MaxRowsAtCompileTime == 10);
    auto g = G{Fleet{}, Eigen::Index{10}};
    EXPECT_EQ(g.valueDimension(), 10);
    static_assert(std::is_same_v<GaussianParams<Linear<Eigen::Dynamic, 6>>::SpaceType, Linear<Eigen::Dynamic, 6>>);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// A size that is not a whole number of repetitions, or over the cap, dies with "fit the space", and
// a space whose resize() does not reach the size is caught.
TEST(GaussianSpaceDeathTest, SizeMustFitTheSpace)
{
    struct Vehicle : Composite<Linear<2>, RadiansPi<>>
    {};

    using World = Composite<Linear<1>, Repeated<Vehicle>>;
    auto g = GaussianParams<World>{World{}, Eigen::Index{4}};
    EXPECT_DEATH(g.resize(5), "fit the space");
    EXPECT_DEATH((GaussianParams<World>{World{}, Eigen::Index{3}}), "fit the space");

    using Capped = Composite<Linear<1>, Repeated<Vehicle, 2>>;
    auto capped = GaussianParams<Capped>{Capped{}, Eigen::Index{7}};
    EXPECT_DEATH(capped.resize(10), "fit the space");

    // Valid for the type, but a user's resize() that doesn't reach the size is caught too.
    struct Stuck : Linear<Eigen::Dynamic>
    {
        void
        resize(Eigen::Index)
        {}
    };

    static_assert(allowsValueDimension<Stuck>(3));
    EXPECT_DEATH((GaussianParams<Stuck>{Stuck{}, Eigen::Index{3}}), "did not give it this dimension");
}

#endif

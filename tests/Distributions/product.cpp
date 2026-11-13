// Product (step 12): a product of independent factors over the blocks of a Composite adds
// their log-densities, samples each factor's rows, in a batch or one draw at a time [allocation], has
// block-diagonal moments (which fail with a Student-t factor below two degrees of freedom), accepts
// its factors in any order and by value, takes every form of the Gaussian [form-densities], and has marginals
// and conditionals that are the factors' own. A factor placed by onEach() covers every repetition of a
// repeated block, and Factors, the factors placed by path in no space, is placed in each space it is given
// ([product-factors]).

#include <EmrysFilter/Distributions/Gamma.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/KernelDensity.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Distributions/StudentT.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Distributions/Uniform.h>

#include <gtest/gtest.h>

#include <cmath>
#include <expected>
#include <numbers>
#include <random>
#include <type_traits>

#include "Distributions/draw_stream.h"

using Emrys::Composite;
using Emrys::Gamma;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::MomentsError;
using Emrys::on;
using Emrys::Product;
using Emrys::RadiansPi;
using Emrys::TagPath;
using Emrys::Uniform;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Heading : RadiansPi<>
{};

struct Speed : Linear<1>
{};

using State = Composite<Position, Heading, Speed>;

GaussianParams<Position>
makePosition()
{
    return GaussianParams<Position>{Eigen::Vector2d{10.0, -5.0}, Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}}};
}

}  // namespace

// A product of a Gaussian, a uniform, and a gamma on the three blocks deduces its type, is not an
// UncertainInput, keeps its space and each factor's rows, and has the sum of its factors'
// log-densities, minus infinity where a factor is, and its exponential as the density.
TEST(Product, LogDensityAddsTheFactors)
{
    auto const position = makePosition();
    auto const positionDensity = position.toDensity();
    ASSERT_TRUE(positionDensity.has_value());
    auto const heading = Uniform<Heading>{};
    auto const speed = Gamma<Speed>::create(4.0, 0.5).value();
    auto const prior = Product{State{}, on<Position>(*positionDensity), on<Heading>(heading), on<Speed>(speed)};
    static_assert(std::is_same_v<
                  std::remove_const_t<decltype(prior)>,
                  Product<State, double, Emrys::Gaussian<Position>, Uniform<Heading>, Gamma<Speed>>>);
    static_assert(!Emrys::UncertainInput<decltype(prior)>);
    EXPECT_EQ(prior.valueDimension(), 4);
    EXPECT_EQ(prior.space(), State{});
    EXPECT_EQ(Eigen::VectorX<Eigen::Index>{prior.indices<2>()}, (Eigen::Matrix<Eigen::Index, 1, 1>{3}));

    auto const x = Eigen::Vector4d{11.0, -4.0, 2.0, 1.5};
    double const expected = positionDensity->logDensity(x.head<2>()) - std::log(2 * pi)
                          + speed.logDensity(Eigen::Matrix<double, 1, 1>{1.5});
    EXPECT_NEAR(prior.logDensity(x), expected, 1e-13);
    EXPECT_NEAR(prior.density(x), std::exp(expected), 1e-15);
    EXPECT_EQ(prior.logDensity(Eigen::Vector4d{11.0, -4.0, 2.0, -1.0}), -std::numeric_limits<double>::infinity());
}

// Samples carry the product's log-density, each factor's rows obey that factor's support, and the
// sample moments show each factor's mean, the Gaussian's correlation, and independence across
// factors.
TEST(Product, SamplesFillEachFactorsRows)
{
    auto const position = makePosition();
    auto const positionDensity = position.toDensity();
    ASSERT_TRUE(positionDensity.has_value());
    auto const prior = Product{
        State{},
        on<Position>(*positionDensity),
        on<Heading>(Uniform<Heading>{}),
        on<Speed>(Gamma<Speed>::create(4.0, 0.5).value())
    };

    auto rng = std::mt19937_64{17};
    auto const drawn = Emrys::drawSamplesWithLogDensity(prior, 50000, rng);
    ASSERT_EQ(drawn.samples.rows(), 4);
    for (Eigen::Index j = 0; j < 20; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), prior.logDensity(drawn.samples.col(j)), 1e-10);
    }
    EXPECT_TRUE((drawn.samples.row(2).array() >= -pi).all() && (drawn.samples.row(2).array() < pi).all());
    EXPECT_TRUE((drawn.samples.row(3).array() >= 0).all());

    // Each factor's rows have that factor's mean: the Gaussian's, zero for the uniform angle, and
    // shape times scale = 2 for the gamma speed. The tolerances are a few standard errors at this count.
    Eigen::Vector4d const mean = drawn.samples.rowwise().mean();
    EXPECT_NEAR(mean(0), 10.0, 0.05);
    EXPECT_NEAR(mean(1), -5.0, 0.05);
    EXPECT_NEAR(mean(2), 0.0, 0.05);
    EXPECT_NEAR(mean(3), 2.0, 0.02);
    Eigen::Matrix<double, 4, Eigen::Dynamic> const centered = drawn.samples.colwise() - mean;
    Eigen::Matrix4d const cov = centered * centered.transpose() / double(centered.cols());
    EXPECT_NEAR(cov(0, 1), 1.0, 0.1);
    EXPECT_NEAR(cov(0, 3), 0.0, 0.05);  // independent factors
}

// moments() stacks the factors' means and places their covariances on the block diagonal (the
// uniform angle's is (2 pi)^2 / 12).
TEST(Product, MomentsAreBlockDiagonal)
{
    auto const position = makePosition();
    auto const positionDensity = position.toDensity();
    ASSERT_TRUE(positionDensity.has_value());
    auto const prior = Product{
        State{},
        on<Position>(*positionDensity),
        on<Heading>(Uniform<Heading>{}),
        on<Speed>(Gamma<Speed>::create(4.0, 0.5).value())
    };
    auto const moments = prior.moments();
    static_assert(std::is_same_v<std::remove_const_t<decltype(moments)>, Emrys::Moments<State>>);
    EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector4d{10.0, -5.0, 0.0, 2.0}));
    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    expected.topLeftCorner<2, 2>() = Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}};
    expected(2, 2) = 4 * pi * pi / 12;
    expected(3, 3) = 1.0;
    EXPECT_TRUE(moments.covarianceLower().isApprox(expected));
}

// With a factor whose moments may not exist, moments() returns an expected: Infinite for a Cauchy
// factor and the finite value otherwise.
TEST(Product, MomentsCanFailWithAStudentTFactor)
{
    using Plane = Composite<Position, Heading>;
    auto const cauchy = Emrys::StudentT<Position>::create(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity(), 1.0);
    auto const t = Emrys::StudentT<Position>::create(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity(), 4.0);
    ASSERT_TRUE(cauchy && t);
    auto const heavy = Product{Plane{}, on<Position>(*cauchy), on<Heading>(Uniform<Heading>{})};
    auto const finite = Product{Plane{}, on<Position>(*t), on<Heading>(Uniform<Heading>{})};
    static_assert(std::is_same_v<decltype(heavy.moments()), std::expected<Emrys::Moments<Plane>, MomentsError>>);
    EXPECT_EQ(heavy.moments().error(), MomentsError::Infinite);
    ASSERT_TRUE(finite.moments().has_value());
    EXPECT_NEAR(finite.moments()->covarianceLower()(0, 0), 2.0, 1e-14);
}

// Factors listed out of block order still land on their blocks: density, support of the samples,
// and moments follow the space's coordinate order.
TEST(Product, FactorsMayBeListedInAnyOrder)
{
    // A uniform on the box, coordinates 1 and 2, listed before an exponential on the lead, coordinate 0.
    struct Lead : Linear<1>
    {};

    struct Box : Linear<2>
    {};

    using Space = Composite<Lead, Box>;
    auto const box =
        Emrys::Uniform<Emrys::Linear<2>>::create(Eigen::Vector2d{0.0, -1.0}, Eigen::Vector2d{4.0, 1.0}).value();
    auto const exponential = Emrys::Gamma<Emrys::Linear<1>>::exponential(2.0).value();
    auto const product = Product{Space{}, on<Box>(box), on<Lead>(exponential)};

    auto const x = Eigen::Vector3d{3.0, 2.0, 0.5};
    EXPECT_NEAR(product.logDensity(x), -std::log(8.0) - std::log(2.0) - 1.5, 1e-14);
    EXPECT_EQ(product.logDensity(Eigen::Vector3d{3.0, 0.5, 2.0}), -std::numeric_limits<double>::infinity());

    auto rng = std::mt19937_64{23};
    auto const samples = Emrys::drawSamples(product, 1000, rng);
    EXPECT_TRUE((samples.row(1).array() >= 0.0).all() && (samples.row(1).array() < 4.0).all());
    EXPECT_TRUE((samples.row(2).array() >= -1.0).all() && (samples.row(2).array() < 1.0).all());
    EXPECT_TRUE((samples.row(0).array() >= 0.0).all());

    auto const moments = product.moments();
    EXPECT_TRUE(moments.mean().isApprox(Eigen::Vector3d{2.0, 2.0, 0.0}));
    EXPECT_NEAR(moments.covarianceLower()(1, 1), 16.0 / 12, 1e-14);
    EXPECT_NEAR(moments.covarianceLower()(2, 2), 4.0 / 12, 1e-14);
    EXPECT_NEAR(moments.covarianceLower()(0, 0), 4.0, 1e-14);

    // Single draws, one drawSample() at a time, take each factor's in its rows and have the
    // product's moments; with their log-densities they are the same draws, each with the sum of the
    // factors' log-densities, which is the product's to rounding.
    Eigen::MatrixXd const draws = EmrysTest::streamOfDraws(product, 40'000, rng);
    EXPECT_TRUE((draws.row(1).array() >= 0.0).all() && (draws.row(1).array() < 4.0).all());
    EXPECT_TRUE((draws.row(2).array() >= -1.0).all() && (draws.row(2).array() < 1.0).all());
    EXPECT_TRUE((draws.row(0).array() >= 0.0).all());
    EmrysTest::expectSampleMoments(draws, moments.mean(), moments.covarianceLower(), 0.05);
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(product, 50, 47, 1e-12);
}

// A KernelDensity view can be a factor; the density and the samples' log-densities include it.
TEST(Product, TakesAKernelDensityFactor)
{
    using Plane = Composite<Position, Heading>;
    auto particles =
        Emrys::ParticleSet<Position>{Position{}, Eigen::Matrix<double, 2, 3>{{0.0, 1.0, 2.0}, {0.0, 1.0, 0.0}}};
    particles.setBandwidth(Eigen::Matrix2d::Identity() * 0.1);
    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde.has_value());
    auto const product = Product{Plane{}, on<Position>(*kde), on<Heading>(Uniform<Heading>{})};
    auto const x = Eigen::Vector3d{1.0, 0.5, 1.0};
    EXPECT_NEAR(product.logDensity(x), kde->logDensity(x.head<2>()) - std::log(2 * pi), 1e-13);
    auto rng = std::mt19937_64{29};
    auto const drawn = Emrys::drawSamplesWithLogDensity(product, 50, rng);
    for (Eigen::Index j = 0; j < 10; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), product.logDensity(drawn.samples.col(j)), 1e-10);
    }
}

// Every form of the Gaussian is a factor as parameters, which the product checks with
// toDensity(): its log-density, moments, and draws' log-densities match the product with the
// covariance form to rounding [form-densities]. Moments that can fail in the parameters can't in their
// density, so the product's moments are plain.
TEST(Product, TakesEveryFormOfTheGaussian)
{
    static_assert(Emrys::ProductFactor<Emrys::SqrtGaussianParams<Position>>);
    static_assert(Emrys::ProductFactor<Emrys::InfoGaussianParams<Position>>);
    static_assert(Emrys::ProductFactor<Emrys::SqrtInfoGaussianParams<Position>>);
    static_assert(!Emrys::ProductFactor<Emrys::Moments<Position>>);

    auto const position = makePosition();
    auto const root = Emrys::toSqrt(position);
    auto const info = Emrys::toInfo(position);
    auto const rootInfo = Emrys::toSqrtInfo(position);
    ASSERT_TRUE(root && info && rootInfo);
    auto const speed = Gamma<Speed>::create(4.0, 0.5).value();
    auto const expected = Product{State{}, on<Position>(position), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)};
    auto const expectedMoments = expected.moments();
    auto const x = Eigen::Vector4d{11.0, -4.0, 2.0, 1.5};

    auto const check = [&](auto const & product) {
        EXPECT_NEAR(product.logDensity(x), expected.logDensity(x), 1e-12);
        auto const moments = product.moments();
        static_assert(std::is_same_v<std::remove_const_t<decltype(moments)>, Emrys::Moments<State>>);
        EXPECT_TRUE(moments.mean().isApprox(expectedMoments.mean(), 1e-13));
        EXPECT_TRUE(moments.covarianceLower().isApprox(expectedMoments.covarianceLower(), 1e-13));
        auto rng = std::mt19937_64{31};
        auto const drawn = Emrys::drawSamplesWithLogDensity(product, 20, rng);
        for (Eigen::Index j = 0; j < drawn.samples.cols(); ++j)
        {
            EXPECT_NEAR(drawn.logDensity(j), expected.logDensity(drawn.samples.col(j)), 1e-10);
        }
    };
    check(Product{State{}, on<Position>(*root), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)});
    check(Product{State{}, on<Position>(*info), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)});
    check(Product{State{}, on<Position>(*rootInfo), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)});
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
// Two factors on the same block die with "partition".
TEST(ProductDeathTest, FactorsMustPartitionTheSpace)
{
    struct Left : Linear<1>
    {};

    struct Right : Linear<1>
    {};

    auto const line =
        Uniform<Linear<1>>::create(Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}).value();
    EXPECT_DEATH((Product{Composite<Left, Right>{}, on<Left>(line), on<Left>(line)}), "partition");
}
#endif

// The internal fitsIn check accepts a factor over the tag, or the tag's own space, on the block it
// names and rejects one over another space.
TEST(Product, FactorSpacesMustBeTheBlocksTheyCover)
{
    using Emrys::internal::OnPath;
    using Emrys::internal::PlacedFactor;

    struct Offset : Linear<1>
    {};

    using Two = Composite<Heading, Offset>;
    static_assert(PlacedFactor<OnPath<Uniform<Heading>, Heading>>::fitsIn<Two>);
    static_assert(PlacedFactor<OnPath<Uniform<RadiansPi<>>, Heading>>::fitsIn<Two>);  // the tag's own space
    static_assert(PlacedFactor<OnPath<Uniform<Offset>, Offset>>::fitsIn<Two>);
    static_assert(!PlacedFactor<OnPath<Gamma<Speed>, Heading>>::fitsIn<Two>);
}

namespace
{

struct East : Linear<1>
{};

struct North : Linear<1>
{};

struct Place : Composite<East, North>
{
    using Composite::Composite;
};

struct Pose : Composite<Place, Heading>
{
    using Composite::Composite;
};

using Vehicle = Composite<Pose, Speed>;

GaussianParams<Place>
makePlace()
{
    return GaussianParams<Place>{Eigen::Vector2d{10.0, -5.0}, Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}}};
}

double
speedLogDensity(double speed)
{
    return Gamma<Speed>::create(4.0, 0.5).value().logDensity(Eigen::Matrix<double, 1, 1>{speed});
}

// A Gaussian place, a uniform heading, and a Gamma speed, with the Gaussian held by value.
auto
makeVehiclePrior()
{
    return Product{
        Vehicle{},
        on<Place>(makePlace()),
        on<Heading>(Uniform<Heading>{}),
        on<Speed>(Gamma<Speed>::create(4.0, 0.5).value())
    };
}

}  // namespace

// A GaussianParams passed to on<>() is stored by value and factorized when evaluated, so the
// product can be returned from a function; its density and samples match.
TEST(Product, HoldsParamsByValue)
{
    auto const prior = makeVehiclePrior();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(prior.factor<0>())>, GaussianParams<Place>>);
    auto const placeGaussian = makePlace();
    auto const place = placeGaussian.toDensity();
    ASSERT_TRUE(place.has_value());
    auto const x = Eigen::Vector4d{11.0, -4.0, 2.0, 1.5};
    EXPECT_NEAR(prior.logDensity(x), place->logDensity(x.head<2>()) - std::log(2 * pi) + speedLogDensity(1.5), 1e-13);
    auto rng = std::mt19937_64{31};
    auto const drawn = Emrys::drawSamplesWithLogDensity(prior, 20, rng);
    for (Eigen::Index j = 0; j < 20; ++j)
    {
        EXPECT_NEAR(drawn.logDensity(j), prior.logDensity(drawn.samples.col(j)), 1e-10);
    }
}

// marginal<Block>() of a whole factor returns a reference to that factor itself, also through a
// qualified path, rather than a copy [element-marginals].
TEST(Product, MarginalOfAFactorIsThatFactor)
{
    auto const prior = makeVehiclePrior();
    static_assert(std::is_same_v<decltype(prior.marginal<Place>()), GaussianParams<Place> const &>);
    auto const & place = prior.marginal<Place>();
    EXPECT_EQ(&place, &prior.factor<0>());
    EXPECT_TRUE(place.mean().isApprox(Eigen::Vector2d{10.0, -5.0}));
    static_assert(std::is_same_v<decltype(prior.marginal<TagPath<Pose, Heading>>()), Uniform<Heading> const &>);
}

// A marginal of part of a factor is that factor's own marginal: a view of a GaussianParams, which
// reads the factor in place, and copy() gives parameters [element-marginals].
TEST(Product, MarginalInsideAFactorIsTheFactorsMarginal)
{
    auto const prior = makeVehiclePrior();
    auto const north = prior.marginal<North>();
    static_assert(std::is_same_v<std::remove_const_t<decltype(north.copy())>, GaussianParams<North>>);
    EXPECT_NEAR(north.mean()(0), -5.0, 1e-15);
    EXPECT_NEAR(north.covarianceLower()(0, 0), 2.0, 1e-15);
    auto const east = prior.marginal<TagPath<Pose, Place, East>>();
    EXPECT_NEAR(east.mean()(0), 10.0, 1e-15);
}

// A marginal covering several whole factors is the product of those factors.
TEST(Product, MarginalOfWholeFactorsIsTheirProduct)
{
    auto const prior = makeVehiclePrior();
    auto const pose = prior.marginal<Pose>();
    static_assert(std::is_same_v<decltype(pose), Product<Pose, double, GaussianParams<Place>, Uniform<Heading>> const>);
    auto const x = Eigen::Vector3d{11.0, -4.0, 2.0};
    auto const place = prior.marginal<Place>();
    EXPECT_NEAR(pose.logDensity(x), place.toDensity()->logDensity(x.head<2>()) - std::log(2 * pi), 1e-13);
}

// Conditioning on a whole factor drops it, since the factors are independent; the result is the
// product of the rest or, with one left, that factor.
TEST(Product, ConditionalOnAWholeFactorDropsIt)
{
    auto const prior = makeVehiclePrior();
    auto const rest = prior.conditional<Speed>(Eigen::Matrix<double, 1, 1>{3.0});
    static_assert(std::is_same_v<decltype(rest), Product<Pose, double, GaussianParams<Place>, Uniform<Heading>> const>);
    auto const heading = prior.conditional<TagPath<Pose, Place>>(Eigen::Vector2d{1.0, 2.0});
    static_assert(std::is_same_v<
                  std::remove_const_t<decltype(heading)>,
                  Product<Composite<Heading, Speed>, double, Uniform<Heading>, Gamma<Speed>>>);
    EXPECT_NEAR(heading.logDensity(Eigen::Vector2d{0.3, 1.5}), -std::log(2 * pi) + speedLogDensity(1.5), 1e-13);
    auto const speed = prior.conditional<Pose>(Eigen::Vector3d{1.0, 2.0, 0.0});
    static_assert(std::is_same_v<decltype(speed), Gamma<Speed> const>);
}

// Conditioning on part of a Gaussian factor replaces it with the Gaussian's conditional, through an
// expected since factorizing it may fail.
TEST(Product, ConditionalInsideAFactorReplacesIt)
{
    auto const prior = makeVehiclePrior();
    auto const given = prior.conditional<East>(Eigen::Matrix<double, 1, 1>{12.0});
    ASSERT_TRUE(given.has_value());  // a Gaussian conditional must factorize, so this can fail
    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(*given)>,
            Product<Composite<North, Heading, Speed>, double, GaussianParams<North>, Uniform<Heading>, Gamma<Speed>>>
    );
    auto const northGaussian = makePlace().conditional<East>(Eigen::Matrix<double, 1, 1>{12.0});
    auto const north = northGaussian.toDensity();
    ASSERT_TRUE(north.has_value());
    auto const x = Eigen::Vector3d{-3.0, 0.5, 2.5};
    EXPECT_NEAR(given->logDensity(x), north->logDensity(x.head<1>()) - std::log(2 * pi) + speedLogDensity(2.5), 1e-13);
}

// Conditioning on part of a Student-t factor replaces it with the Student-t's conditional, with its
// raised degrees of freedom.
TEST(Product, ConditionalInsideAStudentTFactor)
{
    using Plane = Composite<Place, Heading>;
    auto const t =
        Emrys::StudentT<Place>::create(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.5}, {0.5, 1.0}}, 3.0);
    ASSERT_TRUE(t.has_value());
    auto const product = Product{Plane{}, on<Place>(*t), on<Heading>(Uniform<Heading>{})};
    auto const given = product.conditional<North>(Eigen::Matrix<double, 1, 1>{2.0});
    ASSERT_TRUE(given.has_value());
    auto const east = t->conditional<North>(Eigen::Matrix<double, 1, 1>{2.0});
    ASSERT_TRUE(east.has_value());
    EXPECT_EQ(given->factor<0>().dof(), 4.0);
    EXPECT_NEAR(
        given->logDensity(Eigen::Vector2d{1.0, 0.2}),
        east->logDensity(Eigen::Matrix<double, 1, 1>{1.0}) - std::log(2 * pi),
        1e-13
    );
}

namespace
{

struct Landmark : Linear<2>
{};

struct Landmarks : Emrys::Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Landmarks>;

GaussianParams<Landmark>
makeLandmark()
{
    return GaussianParams<Landmark>{Eigen::Vector2d{3.0, 1.0}, Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
}

}  // namespace

// onEach<Landmarks>() places one factor on every repetition the space has: the moments are the
// factor's mean repeated and I (x) R, the log-density is the factor's summed over the repetitions,
// and the marginal of the block is the EachRepetition, with the count of the space it was placed in.
TEST(Product, OnEachPlacesTheFactorOnEveryRepetition)
{
    auto const space = Map{Position{}, Landmarks{Landmark{}, 3}};
    auto const prior = Product{space, on<Position>(makePosition()), Emrys::onEach<Landmarks>(makeLandmark())};
    static_assert(
        std::is_same_v<
            std::remove_const_t<decltype(prior)>,
            Product<Map, double, GaussianParams<Position>, Emrys::EachRepetition<GaussianParams<Landmark>, Landmarks>>>
    );
    EXPECT_EQ(prior.valueDimension(), 8);
    EXPECT_EQ(prior.marginal<Landmarks>().count(), 3);

    auto const moments = prior.moments();
    Eigen::Matrix<double, 8, 1> expectedMean;
    expectedMean << 10.0, -5.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0;
    EXPECT_TRUE(moments.mean().isApprox(expectedMean));
    Eigen::Matrix<double, 8, 8> expectedCov = Eigen::Matrix<double, 8, 8>::Zero();
    expectedCov.topLeftCorner<2, 2>() = makePosition().covarianceLower().selfadjointView<Eigen::Lower>();
    for (int k = 0; k < 3; ++k)
    {
        expectedCov.block<2, 2>(2 + 2 * k, 2 + 2 * k) =
            makeLandmark().covarianceLower().selfadjointView<Eigen::Lower>();
    }
    EXPECT_TRUE(moments.covarianceLower().isApprox(expectedCov));

    auto const positionParams = makePosition();
    auto const landmarkParams = makeLandmark();
    auto const position = positionParams.toDensity();
    auto const landmark = landmarkParams.toDensity();
    ASSERT_TRUE(position.has_value() && landmark.has_value());
    Eigen::Matrix<double, 8, 1> x;
    x << 11.0, -4.0, 2.0, 1.5, 4.0, 0.0, 3.5, 2.0;
    double const expected = position->logDensity(x.head<2>()) + landmark->logDensity(x.segment<2>(2))
                          + landmark->logDensity(x.segment<2>(4)) + landmark->logDensity(x.segment<2>(6));
    EXPECT_NEAR(prior.logDensity(x), expected, 1e-12);
}

// Factors holds the factors placed by path in no space, and Product{space, factors} places them as
// Product{space, factors...} does; the same Factors placed in spaces of other counts follows each.
TEST(Product, FactorsArePlacedInEachSpace)
{
    auto const factors = Emrys::Factors{on<Position>(makePosition()), Emrys::onEach<Landmarks>(makeLandmark())};
    auto const listed = Product{
        Map{Position{}, Landmarks{Landmark{}, 2}},
        on<Position>(makePosition()),
        Emrys::onEach<Landmarks>(makeLandmark())
    };
    auto const two = Product{Map{Position{}, Landmarks{Landmark{}, 2}}, factors};
    static_assert(std::is_same_v<decltype(two), decltype(listed)>);
    Eigen::Matrix<double, 6, 1> const x{{11.0, -4.0, 2.0, 1.5, 4.0, 0.0}};
    EXPECT_EQ(two.logDensity(x), listed.logDensity(x));

    auto const five = Product{Map{Position{}, Landmarks{Landmark{}, 5}}, factors};
    EXPECT_EQ(five.valueDimension(), 12);
    EXPECT_EQ(five.moments().mean().tail<2>(), (Eigen::Vector2d{3.0, 1.0}));
}

// A path may name the whole space, so a repeated space is one onEach() factor.
TEST(Product, OnEachMayCoverTheWholeSpace)
{
    auto const product = Product{Landmarks{Landmark{}, 2}, Emrys::Factors{Emrys::onEach<Landmarks>(makeLandmark())}};
    auto const landmarkParams = makeLandmark();
    auto const landmark = landmarkParams.toDensity();
    ASSERT_TRUE(landmark.has_value());
    Eigen::Vector4d const x{4.0, 0.0, 3.5, 2.0};
    EXPECT_NEAR(product.logDensity(x), landmark->logDensity(x.head<2>()) + landmark->logDensity(x.tail<2>()), 1e-12);
}

// Without a path, onEach<>() covers every repetition of the whole space, which a space with no tag
// to name, such as a sensor's inferred measurement space, needs.
TEST(Product, OnEachWithoutAPathCoversTheWholeSpace)
{
    using Untagged = Emrys::Repeated<Linear<2>, 4>;
    auto const factors = Emrys::Factors{
        Emrys::onEach<>(GaussianParams<Linear<2>>{makeLandmark().mean(), makeLandmark().covarianceLower()})
    };
    auto const product = Product{Untagged{Linear<2>{}, 3}, factors};
    auto const tagged = Product{Landmarks{Landmark{}, 3}, Emrys::Factors{Emrys::onEach<Landmarks>(makeLandmark())}};
    auto const moments = product.moments();
    auto const expected = tagged.moments();
    EXPECT_EQ(moments.mean(), expected.mean());
    EXPECT_EQ(moments.covarianceLower(), expected.covarianceLower());
}

// The draws of an onEach() factor are the factor's own, one repetition after another, so the stream
// is the one drawn from the factor alone; drawSample(), drawSamplesWithLogDensity(), and
// drawSampleWithLogDensity() agree with it.
TEST(Product, OnEachDrawsEachRepetitionInTurn)
{
    auto const space = Landmarks{Landmark{}, 3};
    auto const product = Product{space, Emrys::onEach<Landmarks>(makeLandmark())};
    auto const landmarkParams = makeLandmark();
    auto const landmark = landmarkParams.toDensity();
    ASSERT_TRUE(landmark.has_value());

    auto rng = std::mt19937_64{5};
    auto const drawn = Emrys::drawSamples(product, 4, rng);
    auto replay = std::mt19937_64{5};
    for (Eigen::Index k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 2, Eigen::Dynamic> const one = Emrys::drawSamples(*landmark, 4, replay);
        EXPECT_EQ(Eigen::MatrixXd{drawn.middleRows(2 * k, 2)}, Eigen::MatrixXd{one});
    }

    auto sampleRng = std::mt19937_64{6};
    auto sampleReplay = std::mt19937_64{6};
    auto const sample = Emrys::drawSample(product, sampleRng);
    for (Eigen::Index k = 0; k < 3; ++k)
    {
        EXPECT_EQ(
            Eigen::Vector2d{sample.segment<2>(2 * k)},
            Eigen::Vector2d{Emrys::drawSample(*landmark, sampleReplay)}
        );
    }

    auto const withDensity = Emrys::drawSamplesWithLogDensity(product, 3, rng);
    for (Eigen::Index j = 0; j < 3; ++j)
    {
        EXPECT_NEAR(withDensity.logDensity(j), product.logDensity(withDensity.samples.col(j)), 1e-12);
    }
    EmrysTest::expectSingleDrawsCarryTheirLogDensity(product, 10, 7, 1e-12);
}

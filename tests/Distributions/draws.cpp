// Ranges of draws (roadmap step 17zw, [lazy-draws], [random-draws]): drawSamples() and
// drawSamplesWithLogDensity() return a sized input range whose elements are, bit for bit, the stream
// of drawSample() and drawSampleWithLogDensity() from a generator seeded alike, for every density
// that draws; the range draws when it is read, holds an lvalue density by reference and an rvalue by
// value, and collect() reads it into a SampleSet, a SamplesWithLogDensity, or indices; a kernel
// density's and a mixture's ranges pick as their single draws do, and with a resampler pick every
// element at the first read; ParticleSet, its assign(), and drawParticleSet() read the range into the
// set's storage and deduce the set from it. Headers under test: Distributions/Draws.h, Sampling.h,
// GaussianForm.h, Gamma.h, StudentT.h, Uniform.h, Categorical.h, Product.h, KernelDensity.h,
// Mixture.h, and ParticleSet.h.

#include <EmrysFilter/Distributions.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory_resource>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ParticleSet;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Lead : Linear<1>
{};

struct Box : Linear<2>
{};

struct Cruise
{};

struct Turn
{};

struct Hold
{};

struct Maneuver : Emrys::Named<Cruise, Turn, Hold>
{};

GaussianParams<State>
makeState()
{
    auto const a =
        Eigen::Matrix4d{{2.0, 0.3, 0.1, 0.0}, {0.3, 1.0, 0.0, 0.2}, {0.1, 0.0, 0.5, 0.1}, {0.0, 0.2, 0.1, 0.8}};
    return {Eigen::Vector4d{1.0, -2.0, 0.5, 0.25}, a};
}

GaussianParams<Linear<3>>
makeOdd()
{
    auto const a = Eigen::Matrix3d{{1.0, 0.2, 0.0}, {0.2, 2.0, 0.4}, {0.0, 0.4, 0.5}};
    return {Eigen::Vector3d{0.5, 1.5, -1.0}, a};
}

// The range of count draws from distribution, read element by element, equals the stream of count
// drawSample() calls from a generator seeded alike, bit for bit, and leaves its generator where the
// stream leaves its own, so it takes nothing more; the range with log-densities equals the stream of
// drawSampleWithLogDensity() alike, where the distribution has one.
template <typename Distribution>
void
expectRangeIsTheStream(Distribution const & distribution, Eigen::Index count, std::uint64_t seed)
{
    auto rangeRng = std::mt19937_64{seed};
    auto streamRng = std::mt19937_64{seed};
    auto draws = Emrys::drawSamples(distribution, count, rangeRng);
    EXPECT_EQ(draws.size(), count);
    Eigen::Index read = 0;
    for (auto const & draw : draws)
    {
        auto const single = Emrys::drawSample(distribution, streamRng);
        EXPECT_TRUE(draw == single) << "draw " << read;
        ++read;
    }
    EXPECT_EQ(read, count);
    EXPECT_TRUE(rangeRng == streamRng) << "the range took a different number of values from its generator";

    if constexpr (requires(std::mt19937_64 & rng) { Emrys::drawSampleWithLogDensity(distribution, rng); })
    {
        auto withRng = std::mt19937_64{seed + 1};
        auto singleRng = std::mt19937_64{seed + 1};
        Eigen::Index k = 0;
        for (auto const & draw : Emrys::drawSamplesWithLogDensity(distribution, count, withRng))
        {
            auto const single = Emrys::drawSampleWithLogDensity(distribution, singleRng);
            EXPECT_TRUE(draw.sample == single.sample) << "draw " << k;
            EXPECT_EQ(draw.logDensity, single.logDensity) << "draw " << k;
            ++k;
        }
        EXPECT_EQ(k, count);
        EXPECT_TRUE(withRng == singleRng);
    }
}

}  // namespace

// The range is a sized input range of values in the density's space, which a range-based algorithm
// reads, and it is moved, never copied (the copy is a negative compile test).
TEST(Draws, IsASizedInputRange)
{
    auto const params = makeState();
    auto const density = params.toDensity().value();
    auto rng = std::mt19937_64{1};
    using Range = decltype(Emrys::drawSamples(density, 3, rng));
    static_assert(std::ranges::input_range<Range>);
    static_assert(std::ranges::sized_range<Range>);
    static_assert(!std::ranges::forward_range<Range>);
    static_assert(!std::is_copy_constructible_v<Range> && std::is_move_constructible_v<Range>);
    static_assert(std::same_as<std::ranges::range_value_t<Range>, Emrys::ValueVector<State>>);
    static_assert(std::same_as<typename Range::SpaceType, State>);

    auto draws = Emrys::drawSamples(density, 3, rng);
    auto moved = std::move(draws);
    EXPECT_EQ(std::ranges::distance(moved), 3);
}

// Each density's range is the stream of its single draws from the same seed: a Gaussian in odd and
// even dimensions, where std::normal_distribution may keep a spare normal, in float, and in a
// run-time dimension; the square-root and information forms; a gamma, a Student-t, a uniform box,
// a categorical distribution, and a product of them.
TEST(Draws, EachDensitysRangeIsItsStreamOfSingleDraws)
{
    auto const state = makeState();
    expectRangeIsTheStream(state.toDensity().value(), 25, 1);
    auto const odd = makeOdd();
    expectRangeIsTheStream(odd.toDensity().value(), 25, 2);
    auto const single = GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.5}, Eigen::Matrix<double, 1, 1>{2.0}};
    expectRangeIsTheStream(single.toDensity().value(), 25, 3);
    auto const narrow = GaussianParams<Linear<2>, float>{Eigen::Vector2f{1.0f, 2.0f}, Eigen::Matrix2f::Identity()};
    expectRangeIsTheStream(narrow.toDensity().value(), 25, 4);
    auto const dynamic =
        GaussianParams<Linear<Eigen::Dynamic>>{Eigen::VectorXd::Ones(5), Eigen::MatrixXd::Identity(5, 5)};
    expectRangeIsTheStream(dynamic.toDensity().value(), 25, 5);

    auto const root = Emrys::toSqrt(state);
    ASSERT_TRUE(root);
    expectRangeIsTheStream(root->toDensity().value(), 25, 6);
    auto const info = Emrys::toInfo(state);
    ASSERT_TRUE(info);
    expectRangeIsTheStream(info->toDensity().value(), 25, 7);

    auto const gamma =
        Emrys::Gamma<Linear<2>>::fromMeanAndStdDev(Eigen::Vector2d{10.0, 2.0}, Eigen::Vector2d{2.0, 1.5}).value();
    expectRangeIsTheStream(gamma, 25, 8);
    auto const t = Emrys::StudentT<Linear<3>>::create(Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Matrix3d::Identity(), 5.0);
    ASSERT_TRUE(t);
    expectRangeIsTheStream(*t, 25, 9);
    auto const box = Emrys::Uniform<Box>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    expectRangeIsTheStream(box, 25, 10);

    auto const categoricalParams =
        Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector3d{0.6, 0.1, 0.3});
    auto const categorical = categoricalParams.toDensity().value();
    expectRangeIsTheStream(categorical, 40, 11);

    auto const exponential = Emrys::Gamma<Lead>::exponential(2.0).value();
    auto const product = Emrys::Product{Composite<Lead, Box>{}, Emrys::on<Box>(box), Emrys::on<Lead>(exponential)};
    expectRangeIsTheStream(product, 25, 12);
}

// A kernel density's and a mixture's ranges pick independently, as their single draws do, by the
// running sums of the weights, with a particle or component of zero weight never picked: the range
// is the stream of single draws, bit for bit.
TEST(Draws, KernelDensityAndMixtureRangesAreTheirStreams)
{
    auto rng = std::mt19937_64{20};
    auto const prior = makeState();
    auto particles = ParticleSet{Emrys::drawSamples(prior.toDensity().value(), 200, rng)};
    particles.setLogWeight(3, -std::numeric_limits<double>::infinity());
    particles.setLogWeight(17, 2.0);
    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde);
    expectRangeIsTheStream(*kde, 60, 21);

    auto mixture = Emrys::GaussianMixtureParams<State>{State{}};
    mixture.add(std::log(0.2), prior.mean(), prior.covarianceLower());
    mixture.add(-std::numeric_limits<double>::infinity(), Eigen::Vector4d::Ones(), Eigen::Matrix4d::Identity());
    mixture.add(std::log(0.8), Eigen::Vector4d{5.0, -2.0, 0.0, 1.0}, Eigen::Matrix4d::Identity());
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    expectRangeIsTheStream(*density, 60, 22);
}

// Given a resampler, a kernel density's range picks every particle at the first read (how a
// systematic resampler spreads them is kernel_density.cpp's test): the range with log-densities
// draws the same from the same seed, with the density's log-density at each, and its collect()
// under a policy is collect()'s.
TEST(Draws, AResamplerPicksEveryElementAtTheFirstRead)
{
    auto rng = std::mt19937_64{30};
    auto const prior = makeState();
    auto const particles = ParticleSet{Emrys::drawSamples(prior.toDensity().value(), 10, rng)};
    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde);
    auto first = std::mt19937_64{31};
    auto second = std::mt19937_64{31};
    auto third = std::mt19937_64{31};
    auto const samples = Emrys::drawSamples(*kde, 50, first, Emrys::resamplers::systematic()).collect();
    auto const withLog = Emrys::drawSamplesWithLogDensity(*kde, 50, second, Emrys::resamplers::systematic()).collect();
    auto const underPolicy = Emrys::drawSamplesWithLogDensity(*kde, 50, third, Emrys::resamplers::systematic())
                                 .collect(Emrys::execution::sequential);
    EXPECT_EQ(samples, withLog.samples);
    EXPECT_EQ(underPolicy.samples, withLog.samples);
    EXPECT_EQ(underPolicy.logDensity.matrix(), withLog.logDensity.matrix());
    for (Eigen::Index j = 0; j < 50; ++j)
    {
        EXPECT_EQ(withLog.logDensity(j), kde->logDensity(withLog.samples.col(j)));
    }
}

// The range draws when it is read, not when it is made: a value taken from the generator in between
// comes before the first draw. An empty range takes nothing from its generator.
TEST(Draws, DrawsWhenRead)
{
    auto const params = makeOdd();
    auto const density = params.toDensity().value();
    auto rng = std::mt19937_64{40};
    auto draws = Emrys::drawSamples(density, 4, rng);
    EXPECT_TRUE(rng == std::mt19937_64{40});
    (void)rng();

    auto expected = std::mt19937_64{40};
    (void)expected();
    auto const later = draws.collect();
    for (Eigen::Index j = 0; j < 4; ++j)
    {
        EXPECT_EQ(later.col(j), Emrys::drawSample(density, expected));
    }

    auto empty = std::mt19937_64{41};
    auto none = Emrys::drawSamples(density, 0, empty);
    EXPECT_TRUE(none.begin() == none.end());
    EXPECT_TRUE(empty == std::mt19937_64{41});
    EXPECT_EQ(Emrys::drawSamples(density, 0, empty).collect().cols(), 0);
}

// A density given as an lvalue is held by reference and one given as an rvalue, such as
// toDensity().value(), by value, so a range kept in a variable reads a density that is still alive.
TEST(Draws, HoldsAnRvalueDensityByValue)
{
    auto const params = makeOdd();
    auto const density = params.toDensity().value();
    auto rng = std::mt19937_64{50};
    auto const byReference = Emrys::drawSamples(density, 3, rng);
    EXPECT_EQ(&byReference.density(), &density);

    auto byValue = Emrys::drawSamples(params.toDensity().value(), 3, rng);
    auto again = std::mt19937_64{50};
    auto const read = byValue.collect();
    for (Eigen::Index j = 0; j < 3; ++j)
    {
        EXPECT_EQ(read.col(j), Emrys::drawSample(density, again));
    }
}

// collect() gives a SampleSet in the space, a SamplesWithLogDensity whose log-densities are the
// density's, and a categorical distribution's indices.
TEST(Draws, CollectReadsEveryDraw)
{
    auto const params = makeState();
    auto const density = params.toDensity().value();
    auto rng = std::mt19937_64{60};
    Emrys::SampleSet<State> const samples = Emrys::drawSamples(density, 7, rng).collect();
    EXPECT_EQ(samples.rows(), 4);
    EXPECT_EQ(samples.cols(), 7);

    auto const withLog = Emrys::drawSamplesWithLogDensity(density, 5, rng).collect();
    ASSERT_EQ(withLog.logDensity.size(), 5);
    for (Eigen::Index j = 0; j < 5; ++j)
    {
        EXPECT_NEAR(withLog.logDensity(j), density.logDensity(withLog.samples.col(j)), 1e-12);
    }

    auto const categoricalParams =
        Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector3d{0.6, 0.1, 0.3});
    std::vector<Eigen::Index> const indices =
        Emrys::drawSamples(categoricalParams.toDensity().value(), 9, rng).collect();
    EXPECT_EQ(indices.size(), 9u);
}

// A set built from the range holds its draws, in order, with equal weights, and deduces its space
// and scalar, and with an allocator its allocator type; assign() redraws a set in place, and
// drawParticleSet() is the set built from drawSamples(), in a fixed and in a run-time dimension.
TEST(Draws, ParticleSetReadsTheRange)
{
    auto const params = makeState();
    auto const density = params.toDensity().value();
    auto a = std::mt19937_64{70};
    auto b = std::mt19937_64{70};

    auto particles = ParticleSet{Emrys::drawSamples(density, 50, a)};
    static_assert(std::same_as<decltype(particles), ParticleSet<State>>);
    EXPECT_EQ(particles.states(), Emrys::drawSamples(density, 50, b).collect());
    EXPECT_TRUE(particles.logWeights().isApproxToConstant(-std::log(50.0)));

    particles.assign(Emrys::drawSamples(density, 20, a));
    EXPECT_EQ(particles.size(), 20);
    EXPECT_EQ(particles.states(), Emrys::drawSamples(density, 20, b).collect());
    EXPECT_TRUE(particles.logWeights().isApproxToConstant(-std::log(20.0)));

    auto const drawn = Emrys::drawParticleSet(density, 30, a);
    static_assert(std::same_as<std::remove_const_t<decltype(drawn)>, ParticleSet<State>>);
    EXPECT_EQ(drawn.states(), Emrys::drawSamples(density, 30, b).collect());

    auto resource = std::pmr::monotonic_buffer_resource{};
    auto const allocator = std::pmr::polymorphic_allocator<std::byte>{&resource};
    auto const inResource = ParticleSet{Emrys::drawSamples(density, 10, a), allocator};
    static_assert(std::same_as<
                  std::remove_const_t<decltype(inResource)>,
                  ParticleSet<State, double, std::pmr::polymorphic_allocator<std::byte>>>);
    EXPECT_EQ(inResource.get_allocator(), allocator);
    EXPECT_EQ(inResource.states(), Emrys::drawSamples(density, 10, b).collect());
    auto const named = Emrys::drawParticleSet(density, 10, a, allocator);
    EXPECT_EQ(named.get_allocator(), allocator);
    EXPECT_EQ(named.states(), Emrys::drawSamples(density, 10, b).collect());

    auto const dynamicParams =
        GaussianParams<Linear<Eigen::Dynamic>>{Eigen::VectorXd::Ones(5), Eigen::MatrixXd::Identity(5, 5)};
    auto const dynamicSet = Emrys::drawParticleSet(dynamicParams.toDensity().value(), 12, a);
    EXPECT_EQ(dynamicSet.valueDimension(), 5);
    EXPECT_EQ(dynamicSet.states(), Emrys::drawSamples(dynamicParams.toDensity().value(), 12, b).collect());
}

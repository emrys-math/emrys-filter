// Resizable spaces through the distributions (step 11, [sizes-and-dimensions]): a Resizable carries its size, indices
// are run-time, Gaussian, GaussianMixtureParams, and ParticleSet insert and erase coordinates,
// marginals and conditionals follow the size, a transform and an update run on a resizable state,
// and mismatched sizes die. A space with several resizable blocks [several-resizable-blocks]: its counts and strides,
// its run-time indices, that nothing infers its sizes from a length, insert and erase in a block
// that isn't last, and a measurement update that reads blocks in an order a single resizable block
// once forbade.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Transform.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <type_traits>

using Emrys::Composite;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::idx;
using Emrys::Linear;
using Emrys::normalizeValues;
using Emrys::ParticleSet;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::ScalarOf;
using Emrys::TagPath;
using Emrys::withCross;

namespace
{

constexpr double pi = std::numbers::pi;

// A pose and a variable number of landmarks, each a position and an orientation.
struct Pose : Linear<3>
{};

struct Orientation : RadiansPi<>
{};

struct Landmark : Composite<Linear<2>, Orientation>
{};

struct Landmarks : Repeated<Landmark>
{
    using Repeated::Repeated;
};

using Map = Composite<Pose, Landmarks>;

Map
makeMap(Eigen::Index landmarks)
{
    return Map{Pose{}, Landmarks{Landmark{}, landmarks}};
}

// A small mean and a covariance with every pair correlated, so misplaced blocks show.
Eigen::VectorXd
iota(Eigen::Index n)
{
    return Eigen::VectorXd::LinSpaced(n, 0.0, double(n - 1)) * 0.1;
}

Eigen::MatrixXd
makeCov(Eigen::Index n)
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Constant(n, n, 0.01);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            cov(i, j) += 0.001 * double(i + j);
        }
        cov(i, i) += 1.0 + double(i);
    }
    return cov;
}

Eigen::Vector3d
makeLandmark()
{
    return {5.0, -5.0, 4.0};
}  // the orientation needs wrapping

}  // namespace

// A Composite with a Repeated block has dynamic size and a stride, reports its size and offsets
// from the count, resizes, and normalizes every repetition's orientation.
TEST(ResizableSpace, CarriesItsSize)
{
    static_assert(Emrys::valueDimensionAtCompileTime<Map> == Eigen::Dynamic && Map::ValueDimensionStep == 3);
    auto map = Map{makeMap(2)};
    EXPECT_EQ(map.valueDimension(), 9);
    EXPECT_EQ(map.offset<1>(), 3);
    map.resize(12);
    EXPECT_EQ(map.block<1>().count(), 3);

    Eigen::VectorXd x = Eigen::VectorXd::Constant(12, 4.0);
    normalizeValues(map, x);
    for (Eigen::Index const row : {5, 8, 11})
    {
        EXPECT_NEAR(x(row), 4.0 - 2 * pi, 1e-15) << "row " << row;
    }
    EXPECT_EQ(x(4), 4.0);
}

// idx() takes the space instance and selects a repeated block, a path into each repetition, and
// one repetition; a fixed block keeps its compile-time size and agrees with the constant idx().
TEST(ResizableSpace, RuntimeIndices)
{
    auto const map = Map{makeMap(2)};
    Eigen::VectorXd const x = Eigen::VectorXd::LinSpaced(9, 0.0, 8.0);
    EXPECT_EQ(Eigen::VectorXd{x(idx<Landmarks>(map))}, Eigen::VectorXd::LinSpaced(6, 3.0, 8.0));
    EXPECT_EQ(Eigen::VectorXd{x(idx<TagPath<Landmarks, Orientation>>(map))}, Eigen::Vector2d(5.0, 8.0));
    EXPECT_EQ(Eigen::Vector3d{x(idx<Landmarks>(map, 1))}, Eigen::Vector3d(6.0, 7.0, 8.0));
    EXPECT_EQ((idx<TagPath<Landmarks, Orientation>>(map, 0)), 5);

    // A fixed block keeps its compile-time size, and the constant idx() agrees.
    auto const pose = x(idx<Pose>(map));
    static_assert(decltype(pose)::RowsAtCompileTime == 3);
    EXPECT_EQ(Eigen::VectorXd{x(Emrys::idx<Map, Landmarks>())}, Eigen::VectorXd{x(idx<Landmarks>(map))});
}

// insert<Landmarks>(i, mean, cov, cross) grows the Gaussian by one element at position i, placing
// its mean (wrapped), covariance, and cross-covariance and shifting the rest; erase() restores the
// original exactly and dropping the first element keeps the other's blocks.
TEST(ResizableSpace, GaussianInsertAndErase)
{
    Eigen::VectorXd const mean = iota(9);
    Eigen::MatrixXd const cov = makeCov(9);
    auto g = GaussianParams<Map>{makeMap(2), mean, cov};

    // A new landmark between the two, correlated with the pose's first coordinate only.
    auto const landmarkCov = Eigen::Matrix3d{{0.5, 0.1, 0.0}, {0.1, 0.4, 0.0}, {0.0, 0.0, 0.2}};
    Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(3, 9);
    cross(0, 0) = 0.2;
    g.insert<Landmarks>(1, makeLandmark(), landmarkCov, cross);
    EXPECT_EQ(g.valueDimension(), 12);
    EXPECT_EQ(g.space().block<1>().count(), 3);
    EXPECT_EQ(g.mean().segment<2>(6), makeLandmark().head<2>());
    EXPECT_NEAR(g.mean()(8), 4.0 - 2 * pi, 1e-15);  // wrapped like any stored orientation
    EXPECT_EQ(g.mean().head<6>(), mean.head<6>());
    EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());
    EXPECT_EQ((g.covarianceLower().block<3, 3>(6, 6)), landmarkCov);
    EXPECT_EQ(g.covarianceLower()(6, 0), 0.2);
    EXPECT_EQ(g.covarianceLower()(0, 6), 0.2);
    EXPECT_EQ((g.covarianceLower().block<6, 6>(0, 0)), (cov.block<6, 6>(0, 0)));
    EXPECT_EQ((g.covarianceLower().block<3, 3>(9, 9)), (cov.block<3, 3>(6, 6)));
    EXPECT_EQ((g.covarianceLower().block<6, 3>(0, 9)), (cov.block<6, 3>(0, 6)));

    // Erasing it restores the original exactly: the marginal of the rest.
    g.erase<Landmarks>(1);
    EXPECT_EQ(g.space(), makeMap(2));
    EXPECT_EQ(g.mean(), mean);
    EXPECT_EQ(g.covarianceLower(), cov);

    // Dropping the first landmark.
    g.erase<Landmarks>(0);
    EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());
    EXPECT_EQ((g.covarianceLower().block<3, 3>(3, 3)), (cov.block<3, 3>(6, 6)));
    EXPECT_TRUE(g.toDensity().has_value());
}

// Marginals of the repeated block and of a path into it carry the current count, and a conditional
// on the landmarks is a Gaussian over the fixed pose.
TEST(ResizableSpace, MarginalsAndConditionalsFollowTheSize)
{
    auto const g = GaussianParams<Map>{makeMap(2), iota(9), makeCov(9)};
    auto const landmarks = g.marginal<Landmarks>().copy();
    static_assert(std::is_same_v<decltype(landmarks)::SpaceType, Landmarks>);
    EXPECT_EQ(landmarks.space().count(), 2);
    EXPECT_EQ(landmarks.mean(), iota(9).tail<6>());

    auto const orientations = g.marginal<TagPath<Landmarks, Orientation>>().copy();
    EXPECT_EQ(orientations.space().count(), 2);

    auto const pose = g.conditional<Landmarks>(iota(9).tail<6>());
    static_assert(std::is_same_v<decltype(pose)::SpaceType, Pose>);
    EXPECT_LT((pose.mean() - iota(3)).norm(), 1e-12);
}

// A mixture inserts and erases an element in every component at once, keeping each component's
// space equal to the mixture's.
TEST(ResizableSpace, MixtureInsertAndErase)
{
    auto mixture = GaussianMixtureParams<Map>{makeMap(1)};
    mixture.add(0.0, Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));
    mixture.add(-1.0, Eigen::VectorXd::Ones(6), 2.0 * Eigen::MatrixXd::Identity(6, 6));
    mixture.insert<Landmarks>(0, makeLandmark(), Eigen::Matrix3d::Identity());
    EXPECT_EQ(mixture.valueDimension(), 9);
    for (auto const & component : mixture)
    {
        EXPECT_EQ(component.component.space(), mixture.space());
        EXPECT_EQ(component.component.mean().segment<2>(3), makeLandmark().head<2>());
    }
    mixture.erase<Landmarks>(1);  // the original landmark; the inserted one remains
    EXPECT_EQ(mixture.valueDimension(), 6);
    EXPECT_EQ(mixture.begin()->component.mean().segment<2>(3), makeLandmark().head<2>());
    EXPECT_TRUE(mixture.toDensity().has_value());
}

// Erasing an element from a particle set drops its rows and marginalizes a stored bandwidth;
// inserting one adds the given rows and clears the bandwidth, which no longer fits.
TEST(ResizableSpace, ParticleInsertAndErase)
{
    Eigen::MatrixXd const states = Eigen::MatrixXd::Random(6, 30);
    auto set = ParticleSet<Map>{makeMap(1), states};
    set.setBandwidth(Eigen::MatrixXd::Identity(6, 6));
    set.erase<Landmarks>(0);
    EXPECT_EQ(set.valueDimension(), 3);
    EXPECT_EQ(set.bandwidth()->rows(), 3);  // marginalized
    EXPECT_EQ(set.states(), states.topRows<3>());

    // Spread across particles, so Silverman's rule has something to smooth.
    Eigen::MatrixXd const landmarks = Eigen::MatrixXd::Random(3, 30);
    set.insert<Landmarks>(0, landmarks);
    EXPECT_EQ(set.valueDimension(), 6);
    EXPECT_FALSE(set.bandwidth().has_value());  // no longer fits, so cleared
    EXPECT_EQ(set.states().bottomRows<3>(), landmarks);
    EXPECT_TRUE(set.toDensity().has_value());
}

// A linearized transform of a resizable state to a repeated output has a dynamic cross matrix sized
// by both, update() returns a posterior over the same space, with no noise the readings are matched
// exactly, and an unnamed output is Linear sized by the function.
TEST(ResizableSpace, TransformAndUpdateAResizableState)
{
    // Each landmark's position is observed relative to the pose; there is a reading per landmark.
    struct Reading : Linear<2>
    {};

    struct Readings : Repeated<Reading>
    {
        using Repeated::Repeated;
    };

    auto const map = Map{makeMap(2)};
    auto const x = GaussianParams<Map>{map, iota(9), makeCov(9)};
    auto const z = Eigen::Vector4d{0.4, 0.4, 0.7, 0.7};
    auto const readings = []<typename T>(auto const & state) {
        auto result = Eigen::Matrix<T, Eigen::Dynamic, 1>(4);
        for (Eigen::Index k = 0; k < 2; ++k)
        {
            result.segment(2 * k, 2) = state.segment(3 + 3 * k, 2) - state.head(2);
        }
        return result;
    };
    auto const predicted = Emrys::transform(
        Emrys::methods::linearized(),
        readings,
        Emrys::outputSpace(Readings{Reading{}, 2}),
        withCross(x)
    );
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(predicted.crossCovariance())>, Eigen::MatrixXd>);
    EXPECT_EQ(predicted.crossCovariance().rows(), 9);
    EXPECT_EQ(predicted.crossCovariance().cols(), 4);
    EXPECT_EQ(predicted.output().space().count(), 2);

    auto const updated = Emrys::condition(x, predicted, z);
    ASSERT_TRUE(updated);
    auto const & [posterior, observed] = *updated;
    static_assert(std::is_same_v<typename decltype(posterior)::SpaceType, Map>);
    EXPECT_EQ(posterior.space(), map);
    EXPECT_TRUE(std::isfinite(observed.logLikelihood()));
    // With no measurement noise, the relative positions now match the readings exactly.
    for (Eigen::Index k = 0; k < 2; ++k)
    {
        EXPECT_LT(
            (posterior.mean().template segment<2>(3 + 3 * k) - posterior.mean().template head<2>()
             - z.segment<2>(2 * k))
                .norm(),
            1e-9
        );
    }

    // With no output space given, a dynamic return is Linear and sized by f.
    auto const unnamed = Emrys::transform(Emrys::methods::linearized(), readings, x);
    static_assert(std::is_same_v<decltype(unnamed)::SpaceType, Linear<Eigen::Dynamic>>);
    EXPECT_EQ(unnamed.space().valueDimension(), 4);
}

namespace
{

// A pose, a set of tracks, and a set of landmarks: two growing sets with different strides.
struct TrackPosition : Linear<2>
{};

struct TrackVelocity : Linear<2>
{};

struct Track : Composite<TrackPosition, TrackVelocity>
{};

struct Tracks : Repeated<Track>
{
    using Repeated::Repeated;
};

using Scene = Composite<Pose, Tracks, Landmarks>;

Scene
makeScene(Eigen::Index tracks, Eigen::Index landmarks)
{
    return Scene{Pose{}, Tracks{Track{}, tracks}, Landmarks{Landmark{}, landmarks}};
}

// The tracks and landmarks as a named block, so that the second resizable block is nested.
struct Sets : Composite<Tracks, Landmarks>
{
    using Composite::Composite;
};

using NestedScene = Composite<Pose, Sets>;

// A resizable block first, then a fixed one: a single resizable block anywhere.
using LandmarksFirst = Composite<Landmarks, Pose>;

// Whether a T, a space or a distribution, can be resized to a length.
template <typename T>
constexpr bool hasResize = requires(T & t) { t.resize(3); };

// Whether a Gaussian in S can be resized to a length.
template <typename S>
constexpr bool resizableByLength = hasResize<GaussianParams<S>>;

// Whether every component of the mixture M can be resized to a length.
template <typename M>
constexpr bool hasResizeComponents = requires(M & m) { m.resizeComponents(3); };

}  // namespace

// A composite counts its resizable blocks at any depth: one gives it resize(n) wherever it sits, two
// leave it without, and its stride is the greatest common divisor of theirs (4 and 3 give 1).
// allowsValueDimension() is then only necessary: 4 passes but no Scene has 4 coordinates.
TEST(SeveralResizableBlocks, CountsAndStrides)
{
    static_assert(Emrys::Space<Scene> && Emrys::Space<NestedScene> && Emrys::Space<LandmarksFirst>);
    static_assert(Emrys::internal::resizableBlockCount<Scene> == 2);
    static_assert(Emrys::internal::resizableBlockCount<NestedScene> == 2);
    static_assert(Emrys::internal::resizableBlockCount<LandmarksFirst> == 1);
    static_assert(Emrys::internal::resizableBlockCount<Map> == 1);
    static_assert(Emrys::internal::resizableBlockCount<Pose> == 0);
    static_assert(Scene::FixedValueDimension == 3 && Scene::ValueDimensionStep == 1);
    static_assert(Scene::MaxValueDimension == Eigen::Dynamic);
    static_assert(!hasResize<Scene> && !hasResize<NestedScene> && hasResize<LandmarksFirst>);
    static_assert(Emrys::allowsValueDimension<Scene>(4));

    auto first = LandmarksFirst{Landmarks{Landmark{}, 1}, Pose{}};
    first.resize(9);
    EXPECT_EQ(first.block<0>().count(), 2);
    EXPECT_EQ(first.offset<1>(), 6);

    auto const scene = makeScene(2, 3);
    EXPECT_EQ(scene.valueDimension(), 3 + 8 + 9);
    EXPECT_EQ(scene.offset<2>(), 11);
    auto const nested = NestedScene{Pose{}, Sets{Tracks{Track{}, 2}, Landmarks{Landmark{}, 3}}};
    EXPECT_EQ(nested.valueDimension(), 20);
}

// idx<Path>(space) places every block from the counts before it: a block after the first resizable
// one, one repetition of it, and a path through it; a fixed block before both keeps its constant
// index. Normalizing wraps the landmarks' orientations and nothing else.
TEST(SeveralResizableBlocks, RuntimeIndices)
{
    auto const scene = makeScene(2, 3);
    Eigen::VectorXd const x = Eigen::VectorXd::LinSpaced(20, 0.0, 19.0);
    EXPECT_EQ(Eigen::VectorXd{x(idx<Tracks>(scene))}, Eigen::VectorXd::LinSpaced(8, 3.0, 10.0));
    EXPECT_EQ(Eigen::VectorXd{x(idx<Landmarks>(scene))}, Eigen::VectorXd::LinSpaced(9, 11.0, 19.0));
    EXPECT_EQ(Eigen::Vector3d{x(idx<Landmarks>(scene, 1))}, Eigen::Vector3d(14.0, 15.0, 16.0));
    EXPECT_EQ(Eigen::Vector3d{x(idx<TagPath<Landmarks, Orientation>>(scene))}, Eigen::Vector3d(13.0, 16.0, 19.0));
    EXPECT_EQ(Eigen::Vector2d{x(idx<TagPath<Tracks, TrackVelocity>>(scene, 1))}, Eigen::Vector2d(9.0, 10.0));
    EXPECT_EQ(Eigen::Vector3d{x(Emrys::idx<Scene, Pose>())}, Eigen::Vector3d(0.0, 1.0, 2.0));

    auto const nested = NestedScene{Pose{}, Sets{Tracks{Track{}, 2}, Landmarks{Landmark{}, 3}}};
    EXPECT_EQ(Eigen::VectorXd{x(idx<TagPath<Sets, Landmarks>>(nested))}, Eigen::VectorXd::LinSpaced(9, 11.0, 19.0));

    Eigen::VectorXd y = Eigen::VectorXd::Constant(20, 4.0);
    normalizeValues(scene, y);
    for (Eigen::Index row = 0; row < 20; ++row)
    {
        bool const orientation = row == 13 || row == 16 || row == 19;
        EXPECT_NEAR(y(row), orientation ? 4.0 - 2 * pi : 4.0, 1e-15) << "row " << row;
    }
}

// Nothing builds a distribution in a space with several resizable blocks from a length alone: the
// space-less constructors and resize(n) are absent, while a space with one keeps them.
TEST(SeveralResizableBlocks, SizesComeFromTheSpace)
{
    static_assert(!std::is_constructible_v<GaussianParams<Scene>, Eigen::VectorXd, Eigen::MatrixXd>);
    static_assert(!std::is_constructible_v<GaussianParams<Scene>, Eigen::Index>);
    static_assert(std::is_constructible_v<GaussianParams<Scene>, Scene, Eigen::VectorXd, Eigen::MatrixXd>);
    static_assert(!resizableByLength<Scene> && resizableByLength<Map> && resizableByLength<LandmarksFirst>);
    static_assert(!std::is_constructible_v<ParticleSet<Scene>, Eigen::MatrixXd>);
    static_assert(std::is_constructible_v<ParticleSet<Scene>, Scene, Eigen::MatrixXd>);
    static_assert(!std::is_constructible_v<GaussianMixtureParams<Scene>, Eigen::Index>);
    static_assert(
        !hasResizeComponents<GaussianMixtureParams<Scene>> && hasResizeComponents<GaussianMixtureParams<Map>>
    );

    auto const g = GaussianParams<Scene>{makeScene(1, 1), iota(10), makeCov(10)};
    EXPECT_EQ(g.space(), makeScene(1, 1));
    EXPECT_TRUE(g.toDensity().has_value());
}

// A Gaussian grows the tracks, which aren't last, and the landmarks after them: each insert places
// its mean and covariance at the block's run-time offset and shifts what follows, and erasing both
// restores the original exactly.
TEST(SeveralResizableBlocks, GaussianInsertAndEraseInEachBlock)
{
    Eigen::VectorXd const mean = iota(10);
    Eigen::MatrixXd const cov = makeCov(10);
    auto g = GaussianParams<Scene>{makeScene(1, 1), mean, cov};

    auto const track = Eigen::Vector4d{1.0, 2.0, 3.0, 4.0};
    Eigen::Matrix4d const trackCov = 0.5 * Eigen::Matrix4d::Identity();
    g.insert<Tracks>(1, track, trackCov);
    EXPECT_EQ(g.space(), makeScene(2, 1));
    EXPECT_EQ(g.mean().segment<4>(7), track);
    EXPECT_EQ((g.covarianceLower().block<4, 4>(7, 7)), trackCov);
    EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());  // the landmark moved down by a track
    EXPECT_EQ((g.covarianceLower().block<3, 3>(11, 11)), (cov.block<3, 3>(7, 7)));
    EXPECT_EQ((g.covarianceLower().block<7, 3>(0, 11)), (cov.block<7, 3>(0, 7)));

    g.insert<Landmarks>(0, makeLandmark(), Eigen::Matrix3d::Identity());
    EXPECT_EQ(g.space(), makeScene(2, 2));
    EXPECT_EQ(g.mean().segment<2>(11), makeLandmark().head<2>());
    EXPECT_EQ(g.mean().tail<3>(), mean.tail<3>());

    g.erase<Landmarks>(0);
    g.erase<Tracks>(1);
    EXPECT_EQ(g.space(), makeScene(1, 1));
    EXPECT_EQ(g.mean(), mean);
    EXPECT_EQ(g.covarianceLower(), cov);
}

// The marginal of a block that isn't last carries its count and its rows; the conditional on the
// landmarks matches the plain-Eigen Gaussian conditional mu_a + S_ab S_bb^-1 (x_b - mu_b) to
// rounding. The landmark's orientation is 2.7 from the mean's, inside the period, so the plain
// difference is the space's.
TEST(SeveralResizableBlocks, MarginalsAndConditionals)
{
    auto const g = GaussianParams<Scene>{makeScene(2, 1), iota(14), makeCov(14)};
    auto const tracks = g.marginal<Tracks>().copy();
    static_assert(std::is_same_v<decltype(tracks)::SpaceType, Tracks>);
    EXPECT_EQ(tracks.space().count(), 2);
    EXPECT_EQ(tracks.mean(), iota(14).segment<8>(3));
    EXPECT_EQ(tracks.covarianceLower(), (makeCov(14).block<8, 8>(3, 3)));

    Eigen::Vector3d const landmark = makeLandmark();
    auto const rest = g.conditional<Landmarks>(landmark);
    Eigen::MatrixXd const cov = makeCov(14);
    Eigen::VectorXd const mu = iota(14);
    Eigen::VectorXd const expected =
        mu.head(11) + cov.topRightCorner(11, 3) * cov.bottomRightCorner(3, 3).lu().solve(landmark - mu.tail(3));
    EXPECT_EQ(rest.valueDimension(), 11);
    EXPECT_LT((rest.mean() - expected).norm(), 1e-12);
}

// A mixture and a particle set grow and shrink the tracks, the block that isn't last, in every
// component or particle at once.
TEST(SeveralResizableBlocks, MixtureAndParticlesInsertInTheMiddle)
{
    auto mixture = GaussianMixtureParams<Scene>{makeScene(0, 1)};
    mixture.add(0.0, Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));
    mixture.add(-1.0, Eigen::VectorXd::Ones(6), 2.0 * Eigen::MatrixXd::Identity(6, 6));
    auto const track = Eigen::Vector4d{1.0, 2.0, 3.0, 4.0};
    mixture.insert<Tracks>(0, track, Eigen::Matrix4d::Identity());
    EXPECT_EQ(mixture.space(), makeScene(1, 1));
    for (auto const & entry : mixture)
    {
        EXPECT_EQ(entry.component.space(), mixture.space());
        EXPECT_EQ(entry.component.mean().segment<4>(3), track);
    }
    EXPECT_EQ(mixture.begin()->component.mean().tail<3>(), Eigen::Vector3d::Zero());
    mixture.erase<Tracks>(0);
    EXPECT_EQ(mixture.space(), makeScene(0, 1));
    EXPECT_TRUE(mixture.toDensity().has_value());

    Eigen::MatrixXd const states = Eigen::MatrixXd::Random(6, 30);
    auto set = ParticleSet<Scene>{makeScene(0, 1), states};
    Eigen::MatrixXd const tracks = Eigen::MatrixXd::Random(4, 30);
    set.insert<Tracks>(0, tracks);
    EXPECT_EQ(set.space(), makeScene(1, 1));
    EXPECT_EQ(set.states().middleRows<4>(3), tracks);
    EXPECT_EQ(set.states().bottomRows<3>(), states.bottomRows<3>());
    set.erase<Tracks>(0);
    EXPECT_EQ(set.states(), states);
}

namespace
{

GaussianParams<Linear<2>> const landmarkFromTrackNoise{
    Eigen::Vector2d::Zero(),
    Eigen::Matrix2d{{0.04, 0.0}, {0.0, 0.09}}
};

// The first landmark's position relative to the first track, read in the order landmarks, tracks,
// pose: a joint space with two resizable blocks, the first of them first.
auto
makeLandmarkFromTrack(Eigen::Vector2d const & z)
{
    return Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            []<typename T>(auto const & in, auto const &, auto const & x) {
                return Eigen::Matrix<T, 2, 1>{
                    Emrys::block<Landmarks>(in, x).template head<2>() - Emrys::block<Tracks>(in, x).template head<2>()
                };
            },
            Emrys::reads(Emrys::path<Landmarks>, Emrys::path<Tracks>, Emrys::path<Pose>),
            Emrys::additiveNoise(landmarkFromTrackNoise)
        ),
        z
    );
}

}  // namespace

// update(prior, measurement) over blocks read in an order that once failed to compile matches the
// plain-Eigen Kalman update with the measurement's linear H, to rounding, and keeps the space.
TEST(SeveralResizableBlocks, UpdateReadsBlocksInAnyOrder)
{
    auto const scene = makeScene(2, 2);
    Eigen::VectorXd const mu = iota(17);
    Eigen::MatrixXd const cov = makeCov(17);
    auto const prior = GaussianParams<Scene>{scene, mu, cov};
    auto const z = Eigen::Vector2d{0.5, -0.3};
    auto const updated = Emrys::update(prior, makeLandmarkFromTrack(z));
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 1u);
    auto const & posterior = updated->branches.front().posterior;
    EXPECT_EQ(posterior.space(), scene);

    // h(x) is the first landmark's position (rows 11, 12) minus the first track's (rows 3, 4).
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(2, 17);
    h(0, 11) = h(1, 12) = 1.0;
    h(0, 3) = h(1, 4) = -1.0;
    Eigen::MatrixXd const s = h * cov * h.transpose() + landmarkFromTrackNoise.covarianceLower();
    Eigen::MatrixXd const gain = cov * h.transpose() * s.inverse();
    Eigen::VectorXd const mean = mu + gain * (z - h * mu);
    Eigen::MatrixXd const covariance = cov - gain * s * gain.transpose();
    EXPECT_LT((posterior.mean() - mean).norm(), 1e-12);
    EXPECT_LT((posterior.covarianceLower() - covariance).norm(), 1e-12);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// A space with several resizable blocks is not resized to the data, so data of another length dies,
// even a length that some other split would give (11 coordinates are two tracks and no landmarks)
// or that the type allows but no split gives (4 coordinates).
TEST(SeveralResizableBlocksDeathTest, TheSpaceMustFitTheData)
{
    EXPECT_DEATH((GaussianParams<Scene>{makeScene(1, 1), iota(11), makeCov(11)}), "fit the space");
    EXPECT_DEATH((GaussianParams<Scene>{makeScene(0, 0), iota(4), makeCov(4)}), "fit the space");
}

// A resize that the space cannot take, an insert or erase out of range or of a partial element, and
// an insert past a capped Repeated's bound die with the named messages.
TEST(ResizableSpaceDeathTest, SizesMustMatchTheSpace)
{
    auto g = GaussianParams<Map>{makeMap(2), iota(9), makeCov(9)};
    EXPECT_DEATH(g.resize(10), "fit the space");
    EXPECT_DEATH(g.insert<Landmarks>(3, makeLandmark(), Eigen::Matrix3d::Identity()), "out of range");
    EXPECT_DEATH(g.insert<Landmarks>(0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()), "whole elements");
    EXPECT_DEATH(g.erase<Landmarks>(1, 2), "out of range");

    struct Capped : Repeated<Landmark, 2>
    {
        using Repeated::Repeated;
    };

    auto capped = GaussianParams<Composite<Pose, Capped>>{
        Composite<Pose, Capped>{Pose{}, Capped{Landmark{}, 2}},
        iota(9),
        makeCov(9)
    };
    EXPECT_DEATH(capped.insert<Capped>(0, makeLandmark(), Eigen::Matrix3d::Identity()), "fit the space");
}

#endif

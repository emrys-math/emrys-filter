// Selections of repetitions through compositions and placements (roadmap step 17zu, decisions.md
// [composed-selections], [selected-reads]): compose() of measurement models and of process models whose
// reads() select repetitions of a repeated block, and on<Path>() of such a model. As in
// selected_reads.cpp, a state written with one tag per repetition, Composite<Position, Heading, First,
// Second, Third>, has the coordinates of Composite<Position, Heading, Landmarks> with three landmarks,
// in the same order, so an update or a prediction through a selection must equal the one through the
// tagged blocks; the two differ only in the sizes Eigen fixes at compile time, hence the tolerance.
// Also covered: the selections a composition and a placement hold, parts that spell one selection in
// two ways, and the assertion on parts that read different repetitions under one name, in a
// composition and in a composition that is placed.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::compose;
using Emrys::Composite;
using Emrys::continuousNoise;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::processModel;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::writes;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Heading : RadiansPi<>
{};

struct Landmark : Linear<2>
{};

struct Landmarks : Repeated<Landmark, 6>
{
    using Repeated::Repeated;
};

/// a vehicle and landmarks, as a state that grows holds them
using Map = Composite<Position, Heading, Landmarks>;

struct First : Linear<2>
{};

struct Second : Linear<2>
{};

struct Third : Linear<2>
{};

/// the coordinates of Map with three landmarks, a tag per landmark
using Tagged = Composite<Position, Heading, First, Second, Third>;

/// the names two parts of a composition read two landmarks under
struct Near : Linear<2>
{};

struct Far : Linear<2>
{};

struct Range : Linear<1>
{};

struct Bearing : RadiansPi<>
{};

using RangeBearing = Composite<Range, Bearing>;

/// the sightings two parts of a composition write
struct NearSight : Composite<Range, Bearing>
{};

struct FarSight : Composite<Range, Bearing>
{};

/// what two parts that read two landmarks under one name write
struct TwoRanges : Linear<2>
{};

struct Spread : Linear<1>
{};

/// a block beside the map, so that the map's rows are not the state's first
struct Drift : Linear<1>
{};

/// the map as a block of a larger state, and the same coordinates with a tag per landmark
struct Survey : Composite<Position, Heading, Landmarks>
{
    using Composite::Composite;
};

struct TaggedSurvey : Composite<Position, Heading, First, Second, Third>
{};

using Site = Composite<Drift, Survey>;
using TaggedSite = Composite<Drift, TaggedSurvey>;

using Vector9 = Eigen::Matrix<double, 9, 1>;
using Matrix9 = Eigen::Matrix<double, 9, 9>;
using Vector10 = Eigen::Matrix<double, 10, 1>;
using Matrix10 = Eigen::Matrix<double, 10, 10>;

// A drift of 0.2, a vehicle at (1, 2) heading 0.3, and landmarks at (10, 4), (-3, 8), and (6, -7).
Vector10
siteMean()
{
    Vector10 mean;
    mean << 0.2, 1.0, 2.0, 0.3, 10.0, 4.0, -3.0, 8.0, 6.0, -7.0;
    return mean;
}

// A covariance with every block correlated with every other: A A^T + I / 2 for a fixed full A.
Matrix10
siteCovariance()
{
    Matrix10 a;
    for (Eigen::Index i = 0; i < 10; ++i)
    {
        for (Eigen::Index j = 0; j < 10; ++j)
        {
            a(i, j) = 0.3 * std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
        }
    }
    return a * a.transpose() + 0.5 * Matrix10::Identity();
}

// The site without its drift.
Vector9
mapMean()
{
    return siteMean().tail<9>();
}

Matrix9
mapCovariance()
{
    return siteCovariance().bottomRightCorner<9, 9>();
}

GaussianParams<Map>
mapPrior()
{
    return {Map{Position{}, Heading{}, Landmarks{Landmark{}, 3}}, mapMean(), mapCovariance()};
}

GaussianParams<Tagged>
taggedPrior()
{
    return {mapMean(), mapCovariance()};
}

GaussianParams<Site>
sitePrior()
{
    return {Site{Drift{}, Survey{Position{}, Heading{}, Landmarks{Landmark{}, 3}}}, siteMean(), siteCovariance()};
}

GaussianParams<TaggedSite>
taggedSitePrior()
{
    return {siteMean(), siteCovariance()};
}

Eigen::Matrix2d const sightingNoise = Eigen::Vector2d{0.04, 0.0025}.asDiagonal();

// The range and the bearing from the vehicle to the landmark read under the name Name.
template <typename Name>
auto
sightingOf()
{
    return []<typename T>(auto const & in, auto const & x) {
        Eigen::Vector<T, 2> const d = block<Name>(in, x) - block<Position>(in, x);
        return Eigen::Vector<T, 2>{
            T(Eigen::numext::sqrt(T(d.squaredNorm()))),
            T(Eigen::numext::atan2(d(1), d(0)) - block<Heading>(in, x)(0))
        };
    };
}

// A sighting of the landmark that \a item reads under Name, written as the block Sight.
template <typename Name, typename Sight, typename Item>
auto
sightingAs(Item const & item)
{
    return measurementModel(
        sightingOf<Name>(),
        reads(path<Position>, path<Heading>, item),
        writes(path<Sight>),
        additiveNoise(GaussianParams<Sight>{Eigen::Vector2d::Zero(), sightingNoise})
    );
}

// The range from the vehicle to the landmark that \a item reads under Landmark.
template <typename Item>
auto
rangeTo(Item const & item)
{
    return measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            return Eigen::Vector<T, 1>{
                T(Eigen::numext::sqrt(T((block<Landmark>(in, x) - block<Position>(in, x)).squaredNorm())))
            };
        },
        reads(path<Position>, item),
        writes(path<Range>),
        additiveNoise(GaussianParams<Range>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.04}})
    );
}

// The bearing from the vehicle to the landmark that \a item reads under Landmark.
template <typename Item>
auto
bearingTo(Item const & item)
{
    return measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            Eigen::Vector<T, 2> const d = block<Landmark>(in, x) - block<Position>(in, x);
            return Eigen::Vector<T, 1>{T(Eigen::numext::atan2(d(1), d(0)) - block<Heading>(in, x)(0))};
        },
        reads(path<Position>, path<Heading>, item),
        writes(path<Bearing>),
        additiveNoise(GaussianParams<Bearing>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.0025}})
    );
}

// The ranges from the vehicle, the first block of the input, to the two landmarks after it, whatever
// the blocks they are read as.
template <typename... Items>
auto
twoRanges(Items const &... items)
{
    return measurementModel(
        []<typename T>(auto const & x) {
            return Eigen::Vector<T, 2>{
                T(Eigen::numext::sqrt(T((x.template segment<2>(2) - x.template head<2>()).squaredNorm()))),
                T(Eigen::numext::sqrt(T((x.template segment<2>(4) - x.template head<2>()).squaredNorm())))
            };
        },
        reads(path<Position>, items...),
        writes(path<TwoRanges>),
        additiveNoise(GaussianParams<TwoRanges>{Eigen::Vector2d::Zero(), 0.04 * Eigen::Matrix2d::Identity()})
    );
}

// The distance between the two landmarks of the input, whatever the blocks they are read as.
template <typename... Items>
auto
spreadOf(Items const &... items)
{
    return measurementModel(
        []<typename T>(auto const & x) {
            return Eigen::Vector<T, 1>{
                T(Eigen::numext::sqrt(T((x.template segment<2>(2) - x.template head<2>()).squaredNorm())))
            };
        },
        reads(items...),
        writes(path<Spread>),
        additiveNoise(GaussianParams<Spread>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.09}})
    );
}

// The vehicle moves a part of the way to a point between the three landmarks after it in the input,
// weighted by the order they are read in, whatever the blocks they are read as.
template <typename... Items>
auto
approach(Items const &... items)
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            Eigen::Vector<T, 2> const goal =
                0.5 * x.template segment<2>(2) + 0.3 * x.template segment<2>(4) + 0.2 * x.template segment<2>(6);
            return Eigen::Vector<T, 2>{x.template head<2>() + 0.4 * dt * (goal - x.template head<2>())};
        },
        reads(path<Position>, items...),
        writes(path<Position>),
        continuousNoise(on<Position>(0.05))
    );
}

// The three landmarks of the input, read as the state's third, first, and second, each moved toward
// the origin and written in the state's order.
template <typename Written, typename... Items>
auto
settle(Written const & written, Items const &... items)
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            Eigen::Vector<T, 6> y;
            y << x.template segment<2>(2), x.template segment<2>(4), x.template head<2>();
            return Eigen::Vector<T, 6>{(1.0 - 0.1 * dt) * y};
        },
        reads(items...),
        written
    );
}

// The map's process model with the landmarks read by the selections \a toApproach and \a toSettle.
template <typename ToApproach, typename ToSettle>
auto
mapProcess(ToApproach const & toApproach, ToSettle const & toSettle)
{
    return compose(
        approach(path<Landmarks>[toApproach]),
        RandomWalk<Heading>{0.01},
        settle(writes(path<Landmarks>), path<Landmarks>[toSettle])
    );
}

// The same model of the state with a tag per landmark.
auto
taggedProcess()
{
    return compose(
        approach(path<Third>, path<First>, path<Second>),
        RandomWalk<Heading>{0.01},
        settle(writes(path<First>, path<Second>, path<Third>), path<Third>, path<First>, path<Second>)
    );
}

// The posteriors of two updates agree in their means and in their covariances' lower triangles,
// and the update moved the state from the mean \a prior.
template <typename Selected, typename ByTag, typename Prior>
void
expectSamePosterior(Selected const & selected, ByTag const & byTag, Prior const & prior)
{
    ASSERT_TRUE(selected);
    ASSERT_TRUE(byTag);
    auto const & a = selected->branches[0].posterior;
    auto const & b = byTag->branches[0].posterior;
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    Eigen::MatrixXd const lowerA = a.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = b.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
    EXPECT_GT((a.mean() - prior).norm(), 1e-3);
}

// Two predictions agree in their means and in their covariances' lower triangles, and the
// prediction moved the state from the mean \a prior.
template <typename Selected, typename ByTag, typename Prior>
void
expectSamePrediction(Selected const & selected, ByTag const & byTag, Prior const & prior)
{
    EXPECT_LT((selected.mean() - byTag.mean()).norm(), 1e-12);
    Eigen::MatrixXd const lowerA = selected.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = byTag.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
    EXPECT_GT((selected.mean() - prior).norm(), 1e-3);
}

// Sightings of the landmarks at repetitions 0 and 2 side by side, under two names, against
// sightings of the blocks First and Third.
template <typename Method>
void
expectComposedSightingsAreTheirBlocks(Method const & method)
{
    auto const selected = compose(
        sightingAs<Near, NearSight>(on<Near>(path<Landmarks>[0])),
        sightingAs<Far, FarSight>(on<Far>(path<Landmarks>[2]))
    );
    auto const byTag =
        compose(sightingAs<Near, NearSight>(on<Near>(path<First>)), sightingAs<Far, FarSight>(on<Far>(path<Third>)));
    Eigen::Vector4d const observed{9.5, 0.0, 10.0, -1.3};
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(method, selected, observed)),
        Emrys::update(taggedPrior(), observe(method, byTag, observed)),
        mapMean()
    );
}

}  // namespace

// A composition's reads() is the union of its parts' and holds each selection: the items two
// sightings share once, then each sighting's landmark with its repetition.
TEST(SelectedPlacement, ACompositionHoldsItsPartsSelections)
{
    auto const composed = compose(
        sightingAs<Near, NearSight>(on<Near>(path<Landmarks>[0])),
        sightingAs<Far, FarSight>(on<Far>(path<Landmarks>[2]))
    );
    using ReadsPolicy = decltype(composed)::Policy<Emrys::internal::PolicyKind::reads>;
    static_assert(std::is_same_v<ReadsPolicy::InputSpace, Composite<Position, Heading, Near, Far>>);
    static_assert(ReadsPolicy::Selected);
    auto const & policy = composed.policy<Emrys::internal::PolicyKind::reads>();
    EXPECT_EQ(policy.selection<2>(), 0);
    EXPECT_EQ(policy.selection<3>(), 2);

    // A composition of process models holds them too, and one whose parts select nothing holds nothing.
    auto const process = mapProcess(std::array{2, 0, 1}, std::array{2, 0, 1});
    auto const & processReads = process.policy<Emrys::internal::PolicyKind::reads>();
    EXPECT_EQ(processReads.selection<1>(), (Eigen::Matrix<Eigen::Index, 3, 1>{2, 0, 1}));
    static_assert(
        std::is_empty_v<std::remove_cvref_t<decltype(taggedProcess().policy<Emrys::internal::PolicyKind::reads>())>>
    );
}

// Two sightings side by side, each of one repetition under its own name, update as the sightings of
// those repetitions' tagged blocks, by linearization and by sigma points.
TEST(SelectedPlacement, ComposedSightingsAreTheirTaggedBlocks)
{
    expectComposedSightingsAreTheirBlocks(methods::linearized());
    expectComposedSightingsAreTheirBlocks(methods::unscented());
}

// Two parts that read one repetition under one name read it once: a range and a bearing to the
// landmark at repetition 1 update as the range and the bearing to the block Second.
TEST(SelectedPlacement, PartsShareOneRepetitionUnderOneName)
{
    auto const selected = compose(rangeTo(path<Landmarks>[1]), bearingTo(path<Landmarks>[1]));
    auto const byTag = compose(rangeTo(on<Landmark>(path<Second>)), bearingTo(on<Landmark>(path<Second>)));
    using ReadsPolicy = decltype(selected)::Policy<Emrys::internal::PolicyKind::reads>;
    static_assert(std::is_same_v<ReadsPolicy::InputSpace, Composite<Position, Landmark, Heading>>);
    Eigen::Vector2d const observed{7.0, 1.7};
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(methods::unscented(), selected, observed)),
        Emrys::update(taggedPrior(), observe(methods::unscented(), byTag, observed)),
        mapMean()
    );
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(methods::linearized(), selected, observed)),
        Emrys::update(taggedPrior(), observe(methods::linearized(), byTag, observed)),
        mapMean()
    );
}

// Two parts that read several repetitions under one name may spell them in two ways, a sequence and a
// list, or a sequence that counts from the block's end: the selections are compared when they meet
// the state's count, and the parts update as the parts that read the tagged blocks.
TEST(SelectedPlacement, PartsShareSeveralRepetitionsSpelledTwoWays)
{
    auto const byTag = compose(twoRanges(path<Second>, path<Third>), spreadOf(path<Second>, path<Third>));
    Eigen::Vector3d const observed{7.0, 10.0, 17.0};
    auto const expected = Emrys::update(taggedPrior(), observe(methods::unscented(), byTag, observed));
    auto const lastTwo = std::array{1, 2};
    expectSamePosterior(
        Emrys::update(
            mapPrior(),
            observe(
                methods::unscented(),
                compose(twoRanges(path<Landmarks>[Eigen::seqN(1, 2)]), spreadOf(path<Landmarks>[lastTwo])),
                observed
            )
        ),
        expected,
        mapMean()
    );
    expectSamePosterior(
        Emrys::update(
            mapPrior(),
            observe(
                methods::unscented(),
                compose(twoRanges(path<Landmarks>[lastTwo]), spreadOf(path<Landmarks>[Eigen::placeholders::lastN(2)])),
                observed
            )
        ),
        expected,
        mapMean()
    );
}

// A measurement model placed on a block selects the repetitions it selected in that block: a
// sighting of repetition 1 placed on the map updates the larger state as the sighting of the block
// Second placed on the map with a tag per landmark, and so does a composition that is placed.
TEST(SelectedPlacement, APlacedMeasurementSelectsInItsBlock)
{
    auto const selected = measurementModel(
        sightingOf<Landmark>(),
        reads(path<Position>, path<Heading>, path<Landmarks>[1]),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    auto const byTag = measurementModel(
        sightingOf<Landmark>(),
        reads(path<Position>, path<Heading>, on<Landmark>(path<Second>)),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    auto const placed = on<Survey>(selected);
    EXPECT_EQ(placed.policy<Emrys::internal::PolicyKind::reads>().selection<2>(), 1);
    Eigen::Vector2d const observed{7.0, 1.7};
    expectSamePosterior(
        Emrys::update(sitePrior(), observe(methods::linearized(), placed, observed)),
        Emrys::update(taggedSitePrior(), observe(methods::linearized(), on<TaggedSurvey>(byTag), observed)),
        siteMean()
    );
    expectSamePosterior(
        Emrys::update(sitePrior(), observe(methods::unscented(), placed, observed)),
        Emrys::update(taggedSitePrior(), observe(methods::unscented(), on<TaggedSurvey>(byTag), observed)),
        siteMean()
    );

    auto const composed = on<Survey>(compose(rangeTo(path<Landmarks>[1]), bearingTo(path<Landmarks>[1])));
    auto const composedByTag =
        on<TaggedSurvey>(compose(rangeTo(on<Landmark>(path<Second>)), bearingTo(on<Landmark>(path<Second>))));
    expectSamePosterior(
        Emrys::update(sitePrior(), observe(methods::unscented(), composed, observed)),
        Emrys::update(taggedSitePrior(), observe(methods::unscented(), composedByTag, observed)),
        siteMean()
    );
}

// A composition of process models whose parts read a list of repetitions under one name, in an
// order that is not the state's, predicts as the composition that reads the tagged blocks in that
// order, by linearization and by sigma points.
TEST(SelectedPlacement, AComposedProcessModelReadsASelection)
{
    auto const order = std::array{2, 0, 1};
    auto const selected = mapProcess(order, order);
    auto const byTag = taggedProcess();
    expectSamePrediction(
        Emrys::predict(methods::linearized(), mapPrior(), selected, 0.5),
        Emrys::predict(methods::linearized(), taggedPrior(), byTag, 0.5),
        mapMean()
    );
    expectSamePrediction(
        Emrys::predict(methods::unscented(), mapPrior(), selected, 0.5),
        Emrys::predict(methods::unscented(), taggedPrior(), byTag, 0.5),
        mapMean()
    );
}

// A composition of process models placed on a block selects in that block: the map's model placed
// on the map of a larger state predicts as the tagged model placed on the tagged map.
TEST(SelectedPlacement, APlacedProcessModelSelectsInItsBlock)
{
    auto const order = std::array{2, 0, 1};
    auto const selected = compose(RandomWalk<Drift>{0.02}, on<Survey>(mapProcess(order, order)));
    auto const byTag = compose(RandomWalk<Drift>{0.02}, on<TaggedSurvey>(taggedProcess()));
    expectSamePrediction(
        Emrys::predict(methods::linearized(), sitePrior(), selected, 0.5),
        Emrys::predict(methods::linearized(), taggedSitePrior(), byTag, 0.5),
        siteMean()
    );
    expectSamePrediction(
        Emrys::predict(methods::unscented(), sitePrior(), selected, 0.5),
        Emrys::predict(methods::unscented(), taggedSitePrior(), byTag, 0.5),
        siteMean()
    );
}

// Parts that read different repetitions under one name are refused when the composition meets the
// state: two integers, two lists of the same repetitions in another order, and the parts of a
// composition that is placed, of measurement models and of process models.
TEST(SelectedPlacement, PartsThatDisagreeAssert)
{
    Eigen::Vector2d const observed{7.0, 1.7};
    auto const integers = compose(rangeTo(path<Landmarks>[1]), bearingTo(path<Landmarks>[2]));
    EXPECT_DEATH(
        std::ignore = Emrys::update(mapPrior(), observe(methods::linearized(), integers, observed)),
        "select the same ones"
    );
    auto const lists =
        compose(twoRanges(path<Landmarks>[Eigen::seqN(1, 2)]), spreadOf(path<Landmarks>[std::array{2, 1}]));
    EXPECT_DEATH(
        std::ignore =
            Emrys::update(mapPrior(), observe(methods::linearized(), lists, Eigen::Vector3d{7.0, 10.0, 17.0})),
        "select the same ones"
    );
    EXPECT_DEATH(
        std::ignore = Emrys::update(sitePrior(), observe(methods::linearized(), on<Survey>(integers), observed)),
        "select the same ones"
    );
    auto const process =
        compose(RandomWalk<Drift>{0.02}, on<Survey>(mapProcess(std::array{2, 0, 1}, std::array{0, 1, 2})));
    EXPECT_DEATH(
        std::ignore = Emrys::predict(methods::linearized(), sitePrior(), process, 0.5),
        "select the same ones"
    );
}

// A composition reads no coordinate twice, as a model does not: a part that reads one repetition
// beside a part that reads the whole block under the block's name is refused when it meets the state.
TEST(SelectedPlacement, APartBesideOneThatReadsTheWholeBlockAsserts)
{
    auto const overlapping = compose(rangeTo(path<Landmarks>[1]), spreadOf(path<Landmarks>));
    EXPECT_DEATH(
        std::ignore =
            Emrys::update(mapPrior(), observe(methods::linearized(), overlapping, Eigen::Vector2d{7.0, 17.0})),
        "must not read a coordinate twice"
    );
}

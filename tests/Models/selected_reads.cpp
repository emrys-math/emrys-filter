// Items of reads() that select repetitions of a repeated block (roadmap step 17zu, decisions.md
// [selected-reads], [repetition-selections]): path<Landmarks>[k], a sequence, and a list, read by a
// measurement model. A state written with one tag per repetition, Composite<Position, Heading, First,
// Second, Third>, has the coordinates of Composite<Position, Heading, Landmarks> with three landmarks,
// in the same order, so an update whose model reads a selection must equal the update whose model reads
// the tagged blocks, under every method; the two differ only in the sizes Eigen fixes at compile time,
// hence the tolerance. Also covered: the names and types of the items, the selection held by value, a
// name that bounds the count, affineIn() of a selection, and the assertions.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <vector>

using Emrys::additiveNoise;
using Emrys::affineIn;
using Emrys::block;
using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::Reads;
using Emrys::Repeated;
using Emrys::TagPath;

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

struct Range : Linear<1>
{};

struct Bearing : RadiansPi<>
{};

using RangeBearing = Composite<Range, Bearing>;

struct Ranges : Repeated<Range, 6>
{
    using Repeated::Repeated;
};

/// a name for several landmarks that bounds their count at two
struct Sighted : Repeated<Landmark, 2>
{
    using Repeated::Repeated;
};

using Vector9 = Eigen::Matrix<double, 9, 1>;
using Matrix9 = Eigen::Matrix<double, 9, 9>;

// A vehicle at (1, 2) heading 0.3, and landmarks at (10, 4), (-3, 8), and (6, -7).
Vector9
priorMean()
{
    Vector9 mean;
    mean << 1.0, 2.0, 0.3, 10.0, 4.0, -3.0, 8.0, 6.0, -7.0;
    return mean;
}

// A covariance with every block correlated with every other: A A^T + I / 2 for a fixed full A.
Matrix9
priorCovariance()
{
    Matrix9 a;
    for (Eigen::Index i = 0; i < 9; ++i)
    {
        for (Eigen::Index j = 0; j < 9; ++j)
        {
            a(i, j) = 0.3 * std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
        }
    }
    return a * a.transpose() + 0.5 * Matrix9::Identity();
}

GaussianParams<Map>
mapPrior()
{
    return {Map{Position{}, Heading{}, Landmarks{Landmark{}, 3}}, priorMean(), priorCovariance()};
}

GaussianParams<Tagged>
taggedPrior()
{
    return {priorMean(), priorCovariance()};
}

// The range and the bearing from the vehicle to the landmark read under the name Landmark.
auto const sighting = []<typename T>(auto const & in, auto const & x) {
    Eigen::Vector<T, 2> const d = block<Landmark>(in, x) - block<Position>(in, x);
    return Eigen::Vector<T, 2>{
        T(Eigen::numext::sqrt(T(d.squaredNorm()))),
        T(Eigen::numext::atan2(d(1), d(0)) - block<Heading>(in, x)(0))
    };
};

// The ranges from the vehicle, the first block of the input, to each landmark after it, two
// coordinates each, whatever the blocks they are read as.
auto const ranges = []<typename T>(auto const & x) {
    Eigen::Index const count = (x.size() - 2) / 2;
    Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 6, 1> result(count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        result(i) = T(Eigen::numext::sqrt(T((x.template segment<2>(2 + 2 * i) - x.template head<2>()).squaredNorm())));
    }
    return result;
};

Eigen::Matrix2d const sightingNoise = Eigen::Vector2d{0.04, 0.0025}.asDiagonal();
Eigen::Vector2d const sightingObserved{6.5, 2.2};

// The posteriors of two updates agree in their means and in their covariances' lower triangles.
template <typename Selected, typename ByTag>
void
expectSamePosterior(Selected const & selected, ByTag const & byTag)
{
    ASSERT_TRUE(selected);
    ASSERT_TRUE(byTag);
    auto const & a = selected->branches[0].posterior;
    auto const & b = byTag->branches[0].posterior;
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    Eigen::MatrixXd const lowerA = a.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = b.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
    // The update moved the state: the comparison is not of two priors.
    EXPECT_GT((a.mean() - priorMean()).norm(), 1e-3);
}

// A sighting of the landmark at repetition 1 against a sighting of the block Second.
template <typename Method>
void
expectOneRepetitionIsItsBlock(Method const & method)
{
    auto const selected = measurementModel(
        sighting,
        reads(path<Position>, path<Heading>, path<Landmarks>[1]),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    auto const byTag = measurementModel(
        sighting,
        reads(path<Position>, path<Heading>, on<Landmark>(path<Second>)),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(method, selected, sightingObserved)),
        Emrys::update(taggedPrior(), observe(method, byTag, sightingObserved))
    );
}

// The ranges to the landmarks that \a selection names against the ranges to the blocks \a Tags.
template <typename... Tags, typename Method, typename Selection>
void
expectSeveralAreTheirBlocks(Method const & method, Selection const & selection)
{
    Eigen::Index const count = sizeof...(Tags);
    Eigen::MatrixXd const noise = 0.04 * Eigen::MatrixXd::Identity(count, count);
    Eigen::VectorXd const observed = Eigen::VectorXd::LinSpaced(count, 9.0, 11.0);
    auto const selected = measurementModel(
        ranges,
        reads(path<Position>, path<Landmarks>[selection]),
        outputSpace(Ranges{Range{}, count}),
        additiveNoise(noise)
    );
    auto const byTag = measurementModel(
        ranges,
        reads(path<Position>, path<Tags>...),
        outputSpace(Ranges{Range{}, count}),
        additiveNoise(noise)
    );
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(method, selected, observed)),
        Emrys::update(taggedPrior(), observe(method, byTag, observed))
    );
}

}  // namespace

// An item's name and what it holds: one repetition is read under the repetition's tag, several
// under the repeated block's own, and either under the name on<Name>() gives; the policy holds each
// selection by value, and has no size where no item has one.
TEST(SelectedReads, ItemsNamesAndSelections)
{
    auto const one = reads(path<Position>, path<Landmarks>[2]);
    static_assert(std::is_same_v<
                  decltype(one),
                  Reads<
                      Emrys::internal::ReadItem<TagPath<Position>, Position>,
                      Emrys::internal::ReadItem<TagPath<Landmarks>, Landmark, Eigen::Index>> const>);
    static_assert(std::is_same_v<decltype(one)::InputSpace, Composite<Position, Landmark>>);
    EXPECT_EQ(one.selection<1>(), 2);

    auto const several = reads(path<Landmarks>[{2, 0}]);
    static_assert(std::is_same_v<decltype(several)::InputSpace, Composite<Landmarks>>);
    EXPECT_EQ(several.selection<0>(), (Eigen::Matrix<Eigen::Index, 2, 1>{2, 0}));

    auto const named = reads(on<Sighted>(path<Landmarks>[Eigen::seqN(0, 2)]), on<First>(path<Landmarks>[2]));
    static_assert(std::is_same_v<decltype(named)::InputSpace, Composite<Sighted, First>>);
    EXPECT_EQ(named.selection<1>(), 2);

    static_assert(std::is_empty_v<decltype(reads(path<Position>, path<Heading>))>);
    static_assert(!decltype(reads(path<Position>))::Selected && decltype(one)::Selected);
}

// A model that reads one repetition updates as the model that reads that repetition's tagged block,
// by linearization and by sigma points.
TEST(SelectedReads, OneRepetitionIsItsTaggedBlock)
{
    expectOneRepetitionIsItsBlock(methods::linearized());
    expectOneRepetitionIsItsBlock(methods::unscented());
    expectOneRepetitionIsItsBlock(methods::cubature());
}

// A sigma-point method evaluates a model that reads one repetition as often as the model that reads
// its tagged block, and less often than the model that reads every landmark to use one.
TEST(SelectedReads, OneRepetitionSpreadsOverWhatItReads)
{
    int calls = 0;
    auto const counted = [&calls]<typename T>(auto const & in, auto const & x) {
        ++calls;
        return sighting.template operator()<T>(in, x);
    };
    auto const countedWhole = [&calls]<typename T>(auto const & in, auto const & x) {
        ++calls;
        Eigen::Vector<T, 2> const d = block<Landmarks>(in, x, 1) - block<Position>(in, x);
        return Eigen::Vector<T, 2>{
            T(Eigen::numext::sqrt(T(d.squaredNorm()))),
            T(Eigen::numext::atan2(d(1), d(0)) - block<Heading>(in, x)(0))
        };
    };
    auto const selected = measurementModel(
        counted,
        reads(path<Position>, path<Heading>, path<Landmarks>[1]),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    auto const byTag = measurementModel(
        counted,
        reads(path<Position>, path<Heading>, on<Landmark>(path<Second>)),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    auto const whole = measurementModel(
        countedWhole,
        reads(path<Position>, path<Heading>, path<Landmarks>),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    ASSERT_TRUE(Emrys::update(mapPrior(), observe(methods::unscented(), selected, sightingObserved)));
    int const selectedCalls = std::exchange(calls, 0);
    ASSERT_TRUE(Emrys::update(taggedPrior(), observe(methods::unscented(), byTag, sightingObserved)));
    int const byTagCalls = std::exchange(calls, 0);
    ASSERT_TRUE(Emrys::update(mapPrior(), observe(methods::unscented(), whole, sightingObserved)));
    int const wholeCalls = std::exchange(calls, 0);
    EXPECT_EQ(selectedCalls, byTagCalls);
    // Five coordinates are read, not nine: four fewer coordinates are eight fewer points.
    EXPECT_EQ(wholeCalls - selectedCalls, 8);
}

// A list is read in its order, as the tagged blocks named in that order, and a sequence as the
// tagged blocks it runs over, whether its bounds are integers or count from the block's end.
TEST(SelectedReads, SeveralRepetitionsAreTheirTaggedBlocks)
{
    auto const lastAndFirst = std::array{2, 0};
    expectSeveralAreTheirBlocks<Third, First>(methods::linearized(), lastAndFirst);
    expectSeveralAreTheirBlocks<Third, First>(methods::unscented(), lastAndFirst);
    expectSeveralAreTheirBlocks<Third, First>(methods::unscented(), std::vector<Eigen::Index>{2, 0});
    expectSeveralAreTheirBlocks<Second, Third>(methods::linearized(), Eigen::seqN(1, 2));
    expectSeveralAreTheirBlocks<Second, Third>(methods::unscented(), Eigen::seqN(1, 2));
    expectSeveralAreTheirBlocks<Second, Third>(methods::unscented(), Eigen::placeholders::lastN(2));
    expectSeveralAreTheirBlocks<First, Second, Third>(methods::cubature(), Eigen::seqN(0, 3));
}

// A linear model of the landmark's position relative to the vehicle's, given by its matrix, reads
// one repetition as it reads the tagged block.
TEST(SelectedReads, AMatrixModelReadsOneRepetition)
{
    Eigen::Matrix<double, 2, 4> h;
    h << -Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity();
    Eigen::Matrix2d const noise = 0.01 * Eigen::Matrix2d::Identity();
    Eigen::Vector2d const observed{4.5, -9.5};
    auto const selected = measurementModel(h, reads(path<Position>, path<Landmarks>[2]), additiveNoise(noise));
    auto const byTag = measurementModel(h, reads(path<Position>, path<Third>), additiveNoise(noise));
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(selected, observed)),
        Emrys::update(taggedPrior(), observe(byTag, observed))
    );
}

// A name that bounds the count more tightly than the state's block reads the same repetitions, and
// the function sees the name's block in its input.
TEST(SelectedReads, ANameBoundsTheCount)
{
    Eigen::Matrix2d const noise = 0.04 * Eigen::Matrix2d::Identity();
    Eigen::Vector2d const observed{9.0, 11.0};
    auto const lastAndFirst = std::array{2, 0};
    auto const bounded = measurementModel(
        ranges,
        reads(path<Position>, on<Sighted>(path<Landmarks>[lastAndFirst])),
        outputSpace(Ranges{Range{}, 2}),
        additiveNoise(noise)
    );
    auto const byTag = measurementModel(
        ranges,
        reads(path<Position>, path<Third>, path<First>),
        outputSpace(Ranges{Range{}, 2}),
        additiveNoise(noise)
    );
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(methods::unscented(), bounded, observed)),
        Emrys::update(taggedPrior(), observe(methods::unscented(), byTag, observed))
    );
}

// A selection named in affineIn() is factored out of the sigma points as its tagged block is: the
// sighting's offset is affine in the landmark given the vehicle.
TEST(SelectedReads, AffineInFactorsASelectionOut)
{
    auto const offset = []<typename T>(auto const & in, auto const & x) {
        T const heading = block<Heading>(in, x)(0);
        Eigen::Matrix<T, 2, 2> rotation;
        rotation << Eigen::numext::cos(heading), Eigen::numext::sin(heading), -Eigen::numext::sin(heading),
            Eigen::numext::cos(heading);
        return Eigen::Vector<T, 2>{rotation * (block<Landmark>(in, x) - block<Position>(in, x))};
    };
    Eigen::Matrix2d const noise = 0.01 * Eigen::Matrix2d::Identity();
    Eigen::Vector2d const observed{-2.0, 7.5};
    auto const selected = measurementModel(
        offset,
        reads(path<Position>, path<Heading>, path<Landmarks>[1]),
        affineIn(path<Landmark>, path<Position>),
        outputSpace(Linear<2>{}),
        additiveNoise(noise)
    );
    auto const byTag = measurementModel(
        offset,
        reads(path<Position>, path<Heading>, on<Landmark>(path<Second>)),
        affineIn(path<Landmark>, path<Position>),
        outputSpace(Linear<2>{}),
        additiveNoise(noise)
    );
    expectSamePosterior(
        Emrys::update(mapPrior(), observe(methods::unscented(), selected, observed)),
        Emrys::update(taggedPrior(), observe(methods::unscented(), byTag, observed))
    );
}

// A repetition the block lacks, and a coordinate read by two items, are refused when the model
// meets the state.
TEST(SelectedReads, SelectionAsserts)
{
    auto const beyond = measurementModel(
        sighting,
        reads(path<Position>, path<Heading>, path<Landmarks>[3]),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    EXPECT_DEATH(
        std::ignore = Emrys::update(mapPrior(), observe(methods::linearized(), beyond, sightingObserved)),
        "repetition out of range"
    );
    auto const twice = measurementModel(
        sighting,
        reads(path<Position>, path<Heading>, path<Landmarks>[1], on<Sighted>(path<Landmarks>[Eigen::seqN(0, 2)])),
        outputSpace(RangeBearing{}),
        additiveNoise(sightingNoise)
    );
    EXPECT_DEATH(
        std::ignore = Emrys::update(mapPrior(), observe(methods::linearized(), twice, sightingObserved)),
        "must not read a coordinate twice"
    );
}

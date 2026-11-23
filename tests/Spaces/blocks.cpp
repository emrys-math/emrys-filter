// The block accessors of Spaces/TagPath.h and Spaces/Selection.h (roadmap steps 17z and 17zu,
// decisions.md [tag-addressing], [index-names], [repetition-selections]): path<Tags...> as a value and PathValue, and
// its subscript, which selects repetitions of a repeated block; block() and diffBlock(), which give an Eigen vector's
// or matrix's block for a path with the rows the space's type fixes, in values and in differences, on a fixed space, a
// resizable one, a Repeated block, and a UnitQuaternion block, whose value and difference blocks differ; idx() and
// block() of one repetition, of a sequence, and of a list of them, through the tags that hold the repeated block; the
// space of several repetitions (Spaces/Subspace.h); the size check against the space; and zeros(). Expected addresses
// and sizes are counted by hand from the spaces' layouts.

#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Selection.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::block;
using Emrys::Composite;
using Emrys::diffBlock;
using Emrys::diffIdx;
using Emrys::idx;
using Emrys::Linear;
using Emrys::path;
using Emrys::PathValue;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::SelectedPath;
using Emrys::TagPath;
using Emrys::UnitQuaternion;
using Emrys::zeros;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Velocity, Bias>;

struct Heading : RadiansPi<>
{};

struct Landmarks : Linear<Eigen::Dynamic, 8>
{
    using Linear::Linear;
};

using World = Composite<Position, Landmarks, Heading>;

struct Vehicle : Composite<Position, Heading>
{};

struct Fleet : Repeated<Vehicle, 4>
{
    using Repeated::Repeated;
};

using Convoy = Composite<Position, Fleet>;

/// a fleet inside another tag, so that a path reaches the repeated block through it
struct Theater : Composite<Fleet, Bias>
{
    using Composite::Composite;
};

using Campaign = Composite<Position, Theater>;

struct Attitude : UnitQuaternion
{};

using Pose = Composite<Attitude, Position>;

struct Attitudes : Repeated<Attitude, 4>
{
    using Repeated::Repeated;
};

using Rig = Composite<Position, Attitudes>;

/// a column of indices, as idx() gives for a list of repetitions
template <int Size>
using Indices = Eigen::Matrix<Eigen::Index, Size, 1>;

struct Range : Linear<1>
{};

struct Bearing : RadiansPi<>
{};

using RangeBearing = Composite<Range, Bearing>;

struct Ownship : Composite<Position, Velocity>
{};

template <typename Block>
constexpr int rowsAtCompileTime = std::remove_cvref_t<Block>::RowsAtCompileTime;

template <typename Block>
constexpr int colsAtCompileTime = std::remove_cvref_t<Block>::ColsAtCompileTime;

}  // namespace

// path<Tags...> is the TagPath as a value, a bare tag's value is a path too, a library space type or
// a number is not, and on<Name>(path) names a path by another tag.
TEST(Blocks, PathsAreValues)
{
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(path<Position>)>, TagPath<Position>>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(path<Ownship, Position>)>, TagPath<Ownship, Position>>);
    static_assert(PathValue<decltype(path<Ownship, Position>)> && PathValue<Position> && PathValue<Fleet>);
    static_assert(!PathValue<Linear<2>> && !PathValue<int> && !PathValue<Composite<Position, Velocity>>);
    static_assert(std::is_same_v<Emrys::internal::PathOf<Position const &>, TagPath<Position>>);
    static_assert(
        std::is_same_v<Emrys::internal::PathOf<decltype(path<Ownship, Position>)>, TagPath<Ownship, Position>>
    );
    [[maybe_unused]] auto const renamed = Emrys::on<Range>(path<Ownship, Position>);
    static_assert(std::is_same_v<decltype(renamed), Emrys::internal::OnPath<TagPath<Ownship, Position>, Range> const>);
}

// On a fixed space, block() is a segment of the vector with the block's rows fixed at compile time,
// at the block's offset, writable where the vector is, in both forms; on a matrix of points it is
// those rows of every column; and diffBlock() gives the rows and columns of a covariance, or of a
// Jacobian whose rows are in another space.
TEST(Blocks, FixedSpace)
{
    auto x = Eigen::Vector<double, 5>{1.0, 2.0, 3.0, 4.0, 5.0};
    auto const velocity = block<Velocity>(State{}, x);
    static_assert(rowsAtCompileTime<decltype(velocity)> == 2);
    EXPECT_EQ(velocity.data(), x.data() + 2);
    EXPECT_EQ(velocity, Eigen::Vector2d(3.0, 4.0));
    block<State, Bias>(x)(0) = 7.0;
    EXPECT_EQ(x(4), 7.0);
    static_assert(rowsAtCompileTime<decltype(block<State, Bias>(x))> == 1);
    static_assert(std::is_same_v<
                  decltype(block<Position>(State{}, std::as_const(x))),
                  Eigen::Block<Eigen::Vector<double, 5> const, 2, 1>>);

    auto points = Eigen::Matrix<double, 5, 3>{};
    points.setZero();
    block<Position>(State{}, points).setOnes();
    EXPECT_EQ(points.topRows(2), (Eigen::Matrix<double, 2, 3>::Ones()));
    EXPECT_EQ(points.bottomRows(3), (Eigen::Matrix<double, 3, 3>::Zero()));
    static_assert(rowsAtCompileTime<decltype(block<Position>(State{}, points))> == 2);
    static_assert(colsAtCompileTime<decltype(block<Position>(State{}, points))> == 3);

    auto cov = Eigen::Matrix<double, 5, 5>{};
    cov.setZero();
    diffBlock<Position, Velocity>(State{}, cov) = Eigen::Matrix2d::Constant(0.5);
    EXPECT_EQ((cov.block<2, 2>(0, 2)), Eigen::Matrix2d::Constant(0.5));
    EXPECT_EQ(cov.sum(), 2.0);
    static_assert(rowsAtCompileTime<decltype(diffBlock<State, Position, Velocity>(cov))> == 2);
    static_assert(colsAtCompileTime<decltype(diffBlock<State, Position, Velocity>(cov))> == 2);
    EXPECT_EQ((diffBlock<State, Position, Velocity>(cov).data()), cov.data() + 2 * 5);
    EXPECT_EQ(diffBlock<Bias>(State{}, cov.col(0)).data(), cov.data() + 4);
    static_assert(rowsAtCompileTime<decltype(diffBlock<State, Bias>(cov.col(0)))> == 1);

    auto h = Eigen::Matrix<double, 2, 5>{};
    h.setZero();
    diffBlock<Bearing, Position>(RangeBearing{}, State{}, h) = Eigen::RowVector2d{1.0, -1.0};
    EXPECT_EQ(h.row(1).head<2>(), Eigen::RowVector2d(1.0, -1.0));
    EXPECT_EQ(h.sum(), 0.0);
    static_assert(rowsAtCompileTime<decltype(diffBlock<Bearing, Position>(RangeBearing{}, State{}, h))> == 1);
    static_assert(colsAtCompileTime<decltype(diffBlock<Bearing, Position>(RangeBearing{}, State{}, h))> == 2);
}

// On a resizable space the offset of a block after the resizable one is read from the space's
// sizes, a fixed block keeps its compile-time rows, and the resizable block's rows are dynamic.
TEST(Blocks, ResizableSpace)
{
    auto const space = World{Position{}, Landmarks{Eigen::Index{4}}, Heading{}};
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(7, 0.0, 6.0);
    auto const heading = block<Heading>(space, x);
    static_assert(rowsAtCompileTime<decltype(heading)> == 1);
    EXPECT_EQ(heading.data(), x.data() + 6);
    auto const landmarks = block<Landmarks>(space, x);
    static_assert(rowsAtCompileTime<decltype(landmarks)> == Eigen::Dynamic);
    EXPECT_EQ(landmarks.rows(), 4);
    EXPECT_EQ(landmarks.data(), x.data() + 2);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(7, 7);
    diffBlock<Heading, Landmarks>(space, cov).setOnes();
    EXPECT_EQ(cov.row(6).segment(2, 4).sum(), 4.0);
    EXPECT_EQ(cov.sum(), 4.0);
}

// A Repeated block's repetitions are reached one at a time, the whole repetition or a block inside
// it, in values and in differences, at the repetition's address.
TEST(Blocks, RepeatedBlock)
{
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 3}};
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(11, 0.0, 10.0);
    auto const second = block<Fleet>(convoy, x, 1);
    static_assert(rowsAtCompileTime<decltype(second)> == 3);
    EXPECT_EQ(second.data(), x.data() + 5);
    auto const heading = block<TagPath<Fleet, Heading>>(convoy, x, 2);
    static_assert(rowsAtCompileTime<decltype(heading)> == 1);
    EXPECT_EQ(heading.data(), x.data() + 10);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(11);
    diffBlock<TagPath<Fleet, Position>>(convoy, d, 0).setOnes();
    EXPECT_EQ(d.segment(2, 2).sum(), 2.0);
    EXPECT_EQ(d.sum(), 2.0);
}

// A path that runs through a repeated block takes a subscript: an integer selects one repetition,
// an Eigen sequence several, and a list those it holds, copied at the list's size where its type
// fixes it. The subscripted path is a SelectedPath, which is not a PathValue.
TEST(Blocks, PathSubscripts)
{
    auto const one = path<Fleet>[2];
    static_assert(std::is_same_v<decltype(one), SelectedPath<TagPath<Fleet>, Eigen::Index> const>);
    EXPECT_EQ(one.selection, 2);
    auto const heading = path<Fleet, Heading>[std::size_t{1}];
    static_assert(std::is_same_v<decltype(heading), SelectedPath<TagPath<Fleet, Heading>, Eigen::Index> const>);
    EXPECT_EQ(heading.selection, 1);
    static_assert(!PathValue<decltype(one)>);

    auto const run = path<Fleet>[Eigen::seqN(1, 2)];
    static_assert(std::is_same_v<decltype(run), SelectedPath<TagPath<Fleet>, decltype(Eigen::seqN(1, 2))> const>);
    EXPECT_EQ(run.selection.first(), 1);
    EXPECT_EQ(run.selection.size(), 2);
    [[maybe_unused]] auto const newest = path<Theater, Fleet>[Eigen::placeholders::lastN(2)];

    auto const braced = path<Fleet>[{2, 0}];
    static_assert(std::is_same_v<decltype(braced.selection), Indices<2>>);
    EXPECT_EQ(braced.selection, (Indices<2>{2, 0}));
    auto const bracedOne = path<Fleet>[{1}];
    static_assert(std::is_same_v<decltype(bracedOne.selection), Indices<1>>);
    EXPECT_EQ(bracedOne.selection(0), 1);

    auto const listed = path<Fleet>[std::vector<int>{3, 1, 0}];
    static_assert(std::is_same_v<decltype(listed.selection), Indices<Eigen::Dynamic>>);
    EXPECT_EQ(listed.selection, (Indices<3>{3, 1, 0}));
    auto const array = path<Fleet>[std::array<std::size_t, 2>{0, 3}];
    static_assert(std::is_same_v<decltype(array.selection), Indices<2>>);
    EXPECT_EQ(array.selection, (Indices<2>{0, 3}));
    auto const row = Eigen::Array<int, 1, 3>{2, 1, 0};
    auto const eigen = path<Fleet>[row];
    static_assert(std::is_same_v<decltype(eigen.selection), Indices<3>>);
    EXPECT_EQ(eigen.selection, (Indices<3>{2, 1, 0}));
}

// A repetition is reached through the tags that hold its repeated block, as through the block's
// own tag where that is unique: in Campaign the position is at 0, vehicle k at 2 + 3 k with its
// heading last, and the bias at 11.
TEST(Blocks, RepeatedBlockInsideATag)
{
    auto const campaign = Campaign{Position{}, Theater{Fleet{Vehicle{}, 3}, Bias{}}};
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(12, 0.0, 11.0);
    auto const second = block<TagPath<Theater, Fleet>>(campaign, x, 1);
    static_assert(rowsAtCompileTime<decltype(second)> == 3);
    EXPECT_EQ(second.data(), x.data() + 5);
    auto const heading = block<TagPath<Theater, Fleet, Heading>>(campaign, x, 2);
    static_assert(rowsAtCompileTime<decltype(heading)> == 1);
    EXPECT_EQ(heading.data(), x.data() + 10);
    EXPECT_EQ((idx<TagPath<Theater, Fleet, Heading>>(campaign, 0)), 4);
    EXPECT_EQ((diffIdx<TagPath<Theater, Fleet, Heading>>(campaign, 1)), 7);
    EXPECT_EQ(block<Fleet>(campaign, x, 2).data(), x.data() + 8);
    auto const fromType = idx<Campaign, TagPath<Theater, Fleet>>(1);
    EXPECT_EQ(fromType.first(), 5);
    EXPECT_EQ(fromType.size(), 3);
    auto const vehicle = Emrys::internal::pathElementSubspace<TagPath<Theater, Fleet>>(campaign);
    static_assert(std::is_same_v<decltype(vehicle), Vehicle const>);
    auto const run = idx<TagPath<Theater, Fleet>>(campaign, Eigen::seqN(1, 2));
    EXPECT_EQ(run.first(), 5);
    EXPECT_EQ(run.size(), 6);
}

// idx() and block() take several repetitions as an Eigen sequence or as a list. Whole repetitions
// of a sequence whose increment is one are one run of rows, of fixed size where the sequence's is;
// anything else is the coordinates' indices, repetition after repetition in the order named. In
// Convoy the position is at 0 and vehicle k at 2 + 3 k, its heading last.
TEST(Blocks, SeveralRepetitions)
{
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 4}};
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(14, 0.0, 13.0);

    auto const run = idx<Fleet>(convoy, Eigen::seqN(1, 2));
    EXPECT_EQ(run.first(), 5);
    EXPECT_EQ(run.size(), 6);
    auto const pair = block<Fleet>(convoy, x, Eigen::seqN(1, 2));
    static_assert(rowsAtCompileTime<decltype(pair)> == Eigen::Dynamic);
    EXPECT_EQ(pair.data(), x.data() + 5);
    EXPECT_EQ(pair.rows(), 6);
    auto const fixedPair = block<Fleet>(convoy, x, Eigen::seqN(1, Eigen::fix<2>));
    static_assert(rowsAtCompileTime<decltype(fixedPair)> == 6);
    EXPECT_EQ(fixedPair.data(), x.data() + 5);

    // The last two vehicles, counted from the block's end.
    auto const newest = block<Fleet>(convoy, x, Eigen::placeholders::lastN(2));
    EXPECT_EQ(newest.data(), x.data() + 8);
    EXPECT_EQ(newest.rows(), 6);

    // A list is read in its order: vehicle 3, then vehicle 0.
    auto const lastAndFirst = std::array{3, 0};
    auto const listed = idx<Fleet>(convoy, lastAndFirst);
    static_assert(std::is_same_v<decltype(listed), Indices<6> const>);
    EXPECT_EQ(listed, (Indices<6>{11, 12, 13, 2, 3, 4}));
    Eigen::VectorXd const picked = block<Fleet>(convoy, x, lastAndFirst);
    EXPECT_EQ(picked, (Eigen::Vector<double, 6>{11.0, 12.0, 13.0, 2.0, 3.0, 4.0}));

    // A block inside each repetition: the headings of a list and of a sequence, and the positions
    // of a sequence, whose rows are not one run.
    auto const thirdAndSecond = std::vector<int>{2, 1};
    EXPECT_EQ((idx<TagPath<Fleet, Heading>>(convoy, thirdAndSecond)), (Indices<2>{10, 7}));
    EXPECT_EQ((idx<TagPath<Fleet, Heading>>(convoy, Eigen::seqN(0, 3))), (Indices<3>{4, 7, 10}));
    EXPECT_EQ((idx<TagPath<Fleet, Position>>(convoy, Eigen::seqN(2, 2))), (Indices<4>{8, 9, 11, 12}));

    // Every other vehicle: a sequence whose increment is not one.
    EXPECT_EQ((idx<Fleet>(convoy, Eigen::seqN(0, 2, 2))), (Indices<6>{2, 3, 4, 8, 9, 10}));

    // A view of listed repetitions writes through to the vector.
    auto const firstAndLast = std::array{0, 3};
    block<TagPath<Fleet, Heading>>(convoy, x, firstAndLast).setZero();
    EXPECT_EQ(x(4), 0.0);
    EXPECT_EQ(x(13), 0.0);
    EXPECT_EQ(x(7), 7.0);

    // On a matrix of points, the repetitions' rows of every column.
    Eigen::MatrixXd points = Eigen::MatrixXd::Ones(14, 3);
    auto const middle = std::array{1, 2};
    auto const rows = block<Fleet>(convoy, points, middle);
    EXPECT_EQ(rows.rows(), 6);
    EXPECT_EQ(rows.cols(), 3);

    // From the type alone, where nothing resizable precedes the block.
    auto const second = std::array{1};
    EXPECT_EQ((idx<Convoy, Fleet>(second)), (Indices<3>{5, 6, 7}));
    auto const fromType = idx<Convoy, Fleet>(Eigen::seqN(2, 2));
    EXPECT_EQ(fromType.first(), 8);
    EXPECT_EQ(fromType.size(), 6);
    auto const lastAndSecond = std::array{3, 1};
    EXPECT_EQ((diffIdx<Convoy, TagPath<Fleet, Heading>>(lastAndSecond)), (Indices<2>{13, 7}));
}

// A repetition of a unit quaternion has four value rows and three difference rows, so the rows of
// several differ between a value and a covariance: in Rig the position is at 0, and attitude k at
// 2 + 4 k in a value and at 2 + 3 k in a difference.
TEST(Blocks, SeveralQuaternionRepetitions)
{
    auto const rig = Rig{Position{}, Attitudes{Attitude{}, 3}};
    auto const lastAndFirst = std::array{2, 0};
    EXPECT_EQ((idx<Attitudes>(rig, lastAndFirst)), (Indices<8>{10, 11, 12, 13, 2, 3, 4, 5}));
    EXPECT_EQ((diffIdx<Attitudes>(rig, lastAndFirst)), (Indices<6>{8, 9, 10, 2, 3, 4}));
    Eigen::VectorXd d = Eigen::VectorXd::Zero(11);
    diffBlock<Attitudes>(rig, d, Eigen::seqN(1, 2)).setOnes();
    EXPECT_EQ(d.segment(5, 6).sum(), 6.0);
    EXPECT_EQ(d.sum(), 6.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(14);
    auto const values = block<Attitudes>(rig, x, Eigen::seqN(1, 2));
    EXPECT_EQ(values.data(), x.data() + 6);
    EXPECT_EQ(values.rows(), 8);
}

// The space of several repetitions is the repeated block's own tag with their count, and of a
// block inside each, that block repeated as many times, up to the same maximum count.
TEST(Blocks, RepetitionsSubspace)
{
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 4}};
    auto const two = Emrys::internal::pathRepetitionsSubspace<Fleet>(convoy, 2);
    static_assert(std::is_same_v<decltype(two), Fleet const>);
    EXPECT_EQ(two.count(), 2);
    auto const headings = Emrys::internal::pathRepetitionsSubspace<TagPath<Fleet, Heading>>(convoy, 3);
    static_assert(std::is_same_v<decltype(headings), Repeated<Heading, 4> const>);
    EXPECT_EQ(headings.count(), 3);
}

// A unit quaternion block has four value rows and three difference rows, so the blocks after it
// start at different offsets in a value and in a covariance, and map() views the value block.
TEST(Blocks, UnitQuaternionBlock)
{
    auto x = Eigen::Vector<double, 6>{};
    x << 0.0, 0.0, 0.0, 1.0, 5.0, 6.0;
    static_assert(rowsAtCompileTime<decltype(block<Attitude>(Pose{}, x))> == 4);
    EXPECT_EQ(block<Position>(Pose{}, x).data(), x.data() + 4);
    auto d = Eigen::Vector<double, 5>{};
    d.setZero();
    static_assert(rowsAtCompileTime<decltype(diffBlock<Attitude>(Pose{}, d))> == 3);
    EXPECT_EQ(diffBlock<Position>(Pose{}, d).data(), d.data() + 3);
    Eigen::Matrix<double, 5, 5> cov = Eigen::Matrix<double, 5, 5>::Zero();
    diffBlock<Attitude, Position>(Pose{}, cov).setOnes();
    EXPECT_EQ((cov.block<3, 2>(0, 3)), (Eigen::Matrix<double, 3, 2>::Ones()));
    EXPECT_EQ(cov.sum(), 6.0);
    auto q = UnitQuaternion::map(block<Attitude>(Pose{}, x));
    EXPECT_EQ(q.coeffs().data(), x.data());
    q = Eigen::Quaterniond{Eigen::AngleAxisd{0.5, Eigen::Vector3d::UnitZ()}};
    EXPECT_NEAR(x(2), std::sin(0.25), 1e-15);
}

// zeros<T>() is a zero vector of the space's value dimension in the scalar T, fixed where the space
// is and sized from the instance, with the space's bound, where it is not.
TEST(Blocks, Zeros)
{
    auto y = zeros<float>(State{});
    static_assert(std::is_same_v<decltype(y), Eigen::Vector<float, 5>>);
    EXPECT_TRUE(y.isZero(0));
    auto const space = World{Position{}, Landmarks{Eigen::Index{4}}, Heading{}};
    auto v = zeros<double>(space);
    static_assert(std::is_same_v<decltype(v), Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 11, 1>>);
    EXPECT_EQ(v.size(), 7);
    EXPECT_TRUE(v.isZero(0));
    block<Heading>(space, v)(0) = 1.0;
    EXPECT_EQ(v.sum(), 1.0);
}

// A vector or matrix whose size is not the space's, and a repetition the block lacks, are refused
// where the sizes are known only at run time.
TEST(Blocks, SizeMismatchAsserts)
{
    auto const space = World{Position{}, Landmarks{Eigen::Index{4}}, Heading{}};
    Eigen::VectorXd const x = Eigen::VectorXd::Zero(6);
    EXPECT_DEATH(std::ignore = block<Heading>(space, x), "do not match the space");
    Eigen::MatrixXd const h = Eigen::MatrixXd::Zero(7, 6);
    EXPECT_DEATH((std::ignore = diffBlock<Heading, Position>(space, State{}, h)), "do not match the spaces");
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 2}};
    Eigen::VectorXd const vehicles = Eigen::VectorXd::Zero(8);
    EXPECT_DEATH(std::ignore = block<Fleet>(convoy, vehicles, 2), "repetition out of range");
}

// A list or a sequence that names a repetition the block lacks, one that names a repetition twice,
// and one that names more repetitions than the block has are refused.
TEST(Blocks, SelectionAsserts)
{
    auto const convoy = Convoy{Position{}, Fleet{Vehicle{}, 2}};
    Eigen::VectorXd const x = Eigen::VectorXd::Zero(8);
    auto const beyond = std::array{0, 2};
    EXPECT_DEATH(std::ignore = block<Fleet>(convoy, x, beyond), "repetition out of range");
    EXPECT_DEATH(std::ignore = idx<Fleet>(convoy, Eigen::seqN(1, 2)), "repetition out of range");
    auto const twice = std::array{1, 1};
    EXPECT_DEATH(std::ignore = idx<Fleet>(convoy, twice), "names a repetition twice");
    auto const tooMany = std::vector<int>{0, 1, 0};
    EXPECT_DEATH(std::ignore = idx<Fleet>(convoy, tooMany), "more repetitions than the block has");
}

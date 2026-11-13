// The block accessors of Spaces/TagPath.h (roadmap step 17z, decisions.md [tag-addressing],
// [index-names]): path<Tags...> as a value and PathValue; block() and diffBlock(), which give an
// Eigen vector's or matrix's block for a path with the rows the space's type fixes, in values and
// in differences, on a fixed space, a resizable one, a Repeated block, and a UnitQuaternion block,
// whose value and difference blocks differ; the size check against the space; and zeros().
// Expected addresses and sizes are counted by hand from the spaces' layouts.

#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

using Emrys::block;
using Emrys::Composite;
using Emrys::diffBlock;
using Emrys::Linear;
using Emrys::path;
using Emrys::PathValue;
using Emrys::RadiansPi;
using Emrys::Repeated;
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

struct Attitude : UnitQuaternion
{};

using Pose = Composite<Attitude, Position>;

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
    auto const renamed = Emrys::on<Range>(path<Ownship, Position>);
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

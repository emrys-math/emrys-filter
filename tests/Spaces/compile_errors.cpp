// The compile errors that the spaces claim (Spaces/Coordinates.h, Composite.h, Discrete.h,
// TagPath.h with its block accessors, UnitQuaternion.h): each case below is a misuse that must fail to compile with the
// static_assert or deleted function it names, checked by tests/compile_errors.cmake. Without a case
// the file compiles, which checks what the cases share. The evidence for [coverage]'s negative
// compile tests and for the messages of [tag-addressing] and [discrete-spaces].

#include <EmrysFilter/Spaces.h>

#include <Eigen/Core>

#include <limits>

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<>
{};

struct Attitude : Emrys::UnitQuaternion
{};

/// a resizable block, which makes the offsets of the blocks after it run-time values
struct Landmarks : Emrys::Linear<Eigen::Dynamic, 8>
{};

struct Vehicle : Emrys::Composite<Position, Heading>
{};

struct Fleet : Emrys::Repeated<Vehicle>
{
    using Repeated::Repeated;
};

struct Cruise
{};

struct Turn
{};

struct Hover
{};

struct Maneuver : Emrys::Named<Cruise, Turn>
{};

struct Fault : Emrys::Named<Cruise, Hover>
{};

struct Cell : Emrys::Finite<4>
{};

using State = Emrys::Composite<Position, Heading>;
using Pose = Emrys::Composite<Attitude, Position>;

[[maybe_unused]] void
everyCaseSharesThis()
{
    [[maybe_unused]] constexpr auto heading = Emrys::idx<State, Heading>();
    [[maybe_unused]] auto const maneuver = Emrys::valueOf<Turn>(Maneuver{});
    [[maybe_unused]] constexpr auto cell =
        Emrys::valueOf<Emrys::DiscreteProduct<Maneuver, Cell>, Turn>(Emrys::at<Cell>(3));
}

#if defined(LINEAR_FIXED_MAXIMUM)  // static_assert: Coordinates.h: a fixed size is its own maximum
[[maybe_unused]] Emrys::Linear<2, 3> const space;
#endif

#if defined(PERIODIC_FIXED_MAXIMUM)  // static_assert: Coordinates.h: a fixed size is its own maximum
[[maybe_unused]] Emrys::Periodic<{.lower = 0.0, .upper = 1.0}, 2, 3> const space;
#endif

#if defined(PERIODIC_NO_LOWER)  // static_assert: Coordinates.h: the lower end must be given
[[maybe_unused]] Emrys::Periodic<{.upper = 1.0}> const space;
#endif

#if defined(PERIODIC_NO_UPPER)  // static_assert: Coordinates.h: the upper end must be given
[[maybe_unused]] Emrys::Periodic<{.lower = 0.0}> const space;
#endif

#if defined(PERIODIC_EMPTY)  // static_assert: Coordinates.h: the upper end must be above the lower
[[maybe_unused]] Emrys::Periodic<{.lower = 1.0, .upper = 0.0}> const space;
#endif

#if defined(PERIODIC_INFINITE)  // static_assert: Coordinates.h: the period must be finite
[[maybe_unused]] Emrys::Periodic<{.lower = 0.0, .upper = std::numeric_limits<double>::infinity()}> const space;
#endif

#if defined(COMPOSITE_EMPTY)  // static_assert: Composite.h: a composite needs at least one block
[[maybe_unused]] Emrys::Composite<> const space;
#endif

#if defined(REPEATED_RESIZABLE)  // static_assert: Composite.h: a repeated block must have a fixed, nonzero size
[[maybe_unused]] Emrys::Repeated<Landmarks> const space;
#endif

#if defined(FINITE_EMPTY)  // static_assert: Discrete.h: a discrete space has at least one value
[[maybe_unused]] Emrys::Finite<0> const space;
#endif

#if defined(FINITE_FIXED_MAXIMUM)  // static_assert: Discrete.h: a fixed cardinality is its own maximum
[[maybe_unused]] Emrys::Finite<2, 3> const space;
#endif

#if defined(NAMED_EMPTY)  // static_assert: Discrete.h: a discrete space has at least one value
[[maybe_unused]] Emrys::Named<> const space;
#endif

#if defined(NAMED_REPEATED)  // static_assert: Discrete.h: the values of a Named space must be distinct
[[maybe_unused]] Emrys::Named<Cruise, Cruise> const space;
#endif

#if defined(NAMED_NO_SUCH_VALUE)  // static_assert: Discrete.h: the space has no value with this name
[[maybe_unused]] constexpr auto index = Maneuver::index<Hover>();
#endif

#if defined(DISCRETE_PRODUCT_EMPTY)  // static_assert: Discrete.h: a product needs at least one axis
[[maybe_unused]] Emrys::DiscreteProduct<> const space;
#endif

#if defined(VALUE_OF_NOTHING)  // static_assert: Discrete.h: name a value
[[maybe_unused]] auto const value = Emrys::valueOf<>(Maneuver{});
#endif

#if defined(VALUE_OF_MISSING_VALUE)  // static_assert: Discrete.h: the space has no value with this name
[[maybe_unused]] auto const value = Emrys::valueOf<Hover>(Maneuver{});
#endif

#if defined(VALUE_OF_AMBIGUOUS_VALUE)  // static_assert: Discrete.h: the value names a value of more than one axis
[[maybe_unused]] auto const value = Emrys::valueOf<Cruise>(Emrys::DiscreteProduct<Maneuver, Fault>{});
#endif

#if defined(VALUE_OF_UNTAGGED_AXIS)  // static_assert: Discrete.h: a path names axes by tag
[[maybe_unused]] auto const value = Emrys::valueOf<Emrys::TagPath<Emrys::Named<Cruise, Turn>, Turn>>(
    Emrys::DiscreteProduct<Emrys::Named<Cruise, Turn>, Fault>{}
);
#endif

#if defined(VALUE_OF_MISSING_AXIS)  // static_assert: Discrete.h: the space does not have this axis
[[maybe_unused]] auto const value = Emrys::valueOf<Emrys::TagPath<Fault, Hover>>(Emrys::DiscreteProduct<Maneuver>{});
#endif

#if defined(VALUE_OF_REPEATED_AXIS)  // static_assert: Discrete.h: the axis appears more than once
[[maybe_unused]] auto const value =
    Emrys::valueOf<Emrys::TagPath<Maneuver, Turn>>(Emrys::DiscreteProduct<Maneuver, Maneuver>{});
#endif

#if defined(VALUE_OF_AXIS_TWICE)  // static_assert: Discrete.h: an axis is named more than once
[[maybe_unused]] constexpr auto value = Emrys::valueOf<Maneuver, Turn, Cruise>();
#endif

#if defined(VALUE_OF_AXIS_LEFT_OUT)  // static_assert: Discrete.h: name one value in each axis
[[maybe_unused]] constexpr auto value = Emrys::valueOf<Emrys::DiscreteProduct<Maneuver, Fault>, Turn>();
#endif

#if defined(VALUE_OF_UNNAMED_AXIS_LEFT_OUT)  // static_assert: Discrete.h: an axis without names takes its index from
[[maybe_unused]] constexpr auto value = Emrys::valueOf<Emrys::DiscreteProduct<Maneuver, Cell>, Turn>();
#endif

#if defined(AT_NOT_DISCRETE)  // static_assert: Discrete.h: at<Axis>() names an axis of a discrete space
[[maybe_unused]] constexpr auto index = Emrys::at<Position>(0);
#endif

#if defined(AT_INDEX_COUNT)  // static_assert: Discrete.h: give one index for each leaf axis of the axis named
[[maybe_unused]] constexpr auto index = Emrys::at<Emrys::DiscreteProduct<Maneuver, Cell>>(1);
#endif

#if defined(IDX_UNTAGGED_BLOCK)  // static_assert: TagPath.h: a path names blocks by tag
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Emrys::Linear<2>, Heading>, Emrys::Linear<2>>();
#endif

#if defined(IDX_MISSING_TAG)  // static_assert: TagPath.h: the space does not contain this tag
[[maybe_unused]] constexpr auto index = Emrys::idx<State, Attitude>();
#endif

#if defined(IDX_REPEATED_TAG)  // static_assert: TagPath.h: the tag appears more than once
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Vehicle, Emrys::Composite<Heading>>, Heading>();
#endif

#if defined(IDX_INTO_REPETITION)  // static_assert: TagPath.h: a path through a repeated block must end at a single
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Fleet>, Emrys::TagPath<Fleet, Position>>();
#endif

#if defined(IDX_INTO_REPETITION_AT_RUN_TIME)  // static_assert: TagPath.h: a path through a repeated block must end at
[[maybe_unused]] auto const index =
    Emrys::idx<Emrys::TagPath<Fleet, Position>>(Emrys::Composite<Fleet>{Fleet{Vehicle{}, 2}});
#endif

#if defined(IDX_AFTER_RESIZABLE)  // static_assert: TagPath.h: this path's offset depends on the size of a resizable
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Landmarks, Heading>, Heading>();
#endif

#if defined(IDX_REPEATED_NOT_LAST)  // static_assert: TagPath.h: this repeated block is not at the end of the space
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Fleet, Attitude>, Emrys::TagPath<Fleet, Heading>>();
#endif

#if defined(IDX_RESIZABLE_NOT_LAST)  // static_assert: TagPath.h: this resizable block is not at the end of the space
[[maybe_unused]] constexpr auto index = Emrys::idx<Emrys::Composite<Landmarks, Heading>, Landmarks>();
#endif

#if defined(BLOCK_ROWS_MISSING_TAG)  // static_assert: TagPath.h: the space does not contain this tag
[[maybe_unused]] auto const rows = Emrys::internal::blockRows<Attitude>(State{});
#endif

#if defined(BLOCK_ROWS_INSIDE_REPETITION)  // static_assert: TagPath.h: this path continues inside a repeated block
[[maybe_unused]] auto const rows =
    Emrys::internal::blockRows<Emrys::TagPath<Fleet, Heading>>(Emrys::Composite<Fleet>{Fleet{Vehicle{}, 2}});
#endif

#if defined(TAG_PATH_EMPTY)  // static_assert: TagPath.h: a tag path needs at least one tag
[[maybe_unused]] Emrys::TagPath<> const path;
#endif

#if defined(IDX_ELEMENT_OF_UNREPEATED)  // static_assert: TagPath.h: the element form takes a path that starts at a
[[maybe_unused]] auto const index = Emrys::idx<State, Heading>(0);
#endif

#if defined(BLOCK_ELEMENT_OF_UNREPEATED)  // static_assert: TagPath.h: the element form takes a path that starts at a
[[maybe_unused]] void
blockElement()
{
    Eigen::Vector<double, 3> x = Eigen::Vector<double, 3>::Zero();
    [[maybe_unused]] auto const heading = Emrys::block<Heading>(State{}, x, 0);
}
#endif

#if defined(BLOCK_INSIDE_REPETITION)  // static_assert: TagPath.h: this path continues inside a repeated block
[[maybe_unused]] void
blockRepetition()
{
    auto const space = Emrys::Composite<Fleet>{Fleet{Vehicle{}, 2}};
    Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
    [[maybe_unused]] auto const heading = Emrys::block<Emrys::TagPath<Fleet, Heading>>(space, x);
}
#endif

#if defined(BLOCK_ROWS_OF_ANOTHER_SPACE)  // static_assert: TagPath.h: the rows do not match the space's dimension
[[maybe_unused]] void
blockRows()
{
    Eigen::Vector<double, 4> x = Eigen::Vector<double, 4>::Zero();
    [[maybe_unused]] auto const heading = Emrys::block<Heading>(State{}, x);
}
#endif

#if defined(DIFF_BLOCK_ROWS_OF_VALUES)  // static_assert: TagPath.h: the rows do not match the space's dimension
[[maybe_unused]] void
diffBlockRows()
{
    Eigen::Vector<double, 6> x = Eigen::Vector<double, 6>::Zero();
    [[maybe_unused]] auto const attitude = Emrys::diffBlock<Attitude>(Pose{}, x);
}
#endif

#if defined(DIFF_BLOCK_COLUMNS_OF_ANOTHER_SPACE)  // static_assert: TagPath.h: the columns do not match the space's
[[maybe_unused]] void
diffBlockColumns()
{
    Eigen::Matrix<double, 5, 5> h = Eigen::Matrix<double, 5, 5>::Zero();
    [[maybe_unused]] auto const columns = Emrys::diffBlock<Attitude, Position>(Pose{}, State{}, h);
}
#endif

#if defined(BLOCK_TEMPORARY)  // static_assert: TagPath.h: a block of a temporary matrix would dangle
[[maybe_unused]] auto const position = Emrys::block<State, Position>(Eigen::Vector<double, 3>::Zero().eval());
#endif

#if defined(DIFF_BLOCK_TEMPORARY)  // static_assert: TagPath.h: a block of a temporary matrix would dangle
[[maybe_unused]] auto const position =
    Emrys::diffBlock<Pose, Attitude, Position>(Eigen::Matrix<double, 6, 6>::Zero().eval());
#endif

#if defined(IDX_ELEMENT_AFTER_RESIZABLE)  // static_assert: TagPath.h: this block's offset depends on the size of a
[[maybe_unused]] auto const index = Emrys::idx<Emrys::Composite<Landmarks, Fleet>, Fleet>(0);
#endif

#if defined(MAP_NOT_A_MATRIX)  // static_assert: UnitQuaternion.h: map() takes an Eigen vector of four coordinates
[[maybe_unused]] void
mapNumber()
{
    double coefficient = 1.0;
    [[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(coefficient);
}
#endif

#if defined(MAP_THREE_COORDINATES)  // static_assert: UnitQuaternion.h: map() takes one column of four coordinates
[[maybe_unused]] void
mapThree()
{
    Eigen::Vector3d coefficients = Eigen::Vector3d::Zero();
    [[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(coefficients);
}
#endif

#if defined(MAP_EXPRESSION)  // static_assert: UnitQuaternion.h: map() needs coefficients in memory with unit stride
[[maybe_unused]] void
mapExpression()
{
    Eigen::Vector4d coefficients = Eigen::Vector4d::UnitW();
    [[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(2.0 * coefficients);
}
#endif

#if defined(MAP_TEMPORARY)  // static_assert: UnitQuaternion.h: map() of a temporary matrix would dangle
[[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(Eigen::Vector4d::UnitW().eval());
#endif

#if defined(MAP_HEADING_BLOCK)  // static_assert: UnitQuaternion.h: map() takes one column of four coordinates
[[maybe_unused]] void
mapHeading()
{
    Eigen::Vector<double, 3> x = Eigen::Vector<double, 3>::Zero();
    [[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(Emrys::block<Heading>(State{}, x));
}
#endif

#if defined(MAP_STATE_OF_TWO_COLUMNS)  // static_assert: UnitQuaternion.h: map() takes one column of four coordinates
[[maybe_unused]] void
mapColumns()
{
    Eigen::Matrix<double, 6, 2> x = Eigen::Matrix<double, 6, 2>::Zero();
    [[maybe_unused]] auto const q = Emrys::UnitQuaternion::map(Emrys::block<Attitude>(Pose{}, x));
}
#endif

}  // namespace

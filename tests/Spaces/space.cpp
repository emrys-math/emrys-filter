// The continuous spaces of Spaces/Space.h, Coordinates.h, Composite.h, TagPath.h, and
// Subspace.h (steps 5a, 11; [discrete-spaces], [sizes-and-dimensions], [no-bounded-coordinates]): sizes, strides, and
// compile-time bounds of Linear, Periodic, Composite, and Repeated; wrapping of values and
// differences; tag paths and projections [index-names]; the Space concept, what satisfies it and what a
// user-written space must provide; and the plus()/minus() operations a space may define, which
// Gaussian, ParticleSet, and Sampling then use.

#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>
#include <vector>

using Emrys::allowsValueDimension;
using Emrys::Composite;
using Emrys::Degrees180;
using Emrys::Degrees360;
using Emrys::DifferenceSpace;
using Emrys::differenceSpace;
using Emrys::FixedSizeSpace;
using Emrys::GaussianParams;
using Emrys::hasPlainDifferences;
using Emrys::hasPlainValues;
using Emrys::idx;
using Emrys::isLinear;
using Emrys::Linear;
using Emrys::minus;
using Emrys::normalizeDifferences;
using Emrys::normalizeValues;
using Emrys::ParticleSet;
using Emrys::Periodic;
using Emrys::plus;
using Emrys::Radians2Pi;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::Space;
using Emrys::TagPath;
using Emrys::valueDimension;
using Emrys::valueDimensionAtCompileTime;
using Emrys::writable;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<3>
{};

struct Heading : RadiansPi<>
{};

struct Range : Linear<1>
{};

using State = Composite<Position, Heading, Range>;

struct Ownship : Linear<6>
{};

struct Latitude : Linear<1>
{};

struct Longitude : RadiansPi<>
{};

struct Altitude : Linear<1>
{};

struct Vehicle : Composite<Latitude, Longitude, Altitude>
{};

struct Vehicles : Repeated<Vehicle>
{};

using World = Composite<Ownship, Vehicles>;

// Two groups that both contain an X, so a bare X is ambiguous.
struct X : Linear<1>
{};

struct Y : Linear<1>
{};

struct Left : Composite<X, Y>
{};

struct Right : Composite<X>
{};

using Sides = Composite<Left, Right>;

// The default space takes no storage.
struct WithEmptySpace
{
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Linear<4> space;
    double value;
};

static_assert(sizeof(WithEmptySpace) == sizeof(double));

// A composite stores only its blocks with state: fixed-size composites, nested or not, are
// empty, and a resizable one is just its count.
static_assert(std::is_empty_v<State> && std::is_empty_v<Vehicle> && std::is_empty_v<Sides>);

static_assert(std::is_empty_v<Composite<X, X>>);
static_assert(sizeof(World) == sizeof(Eigen::Index) && sizeof(Vehicles) == sizeof(Eigen::Index));
static_assert(sizeof(GaussianParams<State>) == sizeof(GaussianParams<Linear<5>>));

Eigen::VectorXd
iota(Eigen::Index n)
{
    return Eigen::VectorXd::LinSpaced(n, 0.0, double(n - 1));
}

}  // namespace

// FixedValueDimension, ValueDimensionStep, and valueDimensionAtCompileTime<> of Linear, a fixed
// Composite, and one with a Repeated block; isLinear<> of each; and allowsValueDimension() accepts
// only the fixed part plus whole repetitions.
TEST(Space, ValueDimensionsAndSteps)
{
    static_assert(Linear<3>::FixedValueDimension == 3 && Linear<3>::ValueDimensionStep == 0);
    static_assert(Linear<Eigen::Dynamic>::FixedValueDimension == 0 && Linear<Eigen::Dynamic>::ValueDimensionStep == 1);
    static_assert(
        State::FixedValueDimension == 5 && State::ValueDimensionStep == 0 && valueDimensionAtCompileTime<State> == 5
    );
    static_assert(
        World::FixedValueDimension == 6 && World::ValueDimensionStep == 3
        && valueDimensionAtCompileTime<World> == Eigen::Dynamic
    );
    static_assert(!isLinear<State> && isLinear<Composite<Position, Altitude>> && isLinear<Repeated<Ownship>>);

    EXPECT_TRUE(allowsValueDimension<State>(5));
    EXPECT_FALSE(allowsValueDimension<State>(6));
    EXPECT_TRUE(allowsValueDimension<World>(6));
    EXPECT_TRUE(allowsValueDimension<World>(12));
    EXPECT_FALSE(allowsValueDimension<World>(10));
    EXPECT_FALSE(allowsValueDimension<World>(3));
}

// MaxValueDimension and maxValueDimensionAtCompileTime<> are Dynamic for an uncapped Repeated and
// the sum of bounds for a capped one or a bounded Linear, and allowsValueDimension() respects the
// cap.
TEST(Space, MaximumValueDimensions)
{
    struct Fleet : Repeated<Vehicle, 4>
    {};

    using CappedWorld = Composite<Ownship, Fleet>;
    static_assert(
        World::MaxValueDimension == Eigen::Dynamic && Emrys::maxValueDimensionAtCompileTime<World> == Eigen::Dynamic
    );
    static_assert(
        CappedWorld::MaxValueDimension == 6 + 3 * 4 && Emrys::maxValueDimensionAtCompileTime<CappedWorld> == 18
    );
    static_assert(Emrys::maxValueDimensionAtCompileTime<State> == 5);
    static_assert(Linear<Eigen::Dynamic, 8>::MaxValueDimension == 8);
    EXPECT_TRUE(allowsValueDimension<CappedWorld>(18));
    EXPECT_FALSE(allowsValueDimension<CappedWorld>(21));
    EXPECT_TRUE((allowsValueDimension<Linear<Eigen::Dynamic, 8>>(8)));
    EXPECT_FALSE((allowsValueDimension<Linear<Eigen::Dynamic, 8>>(9)));
}

// ValueVector, DifferenceVector, and DifferenceMatrix take their rows and columns from the space:
// fixed for a fixed space, with the difference space's for differences, so a unit quaternion's
// value has 4 rows and its difference 3; dynamic with the space's bound for a capped Repeated, so
// it never allocates; a single row stored row-major; any scalar, such as automatic
// differentiation's; and Moments names the same types.
TEST(Space, SizedVectorsAndMatrices)
{
    struct Fleet : Repeated<Vehicle, 4>
    {};

    using CappedWorld = Composite<Ownship, Fleet>;
    using Rotation = Emrys::UnitQuaternion;
    using Jet = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    static_assert(std::is_same_v<Emrys::ValueVector<State>, Eigen::Matrix<double, 5, 1>>);
    static_assert(std::is_same_v<Emrys::ValueVector<Position, float>, Eigen::Vector3f>);
    static_assert(std::is_same_v<Emrys::ValueVector<Rotation>, Eigen::Vector4d>);
    static_assert(std::is_same_v<Emrys::DifferenceVector<Rotation>, Eigen::Vector3d>);
    static_assert(std::is_same_v<Emrys::DifferenceMatrix<Rotation>, Eigen::Matrix3d>);
    static_assert(std::is_same_v<Emrys::DifferenceMatrix<Range, Position>, Eigen::RowVector3d>);
    static_assert(std::is_same_v<Emrys::DifferenceMatrix<Position, Rotation>, Eigen::Matrix3d>);
    static_assert(
        std::is_same_v<Emrys::DifferenceMatrix<Composite<Position, Range>, Rotation>, Eigen::Matrix<double, 4, 3>>
    );
    static_assert(
        std::
            is_same_v<Emrys::ValueVector<CappedWorld>, Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 18, 1>>
    );
    static_assert(std::is_same_v<
                  Emrys::DifferenceMatrix<CappedWorld, Position>,
                  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor, 18, 3>>);
    static_assert(std::is_same_v<Emrys::ValueVector<World>, Eigen::VectorXd>);
    static_assert(std::is_same_v<Emrys::DifferenceVector<Rotation, Jet>, Eigen::Matrix<Jet, 3, 1>>);
    static_assert(std::is_same_v<Emrys::Moments<Rotation>::VectorType, Emrys::ValueVector<Rotation>>);
    static_assert(std::is_same_v<Emrys::Moments<Rotation>::DifferenceVectorType, Emrys::DifferenceVector<Rotation>>);
    static_assert(std::is_same_v<Emrys::Moments<CappedWorld>::MatrixType, Emrys::DifferenceMatrix<CappedWorld>>);
}

// A periodic space wraps values into [lower, upper) and differences into the centered period, for
// RadiansPi and an explicit [0, 256) period.
TEST(Space, PeriodicWrapsValuesAndDifferences)
{
    auto const angles = RadiansPi<3>{};
    auto values = Eigen::Vector3d{3.5, -3.5, 7.0};
    normalizeValues(angles, values);
    EXPECT_NEAR(values(0), 3.5 - 2 * pi, 1e-15);
    EXPECT_NEAR(values(1), -3.5 + 2 * pi, 1e-15);
    EXPECT_NEAR(values(2), 7.0 - 2 * pi, 1e-15);

    auto differences = Eigen::Vector3d{6.0, -6.0, 0.5};
    normalizeDifferences(angles, differences);
    EXPECT_NEAR(differences(0), 6.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(differences(1), -6.0 + 2 * pi, 1e-15);
    EXPECT_EQ(differences(2), 0.5);

    auto const bins = Periodic<{.lower = 0.0, .upper = 256.0}>{};
    auto bin = Eigen::Matrix<double, 1, 3>{-1.0, 256.0, 300.0};
    normalizeValues(bins, bin);
    EXPECT_EQ(bin, (Eigen::Matrix<double, 1, 3>{255.0, 0.0, 44.0}));
}

// Values many periods away wrap correctly, a value that rounds onto the upper bound stays inside
// the half-open interval, and NaN and infinity become NaN.
TEST(Space, PeriodicWrapsValuesFarOutside)
{
    auto const bins = Periodic<{.lower = 0.0, .upper = 256.0}>{};
    auto far = Eigen::Vector4d{1000.5, -1000.5, 512.0, -512.0};
    normalizeValues(bins, far);
    EXPECT_EQ(far, Eigen::Vector4d(232.5, 23.5, 0.0, 0.0));

    // -1e-20 + 360 rounds to 360, which the half-open interval excludes.
    auto edge = Eigen::Vector2d{-1e-20, 360.0 + 1e-13};
    normalizeValues(Degrees360<2>{}, edge);
    EXPECT_EQ(edge(0), 0.0);
    EXPECT_GE(edge(1), 0.0);
    EXPECT_LT(edge(1), 360.0);

    auto invalid = Eigen::Vector2d{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    normalizeValues(Degrees360<2>{}, invalid);
    EXPECT_TRUE(std::isnan(invalid(0)));
    EXPECT_TRUE(std::isnan(invalid(1)));
}

// Values within rounding of a seam far outside the interval wrap into it. Far out, the quotient in
// x - p floor((x - a) / p) can round up to the next integer, which put 9.5% of the doubles nearest
// the odd multiples of pi below -pi, by up to 1.1e-10 [coordinate-parameters]. These are the three doubles
// nearest each of the first 200,000 odd multiples of pi on each side.
TEST(Space, PeriodicWrapsValuesNearFarSeams)
{
    constexpr int count = 200'000;
    auto values = Eigen::VectorXd(6 * count);
    for (int k = 0; k < count; ++k)
    {
        double const seam = (2 * k + 1) * pi;
        for (int sign : {-1, 1})
        {
            Eigen::Index const i = 3 * (2 * k + (sign + 1) / 2);
            values(i) = std::nextafter(sign * seam, -std::numeric_limits<double>::infinity());
            values(i + 1) = sign * seam;
            values(i + 2) = std::nextafter(sign * seam, std::numeric_limits<double>::infinity());
        }
    }
    normalizeValues(RadiansPi<Eigen::Dynamic>{values.size()}, values);
    EXPECT_GE(values.minCoeff(), -pi);
    EXPECT_LT(values.maxCoeff(), pi);
}

// plus() and minus() on a periodic space take automatic differentiation scalars, as a model that
// methods::linearized() differentiates calls them, in each of the wrap's three cases: in range, one
// period out, and several out. A wrap subtracts whole periods, so every derivative is the
// identity's.
TEST(Space, PeriodicPlusAndMinusTakeAutoDiffScalars)
{
    using Jet = Eigen::AutoDiffScalar<Eigen::Vector2d>;
    auto const at = [](double value, Eigen::Vector2d const & derivatives) {
        return Eigen::Matrix<Jet, 1, 1>{Jet{value, derivatives}};
    };
    for (double const step : {0.5, 3.0, 20.0})
    {
        auto const sum = plus<RadiansPi<>>(at(3.0, {1.0, 0.0}), at(step, {0.0, 1.0}));
        EXPECT_NEAR(sum(0).value(), std::remainder(3.0 + step, 2 * pi), 1e-14);
        EXPECT_EQ(sum(0).derivatives(), Eigen::Vector2d(1.0, 1.0));

        auto const difference = minus<RadiansPi<>>(at(-3.0, {1.0, 0.0}), at(step, {0.0, 1.0}));
        EXPECT_NEAR(difference(0).value(), std::remainder(-3.0 - step, 2 * pi), 1e-14);
        EXPECT_EQ(difference(0).derivatives(), Eigen::Vector2d(1.0, -1.0));
    }
}

// The named angle spaces have the expected lower bounds and periods, Degrees360 is the Periodic of
// its interval, all are empty types, and each wraps as named.
TEST(Space, NamedAngles)
{
    static_assert(RadiansPi<>::lower() == -pi && RadiansPi<>::period() == 2 * pi);
    static_assert(Radians2Pi<>::lower() == 0.0 && Radians2Pi<>::period() == 2 * pi);
    static_assert(Degrees180<>::lower() == -180.0 && Degrees360<>::lower() == 0.0 && Degrees360<>::period() == 360.0);
    static_assert(std::is_same_v<Periodic<{.lower = 0.0, .upper = 360.0}>, Degrees360<>>);
    static_assert(std::is_empty_v<Degrees360<3>> && std::is_empty_v<RadiansPi<2>>);

    auto centered = Eigen::Vector4d{-1.0, 361.0, 180.0, -180.0};
    normalizeValues(Degrees180<4>{}, centered);
    EXPECT_EQ(centered, Eigen::Vector4d(-1.0, 1.0, -180.0, -180.0));

    auto compass = Eigen::Vector4d{-1.0, 361.0, 180.0, -180.0};
    normalizeValues(Degrees360<4>{}, compass);
    EXPECT_EQ(compass, Eigen::Vector4d(359.0, 1.0, 180.0, 180.0));

    auto radians = Eigen::Vector2d{-1.0, 7.0};
    normalizeValues(Radians2Pi<2>{}, radians);
    EXPECT_NEAR(radians(0), 2 * pi - 1.0, 1e-15);
    EXPECT_NEAR(radians(1), 7.0 - 2 * pi, 1e-15);
}

// normalizeValues() on a Composite applies each block's rule to its rows of every column of a
// sample matrix.
TEST(Space, CompositeNormalizesEachBlockAndEverySample)
{
    auto const state = State{};
    auto samples = Eigen::Matrix<double, 5, 2>{};
    samples.col(0) << 10.0, 20.0, 30.0, 4.0, -1.0;
    samples.col(1) << -10.0, 0.0, 0.0, -4.0, 2.0;
    normalizeValues(state, samples);
    EXPECT_EQ(samples.col(0).head<3>(), Eigen::Vector3d(10.0, 20.0, 30.0));
    EXPECT_NEAR(samples(3, 0), 4.0 - 2 * pi, 1e-15);
    EXPECT_EQ(samples(4, 0), -1.0);
    EXPECT_NEAR(samples(3, 1), -4.0 + 2 * pi, 1e-15);
    EXPECT_EQ(samples(4, 1), 2.0);
}

namespace
{

template <typename S>
concept MinusWithoutInstance = requires(Eigen::VectorXd const & x) { minus<S>(x, x); };

}  // namespace

// A fixed-size space, given as a template argument, does what its default instance does: plus(),
// minus(), and both normalizations, wrapping the heading. A resizable space holds its count, so it
// is not one and must be passed.
TEST(Space, AFixedSizeSpaceNeedsNoInstance)
{
    static_assert(FixedSizeSpace<State> && FixedSizeSpace<Sides> && FixedSizeSpace<Linear<4>>);
    static_assert(!FixedSizeSpace<World> && !FixedSizeSpace<Vehicles>);
    static_assert(MinusWithoutInstance<State> && !MinusWithoutInstance<World>);
    auto const x = Eigen::Vector<double, 5>{1.0, 2.0, 3.0, 3.0, -1.0};
    auto const d = Eigen::Vector<double, 5>{0.5, 0.0, 0.0, 1.0, 2.0};
    EXPECT_EQ(plus<State>(x, d), plus(State{}, x, d));
    EXPECT_EQ(minus<State>(d, x), minus(State{}, d, x));
    EXPECT_NEAR(plus<State>(x, d)(3), 4.0 - 2 * pi, 1e-15);

    Eigen::Vector<double, 5> values = Eigen::Vector<double, 5>::Constant(4.0);
    Eigen::Vector<double, 5> differences = values;
    normalizeValues<State>(values);
    normalizeDifferences<State>(differences);
    EXPECT_NEAR(values(3), 4.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(differences(3), 4.0 - 2 * pi, 1e-15);
    EXPECT_EQ(values(4), 4.0);
}

// A Repeated block wraps the periodic coordinate of every repetition and leaves the linear ones
// alone.
TEST(Space, RepeatedNormalizesEveryRepetition)
{
    auto world = World{};
    world.resize(12);  // two vehicles
    EXPECT_EQ(world.block<1>().count(), 2);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12);
    x(7) = 4.0;    // vehicle 0 longitude
    x(10) = -4.0;  // vehicle 1 longitude
    x(8) = 4.0;    // vehicle 0 altitude, linear
    normalizeValues(world, x);
    EXPECT_NEAR(x(7), 4.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(x(10), -4.0 + 2 * pi, 1e-15);
    EXPECT_EQ(x(8), 4.0);
}

// idx<Space, Tag>() is a constexpr Eigen::Index for a fixed space, and indexing a vector with a
// block's idx gives a fixed-size segment.
TEST(SpaceIndex, FixedTags)
{
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(idx<State, Heading>())>, Eigen::Index>);
    static_assert(idx<State, Heading>() == 3);  // usable in constant expressions
    static_assert(idx<State, Range>() == 4);

    Eigen::Matrix<double, 5, 1> const x = iota(5);
    auto const position = x(idx<State, Position>());
    static_assert(decltype(position)::RowsAtCompileTime == 3);
    EXPECT_EQ(Eigen::Vector3d{position}, Eigen::Vector3d(0.0, 1.0, 2.0));
}

// idx<S, Path>() through a Repeated block selects every repetition's coordinates, and
// idx<S, Path>(k) selects one repetition or one coordinate of it, as diffIdx<S, Path>(k) does of
// the differences, which are the values on these linear blocks.
TEST(SpaceIndex, RepeatedTail)
{
    Eigen::VectorXd const x = iota(12);
    EXPECT_EQ(Eigen::VectorXd{x(idx<World, Ownship>())}, iota(6));
    EXPECT_EQ(Eigen::VectorXd{x(idx<World, Vehicles>())}, (iota(6).array() + 6.0).matrix());
    EXPECT_EQ(Eigen::VectorXd{x(idx<World, TagPath<Vehicles, Altitude>>())}, Eigen::Vector2d(8.0, 11.0));
    EXPECT_EQ(Eigen::VectorXd{x(idx<World, TagPath<Vehicles, Longitude>>())}, Eigen::Vector2d(7.0, 10.0));

    EXPECT_EQ(Eigen::Vector3d{x(idx<World, Vehicles>(1))}, Eigen::Vector3d(9.0, 10.0, 11.0));
    EXPECT_EQ((idx<World, TagPath<Vehicles, Longitude>>(1)), 10);
    EXPECT_EQ((idx<World, TagPath<Vehicles, Altitude>>(0)), 8);
    EXPECT_EQ((Emrys::diffIdx<World, TagPath<Vehicles, Longitude>>(1)), 10);
}

// A tag that appears in two groups is addressed by a TagPath through its group; a unique tag needs
// no qualification.
TEST(SpaceIndex, QualifiedPaths)
{
    static_assert(idx<Sides, TagPath<Left, X>>() == 0);
    static_assert(idx<Sides, TagPath<Right, X>>() == 2);
    static_assert(idx<Sides, Y>() == 1);  // unique, so no qualification needed
}

// The internal complementSubspace() removes a block or a nested path and keeps the rest, replacing
// a composite that loses a block by what remains and keeping a Repeated block's count when one
// coordinate of every repetition goes.
TEST(SpaceProjection, ComplementKeepsTheRemainingBlocks)
{
    using Emrys::internal::complementSubspace;
    // Removing a top-level block leaves the others.
    static_assert(std::is_same_v<decltype(complementSubspace<Heading>(State{})), Composite<Position, Range>>);
    static_assert(std::is_same_v<decltype(complementSubspace<Range>(State{})), Composite<Position, Heading>>);

    // A composite that loses a block is replaced by what it has left.
    static_assert(std::is_same_v<decltype(complementSubspace<Sides, Left, X>(Sides{})), Composite<Y, Right>>);

    // Removing one coordinate of every repetition keeps a Repeated block of the rest, with its
    // count.
    auto world = World{};
    world.resize(12);
    auto const rest = complementSubspace<Vehicles, Longitude>(world);
    static_assert(std::is_same_v<decltype(rest), Composite<Ownship, Repeated<Composite<Latitude, Altitude>>> const>);
    EXPECT_EQ(rest.block<1>().count(), 2);
    EXPECT_EQ(valueDimension(rest), 10);
    auto const ownship = complementSubspace<Vehicles>(world);
    static_assert(std::is_same_v<decltype(ownship), Ownship const>);
}

namespace
{

// Spaces written from scratch, using none of the library's space types: degrees in [0, 360),
// whose differences are degrees in [-180, 180).
struct UserDegreeDifferences
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        writable(block) = block.unaryExpr([](auto d) {
            return d - 360.0 * Eigen::numext::floor(d / 360.0 + 0.5);
        });
    }

    // Differences of differences wrap the same way.
    [[nodiscard]] UserDegreeDifferences
    difference() const
    {
        return *this;
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(UserDegreeDifferences const &, UserDegreeDifferences const &) = default;
};

struct UserDegrees
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        writable(block) = block.unaryExpr([](auto x) {
            return x - 360.0 * Eigen::numext::floor(x / 360.0);
        });
    }

    [[nodiscard]] UserDegreeDifferences
    difference() const
    {
        return {};
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(UserDegrees const &, UserDegrees const &) = default;
};

struct Bearing : UserDegrees
{};

// Each of these breaks one requirement.
struct NormalizeWithoutDifference : UserDegrees
{
    void
    difference() const = delete;
};

struct DifferenceOfAnotherSize : UserDegrees
{
    [[nodiscard]] Linear<2>
    difference() const
    {
        return {};
    }
};

struct ResizableWithoutResize : UserDegrees
{
    static constexpr int FixedValueDimension = 0;
    static constexpr int ValueDimensionStep = 1;
    static constexpr int MaxValueDimension = Eigen::Dynamic;
};

struct MaxDimensionOfAnotherSpace : UserDegrees
{
    static constexpr int MaxValueDimension = 3;
};

struct NonConstantDimension : UserDegrees
{
    static inline int FixedValueDimension = 1;
};

// Operations written for one vector rather than for a block of columns, which the library would
// otherwise skip as missing.
using Vector1 = Eigen::Matrix<double, 1, 1>;

struct VectorNormalize : UserDegrees
{
    void
    normalize(Vector1 & x) const
    {
        x(0) -= 360.0 * Eigen::numext::floor(x(0) / 360.0);
    }
};

struct VectorPlus : UserDegrees
{
    void
    plus(Vector1 const & x, Vector1 const & d, Vector1 & out) const
    {
        out = x + d;
    }
};

struct VectorMinus : UserDegrees
{
    void
    minus(Vector1 const & a, Vector1 const & b, Vector1 & out) const
    {
        out = a - b;
    }
};

struct VectorNormalizedDifferences : UserDegreeDifferences
{
    void
    normalize(Vector1 & d) const
    {
        d(0) -= 360.0 * Eigen::numext::floor(d(0) / 360.0 + 0.5);
    }
};

struct DifferencesWithVectorNormalize : UserDegrees
{
    [[nodiscard]] VectorNormalizedDifferences
    difference() const
    {
        return {};
    }
};

// A linear space written from scratch: no normalize() and no difference().
struct UserMeters
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(UserMeters const &, UserMeters const &) = default;
};

// A tag on Linear that wraps like UserDegrees: its own operations make it non-linear.
struct WrappedOffset : Linear<1>
{
    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block) const
    {
        UserDegrees{}.normalize(block);
    }

    [[nodiscard]] UserDegreeDifferences
    difference() const
    {
        return {};
    }
};

// A heading kept unwrapped, so that it plots without jumps, whose residuals still take the short
// way round: plain values, wrapped differences.
struct UnwrappedHeading : Linear<1>
{
    [[nodiscard]] RadiansPi<>
    difference() const
    {
        return {};
    }
};

// Positive scales whose differences are log ratios: x ⊞ δ = x e^δ and a ⊟ b = log(a / b), which
// no normalize() of a sum or a difference can express.
struct Scale
{
    static constexpr int FixedValueDimension = 1;
    static constexpr int ValueDimensionStep = 0;
    static constexpr int MaxValueDimension = 1;

    [[nodiscard]] static constexpr Eigen::Index
    valueDimension()
    {
        return 1;
    }

    [[nodiscard]] Linear<1>
    difference() const
    {
        return {};
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        writable(out) = (x.array() * d.array().exp()).matrix();
    }

    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> const & out) const
    {
        writable(out) = (a.array() / b.array()).log().matrix();
    }

    [[maybe_unused]] friend bool  // for std::regular, which Space requires; no test compares these
    operator==(Scale const &, Scale const &) = default;
};

struct Gain : Scale
{};

struct Gains : Repeated<Gain>
{
    using Repeated::Repeated;
};

// Its own plus() without saying where its differences live.
struct PlusWithoutDifference : Linear<1>
{
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        Scale{}.plus(x, d, out);
    }
};

// Near misses of the forms: templates the library would have taken as missing, leaving the space
// a plain vector. A normalize() that is not const, a plus() whose output is a non-const reference,
// and a minus() of the same kind; each is otherwise a working space.
struct NonConstNormalize : UserDegrees
{
    template <typename Derived>
    void
    normalize(Eigen::MatrixBase<Derived> const & block)
    {
        UserDegrees{}.normalize(block);
    }
};

struct NonConstOutputPlus : Scale
{
    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> & out) const
    {
        Scale{}.plus(x, d, out);
    }
};

struct NonConstOutputMinus : Scale
{
    template <typename A, typename B, typename Out>
    void
    minus(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b, Eigen::MatrixBase<Out> & out) const
    {
        Scale{}.minus(a, b, out);
    }
};

// Its own plus() and a difference space, but residuals left to subtraction.
struct PlusWithoutMinus : Linear<1>
{
    [[nodiscard]] Linear<1>
    difference() const
    {
        return {};
    }

    template <typename X, typename D, typename Out>
    void
    plus(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out) const
    {
        writable(out) = (x.array() * d.array().exp()).matrix();
    }
};

template <typename S, typename T = double>
concept GaussianAccepts = requires { typename GaussianParams<S, T>; };

// isLinear<> and its parts take only a Space.
template <typename S>
concept HasLinearity =
    requires { isLinear<S>; } || requires { hasPlainValues<S>; } || requires { hasPlainDifferences<S>; };

}  // namespace

// Every library space type satisfies the Space concept.
TEST(SpaceConcept, LibrarySpacesSatisfyIt)
{
    static_assert(Space<Linear<3>> && Space<Linear<Eigen::Dynamic>> && Space<Linear<Eigen::Dynamic, 8>>);
    static_assert(Space<Degrees360<2>> && Space<RadiansPi<Eigen::Dynamic, 4>> && Space<Heading>);
    static_assert(Space<State> && Space<World> && Space<Vehicles> && Space<Sides>);
}

// Scalars, vectors, distributions, and spaces that break one requirement each (no difference space,
// a difference of another size, resizable without resize(), an inconsistent MaxValueDimension, a
// non-constant FixedValueDimension, a normalize(), plus(), or minus() written for one vector, on the space or on its
// difference space, a near miss of their forms, a plus() without a minus()) are rejected, isLinear<>
// does not compile for what is not a space, and Gaussian rejects a misplaced argument.
TEST(SpaceConcept, RejectsWhatIsNotASpace)
{
    static_assert(!Space<double> && !Space<Eigen::Vector3d> && !Space<GaussianParams<State>>);
    static_assert(!Space<NormalizeWithoutDifference>);
    static_assert(!Space<DifferenceOfAnotherSize>);
    static_assert(!Space<ResizableWithoutResize>);
    static_assert(!Space<MaxDimensionOfAnotherSpace>);
    static_assert(!Space<NonConstantDimension>);
    static_assert(!Space<VectorNormalize> && !Space<VectorPlus> && !Space<VectorMinus>);
    static_assert(!Space<DifferencesWithVectorNormalize>);
    static_assert(!Space<NonConstNormalize> && !Space<NonConstOutputPlus> && !Space<NonConstOutputMinus>);
    static_assert(!Space<PlusWithoutMinus>);
    static_assert(Space<Scale>, "the space the near misses of plus() and minus() start from is one");
    static_assert(HasLinearity<Linear<2>> && !HasLinearity<double> && !HasLinearity<Eigen::Vector3d>);
    static_assert(Space<UserDegrees>, "the space each of them breaks is one");

    // A distribution rejects a misplaced argument where it is written, including the old
    // scalar-first order.
    static_assert(GaussianAccepts<State> && GaussianAccepts<State, float>);
    static_assert(!GaussianAccepts<double> && !GaussianAccepts<double, State>);
}

// isLinear<> is true exactly when a space has no normalize() and no difference(), including for
// spaces written from scratch; a Linear tag with its own operations is non-linear and wraps; on a
// linear space the free normalizers do nothing.
TEST(SpaceConcept, LinearityComesFromTheNormalizeOperations)
{
    static_assert(isLinear<Linear<3>> && !isLinear<Degrees360<>> && isLinear<Range> && !isLinear<UserDegrees>);
    static_assert(Space<UserMeters> && isLinear<UserMeters>);
    using Mixed = Composite<Position, WrappedOffset>;
    static_assert(Space<WrappedOffset> && !isLinear<WrappedOffset> && !isLinear<Mixed>);

    auto x = Eigen::Vector4d{1.0, 2.0, 3.0, 370.0};
    normalizeValues(Mixed{}, x);
    EXPECT_EQ(x, Eigen::Vector4d(1.0, 2.0, 3.0, 10.0));

    // A linear space has nothing to call; the free function does nothing.
    auto y = Eigen::Vector3d{-7.0, 400.0, 1e9};
    normalizeValues(Position{}, y);
    normalizeDifferences(Position{}, y);
    EXPECT_EQ(y, Eigen::Vector3d(-7.0, 400.0, 1e9));
}

// The internal leafSupport() reports the whole line for a linear leaf, the interval and periodicity
// for a Periodic tag, and nothing for a non-linear space written from scratch.
TEST(SpaceConcept, LeavesReportWhereTheirValuesLie)
{
    using Emrys::internal::leafSupport;
    constexpr double inf = std::numeric_limits<double>::infinity();
    // A linear leaf, written from scratch or not, covers the whole line.
    static_assert(
        leafSupport<Position>()->lower == -inf && leafSupport<Position>()->upper == inf
        && !leafSupport<Position>()->periodic
    );
    static_assert(leafSupport<UserMeters>().has_value());
    // A tag on Periodic reports its interval.
    static_assert(
        leafSupport<Heading>()->lower == -pi && leafSupport<Heading>()->upper == pi && leafSupport<Heading>()->periodic
    );
    // A non-linear space written from scratch keeps its range to itself.
    static_assert(!leafSupport<UserDegrees>().has_value());
}

// hasPlainValues<> and hasPlainDifferences<> are independent: an unwrapped heading keeps its values
// plain and wraps its differences, and a Composite applies each block's rule to values and to
// differences separately.
TEST(SpaceConcept, ValuesAndDifferencesNormalizeSeparately)
{
    // A space can wrap its differences and keep its values plain.
    static_assert(hasPlainValues<UnwrappedHeading> && !hasPlainDifferences<UnwrappedHeading>);
    static_assert(Space<UnwrappedHeading> && !isLinear<UnwrappedHeading>);

    auto heading = Eigen::Vector2d{7.0, -7.0};
    normalizeValues(UnwrappedHeading{}, heading);
    EXPECT_EQ(heading, Eigen::Vector2d(7.0, -7.0));
    normalizeDifferences(UnwrappedHeading{}, heading);
    EXPECT_NEAR(heading(0), 7.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(heading(1), -7.0 + 2 * pi, 1e-15);

    // A composite normalizes its values and its differences block by block, each in its own way.
    using Track = Composite<Heading, UnwrappedHeading>;
    static_assert(!hasPlainValues<Track> && !hasPlainDifferences<Track>);
    auto track = Eigen::Vector2d{-4.0, 7.0};
    normalizeValues(Track{}, track);
    EXPECT_NEAR(track(0), -4.0 + 2 * pi, 1e-15);
    EXPECT_EQ(track(1), 7.0);
    auto residual = Eigen::Vector2d{-4.0, 7.0};
    normalizeDifferences(Track{}, residual);
    EXPECT_NEAR(residual(0), -4.0 + 2 * pi, 1e-15);
    EXPECT_NEAR(residual(1), 7.0 - 2 * pi, 1e-15);
}

// DifferenceSpace<> is the centered period for a periodic space, the untagged Linear for a
// linear one, a space's own difference() for a user space, and a tagless Composite of the blocks'
// difference spaces, linear or not; a resizable space's difference space carries its sizes, and
// normalizing differences is normalizing values in it.
TEST(SpaceConcept, DifferencesLiveInADifferenceSpace)
{
    static_assert(std::is_same_v<DifferenceSpace<Degrees360<2>>, Degrees180<2>>);
    static_assert(std::is_same_v<DifferenceSpace<Radians2Pi<>>, RadiansPi<>>);
    static_assert(std::is_same_v<DifferenceSpace<RadiansPi<>>, RadiansPi<>>);
    static_assert(std::is_same_v<DifferenceSpace<Linear<3>>, Linear<3>>);
    static_assert(std::is_same_v<DifferenceSpace<UserDegrees>, UserDegreeDifferences>);

    // No tag is kept, linear or not, so a tag declared in another's difference space is a new
    // quantity, not a value of that tag.
    static_assert(std::is_same_v<DifferenceSpace<Position>, Linear<3>>);
    static_assert(std::is_same_v<DifferenceSpace<Heading>, RadiansPi<>>);
    static_assert(std::is_same_v<DifferenceSpace<State>, Composite<Linear<3>, RadiansPi<>, Linear<1>>>);
    static_assert(std::is_same_v<DifferenceSpace<Composite<Position, Altitude>>, Composite<Linear<3>, Linear<1>>>);
    static_assert(std::is_same_v<DifferenceSpace<Vehicles>, Repeated<Composite<Linear<1>, RadiansPi<>, Linear<1>>>>);
    static_assert(isLinear<Position> && isLinear<Composite<Position, Altitude>>);
    static_assert(std::is_empty_v<DifferenceSpace<State>>);

    // A resizable space's difference space has its sizes.
    auto const bins = differenceSpace(Degrees360<Eigen::Dynamic>{4});
    static_assert(std::is_same_v<decltype(bins), Degrees180<Eigen::Dynamic> const>);
    EXPECT_EQ(valueDimension(bins), 4);
    auto world = World{};
    world.resize(12);
    auto const worldDifferences = differenceSpace(world);
    EXPECT_EQ(valueDimension(worldDifferences), 12);
    EXPECT_EQ(worldDifferences.block<1>().count(), 2);

    // Normalizing differences is normalizing values in the difference space.
    auto d = Eigen::Vector2d{190.0, -190.0};
    normalizeDifferences(Degrees360<2>{}, d);
    EXPECT_EQ(d, Eigen::Vector2d(-170.0, 170.0));
    auto v = Eigen::Vector2d{190.0, -190.0};
    normalizeValues(Degrees180<2>{}, v);
    EXPECT_EQ(v, d);
}

// RealScalar accepts the floating-point types and rejects int, complex, and a space, Gaussian
// rejects them as its scalar, and the scalar defaults to double.
TEST(SpaceConcept, ScalarsAreRealNumbers)
{
    static_assert(Emrys::RealScalar<double> && Emrys::RealScalar<float> && Emrys::RealScalar<long double>);
    static_assert(!Emrys::RealScalar<int> && !Emrys::RealScalar<std::complex<double>>);
    static_assert(!Emrys::RealScalar<State>);
    static_assert(!GaussianAccepts<State, int> && !GaussianAccepts<State, std::complex<double>>);

    // The scalar defaults to double.
    static_assert(std::is_same_v<GaussianParams<State>, GaussianParams<State, double>>);
}

// A Gaussian over a space written from scratch wraps its density, keeps the space through a tag
// marginal, and normalizes a conditional's mean with the user's normalize().
TEST(SpaceConcept, UserWrittenSpaceWorksInADistribution)
{
    static_assert(Space<UserDegrees> && Space<Bearing>);

    struct Plane : Linear<2>
    {};

    using Fix = Composite<Plane, Bearing>;
    auto g = GaussianParams<Fix>{
        Fix{},
        Eigen::Vector3d{0.0, 0.0, 359.0},
        Eigen::Matrix3d{Eigen::Vector3d{1.0, 1.0, 4.0}.asDiagonal()}
    };

    auto const density = g.toDensity();
    EXPECT_NEAR(
        density->logDensity(Eigen::Vector3d{0.0, 0.0, 1.0}),
        density->logDensity(Eigen::Vector3d{0.0, 0.0, 357.0}),
        1e-12
    );

    auto const bearing = g.marginal<Bearing>().copy();
    static_assert(std::is_same_v<decltype(bearing)::SpaceType, Bearing>);

    // A conditional that leaves only the user's space returns it, normalizing as it does.
    auto given = g.conditional<Plane>(Eigen::Vector2d{0.0, 0.0});
    static_assert(std::is_same_v<decltype(given)::SpaceType, Bearing>);
    given.setMean(Eigen::Matrix<double, 1, 1>{365.0});
    EXPECT_EQ(given.mean()(0), 5.0);
}

// The free plus() and minus() are the normalized sum and difference, broadcast a one-column operand
// across the other's columns, and are plain arithmetic with fixed sizes on a linear space.
TEST(SpacePlusMinus, DefaultsAreTheNormalizedSumAndDifference)
{
    auto const bearings = Degrees360<2>{};
    auto const x = Eigen::Vector2d{350.0, 10.0};
    EXPECT_EQ(plus(bearings, x, Eigen::Vector2d{20.0, -20.0}), Eigen::Vector2d(10.0, 350.0));
    EXPECT_EQ(minus(bearings, Eigen::Vector2d{10.0, 350.0}, x), Eigen::Vector2d(20.0, -20.0));

    // An operand with one column is used for every column of the other.
    auto const samples = Eigen::Matrix<double, 2, 3>{{10.0, 180.0, 355.0}, {0.0, 0.0, 0.0}};
    Eigen::Matrix<double, 2, 3> const deviations = minus(bearings, samples, x);
    EXPECT_EQ(deviations, (Eigen::Matrix<double, 2, 3>{{20.0, -170.0, 5.0}, {-10.0, -10.0, -10.0}}));
    EXPECT_EQ((Eigen::Matrix<double, 2, 3>{plus(bearings, x, deviations)}), samples);

    // In a linear space they are plain arithmetic, with the operands' fixed sizes.
    auto const sum = plus(Linear<2>{}, x, Eigen::Vector2d{20.0, -20.0});
    static_assert(std::is_same_v<decltype(sum), Eigen::Vector2d const>);
    EXPECT_EQ(sum, Eigen::Vector2d(370.0, -10.0));
}

// less() takes a difference away from a value: plus() of its negation, which wraps as plus() does,
// undoes plus(), broadcasts as plus() does, and is plain subtraction in a linear space.
TEST(SpacePlusMinus, LessTakesADifferenceAwayFromAValue)
{
    auto const bearings = Degrees360<2>{};
    auto const x = Eigen::Vector2d{10.0, 350.0};
    auto const d = Eigen::Vector2d{20.0, -20.0};
    EXPECT_EQ(Emrys::less(bearings, x, d), Eigen::Vector2d(350.0, 10.0));
    EXPECT_EQ(Emrys::less(bearings, plus(bearings, x, d), d), x);

    // An operand with one column is used for every column of the other.
    auto const samples = Eigen::Matrix<double, 2, 3>{{10.0, 180.0, 355.0}, {0.0, 0.0, 0.0}};
    Eigen::Matrix<double, 2, 3> const back = Emrys::less<Degrees360<2>>(samples, d);
    EXPECT_EQ(back, (Eigen::Matrix<double, 2, 3>{{350.0, 160.0, 335.0}, {20.0, 20.0, 20.0}}));

    // A reading less its bias, the bias declared in the reading's difference space.
    struct Gyro : Linear<1>
    {};

    struct Bias : DifferenceSpace<Gyro>
    {};

    auto const reading = Eigen::Matrix<double, 1, 1>{0.25};
    auto const bias = Emrys::ValueVector<Bias>{0.05};
    auto const corrected = Emrys::less<Gyro>(reading, bias);
    static_assert(std::is_same_v<decltype(corrected), Eigen::Matrix<double, 1, 1> const>);
    EXPECT_EQ(corrected(0), 0.25 - 0.05);
}

// A space with its own plus() and minus() (a log-ratio scale) satisfies Space only with a
// difference(), is non-linear, and a Composite or Repeated passes each block to its own operations.
TEST(SpacePlusMinus, ASpaceMayDefineItsOwn)
{
    static_assert(Space<Scale> && Space<Gain> && Space<Gains> && !Space<PlusWithoutDifference>);
    static_assert(!hasPlainValues<Scale> && !hasPlainDifferences<Scale> && !isLinear<Scale>);
    auto const two = Eigen::Matrix<double, 1, 1>{2.0};
    EXPECT_NEAR(plus(Gain{}, two, Eigen::Matrix<double, 1, 1>{std::log(3.0)})(0), 6.0, 1e-14);
    EXPECT_NEAR(minus(Gain{}, Eigen::Matrix<double, 1, 1>{6.0}, two)(0), std::log(3.0), 1e-15);

    // A composite passes each block to that block's own operations, and so does a repeated block.
    using Mixed = Composite<X, Gain, Heading>;
    static_assert(Space<Mixed> && !isLinear<Mixed>);
    auto const x = Eigen::Vector3d{1.0, 2.0, 3.0};
    Eigen::Vector3d const moved = plus(Mixed{}, x, Eigen::Vector3d{0.5, std::log(2.0), 0.5});
    EXPECT_NEAR(moved(0), 1.5, 1e-15);
    EXPECT_NEAR(moved(1), 4.0, 1e-14);
    EXPECT_NEAR(moved(2), 3.5 - 2 * pi, 1e-15);
    Eigen::Vector3d const back = minus(Mixed{}, moved, x);
    EXPECT_LT((back - Eigen::Vector3d{0.5, std::log(2.0), 0.5}).norm(), 1e-14);

    auto const gains = Gains{Gain{}, 3};
    Eigen::VectorXd const doubled =
        plus(gains, Eigen::Vector3d{1.0, 2.0, 4.0}, Eigen::Vector3d::Constant(std::log(2.0)));
    EXPECT_LT((doubled - Eigen::Vector3d{2.0, 4.0, 8.0}).norm(), 1e-14);
}

// A Gaussian on the scale space is lognormal: its residuals are log ratios, its samples are drawn
// in the difference space and moved by plus() so they stay positive, a conditional moves the mean
// by plus(), and the particle mean is the geometric mean.
TEST(SpacePlusMinus, DistributionsUseTheSpacesOwnOperations)
{
    // A Gaussian on a scale is lognormal: its residuals are log ratios.
    auto const g = GaussianParams<Gain>{Gain{}, Eigen::Matrix<double, 1, 1>{2.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    auto const density = g.toDensity();
    ASSERT_TRUE(density.has_value());
    double const atOneSigma = -0.5 * std::log(2 * pi) - 0.5;
    EXPECT_NEAR(density->logDensity(Eigen::Matrix<double, 1, 1>{2.0 * std::exp(1.0)}), atOneSigma, 1e-14);
    EXPECT_NEAR(density->logDensity(Eigen::Matrix<double, 1, 1>{2.0 / std::exp(1.0)}), atOneSigma, 1e-14);

    // Samples are drawn in the difference space and moved by plus(), so they stay positive.
    auto rng = std::mt19937_64{7};
    auto const samples = Emrys::drawSamples(*density, 200, rng);
    EXPECT_TRUE((samples.array() > 0.0).all());

    // A conditional moves the target's mean by plus(): 2 e^{K r} with K = 0.5 and r = 1.
    using Pair = Composite<X, Gain>;
    auto const joint = GaussianParams<Pair>{Pair{}, Eigen::Vector2d{0.0, 2.0}, Eigen::Matrix2d{{1.0, 0.5}, {0.5, 1.0}}};
    auto const given = joint.conditional<X>(Eigen::Matrix<double, 1, 1>{1.0});
    EXPECT_NEAR(given.mean()(0), 2.0 * std::exp(0.5), 1e-14);

    // Weighted moments are tangent-space moments: equal weights on 1, 4, and 16 give the geometric
    // mean 4, where the arithmetic mean would be 7.
    auto const particles = ParticleSet<Gain>{Gain{}, Eigen::Matrix<double, 1, 3>{{1.0, 4.0, 16.0}}};
    auto const moments = particles.moments();
    ASSERT_TRUE(moments.has_value());
    EXPECT_NEAR(moments->mean()(0), 4.0, 1e-12);
}

namespace AdlTrap
{

// A space whose namespace has free plus() and minus() that take anything: an unqualified call
// inside the library would find them by argument-dependent lookup and be ambiguous with
// Emrys::plus() and Emrys::minus(), or call them.
struct Meters : Linear<2>
{};

struct Turn : Emrys::RadiansPi<>
{};

using Track = Composite<Meters, Turn>;

template <typename... Args>
int
plus(Track const &, Args const &...)
{
    return 0;
}

template <typename... Args>
int
minus(Track const &, Args const &...)
{
    return 0;
}

}  // namespace AdlTrap

// The library calls plus() and minus() qualified, so a free plus() or minus() in a space's namespace
// is never found: sampling, a density, and the moments of samples in a space that has them compile
// and wrap the heading as Composite's own operations do. Customization is by members only.
TEST(SpaceConcept, FreeFunctionsInTheSpacesNamespaceAreNotFound)
{
    using AdlTrap::Track;
    auto const g = GaussianParams<Track>{Track{}, Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Matrix3d::Identity()};
    auto const density = g.toDensity().value();
    auto rng = std::mt19937_64{5};
    auto const samples = Emrys::drawSamples(density, 50, rng);
    EXPECT_LT(samples.row(2).maxCoeff(), pi);
    EXPECT_GE(samples.row(2).minCoeff(), -pi);
    EXPECT_TRUE(std::isfinite(density.logDensity(Eigen::Vector3d{1.0, 2.0, 3.0 - 2 * pi})));
    EXPECT_NEAR(density.logDensity(Eigen::Vector3d{1.0, 2.0, 3.0 - 2 * pi}), density.logDensity(g.mean()), 1e-12);
}

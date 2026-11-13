// EMRYS_FILTER_ASSERT defined as a statement (Core/Macros.h): the common do-while shape of an
// override compiles with every header, since the library uses the macro only as a statement of its
// own, and it fires where the default would. DiscreteProduct::value() (Spaces/Discrete.h) was the
// one use inside an expression; valueOf() asserts each index given by at(). The evidence for the assertion hook of
// [embedded-builds], and for [time-steps]: a negative time step asserts at its conversion (Core/Time.h). It is also the
// evidence that a release build, where the assertion does nothing, returns rather than hangs or
// reads an error as a value: chiSquaredQuantile() and resamplers::Residual; and that the Gaussian mixture and
// Rao-Blackwellized particle filters' update() return a WeightError for weights that cannot be normalized, with no
// assertion ([update-outcomes]).
// The test is its own executable because the override changes the body of every inline function
// that asserts, and every translation unit of a program must agree on those.

namespace AssertOverride
{
inline int failures = 0;
}

#define EMRYS_FILTER_ASSERT(x)          \
    do                                  \
    {                                   \
        if (!(x))                       \
        {                               \
            ++AssertOverride::failures; \
        }                               \
    }                                   \
    while (false)

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace reductions = Emrys::reductions;
namespace resamplers = Emrys::resamplers;

namespace
{

using Mode = Emrys::DiscreteProduct<Emrys::Finite<3>, Emrys::Finite<4>>;

// The override is the one the library sees, and a value in range fails nothing. The flat index
// is 2 * 4 + 1, with the last axis varying fastest.
TEST(AssertOverride, ValueInRangeFailsNothing)
{
    AssertOverride::failures = 0;
    auto const mode = Mode{};
    EXPECT_EQ(mode.value({2, 1}), 9);
    EXPECT_EQ(AssertOverride::failures, 0);
}

// Each coordinate out of range fails once. The override returns rather than halting, so value()
// goes on to compute an index, which the test ignores.
TEST(AssertOverride, EachCoordinateOutOfRangeFailsOnce)
{
    AssertOverride::failures = 0;
    auto const mode = Mode{};
    static_cast<void>(mode.value({3, 1}));
    EXPECT_EQ(AssertOverride::failures, 1);
    static_cast<void>(mode.value({-1, 4}));
    EXPECT_EQ(AssertOverride::failures, 3);
}

struct Row : Emrys::Finite<3>
{};

struct Column : Emrys::Finite<4>
{};

// An index out of range in at() fails once in valueOf(), and indices in range fail nothing: the
// flat index of (2, 1) is 2 * 4 + 1.
TEST(AssertOverride, IndexOutOfRangeFailsOnce)
{
    using Grid = Emrys::DiscreteProduct<Row, Column>;
    AssertOverride::failures = 0;
    EXPECT_EQ(Emrys::valueOf(Grid{}, Emrys::at<Row>(2), Emrys::at<Column>(1)), 9);
    EXPECT_EQ(AssertOverride::failures, 0);
    static_cast<void>(Emrys::valueOf(Grid{}, Emrys::at<Grid>(2, 4)));
    EXPECT_EQ(AssertOverride::failures, 1);
}

struct Position : Emrys::Linear<1>
{};

struct Velocity : Emrys::Linear<1>
{};

// A negative step fails where it enters, as a number or as a duration, and a step of zero fails
// nothing: internal::toScalarTime() asserts it for every entry point that takes a step
// [time-steps]. The override returns, so the prediction goes on with the negative step, and the step,
// already a scalar, passes through the conversion again inside the prediction and fails there
// too; the default assertion would have stopped at the first.
TEST(AssertOverride, NegativeTimeStepFails)
{
    using namespace std::chrono_literals;
    using State = Emrys::Composite<Position, Velocity>;
    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto filter = Emrys::KalmanFilter{
        model,
        Emrys::GaussianParams<State>{State{}, Eigen::Vector2d{0.0, 1.0}, Eigen::Matrix2d::Identity()}
    };
    AssertOverride::failures = 0;
    filter.predict(0.0);
    EXPECT_EQ(AssertOverride::failures, 0);
    filter.predict(-0.1);
    int const afterNumber = AssertOverride::failures;
    EXPECT_GT(afterNumber, 0);
    filter.predict(-100ms);
    EXPECT_GT(AssertOverride::failures, afterNumber);
}

// A probability outside (0, 1) fails, and where the assertion returns, the quantile returns the
// limit rather than looping: reductions::gate(95.0) reached the loop.
TEST(AssertOverride, ChiSquaredQuantileOutsideTheUnitIntervalReturns)
{
    double const infinity = std::numeric_limits<double>::infinity();
    AssertOverride::failures = 0;
    EXPECT_EQ(Emrys::chiSquaredQuantile<double>(1.0, 2.0), infinity);
    EXPECT_EQ(Emrys::chiSquaredQuantile<double>(95.0, 2.0), infinity);
    EXPECT_EQ(Emrys::chiSquaredQuantile<double>(0.0, 2.0), 0.0);
    EXPECT_TRUE(std::isnan(Emrys::chiSquaredQuantile<double>(0.5, -1.0)));
    EXPECT_EQ(AssertOverride::failures, 4);
}

// A negative weight fails before the residual resampler counts its copies, and the count it then
// takes is none, not a negative number converted to std::size_t.
TEST(AssertOverride, ResidualResamplerTakesNoCopiesOfANegativeWeight)
{
    auto const weights = std::vector<double>{-1.0, 2.0, 1.0};
    auto ancestors = std::vector<Eigen::Index>(4);
    auto scratch = std::vector<double>(weights.size() + ancestors.size());
    auto rng = std::mt19937_64{3};
    AssertOverride::failures = 0;
    Emrys::resamplers::residual()(
        std::span<double const>{weights},
        std::span<Eigen::Index>{ancestors},
        rng,
        std::span<double>{scratch}
    );
    EXPECT_GT(AssertOverride::failures, 0);
    for (Eigen::Index const ancestor : ancestors)
    {
        EXPECT_GE(ancestor, 0);
        EXPECT_LT(ancestor, 3);
    }
}

// A unit-variance fix of the position at 0.5.
auto
positionFix()
{
    auto const model = Emrys::measurementModel(
        Eigen::Matrix<double, 1, 1>{1.0},
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(
            Emrys::GaussianParams<Emrys::Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}}
        )
    );
    return Emrys::observe(Emrys::methods::linearized(), model, Eigen::Matrix<double, 1, 1>{0.5});
}

// A Gaussian mixture filter's weights that cannot be normalized are an error, not an assertion, whether it
// collapses each component's update or keeps every branch of noise with an outlier component: the update returns
// the WeightError and the components are kept ([update-outcomes]).
TEST(AssertOverride, MixtureFilterReturnsTheErrorOfWeightsThatCannotBeNormalized)
{
    double const zero = -std::numeric_limits<double>::infinity();
    auto mixture = Emrys::GaussianMixtureParams<Position>{Position{}};
    mixture.add(zero, Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0});
    mixture.add(zero, Eigen::Matrix<double, 1, 1>{1.0}, Eigen::Matrix<double, 1, 1>{1.0});
    auto filter = Emrys::GaussianMixtureFilter{Emrys::Constant<Position>{}, mixture};
    AssertOverride::failures = 0;
    auto const collapsed = filter.update(positionFix());
    EXPECT_EQ(AssertOverride::failures, 0);
    ASSERT_FALSE(collapsed);
    EXPECT_EQ(collapsed.error(), Emrys::WeightError::ZeroSum);
    AssertOverride::failures = 0;
    using Vector1 = Eigen::Matrix<double, 1, 1>;
    auto noise = Emrys::GaussianMixtureParams<Emrys::Linear<1>>{Emrys::Linear<1>{}};
    noise.add(std::log(0.9), Vector1{0.0}, Vector1{1.0});
    noise.add(std::log(0.1), Vector1{0.0}, Vector1{100.0});
    auto const robust = Emrys::measurementModel(
        Vector1{1.0},
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(std::move(noise))
    );
    auto const kept =
        filter.update(Emrys::observe(Emrys::methods::linearized(), robust, Vector1{0.5}), Emrys::reductions::keep());
    EXPECT_EQ(AssertOverride::failures, 0);
    ASSERT_FALSE(kept);
    EXPECT_EQ(kept.error(), Emrys::WeightError::ZeroSum);
    EXPECT_EQ(filter.state().size(), 2);
}

// The same for a Rao-Blackwellized particle filter: its particles' weights are left as they were.
TEST(AssertOverride, RbParticleFilterReturnsTheErrorOfWeightsThatCannotBeNormalized)
{
    double const zero = -std::numeric_limits<double>::infinity();
    using State = Emrys::Composite<Position, Velocity>;
    auto const joint = Emrys::GaussianParams<State>{State{}, Eigen::Vector2d{0.0, 1.0}, Eigen::Matrix2d::Identity()};
    auto rng = std::mt19937_64{1};
    auto const drawn = Emrys::drawParticleGaussian<Position>(joint, 2, rng);
    using RbState = std::remove_cvref_t<decltype(drawn)>;
    auto const dead = RbState{
        State{},
        typename RbState::ParticlesType{Position{}, Eigen::RowVector2d{0.0, 1.0}, Eigen::Array2d{zero, zero}},
        drawn.conditional(0)
    };
    auto filter = Emrys::RbParticleFilter{Emrys::ConstantVelocity<Position, Velocity>{0.1}, dead};
    AssertOverride::failures = 0;
    auto const result = filter.update(positionFix());
    EXPECT_EQ(AssertOverride::failures, 0);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), Emrys::WeightError::ZeroSum);
    EXPECT_EQ(filter.state().logWeights()(0), zero);
}

// Two factors over the same block fail the product's partition, though their dimensions add up to
// the space's.
TEST(AssertOverride, ProductOfOverlappingFactorsFails)
{
    using State = Emrys::Composite<Position, Velocity>;
    auto const factor =
        Emrys::GaussianParams<Position>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}};
    AssertOverride::failures = 0;
    std::ignore = Emrys::Product{State{}, Emrys::on<Position>(factor), Emrys::on<Position>(factor)};
    EXPECT_EQ(AssertOverride::failures, 1);
}

}  // namespace

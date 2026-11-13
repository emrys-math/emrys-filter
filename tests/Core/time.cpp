// Time steps (roadmap step 17f; [time-steps]): TimeTraits converts a floating-point number, as it is, and a
// std::chrono::duration, to seconds, to any scalar, an integer clock with its own period included;
// TimeStep admits those, the scalar itself (an autodiff scalar too), and a user's own type
// through a TimeTraits specialization, and rejects integers, Steps, and execution policies.
// Headers under test: Core/Time.h.

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Time.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ratio>

#include <unsupported/Eigen/AutoDiff>

using Emrys::Steps;
using Emrys::TimeStep;
using Emrys::internal::toScalarTime;

namespace
{

// A user's clock, counting milliseconds in a struct of its own.
struct Ticks
{
    std::int64_t value;
};

// The MPEG system clock: 90 kHz, a period no standard duration names.
using MpegTicks = std::chrono::duration<std::int64_t, std::ratio<1, 90000>>;

using AutoDiff = Eigen::AutoDiffScalar<Eigen::Vector2d>;

}  // namespace

template <>
struct Emrys::TimeTraits<Ticks>
{
    template <typename Scalar>
    [[nodiscard]] static Scalar
    toScalar(Ticks t)
    {
        return Scalar(Scalar(t.value) / Scalar(1000));
    }
};

// What is and isn't a time step, in double, float, and an autodiff scalar.
TEST(Time, TimeStepAdmitsFloatingPointDurationsAndTheScalar)
{
    static_assert(TimeStep<double, double> && TimeStep<float, double> && TimeStep<double, float>);
    static_assert(TimeStep<double const &, double>);
    static_assert(TimeStep<std::chrono::milliseconds, double> && TimeStep<std::chrono::milliseconds, float>);
    static_assert(TimeStep<std::chrono::duration<double>, float> && TimeStep<MpegTicks, double>);
    static_assert(TimeStep<Ticks, double> && TimeStep<Ticks, float>);
    static_assert(TimeStep<AutoDiff, AutoDiff> && TimeStep<double, AutoDiff>);
    static_assert(TimeStep<std::chrono::milliseconds, AutoDiff>);

    static_assert(!TimeStep<int, double> && !TimeStep<long, float> && !TimeStep<std::int64_t, double>);
    static_assert(!TimeStep<unsigned, double> && !TimeStep<bool, double>);
    static_assert(!TimeStep<Steps, double> && !TimeStep<Emrys::execution::Sequential, double>);
    static_assert(!TimeStep<Eigen::Vector2d, double>);
}

// What a fixed-step filter's predict(inputs...) rejects as a leading argument: a number of any type
// or a duration, which it would take as an input and ignore ([fixed-step-models]), but not a vector, a
// Steps, which goes to predict(Steps, ...), or nothing.
TEST(Time, FixedStepPredictionRejectsALeadingStep)
{
    using Emrys::internal::leadsWithTimeStep;
    static_assert(leadsWithTimeStep<double> && leadsWithTimeStep<float const &, Eigen::Vector2d>);
    static_assert(leadsWithTimeStep<int> && leadsWithTimeStep<std::chrono::milliseconds>);
    static_assert(!leadsWithTimeStep<> && !leadsWithTimeStep<Eigen::Matrix<double, 1, 1>, double>);
    static_assert(!leadsWithTimeStep<Steps>);
}

// A floating-point number converts by Scalar(dt), so float narrows a double as a functional cast does.
TEST(Time, SecondsPassThrough)
{
    EXPECT_EQ(toScalarTime<double>(0.1), 0.1);
    EXPECT_EQ(toScalarTime<double>(0.1f), double(0.1f));
    EXPECT_EQ(toScalarTime<float>(0.1), 0.1f);
    EXPECT_EQ(toScalarTime<float>(0.1f), 0.1f);
}

// A duration is its count divided by the period's denominator, so a whole number of milliseconds is
// the correctly rounded quotient, the value of the decimal literal, in double and in float.
TEST(Time, DurationsConvertToCorrectlyRoundedSeconds)
{
    using namespace std::chrono_literals;
    EXPECT_EQ(toScalarTime<double>(10ms), 0.01);
    EXPECT_EQ(toScalarTime<double>(100ms), 0.1);
    EXPECT_EQ(toScalarTime<double>(1500us), 0.0015);
    EXPECT_EQ(toScalarTime<double>(250ns), 2.5e-7);
    EXPECT_EQ(toScalarTime<float>(10ms), 0.01f);
    EXPECT_EQ(toScalarTime<float>(100ms), 0.1f);
    EXPECT_EQ(toScalarTime<double>(3s), 3.0);
    EXPECT_EQ(toScalarTime<double>(2min), 120.0);
    EXPECT_EQ(toScalarTime<float>(std::chrono::duration<double>{0.1}), 0.1f);
    EXPECT_EQ(toScalarTime<double>(std::chrono::duration<float, std::milli>{20.0f}), 20.0 / 1000.0);
}

// A clock with a period of its own: the 90 kHz MPEG clock, 9000 ticks of which are 0.1 s, and a
// period with both a numerator and a denominator, 3/2 s, which multiplies and then divides.
TEST(Time, IntegerClocksConvertThroughTheirPeriod)
{
    EXPECT_EQ(toScalarTime<double>(MpegTicks{9000}), 0.1);
    EXPECT_EQ(toScalarTime<float>(MpegTicks{9000}), 0.1f);
    EXPECT_EQ(toScalarTime<double>(MpegTicks{1}), 1.0 / 90000.0);
    using ThreeHalves = std::chrono::duration<std::int64_t, std::ratio<3, 2>>;
    EXPECT_EQ(toScalarTime<double>(ThreeHalves{4}), 6.0);
}

// A TimeTraits specialization makes a user's type a step, converted as it says.
TEST(Time, AUserTypeConvertsThroughItsTraits)
{
    EXPECT_EQ(toScalarTime<double>(Ticks{100}), 0.1);
    EXPECT_EQ(toScalarTime<float>(Ticks{100}), 0.1f);
}

// The scalar itself passes through unchanged, an autodiff scalar's derivatives included, and a
// duration converts into an autodiff scalar with no derivatives.
TEST(Time, AnAutodiffScalarPassesThrough)
{
    using namespace std::chrono_literals;
    auto const dt = AutoDiff{0.25, Eigen::Vector2d{1.0, 0.0}};
    AutoDiff const same = toScalarTime<AutoDiff>(dt);
    EXPECT_EQ(same.value(), 0.25);
    EXPECT_EQ(same.derivatives(), dt.derivatives());
    AutoDiff const fromDuration = toScalarTime<AutoDiff>(250ms);
    EXPECT_EQ(fromDuration.value(), 0.25);
}

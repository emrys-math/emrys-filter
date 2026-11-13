#pragma once

/** \file
 * Time steps (decisions.md [time-steps]): TimeTraits, which converts a time type to a scalar, the TimeStep
 * concept that a filter's predict() takes its step through, and Steps, a count of a fixed-step
 * model's steps. Read Weights.h next.
 */

#include <chrono>
#include <concepts>
#include <type_traits>

#include <Eigen/Core>

#include <EmrysFilter/Core/Macros.h>

namespace Emrys
{

/** \defgroup Time Time steps
 * \ingroup Core
 *
 * What a step of time may be (decisions.md [time-steps]). A filter's predict(dt), the free predict(), a
 * FixedStepModel, and a ContinuousTimeMarkovChain take their step as any TimeStep and convert it
 * once, at that boundary, to their own \c Scalar; everything below, a process model's blocks
 * included, takes a \c Scalar, so a \c float filter does its time arithmetic in \c float:
 *
 * \code
 * using namespace std::chrono_literals;
 * filter.predict(0.01);    // in the unit of the model's parameters
 * filter.predict(10ms);    // 0.01 seconds, bitwise: 10 / 1000 in the filter's scalar
 * \endcode
 *
 *  - A floating-point number is taken as it is, converted to the scalar with \c Scalar(dt). The
 *    scalar itself, such as an autodiff or multiprecision type, passes through unchanged.
 *  - A \c std::chrono::duration converts to seconds as its count times its period, computed in the
 *    scalar: \c Scalar(count) divided by the period's denominator, or times its numerator. An
 *    integer clock is a duration with its own period, such as
 *    <tt>std::chrono::duration<std::int64_t, std::ratio<1, 90000>></tt>, and converts the same way.
 *  - An integer is not a time step: it would read as a time where a count of steps was meant.
 *    A fixed-step model advances by a count as <tt>predict(Steps{n})</tt>.
 *  - Another type becomes one by specializing TimeTraits, in a unit it chooses.
 *
 * The library does no arithmetic that needs a unit of time: a block combines the step with its own
 * continuous-time parameters, such as a spectral density, a time constant, or a chain's rates, so
 * the step must be in the unit those are in, whatever it is. A duration is the one case where the
 * library converts, and it converts to seconds, the unit \c std::chrono states every period in: a
 * model stepped by durations has its parameters per second. Passing a duration to a model whose
 * parameters are per hour gives wrong results the library cannot see.
 */

/** \ingroup Time
 * \brief How a time type converts to a scalar, as a static member template
 * <tt>toScalar<Scalar>(t)</tt> returning \c Scalar, in the unit of the parameters of the models
 * it steps. Empty for a type that is not a time step; specialize it to make one.
 *
 * \code
 * template <>
 * struct Emrys::TimeTraits<Ticks> {  // milliseconds, as seconds
 *   template <typename Scalar>
 *   static Scalar toScalar(Ticks t) { return Scalar(t.value) / Scalar(1000); }
 * };
 * \endcode
 */
template <typename Time>
struct TimeTraits
{};

/** \ingroup Time
 * A floating-point number, taken as it is and converted to the scalar.
 */
template <std::floating_point Time>
struct TimeTraits<Time>
{
    /** \returns \a t as a \a Scalar */
    template <typename Scalar>
    [[nodiscard]] static Scalar
    toScalar(Time t)
    {
        return Scalar(t);
    }
};

/** \ingroup Time
 * A \c std::chrono::duration, converted to seconds in the scalar by its period.
 */
template <typename Rep, typename Period>
struct TimeTraits<std::chrono::duration<Rep, Period>>
{
    /** \returns \a t in seconds as a \a Scalar: its count, converted to \a Scalar, times the
     * period's numerator and divided by its denominator, skipping a factor of one, so that a whole
     * number of milliseconds is the correctly rounded quotient
     */
    template <typename Scalar>
    [[nodiscard]] static Scalar
    toScalar(std::chrono::duration<Rep, Period> const & t)
    {
        // Through the scalar's literal type, the plain number an autodiff scalar is built from, so
        // that the integer's conversion is explicit there too.
        using Real = typename Eigen::NumTraits<Scalar>::Literal;
        Scalar const count = Scalar(Real(t.count()));
        Scalar const numerator = Scalar(Real(Period::num));
        Scalar const denominator = Scalar(Real(Period::den));
        if constexpr (Period::num == 1 && Period::den == 1)
        {
            return count;
        }
        else if constexpr (Period::num == 1)
        {
            return Scalar(count / denominator);
        }
        else if constexpr (Period::den == 1)
        {
            return Scalar(count * numerator);
        }
        else
        {
            return Scalar(Scalar(count * numerator) / denominator);
        }
    }
};

/** \ingroup Time
 * A step of time that converts to \a Scalar: \a Scalar itself, or any type with a TimeTraits,
 * such as a floating-point number or a \c std::chrono::duration. Not an integer.
 */
template <typename Time, typename Scalar>
concept TimeStep = std::same_as<std::remove_cvref_t<Time>, Scalar> || requires(std::remove_cvref_t<Time> const & t) {
    { TimeTraits<std::remove_cvref_t<Time>>::template toScalar<Scalar>(t) } -> std::same_as<Scalar>;
};

/** \ingroup Time
 * A count of a fixed-step model's steps, for <tt>predict(Steps{n})</tt>.
 */
struct Steps
{
    Eigen::Index count;  ///< the number of steps, at least zero
};

namespace internal
{

/** \returns \a t as a \a Scalar: \a t itself if it is one, and otherwise through its TimeTraits.
 * A negative step is a precondition violation: a prediction runs forward, and a smoother is the
 * way back, so every entry point that takes a step asserts it here, once.
 */
template <typename Scalar, typename Time>
    requires TimeStep<Time, Scalar>
[[nodiscard]] Scalar
toScalarTime(Time const & t)
{
    Scalar const step = [&] {
        if constexpr (std::same_as<Time, Scalar>)
        {
            return t;
        }
        else
        {
            return TimeTraits<Time>::template toScalar<Scalar>(t);
        }
    }();
    EMRYS_FILTER_ASSERT(step >= Scalar(0) && "a time step must be nonnegative");
    return step;
}

/** Whether the first of \a Args is a Steps: false for none. A fixed-step predict(inputs...)
 * excludes it, so that a leading count goes to predict(Steps, inputs...).
 */
template <typename... Args>
inline constexpr bool leadsWithSteps = false;

/** Whether \a First, the first of several arguments, is a Steps. */
template <typename First, typename... Rest>
inline constexpr bool leadsWithSteps<First, Rest...> = std::same_as<std::remove_cvref_t<First>, Steps>;

/** Whether the first of \a Args looks like a time step: a number or a type with a TimeTraits, such
 * as a \c std::chrono::duration. A fixed-step filter's predict(inputs...) rejects it, since it
 * takes its own step and would pass the number to the model as an input (decisions.md
 * [fixed-step-models]); an input
 * is given by its tag, as sure<Tag>(value) or a distribution, never as a bare number.
 */
template <typename... Args>
inline constexpr bool leadsWithTimeStep = false;

/** Whether \a First, the first of several arguments, looks like a time step. */
template <typename First, typename... Rest>
inline constexpr bool leadsWithTimeStep<First, Rest...> =
    std::is_arithmetic_v<std::remove_cvref_t<First>> || TimeStep<First, double> || TimeStep<First, float>;

/** Rejects a fixed-step filter's prediction given a time step as its first input, with a message
 * that names the overloads to call instead.
 */
template <typename... Inputs>
consteval void
rejectTimeStep()
{
    static_assert(
        !leadsWithTimeStep<Inputs...>,
        "a fixed-step filter takes its own step: call predict() for one step or predict(Steps{n}) for n, "
        "and give an input by its tag, as sure<Tag>(value) or a distribution over the tag's space"
    );
}

}  // namespace internal

}  // namespace Emrys

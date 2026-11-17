#pragma once

/** \file
 * ByMode (decisions.md [by-mode]): a part of a model that depends on the value of one mode axis, written
 * <tt>byMode<Axis>(when<Value>(part), ...)</tt>, with each alternative labeled by the value it serves. A filter whose
 * mode space has the axis gives each mode the alternative its value names; additiveNoise() of alternatives is the
 * Markov outlier model (ModeNoise), and alternatives that are measurements are a measurement per value of the axis.
 * Read Policies.h next.
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Subspace.h>

namespace Emrys
{

namespace internal
{

/** The weight of an alternative given none. */
struct NoWeight
{};

/** The label of the alternative given to otherwise(): every value no other alternative names. */
struct OtherValues
{};

}  // namespace internal

/** \ingroup Models
 * \brief One alternative of a ByMode: a part of a model, the value of the mode axis it serves, and
 * optionally its weight [by-mode].
 *
 * <tt>when<Turn>(part)</tt> serves the value named \c Turn; <tt>when(part)</tt>, unlabeled, serves
 * the value its position names, on an axis whose values have no names, such as a Finite; and
 * <tt>otherwise(part)</tt> serves every value no other alternative names. <tt>.weight(w)</tt> gives
 * the alternative the probability \a w that a filter without the axis reads it with (see ByMode).
 *
 * \tparam Value   the tag of the value served, \c void for a positional alternative, or
 *                 internal::OtherValues for otherwise()
 * \tparam T       the part
 * \tparam Weight  the weight's scalar type, or internal::NoWeight
 */
template <typename Value, typename T, typename Weight = internal::NoWeight>
class When
{
public:

    using ValueType = Value;  ///< the tag of the value served
    using PartType = T;       ///< the part

    /// whether the alternative has a weight
    static constexpr bool Weighted = !std::is_same_v<Weight, internal::NoWeight>;

    /** \a part, with no weight. */
    explicit When(T part)
        requires(!Weighted)
        : m_part{std::move(part)}
    {}

    /** \a part, with the weight \a weight. */
    When(T part, Weight probability)
        requires Weighted
        : m_part{std::move(part)}
        , m_weight{probability}
    {}

    /** \returns the part */
    [[nodiscard]] T const &
    part() const
    {
        return m_part;
    }

    /** \returns the part, to change its parameters */
    [[nodiscard]] T &
    part()
    {
        return m_part;
    }

    /** \returns the weight */
    [[nodiscard]] Weight
    weight() const
        requires Weighted
    {
        return m_weight;
    }

    /** \returns this alternative with the weight \a weight: the probability of its value, in a filter
     * that does not track the axis (see ByMode)
     */
    template <typename W>
        requires(!Weighted && std::is_arithmetic_v<W>)
    [[nodiscard]] When<Value, T, W>
    weight(W probability) &&
    {
        EMRYS_FILTER_ASSERT(probability >= W(0) && "an alternative's weight is a probability, at least zero");
        return {std::move(m_part), probability};
    }

    /** weight(w) &&, copying the part of an alternative that is not a temporary. */
    template <typename W>
        requires(!Weighted && std::is_arithmetic_v<W>)
    [[nodiscard]] When<Value, T, W>
    weight(W probability) const &
    {
        return When{*this}.weight(probability);
    }

private:

    T m_part;                                        ///< the part
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Weight m_weight;  ///< the weight, if given
};

/** \ingroup Models
 * \returns \a part as the alternative for the value named \a Value of a ByMode's axis:
 * <tt>byMode<Maneuver>(when<Cruise>(cv(0.01)), when<Turn>(cv(4.0)))</tt> [by-mode]
 */
template <typename Value, typename T>
[[nodiscard]] When<Value, std::decay_t<T>>
when(T && part)
{
    return When<Value, std::decay_t<T>>{std::forward<T>(part)};
}

/** \ingroup Models
 * \returns \a part as an alternative served by its position, on an axis whose values have no names,
 * such as a Finite: <tt>byMode<Cell>(when(a).weight(0.5), when(b).weight(0.5))</tt> [by-mode]
 */
template <typename T>
[[nodiscard]] When<void, std::decay_t<T>>
when(T && part)
{
    return When<void, std::decay_t<T>>{std::forward<T>(part)};
}

/** \ingroup Models
 * \returns \a part as the alternative for every value of a ByMode's axis that no other alternative
 * names [by-mode]. It turns off the check that every value has an alternative, so it is never implied.
 */
template <typename T>
[[nodiscard]] When<internal::OtherValues, std::decay_t<T>>
otherwise(T && part)
{
    return When<internal::OtherValues, std::decay_t<T>>{std::forward<T>(part)};
}

namespace internal
{

/** Whether \a T is a When. */
template <typename T>
inline constexpr bool isWhen = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Value, typename T, typename Weight>
inline constexpr bool isWhen<When<Value, T, Weight>> = true;

/// \endcond

/** \a T as an alternative: itself if it is a When, and otherwise a positional When of it. */
template <typename T>
struct AsWhenOf
{
    using type = When<void, T>;  ///< a positional alternative
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Value, typename T, typename Weight>
struct AsWhenOf<When<Value, T, Weight>>
{
    using type = When<Value, T, Weight>;  ///< the alternative itself
};

/// \endcond

/** The alternative \a A with its part replaced by one of type \a U, its value and weight kept. */
template <typename A, typename U>
struct RemappedOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Value, typename T, typename Weight, typename U>
struct RemappedOf<When<Value, T, Weight>, U>
{
    using type = When<Value, U, Weight>;  ///< the alternative with the new part
};

/// \endcond

/** \returns \a alternative with its part replaced by \a part, its value and weight kept */
template <typename Value, typename T, typename Weight, typename U>
[[nodiscard]] When<Value, U, Weight>
remapped(When<Value, T, Weight> const & alternative, U part)
{
    if constexpr (When<Value, T, Weight>::Weighted)
    {
        return {std::move(part), alternative.weight()};
    }
    else
    {
        return When<Value, U, Weight>{std::move(part)};
    }
}

/** \returns \a part as an alternative: itself if it is a When, and otherwise a positional When of it */
template <typename T>
[[nodiscard]] typename AsWhenOf<std::decay_t<T>>::type
asWhen(T && part)
{
    if constexpr (isWhen<std::decay_t<T>>)
    {
        return std::forward<T>(part);
    }
    else
    {
        return When<void, std::decay_t<T>>{std::forward<T>(part)};
    }
}

}  // namespace internal

/** \ingroup Models
 * \brief A part of a model that depends on the value of one mode axis: one alternative per value,
 * each labeled by the value it serves (decisions.md [by-mode]).
 *
 * <tt>byMode<Axis>(when<Value>(part), ...)</tt> goes where the part goes. A filter whose mode space
 * has the axis gives each mode the alternative that its value in the axis names; the mode chain
 * gives the probabilities (the Markov reading). Two places take it:
 *  - additiveNoise() of %Gaussian alternatives is a measurement's noise by mode axis, the Markov
 *    outlier model (see ModeNoise):
 *    <tt>additiveNoise(byMode<Sensor>(when<Inlier>(inlier), when<Outlier>(outlier)))</tt>;
 *  - alternatives that are measurements are a measurement per value of the axis, which a filter
 *    over a Hybrid updates each mode by:
 *    <tt>filter.update(byMode<SensorKind>(when<Radar>(radarFix), when<Lidar>(lidarFix)))</tt>,
 *    or <tt>observe(byMode<SensorKind>(when<Radar>(radar), when<Lidar>(lidar)), z)</tt> of models.
 *
 * The alternatives are given in any order. On an axis whose values are named (see Named), each
 * names its value with <tt>when<Value>()</tt>, and these are compile-time errors: a value the axis
 * lacks, a value named twice, and a value left out, unless <tt>otherwise()</tt> gives the alternative
 * for every value not named. On an axis whose values are not named, such as a Finite, the
 * alternatives are positional, one per value in order, and their count is checked against the
 * axis's cardinality, at compile time where it is fixed.
 *
 * A filter that does not track the axis reads alternatives that carry weights,
 * <tt>when<Inlier>(inlier).weight(0.9)</tt>, as a mixture with those weights, independent from one
 * measurement to the next (the memoryless reading); without weights it refuses them at compile time.
 * For the two readings to describe one sensor, the weights should be the chain's stationary
 * probabilities on the axis. Either every alternative has a weight or none does.
 *
 * Alternatives nest, each reading the axis it depends on: a measurement per value of
 * <tt>SensorKind</tt> may have noise by <tt>Outlier</tt>. An axis chooses at most once along a
 * nesting, since an inner choice by the same axis would only ever take one alternative.
 *
 * \tparam Path_         the mode axis, as a tag or TagPath
 * \tparam Alternatives  the alternatives, each a When
 *
 * \sa byMode(), when(), otherwise(), ModeNoise
 */
template <typename Path_, typename... Alternatives>
class ByMode
{
    static_assert(sizeof...(Alternatives) > 0, "byMode() needs an alternative");
    static_assert((internal::isWhen<Alternatives> && ...));

    /// the tag of the mode axis
    using Axis = internal::PathTag<Path_>;

    /// whether the axis's values have names
    static constexpr bool NamedAxis = internal::IsNamed<Axis>;

    /// whether the alternative is the one given to otherwise()
    template <typename A>
    static constexpr bool isOther = std::is_same_v<typename A::ValueType, internal::OtherValues>;

    /// how many alternatives name \a Value
    template <typename Value>
    static constexpr std::size_t naming =
        (std::size_t{0} + ... + std::size_t{std::is_same_v<typename Alternatives::ValueType, Value>});

    /// how many alternatives were given to otherwise()
    static constexpr std::size_t OtherCount = (std::size_t{0} + ... + std::size_t{isOther<Alternatives>});

    /// fails to compile unless the alternatives give each value of the axis exactly one alternative
    static constexpr bool
    checkLabels()
    {
        if constexpr (NamedAxis)
        {
            static_assert(
                (!std::is_void_v<typename Alternatives::ValueType> && ...),
                "the axis's values have names: label each alternative with when<Value>(), in any order"
            );
            static_assert(
                ((isOther<Alternatives> || Axis::template contains<typename Alternatives::ValueType>) && ...),
                "an alternative names a value that the axis does not have"
            );
            static_assert(
                ((isOther<Alternatives> || naming<typename Alternatives::ValueType> == 1) && ...),
                "two alternatives name the same value of the axis"
            );
            static_assert(OtherCount <= 1, "give otherwise() once");
            static_assert(
                OtherCount == 1 || int(sizeof...(Alternatives)) == Axis::Cardinality,
                "an alternative is missing: name every value of the axis with when<Value>(), or give otherwise() "
                "for the rest"
            );
            static_assert(
                OtherCount == 0 || int(sizeof...(Alternatives)) <= Axis::Cardinality,
                "otherwise() serves no value: every value of the axis already has an alternative"
            );
        }
        else
        {
            static_assert(
                (std::is_void_v<typename Alternatives::ValueType> && ...),
                "the axis's values have no names, so the alternatives are positional, one per value in order, "
                "without when<Value>() or otherwise()"
            );
            static_assert(
                Axis::Cardinality == Eigen::Dynamic || Axis::Cardinality == int(sizeof...(Alternatives)),
                "give one alternative per value of the axis"
            );
        }
        static_assert(
            (Alternatives::Weighted && ...) || (!Alternatives::Weighted && ...),
            "give every alternative a weight, or none"
        );
        return true;
    }

    static_assert(checkLabels());

public:

    using PathType = Path_;  ///< the mode axis

    static constexpr std::size_t Count = sizeof...(Alternatives);  ///< the number of alternatives

    /// whether the alternatives have weights, which a filter without the axis reads them by
    static constexpr bool Weighted = (Alternatives::Weighted && ...);

    /** Alternative \a I, as given. */
    template <std::size_t I>
    using AlternativeType = std::tuple_element_t<I, std::tuple<Alternatives...>>;

    /** The part of alternative \a I. */
    template <std::size_t I>
    using PartType = typename AlternativeType<I>::PartType;

    /** \a alternatives, one per value of the axis \a Path_, in any order. */
    explicit ByMode(Alternatives... alternatives)
        : m_alternatives{std::move(alternatives)...}
    {}

    /** \returns alternative \a I, as given */
    template <std::size_t I>
    [[nodiscard]] AlternativeType<I> const &
    alternative() const
    {
        return std::get<I>(m_alternatives);
    }

    /** \returns alternative \a I, as given, to change its part's parameters */
    template <std::size_t I>
    [[nodiscard]] AlternativeType<I> &
    alternative()
    {
        return std::get<I>(m_alternatives);
    }

    /** \returns the alternatives, as given */
    [[nodiscard]] std::tuple<Alternatives...> const &
    alternatives() const
    {
        return m_alternatives;
    }

    /** \returns the index, among the alternatives as given, of the one that serves the value with
     * index \a value in the axis; for positional alternatives, \a value itself
     */
    [[nodiscard]] static constexpr std::size_t
    alternativeFor(Eigen::Index value)
    {
        if constexpr (NamedAxis)
        {
            EMRYS_FILTER_ASSERT(value >= 0 && value < Axis::Cardinality && "the value is out of the axis's range");
            return Table[std::size_t(value)];
        }
        else
        {
            EMRYS_FILTER_ASSERT(
                value >= 0 && value < Eigen::Index(Count) && "the axis has more values than the alternatives"
            );
            return std::size_t(value);
        }
    }

    /** \returns \a f applied to the part that serves the value with index \a value in the axis; \a f
     * returns one type for every alternative
     */
    template <typename F>
    decltype(auto)
    visit(Eigen::Index value, F && f) const
    {
        return visitAlternative(value, [&](auto const & alternative) -> decltype(auto) {
            return f(alternative.part());
        });
    }

    /** \returns \a f applied to the alternative, the When, that serves the value with index \a value */
    template <typename F>
    decltype(auto)
    visitAlternative(Eigen::Index value, F && f) const
    {
        return visitFrom<0>(alternativeFor(value), f);
    }

    /** \returns the weight of the alternative that serves the value with index \a value */
    [[nodiscard]] auto
    weightFor(Eigen::Index value) const
        requires Weighted
    {
        using W = std::common_type_t<decltype(std::declval<Alternatives const &>().weight())...>;
        return visitAlternative(value, [](auto const & alternative) {
            return W(alternative.weight());
        });
    }

    /** \returns a ByMode on the same axis whose alternatives' parts are \a f applied to these, each
     * serving the same values with the same weight: how observe() makes a measurement per value of
     * models per value
     */
    template <typename F>
    [[nodiscard]] auto
    map(F const & f) const
    {
        return std::apply(
            [&](auto const &... alternative) {
                return ByMode<
                    Path_,
                    typename internal::RemappedOf<
                        std::remove_cvref_t<decltype(alternative)>,
                        std::remove_cvref_t<decltype(f(alternative.part()))>>::type...>{
                    internal::remapped(alternative, f(alternative.part()))...
                };
            },
            m_alternatives
        );
    }

    /** \returns the number of values of the axis the alternatives serve: the axis's cardinality,
     * or for positional alternatives on an axis sized at run time, their count
     */
    [[nodiscard]] static constexpr Eigen::Index
    values()
    {
        if constexpr (NamedAxis || Axis::Cardinality != Eigen::Dynamic)
        {
            return Axis::Cardinality;
        }
        else
        {
            return Eigen::Index(Count);
        }
    }

private:

    /** For each value of a named axis, the index of the alternative that serves it. */
    static constexpr auto Table = [] {
        if constexpr (NamedAxis)
        {
            auto table = std::array<std::size_t, std::size_t(Axis::Cardinality)>{};
            std::size_t other = Count;
            std::size_t i = 0;
            (
                [&] {
                    if constexpr (isOther<Alternatives>)
                    {
                        other = i;
                    }
                    ++i;
                }(),
                ...
            );
            table.fill(other);
            i = 0;
            (
                [&] {
                    if constexpr (!isOther<Alternatives>)
                    {
                        table[std::size_t(Axis::template index<typename Alternatives::ValueType>())] = i;
                    }
                    ++i;
                }(),
                ...
            );
            return table;
        }
        else
        {
            return std::array<std::size_t, 0>{};
        }
    }();

    /** \returns \a f applied to alternative \a index, looked for from alternative \a I on */
    template <std::size_t I, typename F>
    decltype(auto)
    visitFrom(std::size_t index, F & f) const
    {
        if constexpr (I + 1 == Count)
        {
            return f(std::get<I>(m_alternatives));
        }
        else
        {
            if (index == I)
            {
                return f(std::get<I>(m_alternatives));
            }
            return visitFrom<I + 1>(index, f);
        }
    }

    std::tuple<Alternatives...> m_alternatives;  ///< the alternatives, as given
};

/** \ingroup Models
 * \returns the ByMode in which the value of the mode axis named by the path \a P (a tag or a
 * TagPath) selects among \a alternatives, each <tt>when<Value>(part)</tt> on an axis whose values
 * have names, or a part or <tt>when(part)</tt> per value in order on one whose values do not
 * [by-mode]:
 *
 * \code
 * auto const noise = additiveNoise(byMode<Sensor>(when<Inlier>(inlier).weight(0.9),
 * when<Outlier>(outlier).weight(0.1))); auto const fix = byMode<SensorKind>(when<Radar>(radarFix),
 * when<Lidar>(lidarFix));
 * \endcode
 */
template <typename P, typename... Alternatives>
[[nodiscard]] ByMode<P, typename internal::AsWhenOf<std::decay_t<Alternatives>>::type...>
byMode(Alternatives &&... alternatives)
{
    return ByMode<P, typename internal::AsWhenOf<std::decay_t<Alternatives>>::type...>{
        internal::asWhen(std::forward<Alternatives>(alternatives))...
    };
}

namespace internal
{

/** Whether \a T is a ByMode. */
template <typename T>
inline constexpr bool isByModeOf = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Path, typename... Alternatives>
inline constexpr bool isByModeOf<ByMode<Path, Alternatives...>> = true;

/// \endcond

/** Whether \a T, \c const and references aside, is a ByMode. */
template <typename T>
inline constexpr bool isByModePart = isByModeOf<std::remove_cvref_t<T>>;

}  // namespace internal

}  // namespace Emrys

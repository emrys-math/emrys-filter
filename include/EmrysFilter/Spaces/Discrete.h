#pragma once

/** \file
 * Discrete spaces (decisions.md [discrete-spaces]): the DiscreteSpace concept; Finite, Named, and
 * DiscreteProduct; and valueOf(), the flat index of a value named by tags. The internal
 * namespace locates axes and named values by tag path and builds the complement of an axis.
 * Categorical.h is the distribution over them. Read \ref Distributions next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \defgroup Discrete Discrete spaces
 * \ingroup Spaces
 *
 * A discrete space is a finite set of values, such as a maneuver regime, a sensor's
 * inlier/outlier state, or the cells of a grid. It is a family separate from \ref Spaces "Spaces": its
 * values have no differences or means, only probabilities, which stay in a real \c Scalar.
 *
 * The leaf types are Finite, a number of unnamed values, and Named, whose values are tag
 * types. DiscreteProduct combines axes, so a value of a product has one value in each, as a
 * Composite has one block of coordinates for each of its blocks. Axes are named by tag
 * types derived from a discrete space, as subspaces are:
 *
 * \code
 * struct Cruise {};
 * struct Turn {};
 * struct Maneuver : Named<Cruise, Turn> {};
 * struct Inlier {};
 * struct Outlier {};
 * struct Sensor : Named<Inlier, Outlier> {};
 * using Mode = DiscreteProduct<Maneuver, Sensor>;   // 4 values
 * \endcode
 *
 * A value is stored as its flat index in <tt>[0, cardinality())</tt>. In a DiscreteProduct, the last
 * axis varies fastest, as in a C array, so the value of \c Mode with \c Maneuver at \a m
 * and \c Sensor at \a s is <tt>2 * m + s</tt>. A nested DiscreteProduct flattens the same way, so
 * grouping axes doesn't change the flat index. valueOf() gives the flat index of named
 * values, and DiscreteProduct::value() and DiscreteProduct::coordinates() convert between a flat index and one
 * index per axis.
 *
 * An axis is addressed by tag path, as a subspace is (\c Maneuver, or
 * <tt>TagPath<Outer, Maneuver></tt> when it appears more than once); a named value by its tag,
 * or by a TagPath ending in it (<tt>TagPath<Sensor, Outlier></tt>) when several axes
 * share it.
 *
 * Categorical is the distribution over a discrete space, and MarkovChain moves one forward in
 * time.
 */

/** \ingroup Discrete
 * \brief A type usable as a discrete space: a finite set of values.
 *
 * Finite, Named, DiscreteProduct, and tags derived from them satisfy it. A discrete space provides
 * \c Cardinality, its number of values or \c Eigen::Dynamic if that is known only at run time,
 * \c MaxCardinality, a bound on it or \c Eigen::Dynamic, cardinality(), and value semantics
 * (\c std::regular).
 */
template <typename S>
concept DiscreteSpace =
    requires {
        typename std::integral_constant<int, S::Cardinality>;
        typename std::integral_constant<int, S::MaxCardinality>;
    } && (S::Cardinality == Eigen::Dynamic || (S::Cardinality >= 1 && S::MaxCardinality == S::Cardinality))
    && std::regular<S> && requires(S const & space) {
           { space.cardinality() } -> std::convertible_to<Eigen::Index>;
       };

namespace internal
{

/** The cardinality of a Finite space: a compile-time constant when fixed, which takes no storage. */
template <int Cardinality, int MaxCardinality>
class FiniteExtent
{
public:

    /** \returns the number of values, \a Cardinality */
    [[nodiscard]] static constexpr Eigen::Index
    cardinality()
    {
        return Cardinality;
    }

    /** Always equal: a fixed extent has no state. */
    friend bool
    operator==(FiniteExtent const &, FiniteExtent const &) = default;
};

/** The cardinality of a Finite space set at run time: stored, at most \a MaxCardinality, and set
 * only when the space is built [discrete-spaces]. */
template <int MaxCardinality>
class FiniteExtent<Eigen::Dynamic, MaxCardinality>
{
public:

    /** One value. */
    FiniteExtent() = default;

    /** \a n values. */
    constexpr explicit FiniteExtent(Eigen::Index n)
        : m_cardinality{n}
    {
        EMRYS_FILTER_ASSERT(
            n >= 1 && (MaxCardinality == Eigen::Dynamic || n <= MaxCardinality)
            && "the cardinality does not fit the space"
        );
    }

    /** \returns the number of values */
    [[nodiscard]] constexpr Eigen::Index
    cardinality() const
    {
        return m_cardinality;
    }

    /** Equal when the cardinalities are. */
    friend bool
    operator==(FiniteExtent const &, FiniteExtent const &) = default;

private:

    Eigen::Index m_cardinality = 1;  ///< the number of values
};

/** \returns the product of \a Cardinalities, or \c Eigen::Dynamic if any is. */
template <int... Cardinalities>
consteval int
productCardinality()
{
    int result = 1;
    ((result = result == Eigen::Dynamic || Cardinalities == Eigen::Dynamic ? Eigen::Dynamic : result * Cardinalities),
     ...);
    return result;
}

}  // namespace internal

/** \ingroup Discrete
 * \brief \a Cardinality_ unnamed values, \c 0 to <tt>Cardinality_ - 1</tt>, such as the cells of
 * a grid; or a number set at run time, at most \a MaxCardinality_, if \a Cardinality_ is
 * \c Eigen::Dynamic.
 *
 * A fixed Finite is an empty type. A run-time one stores its number of values, which is set when
 * it is built and does not change, since a distribution over it has no meaning to give the
 * values it would gain [discrete-spaces]; a default-constructed one has one value.
 */
template <int Cardinality_, int MaxCardinality_ = Cardinality_>
struct Finite : internal::FiniteExtent<Cardinality_, MaxCardinality_>
{
    static_assert(Cardinality_ == Eigen::Dynamic || Cardinality_ >= 1, "a discrete space has at least one value");

    static_assert(
        Cardinality_ == Eigen::Dynamic || MaxCardinality_ == Cardinality_,
        "a fixed cardinality is its own maximum"
    );

    static_assert(MaxCardinality_ == Eigen::Dynamic || MaxCardinality_ >= 1);
    static constexpr int Cardinality = Cardinality_;        ///< the number of values, or Eigen::Dynamic
    static constexpr int MaxCardinality = MaxCardinality_;  ///< a bound on it, or Eigen::Dynamic

    Finite() = default;

    /** \a n values of a run-time Finite. */
    constexpr explicit Finite(Eigen::Index n)
        requires(Cardinality_ == Eigen::Dynamic)
        : internal::FiniteExtent<Cardinality_, MaxCardinality_>(n)
    {}

    /** Equal when the cardinalities are; a fixed Finite is always equal. */
    friend bool
    operator==(Finite const &, Finite const &) = default;
};

/** \ingroup Discrete
 * \brief Values named by the tag types \a Values, in order: <tt>struct Maneuver : Named<Cruise,
 * Turn> {};</tt>. The value named \c Turn has the index <tt>index<Turn>()</tt>, here 1.
 *
 * An empty type. The tags must be distinct; any type can be one, and it needs no definition.
 */
template <typename... Values>
struct Named
{
    static_assert(sizeof...(Values) >= 1, "a discrete space has at least one value");
    static constexpr int Cardinality = int(sizeof...(Values));  ///< the number of names
    static constexpr int MaxCardinality = Cardinality;          ///< the same: a Named is fixed

    /** Whether \a Value names one of the values. */
    template <typename Value>
    static constexpr bool contains = (std::is_same_v<Value, Values> || ...);

private:

    /** How many times \a Value appears among the names; exactly once for each of them. */
    template <typename Value>
    static constexpr std::size_t occurrences = (std::size_t{0} + ... + std::size_t{std::is_same_v<Value, Values>});

    static_assert(((occurrences<Values> == 1) && ...), "the values of a Named space must be distinct");

public:

    /** \returns the number of values, \c Cardinality */
    [[nodiscard]] static constexpr Eigen::Index
    cardinality()
    {
        return Cardinality;
    }

    /** \returns the index of the value named \a Value */
    template <typename Value>
    [[nodiscard]] static constexpr Eigen::Index
    index()
    {
        static_assert(contains<Value>, "the space has no value with this name");
        Eigen::Index result = 0;
        bool found = false;
        ((found = found || std::is_same_v<Value, Values>, result += found ? 0 : 1), ...);
        return result;
    }

    /** Always equal: a Named has no state. */
    friend bool
    operator==(Named const &, Named const &) = default;
};

/** \ingroup Discrete
 * \brief The product of the discrete spaces \a Axes: a value has one value in each axis.
 *
 * Its cardinality is the product of theirs. Values are flattened with the last axis
 * varying fastest: axis \a I has the stride stride<I>(), the product of the cardinalities
 * after it, and a value's flat index is the sum of each axis's index times its stride.
 */
template <DiscreteSpace... Axes>
class DiscreteProduct : private internal::PartStorage<Axes...>
{
    static_assert(sizeof...(Axes) > 0, "a product needs at least one axis");

    /** The axes with state. */
    using Storage = internal::PartStorage<Axes...>;

public:

    static constexpr std::size_t AxisCount = sizeof...(Axes);  ///< the number of axes

    /// the product of the axes' cardinalities, or Eigen::Dynamic if any is
    static constexpr int Cardinality = internal::productCardinality<Axes::Cardinality...>();

    /// the product of the axes' bounds, or Eigen::Dynamic if any is unbounded
    static constexpr int MaxCardinality = internal::productCardinality<Axes::MaxCardinality...>();

    DiscreteProduct() = default;

    /** The product of \a axes, with their cardinalities. */
    constexpr explicit DiscreteProduct(Axes... axes)
        : Storage{std::move(axes)...}
    {}

    /** \returns axis \a I */
    template <std::size_t I>
    [[nodiscard]] constexpr auto const &
    axis() const
    {
        return Storage::template part<I>();
    }

    /** \returns the number of values: the product of the axes' cardinalities */
    [[nodiscard]] constexpr Eigen::Index
    cardinality() const
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return (Eigen::Index{1} * ... * axis<I>().cardinality());
        }(std::make_index_sequence<AxisCount>{});
    }

    /** \returns the stride of axis \a I: the product of the cardinalities of the axes after it */
    template <std::size_t I>
    [[nodiscard]] constexpr Eigen::Index
    stride() const
    {
        return [&]<std::size_t... J>(std::index_sequence<J...>) {
            return (Eigen::Index{1} * ... * axis<I + 1 + J>().cardinality());
        }(std::make_index_sequence<AxisCount - 1 - I>{});
    }

    /** \returns the flat index of the value with index \a coordinates[i] in axis \a i */
    [[nodiscard]] constexpr Eigen::Index
    value(std::array<Eigen::Index, AxisCount> const & coordinates) const
    {
        auto const cardinalities = [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<Eigen::Index, AxisCount>{axis<I>().cardinality()...};
        }(std::make_index_sequence<AxisCount>{});
        Eigen::Index result = 0;
        for (std::size_t i = 0; i < AxisCount; ++i)
        {
            EMRYS_FILTER_ASSERT(
                coordinates[i] >= 0 && coordinates[i] < cardinalities[i] && "a coordinate is out of range"
            );
            result = result * cardinalities[i] + coordinates[i];
        }
        return result;
    }

    /** \returns the index in each axis of the value with flat index \a value */
    [[nodiscard]] constexpr std::array<Eigen::Index, AxisCount>
    coordinates(Eigen::Index value) const
    {
        EMRYS_FILTER_ASSERT(value >= 0 && value < cardinality() && "the value is out of range");
        auto result = std::array<Eigen::Index, AxisCount>{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((result[I] = (value / stride<I>()) % axis<I>().cardinality()), ...);
        }(std::make_index_sequence<AxisCount>{});
        return result;
    }

    /** Equal when every axis is. */
    friend bool
    operator==(DiscreteProduct const &, DiscreteProduct const &) = default;
};

namespace internal
{

/** Declared, never defined: matched by a pointer to a DiscreteProduct or a tag derived from one. */
template <typename... Axes>
std::tuple<Axes...> *
discreteProductAxes(DiscreteProduct<Axes...> const *);

/** A DiscreteProduct, or a tag derived from one. */
template <typename T>
concept IsDiscreteProduct = requires(T const * t) { discreteProductAxes(t); };

/** The axes of the DiscreteProduct \a T, as a tuple type. */
template <typename T>
using DiscreteProductAxes = std::remove_pointer_t<decltype(discreteProductAxes(std::declval<T const *>()))>;

/** Declared, never defined: true for a pointer to a Named or a tag derived from one. */
template <typename... Values>
std::true_type
namedValues(Named<Values...> const *);

/** The fallback for anything else. */
std::false_type
namedValues(void const *);

/** A Named space, or a tag derived from one. */
template <typename T>
concept IsNamed = decltype(namedValues(std::declval<T const *>()))::value;

/** A library discrete space type itself rather than a tag derived from one, which a path can't name. */
template <typename T>
inline constexpr bool IsUntaggedAxis = false;
/// \cond INTERNAL_SPECIALIZATIONS
template <int Cardinality, int MaxCardinality>
inline constexpr bool IsUntaggedAxis<Finite<Cardinality, MaxCardinality>>{true};

template <typename... Values>
inline constexpr bool IsUntaggedAxis<Named<Values...>>{true};

template <typename... Axes>
inline constexpr bool IsUntaggedAxis<DiscreteProduct<Axes...>>{true};

/// \endcond

/** Number of times the axis \a Tag appears in \a S. */
template <typename Tag, typename S>
consteval int
countAxis()
{
    if constexpr (std::is_same_v<Tag, S>)
    {
        return 1;
    }
    else if constexpr (IsDiscreteProduct<S>)
    {
        return []<typename... Axes>(std::tuple<Axes...> *) {
            return (countAxis<Tag, Axes>() + ... + 0);
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        return 0;
    }
}

/** Number of axes of \a S with a value named \a Value. */
template <typename Value, typename S>
consteval int
countValue()
{
    if constexpr (IsNamed<S>)
    {
        return S::template contains<Value> ? 1 : 0;
    }
    else if constexpr (IsDiscreteProduct<S>)
    {
        return []<typename... Axes>(std::tuple<Axes...> *) {
            return (countValue<Value, Axes>() + ... + 0);
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        return 0;
    }
}

/** Calls \a f(axis, stride) for each axis of the product \a space, in order. */
template <typename S, typename F>
constexpr void
forEachAxis(S const & space, F && f)
{
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f(space.template axis<I>(), space.template stride<I>()), ...);
    }(std::make_index_sequence<std::tuple_size_v<DiscreteProductAxes<S>>>{});
}

/** \returns the stride of the unique axis \a Tag within \a space, and a pointer to it. */
template <typename Tag, typename S>
[[nodiscard]] constexpr std::pair<Eigen::Index, Tag const *>
locateAxis(S const & space)
{
    if constexpr (std::is_same_v<Tag, S>)
    {
        return {1, &space};
    }
    else
    {
        static_assert(
            !IsUntaggedAxis<Tag>,
            "a path names axes by tag; derive one, such as struct Maneuver : Named<Cruise, Turn> {}"
        );
        static_assert(countAxis<Tag, S>() > 0, "the space does not have this axis");
        static_assert(countAxis<Tag, S>() < 2, "the axis appears more than once; qualify it with an enclosing tag");
        auto found = std::pair<Eigen::Index, Tag const *>{0, nullptr};
        forEachAxis(space, [&]<typename Axis>(Axis const & axis, Eigen::Index stride) {
            if constexpr (countAxis<Tag, Axis>() > 0)
            {
                auto const inner = locateAxis<Tag>(axis);
                found = {inner.first * stride, inner.second};
            }
        });
        return found;
    }
}

/** locateAxis() along a path of tags. */
template <typename Tag, typename... Rest, typename S>
[[nodiscard]] constexpr auto
locateAxisPath(S const & space)
{
    auto const [stride, tag] = locateAxis<Tag>(space);
    if constexpr (sizeof...(Rest) == 0)
    {
        return std::pair{stride, tag};
    }
    else
    {
        auto const [innerStride, inner] = locateAxisPath<Rest...>(*tag);
        return std::pair{stride * innerStride, inner};
    }
}

/** Where a named value sits: the value with flat index \a v has it when
 * <tt>(v / stride) % cardinality == index</tt>.
 */
struct ValueLocation
{
    Eigen::Index stride;       ///< the axis's stride in the flat index
    Eigen::Index cardinality;  ///< the axis's number of values
    Eigen::Index index;        ///< the value's index within the axis

    /** \returns whether the value with flat index \a value has this named value */
    [[nodiscard]] constexpr bool
    matches(Eigen::Index value) const
    {
        return (value / stride) % cardinality == index;
    }
};

/** \returns where the value named \a Value sits in \a space; it must name a value of exactly one axis. */
template <typename Value, typename S>
[[nodiscard]] constexpr ValueLocation
locateValue(S const & space)
{
    static_assert(countValue<Value, S>() > 0, "the space has no value with this name");
    static_assert(
        countValue<Value, S>() < 2,
        "the value names a value of more than one axis; qualify it with a TagPath to its axis"
    );
    if constexpr (IsNamed<S>)
    {
        return {1, space.cardinality(), S::template index<Value>()};
    }
    else
    {
        auto found = ValueLocation{};
        forEachAxis(space, [&]<typename Axis>(Axis const & axis, Eigen::Index stride) {
            if constexpr (countValue<Value, Axis>() > 0)
            {
                ValueLocation const inner = locateValue<Value>(axis);
                found = {inner.stride * stride, inner.cardinality, inner.index};
            }
        });
        return found;
    }
}

/** locateValue() for a value path: a value's tag, or a TagPath of axes ending in one. */
template <typename P, typename S>
[[nodiscard]] constexpr ValueLocation
locateValuePath(S const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        constexpr std::size_t count = sizeof...(Tags);
        using Value = std::tuple_element_t<count - 1, std::tuple<Tags...>>;
        if constexpr (count == 1)
        {
            return locateValue<Value>(space);
        }
        else
        {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                auto const [stride, axis] = locateAxisPath<std::tuple_element_t<I, std::tuple<Tags...>>...>(space);
                ValueLocation inner = locateValue<Value>(*axis);
                inner.stride *= stride;
                return inner;
            }(std::make_index_sequence<count - 1>{});
        }
    });
}

/** \returns the axis named by the path \a P (a tag or a TagPath) in \a space and its stride. */
template <typename P, typename S>
[[nodiscard]] constexpr auto
locateAxisByPath(S const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return locateAxisPath<Tags...>(space);
    });
}

/** \returns \a axes as one discrete space: the axis itself if there is one, otherwise their DiscreteProduct. */
template <typename... Axes>
[[nodiscard]] auto
joinAxes(std::tuple<Axes...> axes)
{
    static_assert(sizeof...(Axes) > 0, "nothing is left of the space");
    if constexpr (sizeof...(Axes) == 1)
    {
        return std::get<0>(std::move(axes));
    }
    else
    {
        return std::apply(
            [](Axes &... d) {
                return DiscreteProduct<Axes...>{std::move(d)...};
            },
            axes
        );
    }
}

template <typename Tag, typename... Rest, typename S>
[[nodiscard]] auto
axesWithout(S const & space);  // defined below; axisWithout() recurses into it

/** axesWithout() for one axis of a product: the axis itself if the path's first tag is not in it. */
template <typename Tag, typename... Rest, typename Axis>
[[nodiscard]] auto
axisWithout(Axis const & axis)
{
    if constexpr (countAxis<Tag, Axis>() > 0)
    {
        return axesWithout<Tag, Rest...>(axis);
    }
    else
    {
        return std::tuple<Axis>{axis};
    }
}

/** \returns the axes of \a space left when the one named by the path \a Tag, \a Rest is
 * removed, in order, as a tuple. A DiscreteProduct that loses an axis is replaced by its remaining
 * axes, as complementSubspace() does for a Composite.
 */
template <typename Tag, typename... Rest, typename S>
[[nodiscard]] auto
axesWithout(S const & space)
{
    if constexpr (std::is_same_v<Tag, S>)
    {
        if constexpr (sizeof...(Rest) == 0)
        {
            return std::tuple<>{};
        }
        else
        {
            return axesWithout<Rest...>(space);
        }
    }
    else
    {
        static_assert(
            !IsUntaggedAxis<Tag>,
            "a path names axes by tag; derive one, such as struct Maneuver : Named<Cruise, Turn> {}"
        );
        static_assert(IsDiscreteProduct<S>, "the space does not have this axis");
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::tuple_cat(axisWithout<Tag, Rest...>(space.template axis<I>())...);
        }(std::make_index_sequence<std::tuple_size_v<DiscreteProductAxes<S>>>{});
    }
}

/** \returns the discrete space of the axes of \a space outside the one named by the path \a P. */
template <typename P, typename S>
[[nodiscard]] auto
complementAxes(S const & space)
{
    return withPathTags<P>([&]<typename... Tags>() {
        return joinAxes(axesWithout<Tags...>(space));
    });
}

template <typename Value, typename S>
struct ValueAxisOf;

/** The axis among \a Axes (a tuple of axes) with a value named \a Value: \c void if none. */
template <typename Value, typename Axes>
struct ValueAxisIn
{
    /** No axis has it. */
    using type = void;
};

/** Looks for \a Value in \a Axis, then in \a Rest. */
template <typename Value, typename Axis, typename... Rest>
struct ValueAxisIn<Value, std::tuple<Axis, Rest...>>
{
    /// the owning axis
    using type = typename std::conditional_t<
        (countValue<Value, Axis>() > 0),
        ValueAxisOf<Value, Axis>,
        ValueAxisIn<Value, std::tuple<Rest...>>>::type;
};

/** The Named axis of \a S, or \a S itself, with a value named \a Value. */
template <typename Value, typename S>
struct ValueAxisOf
{
    /** The owning axis. */
    using type = typename ValueAxisIn<Value, DiscreteProductAxes<S>>::type;
};

/** A Named space owns its own values. */
template <typename Value, typename S>
    requires IsNamed<S>
struct ValueAxisOf<Value, S>
{
    /** The space itself. */
    using type = S;
};

/** The Named axis of \a S, or \a S itself, with a value named \a Value. */
template <typename Value, typename S>
using ValueAxis = typename ValueAxisOf<Value, S>::type;

/** The tag path of the axis owning the value named by the bare tag \a P in \a S. */
template <typename P, typename S>
struct ValueAxisPathOf
{
    /** The axis's path. */
    using type = TagPath<ValueAxis<P, S>>;
};

/** The tag path of the axis owning the value named by a TagPath of axes ending in a value: the
 * axes, then the axis that holds the value within the last axis.
 */
template <typename... Tags, typename S>
struct ValueAxisPathOf<TagPath<Tags...>, S>
{
    static constexpr std::size_t Count = sizeof...(Tags);  ///< the path's length

    /** The value's tag, last. */
    using Value = std::tuple_element_t<Count - 1, std::tuple<Tags...>>;

    /** \returns a TagPath of the axes \a I, then the axis that holds the value within the last of them */
    template <std::size_t... I>
    static consteval auto
    make(std::index_sequence<I...>)
    {
        if constexpr (Count == 1)
        {
            return TagPath<ValueAxis<Value, S>>{};
        }
        else
        {
            using Last = std::tuple_element_t<Count - 2, std::tuple<Tags...>>;
            return TagPath<std::tuple_element_t<I, std::tuple<Tags...>>..., ValueAxis<Value, Last>>{};
        }
    }

    /** The axis's path. */
    using type = decltype(make(std::make_index_sequence<Count - 1>{}));
};

/** The tag path, within \a S, of the axis whose values include the one named by the value
 * path \a P (a value's tag, or a TagPath of axes ending in one).
 */
template <typename P, typename S>
using ValueAxisPath = typename ValueAxisPathOf<P, S>::type;

/** The number of leaf axes of the discrete space \a S: one unless it is a product. */
template <typename S>
consteval int
leafAxisCount()
{
    if constexpr (IsDiscreteProduct<S>)
    {
        return []<typename... Axes>(std::tuple<Axes...> *) {
            return (leafAxisCount<Axes>() + ... + 0);
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        return 1;
    }
}

/** The leaf axes \a first to <tt>first + count - 1</tt> of a space, in flat-index order. */
struct LeafRange
{
    int first = 0;  ///< the first leaf axis
    int count = 0;  ///< the number of leaf axes
};

/** \returns the leaf axes of the unique axis \a Tag of \a S, all of them if \a S is \a Tag */
template <typename Tag, typename S>
consteval LeafRange
leafRangeOfAxis()
{
    if constexpr (std::is_same_v<Tag, S>)
    {
        return {0, leafAxisCount<S>()};
    }
    else if constexpr (IsDiscreteProduct<S>)
    {
        return []<typename... Axes>(std::tuple<Axes...> *) {
            LeafRange result{};
            int offset = 0;
            auto const visit = [&]<typename Axis>() {
                if constexpr (countAxis<Tag, Axis>() > 0)
                {
                    LeafRange const inner = leafRangeOfAxis<Tag, Axis>();
                    result = {offset + inner.first, inner.count};
                }
                offset += leafAxisCount<Axis>();
            };
            (visit.template operator()<Axes>(), ...);
            return result;
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        return {};
    }
}

/** \returns the leaf axes of the axis named by the tags \a Tag, \a Rest within \a S */
template <typename S, typename Tag, typename... Rest>
consteval LeafRange
leafRangeOfAxisPath()
{
    LeafRange const outer = leafRangeOfAxis<Tag, S>();
    if constexpr (sizeof...(Rest) == 0)
    {
        return outer;
    }
    else
    {
        LeafRange const inner = leafRangeOfAxisPath<Tag, Rest...>();
        return {outer.first + inner.first, inner.count};
    }
}

/** \returns the leaf axes of the axis named by the path \a P (a tag or a TagPath) within \a S */
template <typename P, typename S>
consteval LeafRange
leafRangeOfPath()
{
    return []<typename... Tags>(TagPath<Tags...> *) {
        return leafRangeOfAxisPath<S, Tags...>();
    }(static_cast<AsTagPath<P> *>(nullptr));
}

/** \returns the leaf axis of \a S with a value named \a Value */
template <typename Value, typename S>
consteval int
leafOfValue()
{
    if constexpr (IsDiscreteProduct<S>)
    {
        return []<typename... Axes>(std::tuple<Axes...> *) {
            int result = 0;
            int offset = 0;
            auto const visit = [&]<typename Axis>() {
                if constexpr (countValue<Value, Axis>() > 0)
                {
                    result = offset + leafOfValue<Value, Axis>();
                }
                offset += leafAxisCount<Axis>();
            };
            (visit.template operator()<Axes>(), ...);
            return result;
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        return 0;
    }
}

/** \returns the leaf axis of \a S holding the value named by the value path \a P */
template <typename P, typename S>
consteval int
leafOfValuePath()
{
    return []<typename... Tags>(TagPath<Tags...> *) {
        constexpr std::size_t count = sizeof...(Tags);
        using Value = std::tuple_element_t<count - 1, std::tuple<Tags...>>;
        if constexpr (count == 1)
        {
            return leafOfValue<Value, S>();
        }
        else
        {
            return []<std::size_t... I>(std::index_sequence<I...>) {
                using Axis = std::tuple_element_t<count - 2, std::tuple<Tags...>>;
                return leafRangeOfAxisPath<S, std::tuple_element_t<I, std::tuple<Tags...>>...>().first
                     + leafOfValue<Value, Axis>();
            }(std::make_index_sequence<count - 1>{});
        }
    }(static_cast<AsTagPath<P> *>(nullptr));
}

/** \returns whether the tags \a Tag, \a Rest name one axis of \a S, as locateAxisPath() requires */
template <typename S, typename Tag, typename... Rest>
consteval bool
axisPathResolves()
{
    if constexpr (!std::is_same_v<Tag, S> && (IsUntaggedAxis<Tag> || countAxis<Tag, S>() != 1))
    {
        return false;
    }
    else if constexpr (sizeof...(Rest) == 0)
    {
        return true;
    }
    else
    {
        return axisPathResolves<Tag, Rest...>();
    }
}

/** \returns whether the path \a P (a tag or a TagPath) names one axis of \a S */
template <typename P, typename S>
consteval bool
pathResolves()
{
    return []<typename... Tags>(TagPath<Tags...> *) {
        return axisPathResolves<S, Tags...>();
    }(static_cast<AsTagPath<P> *>(nullptr));
}

/** \returns whether the value path \a P names a value of one axis of \a S, as locateValuePath() requires */
template <typename P, typename S>
consteval bool
valuePathResolves()
{
    return []<typename... Tags>(TagPath<Tags...> *) {
        constexpr std::size_t count = sizeof...(Tags);
        using Value = std::tuple_element_t<count - 1, std::tuple<Tags...>>;
        if constexpr (count == 1)
        {
            return countValue<Value, S>() == 1;
        }
        else
        {
            return []<std::size_t... I>(std::index_sequence<I...>) {
                using Axis = std::tuple_element_t<count - 2, std::tuple<Tags...>>;
                if constexpr (!axisPathResolves<S, std::tuple_element_t<I, std::tuple<Tags...>>...>())
                {
                    return false;
                }
                else
                {
                    return countValue<Value, Axis>() == 1;
                }
            }(std::make_index_sequence<count - 1>{});
        }
    }(static_cast<AsTagPath<P> *>(nullptr));
}

/** \returns whether each leaf axis of \a S is Named, in flat-index order */
template <typename S>
consteval std::array<bool, std::size_t(leafAxisCount<S>())>
namedLeaves()
{
    auto result = std::array<bool, std::size_t(leafAxisCount<S>())>{};
    if constexpr (IsDiscreteProduct<S>)
    {
        [&]<typename... Axes>(std::tuple<Axes...> *) {
            std::size_t offset = 0;
            auto const append = [&]<typename Axis>() {
                for (bool const named : namedLeaves<Axis>())
                {
                    result[offset++] = named;
                }
            };
            (append.template operator()<Axes>(), ...);
        }(static_cast<DiscreteProductAxes<S> *>(nullptr));
    }
    else
    {
        result[0] = IsNamed<S>;
    }
    return result;
}

/** What valueOf() makes of a set of names and indices. */
enum class Naming
{
    complete,        ///< each leaf axis once
    unresolved,      ///< a name or a path doesn't resolve; locating it reports why
    repeated,        ///< a leaf axis more than once
    missingNamed,    ///< a Named leaf axis not at all
    missingUnnamed,  ///< a leaf axis without names not at all
};

/** Counts each of the leaf axes \a range once more in \a times. */
template <std::size_t N>
consteval void
countLeaves(std::array<int, N> & times, LeafRange range)
{
    for (int k = range.first; k < range.first + range.count; ++k)
    {
        ++times[std::size_t(k)];
    }
}

/** \returns how the value paths \a Values and the axis paths \a Paths given indices by at() name a
 * value of \a S
 */
template <typename S, typename... Values, typename... Paths>
consteval Naming
namingOf(std::tuple<Values...> *, std::tuple<Paths...> *)
{
    if constexpr (!((valuePathResolves<Values, S>() && ...) && (pathResolves<Paths, S>() && ...)))
    {
        return Naming::unresolved;
    }
    else
    {
        auto times = std::array<int, std::size_t(leafAxisCount<S>())>{};
        (countLeaves(times, LeafRange{leafOfValuePath<Values, S>(), 1}), ...);
        (countLeaves(times, leafRangeOfPath<Paths, S>()), ...);
        constexpr auto named = namedLeaves<S>();
        for (int const n : times)
        {
            if (n > 1)
            {
                return Naming::repeated;
            }
        }
        for (std::size_t k = 0; k < times.size(); ++k)
        {
            if (times[k] == 0 && named[k])
            {
                return Naming::missingNamed;
            }
        }
        for (int const n : times)
        {
            if (n == 0)
            {
                return Naming::missingUnnamed;
            }
        }
        return Naming::complete;
    }
}

/** The last tag of the path \a P, a TagPath. */
template <typename P>
struct LastTagOf;

/** The last of \a Tags. */
template <typename... Tags>
struct LastTagOf<TagPath<Tags...>>
{
    /** The tag. */
    using type = std::tuple_element_t<sizeof...(Tags) - 1, std::tuple<Tags...>>;
};

/** The axis named by the path \a P, a tag or a TagPath: its last tag. */
template <typename P>
using LastTag = typename LastTagOf<AsTagPath<P>>::type;

/** \returns the cardinality of each leaf axis of \a space, in flat-index order */
template <typename S>
[[nodiscard]] constexpr std::array<Eigen::Index, std::size_t(leafAxisCount<S>())>
leafCardinalities(S const & space)
{
    auto result = std::array<Eigen::Index, std::size_t(leafAxisCount<S>())>{};
    if constexpr (IsDiscreteProduct<S>)
    {
        std::size_t offset = 0;
        forEachAxis(space, [&]<typename Axis>(Axis const & axis, Eigen::Index) {
            for (Eigen::Index const cardinality : leafCardinalities(axis))
            {
                result[offset++] = cardinality;
            }
        });
    }
    else
    {
        result[0] = space.cardinality();
    }
    return result;
}

/** \returns the part of a flat index that the value named by the value path \a P contributes */
template <typename P, typename S>
[[nodiscard]] constexpr Eigen::Index
namedOffset(S const & space)
{
    ValueLocation const location = locateValuePath<P>(space);
    return location.index * location.stride;
}

}  // namespace internal

/** \ingroup Discrete
 * \brief An index in each leaf axis of the axis named by the path \a Path (a tag or a TagPath), as
 * at() makes it, for valueOf().
 */
template <typename Path, std::size_t N>
struct AxisIndex
{
    std::array<Eigen::Index, N> indices;  ///< one per leaf axis of the one named, in flat-index order
};

/** \ingroup Discrete
 * \returns the value with index \a indices in the axis named by the path \a Path, for valueOf(): one
 * index for an axis, <tt>at<Cell>(3)</tt>, or one per leaf axis, in order, for a product,
 * <tt>at<GridCells<Position>>(3, 7)</tt>. It is how an axis without names, such as a Finite one, is
 * given its value, and a Named axis takes it too, for an index known only at run time.
 */
template <typename Path, std::integral... I>
[[nodiscard]] constexpr AxisIndex<Path, sizeof...(I)>
at(I... indices)
{
    using Axis = internal::LastTag<Path>;
    static_assert(DiscreteSpace<Axis>, "at<Axis>() names an axis of a discrete space");
    static_assert(
        sizeof...(I) == std::size_t(internal::leafAxisCount<Axis>()),
        "give one index for each leaf axis of the axis named"
    );
    return {{Eigen::Index(indices)...}};
}

namespace internal
{

/** \returns the part of a flat index of \a space that \a index contributes */
template <typename Path, std::size_t N, typename S>
[[nodiscard]] constexpr Eigen::Index
indexedOffset(S const & space, AxisIndex<Path, N> const & index)
{
    auto const [stride, axis] = locateAxisByPath<Path>(space);
    auto const cardinalities = leafCardinalities(*axis);
    Eigen::Index flat = 0;
    for (std::size_t k = 0; k < N; ++k)
    {
        EMRYS_FILTER_ASSERT(index.indices[k] >= 0 && index.indices[k] < cardinalities[k] && "an index is out of range");
        flat = flat * cardinalities[k] + index.indices[k];
    }
    return flat * stride;
}

}  // namespace internal

/** \ingroup Discrete
 * \returns the flat index of the value of \a space with the named values \a Values and the indices
 * \a indices (see at()), which together give each leaf axis its value once, in any order. A named
 * value is a value's tag, or a TagPath ending in one when several axes share it:
 * <tt>valueOf<Turn, Outlier>(Mode{})</tt>, <tt>valueOf<Turn>(space, at<Cell>(3))</tt>. Naming an axis
 * twice or leaving one out doesn't compile, and an index out of range is asserted.
 */
template <typename... Values, DiscreteSpace S, typename... Paths, std::size_t... N>
[[nodiscard]] constexpr Eigen::Index
valueOf(S const & space, AxisIndex<Paths, N> const &... indices)
{
    constexpr bool named = sizeof...(Values) + sizeof...(Paths) > 0;
    static_assert(named, "name a value");
    constexpr internal::Naming naming = !named ? internal::Naming::unresolved
                                               : internal::namingOf<S>(
                                                     static_cast<std::tuple<Values...> *>(nullptr),
                                                     static_cast<std::tuple<Paths...> *>(nullptr)
                                                 );
    static_assert(naming != internal::Naming::repeated, "an axis is named more than once");
    static_assert(naming != internal::Naming::missingNamed, "name one value in each axis");
    static_assert(naming != internal::Naming::missingUnnamed, "an axis without names takes its index from at<Axis>(i)");
    return (Eigen::Index{0} + ... + internal::namedOffset<Values>(space))
         + (Eigen::Index{0} + ... + internal::indexedOffset(space, indices));
}

/** \ingroup Discrete
 * valueOf(space, indices...) in the FixedSizeSpace \a S, named as the first template argument:
 * <tt>valueOf<Mode, Turn, Outlier>()</tt> (decisions.md [space-storage]).
 */
template <DiscreteSpace S, typename... Values, typename... Paths, std::size_t... N>
    requires FixedSizeSpace<S>
[[nodiscard]] constexpr Eigen::Index
valueOf(AxisIndex<Paths, N> const &... indices)
{
    return valueOf<Values...>(S{}, indices...);
}

}  // namespace Emrys

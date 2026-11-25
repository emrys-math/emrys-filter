#pragma once

/** \file
 * ModeSequence, the discrete space of the last few values of a mode, and Lagged, the tag of each of
 * its axes: the mode space of a GpbFilter's state, which keeps a component per sequence
 * (roadmap step 17zn). The internal namespace reads the current mode of a value of any discrete
 * space, which is the value itself unless the space is a ModeSequence. Read \ref Distributions next.
 */

#include <cstddef>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Discrete.h>

namespace Emrys
{

/** \ingroup Discrete
 * \brief The mode space \a ModeSpace_ as it was \a Lag_ steps before the current one: the tag of one
 * axis of a ModeSequence, a copy of the mode space that a path can name.
 *
 * <tt>Lagged<Maneuver, 0></tt>, which Current spells, is the mode now, and <tt>Lagged<Maneuver, 1></tt>
 * the mode one step earlier. It derives from the mode space, so it has the mode space's values and
 * axes: <tt>TagPath<Current<Maneuver>, Turn></tt> names the value \c Turn of the current mode.
 *
 * \tparam ModeSpace_  the mode space (see \ref Discrete)
 * \tparam Lag_        how many steps before the current one
 */
template <DiscreteSpace ModeSpace_, std::size_t Lag_>
struct Lagged : ModeSpace_
{
    using ModeSpaceType = ModeSpace_;         ///< the mode space
    static constexpr std::size_t Lag = Lag_;  ///< how many steps before the current one

    Lagged() = default;

    /** The copy of \a space, with its cardinality. */
    constexpr explicit Lagged(ModeSpace_ space)
        : ModeSpace_{std::move(space)}
    {}

    /** Equal when the mode spaces are. */
    friend bool
    operator==(Lagged const &, Lagged const &) = default;
};

/** \ingroup Discrete
 * \brief The current mode's axis of a ModeSequence over \a ModeSpace_: Lagged by no step.
 */
template <DiscreteSpace ModeSpace_>
using Current = Lagged<ModeSpace_, 0>;

namespace internal
{

/** The DiscreteProduct that a ModeSequence of the modes \a Lags steps back is, as \c type. */
template <typename ModeSpace_, typename Lags>
struct ModeSequenceProductOf;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename ModeSpace_, std::size_t... I>
struct ModeSequenceProductOf<ModeSpace_, std::index_sequence<I...>>
{
    /// the product, the oldest mode first
    using type = DiscreteProduct<Lagged<ModeSpace_, sizeof...(I) - 1 - I>...>;

    /** \returns the product with every axis a copy of \a space */
    [[nodiscard]] static constexpr type
    of(ModeSpace_ const & space)
    {
        return type{Lagged<ModeSpace_, sizeof...(I) - 1 - I>{space}...};
    }
};

/// \endcond

}  // namespace internal

/** \ingroup Discrete
 * \brief The sequences of the last \a Length_ values of the mode space \a ModeSpace_, the oldest
 * first and the current one last.
 *
 * It is the DiscreteProduct of \a Length_ copies of the mode space, the axis of the mode \a k steps
 * back tagged <tt>Lagged<ModeSpace_, k></tt>, so everything that reads a product reads it: a
 * marginal over <tt>Current<ModeSpace_></tt> is the distribution of the mode now, and a value is
 * named by a path through an axis. The current mode varies fastest in the flat index, so with
 * \f$ M \f$ modes the sequence \f$ (m_{k-1}, m_k) \f$ has the flat index \f$ M m_{k-1} + m_k \f$.
 *
 * \code
 * using Sequence = ModeSequence<Maneuver, 2>;                    // the last two maneuvers: 4 values
 * valueOf<TagPath<Lagged<Maneuver, 1>, Cruise>, TagPath<Current<Maneuver>, Turn>>(Sequence{});  // 1
 * Sequence{}.currentOf(1);                                       // 1, the index of Turn
 * Sequence{}.following(1, 0);                                    // 2: Turn, then Cruise
 * \endcode
 *
 * A step of the mode drops the oldest value and appends the new one (following()). The filter that
 * keeps a component per sequence is the GpbFilter.
 *
 * \tparam ModeSpace_  the mode space (see \ref Discrete)
 * \tparam Length_     the number of modes in a sequence, at least one
 */
template <DiscreteSpace ModeSpace_, std::size_t Length_>
class ModeSequence : public internal::ModeSequenceProductOf<ModeSpace_, std::make_index_sequence<Length_>>::type
{
    static_assert(Length_ >= 1, "a mode sequence holds at least one mode, the current one");

    /** what makes the product */
    using Maker = internal::ModeSequenceProductOf<ModeSpace_, std::make_index_sequence<Length_>>;

public:

    using ProductType = typename Maker::type;       ///< the product of the copies, the oldest mode first
    using ModeSpaceType = ModeSpace_;               ///< the mode space
    static constexpr std::size_t Length = Length_;  ///< the number of modes in a sequence

    ModeSequence() = default;

    /** The sequences of the values of \a space. */
    constexpr explicit ModeSequence(ModeSpace_ const & space)
        : ProductType{Maker::of(space)}
    {}

    /** \returns the mode space, that of the current mode */
    [[nodiscard]] constexpr ModeSpace_ const &
    current() const
    {
        return this->template axis<Length_ - 1>();
    }

    /** \returns the flat index in the mode space of the current mode of the sequence with flat index
     * \a sequence
     */
    [[nodiscard]] constexpr Eigen::Index
    currentOf(Eigen::Index sequence) const
    {
        EMRYS_FILTER_ASSERT(sequence >= 0 && sequence < this->cardinality() && "the sequence is out of range");
        return sequence % current().cardinality();
    }

    /** \returns the flat index of the sequence that follows the one with flat index \a sequence when
     * the mode moves to the one with flat index \a next: \a sequence less its oldest mode, then
     * \a next
     */
    [[nodiscard]] constexpr Eigen::Index
    following(Eigen::Index sequence, Eigen::Index next) const
    {
        EMRYS_FILTER_ASSERT(sequence >= 0 && sequence < this->cardinality() && "the sequence is out of range");
        Eigen::Index const modes = current().cardinality();
        EMRYS_FILTER_ASSERT(next >= 0 && next < modes && "the mode is out of range");
        return (sequence % (this->cardinality() / modes)) * modes + next;
    }

    /** Equal when the mode spaces are. */
    friend bool
    operator==(ModeSequence const &, ModeSequence const &) = default;
};

namespace internal
{

/** Whether \a S is a ModeSequence. */
template <typename S>
inline constexpr bool isModeSequence = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename ModeSpace_, std::size_t Length_>
inline constexpr bool isModeSequence<ModeSequence<ModeSpace_, Length_>> = true;

/// \endcond

/** The space of the current mode of a value of the discrete space \a S, as \c type: \a S itself,
 * unless it is a ModeSequence, whose values each hold a current mode.
 */
template <typename S>
struct CurrentModeSpaceOf
{
    using type = S;  ///< the space itself
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename ModeSpace_, std::size_t Length_>
struct CurrentModeSpaceOf<ModeSequence<ModeSpace_, Length_>>
{
    using type = ModeSpace_;  ///< the mode space of the sequence
};

/// \endcond

/** The space of the current mode of a value of \a S (see CurrentModeSpaceOf). */
template <typename S>
using CurrentModeSpace = typename CurrentModeSpaceOf<S>::type;

/** \returns the space of the current mode of a value of \a space: \a space itself, or the mode
 * space of a ModeSequence
 */
template <typename S>
[[nodiscard]] constexpr CurrentModeSpace<S> const &
currentModeSpace(S const & space)
{
    if constexpr (isModeSequence<S>)
    {
        return space.current();
    }
    else
    {
        return space;
    }
}

/** \returns the flat index, in currentModeSpace(space), of the current mode of the value of
 * \a space with flat index \a value: \a value itself, or the last mode of a sequence
 */
template <typename S>
[[nodiscard]] constexpr Eigen::Index
currentMode(S const & space, Eigen::Index value)
{
    if constexpr (isModeSequence<S>)
    {
        return space.currentOf(value);
    }
    else
    {
        return value;
    }
}

}  // namespace internal

}  // namespace Emrys

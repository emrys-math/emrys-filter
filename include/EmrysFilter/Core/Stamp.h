#pragma once

/** \file
 * What parameters and their views share (decisions.md [view-stamp], [params-and-density]): Stamp, the modification
 * stamp that parameters keep; ValuesCheck and ShapeCheck, by which a view checks that its parameters have not changed
 * under it; ViewHolder, by which a marginal of a view keeps that view; and CallbackGuard, which checks that a callback
 * leaves its parameters alone. Each detects a change when a view is read and prevents nothing. Everything here is
 * in `internal`. Read Print.h next.
 */

#include <type_traits>

#include <EmrysFilter/Core/Macros.h>

namespace Emrys::internal
{

/** The modification stamp of parameters (decisions.md [view-stamp]): two counters that their mutators
 * advance, one for any change and one for a change of size or structure, which their views compare
 * with what they saw when they were made. Only the parameters write it; a view only reads it.
 *
 * It is kept in every build, so that no configuration changes the parameters' layout; only the
 * checks that read it turn off, with \c EMRYS_FILTER_ASSERT. A copy starts a new count, as copied
 * parameters are new parameters; a move advances the moved-from stamp, and an assignment the
 * assigned-to one, so parameters whose special members are defaulted need nothing more.
 * Hand-written ones call touchShape() themselves.
 *
 * The counters are plain integers, not atomics: the stamp detects a change made before a read,
 * on any thread, and does nothing to order a change and a read made at the same time.
 */
class Stamp
{
public:

    Stamp() = default;

    /** A new count: views of the source do not see the copy. */
    Stamp(Stamp const &) noexcept {}

    /** A new count, and \a other's parameters, moved from, have changed shape. */
    Stamp(Stamp && other) noexcept
    {
        other.touchShape();
    }

    /** This stamp's parameters have been assigned to, which changes their shape. */
    Stamp &
    operator=(Stamp const &) noexcept
    {
        touchShape();
        return *this;
    }

    /** This stamp's parameters have been assigned to, and \a other's moved from. */
    Stamp &
    operator=(Stamp && other) noexcept
    {
        touchShape();
        other.touchShape();
        return *this;
    }

    /** Records a change of values that leaves the size and structure as they were. */
    void
    touchValues() noexcept
    {
        ++m_values;
    }

    /** Records a change of size or structure, such as a resize, insertion, removal, assignment, or
     * move; it is a change of values too.
     */
    void
    touchShape() noexcept
    {
        ++m_values;
        ++m_shape;
    }

    /** \returns how many changes the parameters have had */
    [[nodiscard]] unsigned
    values() const noexcept
    {
        return m_values;
    }

    /** \returns how many changes of size or structure the parameters have had */
    [[nodiscard]] unsigned
    shape() const noexcept
    {
        return m_shape;
    }

private:

    unsigned m_values = 0;  ///< changes of any kind, wrapping
    unsigned m_shape = 0;   ///< changes of size or structure, wrapping
};

/** Checks parameters for any change: reading them through get() after they have changed in any way
 * is a programmer error. It detects the change when the view is read and prevents nothing: a
 * mutator never waits or fails, and no threads are synchronized. \a Source must befriend
 * ValuesCheck and hold its Stamp in \c m_stamp. The source must outlive the check, which does not
 * catch its destruction or reallocation; that is the sanitizer's job.
 */
template <typename Source>
class ValuesCheck
{
public:

    /** Records the stamp of \a source as it is now. */
    explicit ValuesCheck(Source const & source)
        : m_source{&source}
        , m_seen{source.m_stamp.values()}
    {}

    /** \returns the checked source; a programmer error if it has changed since this was made */
    [[nodiscard]] Source const &
    get() const
    {
        EMRYS_FILTER_ASSERT(
            m_source->m_stamp.values() == m_seen && "the parameters changed while a view of them was alive"
        );
        return *m_source;
    }

private:

    Source const * m_source;  ///< the checked source, which outlives this
    unsigned m_seen;          ///< the source's count of changes when this was made
};

/** Checks parameters for a change of size or structure: reading them through get() after one is a
 * programmer error, while their values may change. Like ValuesCheck, it detects and prevents
 * nothing. \a Source must befriend ShapeCheck and hold its Stamp in \c m_stamp. The source must
 * outlive the check.
 */
template <typename Source>
class ShapeCheck
{
public:

    /** Records the stamp of \a source as it is now. */
    explicit ShapeCheck(Source const & source)
        : m_source{&source}
        , m_seen{source.m_stamp.shape()}
    {}

    /** \returns the checked source; a programmer error if its size or structure has changed */
    [[nodiscard]] Source const &
    get() const
    {
        EMRYS_FILTER_ASSERT(
            m_source->m_stamp.shape() == m_seen
            && "the parameters changed size or structure while a marginal of them was alive"
        );
        return *m_source;
    }

private:

    Source const * m_source;  ///< the checked source, which outlives this
    unsigned m_seen;          ///< the source's count of changes of shape when this was made
};

/** Checks that callbacks into user code leave the parameters alone: forEach() and removeIf() make
 * one after their own change of the parameters, and call check() after each callback. Only the
 * element passed in may change, through the handle given for it, which does not touch the stamp.
 */
class CallbackGuard
{
public:

    /** Watches \a stamp from now on. */
    explicit CallbackGuard(Stamp const & stamp) noexcept
        : m_stamp{&stamp}
        , m_seen{stamp.values()}
    {}

    /** A programmer error if the parameters have changed since this guard was made. */
    void
    check() const
    {
        EMRYS_FILTER_ASSERT(m_stamp->values() == m_seen && "the parameters changed inside their own callback");
    }

private:

    [[maybe_unused]] Stamp const * m_stamp;  ///< the watched stamp
    unsigned m_seen;                         ///< its count when the guard was made
};

/** Holds a copy of a read-only view, and with it the view's check. */
template <typename View>
class ViewHolder
{
public:

    /** Copies \a view, and with it its check of the parameters. */
    explicit ViewHolder(View const & view)
        : m_view{view}
    {}

    /** \returns the held view */
    [[nodiscard]] View const &
    get() const
    {
        return m_view;
    }

private:

    View m_view;  ///< the copy of the view
};

/** A ShapeCheck for parameters, a ViewHolder for read-only views. */
template <typename Source>
using HolderFor = std::conditional_t<Source::IsParams, ShapeCheck<Source>, ViewHolder<Source>>;

}  // namespace Emrys::internal

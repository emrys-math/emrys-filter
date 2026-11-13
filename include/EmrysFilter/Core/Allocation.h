#pragma once

/** \file
 * The allocator conventions of decisions.md [allocation]: DefaultAllocator, Capacity, and the internal
 * rebinding alias and buffer type every type with a run-time number of elements uses. Read
 * Execution.h next.
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Macros.h>

namespace Emrys
{

/** \defgroup Allocation Allocation
 * \ingroup Core
 *
 * Storage sized by a space never touches the heap when the space is fixed or bounded (see
 * \ref Spaces), and an unbounded space allocates through Eigen's ordinary heap path. Storage
 * sized by a run-time count of elements, such as a mixture's components or a particle set's
 * particles, belongs on the heap under the caller's control:
 *  - every type with a run-time count takes an \c Allocator template parameter after its
 *    scalar, DefaultAllocator unless given, with \c allocator_type, \c get_allocator(), and an
 *    allocator-extended overload of every constructor. A <tt>std::pmr::polymorphic_allocator<></tt>
 *    is such an allocator, so memory resources need nothing more;
 *  - a type made from an existing one (a copy, a view's copy(), a filter's state, the result of
 *    a merge or a resample) takes the source's allocator. A copy constructor keeps the source's
 *    capacity too, and its scratch's, rather than taking what
 *    \c std::allocator_traits::select_on_container_copy_construction() gives, which for a
 *    <tt>std::pmr::polymorphic_allocator</tt> is the default resource. A copy therefore lives on
 *    its source's resource and must not outlive it; the allocator-extended copy constructor puts
 *    it elsewhere. Copy assignment keeps the destination's allocator;
 *  - a type given a Capacity reserves it once, and growing past it is a precondition
 *    violation;
 *  - scratch sized by a count comes from the same allocator and is reused across calls, so a
 *    filter's steps allocate nothing once it has run one.
 */

/** \ingroup Allocation
 * The allocator a type with a run-time count uses unless given one.
 */
using DefaultAllocator = std::allocator<std::byte>;

/** \ingroup Allocation
 * \brief The most elements a container will hold, reserved once: a mixture's components or a
 * particle set's particles. Going past it is a precondition violation.
 */
struct Capacity
{
    Eigen::Index count;  ///< the most elements, at least zero
};

namespace internal
{

/** \a Allocator re-bound to \a T. */
template <typename Allocator, typename T>
using ReBound = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

/** A contiguous, allocator-aware buffer of \a T: a \c std::vector whose copy keeps the source's
 * allocator and reserves its capacity, where a vector's takes what
 * \c std::allocator_traits::select_on_container_copy_construction() gives and only the size. An
 * implicit copy of the type that holds it then keeps both too, scratch included (see \ref
 * Allocation).
 *
 * It is built with parentheses throughout: it has the vector's initializer-list constructor, which
 * braces would select whenever a \a T can be made from the arguments.
 */
template <typename T, typename Allocator>
class Buffer : public std::vector<T, ReBound<Allocator, T>>
{
public:

    using Base = std::vector<T, ReBound<Allocator, T>>;  ///< the vector it is
    using Base::Base;                                    ///< the vector's constructors

    Buffer() = default;  ///< empty, from a default-constructed allocator

    /** Copies the vector \a vector into storage from its allocator, so that a caller's vector
     * converts where a Buffer is taken, as it did when a Buffer was a vector.
     */
    Buffer(Base const & vector)
        : Base(vector, vector.get_allocator())
    {}

    /** Takes the vector \a vector's storage and allocator. */
    Buffer(Base && vector) noexcept
        : Base(std::move(vector))
    {}

    /** Copies \a other into storage from its allocator, reserving its capacity. */
    Buffer(Buffer const & other)
        : Buffer(other, other.get_allocator())
    {}

    /** Copies \a other into storage from \a allocator, reserving its capacity. */
    Buffer(Buffer const & other, typename Base::allocator_type const & allocator)
        : Base(allocator)
    {
        this->reserve(other.capacity());
        this->assign(other.begin(), other.end());
    }

    Buffer(Buffer &&) noexcept = default;  ///< takes \a other's storage and allocator

    /** Copies into this buffer's storage, which keeps its allocator. */
    Buffer &
    operator=(Buffer const &) = default;

    /** As a \c std::vector's move assignment. */
    Buffer &
    operator=(Buffer &&) = default;

    ~Buffer() = default;
};

/** \returns an empty Buffer that allocates through \a allocator */
template <typename T, typename Allocator>
[[nodiscard]] Buffer<T, Allocator>
bufferOf(Allocator const & allocator)
{
    return Buffer<T, Allocator>(ReBound<Allocator, T>(allocator));
}

/** An empty base for an element that holds a \a T: it declares no \c allocator_type, so a
 * container constructs the element by its plain constructors.
 */
template <typename T>
struct AllocatorTypeOf
{};

/** For a \a T with an allocator: declares its \c allocator_type, so that a container whose
 * allocator passes itself on, as a <tt>std::pmr::polymorphic_allocator</tt> does, builds the
 * element by uses-allocator construction and the \a T inside from the container's allocator.
 */
template <typename T>
    requires requires { typename T::allocator_type; }
struct AllocatorTypeOf<T>
{
    using allocator_type = typename T::allocator_type;  ///< the held object's allocator
};

}  // namespace internal

}  // namespace Emrys

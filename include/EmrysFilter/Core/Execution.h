#pragma once

/** \file
 * Execution policies (decisions.md [execution-policies]): execution::sequential, the ExecutionPolicy concept, and the
 * `internal` primitives that every loop over a run-time count of elements runs through, a for-each
 * over indices or over chunks, a reduction, a scan, a gather, a compaction, and a fill, with the
 * iterators they run over, one over indices and one over elements a view makes when read. Nothing
 * here includes \c \<execution\>: where oneTBB is installed, GCC's library would then make every
 * program that includes this library link TBB, whether or not it passes a policy. Read Time.h
 * next.
 */

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>

namespace Emrys
{

/** \defgroup Execution Execution policies
 * \ingroup Core
 *
 * Where loops run (decisions.md [execution-policies]). An operation whose work grows with a run-time count of
 * elements (a particle set's particles, a grid's cells, a mixture's components, a Monte Carlo
 * estimate's draws) has an overload that takes an execution policy as its first argument, as the
 * standard algorithms do, and the overload without one runs Sequential:
 *
 * \code
 * #include <execution>
 * filter.predict(std::execution::par, rng, dt);
 * filter.update(std::execution::par, fix);
 * upkeep(std::execution::par, filter.state(), rng);
 * auto const moments = filter.state().moments(std::execution::par);
 * \endcode
 *
 *  - execution::sequential, the constant of type execution::Sequential, runs plain loops on the
 *    calling thread. It is what every operation did before it took a policy, bitwise, and it
 *    brings in nothing from \c \<execution\>.
 *  - Any policy of the standard library, such as \c std::execution::par or
 *    \c std::execution::par_unseq, runs the loops through \c std::for_each with that policy. The
 *    caller includes \c \<execution\>; with GCC's library, a parallel run also links TBB, without
 *    which the standard library runs the loop on one thread.
 *  - A policy of the user's own, such as a handle to an engine's thread pool where the standard
 *    library's parallel algorithms are absent or unwanted, is any type with a const
 *    <tt>forEachIndex(count, f)</tt> that calls \a f(i) exactly once for each \c i in
 *    <tt>[0, count)</tt>, in any order and possibly at once, and returns when every call has.
 *    Every loop in the library is built on that one operation. It runs the same chunked loops as
 *    a standard policy, so the rest of this list holds for it as for \c par, except where the
 *    scratch comes from. A policy may also have a const <tt>get_allocator()</tt>, from which the
 *    library takes the scratch of its reductions, scans, and compactions, re-bound to each
 *    element type, instead of from the heap; it allocates on the calling thread, before or after
 *    a loop and never inside one, so that allocator need not be safe to call from several
 *    threads. The library calls nothing else on the policy:
 * \code
 * struct PoolPolicy
 * {
 *     ThreadPool * pool;
 *     std::pmr::memory_resource * scratch;
 *
 *     template <typename F>
 *     void
 *     forEachIndex(Eigen::Index count, F const & f) const
 *     {
 *         pool->parallelFor(0, count, f);  // returns when every f(i) has
 *     }
 *
 *     [[nodiscard]] std::pmr::polymorphic_allocator<>
 *     get_allocator() const
 *     {
 *         return scratch;
 *     }
 * };
 *
 * filter.predict(PoolPolicy{&pool, &arena}, rng, dt);
 * \endcode
 *  - A for-each, which writes one element per index, gives the same result under every policy.
 *    A reduction or a scan, such as a log-sum-exp, a weighted mean, or a cumulative sum, splits
 *    the elements into fixed chunks of internal::chunkSize, runs the chunks under the policy, and
 *    combines their results in order. Its result is the same at any thread count and from run
 *    to run, and differs from Sequential's only by rounding. Below a chunk it is Sequential's.
 *  - The draws from a std::uniform_random_bit_generator stay sequential: they are drawn before a
 *    parallel loop, into scratch, so the loop body is a pure function of its index.
 *  - What a parallel loop calls, such as a model's propagate(), a measurement model's function, or
 *    a ParticleSet::forEach() callback, runs concurrently for different elements, so it must not
 *    write shared state. Under \c par_unseq and \c unseq it must not allocate or synchronize
 *    either, which rules out a space of dynamic size.
 *  - The standard library's parallel algorithms allocate, and so do the chunks' results, so a
 *    filter's steps allocate nothing (\ref Allocation) only under Sequential. Under a standard
 *    policy that memory comes from the heap, not from the state's allocator, which the standard
 *    algorithms give no way to pass. Under a user's policy the library's scratch is the only
 *    memory a step asks for beyond Sequential's, so a policy with get_allocator() keeps a
 *    filter's steps off the heap.
 *  - \c std::execution::seq is a standard policy like the others, not Sequential: it runs on the
 *    calling thread but takes the chunked path, so it gives the parallel policies' results
 *    bitwise, allocates as they do, and, as the standard says of \c sequenced_policy, calls the
 *    elements in no specified order. It reproduces a parallel run on one thread; Sequential is
 *    the plain loops.
 */

/** \ingroup Execution
 * \brief The library's execution policies: the constant execution::sequential and its type. The
 * standard policies are variables, such as \c std::execution::par, so the library's is one too,
 * and both kinds read the same in the argument that takes one (decisions.md [policy-factories]).
 * The device policy of roadmap step 19 joins it.
 */
namespace execution
{

/** \ingroup Execution
 * \brief The type of execution::sequential, the default execution policy: plain loops on the
 * calling thread, in index order. The constant is what a call passes; the type names it where a
 * type is needed.
 */
struct Sequential
{};

/** \ingroup Execution
 * The default execution policy: plain loops on the calling thread, in index order (Sequential).
 */
inline constexpr Sequential sequential{};

}  // namespace execution

namespace internal
{

/** \brief A random-access iterator over the indices themselves, for running the standard
 * algorithms over an index range.
 *
 * It reports \c random_access_iterator_tag both as its \c iterator_concept, which the C++20
 * iterator concepts read, and as its \c iterator_category, which the legacy iterator
 * requirements and so the parallel algorithms read. \c std::views::iota's iterator does not: it
 * yields values rather than references, which the legacy requirements forbid a forward iterator
 * to do, so it reports an input category, and a standard library may run an input range of a
 * parallel algorithm on one thread. This one yields values too and claims the category anyway,
 * which every standard library the library builds with accepts.
 */
class CountingIterator
{
public:

    using iterator_concept = std::random_access_iterator_tag;   ///< random access, for the C++20 iterator concepts
    using iterator_category = std::random_access_iterator_tag;  ///< random access, for the parallel algorithms
    using value_type = Eigen::Index;                            ///< an index
    using difference_type = Eigen::Index;                       ///< the distance between indices
    using pointer = void;                                       ///< no pointer: indices are values
    using reference = Eigen::Index;                             ///< dereferencing gives the index itself

    CountingIterator() = default;

    /** An iterator at \a index. */
    explicit CountingIterator(Eigen::Index index)
        : m_index{index}
    {}

    /** \returns the index */
    [[nodiscard]] Eigen::Index
    operator*() const
    {
        return m_index;
    }

    /** \returns the index \a n past this one */
    [[nodiscard]] Eigen::Index
    operator[](difference_type n) const
    {
        return m_index + n;
    }

    /** Moves to the next index. */
    CountingIterator &
    operator++()
    {
        ++m_index;
        return *this;
    }

    /** Moves to the next index, returning the iterator as it was. */
    CountingIterator
    operator++(int)
    {
        CountingIterator const before = *this;
        ++m_index;
        return before;
    }

    /** Moves to the previous index. */
    CountingIterator &
    operator--()
    {
        --m_index;
        return *this;
    }

    /** Moves to the previous index, returning the iterator as it was. */
    CountingIterator
    operator--(int)
    {
        CountingIterator const before = *this;
        --m_index;
        return before;
    }

    /** Moves \a n indices forward. */
    CountingIterator &
    operator+=(difference_type n)
    {
        m_index += n;
        return *this;
    }

    /** Moves \a n indices back. */
    CountingIterator &
    operator-=(difference_type n)
    {
        m_index -= n;
        return *this;
    }

    /** \returns \a it moved \a n indices forward */
    [[nodiscard]] friend CountingIterator
    operator+(CountingIterator it, difference_type n)
    {
        return CountingIterator{it.m_index + n};
    }

    /** \returns \a it moved \a n indices forward */
    [[nodiscard]] friend CountingIterator
    operator+(difference_type n, CountingIterator it)
    {
        return CountingIterator{it.m_index + n};
    }

    /** \returns \a it moved \a n indices back */
    [[nodiscard]] friend CountingIterator
    operator-(CountingIterator it, difference_type n)
    {
        return CountingIterator{it.m_index - n};
    }

    /** \returns the number of indices from \a b to \a a */
    [[nodiscard]] friend difference_type
    operator-(CountingIterator a, CountingIterator b)
    {
        return a.m_index - b.m_index;
    }

    /** Iterators compare as their indices do. */
    [[nodiscard]] friend auto
    operator<=>(CountingIterator const &, CountingIterator const &) = default;

private:

    Eigen::Index m_index = 0;  ///< the index
};

static_assert(std::random_access_iterator<CountingIterator>);

/** \brief A random-access iterator over the elements of a range that makes each one when it is
 * read, as \c (*range)[i].
 *
 * It is how a view gives a value per element that it does not store, such as a marginal of each
 * component of a mixture, with no allocation: it holds the range and an index, and dereferencing
 * gives the element by value, as CountingIterator gives the index. \a Range must have a
 * \c value_type and a \c const \c operator[] that returns one, and must outlive the iterator.
 */
template <typename Range>
class ElementIterator
{
public:

    using iterator_concept = std::random_access_iterator_tag;   ///< random access, for the C++20 iterator concepts
    using iterator_category = std::random_access_iterator_tag;  ///< random access, for the parallel algorithms
    using value_type = typename Range::value_type;              ///< an element, made when read
    using difference_type = Eigen::Index;                       ///< the distance between elements
    using pointer = void;                                       ///< no pointer: elements are values
    using reference = value_type;                               ///< dereferencing gives the element itself

    ElementIterator() = default;

    /** An iterator at element \a index of \a range. */
    ElementIterator(Range const & range, Eigen::Index index)
        : m_range{&range}
        , m_index{index}
    {}

    /** \returns the element */
    [[nodiscard]] value_type
    operator*() const
    {
        return (*m_range)[m_index];
    }

    /** \returns the element \a n past this one */
    [[nodiscard]] value_type
    operator[](difference_type n) const
    {
        return (*m_range)[m_index + n];
    }

    /** Moves to the next element. */
    ElementIterator &
    operator++()
    {
        ++m_index;
        return *this;
    }

    /** Moves to the next element, returning the iterator as it was. */
    ElementIterator
    operator++(int)
    {
        ElementIterator const before = *this;
        ++m_index;
        return before;
    }

    /** Moves to the previous element. */
    ElementIterator &
    operator--()
    {
        --m_index;
        return *this;
    }

    /** Moves to the previous element, returning the iterator as it was. */
    ElementIterator
    operator--(int)
    {
        ElementIterator const before = *this;
        --m_index;
        return before;
    }

    /** Moves \a n elements forward. */
    ElementIterator &
    operator+=(difference_type n)
    {
        m_index += n;
        return *this;
    }

    /** Moves \a n elements back. */
    ElementIterator &
    operator-=(difference_type n)
    {
        m_index -= n;
        return *this;
    }

    /** \returns \a it moved \a n elements forward */
    [[nodiscard]] friend ElementIterator
    operator+(ElementIterator it, difference_type n)
    {
        return it += n;
    }

    /** \returns \a it moved \a n elements forward */
    [[nodiscard]] friend ElementIterator
    operator+(difference_type n, ElementIterator it)
    {
        return it += n;
    }

    /** \returns \a it moved \a n elements back */
    [[nodiscard]] friend ElementIterator
    operator-(ElementIterator it, difference_type n)
    {
        return it -= n;
    }

    /** \returns the number of elements from \a b to \a a, which iterate one range */
    [[nodiscard]] friend difference_type
    operator-(ElementIterator const & a, ElementIterator const & b)
    {
        return a.m_index - b.m_index;
    }

    /** Iterators over one range compare as their indices do. */
    [[nodiscard]] friend bool
    operator==(ElementIterator const & a, ElementIterator const & b)
    {
        return a.m_index == b.m_index;
    }

    /** Iterators over one range order as their indices do. */
    [[nodiscard]] friend auto
    operator<=>(ElementIterator const & a, ElementIterator const & b)
    {
        return a.m_index <=> b.m_index;
    }

private:

    Range const * m_range = nullptr;  ///< the range, which outlives this
    Eigen::Index m_index = 0;         ///< the element's index
};

/** Whether the standard library runs \c std::for_each with a policy of type \a P, which holds for
 * exactly the types \c std::is_execution_policy_v names.
 *
 * Both standard libraries the library builds with declare the parallel overloads in
 * \c \<algorithm\> and define them in \c \<execution\>, and GCC's names \c std::is_execution_policy
 * only in \c \<execution\>, so asking the algorithm keeps that header, and the TBB link it
 * brings with GCC's library, out of the library. A
 * standard library without parallel algorithms has no such overload, and then only Sequential is
 * a policy.
 */
template <typename P>
concept StandardExecutionPolicy = requires(P && policy, CountingIterator i, void (*f)(Eigen::Index)) {
    std::for_each(std::forward<P>(policy), i, i, f);
};

/** A stand-in for the bodies the library hands a user's policy: callable on an index, through a
 * const reference. Declared only, for UserExecutionPolicy to call in an unevaluated context.
 */
struct IndexFunction
{
    /** Runs the body at index \a i. */
    void
    operator()(Eigen::Index i) const;
};

/** Whether \a P is a policy of the user's own, such as a handle to a thread pool: a type whose
 * const forEachIndex(count, f) can be called with a count and a body. Its contract is that of
 * the parallel standard policies: it calls \a f(i) exactly once for each \c i in
 * <tt>[0, count)</tt>, in any order and possibly at once from several threads, and returns when
 * every call has. It may also supply the scratch's memory (UserPolicyWithAllocator).
 */
template <typename P>
concept UserExecutionPolicy = requires(std::remove_cvref_t<P> const & policy, IndexFunction const & f) {
    policy.forEachIndex(Eigen::Index{}, f);
};

/** Whether \a P is a policy of the user's own that supplies the memory for the primitives'
 * scratch: one with a const get_allocator() as well as its forEachIndex().
 */
template <typename P>
concept UserPolicyWithAllocator =
    UserExecutionPolicy<P> && requires(std::remove_cvref_t<P> const & policy) { policy.get_allocator(); };

}  // namespace internal

/** \ingroup Execution
 * An execution policy: execution::sequential, a policy of the standard library such as
 * \c std::execution::par, or a policy of the user's own with a const forEachIndex(count, f) (see
 * \ref Execution). It constrains the leading argument of the overloads that take one, so that it
 * never collides with a generator, a time step, or a measurement.
 */
template <typename P>
concept ExecutionPolicy = std::same_as<std::remove_cvref_t<P>, execution::Sequential>
                       || internal::UserExecutionPolicy<P> || internal::StandardExecutionPolicy<P>;

namespace internal
{

/** Whether \a P is Sequential, the policy of plain loops: false for \c std::execution::seq, which
 * takes the chunked path as every standard policy does (see \ref Execution).
 */
template <typename P>
inline constexpr bool isSequential = std::same_as<std::remove_cvref_t<P>, execution::Sequential>;

/** Whether the first of \a Args is an execution policy: false for none. A function whose
 * arguments are all a pack, such as a fixed-step predict(inputs...), excludes it, so that a
 * leading policy goes to the overload that takes one rather than being read as an input.
 */
template <typename... Args>
inline constexpr bool leadsWithPolicy = false;

/** Whether \a First, the first of several arguments, is an execution policy. */
template <typename First, typename... Rest>
inline constexpr bool leadsWithPolicy<First, Rest...> = ExecutionPolicy<First>;

/** \returns \a size value-initialized elements of \a T as scratch for a primitive under \a policy:
 * from the policy's get_allocator(), re-bound to \a T, where a user's policy has one, and from the
 * heap otherwise. A primitive allocates its scratch on the calling thread, before or after its
 * loops and never inside them, so that allocator need not be safe to call from several threads.
 */
template <typename T, typename P>
[[nodiscard]] auto
scratchVector(P const & policy, std::size_t size)
{
    if constexpr (UserPolicyWithAllocator<P>)
    {
        using Allocator = ReBound<std::remove_cvref_t<decltype(policy.get_allocator())>, T>;
        return std::vector<T, Allocator>(size, Allocator(policy.get_allocator()));
    }
    else
    {
        return std::vector<T>(size);
    }
}

/** The elements in each chunk of a reduction or a scan under a policy other than Sequential.
 * Fixed, so that the partition, and so the rounding, depends on the count alone, and large
 * enough that a chunk's work outweighs scheduling it.
 */
inline constexpr Eigen::Index chunkSize = 1024;

/** \returns the number of chunks of chunkSize that \a count elements make, the last one short */
[[nodiscard]] constexpr Eigen::Index
chunkCount(Eigen::Index count)
{
    return (count + chunkSize - 1) / chunkSize;
}

/** Calls \a f(i) for each \c i in <tt>[0, count)</tt>: in order under Sequential, through a user's
 * policy's own forEachIndex(), and otherwise through \c std::for_each with \a policy, in any order
 * and possibly at once. Every other primitive is built on this one, so it is all that a user's
 * policy supplies.
 */
template <ExecutionPolicy P, typename F>
void
forEachIndex([[maybe_unused]] P && policy, Eigen::Index count, F && f)
{
    EMRYS_FILTER_ASSERT(count >= 0);
    if constexpr (isSequential<P>)
    {
        for (Eigen::Index i = 0; i < count; ++i)
        {
            f(i);
        }
    }
    else if constexpr (UserExecutionPolicy<P>)
    {
        std::as_const(policy).forEachIndex(count, [&f](Eigen::Index i) {
            f(i);
        });
    }
    else
    {
        std::for_each(std::forward<P>(policy), CountingIterator{0}, CountingIterator{count}, [&f](Eigen::Index i) {
            f(i);
        });
    }
}

/** Calls \a f(begin, length) on ranges that cover <tt>[0, count)</tt> once each: the whole range
 * under Sequential, if it is not empty, and otherwise its chunks of chunkSize, under \a policy.
 * For a body of whole-array Eigen expressions, which keep their vectorization within a chunk.
 */
template <ExecutionPolicy P, typename F>
void
forEachChunk(P && policy, Eigen::Index count, F && f)
{
    EMRYS_FILTER_ASSERT(count >= 0);
    if constexpr (isSequential<P>)
    {
        if (count > 0)
        {
            f(Eigen::Index{0}, count);
        }
    }
    else
    {
        forEachIndex(std::forward<P>(policy), chunkCount(count), [&](Eigen::Index chunk) {
            Eigen::Index const begin = chunk * chunkSize;
            f(begin, std::min(chunkSize, count - begin));
        });
    }
}

/** \returns the reduction of <tt>[0, count)</tt>, in which \a partial(begin, length) gives the
 * result of a range, a plain value such as a matrix rather than an Eigen expression, and
 * \a combine(a, b) joins the results of two adjacent ranges, \a a's before \a b's, associatively.
 *
 * Under Sequential the whole range is one call to \a partial, which is how a reduction ran before
 * it took a policy. Otherwise the ranges are the chunks, \a partial runs on each under \a policy,
 * and their results are combined in order, so the result depends on \a count but not on how the
 * chunks were scheduled. With a single chunk it is Sequential's. This is the two-level form of a
 * [parallel reduction](https://en.wikipedia.org/wiki/Reduction_operator), with the partition
 * fixed rather than taken from the thread count.
 */
template <ExecutionPolicy P, typename Partial, typename Combine>
[[nodiscard]] auto
reduceRanges(P && policy, Eigen::Index count, Partial const & partial, Combine const & combine)
{
    using T = std::remove_cvref_t<std::invoke_result_t<Partial const &, Eigen::Index, Eigen::Index>>;
    EMRYS_FILTER_ASSERT(count >= 0);
    if constexpr (isSequential<P>)
    {
        return T{partial(Eigen::Index{0}, count)};
    }
    else
    {
        Eigen::Index const chunks = chunkCount(count);
        // An empty range, as one chunk, is one call to partial(), as under Sequential.
        if (chunks <= 1)
        {
            return T{partial(Eigen::Index{0}, count)};
        }
        auto partials = scratchVector<T>(policy, static_cast<std::size_t>(chunks));
        forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
            partials[static_cast<std::size_t>(begin / chunkSize)] = partial(begin, length);
        });
        T result = std::move(partials[0]);
        for (std::size_t c = 1; c < partials.size(); ++c)
        {
            result = combine(std::move(result), partials[c]);
        }
        return result;
    }
}

/** \returns \a identity combined with \a f(i) for each \c i in <tt>[0, count)</tt> in order, by
 * \a combine, which must be associative: reduceRanges() with each range folded one element at a
 * time. \a f returns something convertible to \a T.
 */
template <ExecutionPolicy P, typename T, typename Combine, typename F>
[[nodiscard]] T
transformReduce(P && policy, Eigen::Index count, T const & identity, Combine const & combine, F const & f)
{
    return reduceRanges(
        std::forward<P>(policy),
        count,
        [&](Eigen::Index begin, Eigen::Index length) {
            T result = identity;
            for (Eigen::Index i = begin; i < begin + length; ++i)
            {
                result = combine(std::move(result), T(f(i)));
            }
            return result;
        },
        combine
    );
}

/** Writes the running sums of \a in to \a out, <tt>out[i] = in[0] + ... + in[i]</tt>, which may be
 * \a in itself.
 *
 * Under Sequential each sum adds the next element to the one before, in order. Otherwise each
 * chunk is summed under \a policy, the chunks' totals are added in order to give each chunk's
 * offset, and each chunk is then scanned from its offset under \a policy. The sums then depend on
 * the count alone, and differ from Sequential's by rounding past the first chunk. This is the
 * blocked form of a [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithms).
 */
template <ExecutionPolicy P, typename T>
void
cumulativeSum(P && policy, std::span<T const> in, std::span<T> out)
{
    EMRYS_FILTER_ASSERT(in.size() == out.size() && "the scan writes one sum per element");
    Eigen::Index const count = Eigen::Index(in.size());
    // Read each element before writing its sum, so that in and out may be the same.
    auto const scan = [&](Eigen::Index begin, Eigen::Index length, T const * offset) {
        auto const first = std::size_t(begin);
        T sum = offset ? T(*offset + in[first]) : in[first];
        out[first] = sum;
        for (std::size_t i = first + 1; i < std::size_t(begin + length); ++i)
        {
            sum += in[i];
            out[i] = sum;
        }
    };
    if (count == 0)
    {
        return;
    }
    if constexpr (isSequential<P>)
    {
        scan(0, count, nullptr);
    }
    else
    {
        Eigen::Index const chunks = chunkCount(count);
        if (chunks == 1)
        {
            scan(0, count, nullptr);
            return;
        }
        auto offsets = scratchVector<T>(policy, static_cast<std::size_t>(chunks));
        forEachChunk(policy, count, [&](Eigen::Index begin, Eigen::Index length) {
            auto const first = std::size_t(begin);
            T total = in[first];
            for (std::size_t i = first + 1; i < std::size_t(begin + length); ++i)
            {
                total += in[i];
            }
            offsets[std::size_t(begin / chunkSize)] = total;
        });
        // Chunk c starts from the sum of the chunks before it; chunk 0 starts from nothing.
        T carry = offsets[0];
        for (std::size_t c = 1; c < offsets.size(); ++c)
        {
            T const total = offsets[c];
            offsets[c] = carry;
            carry += total;
        }
        forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
            std::size_t const chunk = std::size_t(begin / chunkSize);
            scan(begin, length, chunk == 0 ? nullptr : &offsets[chunk]);
        });
    }
}

/** Copies column <tt>indices[j]</tt> of \a from into column \c j of \a to, for each \c j, under
 * \a policy. \a to must not overlap \a from.
 */
template <ExecutionPolicy P, typename FromDerived, typename To>
void
gather(P && policy, std::span<Eigen::Index const> indices, Eigen::MatrixBase<FromDerived> const & from, To && to)
{
    EMRYS_FILTER_ASSERT(to.cols() == Eigen::Index(indices.size()) && "gather one column per index");
    forEachIndex(std::forward<P>(policy), Eigen::Index(indices.size()), [&](Eigen::Index j) {
        Eigen::Index const source = indices[std::size_t(j)];
        EMRYS_FILTER_ASSERT(source >= 0 && source < from.cols() && "a gathered index is out of range");
        to.col(j) = from.col(source);
    });
}

/** Calls \a move(i, j) for each \c i in <tt>[0, count)</tt> with \a keep(i) true, where \c j is
 * the number of kept indices before \c i, and \returns the number kept: a compaction.
 *
 * Under Sequential the calls are in order of \c i, so \a move may write in place, since
 * <tt>j <= i</tt>. Otherwise \a keep runs under \a policy, a scan of its results gives each \c j,
 * and \a move runs under \a policy in any order, so it must write where neither \a keep nor
 * \a move reads.
 */
template <ExecutionPolicy P, typename Keep, typename Move>
Eigen::Index
copyIfIndex(P && policy, Eigen::Index count, Keep const & keep, Move const & move)
{
    EMRYS_FILTER_ASSERT(count >= 0);
    if constexpr (isSequential<P>)
    {
        Eigen::Index out = 0;
        for (Eigen::Index i = 0; i < count; ++i)
        {
            if (keep(i))
            {
                move(i, out++);
            }
        }
        return out;
    }
    else
    {
        if (count == 0)
        {
            return 0;
        }
        auto positions = scratchVector<Eigen::Index>(policy, static_cast<std::size_t>(count));
        forEachIndex(policy, count, [&](Eigen::Index i) {
            positions[std::size_t(i)] = keep(i) ? 1 : 0;
        });
        cumulativeSum(policy, std::span<Eigen::Index const>{positions}, std::span<Eigen::Index>{positions});
        forEachIndex(std::forward<P>(policy), count, [&](Eigen::Index i) {
            Eigen::Index const before = i == 0 ? 0 : positions[std::size_t(i - 1)];
            if (positions[std::size_t(i)] != before)
            {
                move(i, before);
            }
        });
        return positions.back();
    }
}

/** Sets every element of \a out to \a value, under \a policy. */
template <ExecutionPolicy P, typename T>
void
fill(P && policy, std::span<T> out, T const & value)
{
    forEachChunk(std::forward<P>(policy), Eigen::Index(out.size()), [&](Eigen::Index begin, Eigen::Index length) {
        std::fill_n(out.begin() + begin, length, value);
    });
}

}  // namespace internal

}  // namespace Emrys

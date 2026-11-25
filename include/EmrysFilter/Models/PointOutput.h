#pragma once

/** \file
 * The output of a function at a point that writes its own Jacobian (decisions.md
 * [returned-jacobians]): writesJacobians(f) marks a function that takes its output last and writes
 * its value and the Jacobians the method wants into it, by block of its input, with the free functions
 * value(), wantsJacobian(), wantsJacobians(), and jacobian(), as a batched function writes its
 * BatchedOutput (Batched.h).
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Models/TransformArgs.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

struct PointOutputAccess;

}  // namespace internal

/** \ingroup Models
 * \brief The output of a function at a point marked writesJacobians() (decisions.md
 * [returned-jacobians]): its value, and the Jacobians the method uses on this call, by block of the
 * function's input, in storage the library provides.
 *
 * The function writes <tt>value(y)</tt> whole, and <tt>jacobian<Name>(y)</tt> for each block
 * \c Name of its input for which <tt>wantsJacobian<Name>(y)</tt> is true: the Jacobian's columns
 * for the block that reads() names \c Name, its rows the output's differences. A block not wanted
 * is not read, so the function need not write it. <tt>jacobian(y)</tt> is the Jacobian over the
 * whole input, which a function that computes it whole writes when <tt>wantsJacobians(y)</tt> says
 * some block is wanted; the method reads the wanted blocks' columns alone.
 *
 * \tparam In       the function's input space, a Composite of reads()' names, or the state's space
 * \tparam Out      its output space
 * \tparam Scalar_  the scalar of the value
 */
template <typename In, typename Out, typename Scalar_>
class PointOutput
{
public:

    /// how many blocks the function's input has
    static constexpr std::size_t BlockCount = std::tuple_size_v<internal::InputBlocks<In>>;

    /** Which blocks' Jacobians are wanted, one flag per block of the input in order. */
    using Wanted = std::array<bool, BlockCount>;

    /** the scalar type */
    using Scalar = Scalar_;

    /** The value, in the output space. */
    using ValueType = typename Moments<Out, Scalar_>::VectorType;

    /** The Jacobian over the input's differences. */
    using JacobianType = internal::MatrixOf<
        Scalar_,
        valueDimensionAtCompileTime<DifferenceSpace<Out>>,
        valueDimensionAtCompileTime<DifferenceSpace<In>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<Out>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<In>>>;

    /** The output of a function from \a in to \a out, with the Jacobian of block \a k wanted where
     * \a wanted says. \a in outlives this.
     */
    PointOutput(In const & in, Out const & out, Wanted const & wanted)
        : m_value(valueDimension(out))
        , m_in{in}
        , m_wanted{wanted}
    {
        bool any = false;
        for (bool const w : wanted)
        {
            any = any || w;
        }
        if (any)
        {
            m_jacobian = JacobianType::Zero(differenceDimension(out), differenceDimension(in));
        }
    }

    /** \returns the flags that want every block's Jacobian, or none */
    [[nodiscard]] static Wanted
    every(bool wanted)
    {
        Wanted result{};
        result.fill(wanted);
        return result;
    }

private:

    friend struct internal::PointOutputAccess;

    ValueType m_value;        ///< the value
    JacobianType m_jacobian;  ///< the Jacobian, sized only when a block is wanted
    In const & m_in;          ///< the function's input space
    Wanted m_wanted;          ///< whether each block's Jacobian is wanted
};

namespace internal
{

/** The members of a PointOutput that the free functions on it and the library reach. */
struct PointOutputAccess
{
    /** \returns the value of \a y */
    template <typename Y>
    [[nodiscard]] static auto &
    value(Y & y)
    {
        return y.m_value;
    }

    /** \returns whether \a y wants the Jacobian of the block at position \a K */
    template <std::size_t K, typename Y>
    [[nodiscard]] static bool
    wanted(Y const & y)
    {
        return y.m_wanted[K];
    }

    /** \returns whether \a y wants the Jacobian of any block */
    template <typename Y>
    [[nodiscard]] static bool
    wantedAny(Y const & y)
    {
        for (bool const w : y.m_wanted)
        {
            if (w)
            {
                return true;
            }
        }
        return false;
    }

    /** \returns the Jacobian of \a y over its whole input */
    template <typename Y>
    [[nodiscard]] static auto &
    jacobian(Y & y)
    {
        return y.m_jacobian;
    }

    /** \returns the input space of \a y */
    template <typename Y>
    [[nodiscard]] static auto const &
    in(Y const & y)
    {
        return y.m_in;
    }
};

/** Whether \a T is a PointOutput. */
template <typename T>
inline constexpr bool isPointOutput = false;

/** A PointOutput is one. */
template <typename In, typename Out, typename Scalar>
inline constexpr bool isPointOutput<PointOutput<In, Out, Scalar>> = true;

/** The output space a function names, where \a F names one, for the mark that wraps it. */
template <typename F>
struct MarkedOutputSpace
{};

/** A function that names its output space. */
template <typename F>
    requires requires { typename F::OutputSpaceType; }
struct MarkedOutputSpace<F>
{
    using OutputSpaceType = typename F::OutputSpaceType;  ///< the function's output space
};

}  // namespace internal

/** \ingroup Models
 * \returns the value of \a y, the output of a function marked writesJacobians(), for the function
 * to write whole
 */
template <typename In, typename Out, typename Scalar>
[[nodiscard]] typename PointOutput<In, Out, Scalar>::ValueType &
value(PointOutput<In, Out, Scalar> & y)
{
    return internal::PointOutputAccess::value(y);
}

/** \ingroup Models
 * \returns whether the method uses the Jacobian of the block named \a Name of the function's input
 * on this call, which the function then writes into \a y
 */
template <typename Name, typename In, typename Out, typename Scalar>
[[nodiscard]] bool
wantsJacobian(PointOutput<In, Out, Scalar> const & y)
{
    static_assert(
        internal::blockPosition<Name, internal::InputBlocks<In>> < std::tuple_size_v<internal::InputBlocks<In>>,
        "wantsJacobian<Name>(y) names a block of the function's input, a name that reads() gives"
    );
    return internal::PointOutputAccess::wanted<internal::blockPosition<Name, internal::InputBlocks<In>>>(y);
}

/** \ingroup Models
 * \returns whether the method uses the Jacobian of any block of the function's input on this call,
 * so that a function that computes its Jacobian whole writes it then with jacobian(y): the method
 * reads the columns of the blocks it wants alone
 */
template <typename In, typename Out, typename Scalar>
[[nodiscard]] bool
wantsJacobians(PointOutput<In, Out, Scalar> const & y)
{
    return internal::PointOutputAccess::wantedAny(y);
}

/** \ingroup Models
 * \returns the Jacobian in \a y of the block named \a Name of the function's input, writable, for a
 * block wantsJacobian<Name>(y) says is wanted: its rows are the output's differences and its columns
 * the block's. Without \a Name, the Jacobian over the whole input, when wantsJacobians(y), some
 * block wanted.
 */
template <typename Name = void, typename In, typename Out, typename Scalar>
[[nodiscard]] auto
jacobian(PointOutput<In, Out, Scalar> & y)
{
    auto & whole = internal::PointOutputAccess::jacobian(y);
    if constexpr (std::is_void_v<Name>)
    {
        EMRYS_FILTER_ASSERT(
            internal::PointOutputAccess::wantedAny(y) && "write the whole Jacobian only when wantsJacobians() says"
        );
        return whole.leftCols(whole.cols());
    }
    else
    {
        [[maybe_unused]] bool const wanted = wantsJacobian<Name>(y);
        EMRYS_FILTER_ASSERT(wanted && "write only the Jacobians wantsJacobian() says are wanted");
        static_assert(
            internal::blockPosition<Name, internal::InputBlocks<In>> < std::tuple_size_v<internal::InputBlocks<In>>,
            "jacobian<Name>(y) names a block of the function's input, a name that reads() gives"
        );
        BlockRows const columns = internal::inputBlockRows<Name>(internal::PointOutputAccess::in(y));
        return whole.middleCols(columns.start, columns.size);
    }
}

namespace internal
{

/** A function at a point marked writesJacobians() (decisions.md [returned-jacobians]): \a F takes
 * its output last, a PointOutput, and writes its value and the Jacobians wanted into it. Called with
 * every Jacobian wanted it returns them with its value, as the library reads a function's Jacobian;
 * valueAt() wants none, for a method that reads the value alone.
 *
 * \tparam F  the function
 */
template <typename F>
class WritesJacobians : public MarkedOutputSpace<F>
{
    static_assert(
        !takesBatches<F>,
        "writesJacobians() marks a function at a point; a batched function's mark is batched(f, jacobians::...)"
    );

public:

    /// a function marked to write its Jacobians
    static constexpr bool WritesPointJacobians = true;

    /** \a function marked. */
    explicit WritesJacobians(F function)
        : m_function{std::move(function)}
    {}

    /** \returns the function marked */
    [[nodiscard]] F const &
    function() const
    {
        return m_function;
    }

    /** \returns the function's value at \a x, a vector in the input space \a in, in the output space
     * \a out, with its Jacobian over the input's differences; \a rest is a process model's step
     */
    template <typename In, typename Out, typename X, typename... Rest>
    [[nodiscard]] auto
    operator()(In const & in, Out const & out, X const & x, Rest &&... rest) const
        requires MatrixExpression<X>
    {
        checkOutput<Out>();
        auto every = std::array<bool, std::tuple_size_v<internal::InputBlocks<In>>>{};
        every.fill(true);
        return withWanted(every, in, out, x, std::forward<Rest>(rest)...);
    }

    /** \returns the function's value at \a x with the Jacobian of the blocks \a wanted says, zero in
     * the columns of the others: what a method that reads some blocks' columns alone asks for
     */
    template <typename Wanted, typename In, typename Out, typename X, typename... Rest>
    [[nodiscard]] auto
    withWanted(Wanted const & wanted, In const & in, Out const & out, X const & x, Rest &&... rest) const
    {
        checkOutput<Out>();
        auto output = PointOutput<In, Out, typename X::Scalar>{in, out, wanted};
        callForm(m_function, in, out, x, std::forward<Rest>(rest)..., output);
        return withJacobian(
            std::move(PointOutputAccess::value(output)),
            std::move(PointOutputAccess::jacobian(output))
        );
    }

    /** \returns the function's value alone at \a x, with no Jacobian wanted */
    template <typename In, typename Out, typename X, typename... Rest>
    [[nodiscard]] auto
    valueAt(In const & in, Out const & out, X const & x, Rest &&... rest) const
    {
        checkOutput<Out>();
        using Output = PointOutput<In, Out, typename X::Scalar>;
        auto output = Output{in, out, Output::every(false)};
        callForm(m_function, in, out, x, std::forward<Rest>(rest)..., output);
        return std::move(PointOutputAccess::value(output));
    }

private:

    /** A compile-time error unless  Out, the output space a call names, is a Space. */
    template <typename Out>
    static consteval void
    checkOutput()
    {
        static_assert(
            Space<Out>,
            "a function marked writesJacobians() writes into an output whose space the model names before the call: "
            "give it outputSpace(), writes(), or additive noise over the output"
        );
    }

    F m_function;  ///< the function marked
};

/** \returns \a f called at \a x with the Jacobian of the blocks \a wanted says: a function marked
 * writesJacobians() asked for those alone, and any other called as callForm() calls it
 */
template <typename F, typename Wanted, typename In, typename Out, typename X, typename... Rest>
[[nodiscard]] auto
callWantedForm(F const & f, Wanted const & wanted, In const & in, Out const & out, X const & x, Rest &&... rest)
{
    if constexpr (requires { f.withWanted(wanted, in, out, x, std::forward<Rest>(rest)...); })
    {
        return f.withWanted(wanted, in, out, x, std::forward<Rest>(rest)...);
    }
    else
    {
        return callForm(f, in, out, x, std::forward<Rest>(rest)...);
    }
}

/** \returns the value alone of \a f called at \a x in the form it takes: a function marked
 * writesJacobians() with no Jacobian wanted, and any other called as callForm() calls it, its
 * returned Jacobian dropped
 */
template <typename F, typename In, typename Out, typename X, typename... Rest>
[[nodiscard]] auto
callValueForm(F const & f, In const & in, Out const & out, X const & x, Rest &&... rest)
{
    if constexpr (requires { f.valueAt(in, out, x, std::forward<Rest>(rest)...); })
    {
        return f.valueAt(in, out, x, std::forward<Rest>(rest)...);
    }
    else
    {
        // The value is moved out of what the function returns, which dies here.
        auto returned = callForm(f, in, out, x, std::forward<Rest>(rest)...);
        if constexpr (isWithJacobian<std::remove_cvref_t<decltype(returned)>>)
        {
            return std::move(returned.value);
        }
        else
        {
            return returned;
        }
    }
}

}  // namespace internal

/** \ingroup Models
 * \returns \a function marked as a function at a point that writes its own Jacobian (decisions.md
 * [returned-jacobians]): it takes its output \c y last, after a process model's step,
 * <tt>h(in, out, x, y)</tt> or <tt>f(in, out, x, dt, y)</tt> and the shorter forms, and writes
 * <tt>value(y)</tt> and, for each block the method wants, <tt>jacobian<Name>(y)</tt>, so that it is
 * never differentiated and the Jacobian is computed only where a method reads it.
 *
 * \code
 * auto const h = measurementModel(
 *     writesJacobians([](auto const & in, auto const &, auto const & x, auto & y) {
 *       auto const p = block<Position>(in, x);
 *       value(y) = Eigen::Matrix<double, 1, 1>{p.norm()};
 *       if (wantsJacobian<Position>(y)) { jacobian<Position>(y) = p.transpose() / p.norm(); }
 *     }),
 *     reads(path<Position>), additiveNoise(r));
 * \endcode
 */
template <typename F>
    requires(!MatrixExpression<F>)
[[nodiscard]] internal::WritesJacobians<std::remove_cvref_t<F>>
writesJacobians(F && function)
{
    return internal::WritesJacobians<std::remove_cvref_t<F>>{std::forward<F>(function)};
}

}  // namespace Emrys

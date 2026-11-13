#pragma once

/** \file
 * The arguments of transform() and the joint input they form (decisions.md [prediction-inputs],
 * [affine-coordinates], [linearization-reference]): the types of the withCross() and linearizedAbout()
 * marks, CrossMarked and LinearizedAbout, and, in the internal
 * namespace, how each argument reaches the function, as uncertain or known, and the Transform that
 * concatenates the uncertain ones into one input, with its covariance and root. The functions that make
 * the marks are in Transform.h. Read Transform.h next.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/AutoDiff>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Models/Methods.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** Whether \a T is a WithJacobian: what a function returns when it knows its Jacobian. */
template <typename T>
inline constexpr bool isWithJacobian = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Value, typename Jacobian>
inline constexpr bool isWithJacobian<WithJacobian<Value, Jacobian>>{true};

/// \endcond

/** \returns the value in what a function returned, \a returned: itself, or the value beside its
 * Jacobian. Valid as long as \a returned is.
 */
template <typename Returned>
[[nodiscard]] decltype(auto)
valueOf(Returned && returned)
{
    if constexpr (isWithJacobian<std::remove_cvref_t<Returned>>)
    {
        return (std::forward<Returned>(returned).value);
    }
    else
    {
        return std::forward<Returned>(returned);
    }
}

/** The Jacobian an internal wrapper of a function returns beside its value: \a Blocks is a
 * \c std::tuple with one block per argument of the wrapper, known arguments included, so that
 * the wrapper need not know which of its arguments are uncertain. Transform takes the uncertain
 * ones' blocks and ignores the rest.
 */
template <typename Blocks>
struct ArgumentJacobians
{
    Blocks blocks;  ///< one block per argument
};

/** Whether \a T is an ArgumentJacobians. */
template <typename T>
inline constexpr bool isArgumentJacobians = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Blocks>
inline constexpr bool isArgumentJacobians<ArgumentJacobians<Blocks>>{true};

/// \endcond

/** \returns block \a K of the Jacobian \a jacobian a function returned with one block per
 * uncertain input: the tuple's element, or the one matrix if it is not a tuple
 */
template <std::size_t K, typename Jacobian>
[[nodiscard]] auto const &
jacobianBlock(Jacobian const & jacobian)
{
    if constexpr (requires { std::tuple_size<Jacobian>::value; })
    {
        return std::get<K>(jacobian);
    }
    else
    {
        static_assert(K == 0, "a function with more than one uncertain input must return a tuple of Jacobian blocks");
        return jacobian;
    }
}

/** A sigma-point method \a Points with the coordinates of the joint input that a model's affineIn()
 * names factored out (decisions.md [affine-coordinates]): the points spread over the rest, and the
 * coordinates \a affine, of linear blocks, carried exactly by their %Gaussian given the rest, with the
 * function's Jacobian in them at each point.
 */
template <int AffineSize, int AffineMaxSize, typename Points>
struct FactoredAt
{
    Points points;                                  ///< the sigma points over the rest
    IndexVector<AffineSize, AffineMaxSize> affine;  ///< coordinates of the joint input, in column order
};

/** Whether \a T is a FactoredAt. */
template <typename T>
inline constexpr bool isFactoredAt = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <int AffineSize, int AffineMaxSize, typename Points>
inline constexpr bool isFactoredAt<FactoredAt<AffineSize, AffineMaxSize, Points>>{true};
/// \endcond

/** Sum of \a Sizes, or \c Eigen::Dynamic if any is. */
template <int... Sizes>
inline constexpr int sizeSum = (false || ... || (Sizes == Eigen::Dynamic)) ? Eigen::Dynamic : (0 + ... + Sizes);

/** Product of \a A and \a B, or \c Eigen::Dynamic if either is. */
template <int A, int B>
inline constexpr int sizeProduct = A == Eigen::Dynamic || B == Eigen::Dynamic ? Eigen::Dynamic : A * B;

}  // namespace internal

/** \ingroup Transforms
 * An uncertain input marked by withCross(). Valid until the end of the full expression, so it is
 * passed to transform() as it is made, never kept.
 */
template <typename Input>
struct CrossMarked
{
    Input const & input;  ///< the marked input
};

/** \ingroup Transforms
 * \brief Where a step linearizes its input: a point, or a distribution, in place of the input's own
 * mean and spread (decisions.md [linearization-reference]). Made by linearizedAbout().
 *
 * \tparam Reference  a plain vector, held by value, for a point; or a reference to an UncertainInput
 *                    for a distribution, valid until the end of the full expression
 */
template <typename Reference>
struct LinearizedAbout
{
    /** The point's type, or a reference to the distribution. */
    using ReferenceType = Reference;

    /** Whether the reference is a distribution, which every method linearizes over, rather than a
     * point, which only a linear method (Linearized) linearizes at.
     */
    static constexpr bool IsDistribution = UncertainInput<std::remove_cvref_t<Reference>>;

    Reference reference;  ///< the point, or the distribution

    /** \returns the point, or the distribution's mean */
    [[nodiscard]] decltype(auto)
    mean() const
    {
        if constexpr (IsDistribution)
        {
            return reference.mean();
        }
        else
        {
            return (reference);
        }
    }
};

namespace internal
{

/** Whether \a T is a LinearizedAbout. */
template <typename T>
inline constexpr bool isLinearizedAbout = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Reference>
inline constexpr bool isLinearizedAbout<LinearizedAbout<Reference>>{true};

/// \endcond

/** Whether the first of \a Args is a LinearizedAbout: the reference a transform or a prediction is
 * given before its inputs.
 */
template <typename... Args>
inline constexpr bool leadsWithReference = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename First, typename... Rest>
inline constexpr bool leadsWithReference<First, Rest...> = isLinearizedAbout<First>;

/// \endcond

/** What a step is given where no linearizedAbout() reference is: its input is linearized at its
 * own mean, as usual.
 */
struct NoReference
{};

/** The Space concept as a type trait usable in a requires clause on a template parameter pack. */
template <typename T>
concept IsSpace = Space<T>;

/** How transform() treats an argument: known by default. */
template <typename Arg>
struct ArgTraits
{
    /** The input the argument carries, with any mark removed. */
    using Input = Arg;

    /** The input's scalar type, or void for a known argument. */
    using Scalar = void;

    static constexpr bool isUncertain = false;   ///< whether the argument is a distribution
    static constexpr bool hasCross = false;      ///< whether withCross() marks it
    static constexpr int size = 0;               ///< the input's value dimension, or 0 if known
    static constexpr int maxSize = 0;            ///< a bound on it
    static constexpr int differenceSize = 0;     ///< the input's difference dimension, or 0 if known
    static constexpr int differenceMaxSize = 0;  ///< a bound on it
};

/** An uncertain input: its dimensions come from the distribution. */
template <UncertainInput Arg>
struct ArgTraits<Arg>
{
    /** The input itself. */
    using Input = Arg;

    /** Its scalar type. */
    using Scalar = typename Arg::Scalar;

    static constexpr bool isUncertain = true;                                           ///< a distribution
    static constexpr bool hasCross = false;                                             ///< not marked by withCross()
    static constexpr int size = Arg::ValueDimensionAtCompileTime;                       ///< its value dimension
    static constexpr int maxSize = Arg::MaxValueDimensionAtCompileTime;                 ///< a bound on it
    static constexpr int differenceSize = Arg::DifferenceDimensionAtCompileTime;        ///< its difference dimension
    static constexpr int differenceMaxSize = Arg::MaxDifferenceDimensionAtCompileTime;  ///< a bound on it
};

/** An input marked by withCross(): the input's traits, with the mark. */
template <typename In>
struct ArgTraits<CrossMarked<In>> : ArgTraits<In>
{
    static constexpr bool hasCross = true;  ///< marked by withCross()
};

/** Number of arguments marked by withCross(). */
template <typename... Args>
inline constexpr std::size_t crossCount = (std::size_t{ArgTraits<Args>::hasCross} + ... + 0);

/** \returns the input an argument carries: itself, or the input inside withCross(). */
template <typename Arg>
[[nodiscard]] auto const &
argInput(Arg const & arg)
{
    if constexpr (ArgTraits<Arg>::hasCross)
    {
        return arg.input;
    }
    else
    {
        return arg;
    }
}

/** The first of the types that is not \c void: the uncertain inputs' scalar among the arguments'. */
template <typename First, typename... Rest>
struct FirstNonVoidOf
{
    /** The first non-void type. */
    using type = std::conditional_t<std::is_void_v<First>, typename FirstNonVoidOf<Rest...>::type, First>;
};

/** The last type, whatever it is. */
template <typename First>
struct FirstNonVoidOf<First>
{
    /** The type. */
    using type = First;
};

/** The first of \a T that is not void. */
template <typename... T>
using FirstNonVoid = typename FirstNonVoidOf<T...>::type;

/** Compile-time facts about the arguments of transform(). */
template <typename... Args>
struct TransformArgs
{
    static constexpr std::size_t count = sizeof...(Args);                                 ///< the number of arguments
    static constexpr std::array<bool, count> uncertain{ArgTraits<Args>::isUncertain...};  ///< which are uncertain

    /// how many are uncertain
    static constexpr std::size_t uncertainCount = (std::size_t{ArgTraits<Args>::isUncertain} + ... + 0);

    static_assert(uncertainCount > 0, "transform() needs at least one uncertain input");
    static_assert(crossCount<Args...> <= 1, "withCross() marks at most one input");

    /** The uncertain inputs' common scalar type. */
    using Scalar = FirstNonVoid<typename ArgTraits<Args>::Scalar...>;

    static_assert(
        ((std::is_void_v<typename ArgTraits<Args>::Scalar> || std::is_same_v<typename ArgTraits<Args>::Scalar, Scalar>)
         && ...),
        "every uncertain input must have the same scalar type"
    );

    /** Value dimension of the joint uncertain input, or \c Eigen::Dynamic. */
    static constexpr int size = sizeSum<ArgTraits<Args>::size...>;

    static constexpr int maxSize = sizeSum<ArgTraits<Args>::maxSize...>;  ///< a bound on size

    /** Difference dimension of the joint uncertain input, or \c Eigen::Dynamic. */
    static constexpr int differenceSize = sizeSum<ArgTraits<Args>::differenceSize...>;

    /** A bound on differenceSize. */
    static constexpr int differenceMaxSize = sizeSum<ArgTraits<Args>::differenceMaxSize...>;

    /** Position among the uncertain inputs of each argument (meaningful for uncertain ones). */
    static constexpr std::array<std::size_t, count> ordinal{[] {
        auto result = std::array<std::size_t, count>{};
        std::size_t next = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            result[i] = uncertain[i] ? next++ : 0;
        }
        return result;
    }()};

    /** Argument position of the input marked by withCross(), or \c count if none is. */
    static constexpr std::size_t crossPosition = [] {
        constexpr auto marked = std::array<bool, count>{ArgTraits<Args>::hasCross...};
        for (std::size_t i = 0; i < count; ++i)
        {
            if (marked[i])
            {
                return i;
            }
        }
        return count;
    }();
};

/** A function of the arguments of transform(), evaluated at a point of the joint uncertain input.
 *
 * The joint input is the concatenation of the uncertain inputs. Its values, which \a f receives,
 * and its differences, over which the covariance, the Jacobian, and the sigma-point spread are
 * taken, have their own offsets, which differ where an input's differences have other sizes than
 * its values.
 */
template <typename F, typename OutputSpace, typename... Args>
class Transform
{
public:

    /** The compile-time facts about the arguments. */
    using ArgInfo = TransformArgs<Args...>;

    /** The scalar type. */
    using Scalar = typename ArgInfo::Scalar;

    /** Eigen's index type. */
    using Index = Eigen::Index;

    /** The output's space. */
    using OutputSpaceType = OutputSpace;

    static constexpr int InputSize = ArgInfo::size;                      ///< the joint input's value dimension
    static constexpr int InputMaxSize = ArgInfo::maxSize;                ///< a bound on it
    static constexpr int InputDifferenceSize = ArgInfo::differenceSize;  ///< the joint input's difference dimension
    static constexpr int InputDifferenceMaxSize = ArgInfo::differenceMaxSize;          ///< a bound on it
    static constexpr int OutputSize = valueDimensionAtCompileTime<OutputSpace>;        ///< the output's value dimension
    static constexpr int OutputMaxSize = maxValueDimensionAtCompileTime<OutputSpace>;  ///< a bound on it

    /** The output's difference dimension. */
    static constexpr int OutputDifferenceSize = valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>;

    static constexpr int OutputDifferenceMaxSize =
        maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>;  ///< a bound on it

    /** A joint input value with scalar \a T. */
    template <typename T>
    using InputVector = Eigen::Matrix<T, InputSize, 1, Eigen::ColMajor, InputMaxSize, 1>;

    /** A joint input difference with scalar \a T. */
    template <typename T>
    using InputDifferenceVector = Eigen::Matrix<T, InputDifferenceSize, 1, Eigen::ColMajor, InputDifferenceMaxSize, 1>;

    /** An output value with scalar \a T. */
    template <typename T>
    using OutputVector = Eigen::Matrix<T, OutputSize, 1, Eigen::ColMajor, OutputMaxSize, 1>;

    /** An output difference with scalar \a T. */
    template <typename T>
    using OutputDifferenceVector =
        Eigen::Matrix<T, OutputDifferenceSize, 1, Eigen::ColMajor, OutputDifferenceMaxSize, 1>;

    /** The joint input covariance, over its differences. */
    using InputMatrix = Eigen::Matrix<
        Scalar,
        InputDifferenceSize,
        InputDifferenceSize,
        Eigen::ColMajor,
        InputDifferenceMaxSize,
        InputDifferenceMaxSize>;

    /** The output covariance, over its differences. */
    using OutputMatrix = Eigen::Matrix<
        Scalar,
        OutputDifferenceSize,
        OutputDifferenceSize,
        Eigen::ColMajor,
        OutputDifferenceMaxSize,
        OutputDifferenceMaxSize>;

    /** The Jacobian of the output differences in the joint input differences. */
    using JacobianMatrix =
        MatrixOf<Scalar, OutputDifferenceSize, InputDifferenceSize, OutputDifferenceMaxSize, InputDifferenceMaxSize>;

    /** The cross-covariance of the joint input with the output. */
    using CrossMatrix =
        MatrixOf<Scalar, InputDifferenceSize, OutputDifferenceSize, InputDifferenceMaxSize, OutputDifferenceMaxSize>;

    /** Output values at sigma points, one per column: \a Points of them, at most \a MaxPoints, so
     * a bounded space's need no heap.
     */
    template <int Points = InputDifferenceSize, int MaxPoints = InputDifferenceMaxSize>
    using OutputSamples = MatrixOf<Scalar, OutputSize, Points, OutputMaxSize, MaxPoints>;

    /** Output differences at sigma points, one per column, as OutputSamples. */
    template <int Points = InputDifferenceSize, int MaxPoints = InputDifferenceMaxSize>
    using OutputDeviations = MatrixOf<Scalar, OutputDifferenceSize, Points, OutputDifferenceMaxSize, MaxPoints>;

    /** What a sigma-point method computes over the joint input, the root form (decisions.md [linearization-forms]):
     * the output mean, the block-diagonal root \f$ S \f$ of the joint covariance that spread the
     * points, the whitened cross-covariance \f$ \Psi \f$ with a row per column of \f$ S \f$, and
     * \f$ \Omega \f$, the covariance of the fit's residuals at the points. The output covariance is
     * \f$ \Psi^T \Psi + \Omega \f$, and the cross-covariance \f$ S \Psi \f$.
     */
    struct RootResult
    {
        OutputVector<Scalar> mean;  ///< the output mean, normalized in the output space
        InputMatrix root;           ///< \f$ S \f$, block-diagonal by input
        CrossMatrix whitened;       ///< \f$ \Psi \f$: rows in joint input order, by difference coordinate
        OutputMatrix unexplained;   ///< \f$ \Omega \f$, symmetric
    };

    /** Lays out the joint input of \a args for \a f into \a outputSpace, which is resized to
     * \a f's output if it has one resizable block. One with several must already have the size
     * \a f returns. The arguments must outlive this.
     */
    Transform(F & f, OutputSpace const & outputSpace, Args const &... args)
        : m_f{f}
        , m_outputSpace{outputSpace}
        , m_args{args...}
    {
        forEachUncertain([this]<std::size_t I>(auto const & input) {
            m_offsets[I] = m_inputSize;
            m_differenceOffsets[I] = m_inputDifferenceSize;
            m_inputSize += input.valueDimension();
            m_inputDifferenceSize += input.differenceDimension();
        });
        // A resizable output space with one resizable block takes the size f returns.
        if constexpr (OutputSize == Eigen::Dynamic && requires(Index n) { m_outputSpace.resize(n); })
        {
            Index const outputSize = OutputVector<Scalar>{valueOf(call<Scalar>(m_f, mean()))}.size();
            if (valueDimension(m_outputSpace) != outputSize)
            {
                m_outputSpace.resize(outputSize);
            }
        }
        else if constexpr (OutputSize == Eigen::Dynamic)
        {
            // Several resizable blocks: f's output can't say how to split, so the space must fit it.
            EMRYS_FILTER_ASSERT(
                OutputVector<Scalar>{valueOf(call<Scalar>(m_f, mean()))}.size() == valueDimension(m_outputSpace)
                && "the function's output does not fit the output space"
            );
        }
    }

    /** \returns the output's space, with its sizes */
    [[nodiscard]] OutputSpace const &
    outputSpace() const
    {
        return m_outputSpace;
    }

    /** \returns the value dimension of the joint uncertain input */
    [[nodiscard]] Index
    inputSize() const
    {
        return m_inputSize;
    }

    /** \returns the difference dimension of the joint uncertain input */
    [[nodiscard]] Index
    inputDifferenceSize() const
    {
        return m_inputDifferenceSize;
    }

    /** \returns the difference dimension of the output */
    [[nodiscard]] Index
    outputDifferenceSize() const
    {
        return differenceDimension(m_outputSpace);
    }

    /** \returns the concatenated means of the uncertain inputs */
    [[nodiscard]] InputVector<Scalar>
    mean() const
    {
        InputVector<Scalar> result = InputVector<Scalar>::Zero(m_inputSize);
        forEachUncertain([&]<std::size_t I>(auto const & input) {
            result.segment(m_offsets[I], input.valueDimension()) = input.mean();
        });
        return result;
    }

    /** \returns the block-diagonal joint covariance of the independent uncertain inputs */
    [[nodiscard]] InputMatrix
    covarianceLower() const
    {
        InputMatrix result = InputMatrix::Zero(m_inputDifferenceSize, m_inputDifferenceSize);
        forEachUncertain([&]<std::size_t I>(auto const & input) {
            Index const d = input.differenceDimension();
            result.block(m_differenceOffsets[I], m_differenceOffsets[I], d, d) = symmetricCov(input);
        });
        return result;
    }

    /** \returns a square root \f$ S \f$ of covarianceLower(), \f$ S S^T = \Sigma \f$, block by block
     * (decisions.md [sigma-point-axes]): the input marked by withCross(), or the first uncertain input when none is,
     * by \a axes, and every other by roots::DefaultAxes. An input that holds a root of its own (a
     * \c root() member, as the square-root forms' read blocks do, decisions.md [gaussian-forms]) gives
     * it to its axes as a factor, so that every form spreads the same points; any other gives its
     * covariance, of which the axes read the lower triangle alone.
     *
     * A transform returns no \c std::expected, so \a axes that can fail are refused.
     */
    template <typename A = roots::DefaultAxes>
    [[nodiscard]] InputMatrix
    root(A const & axes = {}) const
    {
        internal::refuseFailingAxes<A>();
        InputMatrix result = InputMatrix::Zero(m_inputDifferenceSize, m_inputDifferenceSize);
        forEachUncertain([&]<std::size_t I>(auto const & input) {
            Index const d = input.differenceDimension();
            auto const take = [&](auto const & entry) {
                if constexpr (requires { input.root(); })
                {
                    return internal::rootValue(internal::rootFromFactor(entry, input.root()));
                }
                else
                {
                    return internal::rootValue(entry.root(covarianceLowerOf(input)));
                }
            };
            constexpr bool spread =
                ArgInfo::crossPosition < ArgInfo::count ? I == ArgInfo::crossPosition : ArgInfo::ordinal[I] == 0;
            if constexpr (spread)
            {
                result.block(m_differenceOffsets[I], m_differenceOffsets[I], d, d) = take(axes);
            }
            else
            {
                result.block(m_differenceOffsets[I], m_differenceOffsets[I], d, d) = take(roots::DefaultAxes{});
            }
        });
        return result;
    }

    /** \returns the joint uncertain input \a x moved by the joint difference \a d: each input's
     * block by plus() in that input's space.
     */
    template <typename X, typename D>
    [[nodiscard]] InputVector<typename X::Scalar>
    plusInputs(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d) const
    {
        InputVector<typename X::Scalar> result = x;
        forEachUncertain([&]<std::size_t I>(auto const & input) {
            using InputSpace = std::remove_cvref_t<decltype(input.space())>;
            constexpr int Values = internal::rowsAtCompileTime<false, InputSpace>;
            constexpr int Differences = internal::rowsAtCompileTime<true, InputSpace>;
            internal::middleRowsOf<Values>(result, m_offsets[I], input.valueDimension()) = Emrys::plus(
                input.space(),
                internal::middleRowsOf<Values>(x, m_offsets[I], input.valueDimension()),
                internal::middleRowsOf<Differences>(d, m_differenceOffsets[I], input.differenceDimension())
            );
        });
        return result;
    }

    /** \returns the joint uncertain input \a x moved by the joint difference \a d without
     * normalizing: <tt>x + d</tt> in each input, except that an input whose space has its own
     * plus() moves by it.
     */
    template <typename X, typename D>
    [[nodiscard]] InputVector<typename X::Scalar>
    stepInputs(Eigen::MatrixBase<X> const & x, Eigen::MatrixBase<D> const & d) const
    {
        InputVector<typename X::Scalar> result = x;
        forEachUncertain([&]<std::size_t I>(auto const & input) {
            using InputSpace = std::remove_cvref_t<decltype(input.space())>;
            constexpr int Values = internal::rowsAtCompileTime<false, InputSpace>;
            constexpr int Differences = internal::rowsAtCompileTime<true, InputSpace>;
            auto values = internal::middleRowsOf<Values>(result, m_offsets[I], input.valueDimension());
            auto const step =
                internal::middleRowsOf<Differences>(d, m_differenceOffsets[I], input.differenceDimension());
            if constexpr (HasPlus<InputSpace>)
            {
                values = Emrys::plus(
                    input.space(),
                    internal::middleRowsOf<Values>(x, m_offsets[I], input.valueDimension()),
                    step
                );
            }
            else
            {
                values += step;
            }
        });
        return result;
    }

    /** \returns \a f at the joint uncertain input \a x, with scalar type \a T: its value, without
     * any Jacobian it returns
     */
    template <typename T>
    [[nodiscard]] OutputVector<T>
    evaluate(InputVector<T> const & x) const
    {
        OutputVector<T> y = valueOf(call<T>(m_f, x));
        EMRYS_FILTER_ASSERT(y.size() == valueDimension(m_outputSpace) && "the output does not fit the output space");
        return y;
    }

    /// whether \a f evaluates a batch of points in one call, as the function objects the resolvers
    /// give for a batched function do (decisions.md [batched-functions])
    static constexpr bool EvaluatesBatches = requires { requires std::remove_cvref_t<F>::EvaluatesBatches; };

    /** Joint input values with scalar \a T, one per column: \a Points of them, at most \a MaxPoints. */
    template <typename T, int Points, int MaxPoints>
    using InputSamples = MatrixOf<T, InputSize, Points, InputMaxSize, MaxPoints>;

    /** Writes \a f's values at \a count points of the joint uncertain input into the columns of
     * \a outputs, which it sizes, in the scalar of \a outputs: point \a j is <tt>pointAt(j)</tt>, an
     * InputVector in that scalar. Where \a f evaluates batches, the points are gathered first and \a f
     * is called once over them all; otherwise it is called at each point in order (decisions.md
     * [batched-functions]).
     */
    template <typename Outputs, typename PointAt>
    void
    evaluateEach(Index count, PointAt const & pointAt, Outputs & outputs) const
    {
        using T = typename Outputs::Scalar;
        outputs.resize(valueDimension(m_outputSpace), count);
        if constexpr (EvaluatesBatches)
        {
            auto x = InputSamples<T, Outputs::ColsAtCompileTime, Outputs::MaxColsAtCompileTime>{};
            x.resize(m_inputSize, count);
            for (Index j = 0; j < count; ++j)
            {
                x.col(j) = pointAt(j);
            }
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                m_f.evaluateBatch(outputs, argumentColumns<I>(x)...);
            }(std::index_sequence_for<Args...>{});
        }
        else
        {
            for (Index j = 0; j < count; ++j)
            {
                outputs.col(j) = evaluate<T>(pointAt(j));
            }
        }
    }

    /** evaluateEach() of a function that evaluates batches, with its Jacobian in the coordinates of the
     * joint input that the factored transform takes as affine written into \a jacobians, columns
     * <tt>j l</tt> to <tt>j l + l - 1</tt> for point \a j, with \f$ l \f$ the number of them, in one
     * call (decisions.md [affine-coordinates], [batched-functions]). The function's object knows those
     * coordinates, since its resolver gave them.
     */
    template <typename Outputs, typename Jacobians, typename PointAt>
    void
    evaluateEachWithAffineJacobians(
        Index count,
        PointAt const & pointAt,
        Outputs & outputs,
        Jacobians & jacobians
    ) const
        requires EvaluatesBatches
    {
        outputs.resize(valueDimension(m_outputSpace), count);
        auto x = InputSamples<Scalar, Outputs::ColsAtCompileTime, Outputs::MaxColsAtCompileTime>{};
        x.resize(m_inputSize, count);
        for (Index j = 0; j < count; ++j)
        {
            x.col(j) = pointAt(j);
        }
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            m_f.evaluateBatchWithAffineJacobians(outputs, jacobians, argumentColumns<I>(x)...);
        }(std::index_sequence_for<Args...>{});
    }

    /** \returns what \a f returns at the joint uncertain input \a x, with scalar type \a T: its
     * value, or a WithJacobian
     */
    template <typename T>
    [[nodiscard]] decltype(auto)
    returned(InputVector<T> const & x) const
    {
        return call<T>(m_f, x);
    }

    /** \returns \a g called with the same arguments as \a f, at the joint uncertain input \a x */
    template <typename T, typename G>
    [[nodiscard]] decltype(auto)
    call(G & g, InputVector<T> const & x) const
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
            return callWithScalar<T>(g, argument<T, I>(x)...);
        }(std::index_sequence_for<Args...>{});
    }

    /** Calls <tt>fn.template operator()\<I\>(input)</tt> for each uncertain argument \a I, in order. */
    template <typename Fn>
    void
    forEachUncertain(Fn && fn) const
    {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (visitIfUncertain<I>(fn), ...);
        }(std::index_sequence_for<Args...>{});
    }

    /** \returns the offset in the joint input's values of uncertain argument \a i */
    [[nodiscard]] Index
    offset(std::size_t i) const
    {
        return m_offsets[i];
    }

    /** \returns the offset in the joint input's differences of uncertain argument \a i */
    [[nodiscard]] Index
    differenceOffset(std::size_t i) const
    {
        return m_differenceOffsets[i];
    }

    /** \returns the distribution of the output from \a result, what the method computed over the
     * joint input: a LinearizationAt for a linear method, or a RootResult. With no input marked by
     * withCross(), a %Gaussian in the output space, with covariance \f$ J \Sigma J^T \f$ or
     * \f$ \Psi^T \Psi + \Omega \f$. With one marked, an OutputWithCross in the form the method
     * computes, its parts split by input (decisions.md [linearization-forms]): the marked input's \f$ H \f$ and
     * \f$ P H^T \f$, or its block of \f$ S \f$ and rows of \f$ \Psi \f$; \f$ H P H^T \f$ from them;
     * and \f$ N \f$, the sum over every other input \f$ k \f$ of \f$ J_k C_k J_k^T \f$ or
     * \f$ \Psi_k^T \Psi_k \f$.
     */
    template <typename MethodResult>
    [[nodiscard]] auto
    assemble(MethodResult const & result) const
    {
        constexpr bool Root = std::is_same_v<MethodResult, RootResult>;
        Index const m = outputDifferenceSize();
        if constexpr (ArgInfo::crossPosition == ArgInfo::count)
        {
            OutputMatrix cov = [&] {
                if constexpr (Root)
                {
                    OutputMatrix parts = result.unexplained;
                    parts.noalias() += multiply(result.whitened.transpose(), result.whitened);
                    return parts;
                }
                else
                {
                    CrossMatrix const cross = multiply(covarianceLower(), result.jacobian.transpose());
                    return OutputMatrix{multiply(result.jacobian, cross)};
                }
            }();
            cov = (Scalar(0.5) * (cov + cov.transpose())).eval();
            return GaussianParams<OutputSpace, Scalar>{m_outputSpace, result.mean, std::move(cov)};
        }
        else
        {
            constexpr std::size_t position = ArgInfo::crossPosition;
            using InputSpace = typename ArgTraits<ArgAt<position>>::Input::SpaceType;
            using Form = std::conditional_t<Root, RootForm, JacobianForm>;
            using WithCrossResult = OutputWithCross<InputSpace, OutputSpace, Form, Scalar>;
            Index const offset = m_differenceOffsets[position];
            Index const rows = argInput(std::get<position>(m_args)).differenceDimension();
            auto const symmetric = [](OutputMatrix & a) {
                a = (Scalar(0.5) * (a + a.transpose())).eval();
            };
            // N: each other input's share, from its own columns of J or rows of Psi.
            OutputMatrix inputs = OutputMatrix::Zero(m, m);
            forEachUncertain([&]<std::size_t I>(auto const & input) {
                if constexpr (I != position)
                {
                    using Input = std::remove_cvref_t<decltype(input)>;
                    using Shared = MatrixOf<
                        Scalar,
                        Input::DifferenceDimensionAtCompileTime,
                        OutputDifferenceSize,
                        Input::MaxDifferenceDimensionAtCompileTime,
                        OutputDifferenceMaxSize>;
                    Index const at = m_differenceOffsets[I];
                    Index const d = input.differenceDimension();
                    if constexpr (Root)
                    {
                        Shared const block = result.whitened.middleRows(at, d);
                        inputs.noalias() += multiply(block.transpose(), block);
                    }
                    else
                    {
                        Shared const shared =
                            multiply(symmetricCov(input), result.jacobian.middleCols(at, d).transpose());
                        inputs.noalias() += multiply(result.jacobian.middleCols(at, d), shared);
                    }
                }
            });
            symmetric(inputs);
            if constexpr (Root)
            {
                typename WithCrossResult::CrossMatrix whitened = result.whitened.middleRows(offset, rows);
                OutputMatrix explained = multiply(whitened.transpose(), whitened);
                symmetric(explained);
                return WithCrossResult{
                    m_outputSpace,
                    result.mean,
                    typename WithCrossResult::RootMatrix{result.root.block(offset, offset, rows, rows)},
                    std::move(whitened),
                    std::move(explained),
                    std::move(inputs),
                    result.unexplained
                };
            }
            else
            {
                typename WithCrossResult::JacobianMatrix jacobian = result.jacobian.middleCols(offset, rows);
                typename WithCrossResult::CrossMatrix cross =
                    multiply(symmetricCov(argInput(std::get<position>(m_args))), jacobian.transpose());
                OutputMatrix explained = multiply(jacobian, cross);
                symmetric(explained);
                return WithCrossResult{
                    m_outputSpace,
                    result.mean,
                    std::move(jacobian),
                    std::move(cross),
                    std::move(explained),
                    std::move(inputs),
                    OutputMatrix::Zero(m, m)
                };
            }
        }
    }

private:

    /** Argument \a I's type. */
    template <std::size_t I>
    using ArgAt = std::tuple_element_t<I, std::tuple<Args...>>;

    /** \returns \a input's covariance made symmetric from its lower triangle */
    template <typename Input>
    [[nodiscard]] static auto
    symmetricCov(Input const & input)
    {
        using Block = Eigen::Matrix<
            Scalar,
            Input::DifferenceDimensionAtCompileTime,
            Input::DifferenceDimensionAtCompileTime,
            Eigen::ColMajor,
            Input::MaxDifferenceDimensionAtCompileTime,
            Input::MaxDifferenceDimensionAtCompileTime>;
        Block cov = covarianceLowerOf(input);
        cov.template triangularView<Eigen::StrictlyUpper>() = cov.transpose();
        return cov;
    }

    /** Calls <tt>fn.template operator()\<I\>(input)</tt> if argument \a I is uncertain. */
    template <std::size_t I, typename Fn>
    void
    visitIfUncertain(Fn & fn) const
    {
        if constexpr (ArgInfo::uncertain[I])
        {
            fn.template operator()<I>(argInput(std::get<I>(m_args)));
        }
    }

    /** \returns argument \a I for \a f: its block of \a x if uncertain, otherwise itself */
    template <typename T, std::size_t I>
    [[nodiscard]] decltype(auto)
    argument(InputVector<T> const & x) const
    {
        auto const & arg = std::get<I>(m_args);
        if constexpr (ArgInfo::uncertain[I])
        {
            using Input = typename ArgTraits<ArgAt<I>>::Input;
            using Block = Eigen::Matrix<
                T,
                Input::ValueDimensionAtCompileTime,
                1,
                Eigen::ColMajor,
                Input::MaxValueDimensionAtCompileTime,
                1>;
            return Block{x.segment(m_offsets[I], argInput(arg).valueDimension())};
        }
        else
        {
            return arg;
        }
    }

    /** \returns argument \a I for a batch evaluation: its rows of \a x, a column per point, if
     * uncertain, otherwise itself
     */
    template <std::size_t I, typename X>
    [[nodiscard]] decltype(auto)
    argumentColumns(X const & x) const
    {
        auto const & arg = std::get<I>(m_args);
        if constexpr (ArgInfo::uncertain[I])
        {
            using Input = typename ArgTraits<ArgAt<I>>::Input;
            return internal::middleRowsOf<Input::ValueDimensionAtCompileTime>(
                x,
                m_offsets[I],
                argInput(arg).valueDimension()
            );
        }
        else
        {
            return (arg);
        }
    }

    /** Declared only, for Returned: the type of \a f called with argument() of each \a I, and with
     * \a T as its explicit template argument where it takes one (WithScalar).
     */
    template <typename T, std::size_t... I>
    static auto returnedAt(std::index_sequence<I...>) -> decltype(std::declval<WithScalar<T, F> const &>()(
        std::declval<Transform const &>().template argument<T, I>(std::declval<InputVector<T> const &>())...
    ));

public:

    /** What \a f returns at a joint input with scalar \a T, found from its declaration without
     * compiling a call, so that linearizeAt() can say what an \a f needs before a conversion
     * inside a call fails.
     */
    template <typename T>
    using Returned = decltype(returnedAt<T>(std::index_sequence_for<Args...>{}));

private:

    F & m_f;                                                   ///< the function
    OutputSpace m_outputSpace;                                 ///< the output's space, sized
    std::tuple<Args const &...> m_args;                        ///< the arguments
    std::array<Index, sizeof...(Args)> m_offsets{};            ///< each argument's offset in the joint values
    std::array<Index, sizeof...(Args)> m_differenceOffsets{};  ///< each argument's offset in the joint differences
    Index m_inputSize = 0;                                     ///< the joint input's value dimension
    Index m_inputDifferenceSize = 0;                           ///< the joint input's difference dimension
};

}  // namespace internal

}  // namespace Emrys

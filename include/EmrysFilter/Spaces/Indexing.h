#pragma once

/** \file
 * What marginals and conditionals are built from: resolving Eigen index sets against a space;
 * the shared implementations of marginal<Path>() and conditional<Path>(); inserting and erasing
 * rows of vectors and symmetric matrices for insert() and erase(); reading a symmetric matrix
 * from its lower triangle; and conditionMoments() and conditionMomentsWithEvidence(), the %Gaussian
 * conditioning every %Gaussian form calls. Everything here is in `internal`. Read Discrete.h next.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys::internal
{

/** A column of resolved indices with compile-time size \a Size and bound \a MaxSize. */
template <int Size, int MaxSize>
using IndexVector = Eigen::Matrix<Eigen::Index, Size, 1, Eigen::ColMajor, MaxSize, 1>;

/** Compile-time size of the subset that \a Indices selects from a vector of size \a SourceSize. */
template <typename Indices, int SourceSize, int SourceMaxSize>
struct IndexedSizeOf
{
private:

    /// a vector of the source's size, to index with \a Indices
    using Probe = Eigen::Matrix<int, SourceSize, 1, Eigen::ColMajor, SourceMaxSize, 1>;

    /// what indexing it gives, whose compile-time rows are the subset's size
    using Expr = std::remove_cvref_t<decltype(std::declval<Probe const &>()(std::declval<Indices const &>()))>;

    static_assert(Expr::ColsAtCompileTime == 1, "marginal and conditional indices must select a set, not one index");

public:

    static constexpr int size = Expr::RowsAtCompileTime;  ///< the subset's size, or Eigen::Dynamic

    /// the subset's bound: its size if fixed, else Eigen's bound on the expression, else the
    /// source's
    static constexpr int maxSize = size != Eigen::Dynamic                       ? size
                                 : Expr::MaxRowsAtCompileTime != Eigen::Dynamic ? int(Expr::MaxRowsAtCompileTime)
                                                                                : SourceMaxSize;

    using type = IndexVector<size, maxSize>;  ///< the resolved indices' type
};

/** The index vector \a Indices resolves to, sized by the subset it selects from a vector of size
 * \a SourceSize.
 */
template <typename Indices, int SourceSize, int SourceMaxSize>
using IndexedSize = typename IndexedSizeOf<Indices, SourceSize, SourceMaxSize>::type;

/** \returns the concrete indices that \a indices selects from \c 0..n-1. Any index
 * type Eigen accepts works, including symbolic ones such as \c lastN(2). Eigen
 * asserts that every index is in range.
 */
template <int SourceSize, int SourceMaxSize, typename Indices>
[[nodiscard]] IndexedSize<Indices, SourceSize, SourceMaxSize>
resolveIndices(Eigen::Index n, Indices const & indices)
{
    using Identity = Eigen::Matrix<Eigen::Index, SourceSize, 1, Eigen::ColMajor, SourceMaxSize, 1>;
    Identity const identity = Identity::LinSpaced(n, 0, n - 1);
    return identity(indices);
}

/** \returns the \a count indices from \a start on, with compile-time size \a Size. */
template <int Size, int MaxSize>
[[nodiscard]] IndexVector<Size, MaxSize>
consecutiveIndices(Eigen::Index start, Eigen::Index count)
{
    return IndexVector<Size, MaxSize>::LinSpaced(count, start, start + count - 1);
}

/** \returns whether the concatenation of \a indices contains no repeats. Compares every pair
 * rather than sorting a copy, so it allocates nothing.
 */
template <typename... Vectors>
[[nodiscard]] bool
distinctIndices(Vectors const &... indices)
{
    bool distinct = true;
    std::size_t position = 0;
    auto const seenBefore = [&](Eigen::Index index, std::size_t before) {
        std::size_t k = 0;
        bool seen = false;
        auto const scan = [&](auto const & vector) {
            for (Eigen::Index const other : vector)
            {
                if (k++ < before && other == index)
                {
                    seen = true;
                }
            }
        };
        (scan(indices), ...);
        return seen;
    };
    auto const check = [&](auto const & vector) {
        for (Eigen::Index const index : vector)
        {
            if (seenBefore(index, position))
            {
                distinct = false;
            }
            ++position;
        }
    };
    (check(indices), ...);
    return distinct;
}

/** \returns the indices in \c 0..n-1 not in \a given, ascending, in storage with the source's
 * bound, so a bounded source allocates nothing. Fixed-size when both \a SourceSize and the size
 * of \a given are known at compile time.
 */
template <int SourceSize, int SourceMaxSize, int GivenSize, int GivenMaxSize>
[[nodiscard]] auto
complementIndices(Eigen::Index n, IndexVector<GivenSize, GivenMaxSize> const & given)
{
    auto taken = Eigen::Matrix<bool, SourceSize, 1, Eigen::ColMajor, SourceMaxSize, 1>{};
    taken.setConstant(n, false);
    for (Eigen::Index const i : given)
    {
        taken(i) = true;
    }
    constexpr int RestSize =
        SourceSize != Eigen::Dynamic && GivenSize != Eigen::Dynamic ? SourceSize - GivenSize : Eigen::Dynamic;
    auto result = IndexVector<RestSize, RestSize == Eigen::Dynamic ? SourceMaxSize : RestSize>{};
    result.resize(n - given.size());
    Eigen::Index out = 0;
    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (!taken(i))
        {
            result(out++) = i;
        }
    }
    return result;
}

/** Coordinates selected from a distribution: their rows in an operand of values, such as a mean
 * or a sample, and in an operand of differences, such as a covariance. The two differ where a
 * block's differences have other sizes than its values.
 */
template <typename Values, typename Differences>
struct Coordinates
{
    Values values;            ///< rows in an operand of values
    Differences differences;  ///< rows in an operand of differences
};

/** \returns the coordinates of \a space selected by \a values and \a differences, index sets
 * Eigen accepts over its values and its differences, resolved
 */
template <typename Space, typename ValueIndices, typename DifferenceIndices>
[[nodiscard]] auto
resolveCoordinates(Space const & space, ValueIndices const & values, DifferenceIndices const & differences)
{
    return Coordinates{
        resolveIndices<valueDimensionAtCompileTime<Space>, maxValueDimensionAtCompileTime<Space>>(
            valueDimension(space),
            asIndexSet(values)
        ),
        resolveIndices<
            valueDimensionAtCompileTime<DifferenceSpace<Space>>,
            maxValueDimensionAtCompileTime<DifferenceSpace<Space>>>(differenceDimension(space), asIndexSet(differences))
    };
}

/** \returns the coordinates of the subspace named by \a Path within \a space, resolved */
template <typename Path, typename Space>
[[nodiscard]] auto
pathCoordinates(Space const & space)
{
    return resolveCoordinates(space, idx<Path>(space), diffIdx<Path>(space));
}

/** The coordinates a marginal selects, resolved, and the space they form. A marginal of a
 * distribution made of elements resolves them once and lends them to the marginal of each element,
 * so that making those allocates nothing and repeats no work.
 */
template <typename IndexVector_, typename DifferenceIndexVector_, typename SubSpace>
struct ResolvedSelection
{
    IndexVector_ indices;                      ///< the selected entries of a value
    DifferenceIndexVector_ differenceIndices;  ///< the selected rows of a covariance
    SubSpace space;                            ///< the space the selected coordinates form
};

/** A selection resolved and held by another marginal, which a marginal of one element borrows. */
template <typename Resolved>
struct SharedSelection
{
    Resolved const * resolved;  ///< the borrowed selection, which outlives the borrower
};

/** Whether \a Selection is a SharedSelection. */
template <typename Selection>
inline constexpr bool isSharedSelection = false;

/** A SharedSelection is one. */
template <typename Resolved>
inline constexpr bool isSharedSelection<SharedSelection<Resolved>> = true;

/** How a marginal of \a Source holds the coordinates \a Selection selects, which form \a SubSpace:
 * resolved against the source, and owned.
 */
template <typename Source, typename Selection, typename SubSpace>
class SelectionHolder
{
public:

    /** The resolved selection, its index vectors sized by the source's bounds. */
    using ResolvedType = ResolvedSelection<
        IndexedSize<
            decltype(Selection::values),
            Source::ValueDimensionAtCompileTime,
            Source::MaxValueDimensionAtCompileTime>,
        IndexedSize<
            decltype(Selection::differences),
            Source::DifferenceDimensionAtCompileTime,
            Source::MaxDifferenceDimensionAtCompileTime>,
        SubSpace>;

    /** Resolves \a selection against \a source; its coordinates form \a space. */
    SelectionHolder(Source const & source, Selection const & selection, SubSpace space)
        : m_resolved{
              resolveIndices<Source::ValueDimensionAtCompileTime, Source::MaxValueDimensionAtCompileTime>(
                  source.valueDimension(),
                  selection.values
              ),
              resolveIndices<Source::DifferenceDimensionAtCompileTime, Source::MaxDifferenceDimensionAtCompileTime>(
                  source.differenceDimension(),
                  selection.differences
              ),
              std::move(space)
          }
    {
        EMRYS_FILTER_ASSERT(
            distinctIndices(m_resolved.indices) && distinctIndices(m_resolved.differenceIndices)
            && "marginal indices must be distinct"
        );
    }

    /** \returns the resolved selection */
    [[nodiscard]] ResolvedType const &
    get() const
    {
        return m_resolved;
    }

private:

    ResolvedType m_resolved;  ///< the resolved selection
};

/** How a marginal of one element holds a selection that the marginal of the whole lends it. */
template <typename Source, typename Resolved, typename SubSpace>
class SelectionHolder<Source, SharedSelection<Resolved>, SubSpace>
{
public:

    using ResolvedType = Resolved;  ///< the borrowed selection's type

    /** Borrows \a shared's selection. */
    explicit SelectionHolder(SharedSelection<Resolved> shared)
        : m_resolved{shared.resolved}
    {}

    /** \returns the borrowed selection */
    [[nodiscard]] ResolvedType const &
    get() const
    {
        return *m_resolved;
    }

private:

    Resolved const * m_resolved;  ///< the borrowed selection, which outlives this
};

/** Shared implementation of conditional<GivenPath>(): conditions a source in \a space on the
 * subspace named by \a GivenPath, with every other coordinate as the target.
 *
 * The given values are in the named subspace, and the target is in the space of the remaining
 * blocks (see complementSubspace()), so a joint conditioned on one block returns the rest in
 * their own spaces.
 *
 * \returns \a condition(target, given, targetSpace, givenSpace), with target and given resolved
 * Coordinates
 */
template <typename GivenPath, typename Space, typename Condition>
[[nodiscard]] auto
tagConditional(Space const & space, Condition condition)
{
    constexpr int N = valueDimensionAtCompileTime<Space>;
    constexpr int NMax = maxValueDimensionAtCompileTime<Space>;
    constexpr int D = valueDimensionAtCompileTime<DifferenceSpace<Space>>;
    constexpr int DMax = maxValueDimensionAtCompileTime<DifferenceSpace<Space>>;
    Eigen::Index const n = valueDimension(space);
    Eigen::Index const d = differenceDimension(space);
    auto const given = pathCoordinates<GivenPath>(space);
    auto const target =
        Coordinates{complementIndices<N, NMax>(n, given.values), complementIndices<D, DMax>(d, given.differences)};
    return condition(target, given, pathComplement<GivenPath>(space), pathSubspace<GivenPath>(space));
}

/** Shared implementation of marginal<Path>() for distributions that own their parameters.
 *
 * \returns \a make(target, targetSpace): the Coordinates of the subspace named by \a Path, and
 * that subspace
 */
template <typename Path, typename Space, typename Make>
[[nodiscard]] auto
marginalOverPath(Space const & space, Make make)
{
    return make(pathCoordinates<Path>(space), pathSubspace<Path>(space));
}

/** Shared implementation of marginal<Path>(element) for distributions that own their
 * parameters, where \a Path starts at a repeated block.
 *
 * \returns \a make(target, targetSpace): the Coordinates of repetition \a element (or of the rest
 * of the path within it), and the space of one repetition (or of the rest of the path)
 */
template <typename Path, typename Space, typename Make>
[[nodiscard]] auto
marginalOverElement(Space const & space, Eigen::Index element, Make make)
{
    return make(
        resolveCoordinates(space, idx<Path>(space, element), diffIdx<Path>(space, element)),
        pathElementSubspace<Path>(space)
    );
}

/** Inserts \a rows into the rows of \a matrix before row \a at. */
template <typename Matrix, typename Derived>
void
insertRows(Matrix & matrix, Eigen::Index at, Eigen::MatrixBase<Derived> const & rows)
{
    Eigen::Index const n = matrix.rows();
    Eigen::Index const k = rows.rows();
    EMRYS_FILTER_ASSERT(at >= 0 && at <= n && rows.cols() == matrix.cols());
    auto result = Matrix{};
    result.resize(n + k, matrix.cols());
    result.topRows(at) = matrix.topRows(at);
    result.middleRows(at, k) = rows;
    result.bottomRows(n - at) = matrix.bottomRows(n - at);
    matrix = std::move(result);
}

/** Removes \a count rows of \a matrix starting at row \a at. */
template <typename Matrix>
void
eraseRows(Matrix & matrix, Eigen::Index at, Eigen::Index count)
{
    Eigen::Index const n = matrix.rows();
    EMRYS_FILTER_ASSERT(at >= 0 && count >= 0 && at + count <= n);
    auto result = Matrix{};
    result.resize(n - count, matrix.cols());
    result.topRows(at) = matrix.topRows(at);
    result.bottomRows(n - at - count) = matrix.bottomRows(n - at - count);
    matrix = std::move(result);
}

/** Inserts rows and columns into the symmetric \a matrix before index \a at: the block
 * \a block on the diagonal, and \a cross (new rows by existing columns) off it.
 */
template <typename Matrix, typename BlockDerived, typename CrossDerived>
void
insertSymmetric(
    Matrix & matrix,
    Eigen::Index at,
    Eigen::MatrixBase<BlockDerived> const & block,
    Eigen::MatrixBase<CrossDerived> const & cross
)
{
    Eigen::Index const n = matrix.rows();
    Eigen::Index const k = block.rows();
    Eigen::Index const after = n - at;
    EMRYS_FILTER_ASSERT(at >= 0 && at <= n && block.cols() == k && cross.rows() == k && cross.cols() == n);
    auto result = Matrix{};
    result.resize(n + k, n + k);
    result.topLeftCorner(at, at) = matrix.topLeftCorner(at, at);
    result.topRightCorner(at, after) = matrix.topRightCorner(at, after);
    result.bottomLeftCorner(after, at) = matrix.bottomLeftCorner(after, at);
    result.bottomRightCorner(after, after) = matrix.bottomRightCorner(after, after);
    result.block(at, at, k, k) = block;
    result.block(at, 0, k, at) = cross.leftCols(at);
    result.block(at, at + k, k, after) = cross.rightCols(after);
    result.block(0, at, at, k) = cross.leftCols(at).transpose();
    result.block(at + k, at, after, k) = cross.rightCols(after).transpose();
    matrix = std::move(result);
}

/** Removes \a count rows and columns of \a matrix starting at index \a at. */
template <typename Matrix>
void
eraseSymmetric(Matrix & matrix, Eigen::Index at, Eigen::Index count)
{
    Eigen::Index const n = matrix.rows();
    Eigen::Index const after = n - at - count;
    EMRYS_FILTER_ASSERT(at >= 0 && count >= 0 && after >= 0);
    auto result = Matrix{};
    result.resize(n - count, n - count);
    result.topLeftCorner(at, at) = matrix.topLeftCorner(at, at);
    result.topRightCorner(at, after) = matrix.topRightCorner(at, after);
    result.bottomLeftCorner(after, at) = matrix.bottomLeftCorner(after, at);
    result.bottomRightCorner(after, after) = matrix.bottomRightCorner(after, after);
    matrix = std::move(result);
}

/** \returns the symmetric matrix whose lower triangle is \a matrix, as a \a Result: copied in one
 * pass from a plain matrix, and evaluated first from any other expression.
 */
template <typename Result, typename Derived>
[[nodiscard]] Result
symmetricFromLower(Eigen::MatrixBase<Derived> const & matrix)
{
    if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<Derived>, Derived>)
    {
        Result result = matrix.template selfadjointView<Eigen::Lower>();
        return result;
    }
    else
    {
        Result result = matrix;
        result.template triangularView<Eigen::StrictlyUpper>() = result.transpose();
        return result;
    }
}

/** \returns a lazy expression for the block of the symmetric matrix whose lower
 * triangle is \a matrix, at rows \a rows and columns \a cols. Only lower-triangle
 * entries of \a matrix are read, whatever order the indices are in. The expression
 * refers to its arguments and must not outlive them.
 */
template <int Rows, int MaxRows, int Cols, int MaxCols, typename Derived>
[[nodiscard]] auto
lowerSymmetricBlock(
    Eigen::MatrixBase<Derived> const & matrix,
    IndexVector<Rows, MaxRows> const & rows,
    IndexVector<Cols, MaxCols> const & cols
)
{
    using Scalar = typename Derived::Scalar;
    using Result = MatrixOf<Scalar, Rows, Cols, MaxRows, MaxCols>;
    return Result::NullaryExpr(
        rows.size(),
        cols.size(),
        [&matrix = matrix.derived(), &rows, &cols](Eigen::Index i, Eigen::Index j) {
            Eigen::Index const r = rows(i);
            Eigen::Index const c = cols(j);
            return r >= c ? matrix(r, c) : matrix(c, r);
        }
    );
}

/** \returns the block of the symmetric matrix whose lower triangle is \a matrix, at \a indices
 * on both axes, which may be any index set Eigen accepts: a coefficient for a single index,
 * otherwise a lazy expression that holds its resolved indices and refers to \a matrix, so it
 * must not outlive \a matrix. Only lower-triangle entries are read.
 */
template <int Size, int MaxSize, typename Derived, typename Indices>
[[nodiscard]] auto
lowerSymmetricSelection(Eigen::MatrixBase<Derived> const & matrix, Indices const & indices)
{
    if constexpr (std::is_integral_v<Indices>)
    {
        return matrix.derived().coeff(indices, indices);
    }
    else
    {
        using Scalar = typename Derived::Scalar;
        using Resolved = IndexedSize<Indices, Size, MaxSize>;
        using Result = MatrixOf<
            Scalar,
            Resolved::RowsAtCompileTime,
            Resolved::RowsAtCompileTime,
            Resolved::MaxRowsAtCompileTime,
            Resolved::MaxRowsAtCompileTime>;
        Resolved resolved = resolveIndices<Size, MaxSize>(matrix.rows(), indices);
        Eigen::Index const n = resolved.size();
        return Result::NullaryExpr(
            n,
            n,
            [&matrix = matrix.derived(), resolved = std::move(resolved)](Eigen::Index i, Eigen::Index j) {
                Eigen::Index const r = resolved(i);
                Eigen::Index const c = resolved(j);
                return r >= c ? matrix(r, c) : matrix(c, r);
            }
        );
    }
}

/** \returns the block of the symmetric matrix whose lower triangle is \a matrix, at rows \a rows
 * and columns \a cols, which may be any index sets Eigen accepts: a coefficient if both are
 * single indices, otherwise an evaluated matrix. Only lower-triangle entries are read.
 */
template <int Size, int MaxSize, typename Derived, typename Rows, typename Cols>
[[nodiscard]] auto
lowerSymmetricCross(Eigen::MatrixBase<Derived> const & matrix, Rows const & rows, Cols const & cols)
{
    if constexpr (std::is_integral_v<Rows> && std::is_integral_v<Cols>)
    {
        return rows >= cols ? matrix.derived().coeff(rows, cols) : matrix.derived().coeff(cols, rows);
    }
    else
    {
        auto const resolvedRows = resolveIndices<Size, MaxSize>(matrix.rows(), asIndexSet(rows));
        auto const resolvedCols = resolveIndices<Size, MaxSize>(matrix.rows(), asIndexSet(cols));
        using RowIndices = std::remove_cvref_t<decltype(resolvedRows)>;
        using ColIndices = std::remove_cvref_t<decltype(resolvedCols)>;
        using Result = MatrixOf<
            typename Derived::Scalar,
            RowIndices::RowsAtCompileTime,
            ColIndices::RowsAtCompileTime,
            RowIndices::MaxRowsAtCompileTime,
            ColIndices::MaxRowsAtCompileTime>;
        return Result{lowerSymmetricBlock(matrix, resolvedRows, resolvedCols)};
    }
}

/** Moments of a %Gaussian conditional in \a TargetSpace. */
template <typename Scalar, typename TargetSpace>
struct ConditionedMoments
{
    /// the conditional mean, in the target space
    Eigen::Matrix<
        Scalar,
        valueDimensionAtCompileTime<TargetSpace>,
        1,
        Eigen::ColMajor,
        maxValueDimensionAtCompileTime<TargetSpace>,
        1>
        mean;

    /// the conditional covariance, over the target space's differences
    MatrixOf<
        Scalar,
        valueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>,
        valueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>>
        cov;
};

/** The moments of a %Gaussian conditional and the likelihood of the conditioning values, from
 * conditionMomentsWithEvidence().
 */
template <typename Scalar, typename TargetSpace>
struct ConditionedMomentsWithEvidence
{
    ConditionedMoments<Scalar, TargetSpace> moments;  ///< the conditional's moments
    Scalar logLikelihood;    ///< the log-density of the conditioning values under their marginal
    Scalar squaredDistance;  ///< squared Mahalanobis distance of the conditioning values from their mean
};

/** The square matrix over the differences of \a GivenSpace, which \f$ \Sigma_{bb} \f$ is. */
template <typename Scalar, typename GivenSpace>
using GivenMatrixOf = MatrixOf<
    Scalar,
    valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>,
    valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>>;

/** The column over the differences of \a GivenSpace, which the residual \f$ v \boxminus \mu_b \f$ is. */
template <typename Scalar, typename GivenSpace>
using GivenVectorOf = Eigen::Matrix<
    Scalar,
    valueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>,
    1,
    Eigen::ColMajor,
    maxValueDimensionAtCompileTime<DifferenceSpace<GivenSpace>>,
    1>;

/** The moments of conditionMoments() and conditionMomentsWithEvidence(), from \a factorization of
 * \f$ \Sigma_{bb} \f$ (an `Eigen::LLT` or `Eigen::LDLT`) and the \a residual
 * \f$ v \boxminus \mu_b \f$.
 */
template <
    typename Factorization,
    typename MeanDerived,
    typename CovDerived,
    typename ResidualDerived,
    typename Target,
    typename Given,
    typename TargetSpace>
[[nodiscard]] ConditionedMoments<typename MeanDerived::Scalar, TargetSpace>
conditionMomentsBy(
    Factorization const & factorization,
    Eigen::MatrixBase<MeanDerived> const & mean,
    Eigen::MatrixBase<CovDerived> const & cov,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<ResidualDerived> const & residual,
    TargetSpace const & targetSpace
)
{
    using Scalar = typename MeanDerived::Scalar;
    constexpr int GivenSize = ResidualDerived::RowsAtCompileTime;
    constexpr int GivenMaxSize = ResidualDerived::MaxRowsAtCompileTime;
    constexpr int TargetSize = valueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>;
    constexpr int TargetMaxSize = maxValueDimensionAtCompileTime<DifferenceSpace<TargetSpace>>;
    using CrossMatrix = MatrixOf<Scalar, TargetSize, GivenSize, TargetMaxSize, GivenMaxSize>;

    auto result = ConditionedMoments<Scalar, TargetSpace>{};
    // With nothing to condition, only the likelihood remains; MSVC can't form an empty gain.
    if constexpr (TargetSize != 0)
    {
        CrossMatrix const cross = lowerSymmetricBlock(cov, target.differences, given.differences);
        CrossMatrix const gain = transposedSolve<CrossMatrix>(factorization, cross.transpose());
        result.cov = lowerSymmetricBlock(cov, target.differences, target.differences);
        result.cov.noalias() -= multiply(gain, cross.transpose());
        if constexpr (hasIdentityReset<TargetSpace>)
        {
            result.mean = Emrys::plus(targetSpace, mean.derived()(target.values), multiply(gain, residual));
        }
        else
        {
            using TargetVector = decltype(result.mean);
            using TargetDifference = Eigen::Matrix<Scalar, TargetSize, 1, Eigen::ColMajor, TargetMaxSize, 1>;
            TargetVector const start = mean.derived()(target.values);
            TargetDifference const step = multiply(gain, residual);
            result.mean = Emrys::plus(targetSpace, start, step);
            resetCovarianceAt(targetSpace, start, step, result.mean, result.cov);
        }
    }
    return result;
}

/** Conditions the %Gaussian with mean \a mean and (lower-triangle) covariance \a cov
 * on the coordinates \a given equal to \a values, for the coordinates \a target, each a
 * Coordinates of resolved indices.
 *
 * The residual \f$ v \boxminus \mu_b \f$ is taken in \a givenSpace and the
 * conditional mean \f$ \mu_a \boxplus K r \f$ in \a targetSpace. Solves against
 * \f$ \Sigma_{bb} \f$ with a pivoted LDLT, which always produces a result. A block known exactly
 * makes \f$ \Sigma_{bb} \f$ singular, and then \f$ \Sigma_{ab} \f$ is zero in the same
 * directions, so skipping the zero pivot gives the exact conditional for values on the support
 * (decisions.md [innovation-cholesky]). If the source is numerically singular, the conditional covariance may not
 * be numerically positive definite. The conditional covariance, computed in the tangent space at
 * \f$ \mu_a \f$, is then carried to the one at the conditional mean by resetCovariance()
 * (decisions.md [covariance-reset]), which does nothing where \a targetSpace has an identity reset.
 *
 * The formulas are the
 * [conditional of a multivariate
 * normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions):
 * \f$ \mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (v - \mu_b) \f$ and
 * \f$ \Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba} \f$, with the
 * Kalman gain \f$ K = \Sigma_{ab} \Sigma_{bb}^{-1} \f$ playing the role it does in
 * \ref Kalman1960 "Kalman (1960)".
 */
template <
    typename MeanDerived,
    typename CovDerived,
    typename ValuesDerived,
    typename Target,
    typename Given,
    typename GivenSpace,
    typename TargetSpace>
[[nodiscard]] ConditionedMoments<typename MeanDerived::Scalar, TargetSpace>
conditionMoments(
    Eigen::MatrixBase<MeanDerived> const & mean,
    Eigen::MatrixBase<CovDerived> const & cov,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<ValuesDerived> const & values,
    GivenSpace const & givenSpace,
    TargetSpace const & targetSpace
)
{
    using Scalar = typename MeanDerived::Scalar;
    EMRYS_FILTER_ASSERT(values.size() == given.values.size());
    GivenVectorOf<Scalar, GivenSpace> const residual = Emrys::minus(givenSpace, values, mean.derived()(given.values));
    auto const ldlt =
        Eigen::LDLT<GivenMatrixOf<Scalar, GivenSpace>>{lowerSymmetricBlock(cov, given.differences, given.differences)};
    return conditionMomentsBy(ldlt, mean, cov, target, given, residual, targetSpace);
}

/** conditionMoments(), with the log-density of \a values under the marginal of \a given, or
 * nothing if \f$ \Sigma_{bb} \f$ is not positive definite.
 *
 * Factors \f$ \Sigma_{bb} \f$ by Cholesky, as an update factors its innovation covariance
 * (decisions.md [innovation-cholesky]). A singular \f$ \Sigma_{bb} \f$ has no density over the given block: on a
 * pivoted LDLT, a zero pivot gave a log-likelihood of \f$ +\infty \f$ whatever the values, and so
 * all the weight of a mixture to that component even where the values rule it out.
 */
template <
    typename MeanDerived,
    typename CovDerived,
    typename ValuesDerived,
    typename Target,
    typename Given,
    typename GivenSpace,
    typename TargetSpace>
[[nodiscard]] std::optional<ConditionedMomentsWithEvidence<typename MeanDerived::Scalar, TargetSpace>>
conditionMomentsWithEvidence(
    Eigen::MatrixBase<MeanDerived> const & mean,
    Eigen::MatrixBase<CovDerived> const & cov,
    Target const & target,
    Given const & given,
    Eigen::MatrixBase<ValuesDerived> const & values,
    GivenSpace const & givenSpace,
    TargetSpace const & targetSpace
)
{
    using Scalar = typename MeanDerived::Scalar;
    using GivenVector = GivenVectorOf<Scalar, GivenSpace>;
    EMRYS_FILTER_ASSERT(values.size() == given.values.size());
    auto const llt =
        Eigen::LLT<GivenMatrixOf<Scalar, GivenSpace>>{lowerSymmetricBlock(cov, given.differences, given.differences)};
    if (llt.info() != Eigen::Success)
    {
        return std::nullopt;
    }
    GivenVector const residual = Emrys::minus(givenSpace, values, mean.derived()(given.values));
    GivenVector whitened = residual;
    llt.matrixL().solveInPlace(whitened);

    Scalar const log2Pi = Emrys::log2Pi<Scalar>();
    Scalar const logDeterminant = Scalar(2) * llt.matrixLLT().diagonal().array().log().sum();
    Scalar const squaredDistance = whitened.squaredNorm();
    return ConditionedMomentsWithEvidence<Scalar, TargetSpace>{
        conditionMomentsBy(llt, mean, cov, target, given, residual, targetSpace),
        Scalar(-0.5) * (Scalar(given.differences.size()) * log2Pi + logDeterminant + squaredDistance),
        squaredDistance
    };
}

}  // namespace Emrys::internal

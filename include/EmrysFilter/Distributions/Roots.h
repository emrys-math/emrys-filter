#pragma once

/** \file
 * Square roots of a covariance (decisions.md [sigma-point-axes]): the Axes and Repair concepts that a root of the
 * user's own meets [user-extension-points], the library's axes roots::cholesky(), roots::pivoted(),
 * roots::principal(), roots::symmetric(), and roots::along(), the repair roots::ridge(), the flat
 * list of fallbacks that axes which can fail take, and roots::squareRoot() and
 * roots::squareRootWithReport(), which take a root by them. Also GaussianError, the reasons a root
 * or a density can fail. Read Gaussian.h next.
 */

#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Core/Triangularization.h>

namespace Emrys
{

namespace internal
{

/** The square matrix of a root of the covariance or factor \a Derived: a row and a column per row
 * of it, with its bounds, so that a bounded matrix allocates nothing [allocation].
 */
template <typename Derived>
using SquareMatrixOf = Eigen::Matrix<
    typename Derived::Scalar,
    Derived::RowsAtCompileTime,
    Derived::RowsAtCompileTime,
    Eigen::ColMajor,
    Derived::MaxRowsAtCompileTime,
    Derived::MaxRowsAtCompileTime>;

/** A column of the root \a Matrix, with its bounds. */
template <typename Matrix>
using RootColumnOf = Eigen::
    Matrix<typename Matrix::Scalar, Matrix::RowsAtCompileTime, 1, Eigen::ColMajor, Matrix::MaxRowsAtCompileTime, 1>;

/** What a root returns: the root \a Matrix, or a \c std::expected of it when it \a CanFail. */
template <bool CanFail, typename Matrix>
using RootResult = std::conditional_t<CanFail, std::expected<Matrix, GaussianError>, Matrix>;

/** A sink for an entry's report that keeps nothing: what roots::squareRoot() gives, so that the
 * plain root costs nothing for the reports.
 */
struct DiscardReport
{
    /** Discards \a report. */
    template <typename Report>
    constexpr void
    operator()(Report const & /*report*/) const noexcept
    {}
};

/** The template constant \a Value in \a Scalar, through its literal type as pi() converts, so that a
 * \c float filter holds no \c double [embedded-builds].
 */
template <typename Scalar, double Value>
[[nodiscard]] Scalar
scalarFromConstant()
{
    using Literal = typename Eigen::NumTraits<Scalar>::Literal;
    return Scalar(static_cast<Literal>(Value));
}

/** Whether the lower triangle of \a a, the part a covariance is read from, is finite. */
template <typename Derived>
[[nodiscard]] bool
lowerFinite(Eigen::MatrixBase<Derived> const & a)
{
    for (Eigen::Index j = 0; j < a.cols(); ++j)
    {
        if (!allFinite(a.col(j).tail(a.rows() - j)))
        {
            return false;
        }
    }
    return true;
}

/** \returns \a cov with a ridge \f$ \delta \f$ added to its diagonal, and \f$ \delta \f$ in
 *           \a delta, or GaussianError::NonFinite if its lower triangle contains NaN or infinity.
 *
 * The ridge comes from the
 * [Gershgorin circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem): every
 * eigenvalue of a symmetric matrix \f$ A \f$ lies in an interval centered on a diagonal element, so
 * \f[ \lambda_{\min} \ge \min_i \left( A_{ii} - \sum_{j \ne i} |A_{ij}| \right). \f]
 * Shifting by \f$ \sigma = \max(0, -\lambda_{\text{bound}}) \f$ makes the matrix positive
 * semidefinite, and a margin of \a Relative times \f$ \sqrt{\epsilon} \f$ times the larger of the
 * largest diagonal magnitude and the shift, plus \a Absolute, makes it positive definite with room
 * for rounding:
 * \f[ \delta = \sigma + r \sqrt{\epsilon} \max\left( \max_i |A_{ii}|, \sigma \right) + a. \f]
 * With \a Absolute zero the margin is zero only for a zero matrix, which therefore still fails. The
 * bound is loose for matrices with large off-diagonal elements, so the ridge can be larger than the
 * smallest one that would succeed.
 */
template <double Relative = 1.0, double Absolute = 0.0, typename Derived>
[[nodiscard]] std::expected<typename Derived::PlainObject, GaussianError>
ridgedCovariance(Eigen::MatrixBase<Derived> const & cov, typename Derived::Scalar * delta = nullptr)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(cov.rows() == cov.cols() && "a covariance must be square");
    Eigen::Index const n = cov.rows();

    // Row i's off-diagonal elements are row i left of the diagonal and, by symmetry, column i
    // below it. NaN or infinity anywhere in the lower triangle reaches some row's bound.
    auto const diagonal = cov.diagonal();
    Scalar bound = Eigen::NumTraits<Scalar>::infinity();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        Scalar const radius = cov.row(i).head(i).cwiseAbs().sum() + cov.col(i).tail(n - i - 1).cwiseAbs().sum();
        Scalar const rowBound = diagonal(i) - radius;
        if (!isFinite(rowBound))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        bound = Eigen::numext::mini(bound, rowBound);
    }

    Scalar const shift = Eigen::numext::maxi(-bound, Scalar(0));
    Scalar const scale = Eigen::numext::maxi(n > 0 ? diagonal.cwiseAbs().maxCoeff() : Scalar(0), shift);
    Scalar const margin =
        scalarFromConstant<Scalar, Relative>() * Eigen::numext::sqrt(Eigen::NumTraits<Scalar>::epsilon()) * scale
        + scalarFromConstant<Scalar, Absolute>();
    typename Derived::PlainObject ridged = cov;
    ridged.diagonal().array() += shift + margin;
    if (delta != nullptr)
    {
        *delta = shift + margin;
    }
    return ridged;
}

/** Flips the sign of each column of \a root whose entry of largest magnitude, the first of any tie,
 * is negative, so that an eigenvector's arbitrary sign does not reach the points a root spreads.
 */
template <typename Matrix>
void
fixColumnSigns(Matrix & root)
{
    for (Eigen::Index j = 0; j < root.cols(); ++j)
    {
        Eigen::Index largest = 0;
        root.col(j).cwiseAbs().maxCoeff(&largest);
        if (root(largest, j) < typename Matrix::Scalar(0))
        {
            root.col(j) *= typename Matrix::Scalar(-1);
        }
    }
}

/** Whether \a a is square and lower triangular with a nonnegative diagonal, as the factor a
 * square-root form holds is.
 */
template <typename Derived>
[[nodiscard]] bool
isTriangularRoot(Eigen::MatrixBase<Derived> const & a)
{
    using Scalar = typename Derived::Scalar;
    if (a.rows() != a.cols())
    {
        return false;
    }
    for (Eigen::Index j = 0; j < a.cols(); ++j)
    {
        if (a(j, j) < Scalar(0))
        {
            return false;
        }
        for (Eigen::Index i = 0; i < j; ++i)
        {
            if (a(i, j) != Scalar(0))
            {
                return false;
            }
        }
    }
    return true;
}

/** The square lower-triangular \f$ L \f$ with a nonnegative diagonal and \f$ L L^T = F F^T \f$ for the
 * factor \a factor, which may be wide: \a factor itself when it is already one, as a square-root
 * form's is, and otherwise its triangularization.
 */
template <typename Derived>
[[nodiscard]] SquareMatrixOf<Derived>
triangularRootOf(Eigen::MatrixBase<Derived> const & factor)
{
    if constexpr (Derived::ColsAtCompileTime == Eigen::Dynamic
                  || Derived::ColsAtCompileTime == Derived::RowsAtCompileTime)
    {
        if (isTriangularRoot(factor))
        {
            return SquareMatrixOf<Derived>{factor};
        }
    }
    return lowerTriangularRoot(factor);
}

/** The eigenvalues and eigenvectors of the covariance \a cov, of which only the lower triangle is read,
 * with the eigenvalues largest first and the negative ones counted in \a clamped and taken as zero:
 * \f$ U \f$ and \f$ \Lambda^{1/2} \f$, so that \f$ U \Lambda^{1/2} \f$ is a root of the positive
 * semidefinite part of \a cov.
 */
template <typename Derived>
struct PrincipalAxes
{
    SquareMatrixOf<Derived> vectors;              ///< \f$ U \f$, a column per eigenvalue
    RootColumnOf<SquareMatrixOf<Derived>> roots;  ///< \f$ \Lambda^{1/2} \f$'s diagonal
    Eigen::Index clamped = 0;                     ///< the eigenvalues below zero, taken as zero
};

/** PrincipalAxes of the covariance \a cov by a symmetric eigensolver. */
template <typename Derived>
[[nodiscard]] PrincipalAxes<Derived>
principalAxesOfCovariance(Eigen::MatrixBase<Derived> const & cov)
{
    using Matrix = SquareMatrixOf<Derived>;
    using Scalar = typename Matrix::Scalar;
    auto const solver = Eigen::SelfAdjointEigenSolver<Matrix>{Matrix{cov}, Eigen::ComputeEigenvectors};
    Eigen::Index const n = cov.rows();
    auto axes = PrincipalAxes<Derived>{};
    axes.vectors.resize(n, n);
    axes.roots.resize(n);
    for (Eigen::Index j = 0; j < n; ++j)
    {
        // The solver sorts its eigenvalues smallest first.
        Scalar const value = solver.eigenvalues()(n - 1 - j);
        if (value < Scalar(0))
        {
            ++axes.clamped;
        }
        axes.vectors.col(j) = solver.eigenvectors().col(n - 1 - j);
        axes.roots(j) = Eigen::numext::sqrt(Eigen::numext::maxi(value, Scalar(0)));
    }
    return axes;
}

/** PrincipalAxes of \f$ F F^T \f$ for the factor \a factor, by a singular value decomposition of its
 * triangular factor, so that the covariance is never formed: \f$ L = U \Sigma V^T \f$ gives
 * \f$ L L^T = U \Sigma^2 U^T \f$.
 */
template <typename Derived>
[[nodiscard]] PrincipalAxes<Derived>
principalAxesOfFactor(Eigen::MatrixBase<Derived> const & factor)
{
    using Matrix = SquareMatrixOf<Derived>;
    Matrix const lower = triangularRootOf(factor);
    auto const svd = Eigen::JacobiSVD<Matrix, Eigen::ComputeFullU>{lower};
    return PrincipalAxes<Derived>{svd.matrixU(), svd.singularValues(), 0};
}

/** The axes \a T of a fallback list or alone: a type meeting Axes. */
template <typename T>
inline constexpr bool isFallbackList = false;

}  // namespace internal

/** \ingroup Distributions
 * \brief Axes: a way of taking a root \f$ S \f$ of a covariance \f$ P \f$, \f$ S S^T = P \f$, whose
 * columns carry a sigma-point rule's unit points into the input (decisions.md [sigma-point-axes]). Every root is
 * \f$ L Q \f$, with \f$ L \f$ the lower Cholesky factor and \f$ Q \f$ orthogonal, so the axes are the
 * choice of \f$ Q \f$, and a rule's moments of a function it does not integrate exactly depend on it.
 *
 * roots::cholesky(), roots::pivoted(), roots::principal(), roots::symmetric(), and roots::along()
 * are the library's, and a list of fallbacks after axes that can fail is axes too. Axes of the
 * user's own are any type with the members below (decisions.md [user-extension-points]); usage.md,
 * "Writing axes", writes one.
 *
 * - <tt>static constexpr bool canFail</tt>: whether a root can fail to be taken, as a Cholesky
 *   factorization fails on a matrix that is not positive definite.
 * - <tt>root(cov)</tt>, a \c const member template taking any square Eigen matrix and reading only its
 *   lower triangle: the root, square with \a cov's rows and its bounds (a fixed or bounded size
 *   allocates nothing), or, when \c canFail, a <tt>std::expected</tt> of it with a GaussianError.
 *
 * Optional:
 *
 * - <tt>rootFromFactor(factor)</tt>: the same root of \f$ F F^T \f$ from a factor \f$ F \f$, which may
 *   be wide, without forming the covariance, as a square-root form holds its factor. Without it the
 *   covariance is formed and given to <tt>root()</tt>.
 * - <tt>root(cov, sink)</tt>: the root, calling \a sink with a report of what was done, such as the
 *   pivots clamped, of the type <tt>Report\<Scalar\></tt> the axes name; roots::squareRootWithReport()
 *   collects it. Without it the axes report roots::NoReport.
 */
template <typename T>
concept Axes = requires {
    { std::bool_constant<T::canFail>{} } -> std::convertible_to<bool>;
} && requires(T const & axes, Eigen::Matrix3d const & cov) {
    { axes.root(cov) } -> std::same_as<internal::RootResult<T::canFail, Eigen::Matrix3d>>;
};

/** \ingroup Distributions
 * \brief A repair: what turns a covariance whose root failed into one to retry (decisions.md [sigma-point-axes]).
 *
 * roots::ridge() is the library's. A repair of the user's own, such as an eigenvalue floor or
 * Higham's nearest positive definite matrix, is any type with the member below
 * (decisions.md [user-extension-points]).
 *
 * - <tt>repaired(cov)</tt>, a \c const member template taking any square Eigen matrix and reading
 *   only its lower triangle: the matrix to retry, of \a cov's plain type, or a GaussianError, as a
 *   <tt>std::expected</tt>.
 *
 * Optional, as for Axes: <tt>repaired(cov, sink)</tt>, which calls \a sink with a report of the
 * type <tt>Report\<Scalar\></tt> the repair names, such as the ridge added.
 */
template <typename T>
concept Repair = requires(T const & repair, Eigen::Matrix3d const & cov) {
    { repair.repaired(cov) } -> std::same_as<std::expected<Eigen::Matrix3d, GaussianError>>;
};

namespace internal
{

/** The repair of what names none: a covariance that fails to factorize fails. Not a Repair, so that
 * no factorization is given it; what takes a repair or this tests for it.
 */
struct NoRepair
{};

}  // namespace internal

/** \ingroup Distributions
 * \brief The axes and repairs, each made by its factory (decisions.md [policy-factories]), and the
 * roots taken by them: cholesky(), pivoted(), principal(), symmetric(), along(), ridge(),
 * squareRoot(), and squareRootWithReport().
 */
namespace roots
{

/** \ingroup Distributions
 * The type of a root of the covariance or factor \a Derived: square, with a row and a column per row
 * of it and its bounds, so that a bounded matrix allocates nothing [allocation]. What axes of the
 * user's own return.
 */
template <typename Derived>
using RootMatrix = internal::SquareMatrixOf<Derived>;

/** \ingroup Distributions
 * The report of axes or a repair with nothing to say, as Cholesky's: it either took the root or
 * failed.
 */
struct NoReport
{};

/** \ingroup Distributions
 * The report of Pivoted, Principal, and Symmetric: how many pivots or eigenvalues were below zero
 * and taken as zero, which only rounding should give a covariance.
 */
struct ClampReport
{
    Eigen::Index clamped = 0;  ///< the pivots or eigenvalues taken as zero
};

/** \ingroup Distributions
 * The report of a Ridge: the ridge \f$ \delta \f$ it added to the diagonal.
 *
 * \tparam Scalar  the scalar of the covariance repaired
 */
template <typename Scalar>
struct RidgeReport
{
    Scalar ridge = 0;  ///< \f$ \delta \f$
};

/** \ingroup Distributions
 * Cholesky axes: the lower Cholesky factor \f$ L \f$, \f$ L L^T = P \f$, by Eigen's \c LLT, which
 * fails on a covariance that is not positive definite. The unique root of a positive definite
 * matrix that is lower triangular with a positive diagonal, an analytic function of \f$ P \f$, the
 * cheapest to take, and the factor the square-root forms hold, so that every form spreads the same
 * points (decisions.md [sigma-point-axes]). Column \f$ j \f$ is the variation of coordinates \f$ j \f$ onward given
 * the earlier ones, so the axes depend on the order of the coordinates.
 *
 * From a factor it triangularizes, which never fails: a factor of a singular matrix gives a root with
 * a zero on its diagonal, where the covariance's factorization fails and a list moves on to its
 * fallbacks. A covariance with NaN or infinity in its lower triangle fails with
 * GaussianError::NonFinite, one that is not positive definite with
 * GaussianError::NotPositiveDefinite.
 *
 * \sa cholesky(), \ref Golub2013 "Golub and Van Loan (2013)",
 * [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Statement)
 */
struct Cholesky
{
    static constexpr bool canFail = true;  ///< a covariance that is not positive definite has no Cholesky factor

    /** \returns the lower Cholesky factor of \a cov, of which only the lower triangle is read */
    template <typename Derived>
    [[nodiscard]] std::expected<internal::SquareMatrixOf<Derived>, GaussianError>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        using Matrix = internal::SquareMatrixOf<Derived>;
        if (!internal::lowerFinite(cov))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        auto const llt = Eigen::LLT<Matrix>{Matrix{cov}};
        if (llt.info() != Eigen::Success)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        return Matrix{llt.matrixL()};
    }

    /** \returns the lower-triangular root of \f$ F F^T \f$ for \a factor, by triangularization */
    template <typename Derived>
    [[nodiscard]] std::expected<internal::SquareMatrixOf<Derived>, GaussianError>
    rootFromFactor(Eigen::MatrixBase<Derived> const & factor) const
    {
        if (!allFinite(factor))
        {
            return std::unexpected{GaussianError::NonFinite};
        }
        return internal::triangularRootOf(factor);
    }
};

/** \ingroup Distributions
 * Pivoted axes: \f$ \Pi^T L D^{1/2} \f$ from Eigen's pivoted
 * [LDL decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition)
 * \f$ P = \Pi^T L D L^T \Pi \f$, with a negative pivot, which rounding can give a singular matrix,
 * taken as zero. It never fails: a semidefinite \f$ P \f$, such as one with a block known exactly,
 * gets an exact root, and an indefinite one a root of its positive semidefinite part. The fallback
 * of the default, cholesky(pivoted()). The pivots follow the diagonal, so the axes jump when two
 * candidates swap, which is why it is not the first choice.
 *
 * Its report is a ClampReport of the pivots taken as zero. A covariance with NaN or infinity gives a
 * root with them.
 *
 * \sa pivoted()
 */
struct Pivoted
{
    static constexpr bool canFail = false;  ///< every positive semidefinite part has a root

    /** The report: the pivots taken as zero. */
    template <typename Scalar>
    using Report = ClampReport;

    /** \returns the pivoted root of \a cov, of which only the lower triangle is read */
    template <typename Derived>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        return root(cov, internal::DiscardReport{});
    }

    /** \returns the pivoted root of \a cov, calling \a sink with the ClampReport */
    template <typename Derived, typename Sink>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov, Sink && sink) const
    {
        using Matrix = internal::SquareMatrixOf<Derived>;
        using Scalar = typename Matrix::Scalar;
        auto const ldlt = Eigen::LDLT<Matrix>{Matrix{cov}};
        auto const pivots = ldlt.vectorD();
        sink(ClampReport{(pivots.array() < Scalar(0)).count()});
        auto const d = pivots.cwiseMax(Scalar(0)).cwiseSqrt().eval();
        Matrix lowerD = ldlt.matrixL();
        lowerD *= d.asDiagonal();
        return ldlt.transpositionsP().transpose() * lowerD;
    }
};

/** \ingroup Distributions
 * Principal axes: \f$ U \Lambda^{1/2} \f$, the eigenvectors of \f$ P = U \Lambda U^T \f$ scaled by the
 * roots of their eigenvalues, largest first, with a negative eigenvalue taken as zero, so that it
 * never fails. Each column's entry of largest magnitude is positive, so that the eigenvectors'
 * arbitrary signs do not move the points. From a factor it takes the singular value decomposition of
 * its triangular factor, \f$ L = U \Sigma V^T \f$, so the covariance is never formed. The points then
 * lie along the directions of greatest and least spread, whatever the order of the coordinates.
 *
 * Its report is a ClampReport of the eigenvalues taken as zero.
 *
 * \sa principal(), \ref Golub2013 "Golub and Van Loan (2013)",
 * [symmetric eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices),
 * [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition#Relation_to_eigenvalue_decomposition)
 */
struct Principal
{
    static constexpr bool canFail = false;  ///< every positive semidefinite part has a root

    /** The report: the eigenvalues taken as zero. */
    template <typename Scalar>
    using Report = ClampReport;

    /** \returns the principal root of \a cov, of which only the lower triangle is read */
    template <typename Derived>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        return root(cov, internal::DiscardReport{});
    }

    /** \returns the principal root of \a cov, calling \a sink with the ClampReport */
    template <typename Derived, typename Sink>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov, Sink && sink) const
    {
        auto const axes = internal::principalAxesOfCovariance(cov);
        sink(ClampReport{axes.clamped});
        return fromAxes<Derived>(axes);
    }

    /** \returns the principal root of \f$ F F^T \f$ for \a factor, from its singular value decomposition */
    template <typename Derived>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    rootFromFactor(Eigen::MatrixBase<Derived> const & factor) const
    {
        return fromAxes<Derived>(internal::principalAxesOfFactor(factor));
    }

private:

    /** \returns \f$ U \Lambda^{1/2} \f$ from \a axes, each column's largest entry made positive */
    template <typename Derived>
    [[nodiscard]] static internal::SquareMatrixOf<Derived>
    fromAxes(internal::PrincipalAxes<Derived> const & axes)
    {
        internal::SquareMatrixOf<Derived> root = axes.vectors * axes.roots.asDiagonal();
        internal::fixColumnSigns(root);
        return root;
    }
};

/** \ingroup Distributions
 * Symmetric axes: the principal square root \f$ P^{1/2} = U \Lambda^{1/2} U^T \f$, the one symmetric
 * positive semidefinite root, with a negative eigenvalue taken as zero, so that it never fails. The
 * one choice that does not depend on the coordinates' basis, rotating with them, and the costliest,
 * an eigendecomposition and a product, which is why it is not the default. From a factor it takes the
 * singular value decomposition of its triangular factor, as Principal does.
 *
 * Its report is a ClampReport of the eigenvalues taken as zero.
 *
 * \sa symmetric(), \ref Golub2013 "Golub and Van Loan (2013)",
 * [principal square root](https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Positive_semidefinite_matrices)
 */
struct Symmetric
{
    static constexpr bool canFail = false;  ///< every positive semidefinite part has a root

    /** The report: the eigenvalues taken as zero. */
    template <typename Scalar>
    using Report = ClampReport;

    /** \returns the principal square root of \a cov, of which only the lower triangle is read */
    template <typename Derived>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        return root(cov, internal::DiscardReport{});
    }

    /** \returns the principal square root of \a cov, calling \a sink with the ClampReport */
    template <typename Derived, typename Sink>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    root(Eigen::MatrixBase<Derived> const & cov, Sink && sink) const
    {
        auto const axes = internal::principalAxesOfCovariance(cov);
        sink(ClampReport{axes.clamped});
        return fromAxes<Derived>(axes);
    }

    /** \returns the principal square root of \f$ F F^T \f$ for \a factor, from its singular value
     * decomposition
     */
    template <typename Derived>
    [[nodiscard]] internal::SquareMatrixOf<Derived>
    rootFromFactor(Eigen::MatrixBase<Derived> const & factor) const
    {
        return fromAxes<Derived>(internal::principalAxesOfFactor(factor));
    }

private:

    /** \returns \f$ U \Lambda^{1/2} U^T \f$ from \a axes */
    template <typename Derived>
    [[nodiscard]] static internal::SquareMatrixOf<Derived>
    fromAxes(internal::PrincipalAxes<Derived> const & axes)
    {
        return axes.vectors * axes.roots.asDiagonal() * axes.vectors.transpose();
    }
};

/** \ingroup Distributions
 * Axes along a direction: a root whose first column is parallel to \f$ d \f$, so that a sigma-point
 * rule puts its first pair of points on the line through the mean along \f$ d \f$, as a split or a
 * probe of the nonlinearity along \f$ d \f$ needs (decisions.md [splitting-hooks]). It is
 * \f$ S = L Q \f$ for the lower Cholesky factor \f$ L \f$ and the
 * [Householder reflection](https://en.wikipedia.org/wiki/Householder_transformation) \f$ Q \f$ that
 * takes \f$ e_1 \f$ to \f$ u = L^{-1} d / \lVert L^{-1} d \rVert \f$, with the first column's sign
 * chosen so that it points along \f$ d \f$, so its first column is
 * \f$ d / \sqrt{d^T P^{-1} d} \f$, a step of one standard deviation along \f$ d \f$.
 *
 * It fails when \f$ L \f$ does, on a covariance that is not positive definite, and when \f$ d \f$ is
 * zero or has no finite length in \f$ P^{-1} \f$, so on a state with a block known exactly it is
 * written <tt>along(d, pivoted())</tt>. From a factor it takes \f$ L \f$ by triangularization, and
 * fails when \f$ d \f$ leaves the range of \f$ L \f$.
 *
 * \tparam Vector  the direction's type, a column vector
 *
 * \sa along()
 */
template <typename Vector>
struct Along
{
    static constexpr bool canFail = true;  ///< a direction outside the covariance's range has no length

    Vector direction;  ///< \f$ d \f$, in the covariance's coordinates

    /** \returns the root of \a cov, of which only the lower triangle is read, whose first column is
     * along the direction
     */
    template <typename Derived>
    [[nodiscard]] std::expected<internal::SquareMatrixOf<Derived>, GaussianError>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        auto const lower = Cholesky{}.root(cov);
        if (!lower)
        {
            return std::unexpected{lower.error()};
        }
        return rotated(*lower);
    }

    /** \returns the root of \f$ F F^T \f$ for \a factor whose first column is along the direction */
    template <typename Derived>
    [[nodiscard]] std::expected<internal::SquareMatrixOf<Derived>, GaussianError>
    rootFromFactor(Eigen::MatrixBase<Derived> const & factor) const
    {
        auto const lower = Cholesky{}.rootFromFactor(factor);
        if (!lower)
        {
            return std::unexpected{lower.error()};
        }
        return rotated(*lower);
    }

private:

    /** \returns the lower Cholesky factor \a lower reflected so that its first column is along the
     * direction, or GaussianError::NotPositiveDefinite if the direction has no finite length in it
     */
    template <typename Matrix>
    [[nodiscard]] std::expected<Matrix, GaussianError>
    rotated(Matrix const & lower) const
    {
        using Scalar = typename Matrix::Scalar;
        using ColumnType = internal::RootColumnOf<Matrix>;
        EMRYS_FILTER_ASSERT(
            direction.size() == lower.rows() && "a direction has a coordinate per coordinate of the covariance"
        );
        ColumnType const whitened =
            lower.template triangularView<Eigen::Lower>().solve(direction.template cast<Scalar>());
        Scalar const length = whitened.norm();
        if (!isFinite(length) || !(length > Scalar(0)))
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        ColumnType const unit = whitened / length;
        // The reflection H = I - 2 v v^T / v^T v with v = e1 - s u takes e1 to s u. The sign s is
        // the one that makes v^T v at least 2, so that no cancellation reaches it, and the first
        // column is then flipped by s so that it points along d.
        Scalar const sign = unit(0) < Scalar(0) ? Scalar(1) : Scalar(-1);
        ColumnType v = -sign * unit;
        v(0) += Scalar(1);
        Matrix root = lower - (Scalar(2) / v.squaredNorm()) * (lower * v) * v.transpose();
        root.col(0) *= sign;
        return root;
    }
};

/** \ingroup Distributions
 * The ridge repair: the Gershgorin ridge of decisions.md [covariance-repair], added to the diagonal of
 * a covariance whose root failed so that the root can be retried. Its margin is \a Relative times
 * \f$ \sqrt{\epsilon} \f$ of the scalar times the larger of the largest diagonal magnitude and the
 * shift that makes the matrix semidefinite, plus \a Absolute: <tt>ridge()</tt>, with no absolute
 * floor, repairs a matrix that is indefinite by rounding and leaves a zero block failing, and a
 * positive \a Absolute repairs that too. Fails with GaussianError::NonFinite on a covariance with
 * NaN or infinity in its lower triangle.
 *
 * Its report is a RidgeReport of the ridge \f$ \delta \f$ added.
 *
 * \tparam Relative  the margin in units of \f$ \sqrt{\epsilon} \f$ of the scale, at least zero
 * \tparam Absolute  the floor added to the margin, at least zero
 *
 * \sa ridge()
 */
template <double Relative = 1.0, double Absolute = 0.0>
struct Ridge
{
    static_assert(Relative >= 0 && Absolute >= 0, "a ridge's margins are at least zero");

    static constexpr double relative = Relative;  ///< \a Relative
    static constexpr double absolute = Absolute;  ///< \a Absolute

    /** The report: the ridge added. */
    template <typename Scalar>
    using Report = RidgeReport<Scalar>;

    /** \returns \a cov with the ridge on its diagonal */
    template <typename Derived>
    [[nodiscard]] std::expected<typename Derived::PlainObject, GaussianError>
    repaired(Eigen::MatrixBase<Derived> const & cov) const
    {
        return internal::ridgedCovariance<Relative, Absolute>(cov);
    }

    /** \returns \a cov with the ridge on its diagonal, calling \a sink with the RidgeReport */
    template <typename Derived, typename Sink>
    [[nodiscard]] std::expected<typename Derived::PlainObject, GaussianError>
    repaired(Eigen::MatrixBase<Derived> const & cov, Sink && sink) const
    {
        auto delta = typename Derived::Scalar(0);
        auto result = internal::ridgedCovariance<Relative, Absolute>(cov, &delta);
        sink(RidgeReport<typename Derived::Scalar>{delta});
        return result;
    }
};

}  // namespace roots

namespace internal
{

/** The report that the axes or repair \a Entry gives over \a Scalar: its \c Report, or roots::NoReport
 * if it names none.
 */
template <typename Entry, typename Scalar>
struct ReportOf
{
    using type = roots::NoReport;  ///< roots::NoReport
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Entry, typename Scalar>
    requires requires { typename Entry::template Report<Scalar>; }
struct ReportOf<Entry, Scalar>
{
    using type = typename Entry::template Report<Scalar>;
};

/// \endcond

/** The report of \a Entry over \a Scalar (ReportOf). */
template <typename Entry, typename Scalar>
using ReportOfT = typename ReportOf<Entry, Scalar>::type;

/** The report that the axes \a A give over \a Scalar when they fail: their \c FailureReport, as a
 * fallback list names one, or their report otherwise.
 */
template <typename A, typename Scalar>
struct FailureReportOf
{
    using type = ReportOfT<A, Scalar>;  ///< the axes' report
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename A, typename Scalar>
    requires requires { typename A::template FailureReport<Scalar>; }
struct FailureReportOf<A, Scalar>
{
    using type = typename A::template FailureReport<Scalar>;
};

/// \endcond

/** The report of the axes \a A over \a Scalar on failure (FailureReportOf). */
template <typename A, typename Scalar>
using FailureReportOfT = typename FailureReportOf<A, Scalar>::type;

/** Whether the entry \a Entry of a fallback list can fail: a repair always can, since the root it
 * retries can fail again, and axes as they say.
 */
template <typename Entry>
[[nodiscard]] consteval bool
entryCanFail()
{
    if constexpr (requires { Entry::canFail; })
    {
        return Entry::canFail;
    }
    else
    {
        return true;
    }
}

/** Whether every entry of \a Fallbacks but the last can fail, so that each is reached. A function
 * rather than a lambda in the static_assert, which MSVC does not evaluate in a class template.
 */
template <typename... Fallbacks>
[[nodiscard]] consteval bool
fallbacksReached()
{
    constexpr bool canFail[] = {entryCanFail<Fallbacks>()..., true};
    for (std::size_t i = 0; i + 1 < sizeof...(Fallbacks); ++i)
    {
        if (!canFail[i])
        {
            return false;
        }
    }
    return true;
}

/** \returns \a axes's root of \a cov, giving \a sink its report if \a axes takes a sink, and
 *           otherwise giving \a sink a roots::NoReport.
 */
template <typename A, typename Derived, typename Sink>
[[nodiscard]] auto
rootWithSink(A const & axes, Eigen::MatrixBase<Derived> const & cov, Sink && sink)
{
    if constexpr (requires { axes.root(cov, sink); })
    {
        return axes.root(cov, sink);
    }
    else
    {
        sink(roots::NoReport{});
        return axes.root(cov);
    }
}

/** \returns \a repair's repair of \a cov, giving \a sink its report as rootWithSink() does. */
template <typename R, typename Derived, typename Sink>
[[nodiscard]] auto
repairWithSink(R const & repair, Eigen::MatrixBase<Derived> const & cov, Sink && sink)
{
    if constexpr (requires { repair.repaired(cov, sink); })
    {
        return repair.repaired(cov, sink);
    }
    else
    {
        sink(roots::NoReport{});
        return repair.repaired(cov);
    }
}

/** \returns \a axes's root of \f$ F F^T \f$ for the factor \a factor: by its own rootFromFactor() if it
 *           has one, and otherwise of the covariance formed.
 */
template <typename A, typename Derived>
[[nodiscard]] auto
rootFromFactor(A const & axes, Eigen::MatrixBase<Derived> const & factor)
{
    if constexpr (requires { axes.rootFromFactor(factor); })
    {
        return axes.rootFromFactor(factor);
    }
    else
    {
        return axes.root(SquareMatrixOf<Derived>{factor * factor.transpose()});
    }
}

/** \returns the root in \a result, a root or a \c std::expected of one, as a \c std::expected. */
template <typename Matrix, typename Result>
[[nodiscard]] std::expected<Matrix, GaussianError>
asExpected(Result && result)
{
    if constexpr (requires { result.has_value(); })
    {
        if (!result)
        {
            return std::unexpected{result.error()};
        }
        return Matrix{std::move(*result)};
    }
    else
    {
        return Matrix{std::forward<Result>(result)};
    }
}

}  // namespace internal

namespace roots
{

/** \ingroup Distributions
 * Axes that can fail, followed by their fallbacks, tried in order (decisions.md [sigma-point-axes]): a repair,
 * which retries \a Head on the covariance it repairs, or other axes, which take the root themselves.
 * The first entry that succeeds gives the root; the list fails, with the last entry's error, only
 * when every entry does, so it can fail exactly when its last entry can. Made by the factories of
 * axes that can fail, given their fallbacks: <tt>cholesky(ridge(), pivoted())</tt>.
 *
 * Refused at compile time, each with a message: a fallback after axes that cannot fail; an entry that
 * is neither axes nor a repair; a fallback after an entry that cannot fail, which is never reached;
 * and a list as an entry, since the list is flat.
 *
 * From a factor, \a Head and each axes after it take their root from the factor; a repair works on
 * the covariance formed from it.
 *
 * Its report, for roots::squareRootWithReport(), is a \c std::variant of the entries' reports whose
 * index is the entry that succeeded, and on failure a \c std::tuple of each entry's report.
 *
 * \tparam Head       axes that can fail
 * \tparam Fallbacks  repairs and axes, the last of which may or may not fail
 */
template <typename Head, typename... Fallbacks>
struct WithFallbacks
{
    static_assert(Axes<Head>, "a fallback list starts with axes");
    static_assert(internal::entryCanFail<Head>(), "axes that cannot fail take no fallback");
    static_assert(((Axes<Fallbacks> || Repair<Fallbacks>) && ...), "a fallback is axes or a repair");
    static_assert(
        (!internal::isFallbackList<Fallbacks> && ...),
        "a fallback list is flat: give a fallback's fallbacks to the list itself"
    );
    static_assert(
        internal::fallbacksReached<Fallbacks...>(),
        "a fallback after axes that cannot fail is never reached"
    );

    /** Whether the list can fail: whether its last entry can. */
    static constexpr bool canFail =
        internal::entryCanFail<std::tuple_element_t<sizeof...(Fallbacks) - 1, std::tuple<Fallbacks...>>>();

    /** The report of the entry that succeeded, at its index. */
    template <typename Scalar>
    using Report = std::variant<internal::ReportOfT<Head, Scalar>, internal::ReportOfT<Fallbacks, Scalar>...>;

    /** The report of every entry, on failure. */
    template <typename Scalar>
    using FailureReport = std::tuple<
        std::optional<internal::ReportOfT<Head, Scalar>>,
        std::optional<internal::ReportOfT<Fallbacks, Scalar>>...>;

    Head head;                           ///< the axes tried first
    std::tuple<Fallbacks...> fallbacks;  ///< the entries tried after it, in order

    /** \returns the root of \a cov, of which only the lower triangle is read, by the first entry
     * that succeeds
     */
    template <typename Derived>
    [[nodiscard]] internal::RootResult<canFail, internal::SquareMatrixOf<Derived>>
    root(Eigen::MatrixBase<Derived> const & cov) const
    {
        return root(cov, internal::DiscardReport{});
    }

    /** \returns the root of \a cov by the first entry that succeeds, calling \a sink with the Report,
     * or on failure with the FailureReport
     */
    template <typename Derived, typename Sink>
    [[nodiscard]] internal::RootResult<canFail, internal::SquareMatrixOf<Derived>>
    root(Eigen::MatrixBase<Derived> const & cov, Sink && sink) const
    {
        using Matrix = internal::SquareMatrixOf<Derived>;
        using Scalar = typename Derived::Scalar;
        auto tried = FailureReport<Scalar>{};
        auto const recordInto = [&tried]<std::size_t I>(std::integral_constant<std::size_t, I>) {
            return [&tried](auto const & report) {
                std::get<I>(tried) = report;
            };
        };
        std::optional<Matrix> found;
        std::size_t used = 0;
        auto error = GaussianError::NotPositiveDefinite;
        auto const keep = [&](std::size_t index, auto && result) {
            auto const root = internal::asExpected<Matrix>(std::forward<decltype(result)>(result));
            if (root)
            {
                found = *root;
                used = index;
            }
            else
            {
                error = root.error();
            }
        };
        keep(0, internal::rootWithSink(head, cov, recordInto(std::integral_constant<std::size_t, 0>{})));
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    if (found)
                    {
                        return;
                    }
                    auto const & entry = std::get<I>(fallbacks);
                    auto const record = recordInto(std::integral_constant<std::size_t, I + 1>{});
                    using Entry = std::tuple_element_t<I, std::tuple<Fallbacks...>>;
                    if constexpr (Repair<Entry>)
                    {
                        auto const fixed = internal::repairWithSink(entry, cov, record);
                        if (!fixed)
                        {
                            error = fixed.error();
                            return;
                        }
                        keep(I + 1, head.root(*fixed));
                    }
                    else
                    {
                        keep(I + 1, internal::rootWithSink(entry, cov, record));
                    }
                }(),
                ...
            );
        }(std::index_sequence_for<Fallbacks...>{});
        if (found)
        {
            sink(successReport<Scalar>(tried, used));
            return std::move(*found);
        }
        if constexpr (canFail)
        {
            sink(tried);
            return std::unexpected{error};
        }
        else
        {
            EMRYS_FILTER_ASSERT(false && "the last fallback cannot fail");
            return Matrix{};
        }
    }

    /** \returns the root of \f$ F F^T \f$ for \a factor by the first entry that succeeds */
    template <typename Derived>
    [[nodiscard]] internal::RootResult<canFail, internal::SquareMatrixOf<Derived>>
    rootFromFactor(Eigen::MatrixBase<Derived> const & factor) const
    {
        using Matrix = internal::SquareMatrixOf<Derived>;
        auto first = internal::asExpected<Matrix>(internal::rootFromFactor(head, factor));
        if (first)
        {
            return std::move(*first);
        }
        std::optional<Matrix> found;
        auto error = first.error();
        std::optional<Matrix> covariance;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    if (found)
                    {
                        return;
                    }
                    auto const & entry = std::get<I>(fallbacks);
                    using Entry = std::tuple_element_t<I, std::tuple<Fallbacks...>>;
                    std::expected<Matrix, GaussianError> root = std::unexpected{error};
                    if constexpr (Repair<Entry>)
                    {
                        if (!covariance)
                        {
                            covariance = Matrix{factor * factor.transpose()};
                        }
                        auto const fixed = entry.repaired(*covariance);
                        if (!fixed)
                        {
                            error = fixed.error();
                            return;
                        }
                        root = internal::asExpected<Matrix>(head.root(*fixed));
                    }
                    else
                    {
                        root = internal::asExpected<Matrix>(internal::rootFromFactor(entry, factor));
                    }
                    if (root)
                    {
                        found = std::move(*root);
                    }
                    else
                    {
                        error = root.error();
                    }
                }(),
                ...
            );
        }(std::index_sequence_for<Fallbacks...>{});
        if (found)
        {
            return std::move(*found);
        }
        if constexpr (canFail)
        {
            return std::unexpected{error};
        }
        else
        {
            EMRYS_FILTER_ASSERT(false && "the last fallback cannot fail");
            return Matrix{};
        }
    }

private:

    /** \returns the report of entry \a used, the one that succeeded, from the reports \a tried */
    template <typename Scalar, std::size_t I = 0>
    [[nodiscard]] static Report<Scalar>
    successReport(FailureReport<Scalar> const & tried, std::size_t used)
    {
        if constexpr (I + 1 < std::tuple_size_v<FailureReport<Scalar>>)
        {
            if (used != I)
            {
                return successReport<Scalar, I + 1>(tried, used);
            }
        }
        EMRYS_FILTER_ASSERT(std::get<I>(tried).has_value());
        return Report<Scalar>{std::in_place_index<I>, *std::get<I>(tried)};
    }
};

}  // namespace roots

namespace internal
{

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Head, typename... Fallbacks>
inline constexpr bool isFallbackList<roots::WithFallbacks<Head, Fallbacks...>>{true};

/// \endcond

/** \returns the root of the covariance \a cov, of which only the lower triangle is read, by
 * roots::DefaultAxes: what every root the library takes outside a sigma-point spread goes through,
 * of a noise, of a kernel for roughening or sampling, of a conditional's given block, so that there
 * is one code path (decisions.md [sigma-point-axes]). It never fails: a semidefinite \a cov, such as a process
 * noise with a noiseless block, gets the pivoted root.
 */
template <typename Derived>
[[nodiscard]] SquareMatrixOf<Derived>
defaultRoot(Eigen::MatrixBase<Derived> const & cov)
{
    return roots::WithFallbacks<roots::Cholesky, roots::Pivoted>{}.root(cov);
}

/** \returns \a head alone, or \a head followed by \a fallbacks as a roots::WithFallbacks. */
template <typename Head, typename... Fallbacks>
[[nodiscard]] constexpr auto
withFallbacks(Head head, Fallbacks... fallbacks)
{
    if constexpr (sizeof...(Fallbacks) == 0)
    {
        return head;
    }
    else
    {
        return roots::WithFallbacks<Head, Fallbacks...>{
            std::move(head),
            std::tuple<Fallbacks...>{std::move(fallbacks)...}
        };
    }
}

}  // namespace internal

namespace roots
{

/** \ingroup Distributions
 * \returns Cholesky axes, followed by \a fallbacks when it fails (WithFallbacks):
 * <tt>cholesky()</tt>, which can fail, or <tt>cholesky(pivoted())</tt>, the default, which cannot
 */
template <typename... Fallbacks>
[[nodiscard]] constexpr auto
cholesky(Fallbacks... fallbacks)
{
    return internal::withFallbacks(Cholesky{}, std::move(fallbacks)...);
}

/** \ingroup Distributions
 * \returns Pivoted axes, which cannot fail and so take no fallback
 */
template <typename... Fallbacks>
[[nodiscard]] constexpr auto
pivoted(Fallbacks... fallbacks)
{
    return internal::withFallbacks(Pivoted{}, std::move(fallbacks)...);
}

/** \ingroup Distributions
 * \returns Principal axes, which cannot fail and so take no fallback
 */
template <typename... Fallbacks>
[[nodiscard]] constexpr auto
principal(Fallbacks... fallbacks)
{
    return internal::withFallbacks(Principal{}, std::move(fallbacks)...);
}

/** \ingroup Distributions
 * \returns Symmetric axes, which cannot fail and so take no fallback
 */
template <typename... Fallbacks>
[[nodiscard]] constexpr auto
symmetric(Fallbacks... fallbacks)
{
    return internal::withFallbacks(Symmetric{}, std::move(fallbacks)...);
}

/** \ingroup Distributions
 * \returns axes along the direction \a direction, a column vector in the covariance's coordinates
 * (Along), followed by \a fallbacks when it fails: <tt>along(d, pivoted())</tt> on a state with a
 * block known exactly
 */
template <typename Derived, typename... Fallbacks>
[[nodiscard]] auto
along(Eigen::MatrixBase<Derived> const & direction, Fallbacks... fallbacks)
{
    static_assert(Derived::ColsAtCompileTime == 1, "along() takes a direction as a column vector");
    return internal::withFallbacks(
        Along<typename Derived::PlainObject>{typename Derived::PlainObject{direction}},
        std::move(fallbacks)...
    );
}

/** \ingroup Distributions
 * \returns the ridge repair with the margins \a Relative and \a Absolute (Ridge):
 * <tt>ridge()</tt> is the ridge of decisions.md [covariance-repair], and <tt>ridge<1.0, 1e-12>()</tt>
 * also repairs a zero block
 */
template <double Relative = 1.0, double Absolute = 0.0>
[[nodiscard]] constexpr Ridge<Relative, Absolute>
ridge()
{
    return {};
}

/** \ingroup Distributions
 * \returns the axes \a axes, which can fail, followed by \a fallbacks (WithFallbacks): what the
 * library's factories do for their own axes, for axes of the user's own,
 * <tt>withFallbacks(MyAxes{}, ridge(), pivoted())</tt>
 */
template <typename Head, typename First, typename... Rest>
[[nodiscard]] constexpr WithFallbacks<Head, First, Rest...>
withFallbacks(Head axes, First first, Rest... rest)
{
    return {std::move(axes), std::tuple<First, Rest...>{std::move(first), std::move(rest)...}};
}

/** \ingroup Distributions
 * The axes every root the library takes uses unless told otherwise, cholesky(pivoted()): the
 * Cholesky factor of a positive definite covariance, and the pivoted root of any other, so it never
 * fails and never repairs (decisions.md [sigma-point-axes]).
 */
using DefaultAxes = WithFallbacks<Cholesky, Pivoted>;

/** \ingroup Distributions
 * \returns a root \f$ S \f$, \f$ S S^T = P \f$, of the covariance \a cov, of which only the lower
 * triangle is read, by \a axes (Axes): the root, or, when \a axes can fail, a \c std::expected of it
 * with the GaussianError of the last entry tried. Square with \a cov's rows and its bounds, so that a
 * bounded covariance allocates nothing.
 *
 * \code
 * auto const s = roots::squareRoot(roots::cholesky(roots::pivoted()), cov);  // a matrix
 * auto const t = roots::squareRoot(roots::cholesky(), cov);  // a std::expected
 * \endcode
 *
 * \sa squareRootWithReport()
 */
template <Axes A, typename Derived>
[[nodiscard]] auto
squareRoot(A const & axes, Eigen::MatrixBase<Derived> const & cov)
{
    EMRYS_FILTER_ASSERT(cov.rows() == cov.cols() && "a covariance must be square");
    return axes.root(cov);
}

/** \ingroup Distributions
 * \brief A root and the report of the axes that took it, from squareRootWithReport().
 *
 * \tparam Matrix  the root's type
 * \tparam Report  the axes' report: for a WithFallbacks, a \c std::variant whose index is the entry
 *                 that succeeded
 */
template <typename Matrix, typename Report>
struct RootWithReport
{
    Matrix root;    ///< \f$ S \f$, \f$ S S^T = P \f$
    Report report;  ///< what the axes did
};

/** \ingroup Distributions
 * \brief Why squareRootWithReport() failed, and what each entry tried did.
 *
 * \tparam Report  the axes' report on failure: for a WithFallbacks, a \c std::tuple of each entry's
 */
template <typename Report>
struct RootFailure
{
    GaussianError error;  ///< the last entry's error
    Report report;        ///< what each entry did
};

/** \ingroup Distributions
 * \returns the root of squareRoot() with a report of what \a axes did (RootWithReport): the pivots or
 * eigenvalues clamped, the ridge added, or a report of the user's own, and for a fallback list which
 * entry succeeded. When \a axes can fail, a \c std::expected of it, whose error is a RootFailure with
 * the error and each entry's report. Transforms and filters carry no reports: a caller diagnosing one
 * calls this on the same covariance with the same axes.
 */
template <Axes A, typename Derived>
[[nodiscard]] auto
squareRootWithReport(A const & axes, Eigen::MatrixBase<Derived> const & cov)
{
    using Scalar = typename Derived::Scalar;
    using Matrix = internal::SquareMatrixOf<Derived>;
    using Report = internal::ReportOfT<A, Scalar>;
    using Failure = internal::FailureReportOfT<A, Scalar>;
    std::optional<Report> report;
    std::optional<Failure> failure;
    auto result = internal::rootWithSink(axes, cov, [&](auto const & given) {
        using Given = std::remove_cvref_t<decltype(given)>;
        if constexpr (std::same_as<Given, Report>)
        {
            report = given;
        }
        if constexpr (std::same_as<Given, Failure>)
        {
            failure = given;
        }
    });
    if constexpr (A::canFail)
    {
        using Result = std::expected<RootWithReport<Matrix, Report>, RootFailure<Failure>>;
        if (!result)
        {
            return Result{std::unexpect, RootFailure<Failure>{result.error(), failure.value_or(Failure{})}};
        }
        return Result{RootWithReport<Matrix, Report>{std::move(*result), report.value_or(Report{})}};
    }
    else
    {
        return RootWithReport<Matrix, Report>{std::move(result), report.value_or(Report{})};
    }
}

}  // namespace roots

}  // namespace Emrys

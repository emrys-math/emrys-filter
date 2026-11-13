#pragma once

/** \file
 * What a transform returns and what a measurement update reads from it (decisions.md
 * [linearization-forms], [innovation-statistics], [innovation-cholesky]): the UncertainInput concept of a
 * transform's uncertain arguments, the linearization forms JacobianForm and RootForm, OutputWithCross, the
 * output's %Gaussian with its linearization in the input marked by withCross(), and Innovation, an
 * observation's residual and its covariance, from which the NIS and the log-likelihood follow. Read
 * Condition.h next.
 */

#include <cmath>
#include <concepts>
#include <expected>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Transforms
 * An argument of transform() that is uncertain: something with a mean, a space, and either a
 * covariance, whose lower triangle \c covarianceLower() gives, or a lower-triangular root
 * \f$ S \f$ of it, \f$ S S^T = \Sigma \f$, which \c root() gives. A GaussianParams, a Gaussian,
 * and a GaussianMarginal give the covariance; a SqrtGaussianParams and a SqrtGaussian give the
 * root, which the sigma-point methods spread their points by as it is (Transform::root()), and
 * whose covariance is formed only where a method needs it. The information forms are not inputs,
 * since their mean fails where the precision is singular: convert one with toSqrt() or
 * toGaussian(), which report that failure.
 */
template <typename T>
concept UncertainInput = requires(T const & input) {
    typename T::Scalar;
    typename T::SpaceType;
    input.mean();
    input.space();
    input.valueDimension();
    input.differenceDimension();
} && (requires(T const & input) { input.covarianceLower(); } || requires(T const & input) { input.root(); });

namespace internal
{

/** \returns the covariance of \a input, an UncertainInput, of which the lower triangle is read:
 * its \c covarianceLower(), or \f$ S S^T \f$ from its \c root() \f$ S \f$ where it has no
 * covarianceLower()
 */
template <UncertainInput Input>
[[nodiscard]] decltype(auto)
covarianceLowerOf(Input const & input)
{
    if constexpr (requires { input.covarianceLower(); })
    {
        return input.covarianceLower();
    }
    else
    {
        using Matrix = std::remove_cvref_t<decltype(input.root())>;
        Matrix result = multiply(asTriangular<Eigen::Lower>(input.root()), input.root().transpose());
        return result;
    }
}

/** \returns \a prior itself where it has a covariance, and otherwise a GaussianParams of its mean
 * and the covariance of its root: what predict() and update() give a step that reads the
 * covariance's blocks, whose result is a GaussianParams in either case
 */
template <UncertainInput Prior>
[[nodiscard]] decltype(auto)
withCovariance(Prior const & prior)
{
    if constexpr (requires { prior.covarianceLower(); })
    {
        return (prior);
    }
    else
    {
        using Result = GaussianParams<typename Prior::SpaceType, typename Prior::Scalar>;
        return Result{
            prior.space(),
            typename Result::VectorType{prior.mean()},
            typename Result::MatrixType{covarianceLowerOf(prior)}
        };
    }
}

}  // namespace internal

/** \ingroup Transforms
 * The Jacobian form of a linearization (decisions.md [linearization-forms]), which methods::linearized() and a
 * Jacobian returned with the value compute (decisions.md [returned-jacobians]): the Jacobian
 * \f$ H \f$ of the output in the marked input at its mean, and the cross-covariance
 * \f$ \Sigma_{xy} = P H^T \f$. OutputWithCross names it. The method fixes the form and no caller
 * chooses it, so it is not a policy and has no factory (decisions.md [policy-factories]).
 */
struct JacobianForm
{};

/** \ingroup Transforms
 * The root form of a linearization (decisions.md [linearization-forms]), which every sigma-point method computes:
 * the root \f$ S \f$ of the marked input's covariance that spread the points, \f$ S S^T = P \f$,
 * and the whitened cross-covariance \f$ \Psi = S^{-1} \Sigma_{xy} = (H S)^T \f$, the weighted sum
 * \f$ \sum_i w_i \xi_i \Delta y_i^T \f$ over the unit points \f$ \xi_i \f$ and the output
 * deviations \f$ \Delta y_i \f$. OutputWithCross names it, and the method fixes it, as
 * JacobianForm's does.
 */
struct RootForm
{};

namespace internal
{

/** Whether \a T is one of the linearization forms, JacobianForm or RootForm. */
template <typename T>
inline constexpr bool isLinearizationForm = std::is_same_v<T, JacobianForm> || std::is_same_v<T, RootForm>;

/** The type of a cross-covariance of an input in \a InputSpace with an output in \a OutputSpace:
 * input differences by output differences.
 */
template <typename InputSpace, typename OutputSpace, typename Scalar>
using CrossMatrixOf = MatrixOf<
    Scalar,
    valueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
    valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
    maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>>;

/** Whether the square \a root is lower triangular with a positive diagonal, as a Cholesky factor
 * is, so that a solve with it is a triangular one.
 */
template <typename Derived>
[[nodiscard]] bool
isInvertibleLowerTriangular(Eigen::MatrixBase<Derived> const & root)
{
    using Scalar = typename Derived::Scalar;
    for (Eigen::Index j = 0; j < root.cols(); ++j)
    {
        if (!(root(j, j) > Scalar(0)))
        {
            return false;
        }
        for (Eigen::Index i = 0; i < j; ++i)
        {
            if (root(i, j) != Scalar(0))
            {
                return false;
            }
        }
    }
    return true;
}

/** \returns a solution \f$ X \f$ of \f$ S^T X = \Psi \f$ for the root \a root, \f$ S \f$, and
 * \a whitened, \f$ \Psi \f$, \f$ S^{-T} \Psi = P^{-1} \Sigma_{xy} \f$ where \f$ S \f$ is invertible, so
 * that \f$ X^T = \Psi^T S^{-1} \f$ is the Jacobian of the root form (decisions.md [linearization-forms]): by a
 * triangular solve when \f$ S \f$ is a Cholesky factor, and otherwise by a pivoted LDLT of
 * \f$ P = S S^T \f$ applied to \f$ \Sigma_{xy} = S \Psi \f$, which allocates nothing on a bounded size.
 * Where a semidefinite input's root has a zero column, whose row of \f$ \Psi \f$ is zero, the LDLT
 * drops the directions \f$ P \f$ does not span, and any solution serves: they differ only in
 * \f$ P \f$'s null space, which \f$ P \f$ and its cross-covariances with the rest of a state map to
 * zero. A solve by the axes that took the root is rejected (decisions.md [linearization-forms]): the root form keeps
 * the root, not the axes, and only a root other than a Cholesky factor would gain.
 */
template <typename Result, typename Root, typename Whitened>
[[nodiscard]] Result
solveWithRoot(Eigen::MatrixBase<Root> const & root, Eigen::MatrixBase<Whitened> const & whitened)
{
    using RootMatrix = typename Root::PlainObject;
    Result result = whitened;
    // MSVC can't form an empty solve.
    if constexpr (Root::RowsAtCompileTime != 0)
    {
        if (isInvertibleLowerTriangular(root))
        {
            root.template triangularView<Eigen::Lower>().transpose().solveInPlace(result);
        }
        else
        {
            auto const ldlt = Eigen::LDLT<RootMatrix>{RootMatrix{multiply(root, root.transpose())}};
            result.noalias() = multiply(root, whitened);
            solveInPlace(ldlt, result);
        }
    }
    return result;
}

}  // namespace internal

/** \ingroup Transforms
 * The result of transform() with an input marked by withCross(): the output's %Gaussian and its
 * linearization in that input, in the \a Form the method computes it, JacobianForm or RootForm
 * (decisions.md [linearization-forms]). condition() conditions the input on an observation of the output.
 *
 * The output covariance is the sum of three parts, each computed as a product of quantities of its
 * own size, so that none is recovered by subtraction: \f$ S_y = H P H^T + N + \Omega \f$, the part
 * the marked input explains (explainedCovariance()), the part the other uncertain inputs explain
 * (inputsCovariance()), such as noise passed to the function, and the part neither explains
 * (unexplainedCovariance()), the function's curvature over the input's spread as the method sees it,
 * which is zero for the Jacobian form. Uncertain inputs are independent, so the joint root is
 * block-diagonal and the root form's \f$ \Psi \f$ splits by input: \f$ H P H^T = \Psi^T \Psi \f$
 * over the marked input's rows, and \f$ N \f$ the same sum over the other inputs' rows.
 *
 * Each form holds what its method computes and derives the rest, as the table below says: the
 * Jacobian form holds \f$ H \f$ and \f$ \Sigma_{xy} \f$, and the root form \f$ S \f$ and
 * \f$ \Psi \f$, from which \f$ \Sigma_{xy} = S \Psi \f$ is a product and \f$ H = \Psi^T S^{-1} \f$
 * a solve.
 *
 * | Asked for | Jacobian form | Root form |
 * |---|---|---|
 * | \f$ H P H^T \f$, \f$ N \f$, \f$ \Omega \f$, and their sum | held | held |
 * | \f$ \Sigma_{xy} \f$, for updates::Standard and updates::Symmetric | held | \f$ S \Psi \f$, a product |
 * | a full state's \f$ P_{xS} H_S^T \f$ from the blocks read | a product | \f$ P_{xS} S^{-T} \Psi \f$, a solve |
 * | \f$ H \f$: information forms, iterated updates, part-state Joseph | held | \f$ \Psi^T S^{-1} \f$, a solve |
 *
 * updates::Joseph over the whole input is \f$ (I - KH) P (I - KH)^T + K R K^T \f$ in the Jacobian
 * form and, since \f$ H S = \Psi^T \f$, \f$ (S - K \Psi^T)(S - K \Psi^T)^T + K R K^T \f$ in the root
 * form: one product of the input's size cubed and no solve, still a sum of two positive
 * semidefinite terms.
 *
 * \tparam InputSpace   space of the marked input
 * \tparam OutputSpace  space of the output
 * \tparam Form         JacobianForm or RootForm
 */
template <typename InputSpace, typename OutputSpace, typename Form, typename Scalar = double>
class OutputWithCross
{
    static_assert(internal::isLinearizationForm<Form>);

    /** Whether the form is the root form. */
    static constexpr bool IsRoot = std::is_same_v<Form, RootForm>;

public:

    /** JacobianForm or RootForm. */
    using FormType = Form;

    /** The output's distribution. */
    using OutputType = GaussianParams<OutputSpace, Scalar>;

    /** An output value. */
    using OutputVector = typename OutputType::VectorType;

    /** A covariance over the output's differences. */
    using OutputMatrix = typename OutputType::MatrixType;

    /** The cross-covariance's type, and \f$ \Psi \f$'s: input differences by output differences. */
    using CrossMatrix = internal::CrossMatrixOf<InputSpace, OutputSpace, Scalar>;

    /** The Jacobian's type: output differences by input differences. */
    using JacobianMatrix = internal::MatrixOf<
        Scalar,
        valueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        valueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<OutputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<InputSpace>>>;

    /** The root's type: over the input's differences. */
    using RootMatrix = internal::MatrixOf<
        Scalar,
        valueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
        valueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<InputSpace>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<InputSpace>>>;

    /** What the form holds beside its cross-covariance: \f$ H \f$ in the Jacobian form, \f$ S \f$ in
     * the root form.
     */
    using FactorMatrix = std::conditional_t<IsRoot, RootMatrix, JacobianMatrix>;

    /** The prediction in \a space with mean \a mean, normalized, and the parts of its linearization:
     * \a factor, \f$ H \f$ or \f$ S \f$ as the form says, \a cross, \f$ \Sigma_{xy} \f$ or
     * \f$ \Psi \f$, and the symmetric \a explained, \a inputs, and \a unexplained, whose sum is the
     * output covariance. Each matrix passed as an rvalue of its stored type is moved. For the
     * library's transforms, which every prediction comes from.
     */
    template <
        MatrixExpression Mean,
        MatrixExpression Factor,
        MatrixExpression Cross,
        MatrixExpression Explained,
        MatrixExpression Inputs,
        MatrixExpression Unexplained>
    OutputWithCross(
        OutputSpace space,
        Mean && mean,
        Factor && factor,
        Cross && cross,
        Explained && explained,
        Inputs && inputs,
        Unexplained && unexplained
    )
        : m_explained{std::forward<Explained>(explained)}
        , m_inputs{std::forward<Inputs>(inputs)}
        , m_unexplained{std::forward<Unexplained>(unexplained)}
        , m_output{std::move(space), std::forward<Mean>(mean), OutputMatrix{m_explained + m_inputs + m_unexplained}}
        , m_factor{std::forward<Factor>(factor)}
        , m_cross{std::forward<Cross>(cross)}
    {
        EMRYS_FILTER_ASSERT(
            m_cross.cols() == m_output.differenceDimension() && m_factor.cols() == m_cross.rows()
            && "the linearization must have a row per input coordinate and a column per output coordinate"
        );
    }

    /** \returns the distribution of the output, with covariance \f$ S_y = H P H^T + N + \Omega \f$ */
    [[nodiscard]] OutputType const &
    output() const &
    {
        return m_output;
    }

    /** \returns the distribution of the output, moved out */
    [[nodiscard]] OutputType
    output() &&
    {
        return std::move(m_output);
    }

    /** \returns \f$ H P H^T \f$, the part of the output covariance that the marked input explains */
    [[nodiscard]] OutputMatrix const &
    explainedCovariance() const
    {
        return m_explained;
    }

    /** \returns \f$ N \f$, the part of the output covariance that the other uncertain inputs explain */
    [[nodiscard]] OutputMatrix const &
    inputsCovariance() const
    {
        return m_inputs;
    }

    /** \returns \f$ \Omega \f$, the part of the output covariance that no input explains: the
     * function's curvature over the inputs' spread as the method sees it, from the residuals of the
     * fit at the points, and zero for the Jacobian form. With a negative sigma-point weight it can
     * be indefinite.
     */
    [[nodiscard]] OutputMatrix const &
    unexplainedCovariance() const
    {
        return m_unexplained;
    }

    /** \returns \f$ \Sigma_{xy} \f$: a row per difference coordinate of the input, a column per
     * difference coordinate of the output
     */
    [[nodiscard]] CrossMatrix const &
    crossCovariance() const
        requires(!IsRoot)
    {
        return m_cross;
    }

    /** \returns \f$ \Sigma_{xy} = S \Psi \f$, formed: a row per difference coordinate of the input, a
     * column per difference coordinate of the output
     */
    [[nodiscard]] CrossMatrix
    crossCovariance() const
        requires IsRoot
    {
        return internal::multiply(m_factor, m_cross);
    }

    /** \returns \f$ H \f$, the Jacobian at the input's mean */
    [[nodiscard]] JacobianMatrix const &
    jacobian() const
        requires(!IsRoot)
    {
        return m_factor;
    }

    /** \returns \f$ H = \Psi^T S^{-1} \f$, the statistical linearization's Jacobian, by a solve with
     * the root
     */
    [[nodiscard]] JacobianMatrix
    jacobian() const
        requires IsRoot
    {
        return internal::solveWithRoot<CrossMatrix>(m_factor, m_cross).transpose();
    }

    /** \returns \f$ S \f$, the root of the marked input's covariance that spread the points */
    [[nodiscard]] RootMatrix const &
    root() const
        requires IsRoot
    {
        return m_factor;
    }

    /** \returns \f$ \Psi = S^{-1} \Sigma_{xy} = (H S)^T \f$, the whitened cross-covariance */
    [[nodiscard]] CrossMatrix const &
    whitenedCross() const
        requires IsRoot
    {
        return m_cross;
    }

private:

    OutputMatrix m_explained;    ///< \f$ H P H^T \f$
    OutputMatrix m_inputs;       ///< \f$ N \f$
    OutputMatrix m_unexplained;  ///< \f$ \Omega \f$
    OutputType m_output;         ///< the output, its covariance the sum of the three
    FactorMatrix m_factor;       ///< \f$ H \f$ or \f$ S \f$
    CrossMatrix m_cross;         ///< \f$ \Sigma_{xy} \f$ or \f$ \Psi \f$
};

namespace internal
{

/** What an innovation's residual \f$ r \f$ and covariance \f$ S \f$ give beside the gain. */
template <typename Scalar>
struct InnovationStatistics
{
    Scalar nis;             ///< the normalized innovation squared, \f$ r^T S^{-1} r \f$
    Scalar logDeterminant;  ///< \f$ \log |S| \f$
};

/** Cholesky-factorizes the innovation covariance \a cov, of which only the lower triangle is read,
 * repairing it by \a R, a Repair or NoRepair, when it is not positive definite: unless \a R is
 * NoRepair, the factorization is retried on the matrix a default-constructed \a R repairs it to,
 * which for roots::Ridge is \a cov with a ridge on its diagonal, and if that succeeds, \a cov is
 * replaced by the repaired matrix, so that the innovation kept with it is the one the gain was taken
 * from (decisions.md [innovation-cholesky]). It is not a root by axes, whose fallback list keeps no
 * repaired matrix, and whose axes that cannot fail would hide a covariance that is not positive
 * definite.
 *
 * \returns the factorization, or GaussianError::NotPositiveDefinite if \a cov does not factorize
 *          and is not repaired, or the repair's error or GaussianError::NotPositiveDefinite, with
 *          \a cov unchanged, if the repair fails
 */
template <typename R, typename Matrix>
[[nodiscard]] std::expected<Eigen::LLT<std::remove_const_t<Matrix>>, GaussianError>
innovationCholesky(Matrix & cov)
{
    auto llt = Eigen::LLT<std::remove_const_t<Matrix>>{cov};
    if (llt.info() == Eigen::Success)
    {
        return llt;
    }
    if constexpr (!std::is_same_v<R, NoRepair>)
    {
        static_assert(!std::is_const_v<Matrix>, "a repaired innovation covariance replaces the one given");
        auto repaired = R{}.repaired(cov);
        if (!repaired)
        {
            return std::unexpected{repaired.error()};
        }
        llt.compute(*repaired);
        if (llt.info() != Eigen::Success)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        cov = std::move(*repaired);
        return llt;
    }
    else
    {
        return std::unexpected{GaussianError::NotPositiveDefinite};
    }
}

/** \returns the NIS and log-determinant of the residual \a residual under the covariance that
 * \a llt factorizes, \f$ S = L L^T \f$: \f$ r^T S^{-1} r = |L^{-1} r|^2 \f$, one forward
 * substitution where a solve takes two, and \f$ \log |S| = 2 \sum_i \log L_{ii} \f$.
 */
template <typename Matrix, typename Residual>
[[nodiscard]] InnovationStatistics<typename Matrix::Scalar>
innovationStatistics(Eigen::LLT<Matrix> const & llt, Eigen::MatrixBase<Residual> const & residual)
{
    using Scalar = typename Matrix::Scalar;
    typename Residual::PlainObject whitened = residual;
    llt.matrixL().solveInPlace(whitened);
    return {whitened.squaredNorm(), Scalar(2) * llt.matrixLLT().diagonal().array().log().sum()};
}

/** Selects the Innovation constructor that makes the covariance where the innovation keeps it,
 * for an update that then conditions on it there (decisions.md [borrowed-intermediates]).
 */
struct InnovationInPlace
{};

/** What an update in place accepts before it writes the state: every innovation. An acceptor is
 * called as <tt>accept(nis, dimension)</tt> once the innovation covariance is factored, and the state
 * is written only if it returns true, so that a policy that rejects a measurement on its NIS, as a
 * Gate does, leaves the state as it was with nothing copied (decisions.md [update-outcomes]).
 */
struct AcceptEvery
{
    /** \returns true */
    template <typename Scalar>
    [[nodiscard]] constexpr bool
    operator()(Scalar /*nis*/, Eigen::Index /*dimension*/) const
    {
        return true;
    }
};

/** Whether \a Accept refuses some innovation, so that an update must compute the NIS to ask it. */
template <typename Accept>
inline constexpr bool acceptsByNis = !std::is_same_v<Accept, AcceptEvery>;

}  // namespace internal

/** \ingroup Transforms
 * \brief The innovation of an update: the residual \f$ r \f$ of the observation from the predicted
 * output, normalized in the output space, and its covariance \f$ S \f$, from which the update's
 * statistics follow (decisions.md [innovation-statistics]).
 *
 * nis() and logLikelihood() are computed from a Cholesky factorization of \f$ S \f$ the first time
 * either is asked for, and kept, unless the update that made the innovation had them already from
 * the factorization that gave its gain, as every update that weights branches by them, or whose
 * filter's policy reads them, does (decisions.md [innovation-statistics]). \f$ S \f$ must be
 * positive definite, as the covariance of every innovation an update returns is (decisions.md
 * [innovation-cholesky]). Only the lower triangle of \f$ S \f$ is read.
 *
 * Keeping the statistics writes through \c const, so the first call to either function is not safe
 * on one innovation from two threads at once. To share an innovation between threads, call nis()
 * or logLikelihood() on it once first; after that, both only read.
 *
 * \tparam Space  the output's space, whose differences the residual is in
 */
template <typename Space, typename Scalar = double>
class Innovation
{
public:

    /// the residual's type
    using VectorType = typename GaussianParams<Space, Scalar>::DifferenceVectorType;

    /// the covariance's type
    using MatrixType = typename GaussianParams<Space, Scalar>::MatrixType;

    /** The innovation with residual \a residual and covariance \a cov, moving either one passed as
     * an rvalue of the stored type, and the statistics \a statistics if the caller has them.
     */
    template <MatrixExpression Residual, MatrixExpression Cov>
    Innovation(
        Residual && residual,
        Cov && cov,
        std::optional<internal::InnovationStatistics<Scalar>> statistics = std::nullopt
    )
        : m_residual{std::forward<Residual>(residual)}
        , m_cov{std::forward<Cov>(cov)}
        , m_statistics{statistics}
    {
        EMRYS_FILTER_ASSERT(
            m_cov.rows() == m_residual.size() && m_cov.cols() == m_residual.size()
            && "the covariance must have a row per residual coordinate"
        );
    }

    /** The innovation with residual \a residual and the covariance \a makeCov returns, made where
     * the innovation keeps it, and the statistics \a condition returns when called with the two as
     * kept: an update conditions on the innovation it returns without copying \f$ S \f$
     * (decisions.md [borrowed-intermediates]), and may replace \f$ S \f$ there with the matrix it repaired it to
     * (decisions.md [innovation-cholesky]). For the library's updates; \a condition returns
     * <tt>std::optional<InnovationStatistics<Scalar>></tt>.
     */
    template <MatrixExpression Residual, typename MakeCov, typename Condition>
    Innovation(internal::InnovationInPlace, Residual && residual, MakeCov const & makeCov, Condition const & condition)
        : m_residual{std::forward<Residual>(residual)}
        , m_cov{makeCov()}
        , m_statistics{condition(std::as_const(m_residual), m_cov)}
    {
        EMRYS_FILTER_ASSERT(
            m_cov.rows() == m_residual.size() && m_cov.cols() == m_residual.size()
            && "the covariance must have a row per residual coordinate"
        );
    }

    /** \returns the residual \f$ r \f$ */
    [[nodiscard]] VectorType const &
    residual() const
    {
        return m_residual;
    }

    /** \returns the covariance \f$ S \f$; only its lower triangle is guaranteed to be read */
    [[nodiscard]] MatrixType const &
    covarianceLower() const
    {
        return m_cov;
    }

    /** \returns the number of coordinates of the residual, the degrees of freedom of the NIS */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return m_residual.size();
    }

    /** \returns the normalized innovation squared \f$ r^T S^{-1} r \f$, chi-squared with
     * differenceDimension() degrees of freedom for a consistent filter
     */
    [[nodiscard]] Scalar
    nis() const
    {
        return statistics().nis;
    }

    /** \returns the log-likelihood of the observation, \f$ \log \mathcal{N}(r; 0, S) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        internal::InnovationStatistics<Scalar> const & known = statistics();
        return Scalar(-0.5) * (Scalar(differenceDimension()) * log2Pi<Scalar>() + known.logDeterminant + known.nis);
    }

private:

    /** \returns the statistics, computing them first if they are not known */
    [[nodiscard]] internal::InnovationStatistics<Scalar> const &
    statistics() const
    {
        if (!m_statistics)
        {
            auto const llt = Eigen::LLT<MatrixType>{m_cov};
            EMRYS_FILTER_ASSERT(llt.info() == Eigen::Success && "the innovation covariance must be positive definite");
            m_statistics = internal::innovationStatistics(llt, m_residual);
        }
        return *m_statistics;
    }

    VectorType m_residual;  ///< the residual
    MatrixType m_cov;       ///< its covariance

    /// the NIS and log-determinant, once computed or given
    mutable std::optional<internal::InnovationStatistics<Scalar>> m_statistics;
};

}  // namespace Emrys

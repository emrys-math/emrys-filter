#pragma once

/** \file
 * PosteriorCramerRaoBound: the Fisher information of the state along true trajectories, whose
 * inverse bounds the mean squared error of any filter run on them (decisions.md [cramer-rao-bound]). Its steps
 * take the true states that simulate() draws, as nees() and NeesMonitor do. Read Consistency.h
 * next.
 */

#include <expected>
#include <limits>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Cholesky.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** The sums over true states that one step of a PosteriorCramerRaoBound averages: of a matrix
 * per state, and whether every state's matrix could be formed.
 */
template <typename Matrix>
struct InfoSums
{
    Matrix first;       ///< the sum of the first matrix, such as \f$ L^{-1} F \f$ with \f$ L L^T = Q \f$
    Matrix second;      ///< the sum of the second matrix, such as \f$ F^T Q^{-1} F \f$
    bool valid = true;  ///< whether every state's terms were formed
};

/** \returns \a a and \a b added: the combination of two ranges' InfoSums. */
template <typename Matrix>
[[nodiscard]] InfoSums<Matrix>
addInfoSums(InfoSums<Matrix> a, InfoSums<Matrix> const & b)
{
    a.first += b.first;
    a.second += b.second;
    a.valid = a.valid && b.valid;
    return a;
}

}  // namespace internal

/** \ingroup Models
 * \brief The posterior Cramér-Rao bound along true trajectories: the Fisher information \f$ J_k \f$
 * of the state, whose inverse bounds the mean squared error of any estimator of \f$ x_k \f$.
 *
 * This is the posterior, or Van Trees, form of the
 * [Cramér-Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound): for any
 * filter, \f$ \mathrm{E}[(\hat x_k \boxminus x_k)(\hat x_k \boxminus x_k)^T] \ge J_k^{-1} \f$, the
 * expectation taken over the true trajectories and their measurements. Compare a filter's mean
 * squared error over Monte Carlo runs with covariance(), and its own covariance with NEES (see
 * NeesMonitor): the bound says how far the error is from the best possible, NEES whether the
 * filter knows its error.
 *
 * It starts from the prior, \f$ J_0 = P_0^{-1} \f$ (create()), or from an information matrix,
 * zero for a state that nothing is known about. Each step is given the true states at its time
 * as the columns of a matrix, a SampleSet: one column for one trajectory, or one per Monte Carlo
 * run, each drawn by simulate() from its own start, so that the bound takes the expectations over
 * them. predict() and update() are the recursion of \ref Tichavsky1998 "Tichavský, Muravchik, and
 * Nehorai (1998)", Proposition 1, eq. (21), with the matrices of eqs. (34) to (36) for additive
 * %Gaussian noise, split in two since \f$ D^{22} \f$ is a sum:
 * \f[ J \leftarrow Q^{-1} - Q^{-1} \bar F (J + D^{11})^{-1} \bar F^T Q^{-1}, \qquad
 *     J \leftarrow J + \mathrm{E}[H^T R^{-1} H], \f]
 * with \f$ \bar F = \mathrm{E}[F] \f$ and \f$ D^{11} = \mathrm{E}[F^T Q^{-1} F] \f$, each
 * expectation the mean over the columns. With one true state and no expectation the prediction
 * is \f$ (Q + F J^{-1} F^T)^{-1} \f$, an extended Kalman filter's covariance recursion with every
 * Jacobian taken at the truth, and for an affine model the Kalman filter's covariance itself.
 *
 * The Jacobians are the Linearized method's at each true state, whatever method a model asks for,
 * so a function that writes its Jacobian (writesJacobians()) is called once per state. On a space
 * with its own plus() and minus() they are in its tangent space, and the bound is that of
 * \ref Chahbazian2024 "Chahbazian et al. (2024)", Proposition 2 and eqs. (68) to (72), for noise
 * that enters as \f$ f(x) \boxplus w \f$, as simulate() draws it. The paper takes \f$ F \f$ at
 * the pair of true states, as the derivative of \f$ x_{k+1} \boxminus f(x_k \boxplus \epsilon) \f$;
 * this takes it at \f$ f(x_k) \f$, which is the same on Linear and agrees to first order in the
 * noise elsewhere, the order of the paper's own truncation.
 *
 * The process noise \f$ Q \f$ must be positive definite: a noiseless block, such as a Constant
 * one, makes the transition density singular, which is Section IV of the paper and not supported
 * yet. A model's inputs are the true ones, as simulate() takes them. The measurements are a
 * measurement model's (see measurementModel()), with no observation, since the bound averages
 * over the measurements the model would draw. Its noise must be additive and %Gaussian
 * (AdditiveNoise of a Gaussian), with \f$ R \f$ positive definite.
 *
 * \code
 * auto bound = PosteriorCramerRaoBound<State>::create(initial);   // J = P0^-1
 * // truths: a column per Monte Carlo run, advanced by simulate() between the steps
 * bound->predict(model, truths, dt);
 * bound->update(truths, rangeSensor);
 * auto const floor = bound->covariance();   // compare with the filters' mean squared error
 * \endcode
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 *
 * \sa simulate(), NeesMonitor
 */
template <Space Space_, RealScalar Scalar_ = double>
class PosteriorCramerRaoBound
{
public:

    using Scalar = Scalar_;                                            ///< the scalar type
    using SpaceType = Space_;                                          ///< the state space
    using VectorType = typename Moments<Space_, Scalar_>::VectorType;  ///< a state value
    using MatrixType = typename Moments<Space_, Scalar_>::MatrixType;  ///< \f$ J \f$, over differences

    /** The bound in \a space with the information matrix \a information, symmetric and positive
     * semidefinite over the space's differences: zero for a state that nothing is known about.
     */
    template <MatrixExpression Info>
    PosteriorCramerRaoBound(Space_ space, Info && info)
        : m_space{std::move(space)}
        , m_info{std::forward<Info>(info)}
        , m_scale{m_info.diagonal()}
    {
        EMRYS_FILTER_ASSERT(
            m_info.rows() == differenceDimension(m_space) && m_info.cols() == m_info.rows()
            && "the information matrix must have a row and a column per difference coordinate"
        );
    }

    /** \returns the bound whose information is the inverse of the covariance of \a prior,
     *          \f$ J_0 = P_0^{-1} \f$, or GaussianError::NotPositiveDefinite where the covariance
     *          has no inverse
     */
    [[nodiscard]] static std::expected<PosteriorCramerRaoBound, GaussianError>
    create(GaussianParams<Space_, Scalar_> const & prior)
    {
        auto const cholesky =
            CholeskyFactorization<MatrixType>{internal::symmetricFromLower<MatrixType>(prior.covarianceLower())};
        if (!cholesky.isPositiveDefinite())
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        return PosteriorCramerRaoBound{prior.space(), inverseOf(cholesky)};
    }

    /** \returns the state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the Fisher information \f$ J \f$, over the space's differences */
    [[nodiscard]] MatrixType const &
    info() const
    {
        return m_info;
    }

    /** \returns the bound itself, \f$ J^{-1} \f$, the least covariance of any estimator's error,
     *          or GaussianError::Singular while the information leaves a direction unknown.
     *
     * A direction is unknown where its information is within rounding of zero, not only where it
     * is exactly zero: from no information a prediction is
     * \f$ Q^{-1} - Q^{-1} F (F^T Q^{-1} F)^{-1} F^T Q^{-1} = 0 \f$, which rounding leaves at about
     * \f$ \epsilon \f$ times the terms subtracted, and that is no information. So the bound keeps
     * the size \f$ s_i \f$ of the terms that formed coordinate \f$ i \f$ of \f$ J \f$, the
     * diagonal of \f$ Q^{-1} \f$ after a prediction, with that of each update's
     * \f$ \mathrm{E}[H^T R^{-1} H] \f$ added, and a direction is unknown unless
     * \f$ J - \tau\, \mathrm{diag}(s) \f$ is positive definite, \f$ \tau = 8 n \epsilon \f$ for
     * \f$ n \f$ coordinates: \f$ J \f$'s least eigenvalue, with each coordinate scaled by its
     * \f$ s_i \f$, must exceed \f$ \tau \f$. Scaling by coordinate keeps the test the same in any
     * units. For a constant-velocity state from no information, over step sizes from 0.01 to 100
     * and noise densities from 1e-6 to 1e4, the rounding left at most \f$ 2.5 \epsilon \f$ and the
     * least information a second fix gave was \f$ 233 \epsilon \f$
     * (`PosteriorCramerRaoBound.TellsRoundingFromInfo`).
     */
    [[nodiscard]] std::expected<MatrixType, GaussianError>
    covariance() const
    {
        auto const cholesky = CholeskyFactorization<MatrixType>{m_info};
        if (!cholesky.isPositiveDefinite())
        {
            return std::unexpected{GaussianError::Singular};
        }
        Scalar_ const tolerance = Scalar_(8) * Scalar_(m_info.rows()) * std::numeric_limits<Scalar_>::epsilon();
        MatrixType margin = m_info;
        margin.diagonal() -= tolerance * m_scale;
        if (!CholeskyFactorization<MatrixType>{margin}.isPositiveDefinite())
        {
            return std::unexpected{GaussianError::Singular};
        }
        return inverseOf(cholesky);
    }

    /** Moves the bound forward by \a dt, any TimeStep (see \ref Time), through the process model
     * \a model with the true inputs \a inputs, from the true states \a truths, a column each:
     * \f$ J \leftarrow Q^{-1} - Q^{-1} \bar F (J + D^{11})^{-1} \bar F^T Q^{-1} \f$, with \a model's
     * Jacobian \f$ F \f$ at each column. A model that gives its transition matrix is not
     * differentiated, and its truths are not read.
     *
     * \returns nothing, or GaussianError::NotPositiveDefinite where \f$ Q \f$ is not positive
     *          definite, and GaussianError::Singular where \f$ J + D^{11} \f$ has no inverse; the
     *          bound is then unchanged
     */
    template <ProcessModel Model, typename Truths, typename... Inputs>
        requires(!ExecutionPolicy<Model>)
    [[nodiscard]] std::expected<void, GaussianError>
    predict(
        Model const & model,
        Eigen::MatrixBase<Truths> const & truths,
        TimeStep<Scalar_> auto const & dt,
        Inputs const &... inputs
    )
    {
        return predict(execution::sequential, model, truths, dt, inputs...);
    }

    /** predict(model, truths, dt, inputs...) under \a policy: the Jacobians at the columns are
     * taken as a reduction (see \ref Execution), so the result depends on the count of truths but
     * not on how the work was scheduled, and with fewer than internal::chunkSize truths it is
     * Sequential's.
     */
    template <ExecutionPolicy P, ProcessModel Model, typename Truths, typename... Inputs>
    [[nodiscard]] std::expected<void, GaussianError>
    predict(
        P && policy,
        Model const & model,
        Eigen::MatrixBase<Truths> const & truths,
        TimeStep<Scalar_> auto const & dt,
        Inputs const &... inputs
    )
    {
        static_assert(
            (!UncertainInput<Inputs> && ...),
            "the bound takes a model's true inputs, as simulate() does; an uncertain input adds noise that the "
            "transition density of the bound does not model (decisions.md [cramer-rao-bound])"
        );
        internal::assertPartitions(model, m_space);
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        auto const call = internal::modelWith(std::cref(model), inputs...);
        auto const resolved = internal::resolveProcess(call, m_space);
        using R = std::remove_cvref_t<decltype(resolved)>;
        auto const noise = internal::symmetricFromLower<MatrixType>(resolved.template noise<Scalar_>(step));
        if constexpr (R::Affine && R::ReturnsJacobian && R::InputCount == 0)
        {
            MatrixType const transition = resolved.template transition<Scalar_>(step);
            return predictFrom(noise, [&](auto const & whiten) {
                internal::InfoSums<MatrixType> sums{whiten(transition), MatrixType{}, true};
                sums.second.noalias() = internal::multiply(sums.first.transpose(), sums.first);
                return sums;
            });
        }
        else
        {
            assertTruths(truths);
            auto const f = [&](auto const & x) {
                return resolved.propagate(x, step);
            };
            using Input = internal::PointInput<Space_, Scalar_>;
            using Problem = internal::Transform<decltype(f) const, Space_, Input>;
            Eigen::Index const count = truths.cols();
            return predictFrom(noise, [&](auto const & whiten) {
                return averageOver(
                    std::forward<P>(policy),
                    count,
                    [&](Eigen::Index i, internal::InfoSums<MatrixType> & sums) {
                        // The problem holds its inputs by reference, so the point is named first.
                        auto const point = Input{m_space, VectorType{truths.col(i)}};
                        auto const problem = Problem{f, m_space, point};
                        auto const linearized = internal::linearizeAt(methods::linearized(), problem);
                        MatrixType const whitened = whiten(linearized.jacobian.leftCols(differenceDimension(m_space)));
                        sums.first += whitened;
                        sums.second.noalias() += internal::multiply(whitened.transpose(), whitened);
                    }
                );
            });
        }
    }

    /** predict() over the step of the fixed-step model \a model, or of fixedStep()'s result, with
     * its process noise.
     */
    template <typename Model, typename Truths, typename... Inputs>
        requires(
            !ExecutionPolicy<Model> && (isFixedStep<Model> || isUnboundStep<Model>)
            && requires(Model const & m) { m.model(); }
        )
    [[nodiscard]] std::expected<void, GaussianError>
    predict(Model const & model, Eigen::MatrixBase<Truths> const & truths, Inputs const &... inputs)
    {
        return predict(execution::sequential, model.model(), truths, model.timeStep(), inputs...);
    }

    /** predict(model, truths, inputs...) over a fixed-step model's step under \a policy. */
    template <ExecutionPolicy P, typename Model, typename Truths, typename... Inputs>
        requires((isFixedStep<Model> || isUnboundStep<Model>) && requires(Model const & m) { m.model(); })
    [[nodiscard]] std::expected<void, GaussianError>
    predict(P && policy, Model const & model, Eigen::MatrixBase<Truths> const & truths, Inputs const &... inputs)
    {
        return predict(std::forward<P>(policy), model.model(), truths, model.timeStep(), inputs...);
    }

    /** Adds the information that a measurement at the true states \a truths, a column each,
     * carries: \f$ J \leftarrow J + \mathrm{E}[H^T R^{-1} H] \f$, with \f$ H \f$ the Jacobian of
     * the measurement model \a model's function at each column, and the call's \a args the sure
     * inputs the model reads, as simulate() takes them. The model's noise must be additive and
     * %Gaussian (additiveNoise() of a GaussianParams), with \f$ R \f$ positive definite. Several
     * measurements at one time are several updates.
     *
     * \returns nothing, or GaussianError::NotPositiveDefinite where \f$ R \f$ is not positive
     *          definite; the bound is then unchanged
     */
    template <typename Truths, MeasurementModel M, typename... Args>
    [[nodiscard]] std::expected<void, GaussianError>
    update(Eigen::MatrixBase<Truths> const & truths, M const & model, Args const &... args)
    {
        return update(execution::sequential, truths, model, args...);
    }

    /** update(truths, model, args...) under \a policy: a reduction over the columns, as in predict(). */
    template <ExecutionPolicy P, typename Truths, MeasurementModel M, typename... Args>
    [[nodiscard]] std::expected<void, GaussianError>
    update(P && policy, Eigen::MatrixBase<Truths> const & truths, M const & model, Args const &... args)
    {
        static_assert(
            (!UncertainInput<Args> && ...),
            "the bound takes a model's true inputs, as simulate() does; an uncertain input adds noise that the "
            "measurement density of the bound does not model (decisions.md [cramer-rao-bound])"
        );
        assertTruths(truths);
        auto const observed = internal::modelWith(std::cref(model), args...);
        auto const resolved = internal::resolveMeasurement(observed, m_space);
        using R = std::remove_cvref_t<decltype(resolved)>;
        static_assert(
            R::HasAdditiveNoise && !R::MixtureNoise,
            "the bound takes a measurement model with additive Gaussian noise, whose likelihood is the Gaussian of "
            "Tichavsky et al. (1998), eq. (33) (decisions.md [cramer-rao-bound])"
        );
        auto const sums = averageOver(
            std::forward<P>(policy),
            truths.cols(),
            [&](Eigen::Index i, internal::InfoSums<MatrixType> & partial) {
                auto const linearization = internal::linearizeMeasurementAtPoint<Scalar_>(
                    m_space,
                    VectorType{truths.col(i)},
                    resolved,
                    methods::linearized()
                );
                using NoiseMatrix = typename std::remove_cvref_t<decltype(linearization)>::NoiseMatrix;
                auto const cholesky = CholeskyFactorization<NoiseMatrix>{linearization.noise};
                if (!cholesky.isPositiveDefinite())
                {
                    partial.valid = false;
                    return;
                }
                using JacobianMatrix = typename std::remove_cvref_t<decltype(linearization)>::JacobianMatrix;
                JacobianMatrix whitened = linearization.jacobian;
                cholesky.whitenInPlace(whitened);
                partial.second.noalias() += internal::multiply(whitened.transpose(), whitened);
            }
        );
        if (!sums.valid)
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        m_info += sums.second;
        m_info = symmetric(std::move(m_info));
        m_scale += sums.second.diagonal();
        return {};
    }

private:

    /** \returns \a matrix made exactly symmetric, the mean of it and its transpose */
    [[nodiscard]] static MatrixType
    symmetric(MatrixType matrix)
    {
        MatrixType result = Scalar_(0.5) * (matrix + matrix.transpose());
        return result;
    }

    /** Asserts that \a truths has a column per true state, each a value in the space. */
    template <typename Truths>
    void
    assertTruths(Eigen::MatrixBase<Truths> const & truths) const
    {
        EMRYS_FILTER_ASSERT(truths.cols() > 0 && "the bound takes at least one true state");
        EMRYS_FILTER_ASSERT(truths.rows() == valueDimension(m_space) && "a true state does not fit the space");
    }

    /** \returns the InfoSums of \a addTerms(i, sums) over the \a count columns under
     *          \a policy, divided by the count: each expectation of the step
     */
    template <ExecutionPolicy P, typename AddTerms>
    [[nodiscard]] internal::InfoSums<MatrixType>
    averageOver(P && policy, Eigen::Index count, AddTerms const & addTerms) const
    {
        Eigen::Index const n = differenceDimension(m_space);
        auto sums = internal::reduceRanges(
            std::forward<P>(policy),
            count,
            [&](Eigen::Index begin, Eigen::Index length) {
                auto range = internal::InfoSums<MatrixType>{MatrixType::Zero(n, n), MatrixType::Zero(n, n), true};
                for (Eigen::Index i = begin; i < begin + length; ++i)
                {
                    addTerms(i, range);
                }
                return range;
            },
            internal::addInfoSums<MatrixType>
        );
        sums.first /= Scalar_(count);
        sums.second /= Scalar_(count);
        return sums;
    }

    /** \returns \f$ A^{-1} \f$ from the factorization \a cholesky of \f$ A \f$, as
     *          \f$ (L^{-1})^T L^{-1} \f$, exactly symmetric
     */
    [[nodiscard]] static MatrixType
    inverseOf(CholeskyFactorization<MatrixType> const & cholesky)
    {
        MatrixType root = MatrixType::Identity(cholesky.rows(), cholesky.rows());
        cholesky.whitenInPlace(root);
        MatrixType result = internal::multiply(root.transpose(), root);
        return symmetric(std::move(result));
    }

    /** The prediction from the process noise \a noise and \a average(whiten), with \a whiten(A)
     * the whitening \f$ L^{-1} A \f$ by the Cholesky factor \f$ L L^T = Q \f$, which gives the
     * whitened mean Jacobian \f$ \bar W = L^{-1} \bar F \f$ and \f$ D^{11} \f$:
     * \f$ J \leftarrow Q^{-1} - Q^{-1} \bar F (J + D^{11})^{-1} \bar F^T Q^{-1}
     *     = L^{-T} (I - V^T V) L^{-1} \f$, \f$ V = C^{-1} \bar W^T \f$ for the Cholesky factor
     * \f$ C C^T = J + D^{11} \f$. Every term is formed through the one whitening, \f$ Q^{-1} \f$
     * too, so that where they cancel they are rounded alike.
     */
    template <typename Average>
    [[nodiscard]] std::expected<void, GaussianError>
    predictFrom(MatrixType const & noise, Average const & average)
    {
        auto const noiseCholesky = CholeskyFactorization<MatrixType>{noise};
        if (!noiseCholesky.isPositiveDefinite())
        {
            return std::unexpected{GaussianError::NotPositiveDefinite};
        }
        auto const whiten = [&](MatrixType const & a) {
            MatrixType result = a;
            noiseCholesky.whitenInPlace(result);
            return result;
        };
        internal::InfoSums<MatrixType> const sums = average(whiten);
        auto const combined = CholeskyFactorization<MatrixType>{MatrixType{m_info + sums.second}};
        if (!combined.isPositiveDefinite())
        {
            return std::unexpected{GaussianError::Singular};
        }
        Eigen::Index const n = m_info.rows();
        MatrixType projected = sums.first.transpose();
        combined.whitenInPlace(projected);
        MatrixType kept = MatrixType::Identity(n, n);
        kept.noalias() -= internal::multiply(projected.transpose(), projected);
        MatrixType root = MatrixType::Identity(n, n);
        noiseCholesky.whitenInPlace(root);
        MatrixType const info = internal::multiply(root.transpose(), MatrixType{internal::multiply(kept, root)});
        m_info = symmetric(info);
        m_scale = root.colwise().squaredNorm().transpose();
        return {};
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the state space
    MatrixType m_info;                              ///< \f$ J \f$
    /// the size of the terms that formed each coordinate of \f$ J \f$, which covariance() compares
    /// its information with
    typename Moments<Space_, Scalar_>::DifferenceVectorType m_scale;
};

}  // namespace Emrys

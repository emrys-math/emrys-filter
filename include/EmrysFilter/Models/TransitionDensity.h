#pragma once

/** \file
 * TransitionDensity and transitionDensity(): the density \f$ p(x_{k+1} \mid x_k) \f$ of a process
 * model over a step, which a particle smoother's backward simulation and a move of a resampled
 * particle weigh states by (roadmap step 17zh). It is the density of the state simulate() draws.
 * Read Simulation.h next.
 */

#include <limits>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Models
 * \brief The density of a process model's next state given its present one over a fixed step,
 * \f$ p(x_{k+1} \mid x_k) \f$: the density of the state that simulate() draws.
 *
 * A process model's step is \f$ x_{k+1} = f(x_k) \boxplus w \f$ with
 * \f$ w \sim \mathcal{N}(0, Q) \f$, so the density is the %Gaussian of the residual
 * \f$ d = x_{k+1} \boxminus f(x_k) \f$ under \f$ Q \f$, in the tangent space at \f$ f(x_k) \f$
 * (decisions.md [tangent-gaussians]):
 * \f[ \log p(x_{k+1} \mid x_k) = -\tfrac12 d^T Q^{+} d - \tfrac12 \log \operatorname{pdet}(2 \pi Q). \f]
 *
 * \f$ Q \f$ is often singular: a Constant block has no noise, and noise that enters through fewer
 * directions than the state has, such as one acceleration driving a position and a velocity,
 * reaches only those directions. The density is then the
 * [degenerate](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Degenerate_case)
 * %Gaussian's: a density on the directions the noise reaches, with the pseudo-inverse and the
 * pseudo-determinant of \f$ Q \f$, taken with respect to the measure on that subspace, and zero,
 * a log-density of \f$ -\infty \f$, at a next state that the noise cannot reach from
 * \f$ f(x_k) \f$. rank() is the number of directions reached. With no noise at all the density is
 * a point mass: zero at \f$ f(x_k) \f$ in the logarithm, and \f$ -\infty \f$ elsewhere. Densities
 * over one step compare with each other, which is what weighing the states of one time needs; a
 * density of one rank does not compare with a density of another.
 *
 * It factorizes \f$ Q \f$ once, when it is made, as a particle filter's prediction does
 * (internal::defaultRoot()): by Cholesky where \f$ Q \f$ is positive definite, and otherwise by
 * the pivoted \f$ L D L^T \f$ factorization, whose pivots that are not positive mark the
 * directions not reached. So a state that a particle filter's prediction or simulate() drew has a
 * finite density from the state it was drawn from. Rounding moves a drawn state off the reached
 * directions by a few units in its last place, so a next state counts as reached when its
 * residual along each direction not reached is within \f$ 16 n \epsilon \f$ of zero relative to
 * the larger of the two states' greatest coordinates, for \f$ n \f$ difference coordinates.
 *
 * The call's inputs are sure ones, <tt>sure<Tag>(value)</tt>, as simulate() takes the true
 * inputs. An uncertain or a sampled input passes noise through the function, and the density of
 * that has no closed form, so it is refused at compile time.
 *
 * \code
 * auto const transition = transitionDensity(space, model, dt);
 * double const logDensity = transition.logDensity(from, to);
 *
 * // Many next states from one present state: the function is evaluated once.
 * auto const mean = transition.mean(from);
 * double const first = transition.logDensityAbout(mean, to);
 * \endcode
 *
 * It holds a copy of the model and the inputs, and the model resolved against the space, which
 * refers to them: a copy resolves the model again, and a density is not assigned.
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Scalar_  scalar type, \c double by default
 * \tparam Model_   the process model, which partitions the space
 * \tparam Inputs   the call's sure inputs
 *
 * \sa transitionDensity(), simulate()
 */
template <Space Space_, RealScalar Scalar_, typename Model_, typename... Inputs>
class TransitionDensity
{
    static_assert(
        ((!UncertainInput<Inputs> && !SampledInput<Inputs>) && ...),
        "a transition density takes a model's sure inputs, sure<Tag>(value), as simulate() takes the true ones: an "
        "uncertain or a sampled input passes noise through the function, and the density of that has no closed form"
    );

    /// the model with the call's inputs, as a prediction reads a model
    using Call = decltype(internal::modelWith(std::declval<Model_ const &>(), std::declval<Inputs const &>()...));

public:

    using Scalar = Scalar_;                                            ///< the scalar type
    using SpaceType = Space_;                                          ///< the state space
    using ModelType = Model_;                                          ///< the process model
    using VectorType = typename Moments<Space_, Scalar_>::VectorType;  ///< a state value
    using MatrixType = typename Moments<Space_, Scalar_>::MatrixType;  ///< a matrix over the state's differences

    /// a difference of two state values
    using DifferenceVectorType = typename Moments<Space_, Scalar_>::DifferenceVectorType;

    /** The density of \a model's step over \a step in \a space, with the call's sure \a inputs.
     * The process noise over the step is factorized here, once.
     */
    TransitionDensity(Space_ space, Model_ const & model, Scalar_ step, Inputs const &... inputs)
        : m_space{std::move(space)}
        , m_call{internal::modelWith(model, inputs...)}
        , m_resolved{internal::resolveProcess(m_call, m_space)}
        , m_step{step}
    {
        internal::assertPartitions(model, m_space);
        factorize(m_resolved.template noise<Scalar_>(m_step));
    }

    /** A copy of \a other, its model resolved again against the copy's own space. */
    TransitionDensity(TransitionDensity const & other)
        : m_space{other.m_space}
        , m_call{other.m_call}
        , m_resolved{internal::resolveProcess(m_call, m_space)}
        , m_step{other.m_step}
        , m_whitening{other.m_whitening}
        , m_rank{other.m_rank}
        , m_logNormalizer{other.m_logNormalizer}
    {}

    /** \returns the state space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the number of directions the process noise reaches, the rank of \f$ Q \f$: the
     * state's difference dimension when \f$ Q \f$ is positive definite, and zero for a model with
     * no noise
     */
    [[nodiscard]] Eigen::Index
    rank() const
    {
        return m_rank;
    }

    /** \returns \f$ f(x_k) \f$ for the present state \a from: the model's step without its noise,
     * normalized, which the density of the next state is centered on
     */
    template <typename From>
    [[nodiscard]] VectorType
    mean(Eigen::MatrixBase<From> const & from) const
    {
        EMRYS_FILTER_ASSERT(from.size() == valueDimension(m_space) && "the state does not fit the space");
        VectorType moved{internal::valueOf(m_resolved.propagate(VectorType{from}, m_step))};
        Emrys::normalizeValues(m_space, moved);
        return moved;
    }

    /** \returns \f$ \log p(x_{k+1} \mid x_k) \f$ for the present state \a from and the next state
     * \a to, or \f$ -\infty \f$ where the noise cannot reach \a to from \f$ f(x_k) \f$
     */
    template <typename From, typename To>
    [[nodiscard]] Scalar_
    logDensity(Eigen::MatrixBase<From> const & from, Eigen::MatrixBase<To> const & to) const
    {
        return logDensityAbout(mean(from), to);
    }

    /** \returns \f$ \log p(x_{k+1} \mid x_k) \f$ for the next state \a to, given \a mean, the
     * mean() of the present state: logDensity() without evaluating the model's function, for the
     * density of many next states from one present state.
     */
    template <typename Mean, typename To>
    [[nodiscard]] Scalar_
    logDensityAbout(Eigen::MatrixBase<Mean> const & mean, Eigen::MatrixBase<To> const & to) const
    {
        EMRYS_FILTER_ASSERT(mean.size() == valueDimension(m_space) && "the mean does not fit the space");
        EMRYS_FILTER_ASSERT(to.size() == valueDimension(m_space) && "the state does not fit the space");
        VectorType const center{mean};
        VectorType const next{to};
        DifferenceVectorType const residual = Emrys::minus(m_space, next, center);
        DifferenceVectorType const whitened = internal::multiply(m_whitening, residual);
        Eigen::Index const unreached = whitened.size() - m_rank;
        if (unreached > 0)
        {
            Scalar_ const scale = Eigen::numext::maxi(
                next.size() == 0 ? Scalar_(0) : Scalar_(next.cwiseAbs().maxCoeff()),
                center.size() == 0 ? Scalar_(0) : Scalar_(center.cwiseAbs().maxCoeff())
            );
            Scalar_ const tolerance =
                Scalar_(16) * Scalar_(whitened.size()) * Eigen::NumTraits<Scalar_>::epsilon() * scale;
            if ((whitened.tail(unreached).array().abs() > tolerance).any())
            {
                return -Eigen::NumTraits<Scalar_>::infinity();
            }
        }
        return m_logNormalizer - Scalar_(0.5) * whitened.head(m_rank).squaredNorm();
    }

private:

    /** Factorizes the process noise \a noise, of which only the lower triangle is read, into the
     * whitening matrix, the rank, and the normalizer. By Cholesky, \f$ Q = L L^T \f$, the
     * whitening matrix is \f$ L^{-1} \f$. Where that fails, by the pivoted
     * \f$ Q = \Pi^T L D L^T \Pi \f$: with \f$ S = L^{-1} \Pi \f$, a row of \f$ S \f$ whose pivot
     * is positive, divided by the pivot's root, whitens a direction reached, and the rows whose
     * pivots are not positive, which roots::Pivoted takes as zero, give the residual along the
     * directions not reached. The reached rows come first. The pseudo-determinant is that of the
     * root's nonzero columns \f$ F \f$, \f$ \det(F^T F) \f$: the product of the positive pivots
     * times the Gram determinant of their columns of \f$ L \f$.
     */
    void
    factorize(MatrixType const & noise)
    {
        Eigen::Index const n = noise.rows();
        EMRYS_FILTER_ASSERT(internal::lowerFinite(noise) && "the process noise must be finite");
        Scalar_ halfLogPseudoDeterminant = 0;
        auto const llt = Eigen::LLT<MatrixType>{noise};
        if (n == 0 || llt.info() == Eigen::Success)
        {
            m_whitening = MatrixType::Identity(n, n);
            m_rank = n;
            if (n > 0)
            {
                llt.matrixL().solveInPlace(m_whitening);
                MatrixType const lower = llt.matrixL();
                for (Eigen::Index j = 0; j < n; ++j)
                {
                    halfLogPseudoDeterminant += Eigen::numext::log(lower(j, j));
                }
            }
        }
        else
        {
            auto const ldlt = Eigen::LDLT<MatrixType>{noise};
            DifferenceVectorType const pivots = ldlt.vectorD();
            MatrixType const lower = ldlt.matrixL();
            MatrixType solved = MatrixType::Identity(n, n);
            solved = ldlt.transpositionsP() * solved;
            ldlt.matrixL().solveInPlace(solved);
            m_rank = (pivots.array() > Scalar_(0)).count();
            m_whitening.resize(n, n);
            // The Gram matrix of the reached columns of L, in the leading block of an identity, so
            // that its factorization has the size of the others.
            MatrixType gram = MatrixType::Identity(n, n);
            Eigen::Index reached = 0;
            Eigen::Index unreached = m_rank;
            for (Eigen::Index j = 0; j < n; ++j)
            {
                if (pivots(j) > Scalar_(0))
                {
                    Scalar_ const root = Eigen::numext::sqrt(pivots(j));
                    m_whitening.row(reached) = solved.row(j) / root;
                    halfLogPseudoDeterminant += Eigen::numext::log(root);
                    Eigen::Index other = 0;
                    for (Eigen::Index k = 0; k <= j; ++k)
                    {
                        if (pivots(k) > Scalar_(0))
                        {
                            gram(reached, other) = lower.col(j).dot(lower.col(k));
                            gram(other, reached) = gram(reached, other);
                            ++other;
                        }
                    }
                    ++reached;
                }
                else
                {
                    m_whitening.row(unreached) = solved.row(j);
                    ++unreached;
                }
            }
            auto const gramLlt = Eigen::LLT<MatrixType>{gram};
            EMRYS_FILTER_ASSERT(gramLlt.info() == Eigen::Success);
            MatrixType const gramLower = gramLlt.matrixL();
            for (Eigen::Index j = 0; j < m_rank; ++j)
            {
                halfLogPseudoDeterminant += Eigen::numext::log(gramLower(j, j));
            }
        }
        m_logNormalizer = -Scalar_(0.5) * Scalar_(m_rank) * log2Pi<Scalar_>() - halfLogPseudoDeterminant;
    }

    Space_ m_space;                                      ///< the state space
    Call m_call;                                         ///< the model and the call's inputs
    internal::ResolvedProcess<Call, Space_> m_resolved;  ///< the call resolved against the space, once
    Scalar_ m_step;                                      ///< the time step
    MatrixType m_whitening;                              ///< the rows that whiten a residual, then the rest
    Eigen::Index m_rank = 0;                             ///< the directions the noise reaches
    Scalar_ m_logNormalizer = 0;                         ///< the density's logarithm at a zero residual
};

/** \ingroup Models
 * \returns the density of the next state given the present one, under the process model \a model
 * over the time step \a dt in \a space, with the call's sure \a inputs (see TransitionDensity):
 * the density of the state that <tt>simulate(space, x, model, dt, rng, inputs...)</tt> draws.
 *
 * \a dt is any TimeStep, converted to \a Scalar (see \ref Time), \c double unless given:
 * <tt>transitionDensity<float>(space, model, dt)</tt>.
 *
 * \code
 * auto const transition = transitionDensity(space, model, 0.1);
 * double const logDensity = transition.logDensity(from, to);
 * \endcode
 */
template <RealScalar Scalar = double, Space Space_, ProcessModel Model, typename... Inputs>
[[nodiscard]] TransitionDensity<Space_, Scalar, Model, Inputs...>
transitionDensity(Space_ const & space, Model const & model, TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
{
    return TransitionDensity<Space_, Scalar, Model, Inputs...>{
        space,
        model,
        internal::toScalarTime<Scalar>(dt),
        inputs...
    };
}

/** \ingroup Models
 * \returns the density of the next state given the present one over the step of the fixed-step
 * model \a model in \a space, with the call's sure \a inputs:
 * transitionDensity(space, model, dt, inputs...) of the model it steps, over its step and in its
 * scalar.
 */
template <Space Space_, typename Model, typename... Inputs>
    requires isFixedStep<Model>
[[nodiscard]] auto
transitionDensity(Space_ const & space, Model const & model, Inputs const &... inputs)
{
    using Stepped = std::remove_cvref_t<decltype(model.model())>;
    using Scalar = std::remove_cvref_t<decltype(model.timeStep())>;
    return TransitionDensity<Space_, Scalar, Stepped, Inputs...>{space, model.model(), model.timeStep(), inputs...};
}

}  // namespace Emrys

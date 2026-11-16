#pragma once

/** \file
 * The continuous-time chain on a discrete space: ContinuousTimeMarkovChain, built from a generator
 * of rates, from independent chains on the axes of a product, as persistent, or from mean sojourn
 * times and exit weights; its transition over an interval as a MarkovChain, through the matrix
 * exponential; predict() over an interval; and stationary(), its stationary distribution
 * (decisions.md [emissions]). Models/Emission.h's ArrivalInterval observes a Markov-modulated
 * Poisson process through it. Read Statistics.h next.
 */

#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/MatrixFunctions>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Distributions
 * \brief A continuous-time Markov chain on a discrete space: the rate of moving from each value
 * to each other value.
 *
 * A time-homogeneous [continuous-time Markov chain](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain)
 * on a finite set. The generator \f$ Q \f$ has a row per value moved from and a column per value
 * moved to: \f$ Q_{ij} \ge 0 \f$ for \f$ i \ne j \f$ is the rate of jumping from \f$ i \f$ to
 * \f$ j \f$, and each row sums to zero, so \f$ -Q_{ii} \f$ is the rate of leaving \f$ i \f$. Its
 * transition over an interval \f$ \Delta t \f$ is the MarkovChain with
 * \f$ \Pi = e^{Q \Delta t} \f$, which transition() computes with Eigen's matrix exponential, by the
 * scaling and squaring method of \ref Higham2005 "Higham (2005)", and
 * predict() moves a CategoricalParams over the interval exactly, as each process model discretizes its
 * own noise [continuous-time-noise].
 *
 * create() builds a chain from its rates off the diagonal, and fromSojournTimes() from mean
 * sojourn times; both check their values and return a ChainError if they fail to define a chain.
 * The constructor that takes a generator is private, and the public one builds the chain that
 * never moves [checked-factories].
 *
 * A chain on a DiscreteProduct can be built from independent chains on its axes. Its generator is
 * their [Kronecker sum](https://en.wikipedia.org/wiki/Kronecker_sum), the rate of one axis
 * jumping while the others stay, since two independent axes never jump at the same instant:
 *
 * \code
 * ContinuousTimeMarkovChain const maneuver = ContinuousTimeMarkovChain<Maneuver>::persistent(0.1);
 * ContinuousTimeMarkovChain const sensor = ContinuousTimeMarkovChain<Sensor>::persistent(2.0);
 * auto const mode = ContinuousTimeMarkovChain<Mode>::independent(maneuver, sensor);
 * p = predict(p, mode, 0.5);   // p e^{Q/2}
 * \endcode
 *
 * \tparam Space_   the discrete space (see \ref Discrete)
 * \tparam Scalar_  scalar type of the rates, \c double by default
 *
 * \sa MarkovChain, ArrivalInterval, HiddenMarkovFilter
 */
template <DiscreteSpace Space_, RealScalar Scalar_ = double>
class ContinuousTimeMarkovChain
{
public:

    using Scalar = Scalar_;                                               ///< the scalar type of the rates
    using SpaceType = Space_;                                             ///< the discrete space
    static constexpr int CardinalityAtCompileTime = Space_::Cardinality;  ///< the number of values, or Eigen::Dynamic
    static constexpr int MaxCardinalityAtCompileTime = Space_::MaxCardinality;  ///< a bound on it, or Eigen::Dynamic

    /** The generator's type, the same as a MarkovChain's transition matrix. */
    using MatrixType = typename MarkovChain<Space_, Scalar_>::MatrixType;

    /** The chain that never moves: every rate zero. */
    explicit ContinuousTimeMarkovChain(Space_ space = {})
        : m_space{std::move(space)}
        , m_generator{MatrixType::Zero(m_space.cardinality(), m_space.cardinality())}
    {}

    /** \returns the chain on \a space that jumps from value \f$ i \f$ to value \f$ j \ne i \f$ at
     * the rate \f$ Q_{ij} \f$, \a rates, a row per value moved from and a column per value moved
     * to, with \f$ Q_{ii} = -\sum_{j \ne i} Q_{ij} \f$; or a ChainError if a rate is negative or
     * not finite.
     *
     * The diagonal of \a rates is not read, so a generator is its own rates, and one written to a
     * few digits needs no rounding to sum to zero. The matrix is cast to the chain's scalar.
     */
    template <MatrixExpression Rates>
    [[nodiscard]] static std::expected<ContinuousTimeMarkovChain, ChainError>
    create(Rates && rates, Space_ space = {})
    {
        Eigen::Index const n = space.cardinality();
        MatrixType generator = internal::castScalar<Scalar>(std::forward<Rates>(rates));
        EMRYS_FILTER_ASSERT(
            generator.rows() == n && generator.cols() == n && "the generator has a row and a column per value"
        );
        generator.diagonal().setZero();
        if (!allFinite(generator))
        {
            return std::unexpected{ChainError::NonFinite};
        }
        if ((generator.array() < Scalar(0)).any())
        {
            return std::unexpected{ChainError::Negative};
        }
        generator.diagonal() = -generator.rowwise().sum();
        if (!allFinite(generator))
        {
            return std::unexpected{ChainError::NonFinite};
        }
        return ContinuousTimeMarkovChain{std::move(space), std::move(generator)};
    }

    /** \returns the chain on the DiscreteProduct \a space whose axes move independently, each by its
     * own chain in \a chains, one per axis in order: the product chain, whose generator is their
     * Kronecker sum
     */
    template <DiscreteSpace... Axes>
        requires(
            internal::IsDiscreteProduct<Space_>
            && sizeof...(Axes) == std::tuple_size_v<internal::DiscreteProductAxes<Space_>>
        )
    [[nodiscard]] static ContinuousTimeMarkovChain
    independent(Space_ space, ContinuousTimeMarkovChain<Axes, Scalar_> const &... chains)
    {
        static_assert(
            std::is_same_v<std::tuple<Axes...>, internal::DiscreteProductAxes<Space_>>,
            "give one chain per axis of the product, in order"
        );
        auto result = ContinuousTimeMarkovChain{std::move(space)};
        Eigen::Index const n = result.m_space.cardinality();
        MatrixType & generator = result.m_generator;
        Eigen::Index stride = n;
        (
            [&] {
                Eigen::Index const count = chains.space().cardinality();
                stride /= count;
                // The axis jumps from i to j while every other coordinate stays: i and j differ
                // only in this axis's coordinate, which a multiple of the stride carries.
                for (Eigen::Index i = 0; i < n; ++i)
                {
                    Eigen::Index const from = (i / stride) % count;
                    Eigen::Index const base = i - from * stride;
                    for (Eigen::Index to = 0; to < count; ++to)
                    {
                        generator(i, base + to * stride) += chains.generator()(from, to);
                    }
                }
            }(),
            ...
        );
        EMRYS_FILTER_ASSERT(stride == 1 && "each chain must have its axis's cardinality");
        return result;
    }

    /** \returns independent() on a DiscreteProduct that is a FixedSizeSpace, named as the template
     * argument: <tt>ContinuousTimeMarkovChain<Mode>::independent(maneuver, sensor)</tt>
     */
    template <DiscreteSpace... Axes>
        requires(
            FixedSizeSpace<Space_> && internal::IsDiscreteProduct<Space_>
            && sizeof...(Axes) == std::tuple_size_v<internal::DiscreteProductAxes<Space_>>
        )
    [[nodiscard]] static ContinuousTimeMarkovChain
    independent(ContinuousTimeMarkovChain<Axes, Scalar_> const &... chains)
    {
        return independent(Space_{}, chains...);
    }

    /** \returns the chain on \a space that leaves its value at the rate \a rate and jumps to any
     * other value with equal probability
     */
    [[nodiscard]] static ContinuousTimeMarkovChain
    persistent(Scalar rate, Space_ space = {})
    {
        EMRYS_FILTER_ASSERT(rate >= Scalar(0) && "the rate must be nonnegative");
        Eigen::Index const n = space.cardinality();
        auto result = ContinuousTimeMarkovChain{std::move(space)};
        if (n > 1)
        {
            result.m_generator.setConstant(rate / Scalar(n - 1));
            result.m_generator.diagonal().setConstant(-rate);
        }
        return result;
    }

    /** \returns the chain on \a space that stays at value \f$ i \f$ for a mean time of
     * \f$ \tau_i \f$, \a sojournTimes, and on leaving jumps to each other value \f$ j \f$ in
     * proportion to \f$ w_{ij} \f$, \a exitWeights: \f$ Q_{ii} = -1/\tau_i \f$ and
     * \f$ Q_{ij} = w_{ij} / (\tau_i \sum_k w_{ik}) \f$.
     *
     * The time at a value is [exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
     * with mean \f$ -1/Q_{ii} \f$, and the jump that ends it goes to \f$ j \f$ with probability
     * \f$ Q_{ij} / -Q_{ii} \f$, the sojourn times and jump chain of a
     * [continuous-time chain](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain), so these
     * are exactly the chain's mean sojourn times and jump probabilities. The sojourn times are in the
     * unit of the time steps given to transition(), seconds for a \c std::chrono::duration (see
     * \ref Time), and positive; an infinite one never leaves. The exit weights need not sum to one,
     * and their diagonal is not read. A ChainError says why the values fail: a sojourn time that is
     * NaN or not positive, an exit weight that is negative or not finite, or a value that is left
     * with no exit weight that isn't zero.
     */
    template <typename SojournTimes, typename ExitWeights>
    [[nodiscard]] static std::expected<ContinuousTimeMarkovChain, ChainError>
    fromSojournTimes(
        Eigen::MatrixBase<SojournTimes> const & sojournTimes,
        Eigen::MatrixBase<ExitWeights> const & exitWeights,
        Space_ space = {}
    )
    {
        EMRYS_FILTER_ASSERT(exitWeights.rows() == space.cardinality() && "give a sojourn time per value");
        return internal::sojournGenerator<Scalar, MatrixType>(sojournTimes, exitWeights)
            .transform([&](MatrixType && generator) {
                return ContinuousTimeMarkovChain{std::move(space), std::move(generator)};
            });
    }

    /** \returns the chain on \a space that stays at value \f$ i \f$ for a mean time of
     * \f$ \tau_i \f$, \a sojournTimes, and on leaving jumps to any other value with equal
     * probability: the other fromSojournTimes() with every exit weight one, or its ChainError
     */
    template <typename SojournTimes>
    [[nodiscard]] static std::expected<ContinuousTimeMarkovChain, ChainError>
    fromSojournTimes(Eigen::MatrixBase<SojournTimes> const & sojournTimes, Space_ space = {})
    {
        Eigen::Index const n = space.cardinality();
        return fromSojournTimes(sojournTimes, MatrixType::Ones(n, n) - MatrixType::Identity(n, n), std::move(space));
    }

    /** \returns the discrete space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the generator \f$ Q \f$, with \f$ Q_{ij} \f$ the rate of jumping from \f$ i \f$ to \f$ j \f$ */
    [[nodiscard]] MatrixType const &
    generator() const
    {
        return m_generator;
    }

    /** \returns the rate of jumping from the value \a from to the value \a to, or minus the rate of
     * leaving \a from if they are the same
     */
    [[nodiscard]] Scalar
    rate(Eigen::Index from, Eigen::Index to) const
    {
        return m_generator(from, to);
    }

    /** \returns the chain over an interval of \a time, any TimeStep (see \ref Time):
     * \f$ \Pi = e^{Q \Delta t} \f$
     */
    [[nodiscard]] MarkovChain<Space_, Scalar_>
    transition(TimeStep<Scalar_> auto const & time) const
    {
        Scalar const dt = internal::toScalarTime<Scalar_>(time);
        MatrixType const scaled = m_generator * dt;
        MatrixType transitions = scaled.exp();
        // The exponential of a generator is stochastic up to rounding: clear the negative
        // rounding off the zeros and rescale each row to sum to one.
        transitions = transitions.cwiseMax(Scalar(0));
        transitions.array().colwise() /= transitions.rowwise().sum().array();
        return MarkovChain<Space_, Scalar_>{m_space, std::move(transitions)};
    }

private:

    /** The chain on \a space with the checked generator \a generator [checked-factories]. */
    ContinuousTimeMarkovChain(Space_ space, MatrixType generator)
        : m_space{std::move(space)}
        , m_generator{std::move(generator)}
    {}

    Space_ m_space;          ///< the discrete space
    MatrixType m_generator;  ///< the generator \f$ Q \f$
};

/** \ingroup Distributions
 * \returns \a prior moved over an interval of \a dt by \a chain: \f$ p' = p\, e^{Q \Delta t} \f$
 */
template <typename Scalar, typename Space>
[[nodiscard]] CategoricalParams<Space, Scalar>
predict(
    CategoricalParams<Space, Scalar> const & prior,
    ContinuousTimeMarkovChain<Space, Scalar> const & chain,
    TimeStep<Scalar> auto const & dt
)
{
    return predict(prior, chain.transition(dt));
}

/** \ingroup Distributions
 * \returns the stationary distribution of \a chain, the probabilities \f$ \pi \f$ that no interval
 * changes, \f$ \pi Q = 0 \f$, and the stationary distribution of its transition over any
 * interval: the prior for a mode that has run long before the first measurement. It exists and
 * is unique when every value can be reached from every other, and otherwise is
 * ChainError::Reducible. It is computed as the discrete chain's is, by the algorithm of
 * \ref Grassmann1985 "Grassmann, Taksar, and Heyman (1985)", which reads only the rates off the
 * diagonal.
 */
template <typename Scalar, typename Space>
[[nodiscard]] std::expected<CategoricalParams<Space, Scalar>, ChainError>
stationary(ContinuousTimeMarkovChain<Space, Scalar> const & chain)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    std::expected<ArrayType, ChainError> const probabilities =
        internal::stationaryProbabilities<Scalar, ArrayType>(chain.generator());
    if (!probabilities)
    {
        return std::unexpected{probabilities.error()};
    }
    return CategoricalParams<Space, Scalar>{chain.space(), probabilities->log()};
}

}  // namespace Emrys

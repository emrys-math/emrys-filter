#pragma once

/** \file
 * The measurement of a discrete state: the EmissionModel concept, which fills one log-likelihood
 * per value; update() of a CategoricalParams by one, returning its CategoricalUpdate; the emission
 * models EmissionTable, DensityEmission, PoissonEmission, and ExponentialEmission; and
 * ArrivalInterval, the exact observation of a Markov-modulated Poisson process over a
 * ContinuousTimeMarkovChain's interval, with its predict() (decisions.md [emissions]). Read
 * Distributions/ContinuousTimeMarkovChain.h first, and Filters/HiddenMarkovFilter.h next.
 */

#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Models
 * The buffer an emission model writes its log-likelihoods into: one coefficient per value of the
 * discrete space \a Space, with as many rows at compile time as \a Space has values, so that a
 * model over a space of fixed cardinality writes a fixed-size array.
 */
template <DiscreteSpace Space, typename Scalar>
using EmissionBuffer = Eigen::Ref<Eigen::Array<Scalar, Space::Cardinality, 1>>;

/** \ingroup Models
 * \brief A measurement of a value in the discrete space \a Space: one log-likelihood per value.
 *
 * An emission model has a \c Scalar and a <tt>logLikelihoods(space, out)</tt> that writes
 * \f$ \log p(z \mid v) \f$ into <tt>out(v)</tt> for every value \f$ v \f$ of \c space, sized by
 * its cardinality, given the measurement \f$ z \f$ it captures, as a continuous measurement
 * captures its observation [sensors]. \c out is an EmissionBuffer, fixed in size where the space's
 * cardinality is. The buffer is the caller's, so a bounded space allocates nothing [allocation].
 * update() applies one to a CategoricalParams.
 */
template <typename E, typename Space>
concept EmissionModel =
    DiscreteSpace<Space>
    && requires(E const & emission, Space const & space, EmissionBuffer<Space, typename E::Scalar> out) {
           requires RealScalar<typename E::Scalar>;
           emission.logLikelihoods(space, out);
       };

/** \ingroup Models
 * \brief The result of updating a CategoricalParams by an emission model: the posterior, whose
 * weights sum to one, and the log-evidence of the measurement.
 *
 * \tparam Space   the discrete space
 * \tparam Scalar  scalar type of the probabilities
 */
template <DiscreteSpace Space, RealScalar Scalar>
struct CategoricalUpdate
{
    CategoricalParams<Space, Scalar> posterior;  ///< the distribution given the measurement
    Scalar logEvidence;                          ///< \f$ \log p(z) = \log \sum_v p_v\, p(z \mid v) \f$
};

namespace internal
{

/** \returns \f$ \log p(z \mid v) \f$ of \a emission for each value \f$ v \f$ of \a state's space, in
 * the emission's scalar: what an update multiplies the weights by
 */
template <typename Scalar, typename Space, EmissionModel<Space> E>
[[nodiscard]] typename CategoricalParams<Space, typename E::Scalar>::ArrayType
emissionLogLikelihoods(CategoricalParams<Space, Scalar> const & state, E const & emission)
{
    using Buffer = typename CategoricalParams<Space, typename E::Scalar>::ArrayType;
    auto logLikelihoods = Buffer(state.cardinality());
    emission.logLikelihoods(state.space(), logLikelihoods);
    return logLikelihoods;
}

}  // namespace internal

/** \ingroup Models
 * \returns \a prior updated by \a emission, by Bayes' rule: the posterior
 * \f$ p_v\, p(z \mid v) / p(z) \f$ and the log-evidence \f$ \log p(z) \f$, as a %Gaussian update
 * returns its log-likelihood; or a WeightError if the measurement is impossible under
 * \a prior, or the weights of \a prior fail to define a distribution
 */
template <typename Scalar, typename Space, EmissionModel<Space> E>
[[nodiscard]] std::expected<CategoricalUpdate<Space, Scalar>, WeightError>
update(CategoricalParams<Space, Scalar> const & prior, E const & emission)
{
    auto result = CategoricalUpdate<Space, Scalar>{prior, Scalar(0)};
    auto const evidence = result.posterior.update(internal::emissionLogLikelihoods(prior, emission));
    if (!evidence)
    {
        return std::unexpected{evidence.error()};
    }
    result.logEvidence = *evidence;
    return result;
}

/** \ingroup Models
 * \brief An emission table: the probability of each observed value given each value of the state.
 *
 * The emission matrix \f$ B \f$ of a hidden Markov model with a finite observation space
 * (\ref Rabiner1989 "Rabiner (1989)", §II), a right
 * [stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix) with a row per value of
 * the state and a column per observed value, \f$ B_{vz} = P(z \mid v) \f$. The table itself is
 * not an emission model: observe() names what was observed and borrows the table, so a filter
 * updated every step doesn't copy the matrix.
 *
 * \code
 * auto const table =
 *     EmissionTable{Weather{}, Activity{}, Eigen::Matrix<double, 2, 3>{{0.1, 0.4, 0.5}, {0.6, 0.3, 0.1}}};
 * auto const evidence = update(prior, table.observe(Activity::index<Walk>()));
 * \endcode
 *
 * \tparam Space_             the discrete space of the state
 * \tparam ObservationSpace_  the discrete space of the observation
 * \tparam Scalar_            scalar type of the probabilities, \c double by default
 */
template <DiscreteSpace Space_, DiscreteSpace ObservationSpace_, RealScalar Scalar_ = double>
class EmissionTable
{
public:

    using Scalar = Scalar_;                          ///< the scalar type of the probabilities
    using SpaceType = Space_;                        ///< the discrete space of the state
    using ObservationSpaceType = ObservationSpace_;  ///< the discrete space of the observation

    /** The emission matrix's type: a row per value of the state, a column per observed value. */
    using MatrixType = Eigen::Matrix<
        Scalar_,
        Space_::Cardinality,
        ObservationSpace_::Cardinality,
        Eigen::RowMajor,
        Space_::MaxCardinality,
        ObservationSpace_::MaxCardinality>;

    /** \brief The emission model of one observed value: the column of the table it selects.
     *
     * It borrows the table and must not outlive it.
     */
    class Observation
    {
    public:

        using Scalar = Scalar_;  ///< the scalar type of the probabilities

        /** The observation of the value with flat index \a observed under \a table. */
        Observation(EmissionTable const & table, Eigen::Index observed)
            : m_table{&table}
            , m_observed{observed}
        {
            EMRYS_FILTER_ASSERT(
                observed >= 0 && observed < table.m_matrix.cols() && "the observed value is out of range"
            );
        }

        /** \returns the flat index of the observed value */
        [[nodiscard]] Eigen::Index
        observed() const
        {
            return m_observed;
        }

        /** Writes \f$ \log B_{vz} \f$ into \a out for each value \f$ v \f$ of \a space. */
        void
        logLikelihoods(Space_ const & space, EmissionBuffer<Space_, Scalar_> out) const
        {
            EMRYS_FILTER_ASSERT(space == m_table->m_space && "the table is over a different space");
            EMRYS_FILTER_ASSERT(out.size() == m_table->m_matrix.rows());
            out = m_table->m_matrix.col(m_observed).array().log();
        }

    private:

        EmissionTable const * m_table;  ///< the table, borrowed
        Eigen::Index m_observed;        ///< the observed value's flat index
    };

    /** The table over \a space and \a observations with emission matrix \a matrix: nonnegative,
     * with each row summing to one.
     */
    template <MatrixExpression Matrix>
    EmissionTable(Space_ space, ObservationSpace_ observations, Matrix && matrix)
        : m_space{std::move(space)}
        , m_observations{std::move(observations)}
        , m_matrix{internal::castScalar<Scalar>(std::forward<Matrix>(matrix))}
    {
        EMRYS_FILTER_ASSERT(
            m_matrix.rows() == m_space.cardinality() && m_matrix.cols() == m_observations.cardinality()
            && "the emission matrix has a row per value and a column per observed value"
        );
        EMRYS_FILTER_ASSERT((m_matrix.array() >= Scalar(0)).all() && "emission probabilities must be nonnegative");
        EMRYS_FILTER_ASSERT(
            ((m_matrix.rowwise().sum().array() - Scalar(1)).abs() <= Eigen::NumTraits<Scalar>::dummy_precision()).all()
            && "each row of the emission matrix must sum to one"
        );
    }

    /** \returns the discrete space of the state */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the discrete space of the observation */
    [[nodiscard]] ObservationSpace_ const &
    observationSpace() const
    {
        return m_observations;
    }

    /** \returns the emission matrix \f$ B \f$, with \f$ B_{vz} = P(z \mid v) \f$ */
    [[nodiscard]] MatrixType const &
    matrix() const
    {
        return m_matrix;
    }

    /** \returns the emission model of observing the value with flat index \a observed */
    [[nodiscard]] Observation
    observe(Eigen::Index observed) const &
    {
        return Observation{*this, observed};
    }

    /** observe() of a temporary table is deleted: the observation refers to the table, which would
     * be gone at the end of the full expression (hidden-markov-filter.md Q4).
     */
    void
    observe(Eigen::Index observed) const && = delete;

private:

    Space_ m_space;                    ///< the discrete space of the state
    ObservationSpace_ m_observations;  ///< the discrete space of the observation
    MatrixType m_matrix;               ///< the emission matrix \f$ B \f$
};

/// \cond DEDUCTION_GUIDES
template <DiscreteSpace Space, DiscreteSpace ObservationSpace, MatrixExpression Matrix>
EmissionTable(Space, ObservationSpace, Matrix &&)
    -> EmissionTable<Space, ObservationSpace, typename std::remove_cvref_t<Matrix>::Scalar>;

/// \endcond

/** \ingroup Models
 * \brief The emission model of an observation drawn from a distribution chosen by the value: the
 * log-density of the observation under the value's distribution.
 *
 * \a distributionOf maps a value's flat index to a distribution with <tt>logDensity()</tt> of the
 * observation, such as a %Gaussian per value for a hidden Markov model with continuous
 * observations. It may return a reference into the caller's own list of distributions or build
 * one on the spot.
 *
 * \code
 * auto const densities = std::array{rainy.toDensity().value(), dry.toDensity().value()};
 * filter.update(DensityEmission{z, [&](Eigen::Index v) -> auto const & { return densities[v]; }});
 * \endcode
 *
 * \tparam Observation     the observation's type
 * \tparam DistributionOf  the callable from a value's flat index to its distribution
 */
template <typename Observation, typename DistributionOf>
class DensityEmission
{
public:

    /** The scalar type of the log-densities. */
    using Scalar = std::remove_cvref_t<decltype(std::declval<DistributionOf const &>()(Eigen::Index{})
                                                    .logDensity(std::declval<Observation const &>()))>;

    /** The emission of \a observation, under the distribution \a distributionOf returns for each value. */
    DensityEmission(Observation observation, DistributionOf distributionOf)
        : m_observation{std::move(observation)}
        , m_distributionOf{std::move(distributionOf)}
    {}

    /** \returns the observation */
    [[nodiscard]] Observation const &
    observation() const
    {
        return m_observation;
    }

    /** Writes the observation's log-density under each value's distribution into \a out. */
    template <DiscreteSpace Space>
    void
    logLikelihoods(Space const & space, EmissionBuffer<Space, Scalar> out) const
    {
        EMRYS_FILTER_ASSERT(out.size() == space.cardinality());
        for (Eigen::Index v = 0; v < out.size(); ++v)
        {
            out(v) = m_distributionOf(v).logDensity(m_observation);
        }
    }

private:

    Observation m_observation;        ///< the observation
    DistributionOf m_distributionOf;  ///< each value's distribution
};

namespace internal
{

/** The rates of a Poisson process, one per value of \a Space: the storage and checks the
 * emission models of arrivals share.
 */
template <DiscreteSpace Space, RealScalar Scalar>
class ValueRates
{
public:

    /** One rate per value, in flat-index order. */
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;

    /** The rates \a rates, nonnegative, one per value. */
    template <typename Derived>
    explicit ValueRates(Eigen::DenseBase<Derived> const & rates)
        : m_rates{rates.derived().array().template cast<Scalar>()}
    {
        EMRYS_FILTER_ASSERT((m_rates >= Scalar(0)).all() && "rates must be nonnegative");
    }

    /** \returns the rate of each value */
    [[nodiscard]] ArrayType const &
    rates() const
    {
        return m_rates;
    }

private:

    ArrayType m_rates;  ///< the rate of each value
};

}  // namespace internal

/** \ingroup Models
 * \brief The emission model of a count of events over an interval, from a Poisson process whose
 * rate is a function of the value.
 *
 * A count \f$ n \f$ over an interval \f$ \Delta t \f$ is
 * [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) with mean
 * \f$ \lambda_v \Delta t \f$, so
 * \f$ \log p(n \mid v) = n \log(\lambda_v \Delta t) - \lambda_v \Delta t - \log n! \f$.
 * It treats the value as fixed over the interval. A ContinuousTimeMarkovChain may jump inside it,
 * so for a Markov-modulated Poisson process this is the approximation for intervals short beside
 * the chain's holding times; ArrivalInterval observes one exactly.
 *
 * \tparam Space   the discrete space
 * \tparam Scalar_ scalar type of the rates, \c double by default
 */
template <DiscreteSpace Space, RealScalar Scalar_ = double>
class PoissonEmission : public internal::ValueRates<Space, Scalar_>
{
public:

    using Scalar = Scalar_;  ///< the scalar type of the rates

    /** The emission of \a count events over an interval of \a dt, any TimeStep (see \ref Time),
     * with \a rates one per value. \a count is an integer, so a step and a count given in each
     * other's place do not compile, where a step converted to a count would have been truncated.
     */
    template <typename Derived, std::integral Count>
        requires(!std::same_as<Count, bool>)
    PoissonEmission(Eigen::DenseBase<Derived> const & rates, Count count, TimeStep<Scalar_> auto const & dt)
        : internal::ValueRates<Space, Scalar_>{rates}
        , m_count{Eigen::Index(count)}
        , m_dt{internal::toScalarTime<Scalar_>(dt)}
    {
        EMRYS_FILTER_ASSERT(count >= 0 && "the count must be nonnegative");
        EMRYS_FILTER_ASSERT(m_dt > 0 && "the interval must be positive");
    }

    /** \returns the number of events */
    [[nodiscard]] Eigen::Index
    count() const
    {
        return m_count;
    }

    /** \returns the length of the interval, in the unit of the rates (see \ref Time) */
    [[nodiscard]] Scalar_
    interval() const
    {
        return m_dt;
    }

    /** Writes the Poisson log-probability of the count under each value's rate into \a out. */
    void
    logLikelihoods(Space const & space, EmissionBuffer<Space, Scalar> out) const
    {
        auto const & rates = this->rates();
        EMRYS_FILTER_ASSERT(out.size() == space.cardinality() && rates.size() == out.size());
        Scalar const n = Scalar(m_count);
        Scalar const logFactorial = logGamma<Scalar>(n + Scalar(1));
        for (Eigen::Index v = 0; v < out.size(); ++v)
        {
            Scalar const mean = rates(v) * m_dt;
            if (mean == Scalar(0))
            {
                out(v) = m_count == 0 ? Scalar(0) : -Eigen::NumTraits<Scalar>::infinity();
            }
            else
            {
                out(v) = n * Eigen::numext::log(mean) - mean - logFactorial;
            }
        }
    }

private:

    Eigen::Index m_count;  ///< the number of events
    Scalar_ m_dt;          ///< the length of the interval, in the unit of the rates
};

/** \ingroup Models
 * \brief The emission model of the time from one event to the next, from a Poisson process whose
 * rate is a function of the value.
 *
 * The [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) density of an
 * inter-arrival time \f$ \tau \f$: \f$ \log p(\tau \mid v) = \log \lambda_v - \lambda_v \tau \f$.
 * Like PoissonEmission, it treats the value as fixed over the interval; ArrivalInterval is the
 * exact observation over a ContinuousTimeMarkovChain.
 *
 * \tparam Space   the discrete space
 * \tparam Scalar_ scalar type of the rates, \c double by default
 */
template <DiscreteSpace Space, RealScalar Scalar_ = double>
class ExponentialEmission : public internal::ValueRates<Space, Scalar_>
{
public:

    using Scalar = Scalar_;  ///< the scalar type of the rates

    /** The emission of an inter-arrival time of \a tau, any TimeStep (see \ref Time), with
     * \a rates one per value.
     */
    template <typename Derived>
    ExponentialEmission(Eigen::DenseBase<Derived> const & rates, TimeStep<Scalar_> auto const & tau)
        : internal::ValueRates<Space, Scalar_>{rates}
        , m_tau{internal::toScalarTime<Scalar_>(tau)}
    {}

    /** \returns the inter-arrival time, in the unit of the rates (see \ref Time) */
    [[nodiscard]] Scalar_
    interval() const
    {
        return m_tau;
    }

    /** Writes the exponential log-density of the inter-arrival time under each value's rate into \a out. */
    void
    logLikelihoods(Space const & space, EmissionBuffer<Space, Scalar> out) const
    {
        auto const & rates = this->rates();
        EMRYS_FILTER_ASSERT(out.size() == space.cardinality() && rates.size() == out.size());
        for (Eigen::Index v = 0; v < out.size(); ++v)
        {
            out(v) = Eigen::numext::log(rates(v)) - rates(v) * m_tau;
        }
    }

private:

    Scalar_ m_tau;  ///< the inter-arrival time, in the unit of the rates
};

/** \ingroup Models
 * \brief The observation of a Markov-modulated Poisson process over an interval: no event inside
 * it, and an event at its end unless it closes the record.
 *
 * In a [Markov-modulated Poisson process](https://en.wikipedia.org/wiki/Markovian_arrival_process)
 * events arrive at the rate \f$ \lambda_v \f$ of the value of a ContinuousTimeMarkovChain, which
 * may jump between events. Its likelihood is not an emission, a function of the value at the end
 * of the interval alone: no event arrived while the chain moved. With
 * \f$ \Lambda = \mathrm{diag}(\lambda) \f$, the interval of length \f$ \tau \f$ ending in an event
 * moves the unnormalized distribution as
 * \f$ p' = p\, e^{(Q - \Lambda) \tau} \Lambda \f$, and the closing interval of a record, which
 * ends without one, as \f$ p' = p\, e^{(Q - \Lambda) \tau} \f$. The product of the steps over a
 * record is its likelihood, the closed form of \ref Fischer1993 "Fischer and Meier-Hellstern (1993)",
 * so predict() with an ArrivalInterval in place of predict() and an emission is the exact
 * Markov-modulated Poisson process filter.
 *
 * \tparam Space   the discrete space
 * \tparam Scalar_ scalar type of the rates, \c double by default
 */
template <DiscreteSpace Space, RealScalar Scalar_ = double>
class ArrivalInterval : public internal::ValueRates<Space, Scalar_>
{
public:

    using Scalar = Scalar_;  ///< the scalar type of the rates

    /** The interval with \a rates one per value, ending in an event if \a endsInArrival. */
    template <typename Derived>
    explicit ArrivalInterval(Eigen::DenseBase<Derived> const & rates, bool endsInArrival = true)
        : internal::ValueRates<Space, Scalar_>{rates}
        , m_endsInArrival{endsInArrival}
    {}

    /** \returns whether the interval ends in an event */
    [[nodiscard]] bool
    endsInArrival() const
    {
        return m_endsInArrival;
    }

private:

    bool m_endsInArrival;  ///< whether the interval ends in an event
};

namespace internal
{

/** The step of an ArrivalInterval as a matrix and a log-scale: the kernel
 * \f$ e^{(Q - \Lambda + \lambda_{\min} I)
 * \Delta t} \f$, times \f$ \Lambda \f$ if the interval ends
 * in an event, and \f$ -\lambda_{\min} \Delta t \f$, the
 * log of the factor it is short of
 * \f$ e^{(Q - \Lambda) \Delta t} [\Lambda] \f$ by.
 */
template <typename Space, typename Scalar>
struct ArrivalKernel
{
    typename ContinuousTimeMarkovChain<Space, Scalar>::MatrixType kernel;  ///< the shifted kernel
    Scalar logScale;  ///< the log of the factor the kernel is short of the step by
};

/** \returns the step of \a interval over \a dt of \a chain, shifted by the slowest rate so that
 * the slowest value's
 * entries stay of order one however long the interval
 */
template <typename Space, typename Scalar>
[[nodiscard]] ArrivalKernel<Space, Scalar>
arrivalKernel(
    ContinuousTimeMarkovChain<Space, Scalar> const & chain,
    Scalar dt,
    ArrivalInterval<Space, Scalar> const & interval
)
{
    EMRYS_FILTER_ASSERT(dt >= 0 && "the interval must be nonnegative");
    using MatrixType = typename ContinuousTimeMarkovChain<Space, Scalar>::MatrixType;
    auto const & rates = interval.rates();
    EMRYS_FILTER_ASSERT(rates.size() == chain.space().cardinality() && "give one rate per value");
    Scalar const shift = rates.minCoeff();
    MatrixType exponent = chain.generator() * dt;
    exponent.diagonal().array() -= (rates - shift) * dt;
    MatrixType kernel = exponent.exp();
    kernel = kernel.cwiseMax(Scalar(0));
    if (interval.endsInArrival())
    {
        kernel = kernel * rates.matrix().asDiagonal();
    }
    return {std::move(kernel), -shift * dt};
}

/** \returns \a prior moved by the shifted \a kernel and normalized, with the log of its total
 * over the prior's plus \a logScale, or a WeightError if nothing is left or the weights of
 * \a prior fail to define a distribution
 */
template <typename Space, typename Scalar, typename Kernel>
[[nodiscard]] std::expected<CategoricalUpdate<Space, Scalar>, WeightError>
moveByKernel(CategoricalParams<Space, Scalar> const & prior, Kernel const & kernel, Scalar logScale)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    auto const priorTotal = prior.logTotalWeight();
    if (!priorTotal)
    {
        return std::unexpected{priorTotal.error()};
    }
    ArrayType const logMoved = movedLogWeights<Scalar>(prior.logWeights(), kernel);
    auto const total = logTotalWeight<Scalar>(logMoved);
    if (!total)
    {
        return std::unexpected{total.error()};
    }
    return CategoricalUpdate<Space, Scalar>{
        CategoricalParams<Space, Scalar>{prior.space(), logMoved - *total},
        *total - *priorTotal + logScale
    };
}

}  // namespace internal

/** \ingroup Models
 * \returns \a prior moved over an interval of \a dt by \a chain while observing \a interval: the
 * normalized \f$ p\, e^{(Q - \Lambda) \Delta t} \Lambda \f$ (without \f$ \Lambda \f$ if the
 * interval ends without an event) and the log of its total, the interval's log-likelihood; or a
 * WeightError if the observation is impossible under \a prior
 *
 * The exponential is taken of \f$ Q - \Lambda + \lambda_{\min} I \f$, whose entries stay of order
 * one for the slowest value however long the interval, and \f$ \lambda_{\min} \Delta t \f$ is
 * subtracted from the log-likelihood.
 */
template <typename Scalar, typename Space>
[[nodiscard]] std::expected<CategoricalUpdate<Space, Scalar>, WeightError>
predict(
    CategoricalParams<Space, Scalar> const & prior,
    ContinuousTimeMarkovChain<Space, Scalar> const & chain,
    TimeStep<Scalar> auto const & dt,
    ArrivalInterval<Space, Scalar> const & interval
)
{
    EMRYS_FILTER_ASSERT(prior.space() == chain.space() && "the chain is on a different space");
    auto const step = internal::arrivalKernel(chain, internal::toScalarTime<Scalar>(dt), interval);
    return internal::moveByKernel(prior, step.kernel, step.logScale);
}

}  // namespace Emrys

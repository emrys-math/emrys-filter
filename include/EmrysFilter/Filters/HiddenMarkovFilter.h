#pragma once

/** \file
 * HiddenMarkovFilter (decisions.md [filters-and-recursions], [emissions], [update-outcomes]): a CategoricalParams state
 * moved by a MarkovChain or a ContinuousTimeMarkovChain and updated by emission models, with the log-evidence of the
 * measurements so far, and HiddenMarkovFilterUpdate. The row of the Filters module's table with nothing left %Gaussian.
 * Read KalmanFilter.h first for the shape it follows, and Models/Emission.h for the measurements it takes.
 */

#include <expected>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/Emission.h>
#include <EmrysFilter/Spaces/Discrete.h>

namespace Emrys
{

namespace internal
{

/** Whether \a Chain is a ContinuousTimeMarkovChain, which predicts over an interval. */
template <typename Chain>
inline constexpr bool IsContinuousTimeChain = false;

/** A ContinuousTimeMarkovChain is one. */
template <typename Space, typename Scalar>
inline constexpr bool IsContinuousTimeChain<ContinuousTimeMarkovChain<Space, Scalar>> = true;

}  // namespace internal

/** \ingroup Filters
 * \brief One prediction of a HiddenMarkovFilter as a smoother needs it: the distribution the step
 * started from, the prediction, and the matrix that moved one to the other.
 *
 * The discrete counterpart of PredictedJoint (decisions.md [smoothing]): with \f$ K \f$ the transitions,
 * \f$ \mathrm{predicted} = K^T \mathrm{filtered} \f$ exactly, so the joint of the values before
 * and after the step is \f$ \mathrm{filtered}_i K_{ij} \f$. For a MarkovChain \f$ K \f$ is
 * \f$ \Pi \f$; for a ContinuousTimeMarkovChain over \f$ \Delta t \f$ it is \f$ e^{Q \Delta t} \f$;
 * and for an ArrivalInterval it is the step \f$ e^{(Q - \Lambda) \Delta t} [\Lambda] \f$ divided
 * by its total, which is no longer stochastic, since the step observed something.
 * CategoricalHistory records them.
 *
 * \tparam Space   the discrete space
 * \tparam Scalar  scalar type of the probabilities
 */
template <DiscreteSpace Space, RealScalar Scalar>
struct CategoricalPredictedJoint
{
    using MatrixType = typename MarkovChain<Space, Scalar>::MatrixType;  ///< the transitions' type

    CategoricalParams<Space, Scalar> filtered;   ///< \f$ p(x_k \mid z_{1:k}) \f$, the state before the step
    CategoricalParams<Space, Scalar> predicted;  ///< the state after it, before the next update
    MatrixType transitions;                      ///< \f$ K \f$, with predicted \f$ = K^T \f$ filtered
};

/** \ingroup Filters
 * \brief What a hidden Markov filter's update did: the measurement's log-evidence under the prior,
 * and whether it was accepted.
 *
 * When \c accepted is false, no value with a nonzero probability could explain the measurement,
 * its log-evidence is \f$ -\infty \f$, and the state was left as it was (decisions.md
 * [update-outcomes]).
 */
template <typename Scalar = double>
class HiddenMarkovFilterUpdate
{
public:

    /** \returns \f$ \log p(z_k \mid z_{1:k-1}) = \log \sum_v p_v \, p(z_k \mid v) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    bool accepted = false;  ///< whether some value with a nonzero probability could explain the measurement

private:

    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the measurement's log-evidence
};

/** \ingroup Filters
 * \brief A hidden Markov model filter: a CategoricalParams state, the chain that moves it, and the
 * log-evidence of the measurements so far.
 *
 * The forward recursion of a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
 * (\ref Rabiner1989 "Rabiner (1989)", §III-A), kept normalized at every update as that paper's
 * §V-A scales it, with the log of each step's normalizer summed into logEvidence(), which is
 * \f$ \log p(z_{1:k}) \f$, the quantity for comparing models. The weights themselves are
 * logs, so a long run of unlikely measurements underflows neither [discrete-spaces]. A state given by the
 * user need not be normalized; an update checks it, and fails with a WeightError if its weights
 * fail to define a distribution. An update conditions the state where it is, and a measurement no
 * value explains is rejected and leaves it as it was [update-outcomes].
 *
 * With a MarkovChain, predict() takes one step of the chain. With a ContinuousTimeMarkovChain,
 * predict(dt) moves the state over an interval exactly, and predict(dt, interval) does so while
 * observing an ArrivalInterval, which makes the filter the Markov-modulated Poisson process
 * filter. update() takes any EmissionModel.
 *
 * \code
 * auto filter = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7), CategoricalParams<Weather>{}};
 * filter.predict();
 * auto const updated = filter.update(table.observe(Activity::index<Walk>()));   // or a WeightError
 * filter.state().toDensity()->probability<Rainy>();
 * filter.logEvidence();
 * \endcode
 *
 * \tparam Space_   the discrete space (see \ref Discrete)
 * \tparam Chain_   MarkovChain or ContinuousTimeMarkovChain on the space
 * \tparam Scalar_  scalar type of the probabilities, \c double by default
 * \tparam Upkeep_  the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                  one; the library has no action for a categorical state, so it is the caller's own
 *
 * \sa CategoricalParams, EmissionModel, \ref Filters
 */
template <
    DiscreteSpace Space_,
    typename Chain_ = MarkovChain<Space_>,
    RealScalar Scalar_ = double,
    typename Upkeep_ = upkeep::None>
class HiddenMarkovFilter : public internal::HeldUpkeep<HiddenMarkovFilter<Space_, Chain_, Scalar_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<HiddenMarkovFilter, Upkeep_>;  ///< what holds the upkeep

    static_assert(
        std::is_same_v<typename Chain_::SpaceType, Space_> && std::is_same_v<typename Chain_::Scalar, Scalar_>,
        "the chain must be on the filter's space, in its scalar"
    );

public:

    using Scalar = Scalar_;                                ///< the scalar type of the probabilities
    using SpaceType = Space_;                              ///< the discrete space
    using ChainType = Chain_;                              ///< the chain
    using StateType = CategoricalParams<Space_, Scalar_>;  ///< the state the filter owns

    /** The filter moved by \a chain, starting from \a state, with upkeep \a upkeep, which maintain()
     * runs.
     */
    HiddenMarkovFilter(Chain_ chain, StateType state, std::type_identity_t<Upkeep_> upkeep = {})
        : UpkeepBase{std::move(upkeep)}
        , m_chain{std::move(chain)}
        , m_state{std::move(state)}
    {
        EMRYS_FILTER_ASSERT(m_chain.space() == m_state.space() && "the chain is on a different space");
    }

    /** The filter moved by \a chain, starting from the uniform distribution over its space. */
    explicit HiddenMarkovFilter(Chain_ chain)
        : UpkeepBase{Upkeep_{}}
        , m_chain{std::move(chain)}
        , m_state{m_chain.space()}
    {}

    /** \returns the state estimate */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the state estimate, to change where it is between steps, as when setting its weights.
     * It must stay on the chain's space; setState() replaces it with another, and checks it. The
     * log-evidence is left as it is.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state estimate with one on the chain's space, as the constructor's must be,
     * leaving the log-evidence as it is.
     */
    void
    setState(StateType state)
    {
        EMRYS_FILTER_ASSERT(m_chain.space() == state.space() && "the state is on a different space");
        m_state = std::move(state);
    }

    /** \returns the chain */
    [[nodiscard]] Chain_ const &
    chain() const
    {
        return m_chain;
    }

    /** \returns the chain, to change between steps */
    [[nodiscard]] Chain_ &
    chain()
    {
        return m_chain;
    }

    /** \returns the log-evidence \f$ \log p(z_{1:k}) \f$ of every measurement since construction or
     * the last resetLogEvidence(): the sum of each update's and each observed interval's
     */
    [[nodiscard]] Scalar
    logEvidence() const
    {
        return m_logEvidence;
    }

    /** Sets the log-evidence back to zero, to start a new record from the current state. */
    void
    resetLogEvidence()
    {
        m_logEvidence = Scalar(0);
    }

    /** Moves the state one step of a MarkovChain: \f$ p' = p\, \Pi \f$. */
    void
    predict()
        requires(!internal::IsContinuousTimeChain<Chain_>)
    {
        m_state = Emrys::predict(m_state, m_chain);
    }

    /** Moves the state over an interval of \a dt, any TimeStep (see \ref Time), of a
     * ContinuousTimeMarkovChain: \f$ p' = p\, e^{Q \Delta t} \f$.
     */
    void
    predict(TimeStep<Scalar> auto const & dt)
        requires internal::IsContinuousTimeChain<Chain_>
    {
        m_state = Emrys::predict(m_state, m_chain, dt);
    }

    /** Moves the state over an interval of \a dt of a ContinuousTimeMarkovChain while observing
     * \a interval of a Markov-modulated Poisson process, and adds its log-likelihood to the
     * log-evidence (see ArrivalInterval).
     *
     * \returns the interval's log-likelihood, or a WeightError if the observation is impossible
     * under the state, which is then unchanged
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    predict(TimeStep<Scalar> auto const & dt, ArrivalInterval<Space_, Scalar_> const & interval)
        requires internal::IsContinuousTimeChain<Chain_>
    {
        return apply(Emrys::predict(m_state, m_chain, dt, interval));
    }

    /** Moves the state one step of a MarkovChain, as predict(), and returns the step's joint for
     * a CategoricalHistory.
     */
    [[nodiscard]] CategoricalPredictedJoint<Space_, Scalar_>
    predictJoint()
        requires(!internal::IsContinuousTimeChain<Chain_>)
    {
        auto predicted = Emrys::predict(m_state, m_chain);
        auto joint = CategoricalPredictedJoint<Space_, Scalar_>{std::move(m_state), predicted, m_chain.matrix()};
        m_state = std::move(predicted);
        return joint;
    }

    /** Moves the state over an interval of \a dt of a ContinuousTimeMarkovChain, as predict(dt),
     * and returns the step's joint for a CategoricalHistory.
     */
    [[nodiscard]] CategoricalPredictedJoint<Space_, Scalar_>
    predictJoint(TimeStep<Scalar> auto const & dt)
        requires internal::IsContinuousTimeChain<Chain_>
    {
        auto step = m_chain.transition(dt);
        auto predicted = Emrys::predict(m_state, step);
        auto joint = CategoricalPredictedJoint<Space_, Scalar_>{std::move(m_state), predicted, step.matrix()};
        m_state = std::move(predicted);
        return joint;
    }

    /** Moves the state over an interval of \a dt while observing \a interval, as
     * predict(dt, interval), and returns the step's joint for a CategoricalHistory.
     *
     * \returns the joint, or a WeightError if the observation is impossible under the state,
     * which is then unchanged
     */
    [[nodiscard]] std::expected<CategoricalPredictedJoint<Space_, Scalar_>, WeightError>
    predictJoint(TimeStep<Scalar> auto const & dt, ArrivalInterval<Space_, Scalar_> const & interval)
        requires internal::IsContinuousTimeChain<Chain_>
    {
        auto step = internal::arrivalKernel(m_chain, internal::toScalarTime<Scalar>(dt), interval);
        auto moved = internal::moveByKernel(m_state, step.kernel, step.logScale);
        if (!moved)
        {
            return std::unexpected{moved.error()};
        }
        // Divide the kernel by the step's total, so that it moves the filtered state onto the
        // normalized prediction exactly.
        step.kernel /= Eigen::numext::exp(Scalar(moved->logEvidence - step.logScale));
        auto joint =
            CategoricalPredictedJoint<Space_, Scalar_>{std::move(m_state), moved->posterior, std::move(step.kernel)};
        m_state = std::move(moved->posterior);
        m_logEvidence += moved->logEvidence;
        return joint;
    }

    /** Updates the state by \a emission, by Bayes' rule, in place, and adds the measurement's
     * log-evidence to the total.
     *
     * A measurement impossible under the state, to which every value with a nonzero probability
     * gives likelihood zero, is rejected (decisions.md [update-outcomes]): the state is unchanged, the
     * result's log-likelihood is \f$ -\infty \f$, and so is logEvidence() from then until
     * resetLogEvidence(), since the measurements so far have probability zero under the model.
     *
     * \returns the log-evidence \f$ \log p(z_k \mid z_{1:k-1}) \f$ as the result's logLikelihood(),
     * and whether the measurement was accepted (see HiddenMarkovFilterUpdate); or the WeightError of
     * a state whose weights fail to define a distribution, or of a log-likelihood that is NaN or
     * \f$ +\infty \f$, with the state and the log-evidence unchanged
     */
    template <EmissionModel<Space_> E>
    [[nodiscard]] std::expected<HiddenMarkovFilterUpdate<Scalar_>, WeightError>
    update(E const & emission)
    {
        auto const prior = m_state.logTotalWeight();
        if (!prior)
        {
            return std::unexpected{prior.error()};
        }
        auto const evidence = m_state.update(internal::emissionLogLikelihoods(m_state, emission));
        auto result = HiddenMarkovFilterUpdate<Scalar_>{};
        if (!evidence)
        {
            // The prior is valid, so a sum of zero is the measurement's: no value explains it.
            if (evidence.error() != WeightError::ZeroSum)
            {
                return std::unexpected{evidence.error()};
            }
            internal::UpdateAccess::logLikelihood(result) = -Eigen::NumTraits<Scalar_>::infinity();
            m_logEvidence = -Eigen::NumTraits<Scalar_>::infinity();
            return result;
        }
        internal::UpdateAccess::logLikelihood(result) = *evidence;
        result.accepted = true;
        m_logEvidence += *evidence;
        return result;
    }

private:

    /** Takes \a result's posterior as the state and adds its log-evidence, or passes on its error. */
    [[nodiscard]] std::expected<Scalar, WeightError>
    apply(std::expected<CategoricalUpdate<Space_, Scalar_>, WeightError> result)
    {
        if (!result)
        {
            return std::unexpected{result.error()};
        }
        m_state = std::move(result->posterior);
        m_logEvidence += result->logEvidence;
        return result->logEvidence;
    }

    Chain_ m_chain;            ///< the chain
    StateType m_state;         ///< the state estimate
    Scalar m_logEvidence = 0;  ///< the log-evidence of the measurements so far
};

/// \cond DEDUCTION_GUIDES
template <typename Chain>
HiddenMarkovFilter(Chain, CategoricalParams<typename Chain::SpaceType, typename Chain::Scalar>)
    -> HiddenMarkovFilter<typename Chain::SpaceType, Chain, typename Chain::Scalar>;

template <typename Chain>
HiddenMarkovFilter(Chain) -> HiddenMarkovFilter<typename Chain::SpaceType, Chain, typename Chain::Scalar>;

template <typename Chain, typename U>
HiddenMarkovFilter(Chain, CategoricalParams<typename Chain::SpaceType, typename Chain::Scalar>, U)
    -> HiddenMarkovFilter<typename Chain::SpaceType, Chain, typename Chain::Scalar, U>;

/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Space_, typename Chain_, typename Scalar_, typename Upkeep_>
void
describe(std::ostream & stream, HiddenMarkovFilter<Space_, Chain_, Scalar_, Upkeep_> const & filter, Indent indent)
{
    describeFilter(stream, "hidden Markov filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

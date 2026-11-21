#pragma once

/** \file
 * ParticleFilter (decisions.md [upkeep]): a ParticleSet moved by sampling the process model and
 * its noise, weighted by the measurement likelihood, with upkeep left to the caller, and
 * ParticleFilterUpdate; its steps also run under an execution policy [execution-policies]. Read
 * RbUpdate.h next; the design review is docs/guide/particle-filter.md.
 */

#include <expected>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Core/Print.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief What a particle filter's update did: the observation's log-likelihood under the
 * particles, and whether it was accepted.
 *
 * When \c accepted is false, the observation had zero likelihood at every particle, its
 * log-likelihood is \f$ -\infty \f$, and the weights were left as they were (decisions.md
 * [update-outcomes]).
 */
template <typename Scalar = double>
class ParticleFilterUpdate
{
public:

    /** \returns \f$ \log \sum_i \hat w_i \, p(z \mid x_i) \f$, with the prior weights normalized */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    bool accepted = false;  ///< whether any particle could explain the observation

private:

    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the observation's log-likelihood
};

namespace internal
{

/** An input to a particle filter's prediction: a known one passed through unchanged, which takes
 * no rows of a particle's standard normal draws.
 */
template <typename Input>
struct DrawnInput
{
    /** Refers to \a known; the draws' rows from \a row on are left to the next input. */
    DrawnInput(Input const & known, Eigen::Index & /*row*/)
        : input{known}
    {}

    static constexpr int RowsAtCompileTime = 0;     ///< the rows of a particle's draws it takes: none
    static constexpr int MaxRowsAtCompileTime = 0;  ///< the bound on those rows: none

    Input const & input;  ///< the known input

    /** Draws nothing: a known input takes no rows of \a draws. */
    template <typename Derived, typename G>
    void
    fill(Derived && /*draws*/, G & /*rng*/) const
    {}

    /** \returns the input itself, for every particle whatever its draws */
    template <typename Derived>
    [[nodiscard]] Input const &
    at(Eigen::MatrixBase<Derived> const & /*draws*/) const
    {
        return input;
    }
};

/** An uncertain input to a particle filter's prediction: an independent draw for each particle,
 * from rows of that particle's standard normal draws, which the inputs take in turn.
 */
template <UncertainInput Input>
struct DrawnInput<Input>
{
    using Space = typename Input::SpaceType;  ///< the input's space
    using Scalar = typename Input::Scalar;    ///< the input's scalar type

    /** Factorizes \a uncertain's covariance once, and takes the draws' rows from \a row on, one per
     * difference coordinate, advancing \a row past them.
     */
    DrawnInput(Input const & uncertain, Eigen::Index & row)
        : input{uncertain}
        , factor{defaultRoot(covarianceLowerOf(uncertain))}
        , firstRow{row}
    {
        row += input.differenceDimension();
    }

    /// the rows of a particle's draws it takes at compile time, or \c Eigen::Dynamic
    static constexpr int RowsAtCompileTime = rowsAtCompileTime<true, Space>;

    /// the bound on those rows, or \c Eigen::Dynamic
    static constexpr int MaxRowsAtCompileTime = maxValueDimensionAtCompileTime<DifferenceSpace<Space>>;

    Input const & input;  ///< the uncertain input

    /** A factor \f$ F \f$ of its covariance, \f$ F F^T = \Sigma \f$. */
    typename Moments<Space, Scalar>::MatrixType factor;

    Eigen::Index firstRow;  ///< the first row of a particle's draws that this input takes

    /** Fills this input's rows of one particle's \a draws with standard normals from \a normal and
     * \a rng.
     */
    template <typename Derived, typename G>
    void
    fill(Derived && draws, G & rng, std::normal_distribution<Scalar> & normal) const
    {
        fillStandardNormal(middleRowsOf<RowsAtCompileTime>(draws, firstRow, input.differenceDimension()), rng, normal);
    }

    /** \returns the draw \f$ \mu \boxplus F z \f$ from the input, with \f$ z \f$ its rows of one
     * particle's standard normal \a draws
     */
    template <typename Derived>
    [[nodiscard]] typename Moments<Space, Scalar>::VectorType
    at(Eigen::MatrixBase<Derived> const & draws) const
    {
        typename Moments<Space, Scalar>::DifferenceVectorType const z =
            middleRowsOf<RowsAtCompileTime>(draws.derived(), firstRow, input.differenceDimension());
        return Emrys::plus(input.space(), input.mean(), multiply(factor, z));
    }
};

/** A sampled input to a particle filter's prediction, one that is not %Gaussian (SampledInput): an
 * independent draw for each particle by its drawSample(), kept in rows of that particle's draws, a
 * row per value coordinate, which the inputs take in turn (decisions.md [sampled-inputs]).
 */
template <SampledInput Input>
struct DrawnInput<Input>
{
    using Space = typename Input::SpaceType;  ///< the input's space
    using Scalar = typename Input::Scalar;    ///< the input's scalar type

    /** Refers to \a input, and takes the draws' rows from \a row on, one per value coordinate,
     * advancing \a row past them.
     */
    DrawnInput(Input const & input, Eigen::Index & row)
        : input{input}
        , firstRow{row}
    {
        row += valueDimension(input.space());
    }

    /// the rows of a particle's draws it takes at compile time, or \c Eigen::Dynamic
    static constexpr int RowsAtCompileTime = valueDimensionAtCompileTime<Space>;

    /// the bound on those rows, or \c Eigen::Dynamic
    static constexpr int MaxRowsAtCompileTime = maxValueDimensionAtCompileTime<Space>;

    Input const & input;  ///< the sampled input

    Eigen::Index firstRow;  ///< the first row of a particle's draws that this input takes

    /** Writes a draw of the input by \a rng into its rows of one particle's \a draws. The draw
     * takes its numbers from \a rng by the input's own drawSample(), so \a normal is left as it is.
     */
    template <typename Derived, typename G, typename Normal>
    void
    fill(Derived && draws, G & rng, Normal & /*normal*/) const
    {
        using Column = typename std::remove_cvref_t<Derived>::Scalar;
        typename Moments<Space, Scalar>::VectorType const value = drawSample(input, rng);
        middleRowsOf<RowsAtCompileTime>(draws, firstRow, value.size()) = value.template cast<Column>();
    }

    /** \returns the draw kept in this input's rows of one particle's \a draws */
    template <typename Derived>
    [[nodiscard]] typename Moments<Space, Scalar>::VectorType
    at(Eigen::MatrixBase<Derived> const & draws) const
    {
        return middleRowsOf<RowsAtCompileTime>(draws.derived(), firstRow, valueDimension(input.space()))
            .template cast<Scalar>();
    }
};

/** \returns the rows of a particle's draws that \a input takes: its difference dimension if it is
 * uncertain, its value dimension if it is sampled, and none if it is known.
 */
template <typename Input>
[[nodiscard]] Eigen::Index
drawnRows([[maybe_unused]] Input const & input)
{
    if constexpr (UncertainInput<Input>)
    {
        return input.differenceDimension();
    }
    else if constexpr (SampledInput<Input>)
    {
        return valueDimension(input.space());
    }
    else
    {
        return 0;
    }
}

}  // namespace internal

/** \ingroup Filters
 * \brief A particle filter: a weighted set of state points, moved by sampling the process model
 * and weighted by the measurement likelihood.
 *
 * predict() moves each particle through the process model and adds a draw of its process noise,
 * so it needs no transform method and the model may be as nonlinear as it likes. update()
 * multiplies each particle's weight by \f$ p(z \mid x_i) \f$ from logLikelihoods(), which any
 * measurement with additive noise provides, including mixture noise, and a model with its own
 * logLikelihood(g), such as one known only by its likelihood, supplies itself (decisions.md
 * [likelihood-measurements]).
 *
 * The filter never resamples on its own. Upkeep, such as resampling when the effective
 * sample size drops, is applied by the caller to state(), when it fits the problem: once after
 * the last update before a prediction, not after each of several updates at one time.
 *
 * With Resample applied by the caller this is the bootstrap filter of
 * \ref Gordon1993 "Gordon, Salmond, and Smith (1993)", the sequential importance resampling
 * filter of \ref Arulampalam2002 "Arulampalam et al. (2002)", §IV.B, with the transition prior
 * as the proposal.
 *
 * Both steps also take an execution policy first (see \ref Execution). A prediction draws every
 * particle's standard normals and sampled inputs before it moves any, in the order a sequential
 * prediction draws them, so the particles then move as a for-each and match Sequential's from the
 * same generator.
 * The normals come from \c std::normal_distribution, so a seed fixes the run within one standard
 * library only (decisions.md [random-draws]).
 * An update's likelihoods are a for-each and its evidence a reduction.
 *
 * Under Sequential a prediction draws one particle's normals at a time, in a vector sized by the
 * noise and the inputs, so it takes no scratch, and an update's likelihoods go into
 * scratch from the state's allocator, reused across updates. On a bounded space its steps
 * therefore allocate nothing under Sequential once it has run one at a steady number of particles
 * (see \ref Allocation). Under another policy the draws go into scratch from the state's
 * allocator too, reserved for the noise and reused, which the first prediction with an uncertain
 * input grows.
 *
 * \code
 * auto filter = ParticleFilter{ConstantVelocity<Position, Velocity>{0.1}, initial};
 * auto const upkeep = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng));
 * filter.predict(dt, rng);
 * filter.update(fix);
 * filter.update(range);
 * upkeep(filter.state());   // resampled and roughened if the ESS is below half
 * filter.predict(std::execution::par, dt, rng);   // the same, on every core
 * \endcode
 *
 * The filter can hold its upkeep, given last to the constructor, and run it where the caller calls
 * maintain():
 *
 * \code
 * auto filter = ParticleFilter{model, initial, upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng),
 *                                                              upkeep::roughen(rng))};
 * filter.predict(dt, rng);
 * filter.update(fix);
 * (void)filter.maintain();   // the same as filter.upkeep()(filter.state())
 * \endcode
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Model_   the process model, a ProcessModel or a FixedStepModel, which
 *                  partition the space; with a FixedStepModel, predict() takes no time step
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  the particles' allocator (see \ref Allocation)
 * \tparam Upkeep_     the upkeep maintain() runs on the particles (see \ref Upkeep), upkeep::None unless
 *                     given one
 *
 * \sa ParticleSet, WhenEss, Resample, \ref Filters
 */
template <
    Space Space_,
    typename Model_,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class ParticleFilter
    : public internal::HeldUpkeep<ParticleFilter<Space_, Model_, Scalar_, Allocator_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<ParticleFilter, Upkeep_>;  ///< what holds the upkeep

public:

    using Scalar = Scalar_;                                      ///< the scalar type
    using SpaceType = Space_;                                    ///< the state space
    using ModelType = Model_;                                    ///< the process model
    using StateType = ParticleSet<Space_, Scalar_, Allocator_>;  ///< the state the filter owns
    using allocator_type = Allocator_;                           ///< the state's allocator

    /** A filter with process model \a model, initial particles \a state, which must not be empty,
     * and upkeep \a upkeep, which maintain() runs. Its scratch comes from the state's allocator, with
     * room for the state's capacity if it has one. The model is not deduced from here, so that the
     * deduction guide below, which binds fixedStep()'s result to the state's space, deduces the
     * filter's arguments.
     */
    ParticleFilter(std::type_identity_t<Model_> model, StateType state, std::type_identity_t<Upkeep_> upkeep = {})
        : UpkeepBase{std::move(upkeep)}
        , m_model{std::move(model)}
        , m_state{std::move(state)}
        , m_logLikelihoods(ScalarAllocator(m_state.get_allocator()))
        , m_draws(ScalarAllocator(m_state.get_allocator()))
        , m_batch(ScalarAllocator(m_state.get_allocator()))
        , m_previous(ScalarAllocator(m_state.get_allocator()))
        , m_centers(ScalarAllocator(m_state.get_allocator()))
        , m_spare(ScalarAllocator(m_state.get_allocator()))
        , m_increments(ScalarAllocator(m_state.get_allocator()))
        , m_ratios(ScalarAllocator(m_state.get_allocator()))
        , m_lookAheads(ScalarAllocator(m_state.get_allocator()))
    {
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the initial particle set must have a particle");
        internal::assertPartitions(m_model, m_state.space());
        reserveScratch();
    }

    /** A filter with process model \a model, initial particles read from \a draws, the range
     * drawSamples() returns, which must not be empty, into the state's storage, and upkeep \a upkeep:
     * <tt>ParticleFilter{model, drawSamples(prior, count, rng)}</tt>. It takes the range rather than
     * what made it, so it holds for any argument drawSamples() takes.
     */
    template <typename R>
        requires internal::ValueDraws<R> && std::constructible_from<StateType, R &&>
    ParticleFilter(std::type_identity_t<Model_> model, R && draws, std::type_identity_t<Upkeep_> upkeep = {})
        : ParticleFilter{std::move(model), StateType{std::forward<R>(draws)}, std::move(upkeep)}
    {}

    /** \returns the allocator of the particles and scratch */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the current particles */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the current particles, to change where they are between steps, as upkeep such as
     * resampling does. The set must keep its space, its allocator, and at least one particle;
     * setState() replaces it with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the particles, as when reinitializing, with a set that has at least one and whose
     * space the process model's writes partition, as the constructor's must. The scratch follows
     * the new set's allocator and capacity.
     */
    void
    setState(StateType state)
    {
        EMRYS_FILTER_ASSERT(!state.empty() && "the particle set must have a particle");
        internal::assertPartitions(m_model, state.space());
        m_state = std::move(state);
        if (m_logLikelihoods.get_allocator() != ScalarAllocator(m_state.get_allocator()))
        {
            // The assignment carried the new set's allocator in, so the scratch follows it.
            m_logLikelihoods = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_draws = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_batch = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_previous = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_centers = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_spare = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_increments = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_ratios = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_lookAheads = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
        }
        m_recorded = false;
        reserveScratch();
    }

    /** \returns the process model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_model;
    }

    /** \returns the process model, to change its parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_model;
    }

    /** Moves each particle forward by \a dt, any TimeStep (see \ref Time), through the process
     * model and adds an independent draw of the process noise \f$ Q \f$, drawn with \a rng.
     * \a inputs reach the model as in Emrys::predict(), except that each uncertain input
     * is drawn once per particle rather than carried as a covariance, and that an input which is
     * not %Gaussian, a SampledInput such as a mixture's density, is also taken and drawn once per
     * particle by its drawSample() (decisions.md [sampled-inputs]). The weights are unchanged.
     *
     * \f$ Q \f$ may be singular, as for a Constant model, whose coordinates then move only as the
     * dynamics move them. Each particle's noise is drawn before its inputs, which are drawn in the
     * order the model reads them.
     *
     * \code
     * filter.predict(dt, rng, slips.toDensity().value());   // slips: a GaussianMixtureParams<Slip>
     * \endcode
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict(TimeStep<Scalar_> auto const & dt, G & rng, Inputs const &... inputs)
    {
        predict(execution::sequential, dt, rng, inputs...);
    }

    /** predict(dt, rng, inputs...) under \a policy: the draws are taken first, as the sequential
     * prediction takes them, and the particles then move as a for-each, so the result is
     * Sequential's (see \ref Execution). The process model's propagate() then runs for several
     * particles at once.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict(P && policy, TimeStep<Scalar_> auto const & dt, G & rng, Inputs const &... inputs)
    {
        m_recorded = false;
        auto const & space = m_state.space();
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        auto const call = internal::modelWith(std::cref(m_model), inputs...);
        auto const resolved = internal::resolveDrawnProcess(call, space);
        std::apply(
            [&](auto const &... uncertain) {
                if constexpr (std::remove_cvref_t<decltype(resolved)>::BatchedFunction)
                {
                    moveInBatches(
                        std::forward<P>(policy),
                        rng,
                        internal::defaultRoot(resolved.template noise<Scalar_>(step)),
                        resolved,
                        step,
                        internal::inputOf(uncertain)...
                    );
                }
                else
                {
                    moveParticles(
                        std::forward<P>(policy),
                        rng,
                        internal::defaultRoot(resolved.template noise<Scalar_>(step)),
                        [&](auto const & x, auto const &... in) {
                            return resolved.propagateValue(x, step, in...);
                        },
                        internal::inputOf(uncertain)...
                    );
                }
            },
            resolved.inputs()
        );
    }

    /** Moves each particle forward by the fixed-step model's step, as predict(dt, rng, inputs...)
     * does, with the model's cached factor of \f$ Q \f$ (FixedStepModel::noiseRoot()).
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    void
    predict(G & rng, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        predict(execution::sequential, rng, inputs...);
    }

    /** predict(rng, inputs...) with a fixed-step model, under \a policy, as
     * predict(policy, dt, rng, inputs...) runs.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    void
    predict(P && policy, G & rng, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        m_recorded = false;
        auto const & space = m_state.space();
        auto const call = internal::modelWith(std::cref(m_model.model()), inputs...);
        auto const resolved = internal::resolveDrawnProcess(call, space);
        std::apply(
            [&](auto const &... uncertain) {
                if constexpr (std::remove_cvref_t<decltype(resolved)>::BatchedFunction)
                {
                    moveInBatches(
                        std::forward<P>(policy),
                        rng,
                        m_model.noiseRoot(),
                        resolved,
                        m_model.timeStep(),
                        internal::inputOf(uncertain)...
                    );
                }
                else
                {
                    moveParticles(
                        std::forward<P>(policy),
                        rng,
                        m_model.noiseRoot(),
                        [&](auto const & x, auto const &... in) {
                            return resolved.propagateValue(x, m_model.timeStep(), in...);
                        },
                        internal::inputOf(uncertain)...
                    );
                }
            },
            resolved.inputs()
        );
    }

    /** Moves each particle forward by \a steps of the fixed-step model, as that many calls of
     * predict(rng, inputs...) do, with \a inputs at every step and the draws in the same order.
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires isFixedStep<Model_>
    void
    predict(G & rng, Steps steps, Inputs const &... inputs)
    {
        predict(execution::sequential, rng, steps, inputs...);
    }

    /** predict(rng, steps, inputs...) under \a policy, as that many calls of
     * predict(policy, rng, inputs...) run.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires isFixedStep<Model_>
    void
    predict(P && policy, G & rng, Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        for (Eigen::Index i = 0; i < steps.count; ++i)
        {
            predict(policy, rng, inputs...);
        }
    }

    /** Multiplies each particle's weight by the observation's likelihood there,
     * \f$ p(z \mid x_i) \f$, and normalizes the weights. If no particle can explain the
     * observation, the weights are unchanged and the measurement is rejected, with log-likelihood
     * \f$ -\infty \f$ (decisions.md [update-outcomes]).
     *
     * \returns the observation's log-likelihood under the particles and whether it was accepted
     * (see ParticleFilterUpdate), or the WeightError of prior weights that fail to define a
     * distribution, as weights set to zero by the caller do, with the state unchanged
     */
    template <internal::WeighsPoints M>
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    update(M const & measurement)
    {
        return update(execution::sequential, measurement);
    }

    /** update() under \a policy: the likelihoods are a for-each over the particles and the
     * evidence and the total weight reductions, which match Sequential's to rounding (see
     * \ref Execution). The measurement's measure() then runs for several particles at once.
     */
    template <ExecutionPolicy P, internal::WeighsPoints M>
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    update(P && policy, M const & measurement)
    {
        m_recorded = false;
        auto const logTotal = m_state.logTotalWeight(policy);
        if (!logTotal)
        {
            return std::unexpected{logTotal.error()};
        }
        m_logLikelihoods.resize(std::size_t(m_state.size()));
        auto logWeights = Eigen::Map<typename StateType::WeightArrayType>{m_logLikelihoods.data(), m_state.size()};
        internal::fillLogLikelihoods(policy, measurement, m_state.space(), m_state.states(), logWeights, m_batch);
        logWeights += m_state.logWeights() - *logTotal;
        // A NaN log-likelihood at any particle makes the evidence NaN.
        Scalar_ const logLikelihood = logSumExp(std::forward<P>(policy), logWeights);
        EMRYS_FILTER_ASSERT(!isNaN(logLikelihood) && "the measurement's log-likelihood must not be NaN");
        auto result = ParticleFilterUpdate<Scalar_>{};
        internal::UpdateAccess::logLikelihood(result) = logLikelihood;
        result.accepted = logLikelihood > -Eigen::NumTraits<Scalar_>::infinity();
        if (result.accepted)
        {
            internal::ParticleSetAccess::setNormalizedLogWeights(m_state, logWeights - logLikelihood);
        }
        return result;
    }

    /** One step with the measurement in hand: moves each particle forward by \a dt, drawing it from
     * a proposal that uses \a measurement, and weights it by the measurement, in one call (roadmap
     * step 17zi, a prototype; docs/design/particles.md).
     *
     * A measurement observed with a method, <tt>observe(method, model, z)</tt>, guides the
     * particles. A particle's next state is \f$ f(x) \boxplus F c \f$ for its draws
     * \f$ c \sim \mathcal{N}(0, I) \f$ and the factor \f$ F F^T = Q \f$ that predict() draws with.
     * The method linearizes the measurement about \f$ f(x) \f$ as it does for a Kalman update, by
     * its Jacobian (Linearized) or by sigma points spread by \f$ F \f$, which gives the %Gaussian
     * the draws have given the observation, and the particle draws \f$ c \f$ from that instead.
     * Its weight is multiplied by \f$ p(z \mid x') \, \mathcal{N}(c; 0, I) / q(c) \f$, which
     * corrects for the proposal whatever the method's error. This is the importance function by
     * local linearization of \ref DoucetGodsill2000 "Doucet, Godsill, and Andrieu (2000)", and
     * with a sigma-point method the proposal of the unscented particle filter of
     * \ref VanDerMerwe2000 "van der Merwe et al. (2000)", both as
     * \ref Arulampalam2002 "Arulampalam et al. (2002)" describe them. For a measurement affine in
     * the state with additive %Gaussian noise it is the optimal importance function: the weight
     * is \f$ p(z \mid x) \f$, whatever the draw. The proposal is over the draws rather than the
     * state, so a singular \f$ Q \f$ needs no special case: a direction the noise does not reach
     * has a zero column in \f$ F \f$.
     *
     * A measurement observed without a method, <tt>observe(model, z)</tt>, draws from the model
     * alone, and the step is predict() followed by update(), with the same draws.
     *
     * The filter keeps each particle's state before the step and its weight's factor, which
     * move() reads. \a inputs are sure ones; predict() takes uncertain and sampled inputs.
     *
     * \returns what update() returns: the observation's log-likelihood under the particles and
     * whether it was accepted, or the WeightError of prior weights that define no distribution
     */
    template <internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    step(TimeStep<Scalar_> auto const & dt, M const & measurement, G & rng, Inputs const &... inputs)
    {
        return stepToward(execution::sequential, false, dt, measurement, rng, inputs...);
    }

    /** step() under \a policy: the draws are taken first, in order, and the particles then move as a
     * for-each (see \ref Execution).
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    step(P && policy, TimeStep<Scalar_> auto const & dt, M const & measurement, G & rng, Inputs const &... inputs)
    {
        return stepToward(std::forward<P>(policy), false, dt, measurement, rng, inputs...);
    }

    /** One step of the auxiliary particle filter of \ref Pitt1999 "Pitt and Shephard (1999)", as
     * \ref Arulampalam2002 "Arulampalam et al. (2002)" describe it: step(), with the particles
     * first weighted by how well each is about to explain \a measurement, and the filter's
     * upkeep run on them, before they move (roadmap step 17zi, a prototype).
     *
     * Each particle's weight is first multiplied by its look-ahead likelihood: the likelihood of
     * the observation at \f$ f(x) \f$, the particle moved without noise, or, for a measurement
     * observed with a method, the %Gaussian likelihood of the innovation that the method's
     * linearization gives. maintain() then runs the upkeep the filter holds, which is where the
     * caller's resampling takes place: with <tt>upkeep::resample(rng)</tt> at every step, as the
     * source resamples, or under a trigger such as whenEss(). The particles then move as in
     * step(), and each weight is divided by the look-ahead likelihood of the state it moved from.
     * The filter resamples nowhere else, and not at all without an upkeep that does.
     *
     * \returns as step() does, with the log-likelihood the sum of the two stages' logarithms
     */
    template <internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    auxiliaryStep(TimeStep<Scalar_> auto const & dt, M const & measurement, G & rng, Inputs const &... inputs)
    {
        return stepToward(execution::sequential, true, dt, measurement, rng, inputs...);
    }

    /** auxiliaryStep() under \a policy, as step() runs under one. */
    template <ExecutionPolicy P, internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    auxiliaryStep(
        P && policy,
        TimeStep<Scalar_> auto const & dt,
        M const & measurement,
        G & rng,
        Inputs const &... inputs
    )
    {
        return stepToward(std::forward<P>(policy), true, dt, measurement, rng, inputs...);
    }

    /** Moves each particle by one Metropolis-Hastings step that leaves the posterior as it is, the
     * move of the resample-move algorithm of \ref Gilks2001 "Gilks and Berzuini (2001)": after a
     * resampling has left copies of one particle, the copies that move are distinct again, and the
     * set's moments are not inflated, as roughening inflates them (roadmap step 17zi, a prototype).
     *
     * It follows the step() or auxiliaryStep() that brought the particles to the present time,
     * with the same \a dt, \a measurement, and \a inputs, and at most one resampling between them.
     * Each particle proposes a new state from the state it had before that step, by the proposal
     * the step drew from, and takes it with probability
     * \f$ \min(1, w' / w) \f$, the ratio of the proposed state's weight factor
     * \f$ p(z \mid x') \, \mathcal{N}(c'; 0, I) / q(c') \f$ to the present state's: an
     * independence sampler (\ref Tierney1994 "Tierney (1994)") whose target is the density of the
     * particle's state given the state it moved from and the observation. The weights are
     * unchanged. It may be called again for another sweep.
     *
     * \returns the number of particles that moved
     */
    template <internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] Eigen::Index
    move(TimeStep<Scalar_> auto const & dt, M const & measurement, G & rng, Inputs const &... inputs)
    {
        return move(execution::sequential, dt, measurement, rng, inputs...);
    }

    /** move() under \a policy: the draws are taken first, in order, and the proposals are then made
     * as a for-each (see \ref Execution).
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] Eigen::Index
    move(P && policy, TimeStep<Scalar_> auto const & dt, M const & measurement, G & rng, Inputs const &... inputs)
    {
        static_assert(
            ((!UncertainInput<Inputs> && !SampledInput<Inputs>) && ...),
            "a step that uses the measurement takes a model's sure inputs, sure<Tag>(value); predict() and update() "
            "take uncertain and sampled inputs"
        );
        EMRYS_FILTER_ASSERT(
            m_recorded && "a move follows the step() or auxiliaryStep() whose time step and measurement it is given"
        );
        using Vector = ValueVector<Space_, Scalar_>;
        using Difference = DifferenceVector<Space_, Scalar_>;
        using States = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
        using Weights = typename StateType::WeightArrayType;
        using Draws = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
        auto const & space = m_state.space();
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        auto const call = internal::modelWith(std::cref(m_model), inputs...);
        auto const process = internal::resolveProcess(call, space);
        auto const factor = internal::defaultRoot(process.template noise<Scalar_>(step));
        auto const sensor = internal::resolveMeasurement(measurement, space);
        Eigen::Index const count = m_state.size();
        Eigen::Index const rows = valueDimension(space);
        Eigen::Index const noiseRows = differenceDimension(space);

        // The record follows the particles through a resampling since the step.
        Eigen::Index const resamplings = m_state.resampleCount() - m_recordedResamplings;
        EMRYS_FILTER_ASSERT(
            (resamplings == 0 || resamplings == 1) && "resample at most once between a step and the move after it"
        );
        if (resamplings == 1)
        {
            auto const ancestors = m_state.ancestors();
            Eigen::Index const recorded = Eigen::Index(m_increments.size());
            m_spare.resize(std::size_t((rows + 1) * count));
            auto const from = Eigen::Map<States const>{m_previous.data(), rows, recorded};
            auto to = Eigen::Map<States>{m_spare.data(), rows, count};
            auto weights = Eigen::Map<Weights>{m_spare.data() + rows * count, count};
            auto const recordedWeights = Eigen::Map<Weights const>{m_increments.data(), recorded};
            for (Eigen::Index j = 0; j < count; ++j)
            {
                to.col(j) = from.col(ancestors[std::size_t(j)]);
                weights(j) = recordedWeights(ancestors[std::size_t(j)]);
            }
            m_previous.resize(std::size_t(rows * count));
            m_increments.resize(std::size_t(count));
            Eigen::Map<States>{m_previous.data(), rows, count} = to;
            Eigen::Map<Weights>{m_increments.data(), count} = weights;
            m_recordedResamplings = m_state.resampleCount();
        }
        EMRYS_FILTER_ASSERT(Eigen::Index(m_increments.size()) == count && "the particles changed since the step");

        // A particle's normals, then the uniform it accepts by.
        m_draws.resize(std::size_t((noiseRows + 1) * count));
        auto draws = Eigen::Map<Draws>{m_draws.data(), noiseRows + 1, count};
        auto normal = std::normal_distribution<Scalar_>{};
        auto uniform = std::uniform_real_distribution<Scalar_>{};
        for (Eigen::Index j = 0; j < count; ++j)
        {
            internal::fillStandardNormal(draws.col(j).head(noiseRows), rng, normal);
            draws(noiseRows, j) = uniform(rng);
        }
        m_centers.resize(std::size_t(rows * count));
        m_ratios.resize(std::size_t(count));
        m_logLikelihoods.resize(std::size_t(count));
        auto const previous = Eigen::Map<States const>{m_previous.data(), rows, count};
        auto proposed = Eigen::Map<States>{m_centers.data(), rows, count};
        auto ratios = Eigen::Map<Weights>{m_ratios.data(), count};
        auto candidates = Eigen::Map<Weights>{m_logLikelihoods.data(), count};
        auto increments = Eigen::Map<Weights>{m_increments.data(), count};
        internal::forEachIndex(policy, count, [&](Eigen::Index j) {
            Vector const moved{internal::valueOf(process.propagate(Vector{previous.col(j)}, step))};
            Difference const normals = draws.col(j).head(noiseRows);
            auto const drawn = proposeDraws<M::HasMethod>(sensor, moved, factor, normals);
            Vector next = Emrys::plus(space, moved, Difference{internal::multiply(factor, drawn.draws)});
            Emrys::normalizeValues(space, next);
            proposed.col(j) = next;
            ratios(j) = drawn.logRatio;
        });
        internal::fillLogLikelihoods(
            std::forward<P>(policy),
            measurement,
            space,
            Eigen::Map<States const>{m_centers.data(), rows, count},
            candidates,
            m_batch
        );
        candidates += ratios;
        Eigen::Index moves = 0;
        for (Eigen::Index j = 0; j < count; ++j)
        {
            if (Eigen::numext::log(draws(noiseRows, j)) < candidates(j) - increments(j))
            {
                m_state.setState(j, proposed.col(j));
                increments(j) = candidates(j);
                ++moves;
            }
        }
        return moves;
    }

private:

    using ScalarAllocator = internal::ReBound<Allocator_, Scalar_>;  ///< allocates the scratch

    /** A particle's draws from a proposal: the draws, and what the weight needs of them. */
    struct ProposedDraws
    {
        DifferenceVector<Space_, Scalar_> draws;  ///< \f$ c \f$, which the noise's factor carries to the state
        Scalar_ logRatio = 0;                     ///< \f$ \log \mathcal{N}(c; 0, I) - \log q(c) \f$
        Scalar_ logLookAhead = 0;                 ///< the innovation's %Gaussian log-likelihood; zero without a method
    };

    /** \returns the linearization of the measurement \a sensor, resolved against the state's space,
     * about \a center by its method, as a Kalman update takes it: by the Jacobian at \a center for
     * Linearized, and otherwise by sigma points spread about \a center by the rows of \a factor,
     * a factor of the process noise, at the blocks the measurement reads
     */
    template <typename Sensor, typename Factor>
    [[nodiscard]] auto
    linearizeAbout(Sensor const & sensor, ValueVector<Space_, Scalar_> const & center, Factor const & factor) const
    {
        using Method = std::remove_cvref_t<decltype(sensor.method())>;
        auto const & space = m_state.space();
        if constexpr (internal::isLinearMethod<Method>)
        {
            return internal::linearizeMeasurementAtPoint<Scalar_>(space, center, sensor);
        }
        else
        {
            auto const & reads = sensor.reads();
            return internal::linearizeMeasurementAtRoot<Scalar_>(
                space,
                reads,
                internal::readRootOf(reads, center, factor),
                sensor
            );
        }
    }

    /** \returns a particle's draws \f$ c \f$ from its proposal, made from the standard normals
     * \a normals, for the particle moved without noise to \a moved and the factor \a factor of
     * the process noise, \f$ F F^T = Q \f$.
     *
     * When \a Guided, the measurement \a sensor has a method, and the proposal is the %Gaussian
     * the draws have given the observation under the method's linearization
     * \f$ z \approx \hat z + H F c + e \f$, \f$ e \sim \mathcal{N}(0, S_0) \f$: with
     * \f$ G = H F \f$ and \f$ S = G G^T + S_0 = L L^T \f$,
     * \f[ q(c) = \mathcal{N}(G^T S^{-1} r, \; I - G^T S^{-1} G), \f]
     * whose covariance is positive definite whatever the rank of \f$ Q \f$, and the look-ahead
     * likelihood is \f$ \mathcal{N}(r; 0, S) \f$. Otherwise the proposal is the prior of the
     * draws, and \f$ c \f$ is \a normals.
     */
    template <bool Guided, typename Sensor, typename Factor>
    [[nodiscard]] ProposedDraws
    proposeDraws(
        [[maybe_unused]] Sensor const & sensor,
        [[maybe_unused]] ValueVector<Space_, Scalar_> const & moved,
        [[maybe_unused]] Factor const & factor,
        DifferenceVector<Space_, Scalar_> const & normals
    ) const
    {
        auto result = ProposedDraws{};
        if constexpr (!Guided)
        {
            result.draws = normals;
        }
        else
        {
            using Matrix = typename Moments<Space_, Scalar_>::MatrixType;
            ValueVector<Space_, Scalar_> center = moved;
            Emrys::normalizeValues(m_state.space(), center);
            auto const linearization = linearizeAbout(sensor, center, factor);
            using Linearization = std::remove_cvref_t<decltype(linearization)>;
            using Jacobian = typename Linearization::JacobianMatrix;
            using Innovation = typename Linearization::NoiseMatrix;
            using Residual = typename Linearization::InnovationVector;
            Eigen::Index const n = normals.size();
            Jacobian whitened = internal::multiply(linearization.jacobian, factor);
            Innovation innovation = linearization.noise;
            innovation.noalias() += internal::multiply(whitened, whitened.transpose());
            auto const llt = Eigen::LLT<Innovation>{innovation};
            EMRYS_FILTER_ASSERT(
                llt.info() == Eigen::Success && "the innovation covariance of a guided step must be positive definite"
            );
            llt.matrixL().solveInPlace(whitened);
            Residual residual = linearization.residual;
            llt.matrixL().solveInPlace(residual);
            Matrix covariance = Matrix::Identity(n, n);
            covariance.noalias() -= internal::multiply(whitened.transpose(), whitened);
            auto const covarianceLlt = Eigen::LLT<Matrix>{covariance};
            EMRYS_FILTER_ASSERT(
                covarianceLlt.info() == Eigen::Success && "the proposal's covariance must be positive definite"
            );
            Matrix const lower = covarianceLlt.matrixL();
            Innovation const innovationLower = llt.matrixL();
            result.draws.noalias() = internal::multiply(whitened.transpose(), residual);
            result.draws.noalias() += internal::multiply(lower, normals);
            Scalar_ logRoot = 0;
            for (Eigen::Index j = 0; j < n; ++j)
            {
                logRoot += Eigen::numext::log(lower(j, j));
            }
            Scalar_ logInnovationRoot = 0;
            for (Eigen::Index j = 0; j < residual.size(); ++j)
            {
                logInnovationRoot += Eigen::numext::log(innovationLower(j, j));
            }
            result.logRatio = Scalar_(0.5) * (normals.squaredNorm() - result.draws.squaredNorm()) + logRoot;
            result.logLookAhead = -Scalar_(0.5) * residual.squaredNorm() - logInnovationRoot
                                - Scalar_(0.5) * Scalar_(residual.size()) * log2Pi<Scalar_>();
        }
        return result;
    }

    /** The body of step() and, with \a lookAhead, of auxiliaryStep().
     *
     * With \a lookAhead each weight is first multiplied by the particle's look-ahead likelihood,
     * and the held upkeep runs. Then every particle's standard normals are drawn, in order, as
     * predict() draws them; each particle keeps its state in the record move() reads, moves by its
     * proposal's draws, and has its weight multiplied by the measurement's likelihood at its new
     * state and the ratio of the prior of its draws to their proposal, and divided by its
     * look-ahead likelihood, computed again at the state it moved from, when the first stage
     * multiplied by one. Look-ahead likelihoods that are all zero leave the first stage out.
     */
    template <typename P, typename M, typename G, typename... Inputs>
    [[nodiscard]] std::expected<ParticleFilterUpdate<Scalar_>, WeightError>
    stepToward(
        P && policy,
        bool lookAhead,
        TimeStep<Scalar_> auto const & dt,
        M const & measurement,
        G & rng,
        Inputs const &... inputs
    )
    {
        static_assert(
            ((!UncertainInput<Inputs> && !SampledInput<Inputs>) && ...),
            "a step that uses the measurement takes a model's sure inputs, sure<Tag>(value); predict() and update() "
            "take uncertain and sampled inputs"
        );
        using Vector = ValueVector<Space_, Scalar_>;
        using Difference = DifferenceVector<Space_, Scalar_>;
        using States = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
        using Weights = typename StateType::WeightArrayType;
        constexpr bool Guided = M::HasMethod;
        m_recorded = false;
        auto const & space = m_state.space();
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        auto const call = internal::modelWith(std::cref(m_model), inputs...);
        auto const process = internal::resolveProcess(call, space);
        auto const factor = internal::defaultRoot(process.template noise<Scalar_>(step));
        auto const sensor = internal::resolveMeasurement(measurement, space);
        Eigen::Index const rows = valueDimension(space);
        Eigen::Index const noiseRows = differenceDimension(space);
        auto const initialTotal = m_state.logTotalWeight(policy);
        if (!initialTotal)
        {
            return std::unexpected{initialTotal.error()};
        }
        Scalar_ logFirstStage = 0;
        bool divided = false;
        if (lookAhead)
        {
            Eigen::Index const count = m_state.size();
            m_lookAheads.resize(std::size_t(count));
            m_logLikelihoods.resize(std::size_t(count));
            m_centers.resize(std::size_t(rows * count));
            auto lookAheads = Eigen::Map<Weights>{m_lookAheads.data(), count};
            auto centers = Eigen::Map<States>{m_centers.data(), rows, count};
            Difference const zero = Difference::Zero(noiseRows);
            internal::forEachIndex(policy, count, [&](Eigen::Index i) {
                Vector const prior = std::as_const(m_state).state(i);
                Vector moved{internal::valueOf(process.propagate(prior, step))};
                if constexpr (Guided)
                {
                    lookAheads(i) = proposeDraws<true>(sensor, moved, factor, zero).logLookAhead;
                }
                else
                {
                    Emrys::normalizeValues(space, moved);
                    centers.col(i) = moved;
                }
            });
            if constexpr (!Guided)
            {
                internal::fillLogLikelihoods(
                    policy,
                    measurement,
                    space,
                    Eigen::Map<States const>{m_centers.data(), rows, count},
                    lookAheads,
                    m_batch
                );
            }
            auto first = Eigen::Map<Weights>{m_logLikelihoods.data(), count};
            first = lookAheads + m_state.logWeights() - *initialTotal;
            Scalar_ const logSum = logSumExp(policy, first);
            EMRYS_FILTER_ASSERT(!isNaN(logSum) && "the measurement's log-likelihood must not be NaN");
            if (logSum > -Eigen::NumTraits<Scalar_>::infinity())
            {
                internal::ParticleSetAccess::setNormalizedLogWeights(m_state, first - logSum);
                logFirstStage = logSum;
                divided = true;
                if constexpr (requires { this->maintain(); })
                {
                    [[maybe_unused]] auto const maintained = this->maintain();
                }
            }
        }
        Eigen::Index const count = m_state.size();
        auto const total = m_state.logTotalWeight(policy);
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        m_draws.resize(std::size_t(noiseRows * count));
        auto draws = Eigen::Map<States>{m_draws.data(), noiseRows, count};
        auto normal = std::normal_distribution<Scalar_>{};
        for (Eigen::Index j = 0; j < count; ++j)
        {
            internal::fillStandardNormal(draws.col(j), rng, normal);
        }
        m_previous.resize(std::size_t(rows * count));
        m_centers.resize(std::size_t(rows * count));
        m_increments.resize(std::size_t(count));
        m_ratios.resize(std::size_t(count));
        m_lookAheads.resize(std::size_t(count));
        m_logLikelihoods.resize(std::size_t(count));
        auto previous = Eigen::Map<States>{m_previous.data(), rows, count};
        auto centers = Eigen::Map<States>{m_centers.data(), rows, count};
        auto increments = Eigen::Map<Weights>{m_increments.data(), count};
        auto ratios = Eigen::Map<Weights>{m_ratios.data(), count};
        auto lookAheads = Eigen::Map<Weights>{m_lookAheads.data(), count};
        internal::forEachIndex(policy, count, [&](Eigen::Index j) {
            Vector const prior = std::as_const(m_state).state(j);
            previous.col(j) = prior;
            Vector const moved{internal::valueOf(process.propagate(prior, step))};
            Difference const normals = draws.col(j);
            auto const drawn = proposeDraws<Guided>(sensor, moved, factor, normals);
            m_state.setState(j, Emrys::plus(space, moved, Difference{internal::multiply(factor, drawn.draws)}));
            ratios(j) = drawn.logRatio;
            if constexpr (Guided)
            {
                lookAheads(j) = drawn.logLookAhead;
            }
            else
            {
                Vector center = moved;
                Emrys::normalizeValues(space, center);
                centers.col(j) = center;
            }
        });
        internal::fillLogLikelihoods(policy, measurement, space, m_state.states(), increments, m_batch);
        increments += ratios;
        if constexpr (!Guided)
        {
            if (divided)
            {
                internal::fillLogLikelihoods(
                    policy,
                    measurement,
                    space,
                    Eigen::Map<States const>{m_centers.data(), rows, count},
                    lookAheads,
                    m_batch
                );
            }
        }
        auto logWeights = Eigen::Map<Weights>{m_logLikelihoods.data(), count};
        logWeights = increments + m_state.logWeights() - *total;
        if (divided)
        {
            logWeights -= lookAheads;
        }
        Scalar_ const logSecondStage = logSumExp(std::forward<P>(policy), logWeights);
        EMRYS_FILTER_ASSERT(!isNaN(logSecondStage) && "the measurement's log-likelihood must not be NaN");
        auto result = ParticleFilterUpdate<Scalar_>{};
        internal::UpdateAccess::logLikelihood(result) = logFirstStage + logSecondStage;
        result.accepted = logSecondStage > -Eigen::NumTraits<Scalar_>::infinity();
        if (result.accepted)
        {
            internal::ParticleSetAccess::setNormalizedLogWeights(m_state, logWeights - logSecondStage);
        }
        m_recorded = true;
        m_recordedResamplings = m_state.resampleCount();
        return result;
    }

    /** Reserves the scratch for the state's capacity, or its size if it has none. */
    void
    reserveScratch()
    {
        auto const count = std::size_t(m_state.capacity().value_or(m_state.size()));
        m_logLikelihoods.reserve(count);
        m_draws.reserve(std::size_t(m_state.differenceDimension()) * count);
    }

    /** Moves each particle by \a propagate, given the particle and a draw of each of \a inputs,
     * and adds a draw of the process noise through \a noiseFactor, a factor \f$ G \f$ of it with
     * \f$ G G^T = Q \f$ (see internal::defaultRoot()): the body of every form of predict().
     * It takes the factor rather than \f$ Q \f$ so that a fixed-step model's, computed once, is
     * not computed again at every step.
     *
     * A particle's draws are a column with a row per difference coordinate of the noise and of
     * each uncertain input, and per value coordinate of each sampled input, fixed in size where
     * theirs are: the noise's standard normals, then each input's rows in the order the model reads
     * them, each filled as one draw. Under Sequential each particle's column is drawn
     * on the stack and the particle moved before the next is drawn, so a prediction takes no
     * scratch. Under another policy every column is drawn first, particle by particle, into the
     * filter's scratch, reserved for the noise's rows and grown once if the inputs need more, and
     * the loop that follows reads only its particle's column; the draws are the same.
     */
    template <typename P, typename G, typename Derived, typename Propagate, typename... Inputs>
    void
    moveParticles(
        P && policy,
        G & rng,
        Eigen::MatrixBase<Derived> const & noiseFactor,
        Propagate const & propagate,
        Inputs const &... inputs
    )
    {
        using Vector = ValueVector<Space_, Scalar_>;
        using Difference = DifferenceVector<Space_, Scalar_>;
        constexpr int NoiseRows = internal::rowsAtCompileTime<true, Space_>;
        constexpr int Rows = internal::sizeSum<NoiseRows, internal::DrawnInput<Inputs>::RowsAtCompileTime...>;
        constexpr int MaxRows = internal::sizeSum<
            maxValueDimensionAtCompileTime<DifferenceSpace<Space_>>,
            internal::DrawnInput<Inputs>::MaxRowsAtCompileTime...>;
        auto const & space = m_state.space();
        Eigen::Index const count = m_state.size();
        EMRYS_FILTER_ASSERT(count > 0 && "the particle set must have a particle");
        Eigen::Index const noiseRows = differenceDimension(space);
        // Initialized in order, each input taking its rows after those before it.
        Eigen::Index row = noiseRows;
        auto const drawn = std::tuple{internal::DrawnInput<Inputs>{inputs, row}...};
        Eigen::Index const rows = noiseRows + (Eigen::Index{0} + ... + internal::drawnRows(inputs));
        EMRYS_FILTER_ASSERT(row == rows);
        // One distribution for every draw, which may keep a value between them, so that the draws
        // are those of filling every particle's column at once.
        auto normal = std::normal_distribution<Scalar_>{};
        auto const draw = [&](auto && column) {
            internal::fillStandardNormal(internal::middleRowsOf<NoiseRows>(column, 0, noiseRows), rng, normal);
            std::apply(
                [&](auto const &... inputDraws) {
                    (inputDraws.fill(column, rng, normal), ...);
                },
                drawn
            );
        };
        // Particle i moved by its column of draws.
        auto const moved = [&](Eigen::Index i, auto const & column) {
            Difference const z = internal::middleRowsOf<NoiseRows>(column, 0, noiseRows);
            Vector const propagated = std::apply(
                [&](auto const &... inputDraws) {
                    return Vector{propagate(m_state.state(i), inputDraws.at(column)...)};
                },
                drawn
            );
            return Vector{Emrys::plus(space, propagated, Difference{internal::multiply(noiseFactor, z)})};
        };
        if constexpr (internal::isSequential<P>)
        {
            auto column = internal::MatrixOf<Scalar_, Rows, 1, MaxRows, 1>{};
            column.resize(rows);
            for (Eigen::Index i = 0; i < count; ++i)
            {
                draw(column);
                m_state.setState(i, moved(i, column));
            }
        }
        else
        {
            using Draws = internal::MatrixOf<Scalar_, Rows, Eigen::Dynamic, MaxRows, Eigen::Dynamic>;
            m_draws.resize(std::size_t(rows * count));
            auto draws = Eigen::Map<Draws>{m_draws.data(), rows, count};
            for (Eigen::Index i = 0; i < count; ++i)
            {
                draw(draws.col(i));
            }
            // Through the set's forEach(), which records the change once, before the loop: its
            // setState(i, x) records it at every call, a write each thread would make.
            m_state.forEach(std::forward<P>(policy), [&](auto & particle) {
                particle.setState(moved(particle.index(), draws.col(particle.index())));
            });
        }
    }

    /** Moves each particle as moveParticles() does, by \a resolved, a resolved call of a process
     * model whose function is batched, over \a dt: every particle's draws are taken first, in the
     * order moveParticles() takes them, and the function is then called once over every particle
     * under Sequential and once per chunk of internal::chunkSize particles under another policy
     * (decisions.md [batched-functions]), its input and output in the filter's scratch.
     */
    template <typename P, typename G, typename Derived, typename R, typename TimeScalar, typename... Inputs>
    void
    moveInBatches(
        P && policy,
        G & rng,
        Eigen::MatrixBase<Derived> const & noiseFactor,
        R const & resolved,
        TimeScalar dt,
        Inputs const &... inputs
    )
    {
        using Difference = DifferenceVector<Space_, Scalar_>;
        constexpr int NoiseRows = internal::rowsAtCompileTime<true, Space_>;
        constexpr int Rows = internal::sizeSum<NoiseRows, internal::DrawnInput<Inputs>::RowsAtCompileTime...>;
        constexpr int MaxRows = internal::sizeSum<
            maxValueDimensionAtCompileTime<DifferenceSpace<Space_>>,
            internal::DrawnInput<Inputs>::MaxRowsAtCompileTime...>;
        using Draws = internal::MatrixOf<Scalar_, Rows, Eigen::Dynamic, MaxRows, Eigen::Dynamic>;
        using X = typename R::template Columns<typename R::In, Scalar_>;
        using Y = typename R::template Columns<typename R::Out, Scalar_>;
        auto const & space = m_state.space();
        Eigen::Index const count = m_state.size();
        EMRYS_FILTER_ASSERT(count > 0 && "the particle set must have a particle");
        Eigen::Index const noiseRows = differenceDimension(space);
        Eigen::Index row = noiseRows;
        auto const drawn = std::tuple{internal::DrawnInput<Inputs>{inputs, row}...};
        Eigen::Index const rows = noiseRows + (Eigen::Index{0} + ... + internal::drawnRows(inputs));
        EMRYS_FILTER_ASSERT(row == rows);
        auto normal = std::normal_distribution<Scalar_>{};
        m_draws.resize(std::size_t(rows * count));
        auto draws = Eigen::Map<Draws>{m_draws.data(), rows, count};
        for (Eigen::Index i = 0; i < count; ++i)
        {
            auto column = draws.col(i);
            internal::fillStandardNormal(internal::middleRowsOf<NoiseRows>(column, 0, noiseRows), rng, normal);
            std::apply(
                [&](auto const &... inputDraws) {
                    (inputDraws.fill(column, rng, normal), ...);
                },
                drawn
            );
        }
        Eigen::Index const inRows = valueDimension(resolved.in());
        Eigen::Index const outRows = valueDimension(resolved.out());
        m_batch.resize(std::size_t((inRows + outRows) * count));
        internal::forEachChunk(std::forward<P>(policy), count, [&](Eigen::Index begin, Eigen::Index length) {
            auto x = Eigen::Map<X>{m_batch.data() + inRows * begin, inRows, length};
            auto y = Eigen::Map<Y>{m_batch.data() + inRows * count + outRows * begin, outRows, length};
            for (Eigen::Index j = 0; j < length; ++j)
            {
                std::apply(
                    [&](auto const &... inputDraws) {
                        resolved.template gatherColumn<Scalar_>(
                            x.col(j),
                            m_state.state(begin + j),
                            inputDraws.at(draws.col(begin + j))...
                        );
                    },
                    drawn
                );
            }
            resolved.stepBatch(Eigen::Map<X const>{x.data(), inRows, length}, dt, y);
            for (Eigen::Index j = 0; j < length; ++j)
            {
                Eigen::Index const i = begin + j;
                auto const column = draws.col(i);
                Difference const z = internal::middleRowsOf<NoiseRows>(column, 0, noiseRows);
                auto const moved = resolved.scattered(m_state.state(i), y.col(j));
                m_state.setState(i, Emrys::plus(space, moved, Difference{internal::multiply(noiseFactor, z)}));
            }
        });
    }

    Model_ m_model;                                          ///< the process model
    StateType m_state;                                       ///< the particles
    internal::Buffer<Scalar_, Allocator_> m_logLikelihoods;  ///< scratch, reused across updates
    internal::Buffer<Scalar_, Allocator_> m_draws;     ///< standard normal draws under a policy, a column per particle
    internal::Buffer<Scalar_, Allocator_> m_batch;     ///< a batched function's input and output, a column per particle
    internal::Buffer<Scalar_, Allocator_> m_previous;  ///< each particle's state before the last step, a column each
    internal::Buffer<Scalar_, Allocator_> m_centers;   ///< states moved without noise, or a move's proposed states
    internal::Buffer<Scalar_, Allocator_> m_spare;     ///< where a move carries the record through a resampling
    internal::Buffer<Scalar_, Allocator_> m_increments;  ///< each particle's log weight factor from the last step
    internal::Buffer<Scalar_, Allocator_>
        m_ratios;  ///< the log ratio of the prior of each particle's draws to their proposal
    internal::Buffer<Scalar_, Allocator_> m_lookAheads;  ///< each particle's log look-ahead likelihood
    bool m_recorded = false;                             ///< whether the last step left the record a move reads
    Eigen::Index m_recordedResamplings = 0;              ///< the state's resampleCount() when the record was made
};

/// \cond DEDUCTION_GUIDES
template <typename Space, typename Model, typename Scalar, typename Allocator>
ParticleFilter(Model, ParticleSet<Space, Scalar, Allocator>)
    -> ParticleFilter<Space, internal::BoundStep<Model, Space, Scalar>, Scalar, Allocator>;

template <typename Space, typename Model, typename Scalar, typename Allocator, typename U>
ParticleFilter(Model, ParticleSet<Space, Scalar, Allocator>, U)
    -> ParticleFilter<Space, internal::BoundStep<Model, Space, Scalar>, Scalar, Allocator, U>;

template <typename Model, typename R>
    requires internal::ValueDraws<R>
ParticleFilter(Model, R &&) -> ParticleFilter<
    typename std::remove_cvref_t<R>::SpaceType,
    internal::BoundStep<Model, typename std::remove_cvref_t<R>::SpaceType, typename std::remove_cvref_t<R>::Scalar>,
    typename std::remove_cvref_t<R>::Scalar>;

template <typename Model, typename R, typename U>
    requires internal::ValueDraws<R>
ParticleFilter(Model, R &&, U) -> ParticleFilter<
    typename std::remove_cvref_t<R>::SpaceType,
    internal::BoundStep<Model, typename std::remove_cvref_t<R>::SpaceType, typename std::remove_cvref_t<R>::Scalar>,
    typename std::remove_cvref_t<R>::Scalar,
    DefaultAllocator,
    U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Space_, typename Model_, typename Scalar_, typename Allocator_, typename Upkeep_>
void
describe(
    std::ostream & stream,
    ParticleFilter<Space_, Model_, Scalar_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "particle filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

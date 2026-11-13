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
    /** Refers to \a input; the draws' rows from \a row on are left to the next input. */
    DrawnInput(Input const & input, Eigen::Index & /*row*/)
        : input{input}
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

    /** Factorizes \a input's covariance once, and takes the draws' rows from \a row on, one per
     * difference coordinate, advancing \a row past them.
     */
    DrawnInput(Input const & input, Eigen::Index & row)
        : input{input}
        , factor{defaultRoot(covarianceLowerOf(input))}
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

/** \returns the rows of a particle's standard normal draws that \a input takes: its difference
 * dimension if it is uncertain, and none if it is known.
 */
template <typename Input>
[[nodiscard]] Eigen::Index
drawnRows([[maybe_unused]] Input const & input)
{
    if constexpr (UncertainInput<Input>)
    {
        return input.differenceDimension();
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
 * particle's standard normals before it moves any, in the order a sequential prediction draws
 * them, so the particles then move as a for-each and match Sequential's from the same generator.
 * The normals come from \c std::normal_distribution, so a seed fixes the run within one standard
 * library only (decisions.md [random-draws]).
 * An update's likelihoods are a for-each and its evidence a reduction.
 *
 * Under Sequential a prediction draws one particle's normals at a time, in a vector sized by the
 * noise and the uncertain inputs, so it takes no scratch, and an update's likelihoods go into
 * scratch from the state's allocator, reused across updates. On a bounded space its steps
 * therefore allocate nothing under Sequential once it has run one at a steady number of particles
 * (see \ref Allocation). Under another policy the draws go into scratch from the state's
 * allocator too, reserved for the noise and reused, which the first prediction with an uncertain
 * input grows.
 *
 * \code
 * auto filter = ParticleFilter{ConstantVelocity<Position, Velocity>{0.1}, initial};
 * auto const upkeep = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng));
 * filter.predict(rng, dt);
 * filter.update(fix);
 * filter.update(range);
 * upkeep(filter.state());   // resampled and roughened if the ESS is below half
 * filter.predict(std::execution::par, rng, dt);   // the same, on every core
 * \endcode
 *
 * The filter can hold its upkeep, given last to the constructor, and run it where the caller calls
 * maintain():
 *
 * \code
 * auto filter = ParticleFilter{model, initial, upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng),
 *                                                              upkeep::roughen(rng))};
 * filter.predict(rng, dt);
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
    {
        EMRYS_FILTER_ASSERT(!m_state.empty() && "the initial particle set must have a particle");
        internal::assertPartitions(m_model, m_state.space());
        reserveScratch();
    }

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
        }
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
     * is drawn once per particle rather than carried as a covariance. The weights are unchanged.
     *
     * \f$ Q \f$ may be singular, as for a Constant model, whose coordinates then move only as the
     * dynamics move them. Each particle's noise is drawn before its uncertain inputs, which are
     * drawn in order.
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict(G & rng, TimeStep<Scalar_> auto const & dt, Inputs const &... inputs)
    {
        predict(execution::sequential, rng, dt, inputs...);
    }

    /** predict(rng, dt, inputs...) under \a policy: the draws are taken first, as the sequential
     * prediction takes them, and the particles then move as a for-each, so the result is
     * Sequential's (see \ref Execution). The process model's propagate() then runs for several
     * particles at once.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict(P && policy, G & rng, TimeStep<Scalar_> auto const & dt, Inputs const &... inputs)
    {
        auto const & space = m_state.space();
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        auto const call = internal::modelWith(std::cref(m_model), inputs...);
        auto const resolved = internal::resolveProcess(call, space);
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
                            return internal::valueOf(resolved.propagate(x, step, in...));
                        },
                        internal::inputOf(uncertain)...
                    );
                }
            },
            resolved.inputs()
        );
    }

    /** Moves each particle forward by the fixed-step model's step, as predict(rng, dt, inputs...)
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
     * predict(policy, rng, dt, inputs...) runs.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    void
    predict(P && policy, G & rng, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        auto const & space = m_state.space();
        auto const call = internal::modelWith(std::cref(m_model.model()), inputs...);
        auto const resolved = internal::resolveProcess(call, space);
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
                            return internal::valueOf(resolved.propagate(x, m_model.timeStep(), in...));
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

private:

    using ScalarAllocator = internal::ReBound<Allocator_, Scalar_>;  ///< allocates the scratch

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
     * A particle's standard normals are a column with a row per difference coordinate of the noise
     * and of each uncertain input, fixed in size where theirs are: the noise's rows, then each
     * uncertain input's, each filled as one draw. Under Sequential each particle's column is drawn
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
        auto const move = [&](Eigen::Index i, auto const & column) {
            Difference const z = internal::middleRowsOf<NoiseRows>(column, 0, noiseRows);
            Vector const moved = std::apply(
                [&](auto const &... inputDraws) {
                    return Vector{propagate(m_state.state(i), inputDraws.at(column)...)};
                },
                drawn
            );
            m_state.setState(i, Emrys::plus(space, moved, Difference{internal::multiply(noiseFactor, z)}));
        };
        if constexpr (internal::isSequential<P>)
        {
            auto column = internal::MatrixOf<Scalar_, Rows, 1, MaxRows, 1>{};
            column.resize(rows);
            for (Eigen::Index i = 0; i < count; ++i)
            {
                draw(column);
                move(i, column);
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
            internal::forEachIndex(std::forward<P>(policy), count, [&](Eigen::Index i) {
                move(i, draws.col(i));
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
    internal::Buffer<Scalar_, Allocator_> m_draws;  ///< standard normal draws under a policy, a column per particle
    internal::Buffer<Scalar_, Allocator_> m_batch;  ///< a batched function's input and output, a column per particle
};

/// \cond DEDUCTION_GUIDES
template <typename Space, typename Model, typename Scalar, typename Allocator>
ParticleFilter(Model, ParticleSet<Space, Scalar, Allocator>)
    -> ParticleFilter<Space, internal::BoundStep<Model, Space, Scalar>, Scalar, Allocator>;

template <typename Space, typename Model, typename Scalar, typename Allocator, typename U>
ParticleFilter(Model, ParticleSet<Space, Scalar, Allocator>, U)
    -> ParticleFilter<Space, internal::BoundStep<Model, Space, Scalar>, Scalar, Allocator, U>;
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

#pragma once

/** \file
 * RbParticleFilter (decisions.md [rb-particle-filter]): particles for one block of the state, each
 * carrying a Kalman filter over the rest, or an IMM filter under a switching model, with a coupled
 * prediction that draws the block from the predicted joint and conditions the rest on it. Read
 * RbPointMassFilter.h next; the design review is docs/guide/rao-blackwellized-filters.md.
 */

#include <expected>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/RbUpdate.h>
#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Transform.h>

namespace Emrys
{

namespace internal
{

/** The whole state's step \a Step given a particle's value \a Pinned, as a function of the rest of
 * the state, \a Split assembling the two (see ConditionalSplit): what a transform over a particle's
 * %Gaussian calls, at one rest or, where the step evaluates batches, at a batch of rests in one call
 * (decisions.md [batched-functions]). Valid while its parts are.
 */
template <typename Step, typename Split, typename Pinned>
class PinnedStep
{
public:

    /// whether the transforms evaluate a batch of points in one call, evaluateBatch()
    static constexpr bool EvaluatesBatches = requires { requires Step::EvaluatesBatches; };

    /** The step \a step given the particle \a pinned, assembled by \a split; all three outlive this. */
    PinnedStep(Step const & step, Split const & split, Pinned const & pinned)
        : m_step{step}
        , m_split{split}
        , m_pinned{pinned}
    {}

    /** \returns the whole state assembled from \a rest and the particle, moved forward at the
     * uncertain inputs \a inputs
     */
    template <typename Rest, typename... Inputs>
    [[nodiscard]] auto
    operator()(Rest const & rest, Inputs const &... inputs) const
    {
        return m_step(m_split.assemble(rest, m_pinned), inputs...);
    }

    /** Writes into \a y the whole states assembled from the columns of \a rests and the particle,
     * moved forward at the columns of each of \a inputs, by one call of the step's batch
     */
    template <typename Y, typename Rests, typename... Inputs>
    void
    evaluateBatch(Y & y, Rests const & rests, Inputs const &... inputs) const
        requires EvaluatesBatches
    {
        using T = typename Rests::Scalar;
        using States =
            MatrixOf<T, Y::RowsAtCompileTime, Y::ColsAtCompileTime, Y::MaxRowsAtCompileTime, Y::MaxColsAtCompileTime>;
        auto states = States{};
        states.resize(y.rows(), rests.cols());
        for (Eigen::Index j = 0; j < rests.cols(); ++j)
        {
            states.col(j) = m_split.assemble(rests.col(j), m_pinned);
        }
        m_step.evaluateBatch(y, states, inputs...);
    }

private:

    Step const & m_step;      ///< the whole state's step
    Split const & m_split;    ///< assembles a whole state
    Pinned const & m_pinned;  ///< the particle's value
};

/** A particle's distribution under \a Model over \a Rest, the rest of the state: a %Gaussian, or a
 * hybrid over the switching model's modes from \a Allocator.
 */
template <typename Rest, typename Scalar, typename Allocator, typename Model, bool = isSwitchingPointModel<Model>>
struct ParticleComponentOf
{
    using type = Gaussian<Rest, Scalar>;  ///< the component
};

/** A switching model's particles each hold a hybrid of Gaussians, from the state's allocator. */
template <typename Rest, typename Scalar, typename Allocator, typename Model>
struct ParticleComponentOf<Rest, Scalar, Allocator, Model, true>
{
    using type = Hybrid<Gaussian<Rest, Scalar>, typename Model::ModeSpaceType, Allocator>;  ///< the component
};

/** \returns the log-density of \a offset, a draw of the block that \a split names less its mean, under
 * the block's covariance in the joint covariance \a cov, without the constant that every joint over
 * the same block shares; or NotPositiveDefinite if the block's covariance is not positive definite.
 */
template <typename Split, typename CovDerived, typename OffsetDerived>
[[nodiscard]] std::expected<typename CovDerived::Scalar, NumericError>
pinnedLogDensity(
    Split const & split,
    Eigen::MatrixBase<CovDerived> const & cov,
    Eigen::MatrixBase<OffsetDerived> const & offset
)
{
    using Scalar = typename CovDerived::Scalar;
    using PinnedMatrix = typename Moments<typename Split::PinnedSpace, Scalar>::MatrixType;
    constexpr int PinnedRows = Moments<typename Split::PinnedSpace, Scalar>::DifferenceDimensionAtCompileTime;
    BlockRows const rows = split.pinnedDifferenceRows();
    PinnedMatrix const pinned = blockOf<PinnedRows, PinnedRows>(cov, rows.start, rows.start, rows.size, rows.size);
    auto const llt = Eigen::LLT<PinnedMatrix>{pinned};
    if (llt.info() != Eigen::Success)
    {
        return std::unexpected{NumericError::NotPositiveDefinite};
    }
    auto const whitened = llt.matrixL().solve(offset.derived()).eval();
    return Scalar(-0.5) * whitened.squaredNorm() - llt.matrixLLT().diagonal().array().log().sum();
}

}  // namespace internal

/** \ingroup Filters
 * \brief What a Rao-Blackwellized particle filter's update did, per particle and in total.
 *
 * A particle whose update the policy rejected has log-likelihood \f$ -\infty \f$ and NIS NaN, and
 * weight zero afterwards. When \c accepted is false, no particle could explain the observation,
 * and the state was left as it was.
 *
 * The result owns its per-particle arrays, in storage from \a Allocator. An update into a result
 * (RbParticleFilter::update(measurement, result)) refills that storage, so a result kept across
 * steps allocates nothing once it has held as many particles (see \ref Allocation).
 */
template <typename Scalar = double, typename Allocator = DefaultAllocator>
class RbParticleFilterUpdate
{
public:

    using ArrayType = Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const>;  ///< a per-particle array
    using allocator_type = Allocator;  ///< where the per-particle arrays come from

    /** An empty result whose arrays come from \a allocator. */
    explicit RbParticleFilterUpdate(Allocator const & allocator = {})
        : m_points{allocator}
    {}

    /** \returns \f$ \log p(z \mid i) \f$ per particle */
    [[nodiscard]] ArrayType
    logLikelihoods() const
    {
        return m_points.logLikelihoods();
    }

    /** \returns each particle's NIS; with mixture noise, its most probable branch's, and with a
     * hybrid per particle, its most probable mode's after the update
     */
    [[nodiscard]] ArrayType
    nis() const
    {
        return m_points.nis();
    }

    /** \returns the difference dimension of the innovation */
    [[nodiscard]] Eigen::Index
    innovationSize() const
    {
        return m_points.innovationSize;
    }

    /** \returns the allocator the per-particle arrays come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_points.logLikelihoodData.get_allocator());
    }

    /** \returns \f$ \log \sum_i \hat w_i \, p(z \mid i) \f$, with the prior weights normalized */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    bool accepted = false;  ///< whether any particle could explain the observation

private:

    friend struct internal::PointLikelihoodsAccess;
    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the observation's log-likelihood

    internal::PointLikelihoods<Scalar, Allocator> m_points;  ///< the per-particle arrays
};

/** \ingroup Filters
 * \brief A Rao-Blackwellized particle filter: particles for one block of the state, each carrying
 * a Kalman filter over the rest.
 *
 * The state is a ParticleGaussianParams, whose density, a ParticleGaussian, is
 * \f$ p(x) = \sum_i w_i \, \delta(x_p - p_i) \, \mathcal{N}(x_r; \mu_i, \Sigma_i) \f$. The
 * process model covers the whole state, and the sampled block's motion may depend on the rest,
 * as a position moves with a velocity kept %Gaussian.
 *
 * predict() moves each particle with its %Gaussian: it predicts the whole state given the
 * particle by the filter's method, with the process noise, which gives a %Gaussian over the new
 * state; draws the particle's new value from that %Gaussian's marginal on the sampled block; and
 * conditions the rest on the draw. The draw's offset tells the rest about the block's motion, as a
 * measurement would: a velocity is learned from the positions it moves. The weights don't change.
 * update() conditions each particle's %Gaussian on a measurement given the particle (see
 * update(prior, measurement, given)) and multiplies its weight by \f$ p(z \mid i) \f$.
 *
 * With \a Covariance_ SharedCovariance, every particle's %Gaussian has the same covariance: one
 * transform, at the heaviest particle, serves every particle in predict() and in update(), and
 * each particle costs a propagation or a measurement and a mean update. This is exact when the
 * dynamics and the measurement are affine in the rest of the state with the same Jacobian at every
 * particle, as in the classic mixed linear/nonlinear model; the measurement may depend on the
 * sampled block arbitrarily. Its noise must be one %Gaussian. With PerPointCovariance, each particle
 * is its own Kalman filter, predicted and updated by its own transform, and an update's branches
 * are reduced by a policy (Collapse, Select, or a Gate) as a KalmanFilter's are. A measurement
 * known only by its likelihood of the sampled block (a model with logLikelihood(g) and no
 * function) multiplies each particle's weight by it and leaves the Gaussians alone.
 *
 * With \a Model_ a SwitchingModel over the whole state, each particle holds a HybridGaussian over
 * the rest, an IMM filter's state, so each particle is an IMM filter given its value. predict()
 * mixes each particle's modes through the chain, predicts the whole state given the particle
 * through each mode's model, draws a mode from the predicted mode probabilities and the particle's
 * new value from that mode's prediction, and conditions every mode's %Gaussian on the draw, with
 * each mode's probability multiplied by the density of the draw under that mode's prediction: the
 * draw tells the modes apart, as it tells the rest about the block's motion. The weights don't
 * change. update() updates each particle as ImmRecursion::update(state, measurement, given,
 * policy) does and multiplies its weight by \f$ p(z \mid i) = \sum_m P(m \mid i) \, p(z \mid m, i) \f$,
 * so noise by mode axis (ModeNoise) updates each mode with the component its value names, as in
 * an ImmFilter. The covariance form is then PerPointCovariance, and the particles' steps share
 * the recursion's scratch, so they run one at a time: predict() and update() take Sequential
 * alone, except an update by a likelihood alone, which leaves the hybrids as they were.
 * state().modes() sums the particles' modes.
 *
 * The filter never resamples on its own; the caller applies upkeep to state() (see
 * \ref Upkeep), as with ParticleFilter.
 *
 * The Rao-Blackwellized particle filter of \ref Doucet2000 "Doucet et al. (2000)" and the
 * marginalized particle filter of \ref Schon2005 "Schön, Gustafsson, and Nordlund (2005)",
 * whose §IV gives the prediction that draws the nonlinear substate from the predicted joint and
 * conditions the linear one on the draw.
 *
 * Its scratch (the draws, the posterior Gaussians, and the per-particle likelihoods) comes from
 * the state's allocator and is reused, so its steps allocate nothing once it has run one at a
 * steady number of particles (see \ref Allocation), except that an update of hybrid particles
 * allocates the IMM result it reuses from particle to particle.
 *
 * \code
 * using State = Composite<Position, Velocity>;
 * auto initial = drawParticleGaussian<Position>(prior, 500, rng);
 * auto filter =
 *     RbParticleFilter{ConstantVelocity<Position, Velocity>{0.1},
 * std::move(initial)}; filter.predict(rng, dt);
 * auto const updated = filter.update(range);  // nonlinear in Position
 * upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))(filter.state());
 * filter.state().toDensity()->collapsed();
 * \endcode
 *
 * \tparam Space_       the whole state space
 * \tparam Path_        the block the particles sample, as a tag or a TagPath
 * \tparam Model_       the process model over the whole state, a ProcessModel or a
 *                      FixedStepModel; with a FixedStepModel, predict() takes no time step; a
 *                      SwitchingModel gives each particle a hybrid
 * \tparam Covariance_  SharedCovariance or PerPointCovariance; PerPointCovariance under a
 *                      SwitchingModel
 * \tparam Method_      the prediction method: Linearized or Unscented; a FixedStepModel's
 *                      own; under a SwitchingModel, that of each mode whose model carries
 *                      none of its own (see withMethod()), and ModelsMethod with fixed-step models
 * \tparam Scalar_      scalar type, \c double by default
 * \tparam Allocator_   the state's allocator (see \ref Allocation)
 * \tparam Upkeep_      the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                      one
 *
 * \sa ParticleGaussianParams, ParticleFilter, RbPointMassFilter, \ref Filters
 */
template <
    Space Space_,
    typename Path_,
    typename Model_,
    typename Covariance_ = internal::PointCovarianceFor<Model_>,
    typename Method_ = DefaultMethodFor<Model_>,
    RealScalar Scalar_ = double,
    typename Allocator_ = DefaultAllocator,
    typename Upkeep_ = upkeep::None>
class RbParticleFilter
    : public internal::HeldUpkeep<
          RbParticleFilter<Space_, Path_, Model_, Covariance_, Method_, Scalar_, Allocator_, Upkeep_>,
          Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<RbParticleFilter, Upkeep_>;  ///< what holds the upkeep

    static constexpr bool Shared = std::is_same_v<Covariance_, SharedCovariance>;  ///< one covariance for all particles

    /** Whether each particle holds a hybrid, an IMM filter's state, under a switching model. */
    static constexpr bool Switching = internal::isSwitchingPointModel<Model_>;

    using RestSpace = typename internal::BlockSplit<Space_, Path_>::RestSpace;  ///< the rest of the state

    /** A compile-time error unless the particles' steps may run under \a P. */
    template <typename P>
    static constexpr void
    assertParticlesRunUnder()
    {
        static_assert(
            !Switching || internal::isSequential<P>,
            "a particle's IMM step writes the recursion's scratch, so the particles of a hybrid state run in "
            "order: pass Sequential"
        );
    }

    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with a fixed-step model predicts by the model's method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

public:

    using Scalar = Scalar_;      ///< the scalar type
    using SpaceType = Space_;    ///< the state space
    using ModelType = Model_;    ///< the process model
    using MethodType = Method_;  ///< the prediction method
    /** The state the filter owns: a %Gaussian per particle, or a hybrid under a switching model. */
    using StateType = ParticleGaussianParams<
        Space_,
        Path_,
        Covariance_,
        Scalar_,
        Allocator_,
        typename internal::ParticleComponentOf<RestSpace, Scalar_, Allocator_, Model_>::type>;
    using ConditionalType = typename StateType::ConditionalType;  ///< a particle's distribution over the rest
    using allocator_type = Allocator_;                            ///< the state's allocator

    /** A filter with process model \a model over the whole state, initial state \a state, prediction
     * method \a method, which a fixed-step model carries instead, and upkeep \a upkeep, which
     * maintain() runs. Its scratch comes from the state's allocator, with room for the state's
     * capacity if it has one. The model is not deduced from here, so that the deduction guide below,
     * which binds fixedStep()'s result to the state's space, deduces the filter's arguments.
     */
    RbParticleFilter(
        std::type_identity_t<Model_> model,
        StateType state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : UpkeepBase{std::move(upkeep)}
        , m_steps{std::move(model), std::move(method), state.get_allocator()}
        , m_state{std::move(state)}
        , m_posterior{m_state.m_conditionals, m_state.get_allocator()}
        , m_normals(ScalarAllocator(m_state.get_allocator()))
        , m_moved(ScalarAllocator(m_state.get_allocator()))
    {
        assertModelPartitions(m_state);
        reserveScratch();
    }

    /** A filter with process model \a model over the whole state, initial state \a state, the
     * default prediction method, and upkeep \a upkeep, which maintain() runs.
     */
    RbParticleFilter(std::type_identity_t<Model_> model, StateType state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : RbParticleFilter{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}

    /** \returns the allocator of the state and scratch */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the particles and their Gaussians */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the state, to change where it is between steps, as upkeep such as resampling does.
     * It must keep its space and allocator; setState() replaces it with another, and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state, as when reinitializing, with one whose space the process model's writes
     * partition, as the constructor's must. The scratch follows the new state's allocator and
     * capacity.
     */
    void
    setState(StateType state)
    {
        assertModelPartitions(state);
        m_state = std::move(state);
        if (m_normals.get_allocator() != ScalarAllocator(m_state.get_allocator()))
        {
            // The assignment carried the new state's allocator in, so the scratch follows it.
            m_posterior = decltype(m_posterior){m_state.m_conditionals, m_state.get_allocator()};
            m_normals = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            m_moved = internal::Buffer<Scalar_, Allocator_>(ScalarAllocator(m_state.get_allocator()));
            if constexpr (Switching)
            {
                m_steps.resetScratch(m_state.get_allocator());
            }
        }
        reserveScratch();
    }

    /** \returns the process model */
    [[nodiscard]] Model_ const &
    model() const
    {
        return m_steps.model();
    }

    /** \returns the process model, to change its parameters */
    [[nodiscard]] Model_ &
    model()
    {
        return m_steps.model();
    }

    /** \returns the prediction method */
    [[nodiscard]] Method_ const &
    method() const
    {
        return m_steps.method();
    }

    /** Moves each particle and its %Gaussian forward by \a dt, any TimeStep (see \ref Time):
     * predicts the whole state given the particle through the process model, with \a inputs
     * passed as in Emrys::predict(), draws the particle's new value from the prediction's marginal
     * on the sampled block with \a rng, and conditions the rest on it. The weights are unchanged.
     *
     * The draws are taken first, one standard normal vector per particle. With a hybrid per
     * particle, each particle then draws its mode with one uniform draw, in order.
     *
     * \returns nothing; with a hybrid per particle, a \c std::expected with NotPositiveDefinite if
     * a mode the chain reaches predicts the sampled block with a covariance that is not positive
     * definite, so that the draw has no density under it, with the state unchanged
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(G & rng, TimeStep<Scalar_> auto const & dt, Inputs const &... inputs)
    {
        return predict(execution::sequential, rng, dt, inputs...);
    }

    /** predict(rng, dt, inputs...) under \a policy: the draws are taken first, as the sequential
     * prediction takes them, and each particle and its %Gaussian then move as a for-each, so the
     * result is Sequential's (see \ref Execution). The process model's propagate() then runs for
     * several particles at once. Hybrid particles take Sequential alone.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(!isFixedStep<Model_>)
    [[nodiscard]] auto
    predict(P && policy, G & rng, TimeStep<Scalar_> auto const & dt, Inputs const &... inputs)
    {
        Scalar_ const step = internal::toScalarTime<Scalar_>(dt);
        if constexpr (Switching)
        {
            assertParticlesRunUnder<P>();
            return moveHybridParticles(rng, [&](auto const & model, auto const & method, auto const & predictBy) {
                auto const call = internal::modelWith(std::cref(model), inputs...);
                auto const resolved = internal::resolveProcess(call, m_state.space());
                return std::apply(
                    [&](auto const &... uncertain) {
                        return predictBy(
                            method,
                            resolved.template noise<Scalar_>(step),
                            internal::ProcessStepValues<std::remove_cvref_t<decltype(resolved)>, Scalar_>{
                                resolved,
                                step
                            },
                            internal::inputOf(uncertain)...
                        );
                    },
                    resolved.inputs()
                );
            });
        }
        else
        {
            predictOne(std::forward<P>(policy), rng, step, inputs...);
        }
    }

    /** Moves each particle and its %Gaussian forward by the fixed-step model's step, as
     * predict(rng, dt, inputs...) does, with the model's cached \f$ Q \f$. Under a cached
     * transition and with no inputs, the prediction given each particle is
     * \f$ F_r P_i F_r^T + Q \f$, with \f$ F_r \f$ the transition's columns for the rest of the
     * state, rather than a transform; with a hybrid per particle, each mode predicts by its model's
     * transform.
     *
     * \returns as predict(rng, dt, inputs...) does
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(G & rng, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        return predict(execution::sequential, rng, inputs...);
    }

    /** predict(rng, inputs...) under \a policy, as predict(policy, rng, dt, inputs...) runs. */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    [[nodiscard]] auto
    predict(P && policy, G & rng, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        if constexpr (Switching)
        {
            assertParticlesRunUnder<P>();
            return moveHybridParticles(rng, [&](auto const & model, auto const & method, auto const & predictBy) {
                auto const call = internal::modelWith(std::cref(model.model()), inputs...);
                auto const resolved = internal::resolveProcess(call, m_state.space());
                return std::apply(
                    [&](auto const &... uncertain) {
                        return predictBy(
                            method,
                            model.noise(),
                            internal::ProcessStepValues<
                                std::remove_cvref_t<decltype(resolved)>,
                                std::remove_cvref_t<decltype(model.timeStep())>>{resolved, model.timeStep()},
                            internal::inputOf(uncertain)...
                        );
                    },
                    resolved.inputs()
                );
            });
        }
        else
        {
            predictOneFixed(std::forward<P>(policy), rng, inputs...);
        }
    }

    /** Moves each particle and its %Gaussian forward by \a steps of the fixed-step model, as that
     * many calls of predict(rng, inputs...) do, with \a inputs at every step and the draws in the
     * same order.
     *
     * \returns as predict(rng, inputs...) does; with a hybrid per particle, the first failed step's
     * error, the steps after it not taken
     */
    template <std::uniform_random_bit_generator G, typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(G & rng, Steps steps, Inputs const &... inputs)
    {
        return predict(execution::sequential, rng, steps, inputs...);
    }

    /** predict(rng, steps, inputs...) under \a policy, as that many calls of
     * predict(policy, rng, inputs...) run.
     */
    template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename... Inputs>
        requires isFixedStep<Model_>
    [[nodiscard]] auto
    predict(P && policy, G & rng, Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        if constexpr (Switching)
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                if (auto moved = predict(policy, rng, inputs...); !moved)
                {
                    return moved;
                }
            }
            return std::expected<void, NumericError>{};
        }
        else
        {
            for (Eigen::Index i = 0; i < steps.count; ++i)
            {
                predict(policy, rng, inputs...);
            }
        }
    }

    /** The result of update(): the per-particle arrays from the filter's allocator. */
    using UpdateType = RbParticleFilterUpdate<Scalar, Allocator_>;

    /** Conditions each particle's %Gaussian on \a measurement given the particle, by the
     * measurement's own method, and multiplies each particle's weight by \f$ p(z \mid i) \f$.
     * With SharedCovariance, the measurement's noise must be one %Gaussian; with
     * PerPointCovariance, each particle's branches are collapsed, as update(measurement, policy)
     * does with Collapse. If no particle can explain the observation, the state is unchanged.
     *
     * A measurement whose model supplies its own logLikelihood(g), with or without a function,
     * instead multiplies each particle's weight by that likelihood at the particle and leaves every
     * particle's %Gaussian as it was (decisions.md [likelihood-measurements]). Every block it reads
     * must lie in the sampled block, which is checked at compile time: given the rest, the rest
     * would no longer be %Gaussian. Its result has NIS NaN and an innovation of size 0.
     *
     * \returns each particle's log-likelihood and NIS, the total log-likelihood, and whether the
     * measurement was accepted (see RbParticleFilterUpdate), in arrays from the filter's
     * allocator; or the WeightError of particle weights that cannot be normalized, or, if a
     * particle's update failed, as one whose innovation covariance is not positive definite does,
     * its error, either with the state unchanged (decisions.md [innovation-cholesky], [update-outcomes]),
     * or with a hybrid per particle, which is updated where it is, with the particles before the
     * failed one updated and the weights as they were
     */
    template <internal::WeighsPoints M>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(M const & measurement)
    {
        return update(execution::sequential, measurement);
    }

    /** update() under \a executionPolicy: each particle's update is a for-each and the evidence a
     * reduction, which matches Sequential's to rounding (see \ref Execution).
     */
    template <ExecutionPolicy P, internal::WeighsPoints M>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(P && executionPolicy, M const & measurement)
    {
        auto result = UpdateType{get_allocator()};
        auto const updated = update(std::forward<P>(executionPolicy), measurement, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(measurement), but into \a result, whose storage is reused: an update into a
     * result kept across steps allocates nothing once it has held as many particles.
     *
     * \returns nothing, or the error update(measurement) would return
     */
    template <internal::WeighsPoints M, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(M const & measurement, RbParticleFilterUpdate<Scalar, ResultAllocator> & result)
    {
        return update(execution::sequential, measurement, result);
    }

    /** update(measurement, result) under \a executionPolicy, as update(executionPolicy,
     * measurement) runs.
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(P && executionPolicy, M const & measurement, RbParticleFilterUpdate<Scalar, ResultAllocator> & result)
    {
        if constexpr (!internal::hasOwnLikelihood<M> && !Shared)
        {
            return update(std::forward<P>(executionPolicy), measurement, reductions::collapse(), result);
        }
        else
        {
            if (auto const total = m_state.logTotalWeight(executionPolicy); !total)
            {
                return std::unexpected{total.error()};
            }
            auto & points = internal::PointLikelihoodsAccess::points(result);
            if constexpr (internal::hasOwnLikelihood<M>)
            {
                internal::updateGivenLikelihood(
                    executionPolicy,
                    m_state.points(),
                    m_state.m_conditionals,
                    measurement,
                    points
                );
                finish<false>(std::forward<P>(executionPolicy), result);
            }
            else
            {
                auto const updated = internal::updateGivenShared(
                    executionPolicy,
                    m_state.points(),
                    m_state.m_conditionals,
                    m_state.mostProbable(),
                    measurement,
                    points,
                    m_posterior
                );
                if (!updated)
                {
                    return std::unexpected{updated.error()};
                }
                finish(std::forward<P>(executionPolicy), result);
            }
            return {};
        }
    }

    /** As update(measurement), with PerPointCovariance, but with each particle's branches reduced
     * by \a policy (Collapse, Select, or a Gate); a particle whose update the policy rejects keeps
     * its %Gaussian and gets weight zero. A measurement weighted by its own likelihood has no
     * branches, so it takes no policy.
     */
    template <internal::WeighsPoints M, typename Policy>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(M const & measurement, Policy const & policy)
        requires(!Shared)
    {
        return update(execution::sequential, measurement, policy);
    }

    /** update(measurement, policy) under \a executionPolicy: each particle's Kalman update and
     * reduction by \a policy is a for-each, and the evidence a reduction, which matches
     * Sequential's to rounding (see \ref Execution).
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, typename Policy>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(P && executionPolicy, M const & measurement, Policy const & policy)
        requires(!Shared)
    {
        auto result = UpdateType{get_allocator()};
        auto const updated = update(std::forward<P>(executionPolicy), measurement, policy, result);
        if (!updated)
        {
            return std::unexpected{updated.error()};
        }
        return result;
    }

    /** As update(measurement, policy), but into \a result, whose storage is reused (see
     * update(measurement, result)).
     *
     * \returns nothing, or the error update(measurement, policy) would return
     */
    template <internal::WeighsPoints M, typename Policy, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(M const & measurement, Policy const & policy, RbParticleFilterUpdate<Scalar, ResultAllocator> & result)
        requires(!Shared)
    {
        return update(execution::sequential, measurement, policy, result);
    }

    /** update(measurement, policy, result) under \a executionPolicy, as update(executionPolicy,
     * measurement, policy) runs.
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, typename Policy, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(
        P && executionPolicy,
        M const & measurement,
        Policy const & policy,
        RbParticleFilterUpdate<Scalar, ResultAllocator> & result
    )
        requires(!Shared)
    {
        static_assert(
            !internal::hasOwnLikelihood<M>,
            "a measurement weighted by its own logLikelihood() has no branches to reduce, so it takes no policy"
        );
        // Weights the caller left undefined fail the update before any particle is conditioned.
        if (auto const total = m_state.logTotalWeight(executionPolicy); !total)
        {
            return std::unexpected{total.error()};
        }
        if constexpr (Switching && internal::isByMode<M>)
        {
            // Each mode's measurement holds its own noise component, so the recursion resolves it.
            return updateParticles(
                std::forward<P>(executionPolicy),
                m_steps.template emptyGivenUpdate<M, Space_, Path_>(get_allocator(), modeCount()),
                [&](auto & hybrid, auto const & pinned, auto & one) {
                    using Value = std::remove_cvref_t<decltype(pinned)>;
                    return m_steps.update(
                        hybrid,
                        measurement,
                        GivenBlock<Space_, Path_, Value>{m_state.space(), pinned},
                        policy,
                        one
                    );
                },
                result
            );
        }
        else if constexpr (Switching)
        {
            auto const f = internal::PinnedMeasurement<Space_, Path_, M>{m_state.space(), measurement};
            return updateParticles(
                std::forward<P>(executionPolicy),
                m_steps.template emptyGivenUpdate<M, Space_, Path_>(get_allocator(), modeCount()),
                [&](auto & hybrid, auto const & pinned, auto & one) {
                    return m_steps.updateGiven(hybrid, f, pinned, policy, one);
                },
                result
            );
        }
        else
        {
            auto const updated = internal::updateGivenEach(
                executionPolicy,
                m_state.points(),
                m_state.m_conditionals,
                measurement,
                policy,
                internal::PointLikelihoodsAccess::points(result),
                m_posterior
            );
            if (!updated)
            {
                return std::unexpected{updated.error()};
            }
            finish(std::forward<P>(executionPolicy), result);
            return {};
        }
    }

private:

    using ScalarAllocator = internal::ReBound<Allocator_, Scalar_>;  ///< allocates the draw scratch

    /** Asserts that the process model's writes, or every mode's under a switching model, partition
     * \a state's space: the check of the constructor and of setState().
     */
    void
    assertModelPartitions(StateType const & state) const
    {
        internal::assertPartitions(
            m_steps.model(),
            state.space(),
            Switching ? "every mode's process model must write every coordinate of the state space once"
                      : "the process model must write every coordinate of the state space once"
        );
    }

    /** \returns the number of modes of a hybrid particle */
    [[nodiscard]] Eigen::Index
    modeCount() const
        requires Switching
    {
        return m_steps.modeSpace().cardinality();
    }

    /** Moves the particles by the process model, \a step forward, with \a inputs: the body of
     * predict(policy, rng, dt, inputs...) without a switching model.
     */
    template <typename P, typename G, typename... Inputs>
    void
    predictOne(P && policy, G & rng, Scalar_ step, Inputs const &... inputs)
    {
        auto const call = internal::modelWith(std::cref(m_steps.model()), inputs...);
        auto const resolved = internal::resolveProcess(call, m_state.space());
        std::apply(
            [&](auto const &... uncertain) {
                moveParticles(
                    std::forward<P>(policy),
                    rng,
                    resolved.template noise<Scalar_>(step),
                    [&](auto const & x, auto const &... in) {
                        return internal::valueOf(resolved.propagate(x, step, in...));
                    },
                    internal::inputOf(uncertain)...
                );
            },
            resolved.inputs()
        );
    }

    /** Moves the particles by the fixed-step model's step, with \a inputs: the body of
     * predict(policy, rng, inputs...) without a switching model.
     */
    template <typename P, typename G, typename... Inputs>
    void
    predictOneFixed(P && policy, G & rng, Inputs const &... inputs)
    {
        auto const & model = m_steps.model();
        auto const call = internal::modelWith(std::cref(model.model()), inputs...);
        auto const resolved = internal::resolveProcess(call, m_state.space());
        std::apply(
            [&](auto const &... uncertain) {
                moveParticles(
                    std::forward<P>(policy),
                    rng,
                    model.noise(),
                    [&](auto const & x, auto const &... in) {
                        return internal::valueOf(resolved.propagate(x, model.timeStep(), in...));
                    },
                    internal::inputOf(uncertain)...
                );
            },
            resolved.inputs()
        );
    }

    /** \returns the whole state predicted given a particle at \a pinned with \a conditional over the
     * rest, through \a propagate (the whole state's step) by \a method with \a inputs, with the
     * process noise \a noise added
     */
    template <typename M, typename NoiseDerived, typename Propagate, typename PinnedVector, typename... Inputs>
    [[nodiscard]] GaussianParams<Space_, Scalar_>
    predictWhole(
        M const & method,
        Eigen::MatrixBase<NoiseDerived> const & noise,
        Propagate const & propagate,
        PinnedVector const & pinned,
        GaussianParams<RestSpace, Scalar_> const & conditional,
        Inputs const &... inputs
    ) const
    {
        auto const & split = m_state.m_split;
        auto const f = internal::PinnedStep<Propagate, std::remove_cvref_t<decltype(split)>, PinnedVector>{
            propagate,
            split,
            pinned
        };
        auto predicted = internal::transformBy(method, f, m_state.space(), conditional, inputs...);
        predicted.setCovariance(predicted.covarianceLower() + noise);
        return predicted;
    }

    /** Moves each particle's value and hybrid forward: the body of every form of predict() under a
     * switching model. \a stepOf(model, method, predictBy) calls \a predictBy(method, noise,
     * propagate, inputs...) with one mode's prediction method, which is the model's own where it
     * carries one and the filter's otherwise [switching-models], process noise, whole state's step,
     * and uncertain inputs.
     *
     * The standard normal draws are taken first, one vector per particle. Then, particle by
     * particle, each mode's component is mixed through the chain as ImmRecursion::predict() mixes it,
     * a mode is drawn from the predicted mode probabilities with one uniform draw, and the whole
     * state is predicted given the particle through each mode's model, the drawn mode's first. The
     * particle's new value is drawn from the drawn mode's prediction, every mode's %Gaussian over the
     * rest is conditioned on it, and every mode's probability is multiplied by the density of the
     * draw under that mode's prediction. The new hybrids are built in the posterior store and swapped
     * in once every particle has moved.
     *
     * \returns nothing, or NotPositiveDefinite if a mode the chain reaches predicts the sampled block
     * with a covariance that is not positive definite, with the state unchanged
     */
    template <typename G, typename StepOf>
    [[nodiscard]] std::expected<void, NumericError>
    moveHybridParticles(G & rng, StepOf const & stepOf)
    {
        using PinnedSpace = typename StateType::PinnedSpace;
        using PinnedSamples = SampleSet<PinnedSpace, Scalar_>;
        using PinnedNormals = SampleSet<DifferenceSpace<PinnedSpace>, Scalar_>;
        using PinnedVector = typename Moments<PinnedSpace, Scalar_>::VectorType;
        using RestVector = typename Moments<RestSpace, Scalar_>::VectorType;
        using ComponentType = typename ConditionalType::ComponentType;
        using ModesType = typename ConditionalType::ModesType;
        using Whole = GaussianParams<Space_, Scalar_>;
        constexpr int PinnedValues = valueDimensionAtCompileTime<PinnedSpace>;
        constexpr Scalar_ none = -Eigen::NumTraits<Scalar_>::infinity();
        auto const & split = m_state.m_split;
        BlockRows const rows = split.pinnedRows();
        Eigen::Index const count = m_state.size();
        Eigen::Index const normalSize = split.pinnedDifferenceRows().size;
        m_normals.resize(std::size_t(normalSize * count));
        auto z = Eigen::Map<PinnedNormals>{m_normals.data(), normalSize, count};
        internal::fillStandardNormal(z, rng);
        m_moved.resize(std::size_t(rows.size * count));
        auto moved = Eigen::Map<PinnedSamples>{m_moved.data(), rows.size, count};
        if (m_posterior.size() != count)
        {
            m_posterior = m_state.m_conditionals;
        }
        auto const & model = m_steps.model();
        auto const & transitions = model.chain().matrix();
        // Under the identity chain, each mode starts from its own component, as ImmRecursion's does.
        bool const mixes = !transitions.isIdentity(Scalar_(0));
        Eigen::Index const modes = modeCount();
        auto logMixing = typename ModesType::ArrayType{};
        logMixing.resize(modes);
        auto logLikelihoods = typename ModesType::ArrayType{};
        logLikelihoods.resize(modes);
        for (Eigen::Index i = 0; i < count; ++i)
        {
            auto const & prior = m_state.m_conditionals.conditional(i);
            auto & posterior = m_posterior.conditional(i);
            PinnedVector const pinned = m_state.particle(i);
            ModesType predictedModes = mixes ? Emrys::predict(prior.modes(), model.chain()) : prior.modes();
            auto const logTotal = predictedModes.logTotalWeight();
            if (!logTotal)
            {
                return std::unexpected{logTotal.error()};
            }
            Eigen::Index const drawn = internal::drawIndex<Scalar_>(
                modes,
                [&](Eigen::Index m) {
                    return Eigen::numext::exp(predictedModes.logWeight(m) - *logTotal);
                },
                rng
            );
            logLikelihoods.setConstant(none);
            for (Eigen::Index pass = 0; pass < modes; ++pass)
            {
                // The drawn mode first, since every mode is conditioned on its draw.
                Eigen::Index const j = pass == 0 ? drawn : (pass - 1 < drawn ? pass - 1 : pass);
                bool const reachable = predictedModes.logWeight(j) > none;
                auto const predictFrom = [&](ComponentType const & start) {
                    auto predicted = std::optional<Whole>{};
                    model.visit(j, m_steps.method(), [&](auto const & mode, auto const & modeMethod) {
                        predicted.emplace(stepOf(
                            mode,
                            modeMethod,
                            [&](auto const & method, auto const & noise, auto const & propagate, auto const &... in) {
                                return predictWhole(method, noise, propagate, pinned, start, in...);
                            }
                        ));
                    });
                    return std::move(*predicted);
                };
                Whole const predicted = [&] {
                    if (!mixes)
                    {
                        return predictFrom(prior.conditional(j));
                    }
                    // A mode the chain can't reach starts from its own component, which is never weighted.
                    for (Eigen::Index m = 0; m < modes; ++m)
                    {
                        logMixing(m) = reachable ? Eigen::numext::log(transitions(m, j)) + prior.modes().logWeight(m)
                                                 : (m == j ? Scalar_(0) : none);
                    }
                    ComponentType const mixed = ComponentTraits<ComponentType>::collapse(
                        modes,
                        [&](Eigen::Index m) -> ComponentType const & {
                            return prior.conditional(m);
                        },
                        [&](Eigen::Index m) {
                            return logMixing(m);
                        }
                    );
                    return predictFrom(mixed);
                }();
                auto conditioning = internal::pinnedConditioning(split, predicted.covarianceLower());
                auto const pinnedMean = internal::middleRowsOf<PinnedValues>(predicted.mean(), rows.start, rows.size);
                auto mean = RestVector{};
                if (pass == 0)
                {
                    internal::drawPinned(split, conditioning, predicted.mean(), z.col(i), moved.col(i), mean);
                }
                auto const offset = Emrys::minus(split.pinnedSpace(), moved.col(i), pinnedMean).eval();
                if (pass != 0)
                {
                    mean = Emrys::plus(
                        split.restSpace(),
                        predicted.mean()(split.restIndices()),
                        internal::multiply(conditioning.gain, offset)
                    );
                }
                if (reachable)
                {
                    auto const logDensity = internal::pinnedLogDensity(split, predicted.covarianceLower(), offset);
                    if (!logDensity)
                    {
                        return std::unexpected{logDensity.error()};
                    }
                    logLikelihoods(j) = *logDensity;
                }
                posterior.setConditional(
                    j,
                    ComponentType{split.restSpace(), std::move(mean), std::move(conditioning.cov)}
                );
            }
            // The drawn mode has a finite density, so the posterior modes define a distribution.
            [[maybe_unused]] auto const evidence = predictedModes.update(logLikelihoods);
            EMRYS_FILTER_ASSERT(evidence && "the draw's mode has a density");
            posterior.setModes(std::move(predictedModes));
        }
        // The particles and their hybrids change here, so a density of the state is stale.
        m_state.m_stamp.touchValues();
        m_state.m_conditionals.swap(m_posterior);
        m_state.m_particles.setStates(moved);
        return {};
    }

    /** Updates each particle's hybrid where it is by \a updateOne(hybrid, pinned, oneResult), with
     * \a oneResult reused from particle to particle, then reweights the particles by their
     * evidence as finish() does: the body of every hybrid update but one by a likelihood alone (see
     * internal::updateHybridsGiven()).
     *
     * \returns nothing, or the first failed particle's error
     */
    template <typename P, typename OneResult, typename UpdateOne, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    updateParticles(
        P && policy,
        OneResult oneResult,
        UpdateOne const & updateOne,
        RbParticleFilterUpdate<Scalar, ResultAllocator> & result
    )
    {
        assertParticlesRunUnder<P>();
        // Each particle's hybrid is updated in place below, so a density of the state is stale.
        m_state.m_stamp.touchValues();
        auto const updated = internal::updateHybridsGiven(
            m_state.points(),
            m_state.m_conditionals,
            internal::PointLikelihoodsAccess::points(result),
            oneResult,
            updateOne
        );
        if (!updated)
        {
            return updated;
        }
        finish<false>(std::forward<P>(policy), result);
        return {};
    }

    /** Reserves the scratch for the particles' capacity, or their number if they have none. */
    void
    reserveScratch()
    {
        Eigen::Index const count = m_state.particles().capacity().value_or(m_state.size());
        m_posterior.reserve(count);
        m_normals.reserve(std::size_t(m_state.pinnedDifferenceRows().size * count));
        m_moved.reserve(std::size_t(m_state.pinnedRows().size * count));
    }

    /** Draws each particle's new value from the prediction given it, by \a propagate (the whole
     * state's step) with \a inputs and the process noise \a noise, and conditions the rest on the
     * draw, the particles a for-each under \a policy: the body of every form of predict().
     */
    template <typename P, typename G, typename Derived, typename Propagate, typename... Inputs>
    void
    moveParticles(
        P && policy,
        G & rng,
        Eigen::MatrixBase<Derived> const & noise,
        Propagate const & propagate,
        Inputs const &... inputs
    )
    {
        using PinnedSamples = SampleSet<typename StateType::PinnedSpace, Scalar_>;
        using PinnedNormals = SampleSet<DifferenceSpace<typename StateType::PinnedSpace>, Scalar_>;
        using PinnedVector = typename Moments<typename StateType::PinnedSpace, Scalar_>::VectorType;
        using WholeMoments = Moments<Space_, Scalar_>;
        using RestMoments = Moments<typename StateType::RestSpace, Scalar_>;
        // The transition's columns for the rest, and their product with a rest covariance.
        using RestColumns = internal::MatrixOf<
            Scalar_,
            WholeMoments::DifferenceDimensionAtCompileTime,
            RestMoments::DifferenceDimensionAtCompileTime,
            WholeMoments::MaxDifferenceDimensionAtCompileTime,
            RestMoments::MaxDifferenceDimensionAtCompileTime>;
        using RestCross = internal::MatrixOf<
            Scalar_,
            RestMoments::DifferenceDimensionAtCompileTime,
            WholeMoments::DifferenceDimensionAtCompileTime,
            RestMoments::MaxDifferenceDimensionAtCompileTime,
            WholeMoments::MaxDifferenceDimensionAtCompileTime>;
        constexpr bool cached = internal::cachesTransition<Model_> && sizeof...(Inputs) == 0;
        auto const & space = m_state.space();
        auto const & split = m_state.m_split;
        Eigen::Index const count = m_state.size();
        Eigen::Index const pinnedSize = split.pinnedRows().size;
        Eigen::Index const normalSize = split.pinnedDifferenceRows().size;
        m_normals.resize(std::size_t(normalSize * count));
        auto z = Eigen::Map<PinnedNormals>{m_normals.data(), normalSize, count};
        internal::fillStandardNormal(z, rng);
        // The particles and their Gaussians change below, so a density of the state is stale.
        m_state.m_stamp.touchValues();
        auto & conditionals = m_state.m_conditionals;
        auto const restColumns = [&] {
            if constexpr (cached)
            {
                return RestColumns{
                    m_steps.model().transition()(Eigen::placeholders::all, split.restDifferenceIndices())
                };
            }
            else
            {
                return internal::NoTransition{};
            }
        }();
        auto const predictGiven = [&](Eigen::Index i) -> GaussianParams<Space_, Scalar_> {
            PinnedVector const pinned = m_state.particle(i);
            auto const & conditional = conditionals.conditional(i);
            if constexpr (cached)
            {
                typename WholeMoments::VectorType mean = propagate(split.assemble(conditional.mean(), pinned));
                normalizeValues(space, mean);
                RestCross const cross = internal::multiply(conditional.covarianceLower(), restColumns.transpose());
                typename WholeMoments::MatrixType cov = internal::multiply(restColumns, cross);
                cov = (Scalar_(0.5) * (cov + cov.transpose()) + noise).eval();
                return {space, std::move(mean), std::move(cov)};
            }
            else
            {
                return predictWhole(m_steps.method(), noise, propagate, pinned, conditional, inputs...);
            }
        };

        m_moved.resize(std::size_t(pinnedSize * count));
        auto moved = Eigen::Map<PinnedSamples>{m_moved.data(), pinnedSize, count};
        if constexpr (Shared)
        {
            auto const conditioning =
                internal::pinnedConditioning(split, predictGiven(m_state.mostProbable()).covarianceLower());
            auto means = conditionals.means();
            internal::forEachIndex(std::forward<P>(policy), count, [&](Eigen::Index i) {
                auto mean = propagate(split.assemble(means.col(i), m_state.particle(i)), internal::meanOf(inputs)...);
                normalizeValues(space, mean);
                internal::drawPinned(split, conditioning, mean, z.col(i), moved.col(i), means.col(i));
            });
            conditionals.sharedCovariance() = conditioning.cov;
        }
        else
        {
            internal::forEachIndex(std::forward<P>(policy), count, [&](Eigen::Index i) {
                auto mean = typename Moments<typename StateType::RestSpace, Scalar_>::VectorType{};
                auto const predicted = predictGiven(i);
                auto conditioning = internal::pinnedConditioning(split, predicted.covarianceLower());
                internal::drawPinned(split, conditioning, predicted.mean(), z.col(i), moved.col(i), mean);
                conditionals.conditional(i) =
                    ConditionalType{split.restSpace(), std::move(mean), std::move(conditioning.cov)};
            });
        }
        m_state.m_particles.setStates(moved);
    }

    /** Reweights the particles by the likelihoods just computed into \a result, under \a policy,
     * swaps the posterior Gaussians in where \a Posterior says the update wrote them there, and
     * records the total in \a result. An update that changes no Gaussian, as a measurement weighted
     * by its own likelihood does, passes false, and the Gaussians stay where they are.
     */
    template <bool Posterior = true, typename P, typename ResultAllocator>
    void
    finish(P && policy, RbParticleFilterUpdate<Scalar, ResultAllocator> & result)
    {
        auto & points = internal::PointLikelihoodsAccess::points(result);
        auto const [logLikelihood, accepted] = [&] {
            if constexpr (Posterior)
            {
                return m_state.reweight(std::forward<P>(policy), points.logLikelihoods(), m_posterior);
            }
            else
            {
                return m_state.reweight(std::forward<P>(policy), points.logLikelihoods());
            }
        }();
        internal::UpdateAccess::logLikelihood(result) = logLikelihood;
        result.accepted = accepted;
    }

    /** The particles' steps: the model and the method, or under a switching model the IMM
     * recursion, which holds them and the scratch of the hybrid updates.
     */
    using StepsType = std::conditional_t<
        Switching,
        ImmRecursion<Gaussian<RestSpace, Scalar_>, Model_, Method_, Allocator_>,
        internal::PointSteps<Model_, Method_>>;

    StepsType m_steps;  ///< the process model over the whole state and the prediction method
    StateType m_state;  ///< the particles and their distributions

    // Scratch, reused across steps.

    typename StateType::ConditionalsType m_posterior;  ///< the updated or predicted distributions
    internal::Buffer<Scalar_, Allocator_> m_normals;   ///< standard normal draws, one column per particle
    internal::Buffer<Scalar_, Allocator_> m_moved;     ///< the particles' new values
};

/// \cond DEDUCTION_GUIDES
// One guide with a defaulted method rather than one per arity, its parameters in the class's
// order: otherwise MSVC deduces the scalar as the method. The state's component follows from the
// model, so it is not carried over.
template <
    typename Space,
    typename Path,
    typename Model,
    typename Covariance,
    typename Method = DefaultMethodFor<Model>,
    typename Scalar,
    typename Allocator,
    typename Component>
    requires(!internal::UpkeepArgument<Method>)
RbParticleFilter(Model, ParticleGaussianParams<Space, Path, Covariance, Scalar, Allocator, Component>, Method = {})
    -> RbParticleFilter<Space, Path, internal::BoundStep<Model, Space, Scalar>, Covariance, Method, Scalar, Allocator>;

template <
    typename Space,
    typename Path,
    typename Model,
    typename Covariance,
    typename Scalar,
    typename Allocator,
    typename Component,
    typename U>
    requires internal::UpkeepArgument<U>
RbParticleFilter(Model, ParticleGaussianParams<Space, Path, Covariance, Scalar, Allocator, Component>, U)
    -> RbParticleFilter<
        Space,
        Path,
        internal::BoundStep<Model, Space, Scalar>,
        Covariance,
        DefaultMethodFor<Model>,
        Scalar,
        Allocator,
        U>;

template <
    typename Space,
    typename Path,
    typename Model,
    typename Covariance,
    typename Method,
    typename Scalar,
    typename Allocator,
    typename Component,
    typename U>
RbParticleFilter(
    Model,
    ParticleGaussianParams<Space, Path, Covariance, Scalar, Allocator, Component>,
    Method,
    U
) -> RbParticleFilter<Space, Path, internal::BoundStep<Model, Space, Scalar>, Covariance, Method, Scalar, Allocator, U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <
    typename Space_,
    typename Path_,
    typename Model_,
    typename Covariance_,
    typename Method_,
    typename Scalar_,
    typename Allocator_,
    typename Upkeep_>
void
describe(
    std::ostream & stream,
    RbParticleFilter<Space_, Path_, Model_, Covariance_, Method_, Scalar_, Allocator_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "Rao-Blackwellized particle filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

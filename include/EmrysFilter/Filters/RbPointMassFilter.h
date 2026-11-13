#pragma once

/** \file
 * RbPointMassFilter (decisions.md [rb-point-mass] and [rao-blackwellized-core]): fixed points for one block of the
 * state, each carrying a Kalman filter over the rest, or an IMM filter under a switching model, with the
 * PointMassFilter alias for a grid that covers the whole state. The last header; read
 * docs/guide/rao-blackwellized-filters.md for the review.
 */

#include <expected>
#include <span>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Filters/FilterUpkeep.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Filters/RbUpdate.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Core/Print.h>

namespace Emrys
{

namespace internal
{

/** A cell's distribution under \a Model over the rest of \a Grid's state: a %Gaussian, or a hybrid
 * over the switching model's modes.
 */
template <typename Grid, typename Model, bool = isSwitchingPointModel<Model>>
struct CellComponentOf
{
    using type = Gaussian<typename Grid::RestSpace, typename Grid::Scalar>;  ///< the component
};

/** A switching model's cells each hold a hybrid of Gaussians, from the grid's allocator. */
template <typename Grid, typename Model>
struct CellComponentOf<Grid, Model, true>
{
    /// the component
    using type = Hybrid<
        Gaussian<typename Grid::RestSpace, typename Grid::Scalar>,
        typename Model::ModeSpaceType,
        typename Grid::allocator_type>;
};

}  // namespace internal

/** \ingroup Filters
 * \brief What a Rao-Blackwellized point-mass filter's update did, per live cell and in total.
 *
 * The arrays have one entry per cell that was alive when the update began, in the order of
 * cells(). A cell whose update the policy rejected has log-likelihood \f$ -\infty \f$ and NIS
 * NaN, and has been removed from the state.
 *
 * The result owns its cells and arrays, in storage from \a Allocator. An update into a result
 * (RbPointMassFilter::update(measurement, result)) refills that storage, so a result kept across
 * steps allocates nothing once it has held as many cells (see \ref Allocation).
 */
template <typename Scalar = double, typename Allocator = DefaultAllocator>
class RbPointMassFilterUpdate
{
public:

    using ArrayType = Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 1> const>;  ///< a per-cell array
    using allocator_type = Allocator;  ///< where the cells and per-cell arrays come from

    /** An empty result whose cells and arrays come from \a allocator. */
    explicit RbPointMassFilterUpdate(Allocator const & allocator = {})
        : m_cells(internal::ReBound<Allocator, Eigen::Index>(allocator))
        , m_points{allocator}
    {}

    /** \returns the flat index of each cell updated */
    [[nodiscard]] std::span<Eigen::Index const>
    cells() const
    {
        return {m_cells.data(), m_cells.size()};
    }

    /** \returns \f$ \log p(z \mid c) \f$ per cell */
    [[nodiscard]] ArrayType
    logLikelihoods() const
    {
        return m_points.logLikelihoods();
    }

    /** \returns each cell's NIS; with mixture noise, its most probable branch's, and with a hybrid
     * per cell, its most probable mode's after the update
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

    /** \returns the allocator the cells and per-cell arrays come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_cells.get_allocator());
    }

    /** \returns \f$ \log p(z) = \log \sum_c P(c) \, p(z \mid c) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return m_logLikelihood;
    }

    bool accepted = false;  ///< whether any cell accepted the measurement

private:

    friend struct internal::PointLikelihoodsAccess;
    friend struct internal::UpdateAccess;

    Scalar m_logLikelihood = 0;  ///< the observation's log-likelihood

    internal::Buffer<Eigen::Index, Allocator> m_cells;       ///< the cells updated
    internal::PointLikelihoods<Scalar, Allocator> m_points;  ///< the per-cell arrays
};

/** \ingroup Filters
 * \brief A Rao-Blackwellized point-mass filter: fixed points for one block of the state, such as
 * a coarse position, each carrying a Kalman filter over the rest.
 *
 * The state is a GridGaussianParams, whose density, a GridGaussian, is
 * \f$ p(x) = \sum_c P(c) \, \delta(x_p - p_c) \, \mathcal{N}(x_r; \mu_c, \Sigma_c) \f$:
 * the block \f$ x_p \f$ named by the grid's path sits at the cell's center \f$ p_c \f$, and the
 * rest \f$ x_r \f$ is %Gaussian given the cell. It is a Rao-Blackwellized particle filter whose
 * particles never move: the points are fixed, the process model moves only the rest, and no
 * probability passes between cells. predict() moves each cell's %Gaussian through the process
 * model, which covers the rest of the state. update() updates each cell's %Gaussian given its
 * center (see update(prior, measurement, given)) and multiplies each cell's probability by
 * \f$ p(z \mid c) \f$, so measurements nonlinear in the fixed block, such as ranges or a terrain
 * map, weigh the cells exactly, while the rest gets Kalman updates.
 *
 * With \a Covariance_ SharedCovariance, every cell has the same covariance: one transform, at the
 * most probable cell, gives the innovation covariance and gain for all of them, and each cell
 * costs an evaluation of the measurement function and a mean update. This is exact when the dynamics and
 * the measurement are affine in the rest of the state with the same Jacobian in every cell; the
 * measurement may depend on the fixed block arbitrarily. Its noise must be one %Gaussian. With
 * PerPointCovariance, each cell is its own Kalman filter, its measurement transformed and its
 * branches reduced by a policy (Collapse, Select, or a Gate) as a KalmanFilter's are. A
 * measurement known only by its likelihood of the fixed block (a model with logLikelihood(g) and
 * no function) multiplies each cell's probability by it and leaves the Gaussians alone.
 *
 * A cell's covariance about its center (see PointGrid) enters each update as an independent
 * uncertain input: it widens the innovation by the measurement's sensitivity to the fixed block,
 * which the measurement's method computes, so a measurement nonlinear in that block needs
 * Linearized or Unscented, as any measurement does. The offset from the center is not
 * tracked, so successive innovations are treated as independent although the offset they share
 * correlates them. Cells with covariances of their own (PointGrid with PerPointCovariance) widen
 * each cell's innovation differently, so they need a PerPointCovariance filter. With
 * SharedCovariance, one cell covariance shared by all is exact only if the measurement's
 * sensitivity to the fixed block is the same in every cell; otherwise the most probable cell's
 * stands for all.
 *
 * Only live cells are stored, and a cell whose probability becomes zero is removed. The caller
 * keeps the number of live cells in bounds through state() (see \ref Upkeep): PruneBelow,
 * PruneTail, and PruneToSize remove the improbable cells, and Subdivide splits every live cell of a regular grid, to
 * resolve the fixed block more finely where the probability has gathered.
 *
 * With \a Model_ a SwitchingModel over the rest, each cell holds a HybridGaussian over the rest,
 * an IMM filter's state, so each cell is an IMM filter given its center: predict() mixes each
 * cell's modes, predicts each through its mode's model, and steps the chain, and update() updates
 * each cell as ImmRecursion::update(state, measurement, given, policy) does and multiplies each
 * cell's probability by \f$ p(z \mid c) = \sum_m P(m \mid c) \, p(z \mid m, c) \f$, so noise by
 * mode axis (ModeNoise) updates each mode with the component its value names, as in an ImmFilter.
 * The covariance form is then PerPointCovariance, and the cells' steps share the recursion's scratch,
 * so they run one at a time: predict() and update() take Sequential alone, except an update by a
 * likelihood alone, which leaves the hybrids as they were. state().modes() sums the cells' modes.
 * No publication found states this form: it composes the IMM filter, given a cell's center, with
 * the Rao-Blackwellized point-mass filter, and is one of two exceptions to
 * [published-algorithms-only].
 *
 * The Rao-Blackwellized point-mass filter of \ref Smidl2013 "Šmídl and Gašperin (2013)" and
 * \ref Dunik2019 "Duník et al. (2019)", as given in \ref Pocoma2023 "Pocoma et al. (2023)",
 * §II-C and §II-D: a grid over one block and a Kalman filter per cell. The update given each
 * cell is the Rao-Blackwellized update the particle filters share (see RbUpdate.h); where the
 * measurement does not depend on the rest it reduces to Pocoma et al.'s correction, Algorithm 1,
 * equations (17) and (18). It departs from the published filter in its prediction, which moves
 * only the rest: the published one also moves probability between cells, with the rest as noise
 * on the block (§III-D) [rb-point-mass]. The cell covariance about each center has no source and
 * is the other exception to [published-algorithms-only].
 *
 * The state and the filter's scratch come from the grid's allocator (see \ref Allocation), and the
 * scratch is reused, so steps allocate nothing once one has run with as many live cells, except
 * that an update of hybrid cells allocates the IMM result it reuses from cell to cell.
 *
 * \code
 * using State = Composite<Position, ClockBias>;
 * auto const grid = regularGrid<State, Position>(first, spacing, {200, 200});
 * auto filter = RbPointMassFilter{RandomWalk<ClockBias>{0.01}, grid, initialBias};
 * filter.predict(dt);
 * auto const updated = filter.update(range);  // reads Position and ClockBias
 * filter.state().marginal<GridAxis<Position, 0>>();
 * filter.state().toDensity()->collapsed();    // the whole state as one GaussianParams
 * upkeep::pruneBelow(1e-12)(filter.state());
 * \endcode
 *
 * \tparam Grid_        the fixed points (see PointGrid)
 * \tparam Model_       the process model over the rest of the state, such as a ProcessModel, or
 *                      a FixedStepModel over the grid's restSpace(), with which predict() takes
 *                      no time step; a SwitchingModel gives each cell a hybrid
 * \tparam Covariance_  SharedCovariance or PerPointCovariance; PerPointCovariance under a
 *                      SwitchingModel
 * \tparam Method_      the prediction method; a FixedStepModel's own; under a SwitchingModel,
 *                      that of each mode whose model carries none of its own (see withMethod()),
 *                      and ModelsMethod with fixed-step models
 * \tparam Upkeep_      the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                      one, such as pruning and subdividing
 *
 * \sa GridGaussianParams, PointGrid, update(prior, measurement, given), \ref Filters
 */
template <
    typename Grid_,
    typename Model_,
    typename Covariance_ = internal::PointCovarianceFor<Model_>,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Upkeep_ = upkeep::None>
class RbPointMassFilter
    : public internal::HeldUpkeep<RbPointMassFilter<Grid_, Model_, Covariance_, Method_, Upkeep_>, Upkeep_>
{
    using UpkeepBase = internal::HeldUpkeep<RbPointMassFilter, Upkeep_>;  ///< what holds the upkeep

    static constexpr bool Shared = std::is_same_v<Covariance_, SharedCovariance>;  ///< one covariance for all cells

    /** Whether each cell holds a hybrid, an IMM filter's state, under a switching model. */
    static constexpr bool Switching = internal::isSwitchingPointModel<Model_>;

    /** A compile-time error unless the cells' steps may run under \a P. */
    template <typename P>
    static constexpr void
    assertCellsRunUnder()
    {
        static_assert(
            !Switching || internal::isSequential<P>,
            "a cell's IMM step writes the recursion's scratch, so the cells of a hybrid grid run in order: pass "
            "Sequential"
        );
    }

    static_assert(
        internal::predictsBy<Model_, Method_>,
        "a filter with a fixed-step model predicts by the model's method"
    );
    static_assert(internal::checkPredictionMethod<Method_>());

public:

    using GridType = Grid_;                       ///< the fixed points
    using Scalar = typename Grid_::Scalar;        ///< the scalar type
    using SpaceType = typename Grid_::SpaceType;  ///< the state space
    using ModelType = Model_;                     ///< the process model
    using MethodType = Method_;                   ///< the prediction method
    /** The state the filter owns: a %Gaussian per cell, or a hybrid under a switching model. */
    using StateType = GridGaussianParams<Grid_, Covariance_, typename internal::CellComponentOf<Grid_, Model_>::type>;
    using ConditionalType = typename StateType::ConditionalType;  ///< a cell's distribution over the rest
    using allocator_type = typename StateType::allocator_type;    ///< the state's allocator

    /** A filter with process model \a model over the rest of the state, points \a grid,
     * \a conditional as every cell's %Gaussian over the rest, or its hybrid under a switching
     * model, with every cell equally probable, prediction method \a method, which a fixed-step
     * model carries instead, and upkeep \a upkeep, which maintain() runs. The model is not deduced
     * from here, so that the deduction guide below, which binds fixedStep()'s result to the rest of
     * the state, deduces the filter's arguments.
     */
    RbPointMassFilter(
        std::type_identity_t<Model_> model,
        Grid_ grid,
        ConditionalType const & conditional,
        Covariance_ = {},
        MethodFor<Model_, Method_> method = {}
    )
        : RbPointMassFilter{std::move(model), std::move(grid), conditional, Covariance_{}, std::move(method), Upkeep_{}}
    {}

    /** As RbPointMassFilter(model, grid, conditional, covariance, method), with upkeep \a upkeep, which
     * maintain() runs.
     */
    RbPointMassFilter(
        std::type_identity_t<Model_> model,
        Grid_ grid,
        ConditionalType const & conditional,
        Covariance_,
        MethodFor<Model_, Method_> method,
        std::type_identity_t<Upkeep_> upkeep
    )
        : UpkeepBase{std::move(upkeep)}
        , m_steps{std::move(model), std::move(method), grid.get_allocator()}
        , m_state{std::move(grid), conditional}
        , m_posterior{m_state.m_conditionals, m_state.get_allocator()}
    {
        assertModelPartitions(m_state);
        reserve();
    }

    /** As RbPointMassFilter(model, grid, conditional), with upkeep \a upkeep, which maintain() runs. */
    RbPointMassFilter(
        std::type_identity_t<Model_> model,
        Grid_ grid,
        ConditionalType const & conditional,
        std::type_identity_t<Upkeep_> upkeep
    )
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : RbPointMassFilter{
              std::move(model),
              std::move(grid),
              conditional,
              Covariance_{},
              MethodFor<Model_, Method_>{},
              std::move(upkeep)
          }
    {}

    /** As RbPointMassFilter(model, grid, conditional, covariance), with upkeep \a upkeep, which
     * maintain() runs.
     */
    RbPointMassFilter(
        std::type_identity_t<Model_> model,
        Grid_ grid,
        ConditionalType const & conditional,
        Covariance_ covariance,
        std::type_identity_t<Upkeep_> upkeep
    )
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : RbPointMassFilter{
              std::move(model),
              std::move(grid),
              conditional,
              covariance,
              MethodFor<Model_, Method_>{},
              std::move(upkeep)
          }
    {}

    /** A filter whose points cover the whole state (see PointMassFilter), with every cell equally
     * probable.
     */
    explicit RbPointMassFilter(Grid_ grid, Covariance_ = {})
        requires(std::is_same_v<typename Grid_::RestSpace, Linear<0>> && std::is_same_v<Model_, NoProcessModel>)
        : UpkeepBase{Upkeep_{}}
        , m_state{std::move(grid), ConditionalType{}}
        , m_posterior{m_state.m_conditionals, m_state.get_allocator()}
    {
        reserve();
    }

    /** As RbPointMassFilter(grid, covariance), with upkeep \a upkeep, which maintain() runs. */
    RbPointMassFilter(Grid_ grid, Covariance_, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None> && std::is_same_v<Model_, NoProcessModel>
                 && std::is_same_v<typename Grid_::RestSpace, Linear<0>>)
        : UpkeepBase{std::move(upkeep)}
        , m_state{std::move(grid), ConditionalType{}}
        , m_posterior{m_state.m_conditionals, m_state.get_allocator()}
    {
        reserve();
    }

    /** As RbPointMassFilter(grid), with upkeep \a upkeep, which maintain() runs. */
    RbPointMassFilter(Grid_ grid, std::type_identity_t<Upkeep_> upkeep)
        requires(
            !std::same_as<Upkeep_, Emrys::upkeep::None> && std::is_same_v<Model_, NoProcessModel>
            && std::is_same_v<typename Grid_::RestSpace, Linear<0>>
        )
        : RbPointMassFilter{std::move(grid), Covariance_{}, std::move(upkeep)}
    {}

    /** \returns the allocator of the state and scratch: the grid's */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return m_state.get_allocator();
    }

    /** \returns the live cells, their probabilities, and their Gaussians */
    [[nodiscard]] StateType const &
    state() const
    {
        return m_state;
    }

    /** \returns the state, to change where it is between steps, as upkeep such as pruning or
     * subdividing does. It must keep its space and allocator; setState() replaces it with another,
     * and checks it.
     */
    [[nodiscard]] StateType &
    state()
    {
        return m_state;
    }

    /** Replaces the state, as when reinitializing, with one whose rest space the process model's
     * writes partition, as the constructor's must. The scratch follows the new state's allocator
     * and its number of live cells.
     */
    void
    setState(StateType state)
    {
        if constexpr (!std::is_same_v<Model_, NoProcessModel>)
        {
            assertModelPartitions(state);
        }
        auto const before = m_state.get_allocator();
        m_state = std::move(state);
        if (m_state.get_allocator() != before)
        {
            // The assignment carried the new state's allocator in, so the scratch follows it.
            m_posterior = decltype(m_posterior){m_state.m_conditionals, m_state.get_allocator()};
            if constexpr (Switching)
            {
                m_steps.resetScratch(m_state.get_allocator());
            }
        }
        reserve();
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

    /** Moves each live cell's %Gaussian forward by \a dt, any TimeStep (see \ref Time), through the
     * process model, with \a inputs passed as in Emrys::predict(). The points and their
     * probabilities don't change.
     */
    template <typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict(TimeStep<Scalar> auto const & dt, Inputs const &... inputs)
    {
        predict(execution::sequential, dt, inputs...);
    }

    /** predict(dt, inputs...) under \a policy, a for-each over the live cells, so the result is
     * Sequential's (see \ref Execution). The process model's function then runs for several
     * cells at once. A PointMassFilter, whose points cover the whole state, has nothing to move.
     * Hybrid cells take Sequential alone.
     */
    template <ExecutionPolicy P, typename... Inputs>
        requires(!isFixedStep<Model_>)
    void
    predict([[maybe_unused]] P && policy, TimeStep<Scalar> auto const & dt, [[maybe_unused]] Inputs const &... inputs)
    {
        [[maybe_unused]] Scalar const step = internal::toScalarTime<Scalar>(dt);
        if constexpr (Switching)
        {
            assertCellsRunUnder<P>();
            m_state.m_stamp.touchValues();
            for (Eigen::Index i = 0; i < m_state.size(); ++i)
            {
                m_steps.predict(m_state.m_conditionals.conditional(i), step, inputs...);
            }
        }
        else if constexpr (!std::is_same_v<Model_, NoProcessModel>)
        {
            moveCells(
                std::forward<P>(policy),
                [&](auto const & cell) {
                    return Emrys::predict(m_steps.method(), cell, m_steps.model(), step, inputs...);
                },
                [&](auto const & mean) {
                    return Emrys::propagate(m_state.grid().restSpace(), mean, m_steps.model(), step, inputs...);
                }
            );
        }
        else
        {
            static_assert(sizeof...(Inputs) == 0, "a point-mass filter over the whole state takes no input");
        }
    }

    /** Moves each live cell's %Gaussian forward by the fixed-step model's step, with \a inputs
     * passed as in Emrys::predict(). The points and their probabilities don't change.
     */
    template <typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithPolicy<Inputs...> && !internal::leadsWithSteps<Inputs...>)
    void
    predict(Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        predict(execution::sequential, inputs...);
    }

    /** predict(inputs...) under \a policy, as predict(policy, dt, inputs...) runs. */
    template <ExecutionPolicy P, typename... Inputs>
        requires(isFixedStep<Model_> && !internal::leadsWithSteps<Inputs...>)
    void
    predict(P && policy, Inputs const &... inputs)
    {
        internal::rejectTimeStep<Inputs...>();
        if constexpr (Switching)
        {
            assertCellsRunUnder<P>();
            m_state.m_stamp.touchValues();
            for (Eigen::Index i = 0; i < m_state.size(); ++i)
            {
                m_steps.predict(m_state.m_conditionals.conditional(i), inputs...);
            }
        }
        else
        {
            moveCells(
                std::forward<P>(policy),
                [&](auto const & cell) {
                    return Emrys::predict(cell, m_steps.model(), inputs...);
                },
                [&](auto const & mean) {
                    return Emrys::propagate(
                        m_state.grid().restSpace(),
                        mean,
                        m_steps.model().model(),
                        m_steps.model().timeStep(),
                        inputs...
                    );
                }
            );
        }
    }

    /** Moves each live cell's %Gaussian forward by \a steps of the fixed-step model, as that many
     * calls of predict(inputs...) do, with \a inputs at every step.
     */
    template <typename... Inputs>
        requires isFixedStep<Model_>
    void
    predict(Steps steps, Inputs const &... inputs)
    {
        predict(execution::sequential, steps, inputs...);
    }

    /** predict(steps, inputs...) under \a policy, as that many calls of predict(policy, inputs...)
     * run.
     */
    template <ExecutionPolicy P, typename... Inputs>
        requires isFixedStep<Model_>
    void
    predict(P && policy, Steps steps, Inputs const &... inputs)
    {
        EMRYS_FILTER_ASSERT(steps.count >= 0 && "the number of steps must be nonnegative");
        for (Eigen::Index i = 0; i < steps.count; ++i)
        {
            predict(policy, inputs...);
        }
    }

    /** The result of update(): the cells and per-cell arrays from the filter's allocator. */
    using UpdateType = RbPointMassFilterUpdate<Scalar, allocator_type>;

    /** Conditions each live cell's %Gaussian on \a measurement given the cell's center, by the
     * measurement's own method, and multiplies each cell's probability by \f$ p(z \mid c) \f$.
     * With SharedCovariance, the measurement's noise must be one %Gaussian; with
     * PerPointCovariance, each cell's branches are collapsed, as update(measurement, policy) does
     * with Collapse. If every cell has likelihood zero, the state is unchanged.
     *
     * A measurement whose model supplies its own logLikelihood(g), with or without a function,
     * instead multiplies each cell's probability by that likelihood at the cell's center and leaves every
     * cell's %Gaussian as it was (decisions.md [likelihood-measurements]). Every block it reads must
     * lie in the grid's block, which is checked at compile time: given the rest, the rest would no
     * longer be %Gaussian. Its result has NIS NaN and an innovation of size 0. A cell's covariance
     * about its center is ignored, so a likelihood narrower than the cells needs finer cells, from
     * Subdivide.
     *
     * \returns the cells updated, each one's log-likelihood and NIS, the total log-likelihood, and
     * whether the measurement was accepted (see RbPointMassFilterUpdate), in storage from the
     * filter's allocator; or, if a cell's update failed, as one whose innovation covariance is not
     * positive definite does, its error (decisions.md [innovation-cholesky]), with the state
     * unchanged, or with a hybrid per cell, which is updated where it is, with the cells before the
     * failed one updated and the probabilities as they were [update-outcomes]
     */
    template <internal::WeighsPoints M>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(M const & measurement)
    {
        return update(execution::sequential, measurement);
    }

    /** update() under \a executionPolicy: each cell's update is a for-each and the evidence a
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
     * result kept across steps allocates nothing once it has held as many cells.
     *
     * \returns nothing, or the error update(measurement) would return
     */
    template <internal::WeighsPoints M, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(M const & measurement, RbPointMassFilterUpdate<Scalar, ResultAllocator> & result)
    {
        return update(execution::sequential, measurement, result);
    }

    /** update(measurement, result) under \a executionPolicy, as update(executionPolicy,
     * measurement) runs.
     */
    template <ExecutionPolicy P, internal::WeighsPoints M, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    update(P && executionPolicy, M const & measurement, RbPointMassFilterUpdate<Scalar, ResultAllocator> & result)
    {
        if constexpr (!internal::hasOwnLikelihood<M> && !Shared)
        {
            return update(std::forward<P>(executionPolicy), measurement, reductions::collapse(), result);
        }
        else
        {
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

    /** As update(measurement), with PerPointCovariance, but with each cell's branches reduced by
     * \a policy (Collapse, Select, or a Gate); a cell whose update the policy rejects is removed.
     * A measurement weighted by its own likelihood has no branches, so it takes no policy.
     */
    template <internal::WeighsPoints M, typename Policy>
    [[nodiscard]] std::expected<UpdateType, NumericError>
    update(M const & measurement, Policy const & policy)
        requires(!Shared)
    {
        return update(execution::sequential, measurement, policy);
    }

    /** update(measurement, policy) under \a executionPolicy: each cell's Kalman update and
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
    update(M const & measurement, Policy const & policy, RbPointMassFilterUpdate<Scalar, ResultAllocator> & result)
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
        RbPointMassFilterUpdate<Scalar, ResultAllocator> & result
    )
        requires(!Shared)
    {
        static_assert(
            !internal::hasOwnLikelihood<M>,
            "a measurement weighted by its own logLikelihood() has no branches to reduce, so it takes no policy"
        );
        if constexpr (Switching && internal::isByMode<M>)
        {
            // Each mode's measurement holds its own noise component, so the recursion resolves it.
            using PathType = typename Grid_::PathType;
            return updateCells(
                std::forward<P>(executionPolicy),
                m_steps.template emptyGivenUpdate<M, SpaceType, PathType>(get_allocator(), modeCount()),
                [&](auto & cell, auto const & pinned, auto & cellResult) {
                    using Value = std::remove_cvref_t<decltype(pinned)>;
                    return m_steps.update(
                        cell,
                        measurement,
                        GivenBlock<SpaceType, PathType, Value>{m_state.grid().space(), pinned},
                        policy,
                        cellResult
                    );
                },
                result
            );
        }
        else if constexpr (Switching)
        {
            auto const f = internal::PinnedMeasurement<SpaceType, typename Grid_::PathType, M>{
                m_state.grid().space(),
                measurement
            };
            return updateCells(
                std::forward<P>(executionPolicy),
                m_steps.template emptyGivenUpdate<M, SpaceType, typename Grid_::PathType>(get_allocator(), modeCount()),
                [&](auto & cell, auto const & pinned, auto & cellResult) {
                    return m_steps.updateGiven(cell, f, pinned, policy, cellResult);
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

    /** Asserts that the process model's writes, or every mode's under a switching model, partition
     * the rest of \a state's space: the check of the constructor that takes a model, and of
     * setState().
     */
    void
    assertModelPartitions([[maybe_unused]] StateType const & state) const
    {
        if constexpr (Switching)
        {
            EMRYS_FILTER_ASSERT(
                m_steps.partitions(state.grid().restSpace())
                && "every mode's process model must write every coordinate of the rest of the state once"
            );
        }
        else
        {
            internal::assertPartitions(
                m_steps.model(),
                state.grid().restSpace(),
                "the process model must write every coordinate of the rest of the state once"
            );
        }
    }

    /** \returns the number of modes of a hybrid cell */
    [[nodiscard]] Eigen::Index
    modeCount() const
        requires Switching
    {
        return m_steps.modeSpace().cardinality();
    }

    /** Updates each live cell's hybrid where it is by \a updateCell(cell, pinned, cellResult), with
     * the cell's center or its kernel as \a pinned and \a cellResult reused from cell to cell, then
     * reweights the cells by their evidence as finish() does. A cell no mode accepts keeps its
     * hybrid, which the recursion leaves as it was, and gets likelihood zero, so a measurement no
     * cell accepts leaves the state as it was. Nothing is copied to undo a failure (decisions.md
     * [update-outcomes]): a cell whose update fails keeps its hybrid, and the cells before it keep
     * theirs updated, with the probabilities as they were.
     *
     * \returns nothing, or the first failed cell's error
     */
    template <typename P, typename CellResult, typename UpdateCell, typename ResultAllocator>
    [[nodiscard]] std::expected<void, NumericError>
    updateCells(
        P && policy,
        CellResult cellResult,
        UpdateCell const & updateCell,
        RbPointMassFilterUpdate<Scalar, ResultAllocator> & result
    )
    {
        assertCellsRunUnder<P>();
        // Each cell's hybrid is updated in place below, so a density of the state is stale.
        m_state.m_stamp.touchValues();
        auto const updated = internal::updateHybridsGiven(
            m_state.points(),
            m_state.m_conditionals,
            internal::PointLikelihoodsAccess::points(result),
            cellResult,
            updateCell
        );
        if (!updated)
        {
            return updated;
        }
        finish<false>(std::forward<P>(policy), result);
        return {};
    }

    /** Moves each live cell's %Gaussian by \a predictCell, or under SharedCovariance, the most
     * probable cell's by \a predictCell for the shared covariance and each mean by \a propagate:
     * the body of every form of predict(), the cells a for-each under \a policy.
     */
    template <typename P, typename PredictCell, typename Propagate>
    void
    moveCells(P && policy, PredictCell const & predictCell, Propagate const & propagate)
    {
        m_state.m_stamp.touchValues();
        auto & conditionals = m_state.m_conditionals;
        static_assert(!Switching);
        if constexpr (Shared)
        {
            ConditionalType const predicted = predictCell(conditionals.conditional(m_state.mostProbable()));
            auto means = conditionals.means();
            internal::forEachIndex(std::forward<P>(policy), means.cols(), [&](Eigen::Index i) {
                means.col(i) = propagate(means.col(i));
            });
            normalizeValues(m_state.grid().restSpace(), means);
            conditionals.sharedCovariance() = predicted.covarianceLower();
        }
        else
        {
            internal::forEachIndex(std::forward<P>(policy), conditionals.size(), [&](Eigen::Index i) {
                auto & cell = conditionals.conditional(i);
                cell = predictCell(cell);
            });
        }
    }

    /** Reserves the scratch for the current number of live cells. */
    void
    reserve()
    {
        m_posterior.reserve(m_state.size());
    }

    /** Records the cells updated in \a result, reweights them by the likelihoods just computed
     * into it, under \a policy, swaps the posterior Gaussians in where \a Posterior says the update
     * wrote them there, and records the total in \a result. An update that wrote the cells where
     * they are, or changed none, passes false.
     */
    template <bool Posterior = true, typename P, typename ResultAllocator>
    void
    finish(P && policy, RbPointMassFilterUpdate<Scalar, ResultAllocator> & result)
    {
        auto const cells = m_state.cells();
        internal::PointLikelihoodsAccess::cells(result).assign(cells.begin(), cells.end());
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

    /** The cells' steps: the model and the method, or under a switching model the IMM recursion,
     * which holds them and its scratch.
     */
    using StepsType = std::conditional_t<
        Switching,
        ImmRecursion<Gaussian<typename Grid_::RestSpace, Scalar>, Model_, Method_, allocator_type>,
        internal::PointSteps<Model_, Method_>>;

    StepsType m_steps;  ///< the model and the method, and the IMM's scratch
    StateType m_state;  ///< the live cells and their distributions

    // Scratch, reused across steps.
    typename StateType::ConditionalsType m_posterior;  ///< the updated distributions
};

namespace internal
{

/** Whether \a U, given to a point-mass filter's constructor where a covariance form or a prediction
 * method may stand, is the filter's upkeep instead.
 */
template <typename U>
concept GridUpkeepArgument =
    UpkeepArgument<U> && !std::same_as<U, SharedCovariance> && !std::same_as<U, PerPointCovariance>;

}  // namespace internal

/// \cond DEDUCTION_GUIDES
// The method argument's type depends on the model, so it is deduced here; the parameters are in
// the class's order so MSVC also deduces through the PointMassFilter alias. An upkeep, given last,
// is told from a covariance form or a method by its type.
template <
    typename Grid_,
    typename Model_,
    typename Covariance_ = internal::PointCovarianceFor<Model_>,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Conditional>
    requires(!internal::GridUpkeepArgument<Covariance_> && !internal::UpkeepArgument<Method_>)
RbPointMassFilter(Model_, Grid_, Conditional, Covariance_ = {}, Method_ = {}) -> RbPointMassFilter<
    Grid_,
    internal::BoundStep<Model_, typename Grid_::RestSpace, typename Grid_::Scalar>,
    Covariance_,
    Method_>;

template <
    typename Grid_,
    typename Model_,
    typename Covariance_ = internal::PointCovarianceFor<Model_>,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Upkeep_,
    typename Conditional>
    requires internal::GridUpkeepArgument<Upkeep_>
RbPointMassFilter(Model_, Grid_, Conditional, Upkeep_) -> RbPointMassFilter<
    Grid_,
    internal::BoundStep<Model_, typename Grid_::RestSpace, typename Grid_::Scalar>,
    Covariance_,
    Method_,
    Upkeep_>;

template <
    typename Grid_,
    typename Model_,
    typename Covariance_,
    typename Method_ = DefaultMethodFor<Model_>,
    typename Upkeep_,
    typename Conditional>
    requires(!internal::GridUpkeepArgument<Covariance_> && internal::GridUpkeepArgument<Upkeep_>)
RbPointMassFilter(Model_, Grid_, Conditional, Covariance_, Upkeep_) -> RbPointMassFilter<
    Grid_,
    internal::BoundStep<Model_, typename Grid_::RestSpace, typename Grid_::Scalar>,
    Covariance_,
    Method_,
    Upkeep_>;

template <
    typename Grid_,
    typename Model_,
    typename Covariance_,
    typename Method_,
    typename Upkeep_,
    typename Conditional>
RbPointMassFilter(Model_, Grid_, Conditional, Covariance_, Method_, Upkeep_) -> RbPointMassFilter<
    Grid_,
    internal::BoundStep<Model_, typename Grid_::RestSpace, typename Grid_::Scalar>,
    Covariance_,
    Method_,
    Upkeep_>;

template <
    typename Grid_,
    typename Model_ = NoProcessModel,
    typename Covariance_ = SharedCovariance,
    typename Method_ = methods::Linearized,
    typename Upkeep_ = upkeep::None>
    requires(!internal::GridUpkeepArgument<Covariance_>)
RbPointMassFilter(Grid_, Covariance_ = {}) -> RbPointMassFilter<Grid_, Model_, Covariance_, Method_, Upkeep_>;

template <
    typename Grid_,
    typename Model_ = NoProcessModel,
    typename Covariance_ = SharedCovariance,
    typename Method_ = methods::Linearized,
    typename Upkeep_>
    requires internal::GridUpkeepArgument<Upkeep_>
RbPointMassFilter(Grid_, Upkeep_) -> RbPointMassFilter<Grid_, Model_, Covariance_, Method_, Upkeep_>;

template <
    typename Grid_,
    typename Model_ = NoProcessModel,
    typename Covariance_,
    typename Method_ = methods::Linearized,
    typename Upkeep_>
    requires(!internal::GridUpkeepArgument<Covariance_> && internal::GridUpkeepArgument<Upkeep_>)
RbPointMassFilter(Grid_, Covariance_, Upkeep_) -> RbPointMassFilter<Grid_, Model_, Covariance_, Method_, Upkeep_>;
/// \endcond

/** \ingroup Filters
 * \brief A Bayesian grid filter: an RbPointMassFilter whose points cover the whole
 * state, so nothing is left %Gaussian.
 *
 * The state is \f$ p(x) = \sum_c P(c) \, \delta(x - p_c) \f$ over the grid's cells, and an update
 * multiplies each cell's probability by \f$ p(z \mid c) \f$. As in the Rao-Blackwellized filter,
 * no probability passes between cells, so the filter suits a state that doesn't move, or moves
 * little against the cell spacing, such as a receiver's position or a fixed bias. The grid's path
 * names a block that is the whole state: <tt>regularGrid<Composite<Position>, Position>(...)</tt>.
 * Cells may have covariances about their centers as in PointGrid. Mixture noise and gates need
 * PerPointCovariance, under which each cell's innovation is evaluated on its own. A measurement
 * known only by its likelihood (a model with logLikelihood(g) and no function), whatever the shape
 * of its error, weights the cells exactly under either form (decisions.md [likelihood-measurements]).
 *
 * The point-mass filter of \ref Bucy1971 "Bucy and Senne (1971)" for a state that does not
 * move, as \ref Bergman1999 "Bergman (1999)", chapter 5, uses it for a position on a terrain
 * map.
 *
 * \code
 * auto filter = PointMassFilter{regularGrid<Composite<Position>, Position>(first, spacing, {200, 200})};
 * filter.update(range);                       // reads Position
 * filter.state().marginal<GridAxis<Position, 0>>();
 * \endcode
 *
 * Its upkeep, given last, is run by maintain():
 *
 * \code
 * auto refining = PointMassFilter{grid, upkeep::sequence(upkeep::pruneBelow(1e-6),
 *                                                        upkeep::whenSize(std::less{}, 100, upkeep::subdivide(2)))};
 * (void)refining.update(range);
 * (void)refining.maintain();   // pruned, then refined if fewer than 100 cells are left
 * \endcode
 */
template <typename Grid_, typename Covariance_ = SharedCovariance, typename Upkeep_ = upkeep::None>
using PointMassFilter = RbPointMassFilter<Grid_, NoProcessModel, Covariance_, methods::Linearized, Upkeep_>;

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Grid_, typename Model_, typename Covariance_, typename Method_, typename Upkeep_>
void
describe(
    std::ostream & stream,
    RbPointMassFilter<Grid_, Model_, Covariance_, Method_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "Rao-Blackwellized point-mass filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

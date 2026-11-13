#pragma once

/** \file
 * Upkeep (decisions.md [upkeep], [rao-blackwellized-core]): the actions Resample, Roughen, PruneBelow,
 * PruneTail, PruneToSize, Subdivide, FloorModes, and RepairCovariance, the triggers When, WhenEss, and
 * WhenSize that run them, the Sequence that runs them all, and None, which a caller applies to a filter's
 * state between steps or gives the filter to run by maintain(). Read Divergence.h next.
 */

#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Distributions/Roots.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Core/Weights.h>

namespace Emrys
{

/** \defgroup Upkeep Upkeep
 * \ingroup Distributions
 * \ingroup Filters
 *
 * Upkeep keeps a filter's representation in bounds without meaning to change the
 * distribution much: resampling particles whose weights have degenerated, roughening them after,
 * pruning the improbable particles, components, or cells, or repairing a covariance that rounding has
 * left not positive definite. The filters never do it on their own; the caller applies it to the
 * filter's state() where it fits the problem, such as once after the last update before a prediction.
 * Upkeep is not a reduction policy, which acts inside an update on the branches of one measurement
 * (decisions.md [reduction-policies]); upkeep acts between steps on the whole state.
 *
 * Every action leaves a state of the type it found, since the filter fixed that type for its cost (a
 * shared covariance, a regular grid) and state() hands out a reference to it (decisions.md
 * [rao-blackwellized-core]). Within the type it may change the size: the particles, components, or cells
 * a prune leaves, the particles a Resample draws, a grid's cells after Subdivide. What changes the type,
 * such as drawSamples() from a mixture, fitMixture() from particles, or collapsing a mixture into one
 * %Gaussian, is a conversion the caller makes outside the filter.
 *
 * Each operation applies to the states with the members it calls, and UpkeepOperation answers whether
 * it does (\c Prune is all three prunes):
 *
 * | State, and its filter                                 | Resample | Roughen | Prune | Subdivide | Floor | Repair |
 * |-------------------------------------------------------|:--------:|:-------:|:-----:|:---------:|:-----:|:------:|
 * | GaussianParams: KalmanFilter                          |          |         |       |           |       | yes    |
 * | ParticleSet: ParticleFilter                           | yes      | yes     | yes   |           |       |        |
 * | ParticleGaussianParams: RbParticleFilter              | yes      | yes     | yes   |           |       |        |
 * | MixtureParams of Gaussians: MixtureFilter             |          |         | yes   |           |       | yes    |
 * | MixtureParams of hybrids: MixtureFilter               |          |         | yes   |           | yes   | yes    |
 * | HybridGaussianParams: ImmFilter, AmmFilter            |          |         |       |           | yes   | yes    |
 * | GridGaussianParams, regular: the point-mass filters   |          |         | yes   | yes       |       |        |
 * | GridGaussianParams of given points                    |          |         | yes   |           |       |        |
 * | CategoricalParams: HiddenMarkovFilter                 |          |         |       |           | yes   |        |
 *
 * WhenSize needs only \c size(), and WhenEss \c ess(), which the particle states have. A grid isn't
 * resampled, since two copies of a cell would evolve identically. Of the %Gaussian forms only the
 * covariance form repairs: the square-root form refuses a repair at compile time, and the information
 * forms have none.
 *
 * No action merges or splits mixture components (decisions.md [upkeep]). How best to search
 * for the merges to make depends on the use and its hardware, a software engineering choice and
 * not a mathematical one, so the caller writes that action from MixtureParams::merged(), merge(),
 * and the divergences (see \ref Divergences); examples/gaussian_mixture.cpp merges by Runnalls'
 * cost. The literature has no rule for splitting that the library takes up
 * (decisions.md [splitting-hooks]).
 *
 * Each upkeep object is a function object called with the state alone. It returns whether it
 * changed the state, or a NumericError: a WeightError if the weights are invalid, or a GaussianError
 * if a covariance can't be repaired. Actions do the work unconditionally, and each the library names
 * leaves the state unchanged when it returns an error. Triggers decide whether to run the actions
 * they hold, in order, on the same state; a trigger stops at the first error and returns it, and the
 * actions that ran before it keep their effect (decisions.md [upkeep]), so a caller who needs all or
 * nothing copies the state before the call. An action returns true when it did its work and false when
 * the work would have been nothing: a prune that removed nothing, Subdivide by one, Roughen by zero,
 * FloorModes with no mode below the floor, RepairCovariance on covariances that factorize. Resample
 * returns true even on equal weights, since it redraws the particles.
 * What else an action needs is given to its factory: Resample and Roughen hold a reference to the
 * random generator they draw from, which must outlive them, and a caller's action captures what it
 * needs.
 *
 * An execution policy, when wanted, is given to the factory too, first, as it comes before the range
 * in a standard algorithm (see \ref Execution): Resample and Roughen run their loops under it, and
 * <tt>whenEss(policy, ...)</tt> takes the effective sample size under it. The other actions run
 * sequentially and take none.
 *
 * Upkeep runs on the filter's state, which allocates through its allocator (decisions.md
 * [allocation]). Under execution::sequential, Resample to a size within the set's capacity, Roughen on
 * a ParticleSet, the prunes once the state's capacity is set, and FloorModes allocate nothing. Subdivide
 * allocates the larger grid, Roughen on a Rao-Blackwellized state a copy of its particle set, and
 * RepairCovariance a copy of each covariance it repairs. Under another policy the standard algorithms
 * allocate, as [execution-policies] allows.
 *
 * \code
 * auto const resample = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng));  // particles
 * resample(particleFilter.state());
 * auto const inParallel = upkeep::whenEss(std::execution::par, std::less{}, 0.5,
 *                                         upkeep::resample(std::execution::par, rng),
 *                                         upkeep::roughen(std::execution::par, rng));
 * inParallel(particleFilter.state());
 * auto const thin = upkeep::whenSize(std::greater{}, 4096, upkeep::pruneTail(1e-6));  // grid cells
 * thin(pointMassFilter.state());
 * auto const prune = upkeep::pruneBelow(1e-6);  // mixture components
 * prune(mixtureFilter.state());
 * auto const bound = upkeep::whenSize(std::greater{}, 64, upkeep::pruneToSize(32));  // or particles
 * bound(mixtureFilter.state());
 * auto const repair = upkeep::repairCovariance();  // a Gaussian, as a Kalman filter's
 * repair(kalmanFilter.state());
 * \endcode
 *
 * Actions: Resample, Roughen, PruneBelow, PruneTail, PruneToSize, Subdivide, FloorModes, RepairCovariance.
 * Triggers: When, which
 * runs its actions when a predicate of the state holds, and WhenEss and WhenSize, the two Whens the library
 * names, each testing what it is named for with a comparator given first, a standard one such as \c std::less
 * or the caller's own. A trigger runs a caller's action too: any function object of the state that returns
 * <tt>std::expected<bool, WeightError></tt>. A Sequence runs its actions every time, so that a filter's
 * whole upkeep is one value.
 *
 * Every filter holds one upkeep, given last to its constructor and None unless given, and runs it on its
 * state when the caller calls \c maintain():
 *
 * \code
 * auto filter = ParticleFilter{model, initial, upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng))};
 * (void)filter.update(fix);
 * (void)filter.maintain();   // the same as filter.upkeep()(filter.state())
 * \endcode
 *
 * Resampling when the effective sample size falls is the sequential importance resampling of
 * \ref Arulampalam2002 "Arulampalam et al. (2002)", §V-A, Algorithm 3; roughening after it is the
 * regularized particle filter of \ref Musso2001 "Musso, Oudjane, and Le Gland (2001)", as given in
 * \ref Arulampalam2002 "Arulampalam et al. (2002)", §V-B3, Algorithm 6, with a %Gaussian kernel;
 * and the floor on mode probabilities is \ref Athans1977 "Athans et al. (1977)".
 */

namespace internal
{

/** Runs \a actions in order on \a state, stopping at the first error.
 * \returns whether any changed the state
 */
template <typename Actions, typename State>
[[nodiscard]] std::expected<bool, WeightError>
runActions(Actions const & actions, State & state)
{
    return std::apply(
        [&](auto const &... action) {
            auto changed = std::expected<bool, WeightError>{false};
            auto const run = [&](auto const & next) {
                if (!changed)
                {
                    return;
                }
                std::expected<bool, WeightError> const result = next(state);
                if (!result)
                {
                    changed = std::unexpected{result.error()};
                }
                else
                {
                    changed = *changed || *result;
                }
            };
            (run(action), ...);
            return changed;
        },
        actions
    );
}

/** \returns whether a prune that returned \a removed, the log of the weight it removed or a
 * std::expected of it, removed anything, or its WeightError
 */
template <typename Removed>
[[nodiscard]] std::expected<bool, WeightError>
prunedAny(Removed const & removed)
{
    if constexpr (isExpected<Removed>)
    {
        using Scalar = typename Removed::value_type;
        return removed.transform([](Scalar logRemoved) {
            return logRemoved > -Eigen::NumTraits<Scalar>::infinity();
        });
    }
    else
    {
        return removed > -Eigen::NumTraits<Removed>::infinity();
    }
}

/** \returns the outcome of a factorization, \a result, with its error as a NumericError: a
 * MixtureError's WeightError, or the NumericError of its ComponentError without the component's index
 */
template <typename Result>
[[nodiscard]] std::expected<void, NumericError>
factorizeOutcome(Result const & result)
{
    if (result)
    {
        return {};
    }
    if constexpr (std::same_as<typename Result::error_type, NumericError>)
    {
        return std::unexpected{result.error()};
    }
    else
    {
        return std::unexpected{std::visit(
            [](auto const & error) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(error)>, NumericError>)
                {
                    return error;
                }
                else
                {
                    return NumericError(error.error);
                }
            },
            result.error()
        )};
    }
}

/** Whether \a Test, called with \a Args, is the predicate of an upkeep::When: it returns something
 * that tests as a \c bool, or <tt>std::expected<bool, WeightError></tt> for a test that can fail.
 */
template <typename Test, typename... Args>
concept UpkeepTest = std::predicate<Test const &, Args...>
                  || (std::invocable<Test const &, Args...>
                      && std::same_as<std::invoke_result_t<Test const &, Args...>, std::expected<bool, WeightError>>);

/** \returns the outcome of an upkeep test, \a result, as an <tt>std::expected<bool, WeightError></tt> */
template <typename Result>
[[nodiscard]] std::expected<bool, WeightError>
testOutcome(Result const & result)
{
    if constexpr (std::same_as<Result, std::expected<bool, WeightError>>)
    {
        return result;
    }
    else
    {
        return bool(result);
    }
}

/** The predicate of upkeep::whenEss(): whether \c compare holds between a set's effective sample
 * size, taken under \c policy if the state can, and \c fraction of its size, in the state's scalar,
 * or the WeightError of its \c ess().
 */
template <typename Compare, RealScalar Scalar_, ExecutionPolicy Policy_ = execution::Sequential>
    requires std::predicate<Compare const &, Scalar_, Scalar_>
struct EssTest
{
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Compare compare;   ///< called as <tt>compare(ess, fraction * size)</tt>
    Scalar_ fraction;                                 ///< the threshold, as a fraction of the size
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Policy_ policy{};  ///< the policy the effective sample size is taken under

    /** \returns whether the test holds for \a state, or the WeightError of its \c ess() */
    template <typename State>
        requires requires(State const & state) {
            state.ess();
            state.size();
        }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State const & state) const
    {
        if constexpr (!internal::isSequential<Policy_> && requires { state.ess(policy); })
        {
            return holds(state.ess(policy), state.size());
        }
        else
        {
            return holds(state.ess(), state.size());
        }
    }

private:

    /** \returns whether the test holds for the effective sample size \a ess of \a size elements, in
     * the scalar of \a ess, or the error of \a ess
     */
    template <typename Ess, typename Index>
    [[nodiscard]] std::expected<bool, WeightError>
    holds(Ess const & ess, Index size) const
    {
        if (!ess)
        {
            return std::unexpected{ess.error()};
        }
        using Scalar = typename Ess::value_type;
        return bool(compare(*ess, Scalar(Scalar(fraction) * scalarFromCount<Scalar>(size))));
    }
};

/** The predicate of upkeep::whenSize(): whether \c compare holds between a state's number of
 * elements and \c limit.
 */
template <typename Compare>
    requires std::predicate<Compare const &, std::ptrdiff_t, std::ptrdiff_t>
struct SizeTest
{
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Compare compare;  ///< called as <tt>compare(size, limit)</tt>
    std::ptrdiff_t limit;                            ///< the number of elements the size is compared with

    /** \returns whether the test holds for \a state */
    template <typename State>
        requires requires(State const & state) { state.size(); }
    [[nodiscard]] bool
    operator()(State const & state) const
    {
        return bool(compare(std::ptrdiff_t(state.size()), limit));
    }
};

/** Whether an upkeep::Resample drawing from \a G with the resampler \a R under the policy \a P applies
 * to \a State: it has \c size() and <tt>resample(rng, newSize, resampler)</tt>, or under a policy other
 * than execution::sequential <tt>resample(policy, rng, newSize, resampler)</tt>.
 */
template <typename State, typename G, typename R, typename P>
concept ResamplableBy = requires(State & state, G & rng, Eigen::Index newSize, R const & resampler, P const & policy) {
    state.size();
    requires(!isSequential<P>) || requires { state.resample(rng, newSize, resampler); };
    requires isSequential<P> || requires { state.resample(policy, rng, newSize, resampler); };
};

/** The predicate of upkeep::sequence(): it holds for every state. */
struct Always
{
    /** \returns true */
    template <typename State>
    [[nodiscard]] constexpr bool
    operator()(State const &) const
    {
        return true;
    }
};

}  // namespace internal

/** \ingroup Upkeep
 * \brief An upkeep operation on a \a State: an action or a trigger, the library's or the caller's
 * own, a function object called as <tt>operation(state)</tt> that returns whether it changed the
 * state, or the NumericError that stopped it, a WeightError or a GaussianError (decisions.md
 * [upkeep], [user-extension-points]). A trigger runs any of them. What else an operation needs, such
 * as a generator, it holds.
 *
 * Each of the library's operations is constrained on the members of the state it calls, and a
 * trigger on its predicate and on each of its actions being an UpkeepOperation of the state, so the
 * concept answers whether an operation applies to a state, and a call on a state it doesn't apply to
 * fails at the call.
 */
template <typename T, typename State>
concept UpkeepOperation = requires(T const & operation, State & state) {
    { operation(state) } -> std::same_as<std::expected<bool, NumericError>>;
};

/** \ingroup Upkeep
 * \brief The upkeep operations, each made by its factory (decisions.md [policy-factories],
 * [upkeep]): the actions resample(), roughen(), pruneBelow(), pruneTail(), pruneToSize(), subdivide(),
 * floorModes(), and repairCovariance(), the
 * triggers when(), whenEss(), and whenSize(), sequence(), and none(). The types beside them name an
 * operation where a type is needed, as a class member or a filter's template argument.
 */
namespace upkeep
{

/** \ingroup Upkeep
 * \brief Action that resamples a set in proportion to its weights, drawing from the generator \a G
 * with the resampler \a Resampler_ (systematic by default) under the execution policy \a Policy_
 * (sequential by default), to a given size if one is given and to the set's size otherwise.
 *
 * The state is anything with <tt>resample(rng, newSize, resampler)</tt> and \c size(), or under a
 * policy other than execution::sequential <tt>resample(policy, rng, newSize, resampler)</tt>, such as
 * a ParticleSet or a ParticleGaussianParams; its weights become equal. A size lets upkeep change the
 * particle count. A caller computing the count from the state, as KLD sampling does, makes the action
 * at each step, since an action is a small value. The action holds a reference to the generator,
 * which must outlive it, and draws from it at each call; a generator made in the call to resample()
 * does not compile.
 *
 * \sa WhenEss, \ref Resamplers
 */
template <
    std::uniform_random_bit_generator G,
    typename Resampler_ = resamplers::Systematic,
    ExecutionPolicy Policy_ = execution::Sequential>
class Resample
{
public:

    /** Draws from \a generator with \a resampler under \a policy, to \a size particles if one is
     * given and to the state's size otherwise.
     */
    constexpr explicit Resample(
        G & generator,
        Resampler_ resampler = {},
        std::optional<Eigen::Index> size = std::nullopt,
        Policy_ policy = {}
    )
        : generator_{&generator}
        , resampler_{std::move(resampler)}
        , size_{size}
        , policy_{std::move(policy)}
    {}

    /** Resamples \a state. \returns true, or the WeightError of the state's resample() */
    template <typename State>
        requires internal::ResamplableBy<State, G, Resampler_, Policy_>
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        auto const resampled = [&] {
            if constexpr (internal::isSequential<Policy_>)
            {
                return state.resample(*generator_, newSize(state), resampler_);
            }
            else
            {
                return state.resample(policy_, *generator_, newSize(state), resampler_);
            }
        }();
        if (!resampled)
        {
            return std::unexpected{resampled.error()};
        }
        return true;
    }

    /** \returns the generator drawn from */
    [[nodiscard]] constexpr G &
    generator() const noexcept
    {
        return *generator_;
    }

    /** \returns the resampler (see \ref Resamplers) */
    [[nodiscard]] constexpr Resampler_ const &
    resampler() const noexcept
    {
        return resampler_;
    }

    /** \returns the size resampled to, or none to keep the state's */
    [[nodiscard]] constexpr std::optional<Eigen::Index>
    size() const noexcept
    {
        return size_;
    }

    /** \returns the execution policy */
    [[nodiscard]] constexpr Policy_ const &
    policy() const noexcept
    {
        return policy_;
    }

private:

    /** \returns the size given, else the size of \a state */
    template <typename State>
    [[nodiscard]] Eigen::Index
    newSize(State const & state) const
    {
        return size_ ? *size_ : Eigen::Index(state.size());
    }

    G * generator_;                                        ///< the generator the draws come from
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Resampler_ resampler_;  ///< the resampler
    std::optional<Eigen::Index> size_;                     ///< the size resampled to, or none
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Policy_ policy_;        ///< the execution policy
};

/** \ingroup Upkeep
 * \returns a Resample action drawing from \a generator with \a resampler, systematic by default (see
 * \ref Resamplers), that keeps the state's size
 */
template <std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
    requires(!std::integral<R>)
[[nodiscard]] constexpr Resample<G, R>
resample(G & generator, R resampler = {})
{
    return Resample<G, R>{generator, std::move(resampler)};
}

/** \ingroup Upkeep
 * \returns a Resample action drawing from \a generator with \a resampler, systematic by default, that
 * resamples to \a size particles, which must be positive
 */
template <std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
    requires(!std::integral<R>)
[[nodiscard]] constexpr Resample<G, R>
resample(G & generator, Eigen::Index size, R resampler = {})
{
    EMRYS_FILTER_ASSERT(size > 0 && "a resample needs a positive size");
    return Resample<G, R>{generator, std::move(resampler), size};
}

/** \ingroup Upkeep
 * \returns a Resample action as resample(generator, resampler) makes, under \a policy (see
 * \ref Execution)
 */
template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
    requires(!std::integral<R>)
[[nodiscard]] constexpr Resample<G, R, std::remove_cvref_t<P>>
resample(P && policy, G & generator, R resampler = {})
{
    return Resample<G, R, std::remove_cvref_t<P>>{
        generator,
        std::move(resampler),
        std::nullopt,
        std::forward<P>(policy)
    };
}

/** \ingroup Upkeep
 * \returns a Resample action as resample(generator, size, resampler) makes, under \a policy (see
 * \ref Execution)
 */
template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename R = resamplers::Systematic>
    requires(!std::integral<R>)
[[nodiscard]] constexpr Resample<G, R, std::remove_cvref_t<P>>
resample(P && policy, G & generator, Eigen::Index size, R resampler = {})
{
    EMRYS_FILTER_ASSERT(size > 0 && "a resample needs a positive size");
    return Resample<G, R, std::remove_cvref_t<P>>{generator, std::move(resampler), size, std::forward<P>(policy)};
}

/** \ingroup Upkeep
 * \brief Action that moves each particle by an independent draw from a %Gaussian kernel, to
 * restore the diversity resampling removes: the regularized particle filter.
 *
 * The kernel's covariance is \c scale times the set's kdeBandwidth(), Silverman's rule unless a
 * bandwidth is stored, so after an equal-weight resample the particles become a draw from their
 * kernel density estimate. resample() clears a stored bandwidth, so one stored before it is not
 * used after it. A smaller \c scale adds less noise than the estimate would. The draws are added
 * in the tangent space and normalized. A stored bandwidth is kept, as it is when the states
 * change other than by resampling. \a Scalar_ is the type of \c scale, deduced from the one
 * given (<tt>upkeep::roughen(rng, 0.25f)</tt> holds a \c float), so that a \c float filter holds no \c double.
 * A \c scale of zero turns roughening off: <tt>upkeep::roughen(rng, 0)</tt> checks the weights and
 * returns false, drawing nothing from the generator and leaving the states as they were, so that a run
 * with it is the run without it.
 *
 * The action draws from the generator \a G, which it holds by reference and which must outlive it,
 * under the execution policy \a Policy_ (sequential by default): the bandwidth is a reduction over the
 * particles, and the moves are a for-each over them (see \ref Execution). Sequential draws each
 * particle's normals as it moves it, so it allocates nothing; another policy draws them all first, in
 * the same order, into scratch from the policy [execution-policies].
 *
 * The weights need not be equal: Silverman's rule weights by the effective sample size, so roughening
 * before a resample takes the weighted spread.
 *
 * \sa Resample, ParticleSet::kdeBandwidth()
 */
template <
    std::uniform_random_bit_generator G,
    RealScalar Scalar_ = double,
    ExecutionPolicy Policy_ = execution::Sequential>
class Roughen
{
public:

    /** Draws from \a generator a kernel of covariance \a scale times the bandwidth, under \a policy. */
    constexpr explicit Roughen(G & generator, Scalar_ scale = 1, Policy_ policy = {})
        : generator_{&generator}
        , scale_{scale}
        , policy_{std::move(policy)}
    {}

    /** Moves each of \a particles by a draw from the kernel. \returns whether it moved them, false
     * only for a zero scale, or a WeightError if the weights are invalid
     */
    template <typename Space, typename Scalar, typename Allocator>
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(ParticleSet<Space, Scalar, Allocator> & particles) const
    {
        using Set = ParticleSet<Space, Scalar, Allocator>;
        using Vector = typename Set::VectorType;
        using Difference = typename Set::DifferenceVectorType;
        static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
        EMRYS_FILTER_ASSERT(scale_ >= Scalar_(0) && "the kernel scale must not be negative");
        G & rng = *generator_;
        if (scale_ == Scalar_(0))
        {
            return particles.logTotalWeight(policy_).transform([](Scalar) {
                return false;
            });
        }
        auto const bandwidth = particles.kdeBandwidth(policy_);
        if (!bandwidth)
        {
            return std::unexpected{bandwidth.error()};
        }
        auto const kernel = internal::defaultRoot((Scalar(scale_) * *bandwidth).eval());
        auto const & space = particles.space();
        Eigen::Index const rows = particles.differenceDimension();
        if constexpr (internal::isSequential<Policy_>)
        {
            // One distribution for every draw, which may keep a value between them, so that the draws
            // are those of filling all the particles' normals at once, as another policy does.
            auto normal = std::normal_distribution<Scalar>{};
            auto z = Difference{};
            z.resize(rows);
            for (Eigen::Index i = 0; i < particles.size(); ++i)
            {
                internal::fillStandardNormal(z, rng, normal);
                Vector const moved = Emrys::plus(space, particles.state(i), Difference{internal::multiply(kernel, z)});
                particles.setState(i, moved);
            }
        }
        else
        {
            using Normals = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
            auto draws = internal::scratchVector<Scalar>(policy_, std::size_t(rows * particles.size()));
            auto normals = Eigen::Map<Normals>{draws.data(), rows, particles.size()};
            internal::fillStandardNormal(normals, rng);
            particles.forEach(policy_, [&](auto & particle) {
                Vector const moved =
                    Emrys::plus(space, particle.state(), Difference{kernel * normals.col(particle.index())});
                particle.setState(moved);
            });
        }
        return true;
    }

    /** Roughens the particles of a Rao-Blackwellized state, such as a ParticleGaussianParams, by their
     * own bandwidth. Their Gaussians are left as they were, as if each particle had always been
     * where it is moved to: an approximation, as roughening always is. It roughens a copy of the
     * particle set and writes the states back, so it allocates the copy at each call.
     */
    template <typename State>
        requires requires(State & state) { state.setParticleStates(state.particles().states()); }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        auto particles = state.particles();
        std::expected<bool, WeightError> const roughened = (*this)(particles);
        if (roughened && *roughened)
        {
            state.setParticleStates(particles.states());
        }
        return roughened;
    }

    /** \returns the generator drawn from */
    [[nodiscard]] constexpr G &
    generator() const noexcept
    {
        return *generator_;
    }

    /** \returns the kernel covariance as a multiple of the bandwidth */
    [[nodiscard]] constexpr Scalar_
    scale() const noexcept
    {
        return scale_;
    }

    /** \returns the execution policy */
    [[nodiscard]] constexpr Policy_ const &
    policy() const noexcept
    {
        return policy_;
    }

private:

    G * generator_;                                  ///< the generator the jitter comes from
    Scalar_ scale_;                                  ///< the kernel covariance as a multiple of the bandwidth
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Policy_ policy_;  ///< the execution policy
};

/** \ingroup Upkeep
 * \returns a Roughen action drawing from \a generator a kernel of covariance \a scale times the
 * bandwidth, 1 by default, held in the scalar it is written in (<tt>roughen(rng, 0.25f)</tt> holds a
 * \c float; an integer stands for \c double)
 */
template <std::uniform_random_bit_generator G, typename Value = double>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] constexpr Roughen<G, internal::DeducedScalar<Value>>
roughen(G & generator, Value scale = 1)
{
    return Roughen<G, internal::DeducedScalar<Value>>{generator, internal::DeducedScalar<Value>(scale)};
}

/** \ingroup Upkeep
 * \returns a Roughen action as roughen(generator, scale) makes, under \a policy (see \ref Execution)
 */
template <ExecutionPolicy P, std::uniform_random_bit_generator G, typename Value = double>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] constexpr Roughen<G, internal::DeducedScalar<Value>, std::remove_cvref_t<P>>
roughen(P && policy, G & generator, Value scale = 1)
{
    return Roughen<G, internal::DeducedScalar<Value>, std::remove_cvref_t<P>>{
        generator,
        internal::DeducedScalar<Value>(scale),
        std::forward<P>(policy)
    };
}

/** \ingroup Upkeep
 * \brief Trigger that runs its actions when its predicate holds for the state: the general trigger,
 * of which WhenEss and WhenSize are two.
 *
 * The predicate is called with the state, as a \c const reference, and returns a \c bool, or an
 * <tt>std::expected<bool, WeightError></tt> for a test that can fail, whose error the trigger
 * returns without running anything. The actions are called with the state alone, in order, and stop
 * at the first error, which the trigger returns; actions that ran before it keep their effect.
 *
 * \code
 * auto const refine = upkeep::when([](auto const & grid) { return grid.size() < 64; }, upkeep::subdivide(2));
 * refine(filter.state());
 * \endcode
 */
template <typename Predicate, typename... Actions>
struct When
{
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Predicate predicate;  ///< the test of the state
    std::tuple<Actions...> actions;                      ///< run in order when the test holds

    /** Runs \a toRun, in order, when \a test holds for the state. */
    When(Predicate test, Actions... toRun)
        : predicate{std::move(test)}
        , actions{std::move(toRun)...}
    {}

    /** Tests \a state and runs the actions on it if the test holds.
     * \returns whether any action changed the state, or a WeightError
     */
    template <typename State>
        requires internal::UpkeepTest<Predicate, State const &> && (UpkeepOperation<Actions, State> && ...)
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        std::expected<bool, WeightError> const holds = internal::testOutcome(predicate(std::as_const(state)));
        if (!holds || !*holds)
        {
            return holds;
        }
        return internal::runActions(actions, state);
    }
};

/** \ingroup Upkeep
 * \returns a When trigger running \a actions, in order, when \a predicate holds for the state
 */
template <typename Predicate, typename... Actions>
[[nodiscard]] When<Predicate, Actions...>
when(Predicate predicate, Actions... actions)
{
    return When<Predicate, Actions...>{std::move(predicate), std::move(actions)...};
}

/** \ingroup Upkeep
 * \brief Trigger that runs its actions when \c compare holds between a set's effective sample size
 * and \c fraction of its size: a When whose predicate is an internal::EssTest.
 *
 * The comparator comes first and is required, so the direction of the test is written at the call:
 * <tt>upkeep::whenEss(std::less{}, 0.5, ...)</tt> runs the actions when the effective sample size is
 * below half the size. Any predicate of two scalars serves as the comparator, the caller's own
 * included; it is called as <tt>compare(ess, threshold)</tt>, the effective sample size first and
 * \c fraction times the size second, both in the state's scalar.
 *
 * The state is anything with \c size() and \c ess() returning an \c std::expected, such as a
 * ParticleSet, and the trigger returns the WeightError of \c ess(). Usually its actions start with a
 * Resample. Made with an execution policy first, <tt>whenEss(policy, compare, fraction, ...)</tt>, it
 * takes the effective sample size under the policy if the state can; its actions take their own.
 *
 * \code
 * auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng, 0.25));
 * resampling(filter.state());
 * \endcode
 *
 * \a Scalar_ is the type of the fraction, deduced from the one given
 * (<tt>upkeep::whenEss(std::less{}, 0.5f, ...)</tt> holds a \c float); the test runs in the state's
 * scalar. The fraction isn't checked: above one the trigger runs at every call, and at zero or below
 * never. The alias names the trigger without a policy; with one it is a When of an internal::EssTest
 * that holds the policy.
 */
template <typename Compare, RealScalar Scalar_, typename... Actions>
using WhenEss = When<internal::EssTest<Compare, Scalar_>, Actions...>;

/** \ingroup Upkeep
 * \returns a WhenEss trigger running \a actions, in order, when \a compare holds between the
 * effective sample size and \a fraction of the size, holding the fraction in the scalar it is
 * written in (<tt>whenEss(std::less{}, 0.5f, ...)</tt> holds a \c float)
 */
template <typename Compare, typename Value, typename... Actions>
    requires(!ExecutionPolicy<Compare>) && std::is_arithmetic_v<Value>
         && std::predicate<Compare const &, internal::DeducedScalar<Value>, internal::DeducedScalar<Value>>
[[nodiscard]] WhenEss<Compare, internal::DeducedScalar<Value>, Actions...>
whenEss(Compare compare, Value fraction, Actions... actions)
{
    using Test = internal::EssTest<Compare, internal::DeducedScalar<Value>>;
    return when(Test{std::move(compare), internal::DeducedScalar<Value>(fraction)}, std::move(actions)...);
}

/** \ingroup Upkeep
 * \returns a trigger as whenEss(compare, fraction, actions...) makes, taking the effective sample
 * size under \a policy (see \ref Execution)
 */
template <ExecutionPolicy P, typename Compare, typename Value, typename... Actions>
    requires std::is_arithmetic_v<Value>
          && std::predicate<Compare const &, internal::DeducedScalar<Value>, internal::DeducedScalar<Value>>
[[nodiscard]] When<internal::EssTest<Compare, internal::DeducedScalar<Value>, std::remove_cvref_t<P>>, Actions...>
whenEss(P && policy, Compare compare, Value fraction, Actions... actions)
{
    using Test = internal::EssTest<Compare, internal::DeducedScalar<Value>, std::remove_cvref_t<P>>;
    return when(
        Test{std::move(compare), internal::DeducedScalar<Value>(fraction), std::forward<P>(policy)},
        std::move(actions)...
    );
}

/** \ingroup Upkeep
 * \brief Trigger that runs its actions when \c compare holds between a state's number of elements
 * and a limit, such as a mixture's components or a grid's live cells: a When whose predicate is an
 * internal::SizeTest.
 *
 * The comparator comes first and is required, so the direction of the test is written at the call.
 * <tt>upkeep::whenSize(std::greater{}, 32, reduce)</tt>, with \c reduce a caller's action that merges a
 * mixture down to 16 components, merges whenever there are more than 32, so the work is done in
 * batches rather than at every step (examples/gaussian_mixture.cpp).
 * <tt>upkeep::whenSize(std::less{}, 64, upkeep::subdivide(2))</tt> refines a grid whenever pruning has
 * left fewer than 64 cells; nothing there bounds how fine the grid becomes, and the caller stops
 * refining when the resolution is enough. Any predicate of two \c std::ptrdiff_t serves as the
 * comparator, the caller's own included; it is called as <tt>compare(size, limit)</tt>.
 */
template <typename Compare, typename... Actions>
using WhenSize = When<internal::SizeTest<Compare>, Actions...>;

/** \ingroup Upkeep
 * \returns a WhenSize trigger running \a actions, in order, when \a compare holds between the number
 * of elements and \a limit
 */
template <typename Compare, typename... Actions>
    requires std::predicate<Compare const &, std::ptrdiff_t, std::ptrdiff_t>
[[nodiscard]] WhenSize<Compare, Actions...>
whenSize(Compare compare, std::ptrdiff_t limit, Actions... actions)
{
    return when(internal::SizeTest<Compare>{std::move(compare), limit}, std::move(actions)...);
}

/** \ingroup Upkeep
 * \brief Runs its actions in order on the state every time: a When whose predicate always holds, so
 * that a filter's whole upkeep is one value (decisions.md [upkeep]).
 *
 * It stops at the first action that returns an error and returns it; the actions that ran before
 * it keep their effect. It returns whether any action changed the state.
 *
 * \code
 * auto filter = PointMassFilter{grid, upkeep::sequence(upkeep::pruneBelow(1e-6),
 *                                                      upkeep::whenSize(std::less{}, 100, upkeep::subdivide(2)))};
 * (void)filter.update(range);
 * (void)filter.maintain();   // pruned, then refined if fewer than 100 cells are left
 * \endcode
 */
template <typename... Actions>
using Sequence = When<internal::Always, Actions...>;

/** \ingroup Upkeep
 * \returns a Sequence running \a actions, in order, at every call
 */
template <typename... Actions>
[[nodiscard]] Sequence<Actions...>
sequence(Actions... actions)
{
    return when(internal::Always{}, std::move(actions)...);
}

/** \ingroup Upkeep
 * \brief The upkeep a filter holds when it is given none: it takes any state, changes nothing, and
 * returns \c false, so that \c maintain() can be called on every filter (decisions.md [upkeep]).
 */
struct None
{
    /** \returns false, having changed nothing */
    template <typename State>
    [[nodiscard]] constexpr std::expected<bool, NumericError>
    operator()(State &) const
    {
        return false;
    }
};

/** \ingroup Upkeep
 * \returns the upkeep None, which does nothing
 */
[[nodiscard]] constexpr None
none()
{
    return {};
}

/** \ingroup Upkeep
 * \brief Action that removes every point whose normalized probability is below \c probability,
 * renormalizing the rest; the most probable point is always kept.
 *
 * The state is anything with <tt>pruneBelow(logThreshold)</tt> returning the log of the weight
 * removed, or a <tt>std::expected</tt> of it with a WeightError: a ParticleSet or a
 * ParticleGaussianParams, a MixtureParams, the state of a MixtureFilter, which fails as
 * logTotalWeight() does, or a GridGaussianParams, the state of an RbPointMassFilter, whose prune
 * can't fail. The threshold is absolute, the same whatever the number of points; a caller who wants
 * it relative to something, such as the cell size of a grid that subdivide() refines, adjusts it.
 * It doesn't bound the probability removed, which on \f$ N \f$ points can reach \f$ (N - 1) \f$ times
 * the threshold: <tt>pruneBelow(1e-6)</tt> on a flat distribution over two million cells keeps only the
 * most probable. Under <tt>subdivide(k)</tt> in \f$ d \f$ dimensions each child holds \f$ 1/k^d \f$ of
 * its parent, so the same threshold cuts a density \f$ k^d \f$ times higher after each refinement; a
 * caller who wants the same cut divides the threshold by \f$ k^d \f$. PruneTail bounds the total.
 * \a Scalar_ is the type of \c probability, deduced from the one given
 * (<tt>upkeep::pruneBelow(1e-4f)</tt> holds a \c float).
 *
 * \sa PruneTail, PruneToSize
 */
template <RealScalar Scalar_ = double>
struct PruneBelow
{
    Scalar_ probability;  ///< points less probable than this are removed

    /** Prunes \a state. \returns whether any point was removed, or the WeightError of the state's weights */
    template <typename State>
        requires requires(State & state, typename State::Scalar logThreshold) { state.pruneBelow(logThreshold); }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        using Scalar = typename State::Scalar;
        EMRYS_FILTER_ASSERT(
            probability >= Scalar_(0) && probability < Scalar_(1) && "the threshold is a probability below one"
        );
        return internal::prunedAny(state.pruneBelow(Eigen::numext::log(Scalar(probability))));
    }
};

/** \ingroup Upkeep
 * \returns a PruneBelow action removing the points less probable than \a probability, held in the
 * scalar it is written in (<tt>pruneBelow(1e-4f)</tt> holds a \c float)
 */
template <typename Value>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] constexpr PruneBelow<internal::DeducedScalar<Value>>
pruneBelow(Value probability)
{
    return {internal::DeducedScalar<Value>(probability)};
}

/** \ingroup Upkeep
 * \brief Action that removes the least probable points, as many as can be while their combined
 * probability stays below \c fraction, renormalizing the rest.
 *
 * The state is anything with <tt>pruneTail(logFraction)</tt>, as for PruneBelow. Unlike a threshold
 * on each point, the fraction bounds the probability removed, and so the change in the
 * distribution in total variation, however many points there are, and it removes about the same light
 * tail before a subdivide() as after it. It can remove a whole mode lighter than the fraction, which
 * PruneBelow would keep if each of the mode's points were above its threshold. \a Scalar_ is the type of
 * \c fraction, deduced from the one given (<tt>upkeep::pruneTail(1e-3f)</tt> holds a \c float).
 *
 * \sa PruneBelow, PruneToSize
 */
template <RealScalar Scalar_ = double>
struct PruneTail
{
    Scalar_ fraction;  ///< the probability the points removed stay below

    /** Prunes \a state. \returns whether any point was removed, or the WeightError of the state's weights */
    template <typename State>
        requires requires(State & state, typename State::Scalar logFraction) { state.pruneTail(logFraction); }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        using Scalar = typename State::Scalar;
        EMRYS_FILTER_ASSERT(
            fraction >= Scalar_(0) && fraction < Scalar_(1) && "the fraction pruned is a probability below one"
        );
        return internal::prunedAny(state.pruneTail(Eigen::numext::log(Scalar(fraction))));
    }
};

/** \ingroup Upkeep
 * \returns a PruneTail action removing the least probable points while their combined probability
 * stays below \a fraction, held in the scalar it is written in (<tt>pruneTail(1e-3f)</tt> holds a
 * \c float)
 */
template <typename Value>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] constexpr PruneTail<internal::DeducedScalar<Value>>
pruneTail(Value fraction)
{
    return {internal::DeducedScalar<Value>(fraction)};
}

/** \ingroup Upkeep
 * \brief Action that removes the least probable points until at most \c maxSize remain,
 * renormalizing the rest.
 *
 * The state is anything with <tt>pruneToSize(maxSize)</tt>, as for PruneBelow. Under
 * <tt>upkeep::whenSize(std::greater{}, 64, upkeep::pruneToSize(32))</tt> the work is done in batches rather than at
 * every step.
 *
 * \sa PruneBelow, PruneTail
 */
struct PruneToSize
{
    std::ptrdiff_t maxSize;  ///< the most points left, at least one

    /** Prunes \a state. \returns whether any point was removed, or the WeightError of the state's weights */
    template <typename State>
        requires requires(State & state, Eigen::Index size) { state.pruneToSize(size); }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        return internal::prunedAny(state.pruneToSize(Eigen::Index(maxSize)));
    }
};

/** \ingroup Upkeep
 * \returns a PruneToSize action removing the least probable points until at most \a maxSize remain
 */
[[nodiscard]] constexpr PruneToSize
pruneToSize(std::ptrdiff_t maxSize)
{
    return {maxSize};
}

/** \ingroup Upkeep
 * \brief Action that splits every point into \c factor points along each axis, sharing its
 * probability equally.
 *
 * The state is anything with <tt>subdivide(factor)</tt>, such as the GridGaussianParams of a regular
 * grid with at most one cell covariance (see GridGaussianParams::subdivide()). Splitting every cell at
 * once keeps the cells alike, so a filter that shares one covariance among them still can. On a grid
 * of point masses, with no cell covariance, it widens the distribution; GridGaussianParams::subdivide()
 * says by how much.
 *
 * Prune before subdividing: a PruneBelow after it, at the same threshold, removes more, since each
 * child holds a share of its parent's probability. A flat cell index saved before a subdivide means
 * nothing after it.
 */
struct Subdivide
{
    std::ptrdiff_t factor = 2;  ///< the number of new cells along each axis per old cell

    /** Subdivides \a state. \returns whether the factor was above one, so the cells changed */
    template <typename State>
        requires requires(State & state, Eigen::Index factor) { state.subdivide(factor); }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        state.subdivide(Eigen::Index(factor));
        return factor > 1;
    }
};

/** \ingroup Upkeep
 * \returns a Subdivide action splitting every point into \a factor points along each axis, 2 by
 * default
 */
[[nodiscard]] constexpr Subdivide
subdivide(std::ptrdiff_t factor = 2)
{
    return {factor};
}

/** \ingroup Upkeep
 * \brief Action that raises every mode less probable than \c probability to it, scaling the others
 * down to keep the sum one, so that no mode is ruled out for good.
 *
 * The state is anything with <tt>modes()</tt> and <tt>setModes()</tt>, such as the Hybrid of an
 * ImmFilter; a CategoricalParams, the state of a HiddenMarkovFilter, whose values it floors as modes;
 * or a MixtureParams of hybrids, each of whose components it floors in turn.
 *
 * Under Bayes' rule a mode whose probability is zero can never come back, and one that is merely
 * tiny takes as much evidence to come back as it took to fall. An AmmFilter, whose chain moves no
 * probability between modes, then cannot follow a change of mode; applied after each update, the
 * floor keeps every mode within reach of the next few measurements. \c probability times the
 * number of modes must not exceed one.
 *
 * The lower bound on the model probabilities of \ref Athans1977 "Athans et al. (1977)", §V-F, set
 * there so that the identification stays "sensitive to such changes". The paper does not say how
 * the probabilities are brought back to a sum of one. Here the modes above the floor share what
 * the floored ones leave in proportion to their probabilities, and a mode that sharing takes below
 * the floor is floored too, so every mode ends at least at the floor and the ratios among the modes
 * it doesn't floor are unchanged. Unlike the other actions it changes the distribution on purpose, by at most
 * \c probability times the number of modes in total variation, which is why it is upkeep the caller
 * places rather than an option of the filter (decisions.md [autonomous-multiple-model]).
 *
 * \a Scalar_ is the type of \c probability, deduced from the one given
 * (<tt>upkeep::floorModes(1e-3f)</tt> holds a \c float).
 *
 * \sa AmmFilter, HiddenMarkovFilter
 */
template <RealScalar Scalar_ = double>
struct FloorModes
{
    Scalar_ probability;  ///< no mode is left less probable than this

    /** Floors \a state's mode probabilities, read by <tt>modes()</tt> and written by
     * <tt>setModes()</tt>. \returns whether any mode was raised, or the WeightError of mode
     * weights that fail to define a distribution, which are left as they are
     */
    template <typename State>
        requires requires(State & state) {
            state.modes().logTotalWeight();
            state.modes().logWeights();
            state.setModes(state.modes());
        }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(State & state) const
    {
        using ModesType = std::remove_cvref_t<decltype(state.modes())>;
        auto floored = flooredLogWeights(state.modes());
        if (!floored)
        {
            return std::unexpected{floored.error()};
        }
        if (!*floored)
        {
            return false;
        }
        state.setModes(ModesType{state.modes().space(), **floored});
        return true;
    }

    /** Floors the probabilities of \a categorical, the state of a HiddenMarkovFilter, as those of
     * a hybrid's modes. \returns whether any value was raised, or the WeightError of weights that
     * fail to define a distribution, which are left as they are
     */
    template <typename Space, typename Scalar>
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(CategoricalParams<Space, Scalar> & categorical) const
    {
        auto floored = flooredLogWeights(categorical);
        if (!floored)
        {
            return std::unexpected{floored.error()};
        }
        if (!*floored)
        {
            return false;
        }
        categorical.setLogWeights(**floored);
        return true;
    }

    /** Floors the mode probabilities of each component of \a mixture, a mixture of hybrids, and
     * leaves the components' weights as they are. \returns whether any component's mode was
     * raised, or the WeightError of the first component whose mode weights fail to define a
     * distribution, with no component changed
     */
    template <typename Component, typename Allocator>
        requires requires(typename MixtureParams<Component, Allocator>::ComponentType & hybrid) {
            hybrid.setModes(hybrid.modes());
        }
    [[nodiscard]] std::expected<bool, WeightError>
    operator()(MixtureParams<Component, Allocator> & mixture) const
    {
        // Every component's modes are checked before any is changed, so that a failure leaves the
        // mixture as it was, and a mixture with nothing to floor is not touched.
        using Scalar = typename MixtureParams<Component, Allocator>::ComponentType::ModesType::Scalar;
        Scalar const logFloor = Eigen::numext::log(Scalar(probability));
        bool any = false;
        for (Eigen::Index i = 0; i < mixture.size(); ++i)
        {
            auto const & modes = mixture[i].component.modes();
            auto const total = modes.logTotalWeight();
            if (!total)
            {
                return std::unexpected{total.error()};
            }
            any = any || ((modes.logWeights() - *total) < logFloor).any();
        }
        if (!any)
        {
            return false;
        }
        mixture.forEach([&](auto & entry) {
            entry.updateComponent([&](auto & hybrid) {
                (void)(*this)(hybrid);
            });
        });
        return true;
    }

private:

    /** \returns the log-weights of \a modes with every probability below the floor raised to it
     * and the rest scaled to keep the sum one, nothing when no probability is below the floor, or
     * the WeightError of weights that fail to define a distribution
     */
    template <typename Modes>
    [[nodiscard]] std::expected<std::optional<typename Modes::ArrayType>, WeightError>
    flooredLogWeights(Modes const & modes) const
    {
        using ArrayType = typename Modes::ArrayType;
        using Scalar = typename Modes::Scalar;
        auto const total = modes.logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        ArrayType const logProbabilities = modes.logWeights() - *total;
        Eigen::Index const count = logProbabilities.size();
        EMRYS_FILTER_ASSERT(
            probability >= Scalar_(0) && Scalar(probability) * internal::scalarFromCount<Scalar>(count) <= Scalar(1)
            && "the floor is a probability no more than one over the number of modes"
        );
        Scalar const logFloor = Eigen::numext::log(Scalar(probability));
        // The floored modes are those below a threshold, which rises as each round floors more:
        // the modes above it are scaled by the rest of the mass over their own, and any that falls
        // below the floor lowers that scale.
        Scalar threshold = logFloor;
        Scalar scale = 0;
        Eigen::Index floored = (logProbabilities < threshold).count();
        if (floored == 0)
        {
            return std::optional<ArrayType>{};
        }
        while (true)
        {
            auto kept = LogSumExp<Scalar>{};
            for (Eigen::Index m = 0; m < count; ++m)
            {
                if (!(logProbabilities(m) < threshold))
                {
                    kept.add(logProbabilities(m));
                }
            }
            scale = Eigen::numext::log(Scalar(Scalar(1) - Scalar(floored) * Scalar(probability))) - kept.value();
            threshold = logFloor - scale;
            Eigen::Index const next = (logProbabilities < threshold).count();
            if (next == floored)
            {
                break;
            }
            floored = next;
        }
        return std::optional<ArrayType>{
            (logProbabilities < threshold).select(ArrayType::Constant(count, logFloor), logProbabilities + scale)
        };
    }
};

/** \ingroup Upkeep
 * \returns a FloorModes action raising every mode less probable than \a probability to it, held in
 * the scalar it is written in (<tt>floorModes(1e-3f)</tt> holds a \c float)
 */
template <typename Value>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] constexpr FloorModes<internal::DeducedScalar<Value>>
floorModes(Value probability)
{
    return {internal::DeducedScalar<Value>(probability)};
}

/** \ingroup Upkeep
 * \brief Action that repairs every covariance of the state that is not positive definite, by the
 * Repair \a Repair_ (roots::Ridge by default), so that its density can be read.
 *
 * A Kalman filter's covariance can lose positive definiteness by rounding, as the standard update
 * \f$ P - K \Sigma_{xy}^T \f$ does when a measurement is far more precise than the prior: a prior
 * variance of \f$ 10^6 \f$ and a fix of variance \f$ 10^{-12} \f$ leave a variance of exactly zero
 * (decisions.md [covariance-update-forms]). The Joseph form avoids that in the update, at more cost
 * per update; this action repairs the covariance afterward, whatever caused it. Applied to the
 * filter's state() after the updates, it repairs as <tt>state.factorize(repair)</tt> does
 * (decisions.md [covariance-repair]): it first factorizes without the repair, and if that succeeds it
 * changes nothing and returns false; if it fails with GaussianError::NotPositiveDefinite, it factorizes
 * again with the repair, which replaces each covariance that fails with the repaired matrix, and
 * returns true. A repair changes the distribution, for roots::ridge() by the ridge on the diagonal, so
 * a covariance that is positive definite is never touched. Called at every step, it costs one
 * factorization when nothing fails, which the next read of the density reuses.
 *
 * The state is anything with a \c const <tt>factorize()</tt> and a <tt>factorize(repair)</tt>: a
 * GaussianParams, the state of a KalmanFilter; a MixtureParams of Gaussians, the state of a
 * MixtureFilter; or a HybridGaussianParams, the state of an ImmFilter. A mixture and a hybrid repair
 * every component or none, so an error leaves the state as it was. The square-root form needs no
 * repair, being positive semidefinite by construction, and refuses one at compile time; the
 * information forms have no repair (decisions.md [gaussian-forms]).
 *
 * \returns whether a covariance was repaired; the GaussianError of the first factorization if it is
 * not GaussianError::NotPositiveDefinite, as GaussianError::NonFinite is, with the state unchanged; the
 * GaussianError of the repair, as roots::ridge() gives NotPositiveDefinite for a zero covariance,
 * which <tt>roots::ridge<1.0, 1e-12>()</tt> repairs; or the WeightError of a mixture's or hybrid's
 * weights. A mixture's error for a component comes back as its NumericError, without the index.
 *
 * \sa Repair, roots::Ridge, GaussianParams::factorize(R const &)
 */
template <Repair Repair_ = roots::Ridge<>>
struct RepairCovariance
{
    Repair_ repair{};  ///< the repair, roots::ridge() by default

    /** Repairs \a state's covariances that are not positive definite.
     * \returns whether it repaired one, or the NumericError that stopped it
     */
    template <typename State>
        requires requires(State & state, Repair_ const & repair) {
            std::as_const(state).factorize();
            state.factorize(repair);
        }
    [[nodiscard]] std::expected<bool, NumericError>
    operator()(State & state) const
    {
        std::expected<void, NumericError> const factorized =
            internal::factorizeOutcome(std::as_const(state).factorize());
        if (factorized)
        {
            return false;
        }
        if (factorized.error() != NumericError::NotPositiveDefinite)
        {
            return std::unexpected{factorized.error()};
        }
        std::expected<void, NumericError> const repaired = internal::factorizeOutcome(state.factorize(repair));
        if (!repaired)
        {
            return std::unexpected{repaired.error()};
        }
        return true;
    }
};

/** \ingroup Upkeep
 * \returns a RepairCovariance action repairing each covariance that is not positive definite by
 * \a repair, roots::ridge() by default
 */
template <Repair R = roots::Ridge<>>
[[nodiscard]] constexpr RepairCovariance<R>
repairCovariance(R repair = {})
{
    return {std::move(repair)};
}

}  // namespace upkeep

}  // namespace Emrys

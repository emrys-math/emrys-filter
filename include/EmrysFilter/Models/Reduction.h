#pragma once

/** \file
 * Reducing an update's branches (decisions.md [weighted-branches], [reduction-policies], [composed-features]): the
 * Collapse, Select, and Keep policies, the Gate that rejects branches by a chi-squared test before another policy,
 * the ReductionPolicy and MixtureReductionPolicy concepts a filter's update() requires of its policy, which a
 * policy of the user's own meets [user-extension-points], and
 * filterUpdate(), which turns an update into the FilterUpdate a filter returns. Read
 * Consistency.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <EmrysFilter/Models/Consistency.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Core/Weights.h>

namespace Emrys
{

namespace internal
{

/** Whether \a T is an UpdateBranch. */
template <typename T>
inline constexpr bool isUpdateBranch = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space, typename MeasurementSpace, typename Scalar>
inline constexpr bool isUpdateBranch<UpdateBranch<Space, MeasurementSpace, Scalar>> = true;

/// \endcond

/** The branches a reduction policy is given: a sized range of UpdateBranch read by index, which is
 * an update's branches or the ones a Gate passes.
 */
template <typename R>
concept BranchRange = std::ranges::random_access_range<R> && std::ranges::sized_range<R>
                   && isUpdateBranch<std::remove_cvref_t<std::ranges::range_reference_t<R>>>;

/** The branch type of the BranchRange \a R. */
template <typename R>
using BranchOf = std::remove_cvref_t<std::ranges::range_reference_t<R>>;

/** The scalar of the branches of \a R. */
template <typename R>
using BranchScalarOf = decltype(BranchOf<R>::logWeight);

/** The posterior of a branch of \a R: what a policy that keeps one component returns. */
template <typename R>
using PosteriorOf = decltype(BranchOf<R>::posterior);

/** \returns the log of the summed weights of \a branches: zero for every branch of an update */
template <BranchRange R>
[[nodiscard]] BranchScalarOf<R>
logTotalWeight(R const & branches)
{
    auto total = LogSumExp<BranchScalarOf<R>>{};
    for (auto const & branch : branches)
    {
        total.add(branch.logWeight);
    }
    return total.value();
}

/** Whether the reduction output \a Out holds several components, as a mixture filter's does
 * (see ReductionPolicy): it has <tt>static constexpr bool HoldsSeveral = true</tt>.
 */
template <typename Out>
inline constexpr bool holdsSeveral = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Out>
    requires requires { Out::HoldsSeveral; }
inline constexpr bool holdsSeveral<Out> = Out::HoldsSeveral;

/// \endcond

/** The reduction output of a filter that keeps one component per update, such as a Kalman filter,
 * or one per mode or point: the component a policy adds is written to \a target, and its
 * log-weight kept, by which a filter of several modes or points weights it.
 *
 * \tparam Target  \a Component, or a \c std::optional of it
 */
template <typename Component, typename Scalar, typename Target = Component>
struct OneComponentOutput
{
    using ComponentType = Component;  ///< the component a policy adds

    Target & target;       ///< where the added component goes
    Scalar logWeight = 0;  ///< the added component's log-weight (see ReductionPolicy)
    bool added = false;    ///< whether the policy added a component

    /** Writes \a component, moved, to the target, with log-weight \a weight. */
    void
    add(Scalar weight, Component && component)
    {
        EMRYS_FILTER_ASSERT(!added && "this filter keeps one component per update, so a policy adds at most one");
        target = std::move(component);
        logWeight = weight;
        added = true;
    }

    /** Writes a copy of \a component to the target, with log-weight \a weight. */
    void
    add(Scalar weight, Component const & component)
    {
        add(weight, Component(component));
    }
};

/** A reduction output that holds several components, for checking a policy against a mixture
 * filter's output (see MixtureReductionPolicy); it keeps nothing.
 */
template <typename Component, typename Scalar>
struct SeveralComponentOutput
{
    using ComponentType = Component;            ///< the component a policy adds
    static constexpr bool HoldsSeveral = true;  ///< it holds any number

    /** Discards \a component. */
    void
    add(Scalar, Component &&)
    {}

    /** Discards \a component. */
    void
    add(Scalar, Component const &)
    {}
};

/** Whether \a P, given the branches \a Branches, returns the one component it keeps, of type
 * \a Component, in a \c std::optional, or nothing to keep none.
 */
template <typename P, typename Branches, typename Component>
concept ReturnsComponent = requires(P const & policy, Branches && branches) {
    { policy(static_cast<Branches &&>(branches)) } -> std::same_as<std::optional<Component>>;
};

/** Whether \a P reduces the branches \a Branches into the output \a Out: it adds them to \a Out,
 * or returns the component it keeps (see ReturnsComponent).
 */
template <typename P, typename Branches, typename Out>
concept ReducesInto =
    std::invocable<P const &, Branches, Out &> || ReturnsComponent<P, Branches, typename Out::ComponentType>;

/** Reduces \a branches into \a out by \a policy: a policy that returns its component has it added
 * with the log of the branches' summed weight, since it stands for them all.
 */
template <typename P, BranchRange Branches, typename Out>
    requires ReducesInto<P, Branches, Out>
void
reduceInto(P const & policy, Branches && branches, Out & out)
{
    if constexpr (std::invocable<P const &, Branches, Out &>)
    {
        policy(std::forward<Branches>(branches), out);
    }
    else
    {
        auto const logTotal = logTotalWeight(branches);
        if (auto reduced = policy(std::forward<Branches>(branches)))
        {
            out.add(logTotal, std::move(*reduced));
        }
    }
}

/** The branch at index \a i of the range whose first is \a first: how a Gate reads the branches
 * that pass, by their indices.
 */
template <typename Iterator>
struct BranchAt
{
    Iterator first;  ///< the range's first branch

    /** \returns the branch at index \a i */
    [[nodiscard]] decltype(auto)
    operator()(std::size_t i) const
    {
        return first[std::iter_difference_t<Iterator>(i)];
    }
};

/** The branches of \a R a Gate passes, read through their indices, given as \a R is: the range a
 * Gate gives its inner policy.
 */
template <typename R>
using PassingBranches =
    ForwardedLike<R, std::ranges::transform_view<std::span<std::size_t const>, BranchAt<std::ranges::iterator_t<R>>>>;

}  // namespace internal

/** \ingroup Filters
 * \brief The reduction policies, each made by its factory (decisions.md [policy-factories],
 * [reduction-policies]): collapse(), select(), keep(), and gate(). The types beside them name a
 * policy where a type is needed.
 */
namespace reductions
{

/** \ingroup Filters
 * \brief Reduction policy that moment-matches an update's branches into one %Gaussian: soft
 * outlier rejection, as in the probabilistic data association of
 * \ref BarShalom1975 "Bar-Shalom and Tse (1975)".
 *
 * With normalized weights \f$ \hat w_k \f$, the result has the mean
 * \f$ \mu = \sum_k \hat w_k \mu_k \f$ and covariance
 * \f$ \sum_k \hat w_k \left(\Sigma_k + (\mu_k \boxminus \mu)(\mu_k \boxminus \mu)^T\right) \f$,
 * with a tangent-space mean in a non-linear space, as GaussianMixture::moments() computes.
 *
 * Every branch is collapsed together, whatever its mode label; collapse grouped by mode
 * comes with filters that track modes. It returns the one component it keeps, which stands for
 * every branch it was given (see ReductionPolicy); Collapse always returns one.
 *
 * \sa Select, Gate
 */
struct Collapse
{
    /** \returns \a branches moment-matched into one %Gaussian; one branch's posterior is moved out
     * of \a branches given as an rvalue
     */
    template <typename Branches>
        requires internal::BranchRange<Branches>
    [[nodiscard]] std::optional<internal::PosteriorOf<Branches>>
    operator()(Branches && branches) const
    {
        using Posterior = internal::PosteriorOf<Branches>;
        using Scalar = internal::BranchScalarOf<Branches>;
        using MomentsType = typename Posterior::MomentsType;
        EMRYS_FILTER_ASSERT(!std::ranges::empty(branches) && "an update has at least one branch");
        auto const at = std::ranges::begin(branches);
        auto && first = at[0].posterior;
        Eigen::Index const count = Eigen::Index(std::ranges::size(branches));
        if (count == 1)
        {
            return internal::forwardLike<Branches>(first);
        }

        Scalar const logTotal = internal::logTotalWeight(branches);
        EMRYS_FILTER_ASSERT(isFinite(logTotal) && "the branches' weights must be valid");
        auto const branch = [&](Eigen::Index i) -> auto const & {
            return at[i];
        };
        return Posterior{
            first.space(),
            internal::mixtureMomentsIn<MomentsType>(
                first.space(),
                count,
                [&](Eigen::Index i) -> decltype(auto) {
                    return branch(i).posterior.mean();
                },
                [&](Eigen::Index i) -> decltype(auto) {
                    return branch(i).posterior.covarianceLower();
                },
                [&](Eigen::Index i) {
                    return branch(i).logWeight;
                },
                logTotal
            )
        };
    }
};

/** \ingroup Filters
 * \returns the Collapse policy
 */
[[nodiscard]] constexpr Collapse
collapse()
{
    return {};
}

/** \ingroup Filters
 * \brief Reduction policy that keeps the most probable branch: hard rejection by comparing
 * the noise components with each other. Ties go to the earliest branch. The branch kept stands
 * for every branch it was given (see ReductionPolicy), so a filter weights it by their whole
 * likelihood, as it weights Collapse's: it approximates the posterior's shape by its most probable
 * branch and leaves how probable the component is unchanged.
 *
 * \sa Collapse, Gate
 */
struct Select
{
    /** \returns the posterior of the most probable of \a branches, moved out of \a branches given
     * as an rvalue
     */
    template <typename Branches>
        requires internal::BranchRange<Branches>
    [[nodiscard]] std::optional<internal::PosteriorOf<Branches>>
    operator()(Branches && branches) const
    {
        EMRYS_FILTER_ASSERT(!std::ranges::empty(branches) && "an update has at least one branch");
        auto const at = std::ranges::begin(branches);
        std::ptrdiff_t const count = std::ptrdiff_t(std::ranges::size(branches));
        std::ptrdiff_t best = 0;
        for (std::ptrdiff_t i = 1; i < count; ++i)
        {
            if (at[i].logWeight > at[best].logWeight)
            {
                best = i;
            }
        }
        return internal::forwardLike<Branches>(at[best].posterior);
    }
};

/** \ingroup Filters
 * \returns the Select policy
 */
[[nodiscard]] constexpr Select
select()
{
    return {};
}

/** \ingroup Filters
 * \brief Reduction policy that keeps every branch, each added to the filter's output with its own
 * posterior probability (see ReductionPolicy).
 *
 * For filters whose state is a mixture, such as GaussianMixtureFilter: each update multiplies
 * the number of components by the number of noise components, so keep the growth in check
 * between steps (see GaussianMixtureFilter::state()). A KalmanFilter can't hold the result: its
 * output holds one component, and Keep requires one that holds several.
 *
 * \sa Collapse, Select, Gate
 */
struct Keep
{
    /** Adds each of \a branches to \a out with its log-weight, its posterior moved out of
     * \a branches given as an rvalue.
     */
    template <typename Branches, typename Out>
        requires internal::BranchRange<Branches> && internal::holdsSeveral<Out>
    void
    operator()(Branches && branches, Out & out) const
    {
        for (auto && branch : branches)
        {
            out.add(branch.logWeight, internal::forwardLike<Branches>(branch.posterior));
        }
    }
};

/** \ingroup Filters
 * \returns the Keep policy
 */
[[nodiscard]] constexpr Keep
keep()
{
    return {};
}

}  // namespace reductions

/** \ingroup Filters
 * A Gate's threshold given directly as a normalized innovation squared, rather than as a
 * probability, in the scalar it is written in (<tt>NisThreshold{9.21f}</tt> is a \c float).
 * <tt>NisThreshold{chiSquaredQuantile<double>(p, k)}</tt> is the threshold a gate given \c p
 * computes for an innovation of dimension \c k, computed once rather than on every update.
 */
template <RealScalar Scalar = double>
struct NisThreshold
{
    Scalar value;  ///< the largest NIS that passes
};

/// \cond DEDUCTION_GUIDES
template <typename Value>
    requires std::is_arithmetic_v<Value>
NisThreshold(Value) -> NisThreshold<internal::DeducedScalar<Value>>;

/// \endcond

namespace reductions
{

/** \ingroup Filters
 * \brief Reduction policy that rejects branches by a chi-squared test on their normalized
 * innovation squared, then reduces the rest with \a Inner.
 *
 * A branch passes when its NIS (see UpdateBranch) is at most the threshold: the
 * \a probability quantile of the chi-squared distribution with the innovation's dimension as
 * its degrees of freedom, or a NisThreshold given directly. The passing branches, with their
 * weights as they were, go to \a Inner (Collapse by default), and what \a Inner keeps stands for
 * them alone: a filter of several components, modes, or points weights it by the passing
 * branches' likelihood, \f$ \sum_{k \text{ passes}} w_k \, p(z \mid k) \f$, not the whole
 * update's (see ReductionPolicy). When none pass, the measurement is rejected and the state keeps
 * its prior.
 *
 * This is an absolute test of each branch against its own innovation covariance, and needs
 * no outlier model. It composes with mixture noise, which compares branches with each other:
 * <tt>reductions::gate(0.997, reductions::select())</tt> gates each noise component, then keeps the most probable.
 *
 * The validation gate of \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", as given in
 * \ref BarShalom1975 "Bar-Shalom and Tse (1975)", §4, equations (4.3) to (4.5), applied to an
 * update's branches.
 *
 * \a Scalar_ is the type of the probability or threshold it holds, deduced from the one given
 * (<tt>reductions::gate(0.997f)</tt> holds a \c float). The quantile is computed in the update's scalar, so a
 * \c float filter's gate does no \c double arithmetic, and one given a \c float holds no
 * \c double either.
 *
 * A gate given a probability computes its quantile on every update, an iteration that costs many
 * times a small Kalman update, and the gate keeps no result between calls. Where the innovation's
 * dimension is known and the filter runs at a high rate, compute the threshold once and give it
 * as a NisThreshold, which costs a comparison per branch:
 * \code
 * auto const gate = reductions::gate(NisThreshold{chiSquaredQuantile<double>(0.99, 2)});  // 9.21
 * auto const tested = filter.update(fix(z), gate);
 * \endcode
 *
 * \sa Collapse, Select, chiSquaredQuantile()
 */
template <typename Inner = Collapse, RealScalar Scalar_ = double>
class Gate
{
public:

    /** The type of the probability or threshold it holds. */
    using Scalar = Scalar_;

    /** Passes branches whose NIS is within the \a probability chi-squared quantile. */
    explicit Gate(Scalar probability, Inner inner = {})
        : m_probability{probability}
        , m_inner{std::move(inner)}
    {
        EMRYS_FILTER_ASSERT(probability > Scalar(0) && probability < Scalar(1) && "the probability must be in (0, 1)");
    }

    /** Passes branches whose NIS is at most \a threshold. */
    explicit Gate(NisThreshold<Scalar> threshold, Inner inner = {})
        : m_threshold{threshold.value}
        , m_inner{std::move(inner)}
    {
        EMRYS_FILTER_ASSERT(threshold.value >= Scalar(0) && "the threshold must be nonnegative");
    }

    /** \returns the NIS threshold for an innovation with \a dimension coordinates, computed in
     * \a Result: the gate's own scalar unless named, and the update's when operator() asks
     */
    template <RealScalar Result = Scalar>
    [[nodiscard]] Result
    threshold(Eigen::Index dimension) const
    {
        if (m_probability)
        {
            return chiSquaredQuantile<Result>(Result(*m_probability), internal::scalarFromCount<Result>(dimension));
        }
        return Result(m_threshold);
    }

    /** \returns the reduction policy for the branches that pass */
    [[nodiscard]] Inner const &
    inner() const
    {
        return m_inner;
    }

    /** Reduces the branches of \a branches that pass the gate into \a out by \a Inner, or adds
     * nothing when none pass. When some but not all pass, \a Inner reads them through their
     * indices, held in an array on the stack for an array of branches or a run-time number of up
     * to InlinePassing, and otherwise in a buffer from the branches' allocator, the one case that
     * allocates; posteriors are moved out of \a branches given as an rvalue.
     */
    template <typename Branches, typename Out>
        requires internal::BranchRange<Branches> && internal::ReducesInto<Inner, Branches, Out>
              && internal::ReducesInto<Inner, internal::PassingBranches<Branches>, Out>
    void
    operator()(Branches && branches, Out & out) const
    {
        using UpdateScalar = internal::BranchScalarOf<Branches>;
        EMRYS_FILTER_ASSERT(!std::ranges::empty(branches) && "an update has at least one branch");
        auto const at = std::ranges::begin(branches);
        std::size_t const count = std::size_t(std::ranges::size(branches));
        auto const limit = threshold<UpdateScalar>(at[0].innovation.differenceDimension());
        std::size_t passed = 0;
        for (auto const & branch : branches)
        {
            passed += branch.nis() <= limit ? 1 : 0;
        }
        if (passed == 0)
        {
            return;
        }
        if (passed == count)
        {
            internal::reduceInto(m_inner, std::forward<Branches>(branches), out);
            return;
        }

        auto const reducePassing = [&](auto & indices) {
            std::size_t n = 0;
            for (std::size_t k = 0; k < count; ++k)
            {
                if (at[std::ptrdiff_t(k)].nis() <= limit)
                {
                    indices[n++] = k;
                }
            }
            auto passing = std::ranges::transform_view{
                std::span<std::size_t const>{indices.data(), passed},
                internal::BranchAt<std::ranges::iterator_t<Branches>>{at}
            };
            internal::reduceInto(m_inner, internal::forwardLike<Branches>(passing), out);
        };
        using B = std::remove_cvref_t<Branches>;
        if constexpr (requires { std::tuple_size<B>::value; })
        {
            auto indices = std::array<std::size_t, std::tuple_size_v<B>>{};
            reducePassing(indices);
        }
        else if (count <= InlinePassing)
        {
            auto indices = std::array<std::size_t, InlinePassing>{};
            reducePassing(indices);
        }
        else
        {
            auto indices = internal::bufferOf<std::size_t>(allocatorOf(branches));
            indices.resize(passed);
            reducePassing(indices);
        }
    }

    /** The most branches of a run-time number whose passing indices a gate holds on the stack. */
    static constexpr std::size_t InlinePassing = 32;

private:

    /** \returns the allocator of \a branches, or the default one for a range that has none */
    template <typename Branches>
    [[nodiscard]] static auto
    allocatorOf(Branches const & branches)
    {
        if constexpr (requires { branches.get_allocator(); })
        {
            return branches.get_allocator();
        }
        else
        {
            return DefaultAllocator{};
        }
    }

    std::optional<Scalar> m_probability;           ///< the chi-squared probability, if given that way
    Scalar m_threshold = 0;                        ///< the NIS threshold, if given directly
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Inner m_inner;  ///< the policy for the branches that pass
};

/** \ingroup Filters
 * \returns a Gate passing the branches whose NIS is within the \a probability chi-squared
 * quantile, reduced by \a inner (Collapse by default). It holds the probability in the scalar it
 * is written in (<tt>gate(0.997f)</tt> holds a \c float; an integer stands for \c double). It
 * computes the quantile on every update; see Gate for a threshold computed once.
 */
template <typename Value, typename Inner = Collapse>
    requires std::is_arithmetic_v<Value>
[[nodiscard]] Gate<Inner, internal::DeducedScalar<Value>>
gate(Value probability, Inner inner = {})
{
    using Scalar = internal::DeducedScalar<Value>;
    return Gate<Inner, Scalar>{Scalar(probability), std::move(inner)};
}

/** \ingroup Filters
 * \returns a Gate passing the branches whose NIS is at most \a threshold, reduced by \a inner
 * (Collapse by default)
 */
template <RealScalar Scalar, typename Inner = Collapse>
[[nodiscard]] Gate<Inner, Scalar>
gate(NisThreshold<Scalar> threshold, Inner inner = {})
{
    return Gate<Inner, Scalar>{threshold, std::move(inner)};
}

}  // namespace reductions

/** \ingroup Filters
 * \brief A reduction policy for a filter that keeps a \a State per update: it reduces an update's
 * branches to at most one \a State (decisions.md [reduction-policies]).
 *
 * \a Update is a MeasurementUpdate, and the policy is given its branches, as an rvalue: one
 * UpdateBranch per measurement-noise component, each with its \c logWeight (its posterior
 * probability among them), \c posterior, \c innovation, \c nis(), and \c logLikelihood(). A policy
 * sees the branches and nothing else, not the prior (decisions.md [composed-features]); what it
 * needs of the prior is in the posteriors. It may move the posteriors out: the filter reads only
 * the rest after it. It does one of two things:
 *
 * - returns the one component it keeps in a \c std::optional, or nothing to keep none. What it
 *   returns stands for every branch it was given. Collapse and Select do this.
 * - adds what it keeps to the filter's output \c out, given as a second argument, by
 *   <tt>out.add(logWeight, component)</tt>, moving the component in. \c logWeight is the
 *   log-probability, among the branches, of the ones the component stands for: a branch's own
 *   \c logWeight for a branch kept as it is, and the log of the summed weights of the ones merged
 *   into it. Adding nothing keeps none. Keep and Gate do this.
 *
 * Keeping none rejects the measurement, and a filter of one state keeps its prior. A filter of
 * several components, modes, or points weights each one kept by its prior weight, the
 * observation's likelihood, and the added log-weight, so that branches the policy dropped, such
 * as those a Gate fails, are not credited to what it kept; the evidence counts every branch.
 *
 * A filter whose output holds one component, a KalmanRecursion's and those that reduce per mode
 * or per point, requires this of its policy; a policy that adds twice fails an assertion, and
 * Keep, which requires an output that holds several, fails the constraint. A MixtureFilter's
 * output holds several (see MixtureReductionPolicy). usage.md, "Writing a reduction policy",
 * shows one of the user's own (decisions.md [user-extension-points]).
 */
template <typename P, typename Update, typename State>
concept ReductionPolicy = internal::ReducesInto<
    P,
    typename std::remove_cvref_t<Update>::Container,
    internal::OneComponentOutput<State, typename std::remove_cvref_t<Update>::Scalar>>;

/** \ingroup Filters
 * \brief A reduction policy a MixtureFilter over \a Component_ takes for \a Update: one that
 * reduces the branches into an output that holds several components (see ReductionPolicy), so
 * that one prior component becomes several, as Keep makes it.
 */
template <typename P, typename Update, typename Component_>
concept MixtureReductionPolicy = internal::ReducesInto<
    P,
    typename std::remove_cvref_t<Update>::Container,
    internal::SeveralComponentOutput<Params<Component_>, typename std::remove_cvref_t<Update>::Scalar>>;

/** \ingroup Filters
 * \returns the one \a State that \a policy, a ReductionPolicy, keeps of \a branches, or nothing
 * when it keeps none: how a recursion of the user's own runs the policy it is given, whichever of
 * the two forms of ReductionPolicy it has (decisions.md [user-extension-points]). Posteriors are
 * moved out of \a branches given as an rvalue.
 */
template <typename State, typename Policy, typename Branches>
    requires internal::BranchRange<Branches>
          && internal::ReducesInto<
                 Policy,
                 Branches,
                 internal::OneComponentOutput<State, internal::BranchScalarOf<Branches>, std::optional<State>>>
[[nodiscard]] std::optional<State>
reduceToOne(Policy const & policy, Branches && branches)
{
    auto kept = std::optional<State>{};
    auto out = internal::OneComponentOutput<State, internal::BranchScalarOf<Branches>, std::optional<State>>{kept};
    internal::reduceInto(policy, std::forward<Branches>(branches), out);
    return kept;
}

namespace internal
{

/** Whether \a Policy keeps at most one component of the branches \a Branches of type
 * \a Component: it returns the one it keeps (see ReturnsComponent), or it is a Gate over such a
 * policy. A mixture filter updates a component in place under such a policy.
 */
template <typename Policy, typename Branches, typename Component>
inline constexpr bool keepsOne = ReturnsComponent<Policy, Branches, Component>;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Inner, typename Scalar, typename Branches, typename Component>
inline constexpr bool keepsOne<reductions::Gate<Inner, Scalar>, Branches, Component> =
    keepsOne<Inner, Branches, Component>;

/// \endcond

/** Whether reducing an update with \a Policy may read the branches' NIS or log-likelihood, so
 * that a filter asks the update for them with the gain (decisions.md [innovation-statistics]): any policy other than
 * Collapse, Select, and Keep, which read only the weights and posteriors.
 */
template <typename Policy>
inline constexpr bool policyReadsStatistics = true;

/// \cond INTERNAL_SPECIALIZATIONS
template <>
inline constexpr bool policyReadsStatistics<reductions::Collapse> = false;

template <>
inline constexpr bool policyReadsStatistics<reductions::Select> = false;

template <>
inline constexpr bool policyReadsStatistics<reductions::Keep> = false;

/// \endcond

/** \a Policy as a filter that reads each branch's likelihood passes it on to a recursion: the
 * same reduction, but one for which policyReadsStatistics holds, so the update computes the
 * statistics from the gain's factorization rather than factorizing the innovation covariance a
 * second time when they are read (decisions.md [innovation-statistics]). Valid while \a policy is.
 */
template <typename Policy>
struct WithStatistics
{
    Policy const & policy;  ///< the policy that reduces

    /** Reduces \a branches into \a out as the policy does (see reduceInto()). */
    template <typename Branches, typename Out>
        requires BranchRange<Branches> && ReducesInto<Policy, Branches, Out>
    void
    operator()(Branches && branches, Out & out) const
    {
        reduceInto(policy, std::forward<Branches>(branches), out);
    }
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Policy>
inline constexpr bool policyReadsStatistics<WithStatistics<Policy>> = true;

/// \endcond

/** Whether \a Policy is Collapse, with or without WithStatistics around it: the policy under which
 * an update of one branch is the new state, conditioned in place (decisions.md [borrowed-intermediates]).
 */
template <typename Policy>
inline constexpr bool isCollapsePolicy = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <>
inline constexpr bool isCollapsePolicy<reductions::Collapse> = true;

template <typename Policy>
inline constexpr bool isCollapsePolicy<WithStatistics<Policy>> = isCollapsePolicy<Policy>;

/// \endcond

/** Whether \a Policy reduces one branch to that branch, or to nothing: Collapse, Select, or a Gate
 * over either, with or without WithStatistics around it. Under such a policy an update of one
 * branch is conditioned in place, the gate deciding on the NIS before the state is written, so a
 * rejection leaves the state as it was with nothing copied (decisions.md [update-outcomes]).
 */
template <typename Policy>
inline constexpr bool isOneBranchInPlacePolicy = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <>
inline constexpr bool isOneBranchInPlacePolicy<reductions::Collapse> = true;

template <>
inline constexpr bool isOneBranchInPlacePolicy<reductions::Select> = true;

template <typename Scalar>
inline constexpr bool isOneBranchInPlacePolicy<reductions::Gate<reductions::Collapse, Scalar>> = true;

template <typename Scalar>
inline constexpr bool isOneBranchInPlacePolicy<reductions::Gate<reductions::Select, Scalar>> = true;

template <typename Policy>
inline constexpr bool isOneBranchInPlacePolicy<WithStatistics<Policy>> = isOneBranchInPlacePolicy<Policy>;

/// \endcond

/** The acceptor (see AcceptEvery) by which a Gate decides an update of one branch in place: it
 * accepts an innovation whose NIS is within the gate's threshold for its dimension, the test
 * Gate::operator() makes of a branch. Valid while \a gate is.
 */
template <typename Gate>
struct AcceptByGate
{
    Gate const & gate;  ///< the gate that decides

    /** \returns whether \a nis is within the gate's threshold for an innovation of \a dimension */
    template <typename Scalar>
    [[nodiscard]] bool
    operator()(Scalar nis, Eigen::Index dimension) const
    {
        return nis <= gate.template threshold<Scalar>(dimension);
    }
};

/** \returns the acceptor by which \a policy, a one-branch in-place policy (see
 * isOneBranchInPlacePolicy), decides an update of one branch: AcceptEvery for Collapse and Select,
 * and an AcceptByGate, valid while \a policy is, for a Gate
 */
template <typename Policy>
[[nodiscard]] auto
oneBranchAcceptor(Policy const & policy)
{
    static_assert(isOneBranchInPlacePolicy<Policy>);
    if constexpr (requires { policy.policy; })
    {
        return oneBranchAcceptor(policy.policy);
    }
    else if constexpr (requires { policy.inner(); })
    {
        return AcceptByGate<Policy>{policy};
    }
    else
    {
        return AcceptEvery{};
    }
}

/** \returns \a measured as a FilterUpdate with \a accepted: its branches' weights, innovations,
 * labels, and iteration reports, moved, without their posteriors
 */
template <typename Scalar, typename Space, typename MeasurementSpace, std::size_t Count, typename Allocator>
[[nodiscard]] FilterUpdate<MeasurementSpace, Count, Scalar, Allocator>
filterUpdate(MeasurementUpdate<Space, MeasurementSpace, Count, Scalar, Allocator> && measured, bool accepted)
{
    using Result = FilterUpdate<MeasurementSpace, Count, Scalar, Allocator>;
    using Branch = typename Result::BranchType;
    auto const strip = [](auto & branch) {
        return Branch{
            branch.logWeight,
            std::move(branch.innovation),
            branch.mode,
            branch.component,
            branch.iterations,
            branch.converged
        };
    };
    if constexpr (Count == std::dynamic_extent)
    {
        auto result = Result{
            typename Result::Container(measured.get_allocator()),
            MixtureEvidenceAccess::mixtureLogLikelihood(measured),
            accepted
        };
        result.branches.reserve(measured.branches.size());
        for (auto & branch : measured.branches)
        {
            result.branches.push_back(strip(branch));
        }
        return result;
    }
    else
    {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Result{
                {strip(measured.branches[I])...},
                MixtureEvidenceAccess::mixtureLogLikelihood(measured),
                accepted
            };
        }(std::make_index_sequence<Count>{});
    }
}

/** filterUpdate() into \a result, whose branches are refilled within their capacity, so that a
 * caller who keeps the result allocates nothing once it has held as many (see \ref Allocation).
 * \a measured's innovations are moved out.
 */
template <
    typename Scalar,
    typename Space,
    typename MeasurementSpace,
    std::size_t Count,
    typename MeasuredAllocator,
    typename ResultAllocator>
void
filterUpdateInto(
    MeasurementUpdate<Space, MeasurementSpace, Count, Scalar, MeasuredAllocator> & measured,
    bool accepted,
    FilterUpdate<MeasurementSpace, Count, Scalar, ResultAllocator> & result
)
{
    if constexpr (Count == std::dynamic_extent)
    {
        using Branch = typename FilterUpdate<MeasurementSpace, Count, Scalar, ResultAllocator>::BranchType;
        result.branches.clear();
        result.branches.reserve(measured.branches.size());
        for (auto & branch : measured.branches)
        {
            result.branches.push_back(
                Branch{
                    branch.logWeight,
                    std::move(branch.innovation),
                    branch.mode,
                    branch.component,
                    branch.iterations,
                    branch.converged
                }
            );
        }
        MixtureEvidenceAccess::setMixtureLogLikelihood(result, MixtureEvidenceAccess::mixtureLogLikelihood(measured));
        MixtureEvidenceAccess::setLogKeptWeight(result, Scalar(0));
        result.accepted = accepted;
    }
    else
    {
        result = filterUpdate(std::move(measured), accepted);
    }
}

/** The update of \a prior by \a measurement, reduced by \a reduce(measured), which returns whether
 * it accepted the measurement, and stripped into slot \a i of \a slots, a container of a filter's
 * updates, which is made from \a allocator if \a slots has only \a i so far. A slot of a run-time
 * number of branches (a ReusableFilterUpdate) is measured into its own scratch, so with the same
 * slots at every step nothing is allocated once they have held as many branches (see
 * \ref Allocation); a slot of an array of branches is built from the measured update.
 *
 * \returns nothing, or the update's error, with \a reduce not called (decisions.md [innovation-cholesky])
 */
template <bool Statistics, typename Slots, typename Allocator, UncertainInput Prior, Measurement M, typename Reduce>
[[nodiscard]] std::expected<void, GaussianError>
updateSlot(
    Slots & slots,
    std::size_t i,
    Allocator const & allocator,
    Prior const & prior,
    M const & measurement,
    Reduce const & reduce
)
{
    using Update = typename Slots::value_type;
    EMRYS_FILTER_ASSERT(i <= slots.size() && "the slots before this one must exist");
    if constexpr (requires(Update & update) { update.measured; })
    {
        if (i == slots.size())
        {
            slots.push_back(Update(allocator));
        }
        auto & slot = slots[i];
        auto const measured = updateBranchesInto<Statistics>(prior, measurement, slot.measured);
        if (!measured)
        {
            return measured;
        }
        bool const accepted = reduce(slot.measured);
        filterUpdateInto(slot.measured, accepted, slot);
    }
    else
    {
        auto measured = updateBranches<Statistics>(prior, measurement);
        if (!measured)
        {
            return std::unexpected{measured.error()};
        }
        bool const accepted = reduce(*measured);
        auto update = filterUpdate(std::move(*measured), accepted);
        if (i == slots.size())
        {
            slots.push_back(std::move(update));
        }
        else
        {
            slots[i] = std::move(update);
        }
    }
    return {};
}

/** As updateSlot() of a measurement, but of \a given, a measurement with a block of the state
 * given (see GivenMeasurement), whose update is updateBranchesWith(). A slot that keeps its own
 * scratch receives the branches measured apart.
 *
 * \returns nothing, or the update's error, with \a reduce not called
 */
template <
    bool Statistics,
    typename Slots,
    typename Allocator,
    UncertainInput Prior,
    GivenMeasurementLike G,
    typename Reduce>
[[nodiscard]] std::expected<void, GaussianError>
updateSlot(
    Slots & slots,
    std::size_t i,
    Allocator const & allocator,
    Prior const & prior,
    G const & given,
    Reduce const & reduce
)
{
    using Update = typename Slots::value_type;
    EMRYS_FILTER_ASSERT(i <= slots.size() && "the slots before this one must exist");
    auto measured = updateBranchesWith<Statistics>(prior, given.pinned, given.value);
    if (!measured)
    {
        return std::unexpected{measured.error()};
    }
    bool const accepted = reduce(*measured);
    if constexpr (requires(Update & update) { update.measured; })
    {
        if (i == slots.size())
        {
            slots.push_back(Update(allocator));
        }
        filterUpdateInto(*measured, accepted, slots[i]);
    }
    else
    {
        auto update = filterUpdate(std::move(*measured), accepted);
        if (i == slots.size())
        {
            slots.push_back(std::move(update));
        }
        else
        {
            slots[i] = std::move(update);
        }
    }
    return {};
}

}  // namespace internal

}  // namespace Emrys

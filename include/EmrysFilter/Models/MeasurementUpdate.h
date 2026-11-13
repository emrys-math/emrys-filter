#pragma once

/** \file
 * What an update returns (decisions.md [weighted-branches], [innovation-statistics]): an UpdateBranch per
 * noise component in a MeasurementUpdate, and FilterUpdate and ReusableFilterUpdate, what a filter's update
 * returns without the posteriors, with the internal rebinding of an update to an allocator. Read
 * MeasurementReads.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Models/Linearization.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Models
 * One outcome of an update: a posterior, how probable it is, and its innovation, from which the
 * observation's likelihood under it and its NIS follow (decisions.md [innovation-statistics]).
 *
 * \tparam Space            the state's space
 * \tparam MeasurementSpace  the measurement space
 */
template <typename Space, typename MeasurementSpace, typename Scalar = double>
struct UpdateBranch
{
    Scalar logWeight;                         ///< log of its posterior probability among the branches of one update
    GaussianParams<Space, Scalar> posterior;  ///< the state conditioned on the observation under this branch
    Innovation<MeasurementSpace, Scalar> innovation;  ///< the observation's residual under it and its covariance
    Eigen::Index mode = 0;   ///< flat index of the mode's discrete value, set by the IMM; update() gives 0
    Eigen::Index component;  ///< index of the noise component, such as inlier or outlier

    /** the linearizations the update made: 1, except under the Iterated method */
    std::size_t iterations = 1;

    /** false when an Iterated update stopped at its iteration cap rather than by its tolerance */
    bool converged = true;

    /** \returns the log-likelihood of the observation under this branch */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return innovation.logLikelihood();
    }

    /** \returns the normalized innovation squared, chi-squared with the innovation's difference
     * dimension as its degrees of freedom for a consistent filter (see Innovation::nis())
     */
    [[nodiscard]] Scalar
    nis() const
    {
        return innovation.nis();
    }
};

namespace internal
{

struct MixtureEvidenceAccess;

}  // namespace internal

/** \ingroup Models
 * The result of update(): weighted branches, one per noise component, and the observation's
 * log-likelihood. A single %Gaussian noise gives one branch.
 *
 * \tparam Space            the state's space
 * \tparam MeasurementSpace  the measurement space
 * \tparam Count            number of branches if known at compile time, otherwise \c std::dynamic_extent
 * \tparam Scalar_          the scalar type
 * \tparam Allocator        where a run-time number of branches comes from (see \ref Allocation)
 */
template <
    typename Space,
    typename MeasurementSpace,
    std::size_t Count = std::dynamic_extent,
    typename Scalar_ = double,
    typename Allocator = DefaultAllocator>
struct MeasurementUpdate
{
    using Scalar = Scalar_;                                            ///< the scalar type
    using SpaceType = Space;                                           ///< the state's space
    using MeasurementSpaceType = MeasurementSpace;                     ///< the measurement space
    using BranchType = UpdateBranch<Space, MeasurementSpace, Scalar>;  ///< one branch
    using allocator_type = Allocator;  ///< where a run-time number of branches comes from

    /** The branches' container: an array when their number is known at compile time, and
     * otherwise a buffer from \a Allocator.
     */
    using Container = std::conditional_t<
        Count == std::dynamic_extent,
        internal::Buffer<BranchType, Allocator>,
        std::array<BranchType, Count>>;

    static constexpr std::size_t BranchCount = Count;  ///< the number of branches, or std::dynamic_extent

    Container branches;  ///< one per noise component, weights summing to one

    MeasurementUpdate() = default;  ///< no branches

    /** The branches \a branches of an observation whose log-likelihood, where they are more than
     * the one a single noise component gives, is \a mixtureLogLikelihood
     */
    MeasurementUpdate(Container branches, Scalar mixtureLogLikelihood = 0)
        : branches{std::move(branches)}
        , m_mixtureLogLikelihood{mixtureLogLikelihood}
    {}

    /** \returns the observation's log-likelihood, \f$ \log p(z) = \log \sum_k w_k \, p(z \mid k) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        if constexpr (Count == 1)
        {
            return branches.front().logLikelihood();
        }
        else
        {
            return m_mixtureLogLikelihood;
        }
    }

    /** \returns the difference dimension of the innovation, the degrees of freedom of each NIS */
    [[nodiscard]] Eigen::Index
    innovationSize() const
    {
        return branches.front().innovation.differenceDimension();
    }

    /** \returns the allocator a run-time number of branches comes from, or a default-constructed
     * one for an array of them
     */
    [[nodiscard]] Allocator
    get_allocator() const
    {
        if constexpr (Count == std::dynamic_extent)
        {
            return Allocator(branches.get_allocator());
        }
        else
        {
            return Allocator{};
        }
    }

private:

    friend struct internal::MixtureEvidenceAccess;  ///< the weighting of the branches writes the log-likelihood

    /** \f$ \log \sum_k w_k \, p(z \mid k) \f$, which weighting the branches of more than one noise
     * component computes and logLikelihood() returns; with one, logLikelihood() is the branch's
     */
    Scalar m_mixtureLogLikelihood = 0;
};

namespace internal
{

/** Whether \a T is a MeasurementUpdate. */
template <typename T>
inline constexpr bool isMeasurementUpdate = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space, typename MeasurementSpace, std::size_t Count, typename Scalar, typename Allocator>
inline constexpr bool isMeasurementUpdate<MeasurementUpdate<Space, MeasurementSpace, Count, Scalar, Allocator>> = true;

/// \endcond

}  // namespace internal

/** \ingroup Models
 * One branch of a filter's update (see FilterUpdate): its weight, its innovation, and its labels,
 * without the posterior, which the filter reduced into its state (decisions.md [innovation-statistics]).
 *
 * \tparam MeasurementSpace  the measurement space
 */
template <typename MeasurementSpace, typename Scalar = double>
struct FilterBranch
{
    Scalar logWeight;  ///< log of its posterior probability among the branches of one update
    Innovation<MeasurementSpace, Scalar> innovation;  ///< the observation's residual under it and its covariance
    Eigen::Index mode = 0;       ///< flat index of the mode's discrete value, as UpdateBranch::mode
    Eigen::Index component;      ///< index of the noise component, such as inlier or outlier
    std::size_t iterations = 1;  ///< the linearizations made, as UpdateBranch::iterations
    bool converged = true;       ///< as UpdateBranch::converged

    /** \returns the log-likelihood of the observation under this branch */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        return innovation.logLikelihood();
    }

    /** \returns the normalized innovation squared (see Innovation::nis()) */
    [[nodiscard]] Scalar
    nis() const
    {
        return innovation.nis();
    }
};

/** \ingroup Models
 * \brief What a filter's update did: the branches of the update before reduction, with their
 * weights and innovations but not their posteriors (decisions.md [innovation-statistics]), and whether the
 * reduction policy accepted the measurement.
 *
 * The log-likelihood and each branch's NIS are computed from the innovations when first asked
 * for, unless the filter had them already. When \c accepted is false, the policy (such as a
 * Gate) rejected the measurement and the state kept its prior. Pass it to a NisMonitor as any
 * update.
 *
 * \tparam MeasurementSpace  the measurement space
 * \tparam Count            number of branches if known at compile time, otherwise \c std::dynamic_extent
 * \tparam Scalar_          the scalar type
 * \tparam Allocator        where a run-time number of branches comes from (see \ref Allocation)
 */
template <
    typename MeasurementSpace,
    std::size_t Count = std::dynamic_extent,
    typename Scalar_ = double,
    typename Allocator = DefaultAllocator>
struct FilterUpdate
{
    using Scalar = Scalar_;                                     ///< the scalar type
    using MeasurementSpaceType = MeasurementSpace;              ///< the measurement space
    using BranchType = FilterBranch<MeasurementSpace, Scalar>;  ///< one branch
    using allocator_type = Allocator;                           ///< where a run-time number of branches comes from

    /** The branches' container: an array when their number is known at compile time, and
     * otherwise a buffer from \a Allocator.
     */
    using Container = std::conditional_t<
        Count == std::dynamic_extent,
        internal::Buffer<BranchType, Allocator>,
        std::array<BranchType, Count>>;

    static constexpr std::size_t BranchCount = Count;  ///< the number of branches, or std::dynamic_extent

    Container branches;  ///< one per noise component, weights summing to one

    bool accepted = false;  ///< whether the policy accepted the measurement

    FilterUpdate() = default;  ///< no branches, not accepted

    /** The branches \a branches, as MeasurementUpdate(branches, mixtureLogLikelihood), and whether
     * the policy accepted the measurement, \a accepted
     */
    FilterUpdate(Container branches, Scalar mixtureLogLikelihood = 0, bool accepted = false)
        : branches{std::move(branches)}
        , accepted{accepted}
        , m_mixtureLogLikelihood{mixtureLogLikelihood}
    {}

    /** \returns the observation's log-likelihood, \f$ \log p(z) = \log \sum_k w_k \, p(z \mid k) \f$ */
    [[nodiscard]] Scalar
    logLikelihood() const
    {
        if constexpr (Count == 1)
        {
            return branches.front().logLikelihood();
        }
        else
        {
            return m_mixtureLogLikelihood;
        }
    }

    /** \returns the difference dimension of the innovation, the degrees of freedom of each NIS */
    [[nodiscard]] Eigen::Index
    innovationSize() const
    {
        return branches.front().innovation.differenceDimension();
    }

    /** \returns the allocator a run-time number of branches comes from, or a default-constructed
     * one for an array of them
     */
    [[nodiscard]] Allocator
    get_allocator() const
    {
        if constexpr (Count == std::dynamic_extent)
        {
            return Allocator(branches.get_allocator());
        }
        else
        {
            return Allocator{};
        }
    }

private:

    friend struct internal::MixtureEvidenceAccess;  ///< the weighting of the branches writes the log-likelihood

    /** \f$ \log \sum_k w_k \, p(z \mid k) \f$, which weighting the branches of more than one noise
     * component computes and logLikelihood() returns; with one, logLikelihood() is the branch's
     */
    Scalar m_mixtureLogLikelihood = 0;

    /** The log of the posterior probability of the branches the kept state stands for, as the
     * policy added it (see ReductionPolicy): zero unless the policy dropped some, as a Gate does. A
     * mixture filter weights the component by it.
     */
    Scalar m_logKeptWeight = 0;
};

namespace internal
{

/** Access to the mixture log-likelihood of a MeasurementUpdate or a FilterUpdate, for the code that
 * weights its branches or copies it: their logLikelihood() is its only public reader, since it means
 * nothing where one noise component gives one branch. It also reaches a FilterUpdate's kept
 * weight, which a recursion writes and a mixture filter reads.
 */
struct MixtureEvidenceAccess
{
    /** Sets the log of the posterior probability of the branches \a update's kept state stands
     * for to \a value.
     */
    template <typename U>
    static void
    setLogKeptWeight(U & update, typename U::Scalar value)
    {
        update.m_logKeptWeight = value;
    }

    /** \returns the log of the posterior probability of the branches \a update's kept state stands
     * for
     */
    template <typename U>
    [[nodiscard]] static typename U::Scalar
    logKeptWeight(U const & update)
    {
        return update.m_logKeptWeight;
    }

    /** Sets \a update's mixture log-likelihood to \a value. */
    template <typename U>
    static void
    setMixtureLogLikelihood(U & update, typename U::Scalar value)
    {
        update.m_mixtureLogLikelihood = value;
    }

    /** \returns \a update's mixture log-likelihood, to copy into another update */
    template <typename U>
    [[nodiscard]] static typename U::Scalar
    mixtureLogLikelihood(U const & update)
    {
        return update.m_mixtureLogLikelihood;
    }
};

}  // namespace internal

/** \ingroup Models
 * \brief A FilterUpdate of a run-time number of branches that keeps the MeasurementUpdate it was
 * last filled from, posteriors included, as the scratch of its next fill.
 *
 * A filter's update under mixture noise conditions the prior once per noise component, reduces
 * the posteriors with its policy, and returns the branches without them (decisions.md [innovation-statistics]). The
 * branches with their posteriors are sized by the number of components, so a filter that refills
 * one of these, such as KalmanRecursion::update() into a result, allocates nothing once the
 * result has held as many (see \ref Allocation). The scratch is #measured, whose posteriors are
 * what the policy left: the one it kept is moved from.
 *
 * \tparam Space            the state's space
 * \tparam MeasurementSpace  the measurement space
 * \tparam Scalar           the scalar type
 * \tparam Allocator        where the branches come from
 */
template <typename Space, typename MeasurementSpace, typename Scalar = double, typename Allocator = DefaultAllocator>
struct ReusableFilterUpdate : FilterUpdate<MeasurementSpace, std::dynamic_extent, Scalar, Allocator>
{
    using Base = FilterUpdate<MeasurementSpace, std::dynamic_extent, Scalar, Allocator>;  ///< the update it is

    /** The measurement's branches it keeps. */
    using MeasuredType = MeasurementUpdate<Space, MeasurementSpace, std::dynamic_extent, Scalar, Allocator>;

    ReusableFilterUpdate() = default;  ///< empty, from a default-constructed allocator

    /** Empty, with its branches and scratch from \a allocator. */
    explicit ReusableFilterUpdate(Allocator const & allocator)
        : Base{typename Base::Container(allocator), Scalar(0), false}
        , measured{typename MeasuredType::Container(allocator), Scalar(0)}
    {}

    ReusableFilterUpdate(ReusableFilterUpdate const &) = default;      ///< copies, keeping the source's allocator
    ReusableFilterUpdate(ReusableFilterUpdate &&) noexcept = default;  ///< takes \a other's storage

    /** Copies \a other into storage from \a allocator (see \ref Allocation). */
    ReusableFilterUpdate(ReusableFilterUpdate const & other, Allocator const & allocator)
        : Base{
              typename Base::Container(other.branches, typename Base::Container::allocator_type(allocator)),
              internal::MixtureEvidenceAccess::mixtureLogLikelihood(other),
              other.accepted
          }
        , measured{
              typename MeasuredType::Container(
                  other.measured.branches,
                  typename MeasuredType::Container::allocator_type(allocator)
              ),
              internal::MixtureEvidenceAccess::mixtureLogLikelihood(other.measured)
          }
    {
        internal::MixtureEvidenceAccess::setLogKeptWeight(*this, internal::MixtureEvidenceAccess::logKeptWeight(other));
    }

    /** Moves \a other into storage from \a allocator, taking its storage when the allocators are
     * equal.
     */
    ReusableFilterUpdate(ReusableFilterUpdate && other, Allocator const & allocator)
        : Base{
              typename Base::Container(std::move(other.branches), typename Base::Container::allocator_type(allocator)),
              internal::MixtureEvidenceAccess::mixtureLogLikelihood(other),
              other.accepted
          }
        , measured{
              typename MeasuredType::Container(
                  std::move(other.measured.branches),
                  typename MeasuredType::Container::allocator_type(allocator)
              ),
              internal::MixtureEvidenceAccess::mixtureLogLikelihood(other.measured)
          }
    {
        internal::MixtureEvidenceAccess::setLogKeptWeight(*this, internal::MixtureEvidenceAccess::logKeptWeight(other));
    }

    ReusableFilterUpdate &
    operator=(ReusableFilterUpdate const &) = default;  ///< copies, keeping this allocator

    ReusableFilterUpdate &
    operator=(ReusableFilterUpdate &&) noexcept = default;  ///< takes \a other's storage

    ~ReusableFilterUpdate() = default;

    MeasuredType measured;  ///< the branches of the last fill, posteriors included, reused by the next
};

namespace internal
{

/** Whether \a U is a FilterUpdate, or derives from one as a ReusableFilterUpdate does. */
template <typename MeasurementSpace, std::size_t Count, typename Scalar, typename Allocator>
std::true_type
isFilterUpdateTest(FilterUpdate<MeasurementSpace, Count, Scalar, Allocator> const *);

/** Not a FilterUpdate. */
std::false_type
isFilterUpdateTest(void const *);

/** Whether \a U is a FilterUpdate, or derives from one. */
template <typename U>
inline constexpr bool isFilterUpdate = decltype(isFilterUpdateTest(static_cast<U const *>(nullptr)))::value;

/** The FilterUpdate a filter refills: a ReusableFilterUpdate for a run-time number of branches,
 * which needs scratch, and the plain FilterUpdate otherwise.
 */
template <typename Space, typename MeasurementSpace, std::size_t Count, typename Scalar, typename Allocator>
struct FilterUpdateWithScratchOf
{
    /** An array of branches, which holds nothing from an allocator, so it is the plain update. */
    using type = FilterUpdate<MeasurementSpace, Count, Scalar>;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space, typename MeasurementSpace, typename Scalar, typename Allocator>
struct FilterUpdateWithScratchOf<Space, MeasurementSpace, std::dynamic_extent, Scalar, Allocator>
{
    using type = ReusableFilterUpdate<Space, MeasurementSpace, Scalar, Allocator>;  ///< branches with their scratch
};

/// \endcond

/** The FilterUpdate a filter refills (see FilterUpdateWithScratchOf). */
template <typename Space, typename MeasurementSpace, std::size_t Count, typename Scalar, typename Allocator>
using FilterUpdateWithScratch =
    typename FilterUpdateWithScratchOf<Space, MeasurementSpace, Count, Scalar, Allocator>::type;

/** \a Update, a filter's update type, with \a Allocator as the allocator of whatever in it is
 * sized by a run-time count: a ReusableFilterUpdate's branches and scratch, an ImmFilterUpdate's
 * modes (specialized in ImmFilter.h), or the update inside a form's \c std::expected. A filter
 * that holds a run-time number of elements, such as a MixtureFilter, gives its components'
 * updates its own allocator this way (see \ref Allocation).
 */
template <typename Update, typename Allocator>
struct RebindUpdateAllocatorOf
{
    /** Unchanged: an array of branches holds nothing sized by a count. */
    using type = Update;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Space, typename MeasurementSpace, typename Scalar, typename Old, typename Allocator>
struct RebindUpdateAllocatorOf<ReusableFilterUpdate<Space, MeasurementSpace, Scalar, Old>, Allocator>
{
    using type = ReusableFilterUpdate<Space, MeasurementSpace, Scalar, Allocator>;
};

template <typename Update, typename Error, typename Allocator>
struct RebindUpdateAllocatorOf<std::expected<Update, Error>, Allocator>
{
    using type = std::expected<typename RebindUpdateAllocatorOf<Update, Allocator>::type, Error>;
};

/// \endcond

/** \a Update with \a Allocator for what in it is sized by a run-time count (see
 * RebindUpdateAllocatorOf).
 */
template <typename Update, typename Allocator>
using RebindUpdateAllocator = typename RebindUpdateAllocatorOf<Update, Allocator>::type;

/** Whether \a Update, a filter's update type, can be made empty from an allocator, to be filled
 * in place: a ReusableFilterUpdate, or an ImmFilterUpdate, whose modes come from it. An array of
 * branches can't, since a branch holds an innovation, and is made from the measured update.
 */
template <typename Update, typename Allocator>
concept EmptyFromAllocator = std::is_constructible_v<Update, Allocator const &>;

/** \returns an empty \a Update whose storage comes from \a allocator: a ReusableFilterUpdate's,
 * or an ImmFilterUpdate's modes
 */
template <typename Update, typename Allocator>
    requires EmptyFromAllocator<Update, Allocator>
[[nodiscard]] Update
emptyUpdate(Allocator const & allocator)
{
    return Update(allocator);
}

}  // namespace internal

}  // namespace Emrys

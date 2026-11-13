#pragma once

/** \file
 * The hybrid state of a discrete mode and a component per mode (decisions.md [imm], [component-first]): Hybrid,
 * with its collapse into one component, whole or grouped by a mode axis, its view as a mixture,
 * its marginal, a MixtureMarginal over the modes [element-marginals], and its ComponentTraits and
 * internal::ElementsOf, which make a hybrid state a component in its own right, so a
 * Mixture of hybrids is the state of a mixture of interacting multiple model filters [filters-and-recursions]; and
 * the HybridGaussian alias for the hybrid of Gaussians, the interacting multiple model filter's
 * state. Read ParticleSet.h next; the design review is
 * docs/guide/imm-filter.md.
 */

#include <concepts>
#include <cstddef>
#include <expected>
#include <memory>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

template <internal::ComponentOrInfoForm Component_, DiscreteSpace ModeSpace_, typename Allocator_ = DefaultAllocator>
class Hybrid;

/** \ingroup Distributions
 * \brief The parameters of a hybrid state: a discrete mode, and a component conditioned on each
 * mode.
 *
 * The density, evaluated by its Hybrid from toDensity(), is \f$ p(x, m) = P(m) \, p_m(x) \f$, with
 * \a ModeSpace_ a discrete space (see
 * \ref Discrete) and one component (see Component) for each of its values: a Gaussian, which
 * makes the hybrid a HybridGaussian, \f$ p_m = \mathcal{N}(\mu_m, \Sigma_m) \f$, or any other
 * component. This is the state of an ImmFilter, where the modes are maneuver regimes or sensor
 * states; a grid of Kalman filters, whose cells can't be revived as modes can, is a
 * GridGaussianParams (decisions.md [rao-blackwellized-core]). Unlike a Mixture, every component is
 * labeled by its mode, and there is exactly one per mode.
 *
 * A hybrid is itself a component (decisions.md [component-first]): its log-density is the log-sum-exp over
 * its modes, and several hybrids on one mode space collapse into one whose mode probabilities are
 * their weighted sums and whose conditional for mode \f$ m \f$ is the collapse of theirs with
 * weights \f$ w_c P_c(m) \f$, which keeps each mode's mean and covariance. So a Mixture of
 * hybrids is a mixture of IMM states [filters-and-recursions].
 *
 * The components come from the allocator (see \ref Allocation). The modes are a
 * CategoricalParams, one weight per value of the mode space, which need not sum to one, as a
 * mixture's weights need not: the state is \f$ p(x, m) \propto w_m \, p_m(x) \f$, and
 * toDensity() checks the weights and factorizes the components before the density is read. The
 * moments and the collapses read the components, so they are the density's: they mean something
 * only once every component is a distribution (decisions.md [params-and-density]).
 *
 * \code
 * auto x = HybridGaussianParams<State, Mode>{CategoricalParams<Mode>{}, initial};  // every mode starts at initial
 * x.modes().toDensity()->probability<Turn>();   // probability of turning, any sensor state
 * x.conditional<Turn, Inlier>();          // the GaussianParams given that mode
 * x.marginal<Position>();                  // each mode's log-weight and position, read in place
 * auto const density = x.toDensity();
 * density->collapsed();                    // one GaussianParams, moment-matched over the modes
 * density->collapse<Maneuver>();           // HybridGaussianParams<State, Maneuver>, grouped by maneuver
 * \endcode
 *
 * \tparam Component_  the component given a mode (see Component), which sets the continuous
 *                     state space and the scalar
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Allocator_  allocator for the components (see \ref Allocation)
 *
 * \sa class Hybrid, HybridGaussianParams, class MixtureParams
 */
template <internal::ComponentOrInfoForm Component_, DiscreteSpace ModeSpace_, typename Allocator_ = DefaultAllocator>
class HybridParams
{
    static_assert(
        Component<Component_>,
        "a Gaussian in an information form is not a hybrid's conditional, having no collapse; convert each "
        "with toGaussian() or toSqrt() and hold a Gaussian or a SqrtGaussian (decisions.md [gaussian-forms])"
    );

public:

    using ComponentType = Params<Component_>;                        ///< the component given one mode
    using Scalar = typename ComponentType::Scalar;                   ///< the scalar type
    using SpaceType = typename ComponentType::SpaceType;             ///< the continuous state space
    using ModeSpaceType = ModeSpace_;                                ///< the discrete mode space
    using ModesType = CategoricalParams<ModeSpace_, Scalar>;         ///< the weights of the modes
    using MomentsType = typename ComponentType::MomentsType;         ///< a mean and covariance together
    using allocator_type = Allocator_;                               ///< the allocator the components come from
    using DensityType = Hybrid<Component_, ModeSpace_, Allocator_>;  ///< the view toDensity() returns

    /** The components, one per mode in flat-index order. */
    using ConditionalsType = internal::Buffer<ComponentType, Allocator_>;

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ComponentType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ComponentType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ComponentType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ComponentType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = true;  ///< its own parameters, which views check

    /** Mode weights \a modes, with \a conditional as the component in every mode. */
    HybridParams(ModesType modes, ComponentType const & conditional, Allocator_ const & allocator = {})
        : m_modes{std::move(modes)}
        , m_conditionals(std::size_t(m_modes.cardinality()), conditional, ComponentAllocator(allocator))
    {}

    /** Mode weights \a modes, and \a conditionals, one component per mode in flat-index order. */
    HybridParams(ModesType modes, ConditionalsType conditionals)
        : m_modes{std::move(modes)}
        , m_conditionals{std::move(conditionals)}
    {
        check();
    }

    /** As above, with the components copied into storage from \a allocator. */
    template <typename Conditionals>
    HybridParams(ModesType modes, Conditionals const & conditionals, Allocator_ const & allocator)
        : m_modes{std::move(modes)}
        , m_conditionals(conditionals.begin(), conditionals.end(), ComponentAllocator(allocator))
    {
        check();
    }

    /** Copies the modes and components into storage from \a allocator. */
    HybridParams(HybridParams const & other, Allocator_ const & allocator)
        : m_modes{other.m_modes}
        , m_conditionals{other.m_conditionals, ComponentAllocator(allocator)}
    {}

    /** Moves the modes and components into storage from \a allocator. */
    HybridParams(HybridParams && other, Allocator_ const & allocator)
        : m_modes{std::move(other.m_modes)}
        , m_conditionals{std::move(other.m_conditionals), ComponentAllocator(allocator)}
    {
        other.m_stamp.touchShape();
    }

    HybridParams(HybridParams const &) = default;      ///< copies the modes and components
    HybridParams(HybridParams &&) noexcept = default;  ///< moves the modes and components

    /** Copies the modes and components, reusing the storage held. */
    HybridParams &
    operator=(HybridParams const &) = default;

    /** Moves the modes and components. */
    HybridParams &
    operator=(HybridParams &&) noexcept = default;

    ~HybridParams() = default;

    /** \returns the allocator the components come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_conditionals.get_allocator());
    }

    /** \returns the continuous state space */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return m_conditionals.front().space();
    }

    /** \returns the discrete mode space */
    [[nodiscard]] ModeSpace_ const &
    modeSpace() const
    {
        return m_modes.space();
    }

    /** \returns the number of modes */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return m_modes.cardinality();
    }

    /** \returns the number of coordinates of every component's mean (see \ref Spaces) */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return m_conditionals.front().valueDimension();
    }

    /** \returns the number of rows and columns of every component's covariance (see \ref Spaces) */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return m_conditionals.front().differenceDimension();
    }

    /** \returns the weights of the modes, proportional to \f$ P(m) \f$ */
    [[nodiscard]] ModesType const &
    modes() const
    {
        return m_modes;
    }

    /** Replaces the mode weights. */
    void
    setModes(ModesType modes)
    {
        EMRYS_FILTER_ASSERT(modes.space() == modeSpace() && "the modes are on a different space");
        m_stamp.touchValues();
        m_modes = std::move(modes);
    }

    /** \returns the component given the mode with flat index \a mode, \f$ p(x \mid m) \f$ */
    [[nodiscard]] ComponentType const &
    conditional(Eigen::Index mode) const &
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < size() && "the mode is out of range");
        return m_conditionals[std::size_t(mode)];
    }

    /** A component of a temporary would outlive it; copy it from a named hybrid instead. */
    void
    conditional(Eigen::Index mode) const && = delete;

    /** Replaces the component given the mode with flat index \a mode with \a component, which
     * must be in the hybrid's space, copying into the storage the component holds.
     */
    void
    setConditional(Eigen::Index mode, ComponentType const & component)
    {
        conditionalToSet(mode, component) = component;
    }

    /** As above, moving \a component in. */
    void
    setConditional(Eigen::Index mode, ComponentType && component)
    {
        conditionalToSet(mode, component) = std::move(component);
    }

    /** \returns \a f called with the component given the mode with flat index \a mode, to change
     * it in place, as a filter's step does; \a f must keep its space, which is asserted after
     */
    template <typename F>
        requires std::invocable<F &, ComponentType &>
    decltype(auto)
    updateConditional(Eigen::Index mode, F && f)
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < size() && "the mode is out of range");
        m_stamp.touchValues();
        return internal::keepingSpace(m_conditionals[std::size_t(mode)], f);
    }

    /** \returns the component given the mode with the named values \a Values and the indices
     * \a indices of axes without names, one per axis (see valueOf() and at()):
     * <tt>conditional<Turn>(at<Cell>(3))</tt>
     */
    template <typename... Values, typename... Paths, std::size_t... N>
        requires(sizeof...(Values) + sizeof...(Paths) > 0)
    [[nodiscard]] ComponentType const &
    conditional(AxisIndex<Paths, N> const &... indices) const &
    {
        return conditional(valueOf<Values...>(modeSpace(), indices...));
    }

    /** A component of a temporary would outlive it. */
    template <typename... Values, typename... Paths, std::size_t... N>
        requires(sizeof...(Values) + sizeof...(Paths) > 0)
    void
    conditional(AxisIndex<Paths, N> const &...) const && = delete;

    /** Exchanges the components with \a conditionals, one per mode on the same space, which must
     * come from an equal allocator: how a filter replaces them without copying.
     */
    void
    swapConditionals(ConditionalsType & conditionals)
    {
        EMRYS_FILTER_ASSERT(Eigen::Index(conditionals.size()) == size() && "give one component per mode");
        EMRYS_FILTER_ASSERT(
            conditionals.get_allocator() == m_conditionals.get_allocator()
            && "the components must come from an equal allocator"
        );
        m_stamp.touchShape();
        m_conditionals.swap(conditionals);
    }

    /** \returns a read-only view of the subspace named by the tag path \a Path (see idx()) of the
     * continuous state, whose space is that subspace with its parameters: a range over the modes
     * that gives each one's log-weight and its component's marginal in turn, copying nothing
     * (decisions.md [element-marginals]). It checks this object's shape; copy() gives a Hybrid of the marginals.
     *
     * \sa class MixtureMarginal
     */
    template <typename Path>
    [[nodiscard]] auto
    marginal() const &
    {
        return internal::tagMarginal<MixtureMarginal, Path>(*this);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal() const && = delete;

    /** marginal() of one repetition of a repeated block; see GaussianParams::marginal(element). */
    template <typename Path>
    [[nodiscard]] auto
    marginal(Eigen::Index element) const &
    {
        return internal::elementMarginal<MixtureMarginal, Path>(*this, element);
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal(Eigen::Index element) const && = delete;

    /** \returns the state as an unlabeled Mixture, one component per mode, weighted by the mode
     * weights, for its toDensity(), marginals, and sampling. It has this state's allocator.
     */
    [[nodiscard]] MixtureParams<Component_, Allocator_>
    toMixture() const
    {
        auto mixture = MixtureParams<Component_, Allocator_>{space(), get_allocator()};
        mixture.reserve(size());
        for (Eigen::Index m = 0; m < size(); ++m)
        {
            mixture.add(m_modes.logWeight(m), m_conditionals[std::size_t(m)]);
        }
        return mixture;
    }

    /** Checks the mode weights and factorizes every component that has changed, so that the
     * state's log-density can be read (see ComponentTraits).
     *
     * It is \c const because it writes only the components' cached factorizations, so it is not
     * safe on one object from two threads at once until it has succeeded; after that it only
     * reads (see \ref Distributions).
     *
     * \returns nothing on success, the WeightError of mode weights that fail to define a
     * distribution, or the GaussianError of the first component that failed
     */
    [[nodiscard]] std::expected<void, NumericError>
    factorize() const
    {
        if (auto const total = m_modes.logTotalWeight(); !total)
        {
            return std::unexpected{total.error()};
        }
        for (auto const & conditional : m_conditionals)
        {
            if (auto result = conditional.factorize(); !result)
            {
                return std::unexpected{result.error()};
            }
        }
        return {};
    }

    /** factorize(), repairing each covariance that is not positive definite by \a repair (see
     * GaussianParams::factorize(R const &)).
     *
     * Every repair is tried before any is kept, as MixtureParams::factorize(R const &) does, so a
     * component that can't be repaired leaves the state as it was.
     *
     * \returns nothing on success, the WeightError of mode weights that fail to define a
     * distribution, or the GaussianError of the first component that could not be repaired
     */
    template <typename R>
    [[nodiscard]] std::expected<void, NumericError>
    factorize(R const & repair)
    {
        m_stamp.touchValues();
        if (auto const total = m_modes.logTotalWeight(); !total)
        {
            return std::unexpected{total.error()};
        }
        // Every repair is tried on a copy of a component that needs one before any is kept. Repairs
        // are rare, so the copies are too; a component that factorizes is cached for the loop after.
        for (auto const & conditional : m_conditionals)
        {
            if (!conditional.factorize())
            {
                auto trial = ComponentType{conditional};
                if (auto result = trial.factorize(repair); !result)
                {
                    return std::unexpected{result.error()};
                }
            }
        }
        for (auto & conditional : m_conditionals)
        {
            if (auto result = conditional.factorize(repair); !result)
            {
                return std::unexpected{result.error()};
            }
        }
        return {};
    }

    /** Checks the mode weights and factorizes every component that has changed (factorize()).
     *
     * \returns a Hybrid that checks this object, or the NumericError from factorize()
     */
    [[nodiscard]] std::expected<DensityType, NumericError>
    toDensity() const &
    {
        if (auto result = factorize(); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

    /** toDensity(), repairing covariances that are not positive definite by \a repair (see
     * factorize(R const &)): <tt>x.toDensity(roots::ridge())</tt>.
     *
     * \returns a Hybrid that checks this object, or the NumericError from factorize(R const &)
     */
    template <typename R>
    [[nodiscard]] std::expected<DensityType, NumericError>
    toDensity(R const & repair) &
    {
        if (auto result = factorize(repair); !result)
        {
            return std::unexpected{result.error()};
        }
        return DensityType{*this};
    }

    /** A density of a temporary would outlive it. */
    template <typename R>
    void
    toDensity(R const &) && = delete;

private:

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    /** A marginal checks the stamp. */
    template <typename>
    friend class internal::ShapeCheck;

    using ComponentAllocator = internal::ReBound<Allocator_, ComponentType>;  ///< allocates the components

    /** \returns the component of \a mode, for setConditional() to assign \a component to, after
     * checking the mode and that \a component is in the hybrid's space
     */
    [[nodiscard]] ComponentType &
    conditionalToSet(Eigen::Index mode, [[maybe_unused]] ComponentType const & component)
    {
        EMRYS_FILTER_ASSERT(mode >= 0 && mode < size() && "the mode is out of range");
        EMRYS_FILTER_ASSERT(component.space() == space() && "the component must be in the hybrid's space");
        m_stamp.touchValues();
        return m_conditionals[std::size_t(mode)];
    }

    /** A programmer error unless there is one component per mode, all on one space. */
    void
    check() const
    {
        EMRYS_FILTER_ASSERT(
            Eigen::Index(m_conditionals.size()) == m_modes.cardinality() && "give one component per mode"
        );
        for (auto const & conditional : m_conditionals)
        {
            EMRYS_FILTER_ASSERT(
                conditional.space() == m_conditionals.front().space()
                && "every mode's component must be on the same space"
            );
        }
    }

    ModesType m_modes;                ///< the weights of the modes
    ConditionalsType m_conditionals;  ///< the component given each mode, in flat-index order
    internal::Stamp m_stamp;          ///< counts changes, for the views' checks [view-stamp]
};

/** \ingroup Distributions
 * \brief The density of a hybrid state, a checked view of its HybridParams (decisions.md
 * [params-and-density], [view-stamp]): \f$ p(x, m) = P(m) \, p_m(x) \f$, with
 * \f$ P(m) = w_m / \sum_n w_n \f$.
 *
 * HybridParams::toDensity() makes one after checking the mode weights and factorizing every
 * component, so the log-density, the moments, and the collapses below read components that are
 * distributions. A hybrid is a component (see Component) by this name, as a %Gaussian is by
 * Gaussian: <tt>MixtureParams<HybridGaussian<State, Mode>></tt> stores HybridGaussianParams.
 *
 * \code
 * auto const density = x.toDensity();       // x a HybridGaussianParams<State, Mode>
 * density->logDensity(value);               // log p(value), summed over the modes
 * density->moments();                       // the mean and covariance over every mode
 * density->collapsed();                     // one GaussianParams, moment-matched over the modes
 * density->collapse<Maneuver>();            // HybridGaussianParams<State, Maneuver>, grouped by maneuver
 * \endcode
 *
 * \tparam Component_  the component given a mode (see Component)
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Allocator_  the parameters' allocator (see \ref Allocation)
 *
 * \sa class HybridParams, HybridGaussian
 */
template <internal::ComponentOrInfoForm Component_, DiscreteSpace ModeSpace_, typename Allocator_>
class Hybrid
{
public:

    using ParamsType = HybridParams<Component_, ModeSpace_, Allocator_>;  ///< the parameters it reads
    using ComponentType = typename ParamsType::ComponentType;             ///< the component given one mode
    using Scalar = typename ParamsType::Scalar;                           ///< the scalar type
    using SpaceType = typename ParamsType::SpaceType;                     ///< the continuous state space
    using ModeSpaceType = ModeSpace_;                                     ///< the discrete mode space
    using ModesType = typename ParamsType::ModesType;                     ///< the weights of the modes
    using MomentsType = typename ParamsType::MomentsType;                 ///< a mean and covariance together
    using allocator_type = Allocator_;                                    ///< the parameters' allocator

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = ParamsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = ParamsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = ParamsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = ParamsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr bool IsParams = false;  ///< a view: it checks its parameters

    /** \returns the continuous state space */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return source().space();
    }

    /** \returns the discrete mode space */
    [[nodiscard]] ModeSpace_ const &
    modeSpace() const
    {
        return source().modeSpace();
    }

    /** \returns the number of modes */
    [[nodiscard]] Eigen::Index
    size() const
    {
        return source().size();
    }

    /** \returns the number of coordinates of every component's mean (see \ref Spaces) */
    [[nodiscard]] Eigen::Index
    valueDimension() const
    {
        return source().valueDimension();
    }

    /** \returns the number of rows and columns of every component's covariance (see \ref Spaces) */
    [[nodiscard]] Eigen::Index
    differenceDimension() const
    {
        return source().differenceDimension();
    }

    /** \returns the parameters' allocator */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return source().get_allocator();
    }

    /** \returns the weights of the modes, which define a distribution */
    [[nodiscard]] ModesType const &
    modes() const
    {
        return source().modes();
    }

    /** \returns the component given the mode with flat index \a mode, factorized */
    [[nodiscard]] ComponentType const &
    conditional(Eigen::Index mode) const
    {
        return source().conditional(mode);
    }

    /** \returns the log of the density at \a x, \f$ \log \sum_m P(m) \, p_m(x) \f$, over the
     * modes: read from the components' factorizations under this view's check (see ComponentTraits)
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        return ComponentTraits<ParamsType>::logDensity(source(), x);
    }

    /** \returns the density at \a x, the exponential of logDensity() */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** \returns the mean and covariance of the state, over every mode: a tangent-space mean, and
     * the modes' covariances plus the spread of their means
     */
    [[nodiscard]] MomentsType
    moments() const
    {
        return ComponentTraits<ParamsType>::moments(source());
    }

    /** \returns the state as one component, moment-matched over the modes (see
     * ComponentTraits)
     */
    [[nodiscard]] ComponentType
    collapsed() const
    {
        return ComponentTraits<ParamsType>::collapseModes(source());
    }

    /** \returns the hybrid state over the mode axis named by the path \a P (a tag or a
     * TagPath) alone, as parameters: its marginal weights, and for each of its values, the
     * components of the modes with that value moment-matched into one. This is collapse grouped by
     * mode. It has the parameters' allocator.
     */
    template <typename P>
    [[nodiscard]] auto
    collapse() const
    {
        ParamsType const & hybrid = source();
        auto const [stride, axis] = internal::locateAxisByPath<P>(modeSpace());
        using Axis = std::remove_cvref_t<decltype(*axis)>;
        using Result = HybridParams<Component_, Axis, Allocator_>;
        using LogWeights = typename ModesType::ArrayType;
        Eigen::Index const count = axis->cardinality();
        typename Result::ConditionalsType grouped(internal::ReBound<Allocator_, ComponentType>(hybrid.get_allocator()));
        grouped.reserve(std::size_t(count));
        auto const componentAt = [&](Eigen::Index m) -> ComponentType const & {
            return hybrid.conditional(m);
        };
        for (Eigen::Index value = 0; value < count; ++value)
        {
            LogWeights logWeights = hybrid.modes().logWeights();
            for (Eigen::Index m = 0; m < size(); ++m)
            {
                if ((m / stride) % count != value)
                {
                    logWeights(m) = -Eigen::NumTraits<Scalar>::infinity();
                }
            }
            // A value of probability zero keeps the components' plain average, which is never used.
            if (!internal::logTotalWeight<Scalar>(logWeights))
            {
                logWeights = LogWeights::Constant(size(), -Eigen::NumTraits<Scalar>::infinity());
                for (Eigen::Index m = 0; m < size(); ++m)
                {
                    if ((m / stride) % count == value)
                    {
                        logWeights(m) = Scalar(0);
                    }
                }
            }
            grouped.push_back(ComponentTraits<ComponentType>::collapse(size(), componentAt, [&](Eigen::Index m) {
                return logWeights(m);
            }));
        }
        return Result{hybrid.modes().template marginal<P>(), std::move(grouped)};
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose components are factorized, and records its stamp; only
     * HybridParams::toDensity() calls this.
     */
    explicit Hybrid(ParamsType const & source)
        : m_check{source}
    {}

    /** \returns the source; a programmer error if it has changed since toDensity() */
    [[nodiscard]] ParamsType const &
    source() const
    {
        return m_check.get();
    }

    internal::ValuesCheck<ParamsType> m_check;  ///< the check of the viewed parameters
};

/// \cond DEDUCTION_GUIDES
template <typename Scalar, typename ModeSpace, typename C>
    requires Component<Density<C>> && std::same_as<Scalar, typename C::Scalar>
HybridParams(CategoricalParams<ModeSpace, Scalar>, C) -> HybridParams<Density<C>, ModeSpace>;

template <typename Scalar, typename ModeSpace, typename C, typename Allocator>
    requires Component<Density<C>> && std::same_as<Scalar, typename C::Scalar>
HybridParams(CategoricalParams<ModeSpace, Scalar>, C, Allocator) -> HybridParams<Density<C>, ModeSpace, Allocator>;

template <typename Scalar, typename ModeSpace, typename C, typename ComponentAllocator>
    requires Component<Density<C>> && std::same_as<Scalar, typename C::Scalar>
HybridParams(CategoricalParams<ModeSpace, Scalar>, std::vector<C, ComponentAllocator>)
    -> HybridParams<Density<C>, ModeSpace, internal::ReBound<ComponentAllocator, std::byte>>;
/// \endcond

/** \ingroup Distributions
 * \brief The Hybrid of Gaussians in covariance form, the spelling for the common case (decisions.md
 * [component-first]): \f$ p(x, m) = P(m) \, \mathcal{N}(x; \mu_m, \Sigma_m) \f$, the density of an
 * ImmFilter's state, and the name a mixture of hybrids takes as its component.
 *
 * \tparam Space_      the continuous state space (see \ref Spaces)
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the Gaussians (see \ref Allocation)
 */
template <Space Space_, DiscreteSpace ModeSpace_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
using HybridGaussian = Hybrid<Gaussian<Space_, Scalar_>, ModeSpace_, Allocator_>;

/** \ingroup Distributions
 * \brief The HybridParams of Gaussians in covariance form: the parameters of a HybridGaussian, an
 * ImmFilter's state.
 *
 * \tparam Space_      the continuous state space (see \ref Spaces)
 * \tparam ModeSpace_  the discrete mode space (see \ref Discrete)
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the Gaussians (see \ref Allocation)
 */
template <Space Space_, DiscreteSpace ModeSpace_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
using HybridGaussianParams = HybridParams<Gaussian<Space_, Scalar_>, ModeSpace_, Allocator_>;

/** ComponentTraits of a hybrid state, which make it a component (decisions.md [component-first]). */
template <Component Component_, DiscreteSpace ModeSpace_, typename Allocator_>
struct ComponentTraits<HybridParams<Component_, ModeSpace_, Allocator_>>
{
    using Type = HybridParams<Component_, ModeSpace_, Allocator_>;  ///< the component
    using Scalar = typename Component_::Scalar;                     ///< the scalar type

    /** A marginal of one hybrid in a mixture, a range over its modes, which borrows the selection
     * the mixture's marginal resolved (an internal::SharedSelection) as \a Selection.
     */
    template <typename Selection, typename SubSpace>
    using MarginalType = MixtureMarginal<Type, Selection, SubSpace>;

    /** \returns the log-density of \a hybrid at \a x, \f$ \log \sum_m P(m) \, p_m(x) \f$ with
     * \f$ P(m) = w_m / \sum_n w_n \f$, from its components' factorizations and its mode weights,
     * which must have been checked (see HybridParams::factorize())
     */
    template <typename Derived>
    [[nodiscard]] static Scalar
    logDensity(Type const & hybrid, Eigen::MatrixBase<Derived> const & x)
    {
        auto sum = LogSumExp<Scalar>{};
        auto total = LogSumExp<Scalar>{};
        for (Eigen::Index m = 0; m < hybrid.size(); ++m)
        {
            Scalar const logWeight = hybrid.modes().logWeight(m);
            if (logWeight == -Eigen::NumTraits<Scalar>::infinity())
            {
                continue;
            }
            total.add(logWeight);
            sum.add(logWeight + ComponentTraits<Params<Component_>>::logDensity(hybrid.conditional(m), x));
        }
        return sum.value() - total.value();
    }

    /** \returns the collapse of \a count hybrids on one mode space, \a componentAt(i) with log-weight
     * \a logWeightAt(i), not necessarily normalized: the mode probabilities
     * \f$ P(m) \propto \sum_c w_c P_c(m) \f$, normalized, and for each mode the collapse of the hybrids'
     * components in that mode with weights \f$ w_c P_c(m) \f$, which keeps the mode probabilities
     * and each mode's mean and covariance of \f$ \sum_c w_c P_c(m) \, p_{c,m}(x) \f$. A mode no
     * hybrid gives probability keeps
     * the first hybrid's component, which is never weighted. The result's components come from the
     * first hybrid's allocator.
     */
    template <typename ComponentAt, typename LogWeightAt>
    [[nodiscard]] static Type
    collapse(Eigen::Index count, ComponentAt componentAt, LogWeightAt logWeightAt)
    {
        EMRYS_FILTER_ASSERT(count > 0 && "a collapse needs a component");
        Type const & first = componentAt(0);
        Eigen::Index const modes = first.size();
        auto logModes = typename Type::ModesType::ArrayType{};
        logModes.resize(modes);
        for (Eigen::Index m = 0; m < modes; ++m)
        {
            auto sum = LogSumExp<Scalar>{};
            for (Eigen::Index c = 0; c < count; ++c)
            {
                EMRYS_FILTER_ASSERT(
                    componentAt(c).modeSpace() == first.modeSpace() && "every hybrid must be on one mode space"
                );
                sum.add(Scalar(logWeightAt(c)) + modeLogProbability(componentAt(c), m));
            }
            logModes(m) = sum.value();
        }
        typename Type::ConditionalsType conditionals(
            internal::ReBound<Allocator_, Params<Component_>>(first.get_allocator())
        );
        conditionals.reserve(std::size_t(modes));
        for (Eigen::Index m = 0; m < modes; ++m)
        {
            if (logModes(m) == -Eigen::NumTraits<Scalar>::infinity())
            {
                conditionals.push_back(first.conditional(m));
                continue;
            }
            conditionals.push_back(
                ComponentTraits<Params<Component_>>::collapse(
                    count,
                    [&](Eigen::Index c) -> Params<Component_> const & {
                        return componentAt(c).conditional(m);
                    },
                    [&](Eigen::Index c) {
                        return Scalar(logWeightAt(c)) + modeLogProbability(componentAt(c), m);
                    }
                )
            );
        }
        auto collapsedModes = typename Type::ModesType{first.modeSpace(), logModes};
        // Weights that fail to normalize are kept as they are, for the next factorize() to report.
        static_cast<void>(collapsedModes.normalize());
        return Type{std::move(collapsedModes), std::move(conditionals)};
    }

    /** \returns \a hybrid as one component, moment-matched over its modes, whose components and
     * mode weights must have been checked (see Hybrid::collapsed())
     */
    [[nodiscard]] static Params<Component_>
    collapseModes(Type const & hybrid)
    {
        return ComponentTraits<Params<Component_>>::collapse(
            hybrid.size(),
            [&](Eigen::Index m) -> Params<Component_> const & {
                return hybrid.conditional(m);
            },
            [&](Eigen::Index m) {
                return hybrid.modes().logWeight(m);
            }
        );
    }

    /** \returns the mean and covariance of \a hybrid over every mode, the moments of
     * collapseModes(), under the same condition (see Hybrid::moments())
     */
    [[nodiscard]] static typename Type::MomentsType
    moments(Type const & hybrid)
    {
        return ComponentTraits<Params<Component_>>::moments(collapseModes(hybrid));
    }

    /** \returns the log-probability of mode \a m of \a hybrid, its log-weight less the log of
     * their total. The weights must define a distribution, as the collapse of the components
     * weighted by these probabilities needs.
     */
    [[nodiscard]] static Scalar
    modeLogProbability(Type const & hybrid, Eigen::Index m)
    {
        auto const total = hybrid.modes().logTotalWeight();
        EMRYS_FILTER_ASSERT(total && "a hybrid's mode weights must define a distribution to collapse it");
        return Scalar(hybrid.modes().logWeight(m) - *total);
    }
};

namespace internal
{

/** ElementsOf of a hybrid state, whose components are its modes' (decisions.md [element-marginals]). */
template <Component Component_, DiscreteSpace ModeSpace_, typename Allocator_>
struct ElementsOf<HybridParams<Component_, ModeSpace_, Allocator_>>
{
    using Type = HybridParams<Component_, ModeSpace_, Allocator_>;  ///< the hybrid

    /** The hybrid of components whose parameters are \a ComponentParams, which copy() makes. */
    template <typename ComponentParams>
    using CopyType = HybridParams<Density<ComponentParams>, ModeSpace_, Allocator_>;

    /** \returns the log-weight of mode \a m of \a hybrid */
    [[nodiscard]] static auto
    logWeight(Type const & hybrid, Eigen::Index m)
    {
        return hybrid.modes().logWeight(m);
    }

    /** \returns the component of mode \a m of \a hybrid */
    [[nodiscard]] static auto const &
    component(Type const & hybrid, Eigen::Index m)
    {
        return hybrid.conditional(m);
    }

    /** \returns a hybrid with the mode weights of \a hybrid and the parameters of each of
     * \a marginal's components, in storage from \a allocator
     */
    template <typename Marginal>
    [[nodiscard]] static auto
    copy(Type const & hybrid, Marginal const & marginal, Allocator_ const & allocator)
    {
        using Result = CopyType<typename Marginal::ComponentMarginalType::ParamsType>;
        auto conditionals =
            typename Result::ConditionalsType(ReBound<Allocator_, typename Result::ComponentType>(allocator));
        conditionals.reserve(std::size_t(marginal.size()));
        for (auto const & element : marginal)
        {
            conditionals.push_back(element.component.copy());
        }
        return Result{hybrid.modes(), std::move(conditionals)};
    }
};

}  // namespace internal

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a hybrid: its mode count and dimension, the mean and covariance over every mode,
 * and each mode's component with the mode's probability; or, if the weights fail to define a
 * distribution, the error in place of the moments and each mode's log-weight.
 */
template <typename Component_, typename ModeSpace_, typename Allocator_>
void
describe(std::ostream & stream, HybridParams<Component_, ModeSpace_, Allocator_> const & hybrid, Indent indent)
{
    stream << indent << "hybrid of ";
    printCount(stream, hybrid.size(), "mode");
    printDimension(stream, hybrid.valueDimension(), hybrid.differenceDimension());
    auto const total = hybrid.modes().logTotalWeight();
    if (total)
    {
        printMoments(
            stream,
            indent.deeper(),
            ComponentTraits<HybridParams<Component_, ModeSpace_, Allocator_>>::moments(hybrid)
        );
    }
    else
    {
        stream << '\n';
        describe(stream, total.error(), indent.deeper());
    }
    for (Eigen::Index mode = 0; mode < hybrid.size(); ++mode)
    {
        auto const logWeight = hybrid.modes().logWeight(mode);
        stream << '\n' << indent.deeper() << "mode " << mode;
        if (total)
        {
            stream << ", probability " << Eigen::numext::exp(logWeight - *total) << ":\n";
        }
        else
        {
            stream << ", log-weight " << logWeight << ":\n";
        }
        describe(stream, hybrid.conditional(mode), indent.deeper().deeper());
    }
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

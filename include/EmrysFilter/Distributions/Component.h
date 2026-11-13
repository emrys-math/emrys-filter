#pragma once

/** \file
 * What a container of several Gaussians needs of each one (decisions.md [component-first]): the Component
 * concept, ComponentTraits with the log-density read without a check, the weighted
 * collapse of several components into one, and the type of one component's marginal, and their
 * specializations for Gaussian and for the square-root form, SqrtGaussian. Mixture.h and
 * Hybrid.h hold their components through this, and Hybrid.h specializes it for Hybrid, so that a
 * hybrid state is a component too. Read Mixture.h next.
 */

#include <concepts>
#include <type_traits>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Triangularization.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/Statistics.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** \returns \a f called with \a component, to change it in place, after which its space must be
 * the one it had: how a container hands out a component it holds without letting its space change.
 */
template <typename C, typename F>
decltype(auto)
keepingSpace(C & component, F & f)
{
    [[maybe_unused]] auto const before = typename C::SpaceType{component.space()};
    if constexpr (std::is_void_v<std::invoke_result_t<F &, C &>>)
    {
        f(component);
        EMRYS_FILTER_ASSERT(component.space() == before && "the component must keep its space");
    }
    else
    {
        std::invoke_result_t<F &, C &> result = f(component);
        EMRYS_FILTER_ASSERT(component.space() == before && "the component must keep its space");
        return result;
    }
}

/** The weights of a collapse of \a count components: their log total, and how many have weight,
 * with the last such one's index, for the collapse that one alone makes.
 */
template <typename Scalar>
struct CollapseWeights
{
    Scalar logTotal;    ///< the log of the total weight
    Eigen::Index kept;  ///< how many components have weight
    Eigen::Index only;  ///< the last component with weight
};

/** \returns the CollapseWeights of the \a count log-weights \a logWeightAt(i), which must be finite
 * and not all zero
 */
template <typename Scalar, typename LogWeightAt>
[[nodiscard]] CollapseWeights<Scalar>
collapseWeights(Eigen::Index count, LogWeightAt const & logWeightAt)
{
    EMRYS_FILTER_ASSERT(count > 0 && "a collapse needs a component");
    auto total = LogSumExp<Scalar>{};
    auto result = CollapseWeights<Scalar>{Scalar(0), 0, 0};
    for (Eigen::Index i = 0; i < count; ++i)
    {
        Scalar const logWeight = logWeightAt(i);
        total.add(logWeight);
        if (logWeight > -Eigen::NumTraits<Scalar>::infinity())
        {
            ++result.kept;
            result.only = i;
        }
    }
    result.logTotal = total.value();
    EMRYS_FILTER_ASSERT(isFinite(result.logTotal) && "the weights must be finite and not all zero");
    return result;
}

}  // namespace internal

/** \ingroup Distributions
 * What a container does with a component beyond the component's own members (decisions.md [component-first]):
 * \c logDensity(component, x), the log-density read from the component's factorization without
 * a check of its own, for a container's density view that already holds one [execution-policies];
 * \c collapse(count, componentAt, logWeightAt), the collapse of several components into one by
 * moment matching; \c moments(component), its mean and covariance, read without a check, for a
 * container's density view that has checked it, as a hybrid's needs no member for them; and
 * \c MarginalType, the view of a subset of one component's coordinates that a
 * marginal of the container gives per component [element-marginals]. Specialized for every component type, by
 * the parameters (\c ComponentTraits<GaussianParams<Space>>). The primary template is undefined, so a type without a
 * specialization is not a Component.
 */
template <typename C>
struct ComponentTraits;

/** \ingroup Distributions
 * Whether \a C is a %Gaussian in covariance form: the component for which a container offers the
 * conveniences that take or give moments [component-first].
 */
template <typename C>
inline constexpr bool isGaussian = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <Space Space_, RealScalar Scalar_>
inline constexpr bool isGaussian<GaussianParams<Space_, Scalar_>>{true};

/// \endcond

/** ComponentTraits of a %Gaussian in covariance form. */
template <Space Space_, RealScalar Scalar_>
struct ComponentTraits<GaussianParams<Space_, Scalar_>>
{
    using Type = GaussianParams<Space_, Scalar_>;  ///< the component

    /** A marginal of one component, which borrows the selection the marginal of its container
     * resolved (an internal::SharedSelection) as \a Selection.
     */
    template <typename Selection, typename SubSpace>
    using MarginalType = GaussianMarginal<Type, Selection, SubSpace>;

    /** \returns the log-density of \a gaussian at \a x, from its factorization, which must be valid
     * (see GaussianParams::factorize())
     */
    template <typename Derived>
    [[nodiscard]] static Scalar_
    logDensity(Type const & gaussian, Eigen::MatrixBase<Derived> const & x)
    {
        return internal::FactorizedGaussian::logDensity(gaussian, x);
    }

    /** \returns the moment-matched %Gaussian of \a count Gaussians, \a componentAt(i) with log-weight
     * \a logWeightAt(i), not necessarily normalized: a tangent-space mean, and the weighted
     * covariances plus the spread of the means. Gaussians of zero weight are left out, and when one
     * alone has weight the result is that %Gaussian. It allocates nothing sized by the count.
     *
     * The mixture-to-%Gaussian collapse of the IMM's output and mixing steps
     * (\ref Blom1988 "Blom and Bar-Shalom (1988)", as given in
     * \ref Hashimzade2024 "Hashimzade et al. (2024)", equation (12)), by internal::mixtureMomentsIn().
     */
    template <typename ComponentAt, typename LogWeightAt>
    [[nodiscard]] static Type
    collapse(Eigen::Index count, ComponentAt componentAt, LogWeightAt logWeightAt)
    {
        auto const weights = internal::collapseWeights<Scalar_>(count, logWeightAt);
        if (weights.kept == 1)
        {
            return Type{componentAt(weights.only)};
        }
        Type const & first = componentAt(0);
        return Type{
            first.space(),
            mixtureMomentsIn<typename Type::MomentsType>(
                first.space(),
                count,
                [&](Eigen::Index i) -> decltype(auto) {
                    return componentAt(i).mean();
                },
                [&](Eigen::Index i) -> decltype(auto) {
                    return componentAt(i).covarianceLower();
                },
                logWeightAt,
                weights.logTotal
            )
        };
    }

    /** \returns the mean and covariance \a gaussian holds */
    [[nodiscard]] static typename Type::MomentsType const &
    moments(Type const & gaussian)
    {
        return gaussian.moments();
    }
};

/** ComponentTraits of a %Gaussian in square-root form (decisions.md [component-first], [gaussian-forms]). */
template <Space Space_, RealScalar Scalar_>
struct ComponentTraits<SqrtGaussianParams<Space_, Scalar_>>
{
    using Type = SqrtGaussianParams<Space_, Scalar_>;  ///< the component

    /** A marginal of one component, which borrows the selection the marginal of its container
     * resolved (an internal::SharedSelection) as \a Selection: parameters, since its root is
     * computed.
     */
    template <typename Selection, typename SubSpace>
    using MarginalType = internal::SqrtComponentMarginal<Type, Selection, SubSpace>;

    /** \returns the log-density of \a gaussian at \a x, from its root and the log-determinant
     * factorize() cached, which must be current (see SqrtGaussianParams::factorize())
     */
    template <typename Derived>
    [[nodiscard]] static Scalar_
    logDensity(Type const & gaussian, Eigen::MatrixBase<Derived> const & x)
    {
        typename Type::DifferenceVectorType residual = Emrys::minus(gaussian.space(), x, gaussian.mean());
        gaussian.whitenInPlace(residual);
        Scalar_ const dimension = internal::scalarFromCount<Scalar_>(gaussian.differenceDimension());
        return Scalar_(-0.5) * (dimension * log2Pi<Scalar_>() + gaussian.m_logDeterminant + residual.squaredNorm());
    }

    /** \returns the moment-matched %Gaussian of \a count square-root forms, \a componentAt(i) with
     * log-weight \a logWeightAt(i), not necessarily normalized, in square-root form: the
     * tangent-space mean the covariance form's collapse takes, and the lower-triangular root of
     * \f$ \sum_i \hat w_i (S_i S_i^T + d_i d_i^T) \f$, with \f$ d_i = \mu_i \boxminus \mu \f$, by
     * triangularization of the stacked \f$ \sqrt{\hat w_i} S_i \f$ and \f$ \sqrt{\hat w_i} d_i \f$,
     * so that no covariance is formed. The stack is folded in one component at a time, beside the
     * root so far, so it allocates nothing sized by the count. Components of zero weight are left
     * out, and when one alone has weight the result is that component.
     *
     * The covariance is the covariance form's collapse
     * (\ref Hashimzade2024 "Hashimzade et al. (2024)", equation (12)) where differences are plain;
     * elsewhere the spread of the means is taken about the mean itself.
     */
    template <typename ComponentAt, typename LogWeightAt>
    [[nodiscard]] static Type
    collapse(Eigen::Index count, ComponentAt componentAt, LogWeightAt logWeightAt)
    {
        auto const weights = internal::collapseWeights<Scalar_>(count, logWeightAt);
        if (weights.kept == 1)
        {
            return Type{componentAt(weights.only)};
        }
        Type const & first = componentAt(0);
        auto const & space = first.space();
        typename Type::VectorType mean = pointMomentsIn<typename Type::MomentsType>(
                                             space,
                                             count,
                                             [&](Eigen::Index i) -> decltype(auto) {
                                                 return componentAt(i).mean();
                                             },
                                             logWeightAt,
                                             weights.logTotal
        )
                                             .mean();
        constexpr int Rows = Type::DifferenceDimensionAtCompileTime;
        constexpr int MaxRows = Type::MaxDifferenceDimensionAtCompileTime;
        using Stack = internal::MatrixOf<
            Scalar_,
            Rows,
            Rows == Eigen::Dynamic ? Eigen::Dynamic : 2 * Rows + 1,
            MaxRows,
            MaxRows == Eigen::Dynamic ? Eigen::Dynamic : 2 * MaxRows + 1>;
        Eigen::Index const n = first.differenceDimension();
        typename Type::MatrixType factor = Type::MatrixType::Zero(n, n);
        auto stack = Stack{n, 2 * n + 1};
        for (Eigen::Index i = 0; i < count; ++i)
        {
            Scalar_ const logWeight = logWeightAt(i);
            if (logWeight == -Eigen::NumTraits<Scalar_>::infinity())
            {
                continue;
            }
            Type const & component = componentAt(i);
            Scalar_ const root = Eigen::numext::exp(Scalar_(0.5) * (logWeight - weights.logTotal));
            stack.leftCols(n) = factor;
            stack.middleCols(n, n) = root * component.root();
            stack.col(2 * n) = root * Emrys::minus(space, component.mean(), mean);
            factor = internal::lowerTriangularRoot(stack);
        }
        return Type{space, std::move(mean), std::move(factor)};
    }

    /** \returns the mean and covariance of \a gaussian, its covariance from its root */
    [[nodiscard]] static typename Type::MomentsType
    moments(Type const & gaussian)
    {
        return gaussian.moments();
    }
};

/** \ingroup Distributions
 * The distribution that a container of several holds one of per element (decisions.md [component-first],
 * [params-and-density]): a Mixture's component, a Hybrid's conditional given a mode. Gaussian is one, and so is Hybrid,
 * so a mixture of hybrid states is a Mixture too. A component is named by the type that evaluates
 * (Gaussian and Hybrid, not GaussianParams and HybridParams), and the container stores its Params.
 *
 * Its parameters have the space and dimension contract of a %Gaussian [sizes-and-dimensions],
 * factorize with factorize(), and have a ComponentTraits specialization for the log-density read
 * from that factorization without a check, for the moments, and for the collapse of several
 * components into one by moment matching.
 *
 * The contract is closed (decisions.md [user-extension-points]): a component is one of the library's,
 * since the traits encode the collapse, which a wrong specialization would corrupt silently, and a
 * type derived from a component isn't one, its Params and Density round trip landing on the base.
 * What a user keeps per component lives beside the mixture instead (MixtureLineage::carry()), and
 * what a user does to a component is a recursion of the user's own (Filters/Recursion.h).
 */
template <typename C>
concept Component = std::same_as<Density<Params<C>>, C> && requires(Params<C> const & c) {
    typename Params<C>::SpaceType;
    typename Params<C>::Scalar;
    typename Params<C>::MomentsType;
    { Params<C>::ValueDimensionAtCompileTime } -> std::convertible_to<int>;
    { Params<C>::MaxValueDimensionAtCompileTime } -> std::convertible_to<int>;
    { Params<C>::DifferenceDimensionAtCompileTime } -> std::convertible_to<int>;
    { Params<C>::MaxDifferenceDimensionAtCompileTime } -> std::convertible_to<int>;
    { c.space() } -> std::same_as<typename Params<C>::SpaceType const &>;
    { c.valueDimension() } -> std::same_as<Eigen::Index>;
    { c.differenceDimension() } -> std::same_as<Eigen::Index>;
    { ComponentTraits<Params<C>>::moments(c) } -> std::convertible_to<typename Params<C>::MomentsType>;
    { c.factorize() };
    typename ComponentTraits<Params<C>>::Type;
};

namespace internal
{

/** Whether \a C is the density of a %Gaussian in an information form, which is not a Component,
 * since it has no collapse: MixtureParams and Hybrid take one only to say so, naming the
 * conversions (decisions.md [gaussian-forms]).
 */
template <typename C>
concept InfoFormDensity = GaussianForm<Params<C>> && std::same_as<Density<Params<C>>, C> && isInfoForm<Params<C>>;

/** What MixtureParams, Mixture, and Hybrid take as their component: a Component, or an
 * InfoFormDensity, which their bodies reject with a message.
 */
template <typename C>
concept ComponentOrInfoForm = Component<C> || InfoFormDensity<C>;

}  // namespace internal

}  // namespace Emrys

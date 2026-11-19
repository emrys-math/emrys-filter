#pragma once

/** \file
 * Lie groups as spaces (docs/design/groups.md; a prototype, not yet a decision): the LieGroup concept, which refines
 * Space, and the free functions that reach a group's members, as plus() and minus() reach a space's: compose(),
 * inverse(), identity(), exp(), log(), adjoint(), rightJacobian(), and leftJacobian(). Linear and Periodic are groups
 * under addition, and a Composite or Repeated of groups is one block by block, each without a member of its own. Read
 * ExtendedPose.h next, for the family of groups of a rotation and the vectors it turns.
 */

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** Whether \a S has a group's members of its own: compose(a, b, out), inverse(a, out), identity(out), exp(d, out),
 * log(a, out), adjoint(a, out), and rightJacobian(d, out), each const, taking one column per operand and writing
 * through \a out as plus() does.
 */
template <typename S>
concept HasGroupMembers = requires(S const & space, Eigen::MatrixXd & m) {
    space.compose(m.col(0), m.col(0), m.col(0));
    space.inverse(m.col(0), m.col(0));
    space.identity(m.col(0));
    space.exp(m.col(0), m.col(0));
    space.log(m.col(0), m.col(0));
    space.adjoint(m.col(0), m.middleRows(0, 0));
    space.rightJacobian(m.col(0), m.middleRows(0, 0));
};

template <typename S>
consteval bool
isLieGroup();

/** Whether every one of the blocks in the tuple type \a Blocks is a group. */
template <typename Blocks>
struct AllLieGroups;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Blocks>
struct AllLieGroups<std::tuple<Blocks...>> : std::bool_constant<(isLieGroup<Blocks>() && ...)>
{};

/// \endcond

/** \returns whether \a S, a Space, is a Lie group: it has the members; or it is a Composite or Repeated of groups; or
 * its values and differences are plain, or it is Periodic, so that it is a group under addition. A space with a plus()
 * of its own and no group members is not one.
 */
template <typename S>
consteval bool
isLieGroup()
{
    if constexpr (HasGroupMembers<S>)
    {
        return true;
    }
    else if constexpr (IsComposite<S>)
    {
        return AllLieGroups<CompositeBlocks<S>>::value;
    }
    else if constexpr (IsRepeated<S>)
    {
        return isLieGroup<RepeatedBlock<S>>();
    }
    else
    {
        return isLinear<S> || IsPeriodic<S>;
    }
}

template <typename S>
consteval bool
isAbelianGroup();

/** Whether every one of the blocks in the tuple type \a Blocks is an abelian group. */
template <typename Blocks>
struct AllAbelian;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Blocks>
struct AllAbelian<std::tuple<Blocks...>> : std::bool_constant<(isAbelianGroup<Blocks>() && ...)>
{};

/// \endcond

/** \returns whether the group \a S is abelian, so that its adjoint is the identity: a group under addition, a
 * Composite or Repeated of such, or a group that declares <tt>static constexpr bool abelian = true;</tt>.
 */
template <typename S>
consteval bool
isAbelianGroup()
{
    if constexpr (HasGroupMembers<S>)
    {
        return requires { requires bool(S::abelian); };
    }
    else if constexpr (IsComposite<S>)
    {
        return AllAbelian<CompositeBlocks<S>>::value;
    }
    else if constexpr (IsRepeated<S>)
    {
        return isAbelianGroup<RepeatedBlock<S>>();
    }
    else
    {
        return true;
    }
}

}  // namespace internal

/** \ingroup Spaces
 * \brief A Space that is a Lie group: two values combine into a third, every value has an inverse, and one value is
 * the identity (docs/design/groups.md).
 *
 * In a group the product determines the space's operations,
 * \f[ x \boxplus \delta = x \cdot \mathrm{Exp}(\delta), \qquad a \boxminus b = \mathrm{Log}(b^{-1} \cdot a), \f]
 * with the increment on the right, so that a difference is a step in the body frame. Linear and Periodic are groups
 * under addition; UnitQuaternion and ExtendedPose have their own products; a Composite or Repeated of groups is one
 * block by block. A space with a plus() of its own that is not a group, such as a unit direction, stays a Space.
 *
 * A group written from scratch has, beside what Space asks for, these const members, each taking one column per
 * operand, in any scalar, and writing through \a out as plus() does:
 *  - <tt>compose(a, b, out)</tt>, the product \f$ a \cdot b \f$;
 *  - <tt>inverse(a, out)</tt> and <tt>identity(out)</tt>;
 *  - <tt>exp(d, out)</tt>, the value \f$ \mathrm{Exp}(d) \f$ of a difference, and <tt>log(a, out)</tt>, its inverse;
 *  - <tt>adjoint(a, out)</tt>, the matrix \f$ \mathrm{Ad}_a \f$ over the differences with
 *    \f$ a \cdot \mathrm{Exp}(d) = \mathrm{Exp}(\mathrm{Ad}_a d) \cdot a \f$;
 *  - <tt>rightJacobian(d, out)</tt>, \f$ J_r(d) \f$ with
 *    \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(d) \cdot \mathrm{Exp}(J_r(d) \epsilon) \f$, which is the
 *    space's reset Jacobian (Reset.h).
 *
 * Its plus() and minus() must be the two formulas above; checkGroup() in Spaces/GroupChecks.h checks the members
 * against each other. The library calls them through compose(), inverse(), identity(), exp(), log(), adjoint(),
 * rightJacobian(), and leftJacobian().
 */
template <typename S>
concept LieGroup = Space<S> && internal::isLieGroup<S>();

/** \ingroup Spaces
 * Whether the adjoint of the group \a S is the identity at every value: true of a group under addition and of a
 * Composite or Repeated of such. Where it holds, the adjoint carry of an update does nothing.
 */
template <LieGroup S>
inline constexpr bool hasIdentityAdjoint = internal::isAbelianGroup<S>();

namespace internal
{

/** Calls \a f(block, valueOffset, valueRows, differenceOffset, differenceRows) for each block of the Composite
 * \a space, or each repetition of the Repeated \a space, in order.
 */
template <typename S, typename F>
void
forEachGroupPart(S const & space, F && f)
{
    if constexpr (IsComposite<S>)
    {
        Eigen::Index values = 0;
        Eigen::Index differences = 0;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((f(space.template block<I>(),
                values,
                valueDimension(space.template block<I>()),
                differences,
                differenceDimension(space.template block<I>())),
              values += valueDimension(space.template block<I>()),
              differences += differenceDimension(space.template block<I>())),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<S>>>{});
    }
    else
    {
        auto const & block = space.block();
        Eigen::Index const v = valueDimension(block);
        Eigen::Index const k = differenceDimension(block);
        for (Eigen::Index i = 0; i < space.count(); ++i)
        {
            f(block, i * v, v, i * k, k);
        }
    }
}

/** Whether the group \a S is one block by block: a Composite or Repeated without group members of its own. */
template <typename S>
concept IsBlockwiseGroup = !HasGroupMembers<S> && (IsComposite<S> || IsRepeated<S>);

/** Writes the product \f$ a \cdot b \f$ in the group \a space through \a out. */
template <typename S, typename A, typename B, typename Out>
void
composeInto(
    S const & space,
    Eigen::MatrixBase<A> const & a,
    Eigen::MatrixBase<B> const & b,
    Eigen::MatrixBase<Out> const & out
)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.compose(a, b, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        forEachGroupPart(space, [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index, Eigen::Index) {
            composeInto(block, a.segment(v, n), b.segment(v, n), o.segment(v, n));
        });
    }
    else
    {
        writable(out) = a.template cast<typename Out::Scalar>() + b.template cast<typename Out::Scalar>();
        normalizeValues(space, out);
    }
}

/** Writes the inverse of \a a in the group \a space through \a out. */
template <typename S, typename A, typename Out>
void
inverseInto(S const & space, Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.inverse(a, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        forEachGroupPart(space, [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index, Eigen::Index) {
            inverseInto(block, a.segment(v, n), o.segment(v, n));
        });
    }
    else
    {
        writable(out) = -a.template cast<typename Out::Scalar>();
        normalizeValues(space, out);
    }
}

/** Writes the identity of the group \a space through \a out. */
template <typename S, typename Out>
void
identityInto(S const & space, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.identity(out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        forEachGroupPart(space, [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index, Eigen::Index) {
            identityInto(block, o.segment(v, n));
        });
    }
    else
    {
        writable(out).setZero();
        normalizeValues(space, out);
    }
}

/** Writes \f$ \mathrm{Exp}(d) \f$ in the group \a space through \a out. */
template <typename S, typename D, typename Out>
void
expInto(S const & space, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.exp(d, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        forEachGroupPart(
            space,
            [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index dv, Eigen::Index dn) {
                expInto(block, d.segment(dv, dn), o.segment(v, n));
            }
        );
    }
    else
    {
        writable(out) = d.template cast<typename Out::Scalar>();
        normalizeValues(space, out);
    }
}

/** Writes \f$ \mathrm{Log}(a) \f$ in the group \a space through \a out. */
template <typename S, typename A, typename Out>
void
logInto(S const & space, Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.log(a, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        forEachGroupPart(
            space,
            [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index dv, Eigen::Index dn) {
                logInto(block, a.segment(v, n), o.segment(dv, dn));
            }
        );
    }
    else
    {
        writable(out) = a.template cast<typename Out::Scalar>();
        normalizeDifferences(space, out);
    }
}

/** Writes the adjoint \f$ \mathrm{Ad}_a \f$ of the group \a space through \a out, a square matrix over its
 * differences.
 */
template <typename S, typename A, typename Out>
void
adjointInto(S const & space, Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.adjoint(a, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        o.setZero();
        forEachGroupPart(
            space,
            [&](auto const & block, Eigen::Index v, Eigen::Index n, Eigen::Index dv, Eigen::Index dn) {
                adjointInto(block, a.segment(v, n), o.block(dv, dv, dn, dn));
            }
        );
    }
    else
    {
        writable(out).setIdentity();
    }
}

/** Writes the right Jacobian \f$ J_r(d) \f$ of the group \a space through \a out, a square matrix over its
 * differences.
 */
template <typename S, typename D, typename Out>
void
rightJacobianInto(S const & space, Eigen::MatrixBase<D> const & d, Eigen::MatrixBase<Out> const & out)
{
    if constexpr (HasGroupMembers<S>)
    {
        space.rightJacobian(d, out);
    }
    else if constexpr (IsBlockwiseGroup<S>)
    {
        auto & o = writable(out);
        o.setZero();
        forEachGroupPart(space, [&](auto const & block, Eigen::Index, Eigen::Index, Eigen::Index dv, Eigen::Index dn) {
            rightJacobianInto(block, d.segment(dv, dn), o.block(dv, dv, dn, dn));
        });
    }
    else
    {
        writable(out).setIdentity();
    }
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the product \f$ a \cdot b \f$ of two values of the group \a space: \a b applied after \a a, in \a a's
 * frame. In Linear it is <tt>a + b</tt>; in UnitQuaternion the quaternion product; in an ExtendedPose the composition
 * of the two rigid motions. A fixed-size group is named instead: <tt>compose<Pose>(a, b)</tt>.
 */
template <LieGroup S, typename A, typename B>
[[nodiscard]] ValueVector<S, typename A::Scalar>
compose(S const & space, Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b)
{
    EMRYS_FILTER_ASSERT(a.size() == valueDimension(space) && b.size() == valueDimension(space));
    auto result = ValueVector<S, typename A::Scalar>{};
    result.resize(valueDimension(space));
    internal::composeInto(space, a, b, result);
    return result;
}

/** \ingroup Spaces
 * \returns the inverse \f$ a^{-1} \f$ of a value of the group \a space
 */
template <LieGroup S, typename A>
[[nodiscard]] ValueVector<S, typename A::Scalar>
inverse(S const & space, Eigen::MatrixBase<A> const & a)
{
    EMRYS_FILTER_ASSERT(a.size() == valueDimension(space));
    auto result = ValueVector<S, typename A::Scalar>{};
    result.resize(valueDimension(space));
    internal::inverseInto(space, a, result);
    return result;
}

/** \ingroup Spaces
 * \returns the identity of the group \a space, in \a Scalar: zero in Linear, and (0, 0, 0, 1) in UnitQuaternion
 */
template <typename Scalar = double, LieGroup S>
[[nodiscard]] ValueVector<S, Scalar>
identity(S const & space)
{
    auto result = ValueVector<S, Scalar>{};
    result.resize(valueDimension(space));
    internal::identityInto(space, result);
    return result;
}

/** \ingroup Spaces
 * \returns \f$ \mathrm{Exp}(d) \f$, the value of the group \a space that the difference \a d steps to from the
 * identity, so that <tt>plus(space, x, d)</tt> is <tt>compose(space, x, exp(space, d))</tt>
 */
template <LieGroup S, typename D>
[[nodiscard]] ValueVector<S, typename D::Scalar>
exp(S const & space, Eigen::MatrixBase<D> const & d)
{
    EMRYS_FILTER_ASSERT(d.size() == differenceDimension(space));
    auto result = ValueVector<S, typename D::Scalar>{};
    result.resize(valueDimension(space));
    internal::expInto(space, d, result);
    return result;
}

/** \ingroup Spaces
 * \returns \f$ \mathrm{Log}(a) \f$, the difference that steps from the identity of the group \a space to \a a, so
 * that <tt>minus(space, a, b)</tt> is <tt>log(space, compose(space, inverse(space, b), a))</tt>
 */
template <LieGroup S, typename A>
[[nodiscard]] DifferenceVector<S, typename A::Scalar>
log(S const & space, Eigen::MatrixBase<A> const & a)
{
    EMRYS_FILTER_ASSERT(a.size() == valueDimension(space));
    auto result = DifferenceVector<S, typename A::Scalar>{};
    result.resize(differenceDimension(space));
    internal::logInto(space, a, result);
    return result;
}

/** \ingroup Spaces
 * \returns the adjoint \f$ \mathrm{Ad}_a \f$ of the group \a space at the value \a a, the matrix over its differences
 * with \f$ a \cdot \mathrm{Exp}(d) = \mathrm{Exp}(\mathrm{Ad}_a d) \cdot a \f$: it takes a step in the body frame at
 * \a a to the same step in the world frame, and a covariance \f$ P \f$ in the body frame to
 * \f$ \mathrm{Ad}_a P \mathrm{Ad}_a^T \f$ in the world frame. The identity for a group under addition.
 */
template <LieGroup S, typename A>
[[nodiscard]] DifferenceMatrix<S, S, typename A::Scalar>
adjoint(S const & space, Eigen::MatrixBase<A> const & a)
{
    EMRYS_FILTER_ASSERT(a.size() == valueDimension(space));
    auto result = DifferenceMatrix<S, S, typename A::Scalar>{};
    result.resize(differenceDimension(space), differenceDimension(space));
    internal::adjointInto(space, a, result);
    return result;
}

/** \ingroup Spaces
 * \returns the right Jacobian \f$ J_r(d) \f$ of the group \a space at the difference \a d:
 * \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(d) \cdot \mathrm{Exp}(J_r(d) \epsilon) \f$, which is the reset
 * Jacobian of a group (resetJacobian()). The identity for a group under addition.
 */
template <LieGroup S, typename D>
[[nodiscard]] DifferenceMatrix<S, S, typename D::Scalar>
rightJacobian(S const & space, Eigen::MatrixBase<D> const & d)
{
    EMRYS_FILTER_ASSERT(d.size() == differenceDimension(space));
    auto result = DifferenceMatrix<S, S, typename D::Scalar>{};
    result.resize(differenceDimension(space), differenceDimension(space));
    internal::rightJacobianInto(space, d, result);
    return result;
}

/** \ingroup Spaces
 * \returns the left Jacobian \f$ J_l(d) = J_r(-d) \f$ of the group \a space at the difference \a d:
 * \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(J_l(d) \epsilon) \cdot \mathrm{Exp}(d) \f$
 */
template <LieGroup S, typename D>
[[nodiscard]] DifferenceMatrix<S, S, typename D::Scalar>
leftJacobian(S const & space, Eigen::MatrixBase<D> const & d)
{
    return Emrys::rightJacobian(space, (-d).eval());
}

/** \ingroup Spaces
 * compose() in the fixed-size group \a S, named as a template argument: <tt>compose<Pose>(a, b)</tt>.
 */
template <FixedSizeSpace S, typename A, typename B>
    requires LieGroup<S>
[[nodiscard]] auto
compose(Eigen::MatrixBase<A> const & a, Eigen::MatrixBase<B> const & b)
{
    return Emrys::compose(S{}, a, b);
}

/** \ingroup Spaces
 * inverse() in the fixed-size group \a S, named as a template argument: <tt>inverse<Pose>(a)</tt>.
 */
template <FixedSizeSpace S, typename A>
    requires LieGroup<S>
[[nodiscard]] auto
inverse(Eigen::MatrixBase<A> const & a)
{
    return Emrys::inverse(S{}, a);
}

/** \ingroup Spaces
 * identity() of the fixed-size group \a S, named as a template argument: <tt>identity<Pose>()</tt>.
 */
template <FixedSizeSpace S, typename Scalar = double>
    requires LieGroup<S>
[[nodiscard]] auto
identity()
{
    return Emrys::identity<Scalar>(S{});
}

/** \ingroup Spaces
 * exp() in the fixed-size group \a S, named as a template argument: <tt>exp<Pose>(d)</tt>.
 */
template <FixedSizeSpace S, typename D>
    requires LieGroup<S>
[[nodiscard]] auto
exp(Eigen::MatrixBase<D> const & d)
{
    return Emrys::exp(S{}, d);
}

/** \ingroup Spaces
 * log() in the fixed-size group \a S, named as a template argument: <tt>log<Pose>(a)</tt>.
 */
template <FixedSizeSpace S, typename A>
    requires LieGroup<S>
[[nodiscard]] auto
log(Eigen::MatrixBase<A> const & a)
{
    return Emrys::log(S{}, a);
}

/** \ingroup Spaces
 * adjoint() in the fixed-size group \a S, named as a template argument: <tt>adjoint<Pose>(a)</tt>.
 */
template <FixedSizeSpace S, typename A>
    requires LieGroup<S>
[[nodiscard]] auto
adjoint(Eigen::MatrixBase<A> const & a)
{
    return Emrys::adjoint(S{}, a);
}

/** \ingroup Spaces
 * rightJacobian() in the fixed-size group \a S, named as a template argument: <tt>rightJacobian<Pose>(d)</tt>.
 */
template <FixedSizeSpace S, typename D>
    requires LieGroup<S>
[[nodiscard]] auto
rightJacobian(Eigen::MatrixBase<D> const & d)
{
    return Emrys::rightJacobian(S{}, d);
}

/** \ingroup Spaces
 * leftJacobian() in the fixed-size group \a S, named as a template argument: <tt>leftJacobian<Pose>(d)</tt>.
 */
template <FixedSizeSpace S, typename D>
    requires LieGroup<S>
[[nodiscard]] auto
leftJacobian(Eigen::MatrixBase<D> const & d)
{
    return Emrys::leftJacobian(S{}, d);
}

}  // namespace Emrys

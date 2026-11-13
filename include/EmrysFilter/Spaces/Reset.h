#pragma once

/** \file
 * The covariance reset (decisions.md [covariance-reset]): hasIdentityReset<>, resetJacobian(), and
 * resetCovariance(), which carry a covariance from the tangent space at a value to the one at that
 * value moved by a step, block by block, and compile to nothing on a space whose every block moves
 * its differences unchanged. The Kalman update and the %Gaussian conditional call it (Condition.h,
 * Indexing.h), and so does the smoother (GaussianHistory.h); the information forms of the %Gaussian move
 * their reference through the inverse Jacobians (internal::resetInfoAt(), [gaussian-forms]). Read
 * UnitQuaternion.h next, for a space that supplies its reset in closed form.
 */

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/LU>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** Whether \a S writes its reset Jacobian in closed form: a resetJacobian(x, d, out) the library
 * can call on Eigen blocks.
 */
template <typename S>
concept HasResetJacobian = requires(S const & space, Eigen::MatrixXd & block) {
    space.resetJacobian(block.col(0), block.col(0), block.middleRows(0, 0));
};

/** Whether \a S declares its reset Jacobian the identity, with
 * <tt>static constexpr bool identityReset = true;</tt>.
 */
template <typename S>
concept DeclaresIdentityReset = requires { requires bool(S::identityReset); };

template <typename S>
consteval bool
identityResetOf();

/** Whether every one of the blocks in the tuple type \a Blocks has an identity reset. */
template <typename Blocks>
struct AllIdentityReset;

/** Whether every one of \a Blocks has an identity reset. */
template <typename... Blocks>
struct AllIdentityReset<std::tuple<Blocks...>> : std::bool_constant<(identityResetOf<Blocks>() && ...)>
{};

/** \returns whether the reset Jacobian of \a S is the identity: declared so; otherwise not where
 * \a S writes one; for a Composite or Repeated, whether every block's is; and otherwise whether
 * \a S has no plus() of its own.
 */
template <typename S>
consteval bool
identityResetOf()
{
    if constexpr (DeclaresIdentityReset<S>)
    {
        return true;
    }
    else if constexpr (HasResetJacobian<S>)
    {
        return false;
    }
    else if constexpr (IsComposite<S>)
    {
        return AllIdentityReset<CompositeBlocks<S>>::value;
    }
    else if constexpr (IsRepeated<S>)
    {
        return identityResetOf<RepeatedBlock<S>>();
    }
    else
    {
        return !HasPlus<S>;
    }
}

}  // namespace internal

/** \ingroup Spaces
 * Whether the reset Jacobian of \a S is the identity, so that a covariance needs no change when its
 * mean moves (decisions.md [covariance-reset]). True for a space with no plus() of its own, such as Linear and
 * Periodic; for one that declares
 * <tt>static constexpr bool identityReset = true;</tt>; and for a Composite or Repeated whose blocks
 * all have it. Where it holds, resetCovariance() does nothing at all.
 */
template <typename S>
inline constexpr bool hasIdentityReset = internal::identityResetOf<S>();

namespace internal
{

/** Writes through \a out the reset Jacobian of \a block, a block without an identity reset, at the
 * value \a x and the step \a d, with \a moved its \f$ x \boxplus d \f$: the block's own
 * resetJacobian(), or else the derivative of \f$ (x \boxplus (d + \epsilon)) \boxminus \f$ \a moved
 * in \f$ \epsilon \f$ at zero, by forward-mode automatic differentiation of the block's plus() and
 * minus(), with a derivative vector of the block's difference dimension.
 */
template <typename Block, typename X, typename D, typename Moved, typename Out>
void
blockResetJacobian(
    Block const & block,
    Eigen::MatrixBase<X> const & x,
    Eigen::MatrixBase<D> const & d,
    [[maybe_unused]] Eigen::MatrixBase<Moved> const & moved,
    Eigen::MatrixBase<Out> const & out
)
{
    if constexpr (HasResetJacobian<Block>)
    {
        block.resetJacobian(x, d, out);
    }
    else
    {
        using Scalar = typename Out::Scalar;
        constexpr int K = valueDimensionAtCompileTime<DifferenceSpace<Block>>;
        constexpr int KMax = maxValueDimensionAtCompileTime<DifferenceSpace<Block>>;
        constexpr int V = valueDimensionAtCompileTime<Block>;
        constexpr int VMax = maxValueDimensionAtCompileTime<Block>;
        using Dual = Eigen::AutoDiffScalar<Eigen::Matrix<Scalar, K, 1, Eigen::ColMajor, KMax, 1>>;
        using DualValue = Eigen::Matrix<Dual, V, 1, Eigen::ColMajor, VMax, 1>;
        using DualDifference = Eigen::Matrix<Dual, K, 1, Eigen::ColMajor, KMax, 1>;
        Eigen::Index const k = differenceDimension(block);
        auto step = DualDifference{};
        step.resize(k);
        for (Eigen::Index i = 0; i < k; ++i)
        {
            step(i) = Dual{Scalar(d(i)), int(k), int(i)};
        }
        DualValue const start = x.template cast<Dual>();
        DualValue const end = moved.template cast<Dual>();
        DualValue perturbed = DualValue::Zero(start.size());
        plusInto(block, start, step, perturbed);
        DualDifference error = DualDifference::Zero(k);
        minusInto(block, perturbed, end, error);
        auto & o = writable(out);
        for (Eigen::Index r = 0; r < k; ++r)
        {
            // Of a run-time size, the derivatives of a coordinate that does not depend on the step
            // would be empty; the Jacobian of a space whose minus() undoes its plus() is invertible.
            EMRYS_FILTER_ASSERT(
                error(r).derivatives().size() == k && "each coordinate of the difference must depend on the step"
            );
            o.row(r) = error(r).derivatives().transpose();
        }
    }
}

/** Calls \a f(block, valueOffset, differenceOffset) for each block of \a space whose reset is not
 * the identity, in order, with the offsets of its first value and difference coordinates: each
 * such block of a Composite, each repetition of a Repeated block, and a space with a
 * resetJacobian() of its own or written from scratch as a whole. Blocks with an identity reset are
 * passed over at compile time.
 */
template <typename S, typename F>
void
forEachResetBlock(S const & space, F && f, Eigen::Index valueOffset = 0, Eigen::Index differenceOffset = 0)
{
    if constexpr (hasIdentityReset<S>)
    {}
    else if constexpr (IsComposite<S> && !HasResetJacobian<S>)
    {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((forEachResetBlock(space.template block<I>(), f, valueOffset, differenceOffset),
              valueOffset += valueDimension(space.template block<I>()),
              differenceOffset += differenceDimension(space.template block<I>())),
             ...);
        }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<S>>>{});
    }
    else if constexpr (IsRepeated<S> && !HasResetJacobian<S>)
    {
        auto const & block = space.block();
        Eigen::Index const values = valueDimension(block);
        Eigen::Index const differences = differenceDimension(block);
        for (Eigen::Index k = 0; k < space.count(); ++k)
        {
            forEachResetBlock(block, f, valueOffset + k * values, differenceOffset + k * differences);
        }
    }
    else
    {
        f(space, valueOffset, differenceOffset);
    }
}

/** Replaces \a cov by \f$ G \, \mathrm{cov} \, G^T \f$, where \f$ G \f$ is the identity but for
 * the square block \a g at rows and columns \a offset: the strip \f$ G P_{o,:} \f$ is formed once
 * and written into those rows and, transposed, into those columns, and the diagonal block
 * \f$ g P_{oo} g^T \f$ is made exactly symmetric. Reads and writes \a cov whole.
 */
template <typename Cov, typename G>
void
congruence(Eigen::MatrixBase<Cov> const & cov, Eigen::Index offset, G const & g)
{
    using Scalar = typename Cov::Scalar;
    constexpr int K = G::RowsAtCompileTime;
    constexpr int KMax = G::MaxRowsAtCompileTime;
    constexpr int N = Cov::ColsAtCompileTime;
    constexpr int NMax = Cov::MaxColsAtCompileTime;
    auto & p = writable(cov);
    Eigen::Index const k = g.rows();
    auto strip = MatrixOf<Scalar, K, N, KMax, NMax>{};
    strip.resize(k, p.cols());
    strip.noalias() = multiply(g, middleRowsOf<K>(p, offset, k));
    middleRowsOf<K>(p, offset, k) = strip;
    middleColsOf<K>(p, offset, k) = strip.transpose();
    auto diagonal = MatrixOf<Scalar, K, K, KMax, KMax>{};
    diagonal.resize(k, k);
    diagonal.noalias() = multiply(middleColsOf<K>(strip, offset, k), g.transpose());
    blockOf<K, K>(p, offset, offset, k, k) = Scalar(0.5) * (diagonal + diagonal.transpose());
}

/** Carries \a cov, of which only the whole matrix is read, from the tangent space at \a x to the
 * one at \a moved, which is \f$ x \boxplus d \f$, or the reverse when \a Inverse: by
 * \f$ G \, \mathrm{cov} \, G^T \f$ with \f$ G \f$ the reset Jacobian of \a space at \a x and \a d,
 * or by \f$ G^{-1} \mathrm{cov} \, G^{-T} \f$, block by block (see resetCovariance()). Does nothing
 * where \a space has an identity reset (hasIdentityReset).
 */
template <bool Inverse = false, typename S, typename X, typename D, typename Moved, typename Cov>
void
resetCovarianceAt(
    S const & space,
    Eigen::MatrixBase<X> const & x,
    Eigen::MatrixBase<D> const & d,
    Eigen::MatrixBase<Moved> const & moved,
    Eigen::MatrixBase<Cov> const & cov
)
{
    if constexpr (!hasIdentityReset<S>)
    {
        EMRYS_FILTER_ASSERT(
            x.size() == valueDimension(space) && moved.size() == valueDimension(space)
            && d.size() == differenceDimension(space) && cov.rows() == differenceDimension(space)
            && cov.cols() == differenceDimension(space) && "the value, step, and covariance must fit the space"
        );
        forEachResetBlock(space, [&](auto const & block, Eigen::Index values, Eigen::Index differences) {
            using Block = std::remove_cvref_t<decltype(block)>;
            using Scalar = typename Cov::Scalar;
            constexpr int K = valueDimensionAtCompileTime<DifferenceSpace<Block>>;
            constexpr int V = valueDimensionAtCompileTime<Block>;
            constexpr int KMax = maxValueDimensionAtCompileTime<DifferenceSpace<Block>>;
            Eigen::Index const v = valueDimension(block);
            Eigen::Index const k = differenceDimension(block);
            MatrixOf<Scalar, K, K, KMax, KMax> g = MatrixOf<Scalar, K, K, KMax, KMax>::Zero(k, k);
            blockResetJacobian(
                block,
                middleRowsOf<V>(x, values, v),
                middleRowsOf<K>(d, differences, k),
                middleRowsOf<V>(moved, values, v),
                g
            );
            if constexpr (Inverse)
            {
                using Matrix = MatrixOf<Scalar, K, K, KMax, KMax>;
                g = inverseFrom<Matrix>(Eigen::PartialPivLU<Matrix>{g});
            }
            congruence(cov, differences, g);
        });
    }
}

/** Calls \a f(differenceOffset, gInverse) for each block of \a space whose reset is not the
 * identity, with \a gInverse the inverse of the block's reset Jacobian at the value \a x and the
 * step \a d, where \a moved is \f$ x \boxplus d \f$. The information forms of the %Gaussian apply
 * these to move their reference (decisions.md [covariance-reset], [gaussian-forms]), where a covariance's reset applies
 * \f$ G \f$ itself. Does nothing where \a space has an identity reset (hasIdentityReset).
 */
template <typename S, typename X, typename D, typename Moved, typename F>
void
forEachInverseResetJacobian(
    S const & space,
    Eigen::MatrixBase<X> const & x,
    Eigen::MatrixBase<D> const & d,
    Eigen::MatrixBase<Moved> const & moved,
    F && f
)
{
    if constexpr (!hasIdentityReset<S>)
    {
        EMRYS_FILTER_ASSERT(
            x.size() == valueDimension(space) && moved.size() == valueDimension(space)
            && d.size() == differenceDimension(space) && "the value and step must fit the space"
        );
        forEachResetBlock(space, [&](auto const & block, Eigen::Index values, Eigen::Index differences) {
            using Block = std::remove_cvref_t<decltype(block)>;
            using Scalar = typename X::Scalar;
            constexpr int K = valueDimensionAtCompileTime<DifferenceSpace<Block>>;
            constexpr int V = valueDimensionAtCompileTime<Block>;
            constexpr int KMax = maxValueDimensionAtCompileTime<DifferenceSpace<Block>>;
            Eigen::Index const v = valueDimension(block);
            Eigen::Index const k = differenceDimension(block);
            MatrixOf<Scalar, K, K, KMax, KMax> g = MatrixOf<Scalar, K, K, KMax, KMax>::Zero(k, k);
            blockResetJacobian(
                block,
                middleRowsOf<V>(x, values, v),
                middleRowsOf<K>(d, differences, k),
                middleRowsOf<V>(moved, values, v),
                g
            );
            using Matrix = MatrixOf<Scalar, K, K, KMax, KMax>;
            g = inverseFrom<Matrix>(Eigen::PartialPivLU<Matrix>{g});
            f(differences, g);
        });
    }
}

/** Carries the precision \a information, read and written whole and symmetric, and the
 * information vector \a vector of a %Gaussian in information form from the tangent space at \a x to
 * the one at \a moved, which is \f$ x \boxplus d \f$: \f$ \Lambda \leftarrow G^{-T} \Lambda G^{-1} \f$
 * and \f$ \eta \leftarrow G^{-T} \eta \f$, block by block, with \f$ G \f$ the reset Jacobian of
 * resetJacobian() (decisions.md [covariance-reset], [gaussian-forms]). It is the information form's counterpart of
 * resetCovarianceAt(): where an error \f$ \epsilon \f$ at \a x is \f$ G \epsilon \f$ at \a moved,
 * a quadratic form in the error moves by the inverse. \a vector must already be re-expressed
 * about \a moved, \f$ \eta - \Lambda d \f$, since that is what \f$ G^{-T} \f$ applies to. Does
 * nothing where \a space has an identity reset (hasIdentityReset).
 */
template <typename S, typename X, typename D, typename Moved, typename Info, typename Vector>
void
resetInfoAt(
    S const & space,
    Eigen::MatrixBase<X> const & x,
    Eigen::MatrixBase<D> const & d,
    Eigen::MatrixBase<Moved> const & moved,
    Eigen::MatrixBase<Info> const & info,
    Eigen::MatrixBase<Vector> const & vector
)
{
    forEachInverseResetJacobian(space, x, d, moved, [&](Eigen::Index differences, auto const & gInverse) {
        using GInverse = std::remove_cvref_t<decltype(gInverse)>;
        GInverse const gInverseTransposed = gInverse.transpose();
        congruence(info, differences, gInverseTransposed);
        auto & eta = writable(vector);
        Eigen::Index const k = gInverse.rows();
        middleRowsOf<GInverse::RowsAtCompileTime>(eta, differences, k) =
            gInverseTransposed * middleRowsOf<GInverse::RowsAtCompileTime>(eta, differences, k);
    });
}

}  // namespace internal

/** \ingroup Spaces
 * \returns the reset Jacobian of \a space at the value \a x and the step \a d, dense: the identity
 * but for the blocks without an identity reset, where
 * \f[ G = \frac{\partial}{\partial \epsilon} \Big[ \big(x \boxplus (d + \epsilon)\big) \boxminus (x \boxplus d)
 * \Big]_{\epsilon = 0}, \f] which takes an error at \a x to the same error seen from \f$ x \boxplus d \f$ (decisions.md
 * [covariance-reset]). A block gives its own in closed form with a member <tt>resetJacobian(x, d, out)</tt>, declares
 * it the identity with <tt>static constexpr bool identityReset = true;</tt>, or has it computed by forward-mode
 * automatic differentiation of its plus() and minus(). For tests and for callers who want \f$ G \f$ itself;
 * resetCovariance() applies it without forming it.
 *
 * For a unit quaternion with \f$ q \boxplus \delta = q \otimes \mathrm{Exp}(\delta) \f$ it is the
 * right Jacobian of SO(3), \f$ J_r(d) \f$, in the closed form of
 * \ref Sola2018 "Solà, Deray, and Atchuthan (2018)", eq. (143). The reset is that of the
 * error-state Kalman filter of \ref Sola2017 "Solà (2017)", §6.3, eqs. (284) to (288), which keeps
 * only the first-order \f$ I - \tfrac12 [d]_\times \f$ (eq. 294), and of
 * \ref Mueller2017 "Mueller, Hehn, and D'Andrea (2017)", who derive it for a filter with an attitude
 * and show it is missing from the multiplicative extended Kalman filter of
 * \ref Markley2003 "Markley (2003)", whose reset moves the attitude and leaves the covariance
 * unchanged. The departure from all three is generality: the Jacobian comes from the space,
 * block by block, by differentiating its own \f$ \boxplus \f$ and \f$ \boxminus \f$ unless it gives
 * a closed form, rather than being written for an attitude.
 */
template <Space S, typename XDerived, typename DDerived>
[[nodiscard]] auto
resetJacobian(S const & space, Eigen::MatrixBase<XDerived> const & x, Eigen::MatrixBase<DDerived> const & d)
{
    using Scalar = typename XDerived::Scalar;
    constexpr int N = valueDimensionAtCompileTime<DifferenceSpace<S>>;
    constexpr int NMax = maxValueDimensionAtCompileTime<DifferenceSpace<S>>;
    Eigen::Index const n = differenceDimension(space);
    auto result = internal::MatrixOf<Scalar, N, N, NMax, NMax>{};
    result.setIdentity(n, n);
    if constexpr (!hasIdentityReset<S>)
    {
        auto const moved = Emrys::plus(space, x, d);
        internal::forEachResetBlock(space, [&](auto const & block, Eigen::Index values, Eigen::Index differences) {
            using Block = std::remove_cvref_t<decltype(block)>;
            constexpr int K = valueDimensionAtCompileTime<DifferenceSpace<Block>>;
            constexpr int V = valueDimensionAtCompileTime<Block>;
            Eigen::Index const v = valueDimension(block);
            Eigen::Index const k = differenceDimension(block);
            internal::blockResetJacobian(
                block,
                internal::middleRowsOf<V>(x, values, v),
                internal::middleRowsOf<K>(d, differences, k),
                internal::middleRowsOf<V>(moved, values, v),
                internal::blockOf<K, K>(result, differences, differences, k, k)
            );
        });
    }
    return result;
}

/** \ingroup Spaces
 * Carries the covariance \a cov from the tangent space at \a x to the one at
 * \f$ x \boxplus d \f$, in place: \f$ P \leftarrow G P G^T \f$, with \f$ G \f$ the reset Jacobian
 * of resetJacobian() (decisions.md [covariance-reset]). An update that moves a mean by a step and keeps the
 * covariance computed at the old mean calls it, as the Kalman update and the %Gaussian conditional
 * do.
 *
 * \a cov is read and written whole, and must be symmetric; it may be a block of a larger matrix.
 * Where \a space has an identity reset (hasIdentityReset), this does nothing and compiles to
 * nothing. Otherwise only the blocks
 * that need it are touched: for a block of \f$ k \f$ rows at offset \f$ o \f$ in an \f$ n \f$-row
 * covariance, the strip \f$ G P_{o,:} \f$ is formed once and written into those rows and,
 * transposed, into those columns, and the diagonal block \f$ G P_{oo} G^T \f$ is made exactly
 * symmetric, about \f$ 4 k^2 n \f$ flops. A symmetric \a cov stays exactly symmetric, and as a
 * congruence it keeps a positive semidefinite one so. Allocates nothing for a bounded space.
 *
 * See resetJacobian() for the sources.
 */
template <Space S, typename XDerived, typename DDerived, typename CovDerived>
void
resetCovariance(
    S const & space,
    Eigen::MatrixBase<XDerived> const & x,
    Eigen::MatrixBase<DDerived> const & d,
    Eigen::MatrixBase<CovDerived> const & cov
)
{
    if constexpr (!hasIdentityReset<S>)
    {
        auto const moved = Emrys::plus(space, x, d);
        internal::resetCovarianceAt(space, x, d, moved, cov);
    }
}

}  // namespace Emrys

#pragma once

/** \file
 * A check of a Lie group's members against each other (docs/design/groups.md; a prototype): checkGroup(), for the
 * author of a group written from scratch to run in a test, and GroupCheck, what it returns.
 */

#include <algorithm>

#include <unsupported/Eigen/AutoDiff>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Spaces
 * What checkGroup() found: for each relation between a group's members, the largest absolute coordinate of what
 * should be zero. Each is near rounding error for a consistent group.
 */
struct GroupCheck
{
    double identity = 0;       ///< \f$ a \cdot e \f$ and \f$ e \cdot a \f$ against \f$ a \f$
    double inverse = 0;        ///< \f$ a \cdot a^{-1} \f$ against \f$ e \f$
    double associative = 0;    ///< \f$ (a \cdot b) \cdot c \f$ against \f$ a \cdot (b \cdot c) \f$, with \f$ c = \mathrm{Exp}(d) \f$
    double logOfExp = 0;       ///< \f$ \mathrm{Log}(\mathrm{Exp}(d)) \f$ against \f$ d \f$
    double plus = 0;           ///< plus(a, d) against \f$ a \cdot \mathrm{Exp}(d) \f$
    double minus = 0;          ///< minus(a, b) against \f$ \mathrm{Log}(b^{-1} \cdot a) \f$
    double adjoint = 0;        ///< \f$ a \cdot \mathrm{Exp}(d) \f$ against \f$ \mathrm{Exp}(\mathrm{Ad}_a d) \cdot a \f$
    double rightJacobian = 0;  ///< rightJacobian(d) against the derivative of \f$ \mathrm{Log}(\mathrm{Exp}(d)^{-1} \mathrm{Exp}(d + \epsilon)) \f$

    /** \returns the largest of the residuals */
    [[nodiscard]] double
    worst() const
    {
        return std::max({identity, inverse, associative, logOfExp, plus, minus, adjoint, rightJacobian});
    }
};

/** \ingroup Spaces
 * \returns how far the members of the group \a space are from agreeing with each other at the values \a a and \a b
 * and the difference \a d: the group axioms, plus() and minus() against the product, the exponential, and the
 * logarithm, the adjoint against the product, and the right Jacobian against the automatic derivative of the
 * exponential and logarithm. A test of a group written from scratch asserts that worst() is near rounding error at a
 * few values, with \a d well inside the range where log() inverts exp().
 */
template <LieGroup S, typename A, typename B, typename D>
[[nodiscard]] GroupCheck
checkGroup(
    S const & space,
    Eigen::MatrixBase<A> const & a,
    Eigen::MatrixBase<B> const & b,
    Eigen::MatrixBase<D> const & d
)
{
    using Value = ValueVector<S>;
    using Difference = DifferenceVector<S>;
    auto const distance = [&](Value const & x, Value const & y) {
        Difference const error = Emrys::minus(space, x, y);
        return error.size() == 0 ? 0.0 : double(error.cwiseAbs().maxCoeff());
    };
    auto const size = [](auto const & v) {
        return v.size() == 0 ? 0.0 : double(v.cwiseAbs().maxCoeff());
    };
    Value const va = a;
    Value const vb = b;
    Difference const step = d;
    Value const e = Emrys::identity(space);
    Value const c = Emrys::exp(space, step);

    auto result = GroupCheck{};
    result.identity =
        std::max(distance(Emrys::compose(space, va, e), va), distance(Emrys::compose(space, e, va), va));
    result.inverse = distance(Emrys::compose(space, va, Emrys::inverse(space, va)), e);
    result.associative = distance(
        Emrys::compose(space, Value{Emrys::compose(space, va, vb)}, c),
        Emrys::compose(space, va, Value{Emrys::compose(space, vb, c)})
    );
    result.logOfExp = size(Difference{Emrys::log(space, c) - step});
    result.plus = distance(Value{Emrys::plus(space, va, step)}, Emrys::compose(space, va, c));
    result.minus = size(
        Difference{
            Emrys::minus(space, va, vb)
            - Emrys::log(space, Value{Emrys::compose(space, Value{Emrys::inverse(space, vb)}, va)})
        }
    );
    Difference const turned = Emrys::adjoint(space, va) * step;
    result.adjoint =
        distance(Emrys::compose(space, va, c), Emrys::compose(space, Value{Emrys::exp(space, turned)}, va));

    // The right Jacobian, against the derivative of Log(Exp(d)^-1 Exp(d + eps)) in eps at zero.
    using Dual = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    Eigen::Index const n = differenceDimension(space);
    auto perturbed = DifferenceVector<S, Dual>{};
    perturbed.resize(n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        perturbed(i) = Dual{step(i), Eigen::VectorXd::Unit(n, i)};
    }
    ValueVector<S, Dual> const moved = Emrys::exp(space, perturbed);
    ValueVector<S, Dual> const back = Emrys::inverse(space, c).template cast<Dual>();
    DifferenceVector<S, Dual> const error = Emrys::log(space, ValueVector<S, Dual>{Emrys::compose(space, back, moved)});
    auto const jacobian = Emrys::rightJacobian(space, step);
    for (Eigen::Index r = 0; r < n; ++r)
    {
        for (Eigen::Index k = 0; k < n; ++k)
        {
            double const derivative = error(r).derivatives().size() == n ? error(r).derivatives()(k) : 0.0;
            result.rightJacobian = std::max(result.rightJacobian, std::abs(jacobian(r, k) - derivative));
        }
    }
    return result;
}

}  // namespace Emrys

#pragma once

/** \file
 * InfoFilter (decisions.md [filters-and-recursions], [gaussian-forms]): the KalmanFilter over an InfoGaussianParams,
 * spelled space first, whose update adds information and whose prediction goes through the moments. Read
 * SqrtKalmanFilter.h next.
 */

#include <utility>

#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief The information filter: a KalmanFilter whose state is an InfoGaussianParams, the
 * precision and information vector about a reference (decisions.md [gaussian-forms]).
 *
 * The update adds \f$ H^T S_0^{-1} H \f$ to the precision and the matching vector to the
 * information vector, with \f$ H \f$ and \f$ S_0 \f$ the statistical linearization of the
 * measurement by its own method at the mean [covariance-update-forms]: exact for an affine measurement, the extended
 * information filter of \ref Mutambara1998 "Mutambara (1998)" otherwise, and independent of the
 * order several measurements arrive in. A state with no information in some direction
 * (\f$ \Lambda = 0 \f$ at the start) takes Linearized updates at its reference until the
 * measurements determine every direction; a sigma-point update, and every prediction, go through
 * the covariance and return GaussianError::Singular until then. Prediction converts to the
 * covariance form, predicts as KalmanFilter does, and converts back with the reference at the
 * predicted mean, so each step returns a \c std::expected. Measurement noise has one component
 * and the policy is Collapse; there is no predictJoint().
 *
 * It is the KalmanFilter it names with nothing added, as a derived class rather than an alias so
 * that its arguments deduce on every compiler [filters-and-recursions]: <tt>InfoFilter filter{model,
 * *toInfo(initial)}</tt>.
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Model_   the process model, a ProcessModel or a FixedStepModel
 * \tparam Method_  the prediction method; a FixedStepModel's own
 * \tparam Scalar_  scalar type, \c double by default
 * \tparam Upkeep_  the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                  one; RepairCovariance does not apply to this form
 *
 * \sa KalmanFilter, InfoGaussianParams, SqrtInfoFilter
 */
template <
    Space Space_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    RealScalar Scalar_ = double,
    typename Upkeep_ = upkeep::None>
class InfoFilter
    : public KalmanFilter<
          InfoGaussian<Space_, Scalar_>,
          Model_,
          Method_,
          KalmanRecursion<InfoGaussian<Space_, Scalar_>, Model_, Method_>,
          Upkeep_>
{
public:

    using Base = KalmanFilter<
        InfoGaussian<Space_, Scalar_>,
        Model_,
        Method_,
        KalmanRecursion<InfoGaussian<Space_, Scalar_>, Model_, Method_>,
        Upkeep_>;  ///< the filter this is

    /** A filter with process model \a model, initial state \a state, and prediction method \a method;
     * a fixed-step model carries its own, so there is none to give. Declared rather than inherited
     * so that only the deduction guide below deduces the filter's arguments.
     */
    InfoFilter(
        Model_ model,
        InfoGaussianParams<Space_, Scalar_> state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : Base{std::move(model), std::move(state), std::move(method), std::move(upkeep)}
    {}

    /** A filter with process model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    InfoFilter(Model_ model, InfoGaussianParams<Space_, Scalar_> state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : Base{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}
};

/// \cond DEDUCTION_GUIDES
// One guide with a defaulted method, its parameters in the class's order.
template <typename Space, typename Model, typename Method = DefaultMethodFor<Model>, typename Scalar>
    requires(!internal::UpkeepArgument<Method>)
InfoFilter(Model, InfoGaussianParams<Space, Scalar>, Method = {})
    -> InfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar>;

template <typename Space, typename Model, typename Scalar, typename U>
    requires internal::UpkeepArgument<U>
InfoFilter(Model, InfoGaussianParams<Space, Scalar>, U)
    -> InfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, DefaultMethodFor<Model>, Scalar, U>;

template <typename Space, typename Model, typename Method, typename Scalar, typename U>
InfoFilter(Model, InfoGaussianParams<Space, Scalar>, Method, U)
    -> InfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar, U>;
/// \endcond

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a filter: its name, and its state nested under it. */
template <typename Space_, typename Model_, typename Method_, typename Scalar_, typename Upkeep_>
void
describe(std::ostream & stream, InfoFilter<Space_, Model_, Method_, Scalar_, Upkeep_> const & filter, Indent indent)
{
    describeFilter(stream, "information filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

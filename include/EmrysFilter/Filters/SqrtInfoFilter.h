#pragma once

/** \file
 * SqrtInfoFilter (decisions.md [filters-and-recursions], [gaussian-forms]): the KalmanFilter over a
 * SqrtInfoGaussianParams, spelled space first, Bierman's filter on the data array. Read
 * MixtureFilter.h next.
 */

#include <utility>

#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief The square-root information filter: a KalmanFilter whose state is a
 * SqrtInfoGaussianParams, the data array \f$ [\,R \;\; r\,] \f$ about a reference
 * (decisions.md [gaussian-forms]).
 *
 * The filter of \ref Bierman1977 "Bierman (1977)", with the update as given in
 * \ref Ke2024 "Ke et al. (2024)", §II. The update stacks the
 * whitened rows of the measurement's statistical linearization [covariance-update-forms] under the data array and
 * triangularizes, which never forms a precision or a covariance; the residual it leaves is the
 * innovation's NIS. Whitening needs $ S_0 $ positive definite, and the update returns
 * GaussianError::NotPositiveDefinite where it is not [innovation-cholesky]. Under Linearized the prediction runs the
 * time update through the inverse of the step's Jacobian, so it starts from a state with no information at all
 * (\f$ R = 0 \f$), where InfoFilter cannot predict, and fails instead on a singular
 * Jacobian; a sigma-point prediction, and a sigma-point update, go through the moments and return
 * GaussianError::Singular while the state has no information in some direction. Each step
 * returns a \c std::expected. Measurement noise has one component and the policy is Collapse;
 * there is no predictJoint().
 *
 * It is the KalmanFilter it names with nothing added, as a derived class rather than an alias so
 * that its arguments deduce on every compiler [filters-and-recursions]: <tt>SqrtInfoFilter
 * filter{model, toSqrtInfo(initial).value()}</tt>.
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Model_   the process model, a ProcessModel or a FixedStepModel
 * \tparam Method_  the prediction method; a FixedStepModel's own
 * \tparam Scalar_  scalar type, \c double by default
 * \tparam Upkeep_  the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                  one; RepairCovariance does not apply to this form
 *
 * \sa KalmanFilter, SqrtInfoGaussianParams, InfoFilter, SqrtKalmanFilter
 */
template <
    Space Space_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    RealScalar Scalar_ = double,
    typename Upkeep_ = upkeep::None>
class SqrtInfoFilter
    : public KalmanFilter<
          SqrtInfoGaussian<Space_, Scalar_>,
          Model_,
          Method_,
          KalmanRecursion<SqrtInfoGaussian<Space_, Scalar_>, Model_, Method_>,
          Upkeep_>
{
public:

    /// the filter this is
    using Base = KalmanFilter<
        SqrtInfoGaussian<Space_, Scalar_>,
        Model_,
        Method_,
        KalmanRecursion<SqrtInfoGaussian<Space_, Scalar_>, Model_, Method_>,
        Upkeep_>;

    /** A filter with process model \a model, initial state \a state, and prediction method \a method;
     * a fixed-step model carries its own, so there is none to give. Declared rather than inherited
     * so that only the deduction guide below deduces the filter's arguments.
     */
    SqrtInfoFilter(
        Model_ model,
        SqrtInfoGaussianParams<Space_, Scalar_> state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : Base{std::move(model), std::move(state), std::move(method), std::move(upkeep)}
    {}

    /** A filter with process model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    SqrtInfoFilter(Model_ model, SqrtInfoGaussianParams<Space_, Scalar_> state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : Base{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}
};

/// \cond DEDUCTION_GUIDES
// One guide with a defaulted method, its parameters in the class's order.
template <typename Space, typename Model, typename Method = DefaultMethodFor<Model>, typename Scalar>
    requires(!internal::UpkeepArgument<Method>)
SqrtInfoFilter(Model, SqrtInfoGaussianParams<Space, Scalar>, Method = {})
    -> SqrtInfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar>;

template <typename Space, typename Model, typename Scalar, typename U>
    requires internal::UpkeepArgument<U>
SqrtInfoFilter(Model, SqrtInfoGaussianParams<Space, Scalar>, U)
    -> SqrtInfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, DefaultMethodFor<Model>, Scalar, U>;

template <typename Space, typename Model, typename Method, typename Scalar, typename U>
SqrtInfoFilter(Model, SqrtInfoGaussianParams<Space, Scalar>, Method, U)
    -> SqrtInfoFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar, U>;
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
describe(std::ostream & stream, SqrtInfoFilter<Space_, Model_, Method_, Scalar_, Upkeep_> const & filter, Indent indent)
{
    describeFilter(stream, "square-root information filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

#pragma once

/** \file
 * SqrtKalmanFilter (decisions.md [filters-and-recursions], [gaussian-forms]): the KalmanFilter over a
 * SqrtGaussianParams, spelled space first, whose steps triangularize stacked factors and never form a covariance. Read
 * SqrtInfoFilter.h next.
 */

#include <utility>

#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Filters
 * \brief The square-root Kalman filter: a KalmanFilter whose state is a SqrtGaussianParams, a
 * mean and a lower-triangular factor of the covariance (decisions.md [gaussian-forms]).
 *
 * Prediction triangularizes \f$ [\,F S \;\; Q^{1/2}\,] \f$, with \f$ F \f$ the model's
 * transition or the step's Jacobian, or under Unscented the weighted deviations of sigma points
 * spread from the factor itself (\ref VanDerMerwe2001 "van der Merwe and Wan (2001)"). The update
 * triangularizes the joint array of the innovation and the state, whose factor holds the
 * innovation's factor, the gain, and the posterior factor (the array algorithms of
 * \ref Kaminski1971 "Kaminski, Bryson, and Schmidt (1971)"), with \f$ H \f$ and \f$ S_0 \f$ the
 * statistical linearization of the measurement by its own method [covariance-update-forms]. The factor represents the
 * covariance at twice the precision and stays positive semidefinite where the covariance form's
 * subtraction can lose definiteness, which is the case of a measurement far more precise than the
 * prior. predict() returns a \c std::expected, empty only when an Unscented with a negative
 * center weight leaves the predicted covariance indefinite; update() returns one too, empty only
 * when the innovation's factor has a zero on its diagonal [innovation-cholesky]. Measurement
 * noise has one component and the policy is Collapse, Select, or a Gate over them; there is no
 * predictJoint().
 *
 * It is the KalmanFilter it names with nothing added, as a derived class rather than an alias so
 * that its arguments deduce on every compiler [filters-and-recursions]: <tt>SqrtKalmanFilter filter{model,
 * *toSqrt(initial)}</tt>.
 *
 * \tparam Space_   the state space (see \ref Spaces)
 * \tparam Model_   the process model, a ProcessModel or a FixedStepModel
 * \tparam Method_  the prediction method; a FixedStepModel's own
 * \tparam Scalar_  scalar type, \c double by default
 * \tparam Upkeep_  the upkeep maintain() runs on the state (see \ref Upkeep), upkeep::None unless given
 *                  one; RepairCovariance does not apply to this form
 *
 * \sa KalmanFilter, SqrtGaussianParams, SqrtInfoFilter
 */
template <
    Space Space_,
    typename Model_,
    typename Method_ = DefaultMethodFor<Model_>,
    RealScalar Scalar_ = double,
    typename Upkeep_ = upkeep::None>
class SqrtKalmanFilter
    : public KalmanFilter<
          SqrtGaussian<Space_, Scalar_>,
          Model_,
          Method_,
          KalmanRecursion<SqrtGaussian<Space_, Scalar_>, Model_, Method_>,
          Upkeep_>
{
public:

    using Base = KalmanFilter<
        SqrtGaussian<Space_, Scalar_>,
        Model_,
        Method_,
        KalmanRecursion<SqrtGaussian<Space_, Scalar_>, Model_, Method_>,
        Upkeep_>;  ///< the filter this is

    /** A filter with process model \a model, initial state \a state, and prediction method \a method;
     * a fixed-step model carries its own, so there is none to give. Declared rather than inherited
     * so that only the deduction guide below deduces the filter's arguments.
     */
    SqrtKalmanFilter(
        Model_ model,
        SqrtGaussianParams<Space_, Scalar_> state,
        MethodFor<Model_, Method_> method = {},
        std::type_identity_t<Upkeep_> upkeep = {}
    )
        : Base{std::move(model), std::move(state), std::move(method), std::move(upkeep)}
    {}

    /** A filter with process model \a model, initial state \a state, the default prediction method,
     * and upkeep \a upkeep, which maintain() runs.
     */
    SqrtKalmanFilter(Model_ model, SqrtGaussianParams<Space_, Scalar_> state, std::type_identity_t<Upkeep_> upkeep)
        requires(!std::same_as<Upkeep_, Emrys::upkeep::None>)
        : Base{std::move(model), std::move(state), MethodFor<Model_, Method_>{}, std::move(upkeep)}
    {}
};

/// \cond DEDUCTION_GUIDES
// One guide with a defaulted method, its parameters in the class's order.
template <typename Space, typename Model, typename Method = DefaultMethodFor<Model>, typename Scalar>
    requires(!internal::UpkeepArgument<Method>)
SqrtKalmanFilter(Model, SqrtGaussianParams<Space, Scalar>, Method = {})
    -> SqrtKalmanFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar>;

template <typename Space, typename Model, typename Scalar, typename U>
    requires internal::UpkeepArgument<U>
SqrtKalmanFilter(Model, SqrtGaussianParams<Space, Scalar>, U)
    -> SqrtKalmanFilter<Space, internal::BoundStep<Model, Space, Scalar>, DefaultMethodFor<Model>, Scalar, U>;

template <typename Space, typename Model, typename Method, typename Scalar, typename U>
SqrtKalmanFilter(Model, SqrtGaussianParams<Space, Scalar>, Method, U)
    -> SqrtKalmanFilter<Space, internal::BoundStep<Model, Space, Scalar>, Method, Scalar, U>;
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
describe(
    std::ostream & stream,
    SqrtKalmanFilter<Space_, Model_, Method_, Scalar_, Upkeep_> const & filter,
    Indent indent
)
{
    describeFilter(stream, "square-root Kalman filter", filter.state(), indent);
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

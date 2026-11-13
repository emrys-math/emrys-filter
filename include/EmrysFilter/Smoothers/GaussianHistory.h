#pragma once

/** \file
 * Smoothing the Kalman family (decisions.md [smoothing]): smoothStep(), the Rauch-Tung-Striebel
 * backward step in the tangent space; and GaussianHistory, the record of a Kalman filter's forward
 * pass that a caller keeps, whose smooth() runs the backward pass over it, fixed-interval or
 * fixed-lag. The filter is unchanged but for predictJoint() (see PredictedJoint in
 * ProcessModel.h). The other histories follow its shape. Read KalmanFilter.h first, and
 * HybridGaussianHistory.h next; the design review is docs/guide/smoothing.md.
 */

#include <list>
#include <optional>
#include <utility>

#include <Eigen/Cholesky>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Spaces/Reset.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Smoothers
 * \returns the smoothed %Gaussian at the start of \a joint's step, given \a smoothedNext, the
 * smoothed %Gaussian at its end: the Rauch-Tung-Striebel backward step in the tangent space, with
 * the gain \f$ G_k = C_k P_{k+1 \mid k}^{-1} \f$ from the joint's cross-covariance,
 *
 * \f[ \hat x_{k \mid N} = \hat x_{k \mid k} \boxplus G_k \,(\hat x_{k+1 \mid N} \boxminus \hat x_{k+1 \mid k}), \qquad
 *     P_{k \mid N} = P_{k \mid k} + G_k \,(P_{k+1 \mid N} - P_{k+1 \mid k})\, G_k^T. \f]
 *
 * The smoother of \ref Rauch1965 "Rauch, Tung, and Striebel (1965)", in the form of the %Gaussian
 * smoother of \ref Sarkka2013 "Särkkä (2013)", chapter 10, which writes the gain in terms of the
 * cross-covariance so that the same step serves the affine, linearized, and unscented
 * predictions. The departure is the manifold: the residual is \f$ \boxminus \f$ and the step
 * \f$ \boxplus \f$, so a unit quaternion's smoothed mean stays unit, and each covariance is moved to
 * the tangent space it is combined in (decisions.md [covariance-reset]). \f$ P_{k+1 \mid N} \f$, about the
 * smoothed mean, is carried back to the tangent space at the predicted mean by the inverse of the
 * reset Jacobian at the predicted mean and the correction, and the smoothed covariance, formed at
 * the filtered mean, is carried to the smoothed mean by resetCovariance(). On a space with an
 * identity reset both are nothing. \f$ P_{k+1 \mid k} \f$ is factorized by LDLT, which tolerates a
 * semidefinite prediction.
 *
 * \sa GaussianHistory, PredictedJoint, \ref Smoothers
 */
template <typename Space_, typename Scalar_>
[[nodiscard]] GaussianParams<Space_, Scalar_>
smoothStep(
    PredictedJoint<Space_, Scalar_> const & joint,
    typename GaussianParams<Space_, Scalar_>::MatrixType const & gain,
    GaussianParams<Space_, Scalar_> const & smoothedNext
)
{
    using GaussianType = GaussianParams<Space_, Scalar_>;
    using MatrixType = typename GaussianType::MatrixType;
    auto const & space = joint.filtered.space();
    EMRYS_FILTER_ASSERT(
        joint.cross.rows() == joint.filtered.differenceDimension()
        && joint.cross.cols() == joint.predicted.differenceDimension()
        && "the joint's cross-covariance has the wrong shape"
    );
    EMRYS_FILTER_ASSERT(
        gain.rows() == joint.filtered.differenceDimension() && gain.cols() == joint.predicted.differenceDimension()
        && "the gain has the wrong shape"
    );
    EMRYS_FILTER_ASSERT(
        smoothedNext.differenceDimension() == joint.predicted.differenceDimension()
        && "the smoothed Gaussian must be over the space the joint predicted"
    );
    MatrixType const predictedCov = joint.predicted.covarianceLower().template selfadjointView<Eigen::Lower>();
    MatrixType nextCov = smoothedNext.covarianceLower().template selfadjointView<Eigen::Lower>();
    typename GaussianType::DifferenceVectorType const correction =
        Emrys::minus(joint.predicted.space(), smoothedNext.mean(), joint.predicted.mean());
    internal::resetCovarianceAt<true>(
        joint.predicted.space(),
        joint.predicted.mean(),
        correction,
        smoothedNext.mean(),
        nextCov
    );
    typename GaussianType::DifferenceVectorType const step = internal::multiply(gain, correction);
    typename GaussianType::VectorType mean = Emrys::plus(space, joint.filtered.mean(), step);
    MatrixType cov = joint.filtered.covarianceLower().template selfadjointView<Eigen::Lower>();
    cov.noalias() += internal::multiply(internal::multiply(gain, nextCov - predictedCov), gain.transpose());
    cov = (Scalar_(0.5) * (cov + cov.transpose())).eval();
    internal::resetCovarianceAt(space, joint.filtered.mean(), step, mean, cov);
    return {space, std::move(mean), std::move(cov)};
}

namespace internal
{

/** \returns the Rauch-Tung-Striebel gain of \a joint,
 * \f$ C_k = P_{k, k+1 \mid k} \, P_{k+1 \mid k}^{-1} \f$ from its cross-covariance and its
 * predicted covariance, which an LDLT factorizes, tolerating a semidefinite prediction. It depends
 * on the joint alone, so a history computes it once when it records the joint and applies it at
 * every smooth() rather than factorizing the prediction again each time (see \ref Allocation).
 */
template <typename Space_, typename Scalar_>
[[nodiscard]] typename GaussianParams<Space_, Scalar_>::MatrixType
rtsGain(PredictedJoint<Space_, Scalar_> const & joint)
{
    using MatrixType = typename GaussianParams<Space_, Scalar_>::MatrixType;
    EMRYS_FILTER_ASSERT(
        joint.cross.rows() == joint.filtered.differenceDimension()
        && joint.cross.cols() == joint.predicted.differenceDimension()
        && "the joint's cross-covariance has the wrong shape"
    );
    MatrixType const predictedCov = joint.predicted.covarianceLower().template selfadjointView<Eigen::Lower>();
    auto const ldlt = Eigen::LDLT<MatrixType>{predictedCov};
    return internal::transposedSolve<MatrixType>(ldlt, joint.cross.transpose());
}

}  // namespace internal

/** \ingroup Smoothers
 * smoothStep() with the gain computed here, from \a joint alone (internal::rtsGain()): the form
 * for one step; a history keeps the gain with the step and applies it at every smooth().
 */
template <typename Space_, typename Scalar_>
[[nodiscard]] GaussianParams<Space_, Scalar_>
smoothStep(PredictedJoint<Space_, Scalar_> const & joint, GaussianParams<Space_, Scalar_> const & smoothedNext)
{
    return smoothStep(joint, internal::rtsGain(joint), smoothedNext);
}

/** \ingroup Smoothers
 * \brief The record of a Kalman filter's forward pass that a smoother runs backward over: one
 * PredictedJoint per prediction, each with room for its smoothed %Gaussian.
 *
 * The caller keeps it, outside the filter (decisions.md [smoothing]): push() records the PredictedJoint
 * a filter's predictJoint() returns, and smooth() takes the filter's current state, the latest
 * posterior, and fills every step's smoothed %Gaussian by smoothStep(), from the newest step back.
 * Updates between predictions need no recording: a step's filtered %Gaussian is the posterior at
 * the moment of the prediction that follows them.
 *
 * \code
 * GaussianHistory<State> history;
 * for (auto const & z : measurements)
 * {
 *     history.push(filter.predictJoint(dt));
 *     if (!filter.update(fix(z))) { break; }
 * }
 * history.smooth(filter.state());
 * for (auto const & step : history) { use(step.smoothed); }  // x_{k|N}, oldest first
 * \endcode
 *
 * Without a Lag it keeps every step, for fixed-interval smoothing. With a Lag of \f$ L \f$ it
 * keeps the newest \f$ L \f$, for fixed-lag smoothing: once full, each push() reuses the oldest
 * step's node for the new one, so it allocates nothing after the first \f$ L \f$, and after
 * smooth() the oldest step, front(), holds \f$ \hat x_{k-L \mid k} \f$.
 *
 * Fixed-lag smoothing as the backward pass over the window is that of
 * \ref Sarkka2013 "Särkkä (2013)", chapter 10, which costs \f$ O(L) \f$ backward steps per filter
 * step; the history doesn't keep the smoothed Gaussians between calls, since a new measurement
 * changes every one of them.
 *
 * The steps live in a \c std::list from the allocator, one node per step, so recording never
 * moves the steps already recorded, and a growing history asks its allocator for one node per
 * push(), never for a larger block to copy into. Nothing else is allocated for a bounded space.
 *
 * \tparam Space_      the state space
 * \tparam Scalar_     scalar type, \c double by default
 * \tparam Allocator_  allocator for the steps, re-bound to the list's nodes (see \ref Allocation)
 *
 * \sa smoothStep(), PredictedJoint, KalmanFilter::predictJoint(), \ref Smoothers
 */
template <Space Space_, RealScalar Scalar_ = double, typename Allocator_ = DefaultAllocator>
class GaussianHistory
{
public:

    using Scalar = Scalar_;                                ///< the scalar type
    using SpaceType = Space_;                              ///< the state space
    using GaussianType = GaussianParams<Space_, Scalar_>;  ///< the smoothed %Gaussian's type
    using JointType = PredictedJoint<Space_, Scalar_>;     ///< what each step records
    using allocator_type = Allocator_;                     ///< the allocator the steps come from
    using Index = Eigen::Index;                            ///< Eigen's index type

    /** One recorded prediction and its smoothed %Gaussian. */
    struct Step
    {
        JointType joint;  ///< the prediction as predictJoint() returned it

        /** \f$ p(x_k \mid z_{1:N}) \f$ for the step's \f$ x_k \f$ (joint.filtered), as of the last
         * smooth(); a copy of joint.filtered until then
         */
        GaussianType smoothed;

        /** The step's Rauch-Tung-Striebel gain (internal::rtsGain()), made once at push(). */
        typename GaussianType::MatrixType gain;
    };

private:

    using StepAllocator = internal::ReBound<Allocator_, Step>;  ///< allocates the list's nodes
    using ListType = std::list<Step, StepAllocator>;            ///< the steps, oldest first

public:

    using const_iterator = typename ListType::const_iterator;  ///< iterates the steps, oldest first

    /** An empty history that keeps every step. */
    GaussianHistory() = default;

    /** An empty history that keeps every step, with nodes from \a allocator. */
    explicit GaussianHistory(Allocator_ const & allocator)
        : m_steps(StepAllocator(allocator))
    {}

    /** An empty history that keeps the newest \a lag steps, with nodes from \a allocator. */
    explicit GaussianHistory(Lag lag, Allocator_ const & allocator = {})
        : m_steps(StepAllocator(allocator))
        , m_lag{lag.steps}
    {
        EMRYS_FILTER_ASSERT(lag.steps > 0 && "a fixed-lag history keeps at least one step");
    }

    /** Copies the steps into nodes from \a other's allocator (see \ref Allocation). */
    GaussianHistory(GaussianHistory const & other)
        : GaussianHistory{other, other.get_allocator()}
    {}

    /** Copies \a other's steps into nodes from \a allocator. */
    GaussianHistory(GaussianHistory const & other, Allocator_ const & allocator)
        : m_steps(other.m_steps, StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Takes \a other's steps. */
    GaussianHistory(GaussianHistory && other) = default;

    /** Moves \a other's steps into nodes from \a allocator, taking its nodes when the allocators
     * are equal.
     */
    GaussianHistory(GaussianHistory && other, Allocator_ const & allocator)
        : m_steps(std::move(other.m_steps), StepAllocator(allocator))
        , m_lag{other.m_lag}
    {}

    /** Copies \a other's steps and lag. */
    GaussianHistory &
    operator=(GaussianHistory const & other) = default;

    /** Takes \a other's steps and lag, as \c std::list's move assignment does. */
    GaussianHistory &
    operator=(GaussianHistory && other) = default;

    ~GaussianHistory() = default;

    /** \returns the allocator the steps come from */
    [[nodiscard]] allocator_type
    get_allocator() const
    {
        return allocator_type(m_steps.get_allocator());
    }

    /** \returns the most steps a fixed-lag history keeps, or nothing if it keeps every step */
    [[nodiscard]] std::optional<Index>
    lag() const
    {
        return m_lag > 0 ? std::optional<Index>{m_lag} : std::nullopt;
    }

    /** \returns the number of steps recorded, at most lag() */
    [[nodiscard]] Index
    size() const
    {
        return Index(m_steps.size());
    }

    /** \returns whether no step is recorded */
    [[nodiscard]] bool
    empty() const
    {
        return m_steps.empty();
    }

    /** \returns the oldest step kept; the history must not be empty */
    [[nodiscard]] Step const &
    front() const
    {
        EMRYS_FILTER_ASSERT(!empty() && "the history is empty");
        return m_steps.front();
    }

    /** \returns the newest step; the history must not be empty */
    [[nodiscard]] Step const &
    back() const
    {
        EMRYS_FILTER_ASSERT(!empty() && "the history is empty");
        return m_steps.back();
    }

    /** \returns an iterator to the oldest step */
    [[nodiscard]] const_iterator
    begin() const
    {
        return m_steps.begin();
    }

    /** \returns an iterator past the newest step */
    [[nodiscard]] const_iterator
    end() const
    {
        return m_steps.end();
    }

    /** Records \a joint as the newest step. A full fixed-lag history drops its oldest step and
     * reuses its node.
     */
    void
    push(JointType joint)
    {
        if (m_lag > 0 && size() == m_lag)
        {
            m_steps.splice(m_steps.end(), m_steps, m_steps.begin());
            Step & step = m_steps.back();
            step.joint = std::move(joint);
            step.smoothed = step.joint.filtered;
            step.gain = internal::rtsGain(step.joint);
        }
        else
        {
            GaussianType smoothed = joint.filtered;
            typename GaussianType::MatrixType gain = internal::rtsGain(joint);
            m_steps.push_back(Step{std::move(joint), std::move(smoothed), std::move(gain)});
        }
    }

    /** Fills every step's smoothed %Gaussian by smoothStep(), from the newest back, given
     * \a latest, the filter's state after the newest step: its posterior \f$ p(x_N \mid z_{1:N}) \f$,
     * which is its own smoothed %Gaussian.
     */
    void
    smooth(GaussianType const & latest)
    {
        GaussianType const * next{&latest};
        for (auto step = m_steps.rbegin(); step != m_steps.rend(); ++step)
        {
            step->smoothed = smoothStep(step->joint, step->gain, *next);
            next = &step->smoothed;
        }
    }

    /** Drops every step, keeping the lag. */
    void
    clear()
    {
        m_steps.clear();
    }

private:

    ListType m_steps;  ///< the steps, oldest first
    Index m_lag = 0;   ///< the most steps kept, or zero to keep every step
};

}  // namespace Emrys

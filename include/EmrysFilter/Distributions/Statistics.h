#pragma once

/** \file
 * Weighted statistics over points, all in `internal`, which the containers of weighted elements
 * (GaussianMixtureParams, ParticleSet, the Rao-Blackwellized states) call: the weighted
 * covariance, weighted moments with a tangent-space mean in a nonlinear space, taken one point at a
 * time so that they allocate nothing sized by the count [allocation], the moment-matched mixture of
 * Gaussians, and the effective sample size, the
 * moments and the effective sample size also under an execution policy. Read Component.h next.
 */

#include <utility>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys::internal
{

/** Weighted covariance \f$ \sum_i w_i d_i d_i^T \f$ of the columns of \a deviations with
 * linear weights \a w. Exactly symmetric.
 *
 * The biased weighted sample covariance, with the weights already normalized to sum to one; see
 * [weighted sample covariance](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance).
 */
template <typename MatrixType, typename DeviationsDerived, typename WeightsDerived>
[[nodiscard]] MatrixType
weightedCovariance(Eigen::MatrixBase<DeviationsDerived> const & deviations, Eigen::ArrayBase<WeightsDerived> const & w)
{
    using Scalar = typename MatrixType::Scalar;
    using Deviations = MatrixOf<
        Scalar,
        MatrixType::RowsAtCompileTime,
        DeviationsDerived::ColsAtCompileTime,
        MatrixType::MaxRowsAtCompileTime,
        DeviationsDerived::MaxColsAtCompileTime>;
    Deviations const scaled = deviations * w.sqrt().matrix().asDiagonal();
    auto cov = MatrixType{deviations.rows(), deviations.rows()};
    if constexpr (MatrixType::MaxRowsAtCompileTime == 1)
    {
        // rankUpdate() would treat the single-row matrix as a column vector.
        cov.setConstant(scaled.squaredNorm());
    }
    else
    {
        cov.setZero();
        cov.template selfadjointView<Eigen::Lower>().rankUpdate(scaled);
        for (Eigen::Index j = 1; j < cov.cols(); ++j)
        {
            for (Eigen::Index i = 0; i < j; ++i)
            {
                cov(i, j) = cov(j, i);
            }
        }
    }
    return cov;
}

/** Copies the lower triangle of \a cov into its upper one, so that it is exactly symmetric, by a
 * triangular assignment, which Eigen unrolls at a small fixed size.
 */
template <typename Derived>
void
mirrorLower(Eigen::MatrixBase<Derived> & cov)
{
    cov.template triangularView<Eigen::StrictlyUpper>() = cov.transpose();
}

/** The tangent-space mean of \a count points read one at a time, \a meanAt(i) with log-weight
 * \a logWeightAt(i), and the lower triangle of their weighted covariance about it: the body of
 * weightedMomentsIn() and mixtureMomentsIn(). It gathers no points, so it allocates nothing sized
 * by the count. Points of zero weight are left out, whatever their values.
 *
 * In a space with plain differences the mean is \f$ \sum_i w_i x_i \f$. Otherwise it is the
 * fixed-point iteration for the Karcher (Fréchet) mean,
 * \f$ r \leftarrow r \boxplus \sum_i w_i (x_i \boxminus r) \f$, of
 * \ref Pennec2006 "Pennec (2006)", as given in \ref Hertzberg2013 "Hertzberg et al. (2013)",
 * §3.4, equations (17) and (18), started from the heaviest point and stopped after two steps,
 * where Hertzberg et al. iterate until the step falls below a threshold (§4.2, Table 3). The
 * covariance is the biased weighted sample covariance of the final tangent-space differences, each
 * added as a rank-one update.
 */
template <typename MomentsType, typename Space, typename MeanAt, typename LogWeightAt>
[[nodiscard]] MomentsType
pointMomentsIn(
    Space const & space,
    Eigen::Index count,
    MeanAt meanAt,
    LogWeightAt logWeightAt,
    typename MomentsType::Scalar logTotal
)
{
    using Scalar = typename MomentsType::Scalar;
    using DifferenceVectorType = typename MomentsType::DifferenceVectorType;
    auto const weight = [&](Eigen::Index i) {
        return Eigen::numext::exp(Scalar(logWeightAt(i)) - logTotal);
    };
    auto const skip = [&](Eigen::Index i) {
        return Scalar(logWeightAt(i)) == -Eigen::NumTraits<Scalar>::infinity();
    };
    auto result = MomentsType{valueDimension(space), differenceDimension(space)};
    auto & cov = result.covarianceLower();
    cov.setZero();
    // Eigen's rank update loops at run time whatever the size, and so does a product assigned to a
    // triangle, which Eigen sends to its rank update; at a fixed size the lazy outer product, of
    // the same terms w v_j v_i, is assigned to the triangle coefficient by coefficient with the
    // sizes fixed.
    auto const addCov = [&](DifferenceVectorType const & deviation, Scalar w) {
        if constexpr (DifferenceVectorType::SizeAtCompileTime != Eigen::Dynamic)
        {
            cov.template triangularView<Eigen::Lower>() += deviation.lazyProduct((w * deviation).transpose());
        }
        else
        {
            cov.template selfadjointView<Eigen::Lower>().rankUpdate(deviation, w);
        }
    };
    if constexpr (hasPlainDifferences<Space>)
    {
        result.mean().setZero();
        for (Eigen::Index i = 0; i < count; ++i)
        {
            if (!skip(i))
            {
                result.mean() += weight(i) * meanAt(i);
            }
        }
        for (Eigen::Index i = 0; i < count; ++i)
        {
            if (!skip(i))
            {
                addCov(DifferenceVectorType{meanAt(i) - result.mean()}, weight(i));
            }
        }
        normalizeValues(space, result.mean());
    }
    else
    {
        Eigen::Index reference = 0;
        for (Eigen::Index i = 1; i < count; ++i)
        {
            if (logWeightAt(i) > logWeightAt(reference))
            {
                reference = i;
            }
        }
        result.mean() = meanAt(reference);
        auto shift = DifferenceVectorType{};
        for (int pass = 0; pass < 2; ++pass)
        {
            shift.setZero(differenceDimension(space));
            for (Eigen::Index i = 0; i < count; ++i)
            {
                if (!skip(i))
                {
                    shift += weight(i) * Emrys::minus(space, meanAt(i), result.mean());
                }
            }
            if (pass == 1)
            {
                for (Eigen::Index i = 0; i < count; ++i)
                {
                    if (!skip(i))
                    {
                        addCov(DifferenceVectorType{Emrys::minus(space, meanAt(i), result.mean()) - shift}, weight(i));
                    }
                }
            }
            result.mean() = Emrys::plus(space, result.mean(), shift);
        }
    }
    return result;
}

/** Weighted mean and (biased) covariance of the columns of \a states in \a space, with weights
 * \f$ e^{\mathrm{logWeights}_i - \mathrm{logTotal}} \f$: a tangent-space mean, as pointMomentsIn()
 * computes it, one point at a time, so that it allocates nothing sized by the count. For a space
 * with plain differences, such as Linear, the mean is the weighted mean, normalized. For periodic
 * coordinates it is accurate while their spread is well under half the period. The covariance is
 * exactly symmetric.
 */
template <typename MomentsType, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] MomentsType
weightedMomentsIn(
    Space const & space,
    Eigen::MatrixBase<StatesDerived> const & states,
    Eigen::ArrayBase<WeightsDerived> const & logWeights,
    typename MomentsType::Scalar logTotal
)
{
    MomentsType result = pointMomentsIn<MomentsType>(
        space,
        states.cols(),
        [&](Eigen::Index i) {
            return states.col(i);
        },
        [&](Eigen::Index i) {
            return logWeights.derived()(i);
        },
        logTotal
    );
    mirrorLower(result.covarianceLower());
    return result;
}

/** weightedMomentsIn() under \a policy. Under Sequential it is weightedMomentsIn() itself;
 * otherwise each sum over the points is a reduceRanges() over chunks of them, in the same passes:
 * the heaviest point as the reference, the mean shifted twice, and the covariance of the
 * differences from the second pass's starting point less its shift. Each chunk keeps its
 * whole-array Eigen expressions, and the chunks' sums are added in order, so the result matches
 * Sequential's to rounding and is the same at every thread count.
 */
template <typename MomentsType, ExecutionPolicy P, typename Space, typename StatesDerived, typename WeightsDerived>
[[nodiscard]] MomentsType
weightedMomentsIn(
    P && policy,
    Space const & space,
    Eigen::MatrixBase<StatesDerived> const & states,
    Eigen::ArrayBase<WeightsDerived> const & logWeights,
    typename MomentsType::Scalar logTotal
)
{
    if constexpr (isSequential<P>)
    {
        return weightedMomentsIn<MomentsType>(space, states, logWeights, logTotal);
    }
    else
    {
        using Scalar = typename MomentsType::Scalar;
        using VectorType = typename MomentsType::VectorType;
        using DifferenceVectorType = typename MomentsType::DifferenceVectorType;
        using MatrixType = typename MomentsType::MatrixType;
        using Deviations = SampleSet<DifferenceSpace<typename MomentsType::SpaceType>, Scalar>;
        using Weights = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
        Eigen::Index const count = states.cols();
        auto const weightsOf = [&](Eigen::Index begin, Eigen::Index length) {
            return Weights{(logWeights.derived().segment(begin, length) - logTotal).exp()};
        };
        auto const add = [](auto a, auto const & b) {
            a += b;
            return a;
        };
        // The weighted covariance of the differences of the points from center, less shift.
        auto const covarianceAbout = [&](auto const & center, DifferenceVectorType const & shift) {
            return reduceRanges(
                policy,
                count,
                [&](Eigen::Index begin, Eigen::Index length) {
                    Deviations deviations = Emrys::minus(space, states.middleCols(begin, length), center);
                    deviations.colwise() -= shift;
                    return weightedCovariance<MatrixType>(deviations, weightsOf(begin, length));
                },
                add
            );
        };
        auto result = MomentsType{valueDimension(space), differenceDimension(space)};
        if constexpr (hasPlainDifferences<Space>)
        {
            result.mean() = reduceRanges(
                policy,
                count,
                [&](Eigen::Index begin, Eigen::Index length) {
                    return VectorType{states.middleCols(begin, length) * weightsOf(begin, length).matrix()};
                },
                add
            );
            result.covarianceLower() =
                covarianceAbout(result.mean(), DifferenceVectorType::Zero(differenceDimension(space)));
            normalizeValues(space, result.mean());
        }
        else
        {
            // The first of the heaviest points, as Eigen's maxCoeff() finds it.
            struct Heaviest
            {
                Scalar logWeight;    ///< its log-weight
                Eigen::Index index;  ///< its index
            };

            Heaviest const heaviest = reduceRanges(
                policy,
                count,
                [&](Eigen::Index begin, Eigen::Index length) {
                    Eigen::Index index = 0;
                    Scalar const logWeight = logWeights.derived().segment(begin, length).maxCoeff(&index);
                    return Heaviest{logWeight, begin + index};
                },
                [](Heaviest const & a, Heaviest const & b) {
                    return b.logWeight > a.logWeight ? b : a;
                }
            );
            auto const shiftFrom = [&](VectorType const & center) {
                return reduceRanges(
                    policy,
                    count,
                    [&](Eigen::Index begin, Eigen::Index length) {
                        return DifferenceVectorType{
                            Emrys::minus(space, states.middleCols(begin, length), center)
                            * weightsOf(begin, length).matrix()
                        };
                    },
                    add
                );
            };
            VectorType const reference = states.col(heaviest.index);
            VectorType const center = Emrys::plus(space, reference, shiftFrom(reference));
            DifferenceVectorType const shift = shiftFrom(center);
            result.mean() = Emrys::plus(space, center, shift);
            result.covarianceLower() = covarianceAbout(center, shift);
        }
        return result;
    }
}

/** weightedMomentsIn() of \a count points read one at a time, \a meanAt(i) with log-weight
 * \a logWeightAt(i), plus \f$ \sum_i \hat w_i \Sigma_i \f$ with \a covAt(i) (only its lower
 * triangle read), without gathering the points: the moment-matched mixture of \a count
 * Gaussians. It allocates nothing sized by the count. The covariance is exactly symmetric.
 *
 * The mean and covariance of a [mixture](https://en.wikipedia.org/wiki/Mixture_distribution#Moments):
 * \f$ \mu = \sum_i w_i \mu_i \f$ and
 * \f$ \Sigma = \sum_i w_i \left(\Sigma_i + (\mu_i - \mu)(\mu_i - \mu)^T\right) \f$, with the
 * means combined as pointMomentsIn() combines points.
 */
template <typename MomentsType, typename Space, typename MeanAt, typename CovAt, typename LogWeightAt>
[[nodiscard]] MomentsType
mixtureMomentsIn(
    Space const & space,
    Eigen::Index count,
    MeanAt meanAt,
    CovAt covAt,
    LogWeightAt logWeightAt,
    typename MomentsType::Scalar logTotal
)
{
    using Scalar = typename MomentsType::Scalar;
    MomentsType result = pointMomentsIn<MomentsType>(space, count, meanAt, logWeightAt, logTotal);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        if (Scalar(logWeightAt(i)) != -Eigen::NumTraits<Scalar>::infinity())
        {
            result.covarianceLower().template triangularView<Eigen::Lower>() +=
                Eigen::numext::exp(Scalar(logWeightAt(i)) - logTotal) * covAt(i);
        }
    }
    mirrorLower(result.covarianceLower());
    return result;
}

/** Effective sample size \f$ 1 / \sum_i \hat w_i^2 \f$ of the given log-weights, with
 * \f$ \hat w_i \f$ the normalized weights: the estimate of \ref Kong1994 "Kong, Liu, and Wong
 * (1994)", as given in \ref Bergman1999 "Bergman (1999)", §6.4.1, equation (6.28), that
 * \ref Liu1998 "Liu and Chen (1998)" use to decide when to resample.
 */
template <typename WeightsDerived>
[[nodiscard]] typename WeightsDerived::Scalar
ess(Eigen::ArrayBase<WeightsDerived> const & logWeights, typename WeightsDerived::Scalar logTotal)
{
    using Scalar = typename WeightsDerived::Scalar;
    return Scalar(1) / (logWeights - logTotal).exp().square().sum();
}

/** ess() under \a policy: the sum of the squared weights is a reduceRanges() over
 * chunks, so it matches Sequential's to rounding. Under Sequential it is ess().
 */
template <ExecutionPolicy P, typename WeightsDerived>
[[nodiscard]] typename WeightsDerived::Scalar
ess(P && policy, Eigen::ArrayBase<WeightsDerived> const & logWeights, typename WeightsDerived::Scalar logTotal)
{
    using Scalar = typename WeightsDerived::Scalar;
    if constexpr (isSequential<P>)
    {
        return ess(logWeights, logTotal);
    }
    else
    {
        Scalar const sumOfSquares = reduceRanges(
            std::forward<P>(policy),
            logWeights.size(),
            [&](Eigen::Index begin, Eigen::Index length) {
                return Scalar((logWeights.derived().segment(begin, length) - logTotal).exp().square().sum());
            },
            [](Scalar a, Scalar b) {
                return a + b;
            }
        );
        return Scalar(1) / sumOfSquares;
    }
}

}  // namespace Emrys::internal

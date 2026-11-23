#pragma once

// The distance between two filters' estimates, as the agreement tests measure it, for the tests
// and the runners against other libraries (mherb, tinyekf, and opencv), which check each pair
// before they time it and print how far the pairs that do not agree are from the estimate they are
// compared with.

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace Ranging
{

/// The distance of a mean and a covariance from a reference's, each relative to the size of the
/// reference's.
struct Distance
{
    double mean;  ///< of the means
    double cov;   ///< of the covariances' lower triangles, the part EmrysFilter guarantees
};

/// The distance of the mean \a mean and the covariance \a cov from \a refMean and \a refCov.
template <typename Mean, typename Cov, typename RefMean, typename RefCov>
[[nodiscard]] Distance
distance(Mean const & mean, Cov const & cov, RefMean const & refMean, RefCov const & refCov)
{
    auto const covDifference = (cov.template cast<double>() - refCov.template cast<double>()).eval();
    return Distance{
        (mean.template cast<double>() - refMean.template cast<double>()).norm() / std::max(1.0, double(refMean.norm())),
        covDifference.template triangularView<Eigen::Lower>().toDenseMatrix().norm()
            / std::max(1.0, double(refCov.norm()))
    };
}

/// The larger of \a a and \a b in each part.
[[nodiscard]] inline Distance
largerOf(Distance const & a, Distance const & b)
{
    return Distance{std::max(a.mean, b.mean), std::max(a.cov, b.cov)};
}

/// The covariance of \a filter as distance() takes it: the lower triangle that an EmrysFilter
/// filter guarantees, or the whole matrix of another library's.
template <typename Filter>
[[nodiscard]] auto
covarianceOf(Filter const & filter)
{
    if constexpr (requires { filter.covarianceLower(); })
    {
        return filter.covarianceLower();
    }
    else
    {
        return filter.covariance();
    }
}

/// The largest distance of \a filter from \a reference over the first \a steps steps of a run,
/// where `step(filter, k)` and `referenceStep(reference, k)` run step k on each.
template <typename Filter, typename Step, typename Reference, typename ReferenceStep>
[[nodiscard]] Distance
largestDistance(
    Filter & filter,
    Step const & step,
    Reference & reference,
    ReferenceStep const & referenceStep,
    std::size_t steps
)
{
    auto worst = Distance{0, 0};
    for (std::size_t k = 0; k < steps; ++k)
    {
        referenceStep(reference, k);
        step(filter, k);
        worst =
            largerOf(worst, distance(filter.mean(), covarianceOf(filter), reference.mean(), covarianceOf(reference)));
    }
    return worst;
}

/// The name of \a Scalar, for the tables.
template <typename Scalar>
[[nodiscard]] char const *
scalarName()
{
    return std::is_same_v<Scalar, float> ? "float" : "double";
}

}  // namespace Ranging

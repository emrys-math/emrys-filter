#pragma once

// The distance between two filters' estimates, as the agreement tests measure it, for the runners
// against other libraries (mherb_benchmark.cpp, tinyekf_benchmark.cpp), which check agreement
// before they time and print the distances of the pairs they don't time.

#include <Eigen/Core>

#include <algorithm>
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

/// The name of \a Scalar, for the tables.
template <typename Scalar>
[[nodiscard]] char const *
scalarName()
{
    return std::is_same_v<Scalar, float> ? "float" : "double";
}

}  // namespace Ranging

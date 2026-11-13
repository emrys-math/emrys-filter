#pragma once

// A sigma-point set of the user's own, for tests/Models/sigma_points.cpp: the symmetric set of 2n + 1 points, the
// mean with weight W0 and the 2n points at sqrt(n / (1 - W0)) along the root's columns, each of weight
// (1 - W0) / (2n), which is Unscented<1, 0, kappa> with kappa = n W0 / (1 - W0). It meets PointSet by its two static
// members alone, with nothing of the library's specialized, so it is a TransformMethod a sensor or a filter takes
// [sigma-point-sets], [user-extension-points]. usage.md, "Writing a sigma-point set", shows it.

#include <Eigen/Core>

namespace EmrysTest
{

// The symmetric sigma-point set with center weight CenterWeight, in [0, 1).
template <double CenterWeight>
struct SymmetricSet
{
    static_assert(CenterWeight >= 0 && CenterWeight < 1, "the center weight must be in [0, 1)");

    // The points over n > 0 coordinates: the center and two per coordinate.
    [[nodiscard]] static constexpr Eigen::Index
    pointCount(Eigen::Index n)
    {
        return 2 * n + 1;
    }

    // Fills points (n by 2n + 1) and weights (2n + 1), both sized by the caller, and returns the center, the point the
    // output mean is taken about.
    template <typename Points, typename Weights>
    static Eigen::Index
    unitPoints(Eigen::Index n, Points & points, Weights & weights)
    {
        using Scalar = typename Points::Scalar;
        Scalar const center = Scalar(CenterWeight);
        Scalar const radius = Eigen::numext::sqrt(Scalar(n) / (Scalar(1) - center));
        points.setZero();
        for (Eigen::Index i = 0; i < n; ++i)
        {
            points(i, 1 + i) = radius;
            points(i, 1 + n + i) = -radius;
        }
        weights(0) = center;
        weights.tail(2 * n).setConstant((Scalar(1) - center) / Scalar(2 * n));
        return 0;
    }
};

}  // namespace EmrysTest

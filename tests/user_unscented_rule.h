#pragma once

// A rule of the user's own for the unscented parameters, for tests/Models/sigma_points.cpp: alpha = 1, beta = 0, and
// kappa = max(0, 3 - n) over n coordinates, so that the points match a Gaussian's fourth moment along each column of
// the root while the center weight stays nonnegative, and so FourthMomentUnscented up to three coordinates and the
// cubature rule's Unscented<1, 0, 0> from three on. It meets UnscentedRule by its static member alone, with nothing of
// the library's specialized, so it is a TransformMethod a sensor or a filter takes [sigma-point-sets],
// [user-extension-points]. usage.md, "Choosing the unscented parameters by the dimension", shows it.

#include <EmrysFilter/Models/Transform.h>

#include <Eigen/Core>

namespace EmrysTest
{

// The unscented parameters with kappa = max(0, 3 - n).
struct ClampedFourthMoment
{
    // The parameters over n > 0 coordinates, in the transform's scalar; a constant expression for a constant n.
    template <typename Scalar>
    [[nodiscard]] static constexpr Emrys::UnscentedParams<Scalar>
    params(Eigen::Index n)
    {
        return {Scalar(1), Scalar(0), n < 3 ? Scalar(3 - n) : Scalar(0)};
    }
};

}  // namespace EmrysTest

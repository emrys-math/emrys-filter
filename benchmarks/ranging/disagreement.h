#pragma once

// The check behind a marked pair [benchmarks]: where another library's EKF in float does not agree
// with EmrysFilter's, the pair is timed all the same and its numbers are marked, once a test shows
// that the difference is the other library's. The agreement tests against mherb/kalman, TinyEKF,
// and OpenCV share this check.

#include "distance.h"
#include "emrys_ranging_kinds.h"
#include "scenario.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

namespace Ranging
{

/// Expects that, of a pair of float EKFs, only the other library's strays, over the simulated run
/// \a run of \a scenario. EmrysFilter's float EKF stays within agreementTolerance of its double
/// EKF, the estimate with which every implementation agrees in double (the agreement tests). The
/// other library's, \a other, moved by `otherStep(other, k)`, goes beyond that tolerance, without
/// which the pair needs no mark, and stays within disagreementBound, so that it is still a filter
/// of the same problem.
template <int Dimension, int AnchorCount, typename Other, typename OtherStep>
void
expectOnlyTheOtherStrays(
    Scenario<Dimension, AnchorCount> const & scenario,
    Trajectory<Dimension, AnchorCount> const & run,
    Other & other,
    OtherStep const & otherStep
)
{
    using Reference = EmrysRangingKind<Dimension, AnchorCount, FilterKind::Ekf>;
    using EmrysFloat = EmrysRangingKind<Dimension, AnchorCount, FilterKind::Ekf, float>;
    auto const floatRanges = rangesIn<float>(run);
    std::size_t const steps = run.ranges.size();
    auto const referenceStep = [&](Reference & filter, std::size_t k) {
        filter.predict();
        filter.update(run.ranges[k]);
    };
    auto const emrysStep = [&](EmrysFloat & filter, std::size_t k) {
        filter.predict();
        filter.update(floatRanges[k]);
    };
    auto emrys = EmrysFloat{scenario};
    auto emrysReference = Reference{scenario};
    Distance const emrysDistance = largestDistance(emrys, emrysStep, emrysReference, referenceStep, steps);
    auto otherReference = Reference{scenario};
    Distance const otherDistance = largestDistance(other, otherStep, otherReference, referenceStep, steps);
    double const tolerance = agreementTolerance<float>;
    EXPECT_LT(emrysDistance.mean, tolerance);
    EXPECT_LT(emrysDistance.cov, tolerance);
    EXPECT_GT(std::max(otherDistance.mean, otherDistance.cov), tolerance);
    EXPECT_LT(otherDistance.mean, disagreementBound);
    EXPECT_LT(otherDistance.cov, disagreementBound);
}

}  // namespace Ranging

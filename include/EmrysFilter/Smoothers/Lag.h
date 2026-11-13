#pragma once

/** \file
 * Lag, the number of steps a fixed-lag history keeps, which every history of the Smoothers layer
 * takes (decisions.md [smoothing]). Read GaussianHistory.h next.
 */

#include <Eigen/Core>

namespace Emrys
{

/** \ingroup Smoothers
 * The number of steps a fixed-lag history keeps.
 */
struct Lag
{
    Eigen::Index steps;  ///< the most steps, at least one
};

}  // namespace Emrys

// The particle filter as a program would use it, one translation unit of the compile-cost table:
// COMPILE_COST_EMRYS chooses the EmrysFilter side (1), with COMPILE_COST_KIND its likelihood, or
// the plain-Eigen one (0). filterRun() has external linkage, so the compiler emits all of it
// although nothing calls it.

#include "scenario.h"

#if COMPILE_COST_EMRYS
#    include "emrys_particles.h"
#else
#    include "eigen_particles.h"
#endif

namespace Particles
{

/// Filters every range of \a run from the particles \a initial and returns a sum of the final
/// log weights, which keeps the steps alive under optimization.
double
filterRun(
    Scenario<2, 4> const & scenario,
    Ranging::Trajectory<2, 4> const & run,
    Eigen::Matrix<double, Scenario<2, 4>::StateSize, Eigen::Dynamic> const & initial
)
{
#if COMPILE_COST_EMRYS
    auto filter = EmrysParticles<2, 4, Likelihood::COMPILE_COST_KIND>{scenario, initial, 3};
#else
    auto filter = EigenParticles<2, 4>{scenario, initial, 3};
#endif
    for (auto const & z : run.ranges)
    {
        filter.predict();
        filter.update(z);
        filter.housekeep();
    }
#if COMPILE_COST_EMRYS
    return filter.state().logWeights().sum();
#else
    return filter.logWeights().sum();
#endif
}

}  // namespace Particles

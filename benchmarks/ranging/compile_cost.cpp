// The ranging filter as a program would use it, one translation unit of the compile-cost table:
// COMPILE_COST_EMRYS chooses the EmrysFilter side (1) or the plain-Eigen one (0), and
// COMPILE_COST_KIND the linearization. filterRun() has external linkage, so the compiler emits all
// of it although nothing calls it.

#include "scenario.h"

#if COMPILE_COST_EMRYS
#    include "emrys_ranging.h"
#else
#    include "eigen_ranging.h"
#endif

namespace Ranging
{

/// Filters every range of \a run and returns a sum of the final mean, which keeps the steps alive
/// under optimization.
double
filterRun(Scenario<2, 4> const & scenario, Trajectory<2, 4> const & run)
{
#if COMPILE_COST_EMRYS
    auto filter = EmrysRanging<2, 4, Jacobian::COMPILE_COST_KIND>{scenario};
#else
    auto filter = EigenRanging<2, 4, Jacobian::COMPILE_COST_KIND>{scenario};
#endif
    for (auto const & z : run.ranges)
    {
        filter.predict();
        filter.update(z);
    }
#if COMPILE_COST_EMRYS
    return filter.state().mean().sum();
#else
    return filter.mean().sum();
#endif
}

}  // namespace Ranging

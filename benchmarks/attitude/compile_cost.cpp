// The attitude filter as a program would use it, one translation unit of the compile-cost table:
// COMPILE_COST_EMRYS chooses the EmrysFilter side (1) or the plain-Eigen one (0), and
// COMPILE_COST_KIND the linearization. filterRun() has external linkage, so the compiler emits all
// of it although nothing calls it.

#include "scenario.h"

#if COMPILE_COST_EMRYS
#    include "emrys_attitude.h"
#else
#    include "eigen_attitude.h"
#endif

namespace Orientation
{

/// Filters every reading of \a run and returns a sum of the final estimate, which keeps the steps
/// alive under optimization.
double
filterRun(Scenario const & scenario, Trajectory const & run)
{
#if COMPILE_COST_EMRYS
    auto filter = EmrysAttitude<Jacobian::COMPILE_COST_KIND>{scenario};
#else
    auto filter = EigenAttitude<Jacobian::COMPILE_COST_KIND>{scenario};
#endif
    for (std::size_t k = 0; k < run.gyro.size(); ++k)
    {
        filter.predict(run.gyro[k]);
        filter.update(run.gravity[k], run.magneticField[k]);
    }
#if COMPILE_COST_EMRYS
    return filter.state().mean().sum();
#else
    return filter.attitude().coeffs().sum() + filter.bias().sum();
#endif
}

}  // namespace Orientation

// The square-root, information, and square-root information filters under the Iterated method of
// Models/Transform.h, through the form steps of Models/FormSteps.h, over a direction measured on
// a unit-quaternion attitude: with Linearized they equal the Kalman filter (roadmap step 17b;
// decisions.md [iterated-update]), and with Cubature they agree with one another and with the Kalman filter to
// the transform's dependence on its root (roadmap step 17e; [sigma-point-sets]). The same comparisons over range
// and bearing are in iterated_forms.cpp; each model is a file of its own to shorten the build.

#include <EmrysFilter/Models/Transform.h>

#include <gtest/gtest.h>

#include "Models/iterated_problem.h"

namespace methods = Emrys::methods;

using namespace EmrysTest::IteratedProblem;

// The square-root, information, and square-root information filters under Iterated equal the
// Kalman filter under it, over several steps of a direction on an attitude, in state and
// log-likelihood, and report the same iterations.
TEST(IteratedUpdate, TheFormsEqualTheKalmanFilterOnAnAttitude)
{
    auto const method = methods::iterated(methods::linearized(), {.tolerance = 1e-10, .maxIterations = 20});
    expectFormsEqualKalman(makeAttitudeModel(), makeAttitudeInitial(), [&](int k) {
        return makeDirection(k, method);
    });
}

// The square-root, information, and square-root information filters under Iterated<Cubature>
// equal the Kalman filter to rounding over several steps of a direction on an attitude, as every
// form spreads its points by the same default axes ([sigma-point-axes]).
TEST(IteratedPosteriorLinearization, TheFormsAgreeOnAnAttitude)
{
    auto const method = methods::iterated(methods::cubature(), {.tolerance = 1e-10, .maxIterations = 20});
    expectFormsEqualKalman(makeAttitudeModel(), makeAttitudeInitial(), [&](int k) {
        return makeDirection(k, method);
    });
}

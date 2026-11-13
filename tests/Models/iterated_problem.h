#pragma once

// The problem shared by the tests of the Iterated method (tests/Models/iterated.cpp,
// iterated_posterior_linearization.cpp, iterated_forms.cpp, and iterated_forms_attitude.cpp;
// roadmap steps 17b and 17e, decisions.md [iterated-update] and [sigma-point-sets]): a poor prior far from a target at
// (4, 3), its precise range and bearing, their values and Jacobian by hand, the range-bearing measurement with its
// noise passed through the measurement function, and the runs that compare the square-root and information forms with
// the Kalman filter. The forms tests are split by model, since each form over each model is a large instantiation and
// the methods share most of it.

#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>

#include "Filters/form_filters.h"

#include <cmath>
#include <utility>

namespace EmrysTest::IteratedProblem
{

using namespace EmrysTest::FormFilters;

using Matrix2 = Eigen::Matrix2d;
using Jacobian = Eigen::Matrix<double, 2, 4>;

// A prior far from the target at (4, 3) and wide enough that one linearization at its mean misses
// the posterior's mode by much more than rounding.
inline GaussianParams<State>
makePoorPrior()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    cov(1, 3) = cov(3, 1) = -0.6;
    return {State{}, Vector4{2.0, 0.5, 1.0, 0.5}, cov};
}

inline Eigen::Vector2d const rangeAndBearing{5.0, std::atan2(3.0, 4.0)};

// Range and bearing from the origin of the state x, and their Jacobian in it, by hand.
inline std::pair<Eigen::Vector2d, Jacobian>
rangeBearingOf(Vector4 const & x)
{
    Eigen::Vector2d const p = x.head<2>();
    double const r = p.norm();
    Jacobian h = Jacobian::Zero();
    h.block<1, 2>(0, 0) = p.transpose() / r;
    h.block<1, 2>(1, 0) = Eigen::RowVector2d{-p(1), p(0)} / (r * r);
    return {Eigen::Vector2d{r, std::atan2(p(1), p(0))}, h};
}

// The noise of a range and bearing that passes through the function, an input of the call.
struct RangeBearingNoise : Linear<2>
{};

// Range and bearing with the noise passed through the measurement function, where it only adds.
template <typename Method>
auto
makeRangeBearingWithInputNoise(Eigen::Vector2d const & z, Method const & method)
{
    auto const rangeBearing = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            auto const w = block<RangeBearingNoise>(in, x);
            return Eigen::Matrix<T, 2, 1>{
                Eigen::numext::sqrt(T(p.squaredNorm())) + w(0),
                Eigen::numext::atan2(T(p(1)), T(p(0))) + w(1)
            };
        },
        reads(path<Position>, path<RangeBearingNoise>),
        outputSpace(RangeBearingSpace{})
    );
    return observe(
        method,
        rangeBearing,
        z,
        GaussianParams<RangeBearingNoise>{Eigen::Vector2d::Zero(), rangeBearingNoise}
    );
}

// Runs the Kalman filter and the square-root, information, and square-root information filters
// from initial under model over four steps, each updated by makeMeasurement(k), and checks that
// the forms equal the Kalman filter in state and log-likelihood, to tolerance, and report the same iterations.
void
expectFormsEqualKalman(auto const & model, auto const & initial, auto const & makeMeasurement, double tolerance = 1e-9)
{
    auto kalman = Emrys::KalmanFilter{model, initial};
    auto root = Emrys::SqrtKalmanFilter{model, Emrys::toSqrt(initial).value()};
    auto info = Emrys::InfoFilter{model, Emrys::toInfo(initial).value()};
    auto rootInfo = Emrys::SqrtInfoFilter{model, Emrys::toSqrtInfo(initial).value()};
    for (int k = 1; k <= 4; ++k)
    {
        kalman.predict(timeStep);
        ASSERT_TRUE(root.predict(timeStep));
        ASSERT_TRUE(info.predict(timeStep));
        ASSERT_TRUE(rootInfo.predict(timeStep));
        auto const expected = kalman.update(makeMeasurement(k));
        auto const fromRoot = root.update(makeMeasurement(k));
        auto const fromInfo = info.update(makeMeasurement(k));
        auto const fromRootInfo = rootInfo.update(makeMeasurement(k));
        ASSERT_TRUE(expected && fromRoot && fromInfo && fromRootInfo);
        for (auto const * branch :
             {&fromRoot->branches.front(), &fromInfo->branches.front(), &fromRootInfo->branches.front()})
        {
            EXPECT_EQ(branch->iterations, expected->branches.front().iterations);
            EXPECT_EQ(branch->converged, expected->branches.front().converged);
        }
        EXPECT_NEAR(fromRoot->logLikelihood(), expected->logLikelihood(), tolerance);
        EXPECT_NEAR(fromInfo->logLikelihood(), expected->logLikelihood(), tolerance);
        EXPECT_NEAR(fromRootInfo->logLikelihood(), expected->logLikelihood(), tolerance);
        expectSameState(root.state(), kalman.state(), tolerance);
        expectSameState(info.state(), kalman.state(), tolerance);
        expectSameState(rootInfo.state(), kalman.state(), tolerance);
    }
}

}  // namespace EmrysTest::IteratedProblem

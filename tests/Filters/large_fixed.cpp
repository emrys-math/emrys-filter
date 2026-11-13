// The filters on a state of a fixed size past Eigen's small-product threshold (decisions.md [allocation];
// the audit of fixed sizes of 2026-10-31): a ten-coordinate
// constant-velocity state with a five-coordinate position fix, whose products are all fixed in
// size and from eight on go through internal::multiply(), Eigen's fixed-size product, rather than
// its run-time kernels. Every form and method is checked against the Kalman recursion written in
// plain dynamic-size Eigen, so that a product evaluated into one of its own operands would show.
// Headers under test: Core/Multiply.h, Filters/KalmanFilter.h, Filters/SqrtKalmanFilter.h,
// Filters/SqrtInfoFilter.h, Filters/InfoFilter.h, Models/Measurement.h, Models/FormSteps.h,
// Models/ProcessModel.h, Models/Transform.h.

#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <gtest/gtest.h>

#include <cmath>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::GaussianParams;
using Emrys::InfoFilter;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::SqrtInfoFilter;
using Emrys::SqrtKalmanFilter;
using Emrys::toGaussian;
using Emrys::toInfo;
using Emrys::toSqrt;
using Emrys::toSqrtInfo;

namespace methods = Emrys::methods;

namespace
{

// The shapes Eigen sends to a run-time kernel, by its product_type_selector: a fixed inner
// dimension of eight or more, or two outer ones; not the inner and outer products, nor the small
// ones, nor any with a size known only at run time.
static_assert(!Emrys::internal::isLargeFixedProduct<Eigen::Matrix3d, Eigen::Matrix3d>);
static_assert(Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 12, 12>, Eigen::Matrix<double, 12, 1>>);
static_assert(Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 3, 12>, Eigen::Matrix<double, 12, 3>>);
static_assert(Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 12, 3>, Eigen::Matrix<double, 3, 12>>);
static_assert(!Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 12, 3>, Eigen::Matrix<double, 3, 2>>);
static_assert(!Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 1, 12>, Eigen::Matrix<double, 12, 1>>);
static_assert(!Emrys::internal::isLargeFixedProduct<Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 1, 12>>);
static_assert(!Emrys::internal::isLargeFixedProduct<Eigen::MatrixXd, Eigen::MatrixXd>);

constexpr int Half = 5;           // the position's coordinates, and the velocity's
constexpr double density = 0.1;   // the acceleration's spectral density
constexpr double timeStep = 1.0;  // the step

struct Position : Linear<Half>
{};

struct Velocity : Linear<Half>
{};

using State = Composite<Position, Velocity>;
using StateVector = Eigen::Matrix<double, 2 * Half, 1>;
using StateMatrix = Eigen::Matrix<double, 2 * Half, 2 * Half>;
using FixVector = Eigen::Matrix<double, Half, 1>;
using FixMatrix = Eigen::Matrix<double, Half, Half>;

GaussianParams<State>
makeInitial()
{
    StateMatrix cov = StateMatrix::Zero();
    for (int i = 0; i < Half; ++i)
    {
        cov(i, i) = 4.0 + 0.5 * i;
        cov(Half + i, Half + i) = 1.0 + 0.1 * i;
        cov(i, Half + i) = cov(Half + i, i) = 0.8;
    }
    StateVector mean = StateVector::Zero();
    mean.tail<Half>().setLinSpaced(0.2, 1.0);
    return {State{}, mean, cov};
}

FixMatrix
fixNoise()
{
    FixMatrix r = FixMatrix::Identity();
    for (int i = 0; i + 1 < Half; ++i)
    {
        r(i, i + 1) = r(i + 1, i) = 0.2;
    }
    return r;
}

template <typename Method, typename... Args>
auto
makeFix(FixVector const & z, Method const & method, Args const &... args)
{
    auto const position = [](auto const & in, auto const &, auto const & x) {
        return Emrys::block<Position>(in, x);
    };
    return observe(
        method,
        Emrys::measurementModel(
            position,
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<Half>>{FixVector::Zero(), fixNoise()})
        ),
        z,
        args...
    );
}

FixVector
fixAt(int k)
{
    auto z = FixVector{};
    for (int i = 0; i < Half; ++i)
    {
        z(i) = 0.6 * k * (i + 1) + 0.3 * std::cos(1.7 * k + i);
    }
    return z;
}

// The Kalman recursion of the constant-velocity model and the fix, in dynamic-size Eigen, whose
// products Eigen sends to its run-time kernels: k steps from the initial Gaussian.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
handRecursion(int steps)
{
    Eigen::Index const n = 2 * Half;
    Eigen::MatrixXd const identity = Eigen::MatrixXd::Identity(Half, Half);
    Eigen::MatrixXd f = Eigen::MatrixXd::Identity(n, n);
    f.topRightCorner(Half, Half) = timeStep * identity;
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(n, n);
    q << std::pow(timeStep, 3) / 3 * identity, timeStep * timeStep / 2 * identity, timeStep * timeStep / 2 * identity,
        timeStep * identity;
    q *= density;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(Half, n);
    h.leftCols(Half) = identity;
    Eigen::MatrixXd const r = fixNoise();
    auto const initial = makeInitial();
    Eigen::VectorXd x = initial.mean();
    Eigen::MatrixXd p = initial.covarianceLower();
    p.triangularView<Eigen::StrictlyUpper>() = p.transpose();
    for (int k = 1; k <= steps; ++k)
    {
        x = (f * x).eval();
        p = (f * p * f.transpose() + q).eval();
        Eigen::MatrixXd const s = h * p * h.transpose() + r;
        Eigen::MatrixXd const gain = p * h.transpose() * s.inverse();
        Eigen::VectorXd const z = fixAt(k);
        x = (x + gain * (z - h * x)).eval();
        p = ((Eigen::MatrixXd::Identity(n, n) - gain * h) * p).eval();
    }
    return {x, p};
}

// The state of a filter over a form, as a Gaussian, must equal the recursion's to tolerance.
template <typename Form>
void
expectRecursion(Form const & form, int steps, double tolerance)
{
    auto const gaussian = toGaussian(form);
    ASSERT_TRUE(gaussian) << "the form has no moments";
    auto const [mean, cov] = handRecursion(steps);
    Eigen::MatrixXd full = gaussian->covarianceLower();
    full.triangularView<Eigen::StrictlyUpper>() = full.transpose();
    EXPECT_LT((gaussian->mean() - mean).norm(), tolerance);
    EXPECT_LT((full - cov).norm(), tolerance);
}

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{density};
}

constexpr int steps = 3;

// The tolerance: the recursions differ only in the order of their rounding, over three steps of a
// covariance whose entries are at most about ten.
constexpr double tolerance = 1e-10;

}  // namespace

// The covariance form under Linearized, Unscented (exact on an affine model), and the Joseph
// update of each, which takes H from the blocks read (the Jacobian, or Sigma_Sy^T P_SS^-1).
TEST(LargeFixed, KalmanFilterMatchesTheRecursion)
{
    auto linearized = KalmanFilter{makeModel(), makeInitial()};
    auto unscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    auto josephLinearized = KalmanFilter{makeModel(), makeInitial()};
    auto josephUnscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    for (int k = 1; k <= steps; ++k)
    {
        linearized.predict(timeStep);
        unscented.predict(timeStep);
        josephLinearized.predict(timeStep);
        josephUnscented.predict(timeStep);
        ASSERT_TRUE(linearized.update(makeFix(fixAt(k), methods::linearized())));
        ASSERT_TRUE(unscented.update(makeFix(fixAt(k), methods::unscented())));
        ASSERT_TRUE(josephLinearized.update(makeFix(fixAt(k), methods::linearized(), Emrys::updates::joseph())));
        ASSERT_TRUE(josephUnscented.update(makeFix(fixAt(k), methods::unscented(), Emrys::updates::joseph())));
    }
    expectRecursion(linearized.state(), steps, tolerance);
    expectRecursion(unscented.state(), steps, tolerance);
    expectRecursion(josephLinearized.state(), steps, tolerance);
    expectRecursion(josephUnscented.state(), steps, tolerance);
}

// The iterated update, which on an affine measurement stops at the Kalman update.
TEST(LargeFixed, IteratedMatchesTheRecursion)
{
    auto iterated = KalmanFilter{makeModel(), makeInitial()};
    for (int k = 1; k <= steps; ++k)
    {
        iterated.predict(timeStep);
        ASSERT_TRUE(iterated.update(makeFix(fixAt(k), Emrys::methods::iterated(methods::linearized()))));
    }
    expectRecursion(iterated.state(), steps, tolerance);
}

// The square-root, square-root information, and information forms under Linearized and
// Unscented.
TEST(LargeFixed, FormFiltersMatchTheRecursion)
{
    auto root = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value()};
    auto rootUnscented = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value(), methods::unscented()};
    auto rootInfo = SqrtInfoFilter{makeModel(), toSqrtInfo(makeInitial()).value()};
    auto info = InfoFilter{makeModel(), toInfo(makeInitial()).value()};
    for (int k = 1; k <= steps; ++k)
    {
        ASSERT_TRUE(root.predict(timeStep));
        ASSERT_TRUE(rootUnscented.predict(timeStep));
        ASSERT_TRUE(rootInfo.predict(timeStep));
        ASSERT_TRUE(info.predict(timeStep));
        ASSERT_TRUE(root.update(makeFix(fixAt(k), methods::linearized())));
        ASSERT_TRUE(rootUnscented.update(makeFix(fixAt(k), methods::unscented())));
        ASSERT_TRUE(rootInfo.update(makeFix(fixAt(k), methods::linearized())));
        ASSERT_TRUE(info.update(makeFix(fixAt(k), methods::linearized())));
    }
    expectRecursion(root.state(), steps, tolerance);
    expectRecursion(rootUnscented.state(), steps, tolerance);
    expectRecursion(rootInfo.state(), steps, tolerance);
    expectRecursion(info.state(), steps, tolerance);
}

// multiply() of fixed operands past the threshold equals Eigen's product of the same values at
// run-time sizes; the two differ only in the order of the rounding.
TEST(LargeFixed, MultiplyMatchesTheRunTimeProduct)
{
    using Large = Eigen::Matrix<double, 12, 12>;
    Large const a = Large::Random();
    Eigen::Matrix<double, 12, 3> const b = Eigen::Matrix<double, 12, 3>::Random();
    Eigen::Matrix<double, 12, 3> const product = Emrys::internal::multiply(a, b);
    Eigen::MatrixXd const expected = Eigen::MatrixXd{a} * Eigen::MatrixXd{b};
    EXPECT_TRUE(product.isApprox(expected, 1e-14));
    Eigen::Matrix<double, 12, 1> const column = Emrys::internal::multiply(a, b.col(0).eval());
    EXPECT_TRUE(column.isApprox(expected.col(0), 1e-14));
}

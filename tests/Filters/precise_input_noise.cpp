// A measurement whose noise is an input of the call, read by its function, far more precise than the prior, through the
// filters over each of the four Gaussian forms (Filters/KalmanFilter.h, SqrtKalmanFilter.h, InfoFilter.h,
// SqrtInfoFilter.h), under the linear and the sigma-point methods (roadmap step 17u, [linearization-forms];
// [covariance-update-forms], [gaussian-forms]): the measurement's linearization keeps the input noise's
// part apart as a product, so the precise coordinate's posterior variance is right where recovering
// the noise by subtraction gave zero.

#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>

#include <gtest/gtest.h>

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

// The noise of the fix, an input of the call.
struct FixNoise : Linear<1>
{};

using Vector1 = Eigen::Matrix<double, 1, 1>;

double const big = 1e6;
double const rho = 0.999999;
double const precise = 1e-12;

// A prior of variance 1e6 in each position coordinate, correlated 0.999999 between them.
GaussianParams<State>
makePrior()
{
    Eigen::Matrix4d cov = Eigen::Matrix4d::Identity();
    cov.topLeftCorner<2, 2>() = big * Eigen::Matrix2d{{1.0, rho}, {rho, 1.0}};
    return {State{}, Eigen::Vector4d::Zero(), cov};
}

// A fix of the first position coordinate whose noise of variance 1e-12 is an input of the call that
// its function reads, under updates::joseph(), by method.
template <typename Method>
auto
makeFix(Method const & method)
{
    auto const fix = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{Emrys::block<Position>(in, x)(0) + Emrys::block<FixNoise>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<FixNoise>)
    );
    return observe(
        method,
        fix,
        Vector1{1.0},
        GaussianParams<FixNoise>{Vector1{0.0}, Vector1{precise}},
        updates::joseph()
    );
}

// The posterior variances of the two position coordinates, R P / (P + R) and P (1 - rho^2), from the
// filter's state, to a part in a thousand and a part in a million.
template <typename Form>
void
expectPrecise(Form const & state, char const * name)
{
    auto const gaussian = Emrys::toGaussian(state);
    ASSERT_TRUE(gaussian) << name;
    Eigen::Matrix4d const cov = gaussian->covarianceLower();
    EXPECT_NEAR(cov(0, 0), precise * big / (big + precise), 1e-3 * precise) << name;
    EXPECT_NEAR(cov(1, 1), big * (1 - rho * rho), 1e-6 * big * (1 - rho * rho)) << name;
}

}  // namespace

// Under each method, in each form: the covariance form's Joseph update takes R as the prediction's N,
// J_w C_w J_w^T or the sigma points' Psi_w^T Psi_w, never Sigma_yy less H P H^T, and the square-root and
// information forms fold the same linearization.
TEST(PreciseInputNoise, EveryFormKeepsIt)
{
    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto const check = [&](auto const & method, char const * name) {
        auto const fix = makeFix(method);
        auto covariance = Emrys::KalmanFilter{model, makePrior()};
        ASSERT_TRUE(covariance.update(fix)) << name;
        expectPrecise(covariance.state(), name);
        auto root = Emrys::SqrtKalmanFilter{model, Emrys::toSqrt(makePrior()).value()};
        ASSERT_TRUE(root.update(fix)) << name;
        expectPrecise(root.state(), name);
        auto information = Emrys::InfoFilter{model, Emrys::toInfo(makePrior()).value()};
        ASSERT_TRUE(information.update(fix)) << name;
        expectPrecise(information.state(), name);
        auto rootInformation = Emrys::SqrtInfoFilter{model, Emrys::toSqrtInfo(makePrior()).value()};
        ASSERT_TRUE(rootInformation.update(fix)) << name;
        expectPrecise(rootInformation.state(), name);
    };
    check(methods::linearized(), "linearized");
    check(methods::unscented(), "unscented");
    check(methods::cubature(), "cubature");
}

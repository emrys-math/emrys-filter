// PosteriorCramerRaoBound of Models/CramerRao.h: the information recursion of Tichavsky,
// Muravchik, and Nehorai (1998), Proposition 1, and its tangent-space form of Chahbazian et al.
// (2024). For an affine model it is the Kalman filter's covariance; for one truth measured without
// noise it is an extended Kalman filter's covariance with the filter on the truth, on a Linear state
// and on one with a unit quaternion; over several truths it averages the Jacobian products before
// it inverts, as the paper's expectations do; a fixed-step model steps it as its model does; and a
// singular Q, R, or prior is refused. The evidence for roadmap step 17g and decisions.md [cramer-rao-bound].

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/CramerRao.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

#include "../unequal_dimensions.h"

#include <gtest/gtest.h>

#include <array>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::BlockRows;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianError;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::path;
using Emrys::PosteriorCramerRaoBound;
using Emrys::reads;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

GaussianParams<State>
makeInitial(Vector4 const & mean)
{
    auto const variances = Vector4{4.0, 4.0, 1.0, 1.0};
    return {State{}, mean, Matrix4{variances.asDiagonal()}};
}

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{0.1};
}

Eigen::Matrix2d const fixNoise{{1.0, 0.2}, {0.2, 0.8}};

// A position fix, z = p + v.
auto
makeFixModel()
{
    return measurementModel(
        Eigen::Matrix2d::Identity(),
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), fixNoise})
    );
}

auto
makeFix(Eigen::Vector2d const & z)
{
    return observe(methods::linearized(), makeFixModel(), z);
}

constexpr double rangeVariance = 0.25;

// The range model at the anchor, h(p) = |p - a|, with variance rangeVariance.
template <typename P = Position>
auto
makeRangeModel(Eigen::Vector2d const & anchor)
{
    return measurementModel(
        [anchor]<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, 2, 1> const offset = block<P>(in, x) - anchor.cast<T>();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(offset.squaredNorm()))};
        },
        reads(path<P>),
        additiveNoise(
            GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{rangeVariance}}
        )
    );
}

// The range from the anchor, z = |p - a| + v.
auto
makeRange(Eigen::Vector2d const & anchor, double z)
{
    return observe(methods::linearized(), makeRangeModel(anchor), Eigen::Matrix<double, 1, 1>{z});
}

// The range from anchor to the position of truth, noise-free.
double
rangeOf(Eigen::Vector2d const & anchor, Vector4 const & truth)
{
    return (truth.head<2>() - anchor).norm();
}

// A position that slows as its cube, p' = p - c dt p^3 per coordinate, with white noise of density
// q on it: a model whose Jacobian, I - 3 c dt diag(p^2), depends on the state.
struct Cubic : Linear<2>
{};

using CubicState = Composite<Cubic>;

auto
cubicDrag(double rate, double density)
{
    return Emrys::processModel(
        [rate]<typename T>(auto const &, auto const &, auto const & x, auto dt) {
            Eigen::Matrix<T, 2, 1> y = x;
            for (Eigen::Index i = 0; i < 2; ++i)
            {
                T const value = x(i);
                y(i) -= T(rate * dt) * value * value * value;
            }
            return y;
        },
        reads(path<Cubic>),
        Emrys::writes(path<Cubic>),
        Emrys::continuousNoise(Emrys::on<Cubic>(density))
    );
}

// Random-walk attitude noise on the rotation vector of the quaternion at Path, which doesn't move.
template <typename Path>
auto
attitudeWalk(double spectralDensity)
{
    return Emrys::processModel(
        [](auto const &, auto const &, auto const & x, auto) {
            return x;
        },
        Emrys::reads(Emrys::path<Path>),
        Emrys::writes(Emrys::path<Path>),
        Emrys::continuousNoise(Emrys::on<Path>(spectralDensity))
    );
}

// The bound's covariance, which the tests expect to exist.
template <typename Bound>
auto
boundCovariance(Bound const & bound)
{
    using Matrix = typename Bound::MatrixType;
    auto const covariance = bound.covariance();
    EXPECT_TRUE(covariance.has_value());
    return covariance.value_or(Matrix{Matrix::Zero()});
}

}  // namespace

// For an affine model and a linear measurement the Jacobians are constant, so the recursion is the
// Kalman filter's on the information matrix, which the paper notes below eq. (36): over ten steps
// of prediction and a position fix, the bound's inverse is the filter's covariance. The truths are
// not read, and the tolerance is rounding through two inversions per step.
TEST(PosteriorCramerRaoBound, IsTheKalmanCovarianceForAnAffineModel)
{
    auto const initial = makeInitial(Vector4{0.0, 0.0, 1.0, 0.5});
    auto filter = KalmanFilter{makeModel(), initial};
    auto bound = PosteriorCramerRaoBound<State>::create(initial).value();
    Vector4 const truth = initial.mean();
    for (int k = 0; k < 10; ++k)
    {
        filter.predict(1.0);
        ASSERT_TRUE(bound.predict(makeModel(), truth, 1.0).has_value());
        auto const fix = makeFix(Eigen::Vector2d{double(k), 0.5 * k});
        ASSERT_TRUE(filter.update(fix).value().accepted);
        ASSERT_TRUE(bound.update(truth, makeFixModel()).has_value());
        EXPECT_TRUE(boundCovariance(bound).isApprox(filter.state().covarianceLower(), 1e-10)) << "step " << k;
    }
}

// With one truth that moves without noise and range measurements equal to its true ranges, a
// Kalman filter started on the truth stays on it, since every innovation is zero, so it linearizes
// at the truth as the bound does: the bound's inverse is the filter's covariance at every step. The
// first step is also checked by hand in plain Eigen: J = (Q + F P0 F^T)^-1 + sum h^T h / r.
TEST(PosteriorCramerRaoBound, OneTruthIsAnExtendedKalmanFilterOnTheTruth)
{
    auto const model = makeModel();
    auto const initial = makeInitial(Vector4{1.0, 2.0, 1.0, 0.5});
    auto const anchors = std::array{Eigen::Vector2d{10.0, 0.0}, Eigen::Vector2d{0.0, 10.0}};
    auto filter = KalmanFilter{model, initial};
    auto bound = PosteriorCramerRaoBound<State>::create(initial).value();
    Vector4 truth = initial.mean();
    for (int k = 0; k < 8; ++k)
    {
        Vector4 const before = truth;
        truth = Emrys::propagate(State{}, truth, model, 1.0);
        filter.predict(1.0);
        ASSERT_TRUE(bound.predict(model, before, 1.0).has_value());
        for (auto const & anchor : anchors)
        {
            auto const range = makeRange(anchor, rangeOf(anchor, truth));
            ASSERT_TRUE(filter.update(range).value().accepted);
            ASSERT_TRUE(bound.update(truth, makeRangeModel(anchor)).has_value());
        }
        ASSERT_TRUE(filter.state().mean().isApprox(truth, 1e-12));
        EXPECT_TRUE(boundCovariance(bound).isApprox(filter.state().covarianceLower(), 1e-9)) << "step " << k;
        if (k == 0)
        {
            Matrix4 f = Matrix4::Identity();
            f.topRightCorner<2, 2>() = Eigen::Matrix2d::Identity();
            Matrix4 const predicted =
                f * initial.covarianceLower() * f.transpose() + Emrys::FixedStepModel{model, State{}, 1.0}.noise();
            Matrix4 info = predicted.inverse();
            for (auto const & anchor : anchors)
            {
                Eigen::Matrix<double, 1, 4> h = Eigen::Matrix<double, 1, 4>::Zero();
                h.head<2>() = (truth.head<2>() - anchor).normalized().transpose();
                info += h.transpose() * h / rangeVariance;
            }
            EXPECT_TRUE(bound.info().isApprox(info, 1e-12));
        }
    }
}

// On a state with a unit quaternion the bound is in the tangent space (Chahbazian et al. 2024): with
// the attitude measured by a direction without noise, a Kalman filter started on the truth stays
// on it, and its covariance over the seven tangent coordinates is the bound's inverse.
TEST(PosteriorCramerRaoBound, IsInTheTangentSpaceOfAUnitQuaternion)
{
    namespace Pose = EmrysTest::UnequalDimensions;
    using EmrysTest::rotate;
    using Motion = Pose::Motion;
    using Orientation = Pose::Orientation;
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << Pose::turned, 10.0, -5.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Zero();
    cov.topLeftCorner<5, 5>() = Pose::poseCov();
    cov.bottomRightCorner<2, 2>() = 0.25 * Eigen::Matrix2d::Identity();
    auto const initial = GaussianParams<Motion>{mean, cov};
    auto const model = compose(attitudeWalk<Orientation>(0.01), ConstantVelocity<Pose::Position, Pose::Velocity>{0.1});
    auto filter = KalmanFilter{model, initial};
    auto bound = PosteriorCramerRaoBound<Motion>::create(initial).value();
    Eigen::Matrix<double, 8, 1> truth = mean;
    auto const forward = Eigen::Vector3d{1.0, 0.0, 0.0};
    auto const up = Eigen::Vector3d{0.0, 0.0, 1.0};
    for (int k = 0; k < 5; ++k)
    {
        Eigen::Matrix<double, 8, 1> const before = truth;
        truth = Emrys::propagate(Motion{}, truth, model, 1.0);
        filter.predict(1.0);
        ASSERT_TRUE(bound.predict(model, before, 1.0).has_value());
        for (auto const & axis : {forward, up})
        {
            Eigen::Vector3d const observed = rotate(truth.head<4>(), axis);
            auto const direction = measurementModel(
                [axis]<typename T>(auto const & in, auto const &, auto const & x) {
                    return rotate(Eigen::Matrix<T, 4, 1>{block<Orientation>(in, x)}, axis).eval();
                },
                reads(path<Orientation>),
                additiveNoise(
                    GaussianParams<Linear<3>>{
                        Eigen::Vector3d::Zero(),
                        Eigen::Matrix3d{0.001 * Eigen::Matrix3d::Identity()}
                    }
                )
            );
            ASSERT_TRUE(filter.update(observe(methods::linearized(), direction, observed)).value().accepted);
            ASSERT_TRUE(bound.update(truth, direction).has_value());
        }
        ASSERT_TRUE(filter.state().mean().isApprox(truth, 1e-12));
        EXPECT_TRUE(boundCovariance(bound).isApprox(filter.state().covarianceLower(), 1e-9)) << "step " << k;
    }
}

// Over several truths each expectation is the mean over the columns, taken before the recursion
// inverts, against eqs. (21) and (34) to (36) in plain Eigen: the prediction of a model whose
// Jacobian depends on the state, and the update by a range, whose Jacobian does too.
TEST(PosteriorCramerRaoBound, AveragesTheJacobianProductsOverTheTruths)
{
    auto const model = cubicDrag(0.05, 0.2);
    auto truths = Emrys::SampleSet<CubicState>(2, 3);
    truths << 1.0, -2.0, 0.5,  //
        0.0, 1.5, -1.0;
    Eigen::Matrix2d const prior{{2.0, 0.3}, {0.3, 1.0}};
    auto bound = PosteriorCramerRaoBound<CubicState>{CubicState{}, Eigen::Matrix2d{prior.inverse()}};
    double const dt = 0.5;
    ASSERT_TRUE(bound.predict(model, truths, dt).has_value());

    Eigen::Matrix2d const q = 0.2 * dt * Eigen::Matrix2d::Identity();
    Eigen::Matrix2d const qInverse = q.inverse();
    Eigen::Matrix2d meanJacobian = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d d11 = Eigen::Matrix2d::Zero();
    for (Eigen::Index i = 0; i < truths.cols(); ++i)
    {
        Eigen::Matrix2d const f =
            Eigen::Matrix2d{(1.0 - 3.0 * 0.05 * dt * truths.col(i).array().square()).matrix().asDiagonal()};
        meanJacobian += f / 3.0;
        d11 += f.transpose() * qInverse * f / 3.0;
    }
    Eigen::Matrix2d const predicted =
        qInverse - qInverse * meanJacobian * (prior.inverse() + d11).inverse() * meanJacobian.transpose() * qInverse;
    EXPECT_TRUE(bound.info().isApprox(predicted, 1e-12));

    // Each truth's range: the bound reads only the model's function and noise, so no observation is
    // needed.
    auto const anchor = Eigen::Vector2d{4.0, -3.0};
    ASSERT_TRUE(bound.update(truths, makeRangeModel<Cubic>(anchor)).has_value());
    Eigen::Matrix2d updated = predicted;
    for (Eigen::Index i = 0; i < truths.cols(); ++i)
    {
        Eigen::RowVector2d const h = (truths.col(i) - anchor).normalized().transpose();
        updated += h.transpose() * h / rangeVariance / 3.0;
    }
    EXPECT_TRUE(bound.info().isApprox(updated, 1e-12));
}

// A fixed-step model steps the bound as its model does over its step, bitwise, under a policy too;
// the bound keeps its space.
TEST(PosteriorCramerRaoBound, AFixedStepModelStepsAsItsModel)
{
    auto const model = cubicDrag(0.05, 0.2);
    auto const fixed = FixedStepModel{model, CubicState{}, 0.5};
    auto const truth = Eigen::Vector2d{1.0, -0.5};
    auto byStep = PosteriorCramerRaoBound<CubicState>{CubicState{}, Eigen::Matrix2d::Identity()};
    auto byModel = byStep;
    auto byPolicy = byStep;
    EXPECT_EQ(byStep.space(), CubicState{});
    ASSERT_TRUE(byStep.predict(fixed, truth).has_value());
    ASSERT_TRUE(byModel.predict(model, truth, 0.5).has_value());
    ASSERT_TRUE(byPolicy.predict(Emrys::execution::sequential, fixed, truth).has_value());
    EXPECT_EQ(byStep.info(), byModel.info());
    EXPECT_EQ(byPolicy.info(), byModel.info());
}

// Starting from no information, a fix of the position at each step makes the velocity known
// through the dynamics: after one step the bound has no inverse, and after two it has one.
TEST(PosteriorCramerRaoBound, StartsFromNoInfo)
{
    auto bound = PosteriorCramerRaoBound<State>{State{}, Matrix4::Zero()};
    Vector4 const truth = Vector4::Zero();
    EXPECT_EQ(bound.covariance().error(), GaussianError::Singular);
    ASSERT_TRUE(bound.predict(makeModel(), truth, 1.0).has_value());
    ASSERT_TRUE(bound.update(truth, makeFixModel()).has_value());
    EXPECT_EQ(bound.covariance().error(), GaussianError::Singular);
    ASSERT_TRUE(bound.predict(makeModel(), truth, 1.0).has_value());
    ASSERT_TRUE(bound.update(truth, makeFixModel()).has_value());
    EXPECT_TRUE(bound.covariance().has_value());
}

// What Proposition 1 needs is refused, and the bound is left as it was: a noiseless block makes Q
// singular (Section IV of the paper, not supported yet), a measurement noise of zero makes R
// singular, and a prior covariance with no inverse has no information matrix. From no
// information, a transition with no inverse leaves J + F^T Q^-1 F singular: the drag's Jacobian
// 1 - 3 rate dt x^2 is zero at x = 1 with rate dt = 1/3.
TEST(PosteriorCramerRaoBound, RefusesASingularNoiseOrPrior)
{
    auto const initial = makeInitial(Vector4::Zero());
    auto bound = PosteriorCramerRaoBound<State>::create(initial).value();
    Matrix4 const before = bound.info();
    Vector4 const truth = Vector4::Zero();
    auto const noiseless = compose(Constant<Position>{}, Constant<Velocity>{});
    EXPECT_EQ(bound.predict(noiseless, truth, 1.0).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(bound.info(), before);

    auto const exact = measurementModel(
        Eigen::Matrix2d::Identity(),
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()})
    );
    EXPECT_EQ(bound.update(truth, exact).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(bound.info(), before);

    auto const flat = GaussianParams<State>{State{}, Vector4::Zero(), Matrix4::Zero()};
    EXPECT_EQ(PosteriorCramerRaoBound<State>::create(flat).error(), GaussianError::NotPositiveDefinite);

    auto blind = PosteriorCramerRaoBound<CubicState>{CubicState{}, Eigen::Matrix2d::Zero()};
    auto const stalled = cubicDrag(1.0 / 3.0, 0.2);
    EXPECT_EQ(blind.predict(stalled, Eigen::Vector2d{1.0, 0.5}, 1.0).error(), GaussianError::Singular);
    EXPECT_EQ(blind.info(), Eigen::Matrix2d::Zero());
}

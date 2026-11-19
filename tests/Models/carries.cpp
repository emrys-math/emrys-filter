// What carries a covariance when an update moves its mean (docs/guide/groups.md, a prototype; Spaces/Reset.h,
// Models/Condition.h): carries::none(), carries::reset(), and carries::adjoint() as an argument of observe(), on a pose
// with landmarks seen from the body. Expected values are the two published invariant filters written out in plain
// Eigen: the left-invariant one in the pose's body-frame differences with no carry, and the right-invariant one in
// world-frame differences with no carry there, its covariance then brought to body-frame differences by the adjoint
// at the posterior mean. Also a prediction on the pose, whose Jacobian under a body-frame twist is the adjoint of
// the twist's step at every estimate. Under a sigma-point method the adjoint carry also spreads the points on the
// world side (roots::worldSide()), checked against a cubature filter written out in world-frame differences.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/ExtendedPose.h>
#include <EmrysFilter/Spaces/Group.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

using Emrys::Composite;
using Emrys::ExtendedPose;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::UnitQuaternion;

namespace carries = Emrys::carries;
namespace methods = Emrys::methods;

namespace
{

struct Attitude : UnitQuaternion
{};

struct Position : Linear<3>
{};

struct Bias : Linear<3>
{};

struct Pose : ExtendedPose<Attitude, Position>
{};

struct Sightings : Linear<6>
{};

using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

Eigen::Vector3d const landmarkA{4.0, 1.0, -2.0};
Eigen::Vector3d const landmarkB{-3.0, 5.0, 1.0};

Eigen::Matrix3d
skew(Eigen::Vector3d const & v)
{
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return m;
}

Vector7
priorMean()
{
    Vector7 x;
    x << UnitQuaternion::exp(Eigen::Vector3d{0.4, -0.7, 0.9}).coeffs(), 1.5, -2.0, 0.5;
    return x;
}

Matrix6
priorCovariance()
{
    Matrix6 a;
    a << 0.5, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.4, 0.1, 0.0, 0.3, 0.0, 0.1, 0.0, 0.6, 0.1, 0.0, 0.2, 0.0, 0.2, 0.0, 1.5,
        0.2, 0.0, 0.3, 0.0, 0.1, 0.0, 1.2, 0.3, 0.0, 0.1, 0.0, 0.2, 0.0, 1.8;
    return a * a.transpose();
}

Matrix6
sightingNoise()
{
    return Vector6{0.04, 0.09, 0.02, 0.05, 0.03, 0.08}.asDiagonal();
}

// Two landmarks seen from the body, y_i = R^T (l_i - p): the form y = X^-1 b.
auto
sightings()
{
    return Emrys::measurementModel(
        []<typename T>(auto const & x) {
            Eigen::Quaternion<T> const q{x(3), x(0), x(1), x(2)};
            Eigen::Matrix<T, 3, 1> const p = x.template tail<3>();
            Eigen::Matrix<T, 6, 1> y;
            y.template head<3>() = q.conjugate() * (landmarkA - p).eval();
            y.template tail<3>() = q.conjugate() * (landmarkB - p).eval();
            return y;
        },
        Emrys::outputSpace(Sightings{}),
        Emrys::additiveNoise(sightingNoise())
    );
}

Vector6
observed()
{
    // The sightings from a pose well away from the prior mean, so that the step is large.
    Vector7 const truth = Emrys::plus<Pose>(priorMean(), Vector6{0.5, -0.4, 0.6, 1.0, 0.8, -1.2});
    Eigen::Quaterniond const q{truth(3), truth(0), truth(1), truth(2)};
    Vector6 z;
    z << q.conjugate() * (landmarkA - truth.tail<3>()), q.conjugate() * (landmarkB - truth.tail<3>());
    return z;
}

// The update in the pose's body-frame differences, by hand: the gain, the step, and the covariance before any carry.
struct BodyUpdate
{
    Vector6 step;
    Matrix6 cov;
};

BodyUpdate
bodyUpdate()
{
    Vector7 const x = priorMean();
    Eigen::Quaterniond const q{x(3), x(0), x(1), x(2)};
    Vector6 predicted;
    predicted << q.conjugate() * (landmarkA - x.tail<3>()), q.conjugate() * (landmarkB - x.tail<3>());
    // y = Exp(-phi) R^T (l - p) - rho, so H = [y^, -I]: it depends on the estimate only through y.
    Matrix6 h;
    h << skew(predicted.head<3>()), -Eigen::Matrix3d::Identity(), skew(predicted.tail<3>()),
        -Eigen::Matrix3d::Identity();
    Matrix6 const p = priorCovariance();
    Matrix6 const s = h * p * h.transpose() + sightingNoise();
    Matrix6 const k = p * h.transpose() * s.inverse();
    return {k * (observed() - predicted), p - k * s * k.transpose()};
}

TEST(Carries, NoCarryIsTheLeftInvariantUpdate)
{
    auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
    auto const updated =
        Emrys::update(prior, Emrys::observe(methods::linearized(), sightings(), observed(), carries::none()));
    ASSERT_TRUE(updated.has_value());
    auto const expected = bodyUpdate();
    auto const & posterior = updated->branches[0].posterior;
    Matrix6 const cov = Matrix6{posterior.covarianceLower().selfadjointView<Eigen::Lower>()};
    EXPECT_LT(Emrys::minus<Pose>(posterior.mean(), Emrys::plus<Pose>(priorMean(), expected.step)).norm(), 1e-12);
    EXPECT_LT((cov - expected.cov).norm(), 1e-11);
}

TEST(Carries, ResetIsTheRightJacobianOfTheStep)
{
    auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
    // The default, and named.
    auto const byDefault = Emrys::update(prior, Emrys::observe(methods::linearized(), sightings(), observed()));
    auto const named =
        Emrys::update(prior, Emrys::observe(methods::linearized(), sightings(), observed(), carries::reset()));
    ASSERT_TRUE(byDefault.has_value() && named.has_value());
    auto const expected = bodyUpdate();
    Matrix6 const g = Emrys::rightJacobian<Pose>(expected.step);
    Matrix6 const cov{byDefault->branches[0].posterior.covarianceLower().selfadjointView<Eigen::Lower>()};
    EXPECT_LT((cov - g * expected.cov * g.transpose()).norm(), 1e-11);
    EXPECT_EQ(named->branches[0].posterior.covarianceLower(), byDefault->branches[0].posterior.covarianceLower());
}

TEST(Carries, AdjointCarryIsTheRightInvariantUpdate)
{
    // The right-invariant filter, in world-frame differences X = Exp(zeta) X_hat, with no carry there.
    Vector7 const x = priorMean();
    Eigen::Quaterniond const q{x(3), x(0), x(1), x(2)};
    Eigen::Matrix3d const turnBack = q.toRotationMatrix().transpose();
    Matrix6 const ad = Emrys::adjoint<Pose>(x);
    Matrix6 const worldCov = ad * priorCovariance() * ad.transpose();
    Vector6 predicted;
    predicted << q.conjugate() * (landmarkA - x.tail<3>()), q.conjugate() * (landmarkB - x.tail<3>());
    // y = R^T Exp(-phi) (l - p - phi x p - rho), so H = R^T [l^, -I]: the estimate enters by a rotation alone.
    Matrix6 h;
    h << turnBack * skew(landmarkA), -turnBack, turnBack * skew(landmarkB), -turnBack;
    Matrix6 const s = h * worldCov * h.transpose() + sightingNoise();
    Matrix6 const k = worldCov * h.transpose() * s.inverse();
    Vector6 const zeta = k * (observed() - predicted);
    Vector7 const mean = Emrys::compose<Pose>(Emrys::exp<Pose>(zeta), x);
    Matrix6 const worldPosterior = worldCov - k * s * k.transpose();
    // Brought to body-frame differences at the posterior mean.
    Matrix6 const back = Emrys::adjoint<Pose>(mean).inverse();
    Matrix6 const expected = back * worldPosterior * back.transpose();

    auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
    auto const updated =
        Emrys::update(prior, Emrys::observe(methods::linearized(), sightings(), observed(), carries::adjoint()));
    ASSERT_TRUE(updated.has_value());
    auto const & posterior = updated->branches[0].posterior;
    Matrix6 const cov = Matrix6{posterior.covarianceLower().selfadjointView<Eigen::Lower>()};
    EXPECT_LT(Emrys::minus<Pose>(posterior.mean(), mean).norm(), 1e-11);
    EXPECT_LT((cov - expected).norm(), 1e-10);
    // The three carries differ at this step.
    EXPECT_GT((cov - bodyUpdate().cov).norm(), 1e-3);
}

TEST(Carries, BiasBesideTheGroupIsCarriedUnchanged)
{
    // A Composite of the pose and a bias carries block by block, and a Linear block's adjoint is the identity.
    using State = Composite<Pose, Bias>;
    Eigen::Matrix<double, 10, 1> x;
    x << priorMean(), 0.1, 0.2, 0.3;
    Eigen::Matrix<double, 9, 1> d;
    d << 0.5, -0.4, 0.6, 1.0, 0.8, -1.2, 0.3, 0.2, 0.1;
    Eigen::Matrix<double, 9, 9> a = Eigen::Matrix<double, 9, 9>::Identity();
    a.topRightCorner<6, 3>().setConstant(0.2);
    Eigen::Matrix<double, 9, 9> const p = a * a.transpose();
    Eigen::Matrix<double, 9, 9> carried = p;
    Emrys::carryCovariance(carries::adjoint(), State{}, x, d, carried);
    Eigen::Matrix<double, 9, 9> g = Eigen::Matrix<double, 9, 9>::Identity();
    g.topLeftCorner<6, 6>() = Emrys::adjoint<Pose>(Emrys::exp<Pose>((-d.head<6>()).eval()));
    EXPECT_LT((carried - g * p * g.transpose()).norm(), 1e-12);
    Eigen::Matrix<double, 9, 9> kept = p;
    Emrys::carryCovariance(carries::none(), State{}, x, d, kept);
    EXPECT_EQ(kept, p);
}

TEST(Carries, TwistPredictionHasTheAdjointAsItsJacobian)
{
    // X' = X Exp(u dt): group-affine, so the Jacobian in body-frame differences is Ad(Exp(-u dt)) at every X.
    Vector6 const twist{0.3, -0.2, 0.5, 1.0, 0.5, -0.4};
    Matrix6 const noise = Vector6{0.01, 0.01, 0.02, 0.05, 0.05, 0.03}.asDiagonal();
    auto const motion = Emrys::processModel(
        [twist]<typename T>(auto const & x, double dt) {
            Eigen::Matrix<T, 7, 1> const value = x;
            Eigen::Matrix<T, 6, 1> const step = (dt * twist).template cast<T>();
            return Emrys::plus<Pose>(value, step);
        },
        Emrys::additiveNoise(noise)
    );
    double const dt = 0.5;
    Matrix6 const f = Emrys::adjoint<Pose>(Emrys::exp<Pose>((-dt * twist).eval()));
    for (auto const & method : {0, 1})
    {
        auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
        auto const predicted = method == 0 ? Emrys::predict(methods::linearized(), prior, motion, dt)
                                           : Emrys::predict(methods::unscented(), prior, motion, dt);
        Matrix6 const cov = Matrix6{predicted.covarianceLower().selfadjointView<Eigen::Lower>()};
        // Exactly linear in the differences, so the sigma points give the linearized result.
        EXPECT_LT((cov - (f * priorCovariance() * f.transpose() + noise)).norm(), method == 0 ? 1e-12 : 1e-9);
        EXPECT_LT(
            Emrys::minus<Pose>(predicted.mean(), Emrys::plus<Pose>(priorMean(), (dt * twist).eval())).norm(),
            1e-12
        );
    }
}

// The sightings at one pose, in plain doubles.
Vector6
sightingsAt(Vector7 const & x)
{
    Eigen::Quaterniond const q{x(3), x(0), x(1), x(2)};
    Vector6 y;
    y << q.conjugate() * (landmarkA - x.tail<3>()), q.conjugate() * (landmarkB - x.tail<3>());
    return y;
}

TEST(Carries, AdjointCarryUnderSigmaPointsIsTheWorldSideFilter)
{
    // A cubature filter with its differences on the world side, X = Exp(zeta) X_hat: 12 points at
    // Exp(+-sqrt(6) L_j) X_hat with L the Cholesky factor of the covariance in world-frame differences.
    Vector7 const x = priorMean();
    Matrix6 const ad = Emrys::adjoint<Pose>(x);
    Matrix6 const worldCov = ad * priorCovariance() * ad.transpose();
    Matrix6 const lower = worldCov.llt().matrixL();
    Eigen::Matrix<double, 6, 12> offsets;
    Eigen::Matrix<double, 6, 12> outputs;
    for (int j = 0; j < 6; ++j)
    {
        offsets.col(2 * j) = std::sqrt(6.0) * lower.col(j);
        offsets.col(2 * j + 1) = -offsets.col(2 * j);
    }
    for (int k = 0; k < 12; ++k)
    {
        outputs.col(k) = sightingsAt(Emrys::compose<Pose>(Emrys::exp<Pose>(offsets.col(k).eval()), x));
    }
    Vector6 const predicted = outputs.rowwise().mean();
    Eigen::Matrix<double, 6, 12> const deviations = outputs.colwise() - predicted;
    Matrix6 const s = deviations * deviations.transpose() / 12.0 + sightingNoise();
    Matrix6 const cross = offsets * deviations.transpose() / 12.0;
    Matrix6 const k = cross * s.inverse();
    Vector6 const zeta = k * (observed() - predicted);
    Vector7 const mean = Emrys::compose<Pose>(Emrys::exp<Pose>(zeta), x);
    Matrix6 const worldPosterior = worldCov - k * s * k.transpose();
    Matrix6 const back = Emrys::adjoint<Pose>(mean).inverse();
    Matrix6 const expected = back * worldPosterior * back.transpose();

    auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
    auto const updated =
        Emrys::update(prior, Emrys::observe(methods::cubature(), sightings(), observed(), carries::adjoint()));
    ASSERT_TRUE(updated.has_value());
    auto const & posterior = updated->branches[0].posterior;
    Matrix6 const cov = Matrix6{posterior.covarianceLower().selfadjointView<Eigen::Lower>()};
    EXPECT_LT(Emrys::minus<Pose>(posterior.mean(), mean).norm(), 1e-10);
    EXPECT_LT((cov - expected).norm(), 1e-9);

    // Named in the method, the same; and on the body side with the same carry, another filter.
    auto const named = Emrys::update(
        prior,
        Emrys::observe(
            methods::spread(methods::cubature(), Emrys::roots::worldSide()),
            sightings(),
            observed(),
            carries::adjoint()
        )
    );
    ASSERT_TRUE(named.has_value());
    EXPECT_EQ(named->branches[0].posterior.mean(), posterior.mean());
    auto const body =
        Emrys::update(prior, Emrys::observe(methods::cubature(), sightings(), observed(), carries::none()));
    ASSERT_TRUE(body.has_value());
    EXPECT_GT(Emrys::minus<Pose>(body->branches[0].posterior.mean(), mean).norm(), 1e-3);
}

TEST(Carries, PredictionSpreadsOnTheWorldSideByItsMethod)
{
    // A motion that is not group-affine, so that the side of the spread shows: the position drifts by a
    // function of the attitude.
    auto const motion = Emrys::processModel(
        []<typename T>(auto const & x, double dt) {
            Eigen::Matrix<T, 7, 1> y = x;
            y(4) += T(dt) * x(0) * x(1);
            y(5) += T(dt) * Eigen::numext::sin(x(2));
            return y;
        },
        Emrys::additiveNoise(Matrix6{0.01 * Matrix6::Identity()})
    );
    auto const prior = GaussianParams<Pose>{priorMean(), priorCovariance()};
    auto const body = Emrys::predict(methods::cubature(), prior, motion, 0.5);
    auto const world =
        Emrys::predict(methods::spread(methods::cubature(), Emrys::roots::worldSide()), prior, motion, 0.5);
    Matrix6 const bodyCov = Matrix6{body.covarianceLower().selfadjointView<Eigen::Lower>()};
    Matrix6 const worldCov = Matrix6{world.covarianceLower().selfadjointView<Eigen::Lower>()};
    // Two square roots of one covariance: the results agree to the function's curvature and no further.
    EXPECT_LT((bodyCov - worldCov).norm(), 0.5);
    EXPECT_GT((bodyCov - worldCov).norm(), 1e-6);
    EXPECT_TRUE((Eigen::SelfAdjointEigenSolver<Matrix6>{worldCov}.eigenvalues().array() > 0).all());
}

}  // namespace

// TransitionDensity and transitionDensity() of Models/TransitionDensity.h: the density of a process
// model's next state given its present one. On a linear model with positive definite noise it is
// the Gaussian density of the residual, computed here in plain Eigen. With a singular noise it is
// finite exactly on the states the noise reaches: a noiseless block is a constraint, noise through
// one direction reaches a line, and a model with no noise is a point mass. The states that
// simulate() draws have a finite density whose whitened residuals have unit variance. On a
// periodic coordinate and on a unit quaternion the residual is the tangent one. A fixed-step
// model, a std::chrono step, and float give the same density. The evidence for roadmap step 17zh.

#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Models/TransitionDensity.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::Linear;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::simulate;
using Emrys::transitionDensity;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Velocity>;
using Biased = Composite<Position, Velocity, Bias>;

struct Heading : RadiansPi<1>
{};

using HeadingState = Composite<Heading>;

struct Attitude : Emrys::UnitQuaternion
{};

using AttitudeState = Composite<Attitude>;

// A push on the position, an input of the call.
struct Push : Linear<2>
{};

// Moves the position by dt times a sure input, the model's whole motion, with no noise.
auto
pushed()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const &, auto const & x, auto dt) {
            return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Push>(in, x)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Push>),
        Emrys::writes(Emrys::path<Position>)
    );
}

// The log-density of N(mean, cov) at x, by a Cholesky factorization in plain Eigen.
template <int N>
double
gaussianLogDensity(
    Eigen::Matrix<double, N, 1> const & x,
    Eigen::Matrix<double, N, 1> const & mean,
    Eigen::Matrix<double, N, N> const & cov
)
{
    auto const llt = Eigen::LLT<Eigen::Matrix<double, N, N>>{cov};
    Eigen::Matrix<double, N, 1> const whitened = llt.matrixL().solve(x - mean);
    double logDeterminant = 0;
    for (int j = 0; j < N; ++j)
    {
        logDeterminant += 2 * std::log(llt.matrixLLT()(j, j));
    }
    return -0.5 * whitened.squaredNorm() - 0.5 * logDeterminant - 0.5 * N * std::log(2 * std::numbers::pi);
}

Eigen::Vector4d const origin{1, 2, 0.5, -0.25};

double const infinity = std::numeric_limits<double>::infinity();

}  // namespace

// On a linear model of the whole state with a positive definite Q, the density is the Gaussian
// density of the next state about F x under Q, here by Cholesky in plain Eigen.
TEST(TransitionDensity, EqualsTheGaussianDensityOnALinearModel)
{
    Eigen::Matrix4d f = Eigen::Matrix4d::Identity();
    f.topRightCorner<2, 2>() = 0.5 * Eigen::Matrix2d::Identity();
    auto const q =
        Eigen::Matrix4d{{0.4, 0.1, 0.05, 0.0}, {0.1, 0.3, 0.0, 0.02}, {0.05, 0.0, 0.2, 0.01}, {0.0, 0.02, 0.01, 0.1}};
    auto const model = Emrys::processModel(f, additiveNoise(q));
    auto const transition = transitionDensity(Linear<4>{}, model, 0.5);
    EXPECT_EQ(transition.rank(), 4);
    auto const to = Eigen::Vector4d{1.4, 1.7, 0.3, 0.1};
    Eigen::Vector4d const mean = f * origin;
    EXPECT_NEAR(transition.logDensity(origin, to), gaussianLogDensity<4>(to, mean, q), 1e-12);
    EXPECT_NEAR(transition.logDensity(origin, mean), gaussianLogDensity<4>(mean, mean, q), 1e-12);
}

// On a built-in model the density is the Gaussian density about propagate()'s state under the noise
// that a FixedStepModel caches for the step, and mean() is that state.
TEST(TransitionDensity, EqualsTheDensityOfTheModelsOwnNoise)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto const transition = transitionDensity(State{}, model, 0.5);
    Eigen::Vector4d const mean = Emrys::propagate(State{}, origin, model, 0.5);
    Eigen::Matrix4d const q = FixedStepModel{model, State{}, 0.5}.noise();
    auto const to = Eigen::Vector4d{1.3, 1.8, 0.45, -0.2};
    EXPECT_EQ(transition.rank(), 4);
    EXPECT_EQ(transition.mean(origin), mean);
    EXPECT_NEAR(transition.logDensity(origin, to), gaussianLogDensity<4>(to, mean, q), 1e-11);
    EXPECT_EQ(transition.logDensityAbout(transition.mean(origin), to), transition.logDensity(origin, to));
    // A copy resolves the model against its own space and evaluates alike.
    auto const copy = transition;
    EXPECT_EQ(copy.logDensity(origin, to), transition.logDensity(origin, to));
}

// A block with no noise is a constraint. Beside a constant bias the density is that of the
// kinematic block alone where the bias is unchanged, and zero where the bias differs at all.
TEST(TransitionDensity, ANoiselessBlockIsAConstraint)
{
    auto const kinematic = ConstantVelocity<Position, Velocity>{0.1};
    auto const model = compose(kinematic, Constant<Bias>{});
    auto const transition = transitionDensity(Biased{}, model, 0.5);
    EXPECT_EQ(transition.rank(), 4);
    auto const from = Eigen::Matrix<double, 5, 1>{1, 2, 0.5, -0.25, 0.7};
    Eigen::Vector4d const mean = Emrys::propagate(State{}, origin, kinematic, 0.5);
    Eigen::Matrix4d const q = FixedStepModel{kinematic, State{}, 0.5}.noise();
    auto const moved = Eigen::Vector4d{1.3, 1.8, 0.45, -0.2};
    auto to = Eigen::Matrix<double, 5, 1>{};
    to << moved, 0.7;
    EXPECT_NEAR(transition.logDensity(from, to), gaussianLogDensity<4>(moved, mean, q), 1e-11);
    to(4) = 0.7 + 1e-9;
    EXPECT_EQ(transition.logDensity(from, to), -infinity);
}

// Every state that simulate() draws under a model with a noiseless block has a finite density from
// the state it was drawn from, and twice the drop of its log-density below the peak, the squared
// length of its whitened residual, has mean 4, the rank. The tolerance is about five standard
// errors of a chi-squared mean over 4000 draws, 5 sqrt(8 / 4000) = 0.22.
TEST(TransitionDensity, SimulatedStatesHaveAFiniteDensityWithUnitWhitenedResiduals)
{
    auto const model = compose(ConstantVelocity<Position, Velocity>{0.1}, Constant<Bias>{});
    auto const transition = transitionDensity(Biased{}, model, 0.5);
    auto const from = Eigen::Matrix<double, 5, 1>{1, 2, 0.5, -0.25, 0.7};
    Eigen::Matrix<double, 5, 1> const mean = transition.mean(from);
    double const peak = transition.logDensityAbout(mean, mean);
    auto rng = std::mt19937_64{1};
    int const count = 4000;
    double sum = 0;
    for (int i = 0; i < count; ++i)
    {
        Eigen::Matrix<double, 5, 1> const to = simulate(Biased{}, from, model, 0.5, rng);
        double const logDensity = transition.logDensityAbout(mean, to);
        ASSERT_TRUE(std::isfinite(logDensity));
        sum += 2 * (peak - logDensity);
    }
    EXPECT_NEAR(sum / count, 4.0, 0.22);
}

// One acceleration driving a position and a velocity gives Q = q g g^T, of rank one, whose null
// direction is not a coordinate. The density is finite on the line f(x) + a g, where it is the
// scalar Gaussian of a g's length under the variance q g^T g, by hand; it is zero off the line; and
// every state simulate() draws is on it.
TEST(TransitionDensity, NoiseThroughOneDirectionReachesALine)
{
    double const dt = 0.5;
    double const q = 0.3;
    auto const f = Eigen::Matrix2d{{1.0, dt}, {0.0, 1.0}};
    auto const g = Eigen::Vector2d{dt * dt / 2, dt};
    Eigen::Matrix2d const noise = q * g * g.transpose();
    auto const model = Emrys::processModel(f, additiveNoise(noise));
    auto const transition = transitionDensity(Linear<2>{}, model, dt);
    EXPECT_EQ(transition.rank(), 1);
    auto const from = Eigen::Vector2d{1.0, -2.0};
    Eigen::Vector2d const mean = f * from;
    double const a = 0.8;
    Eigen::Vector2d const onLine = mean + a * g;
    double const variance = q * g.squaredNorm();
    double const along = a * g.norm();
    double const expected = -0.5 * along * along / variance - 0.5 * std::log(2 * std::numbers::pi * variance);
    EXPECT_NEAR(transition.logDensity(from, onLine), expected, 1e-11);
    Eigen::Vector2d const offLine = onLine + 1e-6 * Eigen::Vector2d{g(1), -g(0)};
    EXPECT_EQ(transition.logDensity(from, offLine), -infinity);
    auto rng = std::mt19937_64{2};
    for (int i = 0; i < 1000; ++i)
    {
        ASSERT_TRUE(std::isfinite(transition.logDensity(from, simulate(Linear<2>{}, from, model, dt, rng))));
    }
}

// A model with no noise is a point mass at its step: the log-density is zero there and minus
// infinity anywhere else, and a sure input of the call reaches the model.
TEST(TransitionDensity, AModelWithNoNoiseIsAPointMass)
{
    auto const model = compose(pushed(), Constant<Velocity>{});
    auto const push = Eigen::Vector2d{3, -1};
    auto const transition = transitionDensity(State{}, model, 2.0, Emrys::sure<Push>(push));
    EXPECT_EQ(transition.rank(), 0);
    auto const expected = Eigen::Vector4d{1 + 2 * 3, 2 - 2 * 1, 0.5, -0.25};
    EXPECT_EQ(transition.mean(origin), expected);
    EXPECT_EQ(transition.logDensity(origin, expected), 0.0);
    EXPECT_EQ(transition.logDensity(origin, origin), -infinity);
}

// On a periodic coordinate the residual takes the short way round: from 3.1 to -3.1 the density is
// that of a step of 2 pi - 6.2 from zero.
TEST(TransitionDensity, APeriodicStateTakesTheShortWayRound)
{
    auto const model = RandomWalk<Heading>{0.5};
    auto const transition = transitionDensity(HeadingState{}, model, 1.0);
    auto const nearSeam = Eigen::Matrix<double, 1, 1>{3.1};
    auto const across = Eigen::Matrix<double, 1, 1>{-3.1};
    auto const zero = Eigen::Matrix<double, 1, 1>{0.0};
    auto const step = Eigen::Matrix<double, 1, 1>{2 * std::numbers::pi - 6.2};
    EXPECT_NEAR(transition.logDensity(nearSeam, across), transition.logDensity(zero, step), 1e-12);
    double const variance = FixedStepModel{model, HeadingState{}, 1.0}.noise()(0, 0);
    EXPECT_NEAR(
        transition.logDensity(zero, step),
        -0.5 * step(0) * step(0) / variance - 0.5 * std::log(2 * std::numbers::pi * variance),
        1e-12
    );
}

// On a unit quaternion the density is the Gaussian of the three tangent coordinates: of the
// difference d where the next attitude is the present one plus d.
TEST(TransitionDensity, AUnitQuaternionTakesTheTangentDensity)
{
    auto const model = RandomWalk<Attitude>{0.04};
    auto const transition = transitionDensity(AttitudeState{}, model, 0.5);
    EXPECT_EQ(transition.rank(), 3);
    Eigen::Vector4d const from = Eigen::Vector4d{0.5, -0.5, 0.5, 0.5};
    auto const d = Eigen::Vector3d{0.05, -0.1, 0.02};
    Eigen::Vector4d const to = Emrys::plus(AttitudeState{}, from, d);
    Eigen::Matrix3d const q = FixedStepModel{model, AttitudeState{}, 0.5}.noise();
    EXPECT_NEAR(transition.logDensity(from, to), gaussianLogDensity<3>(d, Eigen::Vector3d::Zero().eval(), q), 1e-10);
}

// A FixedStepModel's density is that of the model it steps over its step, and a std::chrono step
// is the same step in seconds.
TEST(TransitionDensity, AFixedStepAndAChronoStepGiveTheSameDensity)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto const fixed = FixedStepModel{model, State{}, 0.1};
    auto const bySeconds = transitionDensity(State{}, model, 0.1);
    auto const byChrono = transitionDensity(State{}, model, std::chrono::milliseconds{100});
    auto const byStep = transitionDensity(State{}, fixed);
    auto const to = Eigen::Vector4d{1.05, 1.98, 0.52, -0.2};
    EXPECT_EQ(byChrono.logDensity(origin, to), bySeconds.logDensity(origin, to));
    EXPECT_EQ(byStep.logDensity(origin, to), bySeconds.logDensity(origin, to));
}

// In float the density is the double one to float's precision, with a singular noise too.
TEST(TransitionDensity, ComputesInFloat)
{
    auto const model = compose(ConstantVelocity<Position, Velocity>{0.1}, Constant<Bias>{});
    auto const inDouble = transitionDensity(Biased{}, model, 0.5);
    auto const inFloat = transitionDensity<float>(Biased{}, model, 0.5f);
    static_assert(
        std::is_same_v<decltype(inFloat.logDensity(Eigen::Matrix<float, 5, 1>{}, Eigen::Matrix<float, 5, 1>{})), float>
    );
    auto const from = Eigen::Matrix<double, 5, 1>{1, 2, 0.5, -0.25, 0.7};
    auto const to = Eigen::Matrix<double, 5, 1>{1.3, 1.8, 0.45, -0.2, 0.7};
    EXPECT_EQ(inFloat.rank(), 4);
    EXPECT_NEAR(inFloat.logDensity(from.cast<float>(), to.cast<float>()), inDouble.logDensity(from, to), 1e-3);
}

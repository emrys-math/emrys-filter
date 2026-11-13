// simulate() of Models/Simulation.h: a draw of the next true state under a process model. Without
// process noise it is the propagation exactly; its draws have the model's mean and process noise;
// a seed fixes a run, under any time type and through a FixedStepModel; a periodic state stays in
// its interval; and inputs reach the blocks. A model's measurement of a true state, whose draws
// have the measurement function's value and the noise's covariance, stay in a periodic
// measurement space, and pass an uncertain input through the function. A Markov chain's next value,
// discrete or continuous in time, is a draw from its row. The evidence for simulate() in
// the Agent-friendly use section of the roadmap, and for the simulated measurements of [sensors] and
// [printing-and-simulation].

#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Simulation.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::BlockRows;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::simulate;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Heading : RadiansPi<1>
{};

using HeadingState = Composite<Heading>;

// A mode of three values, the space of a chain's simulated values.
struct Low
{};

struct Middle
{};

struct High
{};

struct Gear : Emrys::Named<Low, Middle, High>
{};

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

Eigen::Vector4d const origin{1, 2, 0.5, -0.25};

}  // namespace

// With no process noise the draw is the model's propagation, bitwise: the noise's factor is zero.
TEST(Simulate, WithoutNoiseIsThePropagation)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.0};
    auto rng = std::mt19937_64{1};
    EXPECT_EQ(simulate(State{}, origin, model, rng, 0.5), Emrys::propagate(State{}, origin, model, 0.5));
}

// From one state, 20000 draws have the propagated mean and the model's process noise as their
// covariance. The tolerances are about five standard errors: the largest variance in Q is 0.1, so
// a mean's standard error is sqrt(0.1 / 20000) ≈ 0.0022 and a covariance entry's about 0.001.
TEST(Simulate, DrawsHaveTheModelsMoments)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto rng = std::mt19937_64{2};
    int const count = 20000;
    Eigen::Matrix<double, 4, Eigen::Dynamic> draws(4, count);
    for (int i = 0; i < count; ++i)
    {
        draws.col(i) = simulate(State{}, origin, model, rng, 1.0);
    }
    Eigen::Vector4d const mean = draws.rowwise().mean();
    Eigen::Matrix<double, 4, Eigen::Dynamic> const centered = draws.colwise() - mean;
    Eigen::Matrix4d const covariance = centered * centered.transpose() / (count - 1);
    EXPECT_LT((mean - Emrys::propagate(State{}, origin, model, 1.0)).cwiseAbs().maxCoeff(), 0.011);
    EXPECT_LT((covariance - FixedStepModel{model, State{}, 1.0}.noise()).cwiseAbs().maxCoeff(), 0.005);
}

// A seed fixes a run: the same seed gives the same trajectory, bitwise, whether the step is a
// double in seconds, a std::chrono::duration, or a FixedStepModel's step.
TEST(Simulate, ASeedFixesTheRunUnderAnyTimeStep)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto const fixed = FixedStepModel{model, State{}, 0.1};
    auto seconds = std::mt19937_64{3};
    auto chrono = std::mt19937_64{3};
    auto steps = std::mt19937_64{3};
    Eigen::Vector4d bySeconds = origin;
    Eigen::Vector4d byChrono = origin;
    Eigen::Vector4d bySteps = origin;
    for (int k = 0; k < 10; ++k)
    {
        bySeconds = simulate(State{}, bySeconds, model, seconds, 0.1);
        byChrono = simulate(State{}, byChrono, model, chrono, std::chrono::milliseconds{100});
        bySteps = simulate(State{}, bySteps, fixed, steps);
    }
    EXPECT_NE(bySeconds, origin);
    EXPECT_EQ(byChrono, bySeconds);
    EXPECT_EQ(bySteps, bySeconds);
}

// A heading near the seam with unit noise wraps into [-pi, pi): every draw is in the interval and
// some cross the seam.
TEST(Simulate, PeriodicStatesStayInTheirInterval)
{
    auto const model = RandomWalk<Heading>{1.0};
    auto rng = std::mt19937_64{4};
    auto const nearSeam = Eigen::Matrix<double, 1, 1>{3.1};
    int wrapped = 0;
    for (int i = 0; i < 1000; ++i)
    {
        double const heading = simulate(HeadingState{}, nearSeam, model, rng, 1.0)(0);
        ASSERT_GE(heading, -std::numbers::pi);
        ASSERT_LT(heading, std::numbers::pi);
        wrapped += heading < 0 ? 1 : 0;
    }
    EXPECT_GT(wrapped, 100);
}

// A sure input reaches the model as it is: with no noise the position moves by dt times it.
TEST(Simulate, InputsReachTheBlocks)
{
    auto const model = compose(pushed(), Constant<Velocity>{});
    auto rng = std::mt19937_64{5};
    auto const push = Eigen::Vector2d{3, -1};
    auto const expected = Eigen::Vector4d{1 + 2 * 3, 2 - 2 * 1, 0.5, -0.25};
    EXPECT_EQ(simulate(State{}, origin, model, rng, 2.0, Emrys::sure<Push>(push)), expected);
}

// A sensor's draws from one true state have the measurement function's value as their mean and
// the additive noise's covariance as theirs. The tolerances are about five standard errors, as in
// DrawsHaveTheModelsMoments: the largest variance is 0.1.
TEST(SimulateMeasurement, DrawsHaveTheFunctionsValueAndTheNoisesCovariance)
{
    auto const r = Eigen::Matrix2d{{0.1, 0.02}, {0.02, 0.05}};
    auto const offset = Eigen::Vector2d{-1.0, 0.5};
    auto const fix = measurementModel(
        [offset](auto const & in, auto const &, auto const & x) {
            return (block<Position>(in, x) + offset).eval();
        },
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), r})
    );
    auto rng = std::mt19937_64{6};
    int const count = 20000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> draws(2, count);
    for (int i = 0; i < count; ++i)
    {
        draws.col(i) = simulate(State{}, origin, fix, rng);
    }
    Eigen::Vector2d const mean = draws.rowwise().mean();
    Eigen::Matrix<double, 2, Eigen::Dynamic> const centered = draws.colwise() - mean;
    Eigen::Matrix2d const covariance = centered * centered.transpose() / (count - 1);
    EXPECT_LT((mean - (origin.head<2>() + offset)).cwiseAbs().maxCoeff(), 0.011);
    EXPECT_LT((covariance - r).cwiseAbs().maxCoeff(), 0.005);
}

// A bearing drawn near the seam of a periodic measurement space wraps into [-pi, pi): every draw
// is in the interval and some cross the seam, as the noise is added by the space's plus().
TEST(SimulateMeasurement, APeriodicMeasurementStaysInItsInterval)
{
    auto const bearing = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::atan2(T(p(1)), T(p(0)))};
        },
        reads(path<Position>),
        outputSpace(HeadingState{}),
        additiveNoise(GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.01}})
    );
    auto const behind = Eigen::Vector4d{-5.0, 0.01, 0.0, 0.0};
    auto rng = std::mt19937_64{7};
    int wrapped = 0;
    for (int i = 0; i < 1000; ++i)
    {
        double const drawn = simulate(State{}, behind, bearing, rng)(0);
        ASSERT_GE(drawn, -std::numbers::pi);
        ASSERT_LT(drawn, std::numbers::pi);
        wrapped += drawn < 0 ? 1 : 0;
    }
    EXPECT_GT(wrapped, 100);
}

// An uncertain input of the call passes through the function: a range with a scale-factor error v,
// |p| (1 + v), has mean |p| and variance |p|^2 sigma^2 over 20000 draws, here 25 * 0.0004 = 0.01, so
// the mean's standard error is 0.0007 and the variance's about 0.0001; the tolerances are about five
// of them.
TEST(SimulateMeasurement, InputNoisePassesThroughTheFunction)
{
    struct Scale : Linear<1>
    {};

    auto const scaled = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(p.squaredNorm())) * (T(1) + block<Scale>(in, x)(0))};
        },
        reads(path<Position>, path<Scale>)
    );
    auto const scale = GaussianParams<Scale>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.0004}};
    auto const target = Eigen::Vector4d{3.0, 4.0, 0.0, 0.0};
    auto rng = std::mt19937_64{8};
    int const count = 20000;
    auto draws = Eigen::ArrayXd{count};
    for (int i = 0; i < count; ++i)
    {
        draws(i) = simulate(State{}, target, scaled, rng, scale)(0);
    }
    double const mean = draws.mean();
    double const variance = (draws - mean).square().sum() / (count - 1);
    EXPECT_NEAR(mean, 5.0, 0.0035);
    EXPECT_NEAR(variance, 0.01, 0.0006);
}

// A model known by its likelihood alone draws its observation by its sample(g), after the call's
// uncertain inputs, placed or not: a range with an exponential delay of mean 0.5 and an uncertain
// offset of variance 0.01 has mean 5.5 and variance 0.25 + 0.01 over 20000 draws, so the mean's
// standard error is 0.0036 and the variance's about 0.005; the tolerances are about five of them.
// The placed model draws what the unplaced one does from the same seed, bitwise. The first model's
// likelihood and draw take the scalar of x as their template parameter [function-scalars].
TEST(SimulateMeasurement, AModelKnownByItsLikelihoodDrawsBySample)
{
    struct Offset : Linear<1>
    {};

    struct Ownship : State
    {};

    auto const range = [](auto const & in, auto const & x) {
        return block<Position>(in, x).norm() + block<Offset>(in, x)(0);
    };
    auto const delayed = measurementModel(
        reads(path<Position>, path<Offset>),
        Emrys::logLikelihood([range]<typename T>(auto const & in, auto const & x, auto const & z) {
            T const e = z(0) - range(in, x);
            return e < 0 ? -std::numeric_limits<T>::infinity() : std::log(T(2)) - 2 * e;
        }),
        Emrys::sample([range]<typename T>(auto const & in, auto const & x, auto & rng) {
            return Eigen::Matrix<T, 1, 1>{range(in, x) + std::exponential_distribution<T>{2.0}(rng)};
        })
    );
    auto const offset = GaussianParams<Offset>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.01}};
    auto const target = Eigen::Vector4d{3.0, 4.0, 0.0, 0.0};
    auto rng = std::mt19937_64{10};
    int const count = 20000;
    auto draws = Eigen::ArrayXd{count};
    for (int i = 0; i < count; ++i)
    {
        draws(i) = simulate(State{}, target, delayed, rng, offset)(0);
    }
    double const mean = draws.mean();
    double const variance = (draws - mean).square().sum() / (count - 1);
    EXPECT_NEAR(mean, 5.5, 0.018);
    EXPECT_NEAR(variance, 0.26, 0.025);

    auto const distance = measurementModel(
        reads(path<Position>),
        Emrys::logLikelihood([](auto const & in, auto const & x, auto const & z) {
            return -std::abs(z(0) - block<Position>(in, x).norm());
        }),
        Emrys::sample([](auto const & in, auto const & x, auto & rng) {
            return Eigen::Matrix<double, 1, 1>{
                block<Position>(in, x).norm() + std::exponential_distribution<double>{2.0}(rng)
            };
        })
    );
    auto given = std::mt19937_64{11};
    auto moved = std::mt19937_64{11};
    EXPECT_EQ(
        simulate(Composite<Ownship>{}, target, Emrys::on<Ownship>(distance), moved),
        simulate(State{}, target, distance, given)
    );
}

// A fixed-size space given as a template argument draws what its default instance does, bitwise,
// for a process model, a fixed-step model, and a measurement model.
TEST(Simulate, AFixedSizeSpaceNeedsNoInstance)
{
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto const fixed = FixedStepModel{model, State{}, 0.1};
    auto const gps = measurementModel(
        Eigen::Matrix2d::Identity(),
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()})
    );
    auto given = std::mt19937_64{9};
    auto named = std::mt19937_64{9};
    EXPECT_EQ(simulate<State>(origin, model, named, 0.1), simulate(State{}, origin, model, given, 0.1));
    EXPECT_EQ(simulate<State>(origin, fixed, named), simulate(State{}, origin, fixed, given));
    EXPECT_EQ(simulate<State>(origin, gps, named), simulate(State{}, origin, gps, given));
}

// A chain's next value is a draw from next(from), the very draw drawSample() of it makes from the
// same generator, and never a value the row gives probability zero; a continuous-time chain's is
// its transition() over the interval, in any time type.
TEST(SimulateChain, DrawsTheNextValueFromTheRow)
{
    auto const chain = Emrys::MarkovChain<Gear>{Eigen::Matrix3d{{0.7, 0.2, 0.1}, {0.0, 0.5, 0.5}, {0.3, 0.0, 0.7}}};
    auto byChain = std::mt19937_64{3};
    auto byRow = std::mt19937_64{3};
    Eigen::Index value = 0;
    for (int k = 0; k < 200; ++k)
    {
        Eigen::Index const next = simulate(value, chain, byChain);
        auto const row = chain.next(value);
        EXPECT_EQ(next, Emrys::drawSample(row.toDensity().value(), byRow));
        EXPECT_GT(chain.transition(value, next), 0.0);
        value = next;
    }

    auto const continuous =
        Emrys::ContinuousTimeMarkovChain<Gear>{Eigen::Matrix3d{{-1.0, 0.5, 0.5}, {0.2, -0.4, 0.2}, {0.0, 1.0, -1.0}}};
    auto inSeconds = std::mt19937_64{4};
    auto inMilliseconds = std::mt19937_64{4};
    auto stepped = std::mt19937_64{4};
    for (Eigen::Index from = 0; from < 3; ++from)
    {
        Eigen::Index const drawn = simulate(from, continuous, inSeconds, 0.25);
        EXPECT_EQ(drawn, simulate(from, continuous, inMilliseconds, std::chrono::milliseconds{250}));
        EXPECT_EQ(drawn, simulate(from, continuous.transition(0.25), stepped));
    }
}

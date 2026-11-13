// Time steps through the filters (roadmap step 17f; [time-steps], amending [fixed-step-models]): every filter's
// predict() and predictJoint() take a std::chrono::duration, an integer clock, or a user's time type and give the run
// in double seconds bitwise, since each converts once to the scalar; a float filter predicts in float from float
// blocks; a FixedStepModel takes its step as any of them; and predict(Steps{n}) on a fixed-step filter is n calls of
// predict(), bitwise. Headers under test: Core/Time.h, Models/ProcessModel.h, Models/Emission.h,
// Distributions/ContinuousTimeMarkovChain.h, and Filters/KalmanFilter.h, MixtureFilter.h, ImmFilter.h,
// ParticleFilter.h, RbParticleFilter.h, RbPointMassFilter.h, and HiddenMarkovFilter.h.

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <ratio>
#include <type_traits>
#include <vector>

using Emrys::additiveNoise;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::ScalarOf;
using Emrys::Steps;
using Emrys::SwitchingModel;

namespace methods = Emrys::methods;

using namespace std::chrono_literals;

namespace
{

using Index = Eigen::Index;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

using State = Composite<Position, Velocity>;

// The MPEG system clock: 90 kHz, so 9000 ticks are the 0.1 s step.
using MpegTicks = std::chrono::duration<std::int64_t, std::ratio<1, 90000>>;

constexpr double timeStep = 0.1;
constexpr auto timeStepInMilliseconds = 100ms;
constexpr MpegTicks timeStepInTicks{9000};

template <typename Scalar = double>
auto
makeModel()
{
    return ConstantVelocity<Position, Velocity, Scalar>{Scalar(0.1)};
}

template <typename Scalar = double>
GaussianParams<State, Scalar>
makeInitial()
{
    using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << Scalar(4), Scalar(4), Scalar(1), Scalar(1);
    cov(0, 2) = cov(2, 0) = Scalar(1.2);
    cov(1, 3) = cov(3, 1) = Scalar(-0.8);
    return {State{}, Vector4{Scalar(0), Scalar(0), Scalar(1), Scalar(0.5)}, cov};
}

template <typename Scalar = double>
auto
makeFix(Eigen::Matrix<Scalar, 2, 1> const & z)
{
    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
    using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
    auto const fix = Emrys::measurementModel(
        Matrix2::Identity(),
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(
            GaussianParams<Linear<2>, Scalar>{
                Vector2::Zero(),
                Matrix2{{Scalar(1), Scalar(0.2)}, {Scalar(0.2), Scalar(0.8)}}
            }
        )
    );
    return observe(Emrys::methods::linearized(), fix, z);
}

// Whether \a Filter's predict() takes a step of type \a Time.
template <typename Filter, typename Time>
concept PredictsOver = requires(Filter & filter, Time const & dt) { filter.predict(dt); };

// Expects \a a and \a b to be the same, coefficient for coefficient.
template <typename A, typename B>
void
expectSame(A const & a, B const & b)
{
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    EXPECT_TRUE((a.array() == b.array()).all());
}

// Expects two Gaussians to be the same, mean and covariance.
template <typename G>
void
expectSameGaussian(G const & a, G const & b)
{
    expectSame(a.mean(), b.mean());
    expectSame(a.covarianceLower(), b.covarianceLower());
}

}  // namespace

// Ten predictions and updates in double seconds, in milliseconds, and in ticks of a 90 kHz clock
// give the same Kalman filter, bitwise, since 100 / 1000 and 9000 / 90000 are the correctly rounded
// 0.1. predictJoint() takes a duration as predict() does, and an integer is no step.
TEST(TimeSteps, KalmanFilterTakesDurationsAndIntegerClocks)
{
    using Filter = decltype(KalmanFilter{makeModel(), makeInitial()});
    static_assert(PredictsOver<Filter, double> && PredictsOver<Filter, float>);
    static_assert(PredictsOver<Filter, std::chrono::milliseconds> && PredictsOver<Filter, MpegTicks>);
    static_assert(!PredictsOver<Filter, int> && !PredictsOver<Filter, std::int64_t>);
    static_assert(!PredictsOver<Filter, Steps>);

    auto seconds = KalmanFilter{makeModel(), makeInitial()};
    auto milliseconds = KalmanFilter{makeModel(), makeInitial()};
    auto ticks = KalmanFilter{makeModel(), makeInitial()};
    for (int k = 1; k <= 10; ++k)
    {
        seconds.predict(timeStep);
        milliseconds.predict(timeStepInMilliseconds);
        ticks.predict(timeStepInTicks);
        auto const fix = makeFix(Eigen::Vector2d{0.1 * k, 0.05 * k});
        ASSERT_TRUE(seconds.update(fix));
        ASSERT_TRUE(milliseconds.update(fix));
        ASSERT_TRUE(ticks.update(fix));
    }
    expectSameGaussian(milliseconds.state(), seconds.state());
    expectSameGaussian(ticks.state(), seconds.state());

    auto const joint = seconds.predictJoint(timeStep);
    auto const jointInMilliseconds = milliseconds.predictJoint(timeStepInMilliseconds);
    expectSameGaussian(jointInMilliseconds.predicted, joint.predicted);
    expectSame(jointInMilliseconds.cross, joint.cross);
}

// A float filter over a float model predicts in float: a duration gives the float step's result
// bitwise, the process noise is the float arithmetic of the model's closed form, dt^3 times 1/3
// and so on, and ten steps stay within float rounding (1e-5 relative) of the double filter.
TEST(TimeSteps, AFloatFilterPredictsInFloat)
{
    using Matrix4f = Eigen::Matrix<float, 4, 4>;
    auto const model = makeModel<float>();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(model)>::Scalar, float>);
    auto single = KalmanFilter{model, makeInitial<float>()};
    auto singleInMilliseconds = KalmanFilter{model, makeInitial<float>()};
    auto reference = KalmanFilter{makeModel(), makeInitial()};
    static_assert(std::is_same_v<decltype(single)::Scalar, float>);
    for (int k = 1; k <= 10; ++k)
    {
        single.predict(0.1f);
        singleInMilliseconds.predict(timeStepInMilliseconds);
        reference.predict(timeStep);
        ASSERT_TRUE(single.update(makeFix<float>(Eigen::Vector2f{0.1f * float(k), 0.05f * float(k)})));
        ASSERT_TRUE(singleInMilliseconds.update(makeFix<float>(Eigen::Vector2f{0.1f * float(k), 0.05f * float(k)})));
        ASSERT_TRUE(reference.update(makeFix(Eigen::Vector2d{0.1 * k, 0.05 * k})));
    }
    expectSameGaussian(singleInMilliseconds.state(), single.state());
    EXPECT_LE(
        (single.state().mean().cast<double>() - reference.state().mean()).norm(),
        1e-5 * reference.state().mean().norm()
    );
    EXPECT_LE(
        (single.state().covarianceLower().cast<double>() - reference.state().covarianceLower()).norm(),
        1e-5 * reference.state().covarianceLower().norm()
    );

    float const dt = 0.1f;
    float const density = 0.1f;
    Matrix4f expected = Matrix4f::Zero();
    expected.diagonal() << dt * dt * dt * (1.0f / 3.0f) * density, dt * dt * dt * (1.0f / 3.0f) * density, dt * density,
        dt * density;
    expected(0, 2) = expected(2, 0) = expected(1, 3) = expected(3, 1) = dt * dt * (1.0f / 2.0f) * density;
    auto const fixed =
        FixedStepModel<State, std::remove_cvref_t<decltype(model)>, methods::Linearized, float>{model, State{}, dt};
    expectSame(fixed.noise(), expected);
}

// A fixed-step model takes its step as any time type and caches the same noise and transition; a
// float model converts it to float. predict(Steps{n}) on a Kalman filter is n calls of predict(),
// bitwise, Steps{0} changes nothing, and a form of the Gaussian returns success.
TEST(TimeSteps, FixedStepModelsTakeAnyTimeAndAdvanceByACount)
{
    auto const model = FixedStepModel{makeModel(), State{}, timeStep};
    auto const inMilliseconds = FixedStepModel{makeModel(), State{}, timeStepInMilliseconds};
    auto const inTicks = FixedStepModel{makeModel(), State{}, timeStepInTicks};
    static_assert(std::is_same_v<decltype(inMilliseconds), decltype(model)>);
    EXPECT_EQ(inMilliseconds.timeStep(), timeStep);
    EXPECT_EQ(inTicks.timeStep(), timeStep);
    expectSame(inMilliseconds.noise(), model.noise());
    expectSame(inMilliseconds.transition(), model.transition());

    auto const single = FixedStepModel<State, decltype(makeModel<float>()), methods::Linearized, float>{
        makeModel<float>(),
        State{},
        timeStepInMilliseconds
    };
    EXPECT_EQ(single.timeStep(), 0.1f);

    auto stepped = KalmanFilter{model, makeInitial()};
    auto counted = KalmanFilter{model, makeInitial()};
    for (int k = 0; k < 7; ++k)
    {
        stepped.predict();
    }
    counted.predict(Steps{7});
    expectSameGaussian(counted.state(), stepped.state());
    counted.predict(Steps{0});
    expectSameGaussian(counted.state(), stepped.state());

    auto root = Emrys::SqrtKalmanFilter{model, Emrys::toSqrt(makeInitial()).value()};
    EXPECT_TRUE(root.predict(Steps{3}));
}

// A particle filter drawing from the same seed moves its particles the same, bitwise, over a step
// in seconds and in milliseconds; with a fixed-step model, predict(rng, Steps{3}) draws as three
// calls of predict(rng) do.
TEST(TimeSteps, ParticleFilterTakesDurationsAndAdvancesByACount)
{
    auto rng = std::mt19937_64{11};
    auto const initial = makeInitial();
    auto const particles =
        Emrys::ParticleSet<State>{State{}, Emrys::drawSamples(initial.toDensity().value(), 200, rng)};

    auto seconds = Emrys::ParticleFilter{makeModel(), particles};
    auto milliseconds = Emrys::ParticleFilter{makeModel(), particles};
    auto first = std::mt19937_64{13};
    auto second = std::mt19937_64{13};
    seconds.predict(first, timeStep);
    milliseconds.predict(second, timeStepInMilliseconds);
    expectSame(milliseconds.state().states(), seconds.state().states());

    auto stepped = Emrys::ParticleFilter{FixedStepModel{makeModel(), State{}, timeStep}, particles};
    auto counted = Emrys::ParticleFilter{stepped};
    for (int k = 0; k < 3; ++k)
    {
        stepped.predict(first);
    }
    counted.predict(second, Steps{3});
    expectSame(counted.state().states(), stepped.state().states());
}

// A Gaussian mixture filter predicts each component the same over a step in seconds and in
// milliseconds, through predict() and predictJoint(), and advances by a count with a fixed-step
// model as that many calls do.
TEST(TimeSteps, GaussianMixtureFilterTakesDurationsAndAdvancesByACount)
{
    auto prior = GaussianMixtureParams<State>{State{}};
    prior.add(std::log(0.7), makeInitial().mean(), makeInitial().covarianceLower());
    prior.add(std::log(0.3), Eigen::Vector4d{2, -1, 0, 1}, makeInitial().covarianceLower() * 2.0);
    auto seconds = Emrys::GaussianMixtureFilter{makeModel(), prior};
    auto milliseconds = Emrys::GaussianMixtureFilter{makeModel(), prior};
    seconds.predict(timeStep);
    milliseconds.predict(timeStepInTicks);
    auto const joint = seconds.predictJoint(timeStep);
    auto const jointInMilliseconds = milliseconds.predictJoint(timeStepInMilliseconds);
    ASSERT_EQ(jointInMilliseconds.joints.size(), joint.joints.size());
    for (Index k = 0; k < seconds.state().size(); ++k)
    {
        expectSameGaussian(milliseconds.state()[k].component, seconds.state()[k].component);
        expectSame(jointInMilliseconds.joints[k].cross, joint.joints[k].cross);
    }

    auto stepped = Emrys::GaussianMixtureFilter{FixedStepModel{makeModel(), State{}, timeStep}, prior};
    auto counted = stepped;
    for (int k = 0; k < 4; ++k)
    {
        stepped.predict();
    }
    counted.predict(Steps{4});
    for (Index k = 0; k < stepped.state().size(); ++k)
    {
        expectSameGaussian(counted.state()[k].component, stepped.state()[k].component);
    }
}

// An IMM over two modes predicts the same over a step in seconds and in milliseconds, and with
// fixed-step models advances by a count as that many calls do, the chain stepping each time.
TEST(TimeSteps, ImmFilterTakesDurationsAndAdvancesByACount)
{
    struct Cruise
    {};

    struct Turn
    {};

    struct Maneuver : Named<Cruise, Turn>
    {};

    auto const chain = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.1, 0.9}}};
    auto const conditionals = std::vector<GaussianParams<State>>{makeInitial(), makeInitial()};
    auto const initial =
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0.6, 0.4}), conditionals};
    auto const quiet = makeModel();
    auto const agile = ConstantVelocity<Position, Velocity>{4.0};

    auto seconds = Emrys::ImmFilter{SwitchingModel{chain, quiet, agile}, initial};
    auto milliseconds = Emrys::ImmFilter{SwitchingModel{chain, quiet, agile}, initial};
    seconds.predict(timeStep);
    milliseconds.predict(timeStepInMilliseconds);
    expectSame(
        milliseconds.state().modes().toDensity().value().probabilities(),
        seconds.state().modes().toDensity().value().probabilities()
    );
    for (Index m = 0; m < 2; ++m)
    {
        expectSameGaussian(milliseconds.state().conditional(m), seconds.state().conditional(m));
    }

    auto stepped = Emrys::ImmFilter{
        SwitchingModel{chain, FixedStepModel{quiet, State{}, timeStep}, FixedStepModel{agile, State{}, timeStep}},
        initial
    };
    auto counted = stepped;
    for (int k = 0; k < 5; ++k)
    {
        stepped.predict();
    }
    counted.predict(Steps{5});
    expectSame(
        counted.state().modes().toDensity().value().probabilities(),
        stepped.state().modes().toDensity().value().probabilities()
    );
    for (Index m = 0; m < 2; ++m)
    {
        expectSameGaussian(counted.state().conditional(m), stepped.state().conditional(m));
    }
}

// The Rao-Blackwellized particle filter and point-mass filter predict the same over a step in
// seconds and in milliseconds, and advance by a count with a fixed-step model as that many calls
// do.
TEST(TimeSteps, RbFiltersTakeDurationsAndAdvanceByACount)
{
    auto rng = std::mt19937_64{17};
    auto const particles = Emrys::drawParticleGaussian<Position>(makeInitial(), 100, rng);
    auto seconds = Emrys::RbParticleFilter{makeModel(), particles};
    auto milliseconds = Emrys::RbParticleFilter{makeModel(), particles};
    auto first = std::mt19937_64{19};
    auto second = std::mt19937_64{19};
    seconds.predict(first, timeStep);
    milliseconds.predict(second, timeStepInMilliseconds);
    expectSame(milliseconds.state().particles().states(), seconds.state().particles().states());
    expectSameGaussian(milliseconds.state().conditional(7), seconds.state().conditional(7));

    auto stepped = Emrys::RbParticleFilter{FixedStepModel{makeModel(), State{}, timeStep}, particles};
    auto counted = stepped;
    stepped.predict(first);
    stepped.predict(first);
    counted.predict(second, Steps{2});
    expectSame(counted.state().particles().states(), stepped.state().particles().states());
    expectSameGaussian(counted.state().conditional(7), stepped.state().conditional(7));

    using Grid = Composite<Position, Bias, Drift>;
    using Rest = Composite<Bias, Drift>;
    auto const restModel = ConstantVelocity<Bias, Drift>{0.01};
    auto const rest =
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Vector2d{4, 0.01}.asDiagonal().toDenseMatrix()};
    auto const grid =
        Emrys::regularGrid<Grid, Position>(Eigen::Vector2d{-1.75, -1.75}, Eigen::Vector2d{0.5, 0.5}, {8, 8});
    auto cells = Emrys::RbPointMassFilter{restModel, grid, rest};
    auto cellsInMilliseconds = Emrys::RbPointMassFilter{restModel, grid, rest};
    cells.predict(timeStep);
    cellsInMilliseconds.predict(timeStepInMilliseconds);
    expectSameGaussian(cellsInMilliseconds.state().conditional(5), cells.state().conditional(5));

    auto steppedCells = Emrys::RbPointMassFilter{FixedStepModel{restModel, grid.restSpace(), timeStep}, grid, rest};
    auto countedCells = steppedCells;
    steppedCells.predict();
    steppedCells.predict();
    steppedCells.predict();
    countedCells.predict(Steps{3});
    expectSameGaussian(countedCells.state().conditional(5), steppedCells.state().conditional(5));
}

// A continuous-time chain's transition, a hidden Markov filter's predict() and predictJoint(), a
// Markov-modulated Poisson interval, and the Poisson and exponential emissions take a duration as
// they take seconds, bitwise.
TEST(TimeSteps, ContinuousTimeChainsAndEmissionsTakeDurations)
{
    struct Quiet
    {};

    struct Busy
    {};

    struct Regime : Named<Quiet, Busy>
    {};

    auto const chain = Emrys::ContinuousTimeMarkovChain<Regime>{Eigen::Matrix2d{{-0.4, 0.4}, {1.1, -1.1}}};
    expectSame(chain.transition(250ms).matrix(), chain.transition(0.25).matrix());

    auto const initial = CategoricalParams<Regime>::fromWeights(Regime{}, Eigen::Vector2d{0.7, 0.3});
    auto seconds = Emrys::HiddenMarkovFilter{chain, initial};
    auto milliseconds = Emrys::HiddenMarkovFilter{chain, initial};
    seconds.predict(0.25);
    milliseconds.predict(250ms);
    expectSame(
        milliseconds.state().toDensity().value().probabilities(),
        seconds.state().toDensity().value().probabilities()
    );

    auto const rates = Eigen::Array2d{0.5, 6.0};
    auto const interval = seconds.predict(0.37, Emrys::ArrivalInterval<Regime>{rates});
    auto const intervalInMilliseconds = milliseconds.predict(370ms, Emrys::ArrivalInterval<Regime>{rates});
    ASSERT_TRUE(interval && intervalInMilliseconds);
    EXPECT_EQ(*intervalInMilliseconds, *interval);
    expectSame(
        milliseconds.state().toDensity().value().probabilities(),
        seconds.state().toDensity().value().probabilities()
    );

    auto const joint = seconds.predictJoint(0.25);
    auto const jointInMilliseconds = milliseconds.predictJoint(250ms);
    expectSame(jointInMilliseconds.predicted.logWeights().exp(), joint.predicted.logWeights().exp());

    auto const count = Emrys::PoissonEmission<Regime>{rates, 3, 0.5};
    auto const countInMilliseconds = Emrys::PoissonEmission<Regime>{rates, 3, 500ms};
    EXPECT_EQ(countInMilliseconds.interval(), count.interval());
    auto const gap = Emrys::ExponentialEmission<Regime>{rates, 0.25};
    auto const gapInMilliseconds = Emrys::ExponentialEmission<Regime>{rates, 250ms};
    EXPECT_EQ(gapInMilliseconds.interval(), gap.interval());
}

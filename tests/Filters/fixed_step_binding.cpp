// fixedStep(model, dt) names no space, and each filter binds it to its state's space and scalar ([fixed-step-models]):
// a filter given it deduces the FixedStepModel over that space, a switching model of fixed steps is bound model by
// model, and the filter predicts as one given the FixedStepModel built over the space by hand, bitwise. simulate()
// takes the unbound step over the space it is given. Headers under test: Models/ProcessModel.h,
// Models/SwitchingModel.h, Models/Simulation.h, and the filters' deduction guides in Filters/.

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <type_traits>
#include <vector>

using Emrys::CategoricalParams;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::SwitchingModel;

namespace methods = Emrys::methods;

using namespace std::chrono_literals;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

struct Cruise
{};

struct Turn
{};

struct Maneuver : Named<Cruise, Turn>
{};

using State = Composite<Position, Velocity>;

constexpr double timeStep = 0.1;

template <typename Scalar = double>
auto
makeModel(Scalar density = Scalar(0.1))
{
    return ConstantVelocity<Position, Velocity, Scalar>{density};
}

using Motion = decltype(makeModel());

template <typename Scalar = double>
GaussianParams<State, Scalar>
makeInitial()
{
    using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << Scalar(4), Scalar(4), Scalar(1), Scalar(1);
    cov(0, 2) = cov(2, 0) = Scalar(1.2);
    return {State{}, Vector4{Scalar(0), Scalar(0), Scalar(1), Scalar(0.5)}, cov};
}

auto
makeHybrid()
{
    auto const conditionals = std::vector<GaussianParams<State>>{makeInitial(), makeInitial()};
    return HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Array2d{0.6, 0.4}), conditionals};
}

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

// fixedStep() keeps the model, the step as given, and the method, and is not itself a fixed-step
// model; the FixedStepModel over a fixed-size space converts from it to the one its constructor
// makes there, with the same noise and transition.
TEST(FixedStepBinding, AnUnboundStepConvertsToTheModelOverASpace)
{
    auto const step = Emrys::fixedStep(makeModel(), timeStep);
    using Step = std::remove_cvref_t<decltype(step)>;
    static_assert(Emrys::isUnboundStep<Step> && !Emrys::isFixedStep<Step>);
    static_assert(std::is_same_v<Step, Emrys::UnboundFixedStep<Motion, methods::Linearized, double>>);
    static_assert(std::is_same_v<Emrys::DefaultMethodFor<Step>, methods::Linearized>);
    EXPECT_EQ(step.timeStep(), timeStep);

    FixedStepModel<State, Motion> const bound = step;
    auto const byHand = FixedStepModel{makeModel(), State{}, timeStep};
    EXPECT_EQ(bound.timeStep(), timeStep);
    expectSame(bound.noise(), byHand.noise());
    expectSame(bound.transition(), byHand.transition());

    auto const unscented = Emrys::fixedStep(makeModel(), 100ms, methods::unscented<0.5>());
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(unscented)>,
                  Emrys::UnboundFixedStep<Motion, methods::Unscented<0.5>, std::chrono::milliseconds>>);
    FixedStepModel<State, Motion, methods::Unscented<0.5>> const boundUnscented = unscented;
    EXPECT_EQ(boundUnscented.method().alpha, 0.5);
    EXPECT_EQ(boundUnscented.timeStep(), timeStep);
}

// The Kalman filters deduce the FixedStepModel over their state's space and scalar, and predict as
// the filters given it by hand do, bitwise; a float state binds the step in float.
TEST(FixedStepBinding, KalmanFiltersBindTheStepToTheirState)
{
    auto bound = KalmanFilter{Emrys::fixedStep(makeModel(), timeStep), makeInitial()};
    auto byHand = KalmanFilter{FixedStepModel{makeModel(), State{}, timeStep}, makeInitial()};
    static_assert(std::is_same_v<decltype(bound), decltype(byHand)>);
    bound.predict();
    byHand.predict();
    expectSameGaussian(bound.state(), byHand.state());

    auto unscented = KalmanFilter{Emrys::fixedStep(makeModel(), timeStep, methods::unscented<0.5>()), makeInitial()};
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);

    auto single = KalmanFilter{Emrys::fixedStep(makeModel<float>(), timeStep), makeInitial<float>()};
    static_assert(std::is_same_v<
                  decltype(single)::ModelType,
                  FixedStepModel<State, decltype(makeModel<float>()), methods::Linearized, float>>);
    EXPECT_EQ(single.model().timeStep(), 0.1f);

    auto root = Emrys::SqrtKalmanFilter{Emrys::fixedStep(makeModel(), timeStep), Emrys::toSqrt(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(root)::ModelType, FixedStepModel<State, Motion>>);
    EXPECT_TRUE(root.predict());

    auto info = Emrys::InfoFilter{Emrys::fixedStep(makeModel(), timeStep), Emrys::toInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(info)::ModelType, FixedStepModel<State, Motion>>);
    EXPECT_TRUE(info.predict());

    auto rootInfo =
        Emrys::SqrtInfoFilter{Emrys::fixedStep(makeModel(), timeStep), Emrys::toSqrtInfo(makeInitial()).value()};
    static_assert(std::is_same_v<decltype(rootInfo)::ModelType, FixedStepModel<State, Motion>>);
    EXPECT_TRUE(rootInfo.predict());
}

// The Gaussian mixture filter binds the step to its components' space.
TEST(FixedStepBinding, AMixtureFilterBindsTheStepToItsComponents)
{
    auto prior = Emrys::GaussianMixtureParams<State>{State{}};
    prior.add(0.0, makeInitial().mean(), makeInitial().covarianceLower());
    auto bound = Emrys::GaussianMixtureFilter{Emrys::fixedStep(makeModel(), timeStep), prior};
    auto byHand = Emrys::GaussianMixtureFilter{FixedStepModel{makeModel(), State{}, timeStep}, prior};
    static_assert(std::is_same_v<decltype(bound), decltype(byHand)>);
    bound.predict();
    byHand.predict();
    expectSameGaussian(bound.state().begin()->component, byHand.state().begin()->component);
}

// A switching model of fixedStep() results holds them unbound, and the IMM and AMM filters bind
// each to the state's space, as the switching model of FixedStepModel built by hand.
TEST(FixedStepBinding, SwitchingFiltersBindEachModel)
{
    auto const chain = MarkovChain<Maneuver>{Eigen::Matrix2d{{0.95, 0.05}, {0.1, 0.9}}};
    auto const quiet = Emrys::fixedStep(makeModel(), timeStep);
    auto const agile = Emrys::fixedStep(makeModel(4.0), timeStep);
    auto const unbound = SwitchingModel{chain, quiet, agile};
    static_assert(!Emrys::isFixedStep<std::remove_cvref_t<decltype(unbound)>>);

    using Bound =
        SwitchingModel<Maneuver, double, Maneuver, FixedStepModel<State, Motion>, FixedStepModel<State, Motion>>;
    auto bound = Emrys::ImmFilter{unbound, makeHybrid()};
    static_assert(std::is_same_v<decltype(bound)::ModelType, Bound>);
    static_assert(Emrys::isFixedStep<decltype(bound)::ModelType>);
    auto byHand = Emrys::ImmFilter{
        SwitchingModel{
            chain,
            FixedStepModel{makeModel(), State{}, timeStep},
            FixedStepModel{makeModel(4.0), State{}, timeStep}
        },
        makeHybrid()
    };
    bound.predict();
    byHand.predict();
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        expectSameGaussian(bound.state().conditional(m), byHand.state().conditional(m));
    }

    auto adaptive = Emrys::AmmFilter{makeHybrid(), quiet, agile};
    static_assert(std::is_same_v<decltype(adaptive)::ModelType, Bound>);
    adaptive.predict();
}

// The particle filters bind the step to the space their process model moves: the whole state for
// the particle filter and the Rao-Blackwellized particle filter, and the rest of the state for the
// point-mass filter.
TEST(FixedStepBinding, ParticleFiltersBindTheStepToTheSpaceTheyMove)
{
    auto rng = std::mt19937_64{3};
    auto const initial = makeInitial();
    auto const particles = Emrys::ParticleSet<State>{State{}, Emrys::drawSamples(initial.toDensity().value(), 50, rng)};
    auto particleFilter = Emrys::ParticleFilter{Emrys::fixedStep(makeModel(), timeStep), particles};
    static_assert(std::is_same_v<decltype(particleFilter)::ModelType, FixedStepModel<State, Motion>>);
    particleFilter.predict(rng);

    auto const split = Emrys::drawParticleGaussian<Position>(initial, 50, rng);
    auto rb = Emrys::RbParticleFilter{Emrys::fixedStep(makeModel(), timeStep), split};
    static_assert(std::is_same_v<decltype(rb)::ModelType, FixedStepModel<State, Motion>>);
    rb.predict(rng);

    using Grid = Composite<Position, Bias, Drift>;
    using Rest = Composite<Bias, Drift>;
    auto const restModel = ConstantVelocity<Bias, Drift>{0.01};
    auto const rest =
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Vector2d{4, 0.01}.asDiagonal().toDenseMatrix()};
    auto const grid =
        Emrys::regularGrid<Grid, Position>(Eigen::Vector2d{-1.75, -1.75}, Eigen::Vector2d{0.5, 0.5}, {8, 8});
    auto cells = Emrys::RbPointMassFilter{Emrys::fixedStep(restModel, timeStep), grid, rest};
    auto cellsByHand = Emrys::RbPointMassFilter{FixedStepModel{restModel, grid.restSpace(), timeStep}, grid, rest};
    static_assert(std::is_same_v<decltype(cells), decltype(cellsByHand)>);
    cells.predict();
    cellsByHand.predict();
    expectSameGaussian(cells.state().conditional(5), cellsByHand.state().conditional(5));
}

// simulate() takes fixedStep()'s result over the space it is given, drawing as the model over the
// step does from the same seed.
TEST(FixedStepBinding, SimulateTakesTheUnboundStep)
{
    auto const x = makeInitial().mean();
    auto first = std::mt19937_64{5};
    auto second = std::mt19937_64{5};
    auto third = std::mt19937_64{5};
    Eigen::Vector4d const unbound = Emrys::simulate(State{}, x, Emrys::fixedStep(makeModel(), timeStep), first);
    Eigen::Vector4d const stepped = Emrys::simulate(State{}, x, makeModel(), second, timeStep);
    Eigen::Vector4d const named = Emrys::simulate<State>(x, Emrys::fixedStep(makeModel(), timeStep), third);
    expectSame(unbound, stepped);
    expectSame(named, stepped);
}

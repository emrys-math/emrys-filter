// RbParticleFilter and ParticleGaussian (steps 8b, 9a, 13d; [rao-blackwellized-core], [rb-particle-filter],
// [fixed-step-models], [likelihood-measurements]): deduction and accessors, a draw of the Rao-Blackwellized form of a
// Gaussian, a state assigned to itself unchanged, prediction conditions the rest on each draw, a fixed-step model
// matches the step, prediction agrees across methods on a linear model, periodic blocks wrap, a fix weighs each
// particle by its likelihood, a likelihood over the particles leaves their Gaussians, SharedCovariance matches
// PerPointCovariance on a linear problem, upkeep carries each particle's Gaussian, a gate zeroes contradicted
// particles, the filter matches KalmanFilter on a linear problem and strays less than a ParticleFilter, and it learns a
// velocity from ranges. Under a switching model ([imm]), hybrid particles match an IMM filter per particle, reduce to
// one Gaussian per particle when the modes share their dynamics, sum, collapse, and resample their hybrids, and fail a
// prediction whose draw has no density.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <sstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::drawParticleGaussian;
using Emrys::GaussianParams;
using Emrys::given;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::ParticleFilter;
using Emrys::ParticleGaussianParams;
using Emrys::ParticleSet;
using Emrys::PerPointCovariance;
using Emrys::RbParticleFilter;
using Emrys::ScalarOf;
using Emrys::SharedCovariance;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;
namespace upkeep = Emrys::upkeep;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Vector1 = Eigen::Matrix<double, 1, 1>;

constexpr double pi = std::numbers::pi;
constexpr double accelDensity = 0.1;
constexpr double timeStep = 1.0;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

// Position and velocity correlated, so the velocity given a position depends on it.
GaussianParams<State>
makeInitial()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    cov(1, 3) = cov(3, 1) = -0.8;
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, cov};
}

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

auto
makeFix(Eigen::Vector2d const & z)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R})
        ),
        z
    );
}

std::array<Eigen::Vector2d, 3> const beacons{
    {Eigen::Vector2d{-40, -30}, Eigen::Vector2d{60, -20}, Eigen::Vector2d{10, 70}}
};

// A range to a beacon: nonlinear in position, independent of velocity.
auto
makeRange(double z, Eigen::Vector2d const & beacon)
{
    auto const range = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{
                Eigen::numext::sqrt(T((Emrys::block<Position>(in, x) - beacon.cast<T>()).squaredNorm()))
            };
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.25}})
    );
    return observe(methods::linearized(), range, Vector1{z});
}

// The range rate to a beacon: nonlinear in position, linear in velocity with a Jacobian that
// depends on the position, so each particle's Gaussian needs its own covariance.
auto
makeRangeRate(double z, Eigen::Vector2d const & beacon)
{
    auto const rangeRate = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{
                d.dot(Emrys::block<Velocity>(in, x)) / Eigen::numext::sqrt(T(d.squaredNorm()))
            };
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.01}})
    );
    return observe(methods::linearized(), rangeRate, Vector1{z});
}

// The range rate with an outlier component in its noise: 0.9 of variance 0.01 and 0.1 of 1.
auto
makeRobustRangeRate(double z, Eigen::Vector2d const & beacon)
{
    auto noise = Emrys::GaussianMixtureParams<Linear<1>>{Linear<1>{}};
    noise.add(std::log(0.9), Vector1::Zero(), Vector1{0.01});
    noise.add(std::log(0.1), Vector1::Zero(), Vector1{1.0});
    auto const rangeRate = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{
                d.dot(Emrys::block<Velocity>(in, x)) / Eigen::numext::sqrt(T(d.squaredNorm()))
            };
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        additiveNoise(std::move(noise))
    );
    return observe(methods::linearized(), rangeRate, Vector1{z});
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

// Checks moments against a Gaussian's, to within `sigmas` standard errors of `count`
// independent draws, which bound those of Rao-Blackwellized particles.
template <typename Moments, typename Mean, typename Cov>
void
expectMomentsNear(Moments const & sample, Mean const & mean, Cov const & cov, double count, double sigmas)
{
    for (Eigen::Index i = 0; i < mean.size(); ++i)
    {
        EXPECT_NEAR(sample.mean()(i), mean(i), sigmas * std::sqrt(cov(i, i) / count)) << "mean " << i;
        for (Eigen::Index j = 0; j < mean.size(); ++j)
        {
            double const standardError = std::sqrt((cov(i, i) * cov(j, j) + cov(i, j) * cov(i, j)) / count);
            EXPECT_NEAR(sample.covarianceLower()(i, j), cov(i, j), sigmas * standardError) << "cov " << i << ", " << j;
        }
    }
}

// What operator<< writes, as a string.
template <typename T>
std::string
printed(T const & value)
{
    auto stream = std::ostringstream{};
    stream << value;
    return stream.str();
}

// Everything after the first line of a description.
std::string
afterFirstLine(std::string const & text)
{
    return text.substr(text.find('\n'));
}

}  // namespace

// The deduction guide picks the space, path, model, covariance form, and method from the
// ParticleGaussian and optional method, and the state reports its pinned rows and the indices of
// the rest.
TEST(RbParticleFilter, DeducesItsTypeAndHoldsItsParts)
{
    auto rng = std::mt19937_64{1};
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), 10, rng)};
    static_assert(std::is_same_v<decltype(filter)::StateType, ParticleGaussianParams<State, Position>>);
    static_assert(std::is_same_v<decltype(filter)::MethodType, methods::Linearized>);
    auto unscented = RbParticleFilter{
        makeModel(),
        drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 10, rng),
        methods::unscented()
    };
    static_assert(
        std::is_same_v<decltype(unscented)::StateType, ParticleGaussianParams<State, Position, PerPointCovariance>>
    );
    static_assert(std::is_same_v<decltype(unscented)::MethodType, methods::Unscented<>>);
    EXPECT_EQ(filter.state().size(), 10);
    EXPECT_EQ(filter.state().pinnedRows().start, 0);
    EXPECT_EQ(filter.state().restIndices(), (Eigen::Vector2<Eigen::Index>{2, 3}));
    static_assert(std::is_same_v<decltype(unscented.method()), methods::Unscented<> const &>);
    EXPECT_EQ(&unscented.method(), &std::as_const(unscented).method());
    EXPECT_EQ(&unscented.model(), &std::as_const(unscented).model());
}

// setState() replaces the state with one of another number of particles, under both covariance
// forms: the steps that follow equal those of a filter made with the new state, from the same
// generator.
template <typename Covariance>
void
expectSetStateReplacesTheState()
{
    auto rng = std::mt19937_64{3};
    auto const replacement = drawParticleGaussian<Position, Covariance>(makeInitial(), 40, rng);
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position, Covariance>(makeInitial(), 10, rng)};
    auto reference = RbParticleFilter{makeModel(), replacement};
    auto const fix = makeFix(Eigen::Vector2d{1.0, 0.5});
    filter.predict(rng, timeStep);
    ASSERT_TRUE(filter.update(fix));

    filter.setState(replacement);
    ASSERT_EQ(filter.state().size(), 40);
    for (auto * const each : {&filter, &reference})
    {
        auto stepRng = std::mt19937_64{4};
        each->predict(stepRng, timeStep);
        ASSERT_TRUE(each->update(fix));
    }
    EXPECT_EQ(filter.state().particles().states(), reference.state().particles().states());
    EXPECT_TRUE((filter.state().particles().logWeights() == reference.state().particles().logWeights()).all());
    for (Eigen::Index i = 0; i < filter.state().size(); ++i)
    {
        EXPECT_EQ(filter.state().conditional(i).mean(), reference.state().conditional(i).mean()) << i;
        EXPECT_EQ(filter.state().conditional(i).covarianceLower(), reference.state().conditional(i).covarianceLower())
            << i;
    }
}

TEST(RbParticleFilter, SetStateReplacesTheState)
{
    expectSetStateReplacesTheState<SharedCovariance>();
    expectSetStateReplacesTheState<PerPointCovariance>();
}

// drawParticleGaussian<Position>() draws positions from the joint's marginal and gives each the
// velocity conditional (Schur complement covariance, shared; mean by the gain), so the collapsed
// moments match the joint; a per-particle state starts with the same covariance everywhere.
TEST(ParticleGaussian, DrawsTheRbFormOfAGaussian)
{
    auto rng = std::mt19937_64{2};
    GaussianParams<State> const joint = makeInitial();
    constexpr Eigen::Index count = 40000;
    auto const state = drawParticleGaussian<Position>(joint, count, rng);
    Eigen::Matrix2d const ppp = joint.covarianceLower().topLeftCorner<2, 2>();
    Eigen::Matrix2d const pvp = joint.covarianceLower().bottomLeftCorner<2, 2>();
    Eigen::Matrix2d const gain = pvp * ppp.inverse();
    EXPECT_TRUE(state.sharedCovariance()
                    .isApprox(joint.covarianceLower().bottomRightCorner<2, 2>() - gain * pvp.transpose(), 1e-12));
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        Eigen::Vector2d const expected = joint.mean().tail<2>() + gain * (state.particle(i) - joint.mean().head<2>());
        EXPECT_TRUE(state.conditional(i).mean().isApprox(expected, 1e-12)) << i;
    }
    expectMomentsNear(state.toDensity()->collapsed(), joint.mean(), joint.covarianceLower(), double(count), 5);

    // Every particle of a per-particle state starts with the same covariance.
    auto same = std::mt19937_64{2};
    auto const perParticle = drawParticleGaussian<Position, PerPointCovariance>(joint, count, same);
    EXPECT_EQ(perParticle.particle(7), state.particle(7));
    EXPECT_EQ(perParticle.conditional(7).mean(), state.conditional(7).mean());
    EXPECT_EQ(perParticle.conditional(7).covarianceLower(), state.sharedCovariance());
}

// Assigning a state to itself changes nothing.
TEST(ParticleGaussian, SelfAssignmentChangesNothing)
{
    auto rng = std::mt19937_64{4};
    auto state = drawParticleGaussian<Position>(makeInitial(), 5, rng);
    auto const copy = state;
    auto & alias = state;
    state = alias;
    EXPECT_EQ(state.sharedCovariance(), copy.sharedCovariance());
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        EXPECT_EQ(state.particle(i), copy.particle(i)) << i;
        EXPECT_EQ(state.conditional(i).mean(), copy.conditional(i).mean()) << i;
    }
}

// particleGaussian<Position>() gives each particle one Gaussian over the rest, and a per-particle
// state takes one per particle from any contiguous storage. The state reports its pinned space,
// the difference indices of the rest, and the total weight and effective sample size of its
// particles, with or without a policy.
TEST(ParticleGaussian, BuildsFromParticlesAndGaussians)
{
    auto rng = std::mt19937_64{6};
    auto const drawn = drawParticleGaussian<Position>(makeInitial(), 3, rng);
    auto const shared = Emrys::particleGaussian<Position>(State{}, drawn.particles(), drawn.conditional(0));
    static_assert(std::is_same_v<decltype(shared), decltype(drawn)>);
    EXPECT_EQ(shared.conditional(2).mean(), drawn.conditional(0).mean());
    EXPECT_EQ(shared.pinnedSpace(), Position{});
    EXPECT_EQ(shared.restDifferenceIndices(), (Eigen::Vector2<Eigen::Index>{2, 3}));
    EXPECT_NEAR(shared.logTotalWeight().value(), 0.0, 1e-15);
    EXPECT_NEAR(shared.logTotalWeight(Emrys::execution::sequential).value(), 0.0, 1e-15);
    EXPECT_NEAR(shared.ess(Emrys::execution::sequential).value(), 3.0, 1e-12);

    auto const conditionals = std::vector{drawn.conditional(0), drawn.conditional(1), drawn.conditional(2)};
    auto const perParticle =
        ParticleGaussianParams<State, Position, PerPointCovariance>{State{}, drawn.particles(), conditionals};
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_EQ(perParticle.conditional(i).mean(), drawn.conditional(i).mean()) << i;
    }
}

// The collapse is the density's (decisions.md [params-and-density]): toDensity() checks the weights
// and factorizes each particle's Gaussian, so a covariance that isn't positive definite fails it,
// in either form, rather than giving a collapse of no distribution. A filter's step changes the
// state, so a density made before it is stale [view-stamp].
TEST(ParticleGaussian, ItsDensityChecksTheGaussians)
{
    auto rng = std::mt19937_64{6};
    auto const drawn = drawParticleGaussian<Position>(makeInitial(), 3, rng);
    auto const density = drawn.toDensity();
    ASSERT_TRUE(density);
    EXPECT_EQ(density->moments().mean(), density->collapsed().mean());
    EXPECT_EQ(density->size(), 3);

    auto broken = drawn.conditional(1);
    broken.setCovariance(Eigen::Matrix2d{{1.0, 2.0}, {2.0, 1.0}});
    auto const shared = Emrys::particleGaussian<Position>(State{}, drawn.particles(), broken);
    EXPECT_EQ(shared.toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
    auto const conditionals = std::vector{drawn.conditional(0), broken, drawn.conditional(2)};
    auto const perParticle =
        ParticleGaussianParams<State, Position, PerPointCovariance>{State{}, drawn.particles(), conditionals};
    EXPECT_EQ(perParticle.toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
TEST(ParticleGaussianDeathTest, ADensityReadAfterAStepFails)
{
    auto rng = std::mt19937_64{6};
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), 10, rng)};
    auto const before = filter.state().toDensity();
    ASSERT_TRUE(before);
    EXPECT_DEATH(
        {
            filter.predict(rng, 0.1);
            std::ignore = before->collapsed();
        },
        "changed while a view"
    );
    auto const beforeUpdate = filter.state().toDensity();
    ASSERT_TRUE(beforeUpdate);
    EXPECT_DEATH(
        {
            std::ignore = filter.update(makeFix(Eigen::Vector2d{1.0, 0.5}));
            std::ignore = beforeUpdate->collapsed();
        },
        "changed while a view"
    );
}
#endif

// A particle of weight zero takes no part in the collapse; weights that fail to define a
// distribution fail a resample, which leaves the state as it was, and print as their error.
TEST(ParticleGaussian, WeightsOfZeroAndWeightsThatFail)
{
    double const inf = std::numeric_limits<double>::infinity();
    auto rng = std::mt19937_64{5};
    auto const drawn = drawParticleGaussian<Position>(makeInitial(), 2, rng);
    using State2 = std::remove_cvref_t<decltype(drawn)>;
    auto const positions = Eigen::Matrix<double, 2, 2>{{0.0, 9.0}, {0.0, 9.0}};
    auto const oneDead = State2{
        State{},
        typename State2::ParticlesType{Position{}, positions, Eigen::Array2d{0.0, -inf}},
        drawn.conditional(0)
    };
    EXPECT_EQ(oneDead.toDensity()->collapsed().mean().head<2>(), Eigen::Vector2d::Zero());

    auto allDead = State2{
        State{},
        typename State2::ParticlesType{Position{}, positions, Eigen::Array2d{-inf, -inf}},
        drawn.conditional(0)
    };
    EXPECT_EQ(allDead.resample(rng, 2).error(), Emrys::WeightError::ZeroSum);
    EXPECT_EQ(allDead.size(), 2);
    auto stream = std::ostringstream{};
    stream << allDead;
    EXPECT_NE(stream.str().find("every weight is zero"), std::string::npos) << stream.str();
}

// The prediction of particle i done by hand: the whole state predicted with the particle known,
// its position drawn from the marginal, and the velocity conditioned on the draw.
template <typename Covariance>
void
expectPredictConditionsTheRestOnEachDraw()
{
    auto rng = std::mt19937_64{4};
    auto initial = drawParticleGaussian<Position, Covariance>(makeInitial(), 3, rng);
    auto filter = RbParticleFilter{makeModel(), initial};
    auto copy = std::mt19937_64{rng};
    filter.predict(rng, timeStep);
    using Draws = Emrys::SampleSet<Position>;
    Draws const z = Emrys::internal::standardNormal<Draws>(2, 3, copy);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        auto mean = Vector4{};
        mean << initial.particle(i), initial.conditional(i).mean();
        Matrix4 cov = Matrix4::Zero();
        cov.bottomRightCorner<2, 2>() = initial.conditional(i).covarianceLower();
        auto const joint =
            Emrys::predict(methods::linearized(), GaussianParams<State>{State{}, mean, cov}, makeModel(), timeStep);
        Eigen::Matrix2d const ppp = joint.covarianceLower().topLeftCorner<2, 2>();
        Eigen::Matrix2d const pvp = joint.covarianceLower().bottomLeftCorner<2, 2>();
        Eigen::Matrix2d const gain = pvp * ppp.inverse();
        Eigen::Vector2d const offset = Emrys::internal::defaultRoot(ppp) * z.col(i);
        EXPECT_TRUE(filter.state().particle(i).isApprox(joint.mean().head<2>() + offset, 1e-12)) << i;
        EXPECT_TRUE(filter.state().conditional(i).mean().isApprox(joint.mean().tail<2>() + gain * offset, 1e-12)) << i;
        EXPECT_TRUE(filter.state().conditional(i).covarianceLower().isApprox(
            joint.covarianceLower().bottomRightCorner<2, 2>() - gain * pvp.transpose(),
            1e-12
        )) << i;
        EXPECT_EQ(filter.state().logWeight(i), initial.logWeight(i));
    }
}

// For each particle under both covariance forms, predict() equals the hand computation: the joint
// predicted with the particle known, the position drawn from its marginal with the same standard
// normals, and the velocity conditioned on the draw, with the weight kept.
TEST(RbParticleFilter, PredictConditionsTheRestOnEachDraw)
{
    expectPredictConditionsTheRestOnEachDraw<SharedCovariance>();
    expectPredictConditionsTheRestOnEachDraw<PerPointCovariance>();
}

template <typename Covariance>
void
expectFixedStepMatchesTheStep()
{
    using Emrys::FixedStepModel;

    auto rng = std::mt19937_64{21};
    auto const initial = drawParticleGaussian<Position, Covariance>(makeInitial(), 20, rng);
    auto cached = Emrys::RbParticleFilter{FixedStepModel{makeModel(), State{}, timeStep}, initial};
    static_assert(std::is_same_v<typename decltype(cached)::MethodType, methods::Linearized>);
    auto unscented =
        Emrys::RbParticleFilter{FixedStepModel{makeModel(), State{}, timeStep, methods::unscented<0.5>()}, initial};
    static_assert(std::is_same_v<typename decltype(unscented)::MethodType, methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    auto continuousCached = RbParticleFilter{makeModel(), initial};
    auto continuousUnscented = RbParticleFilter{makeModel(), initial, methods::unscented<0.5>()};

    auto const expectSame = [](auto const & filter, auto const & expected, double tolerance) {
        for (Eigen::Index i = 0; i < 20; ++i)
        {
            auto const & state = filter.state();
            auto const & other = expected.state();
            EXPECT_LE((state.particle(i) - other.particle(i)).norm(), tolerance) << i;
            EXPECT_LE((state.conditional(i).mean() - other.conditional(i).mean()).norm(), tolerance) << i;
            EXPECT_LE(
                (state.conditional(i).covarianceLower() - other.conditional(i).covarianceLower()).norm(),
                tolerance
            ) << i;
            EXPECT_NEAR(state.logWeight(i), other.logWeight(i), tolerance) << i;
        }
    };
    auto draws = std::array<std::mt19937_64, 4>{
        std::mt19937_64{22},
        std::mt19937_64{22},
        std::mt19937_64{22},
        std::mt19937_64{22}
    };
    auto const fixes = std::vector<Eigen::Vector2d>{{1.2, 0.4}, {1.9, 1.1}, {3.2, 1.4}};
    for (Eigen::Vector2d const & z : fixes)
    {
        cached.predict(draws[0]);
        continuousCached.predict(draws[1], timeStep);
        unscented.predict(draws[2]);
        continuousUnscented.predict(draws[3], timeStep);
        ASSERT_TRUE(cached.update(makeFix(z)));
        ASSERT_TRUE(continuousCached.update(makeFix(z)));
        ASSERT_TRUE(unscented.update(makeFix(z)));
        ASSERT_TRUE(continuousUnscented.update(makeFix(z)));
        expectSame(cached, continuousCached, 1e-10);
        expectSame(unscented, continuousUnscented, 0.0);
    }
}

// With a FixedStepModel [fixed-step-models], under both covariance forms, the filter deduces through the
// abbreviation, takes the model's method, and from the same particles and seed matches the same
// filter predicting over the step over three fixes: exactly under Unscented, which takes the same
// path with the same noise, and to rounding under Linearized, which predicts given each particle by
// the cached transition's columns for the rest rather than a transform. The tolerance is looser
// than the Kalman filter's because a rounding difference in a covariance moves each draw.
TEST(RbParticleFilter, AFixedStepModelMatchesTheStep)
{
    expectFixedStepMatchesTheStep<SharedCovariance>();
    expectFixedStepMatchesTheStep<PerPointCovariance>();
}

// On a linear model both methods predict the same: the derivatives Linearized takes and the sigma
// points Unscented spreads with the particle held fixed.
TEST(RbParticleFilter, EveryMethodPredictsTheSameOnALinearModel)
{
    auto const run = [](auto method) {
        auto rng = std::mt19937_64{12};
        auto filter = RbParticleFilter{
            makeModel(),
            drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 20, rng),
            method
        };
        filter.predict(rng, timeStep);
        filter.predict(rng, 0.5);
        return filter.state();
    };
    auto const linearized = run(methods::linearized());
    auto const unscented = run(methods::unscented());
    for (Eigen::Index i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(unscented.particle(i).isApprox(linearized.particle(i), 1e-9)) << i;
        EXPECT_TRUE(unscented.conditional(i).mean().isApprox(linearized.conditional(i).mean(), 1e-9)) << i;
        EXPECT_TRUE(
            unscented.conditional(i).covarianceLower().isApprox(linearized.conditional(i).covarianceLower(), 1e-9)
        ) << i;
    }
}

// Predicting a particle block of headings near pi keeps them in [-pi, pi), most having crossed the
// seam, with the collapsed mean just past it.
TEST(RbParticleFilter, PredictWrapsAPeriodicBlock)
{
    struct Heading : Emrys::RadiansPi<1>
    {};

    struct Rate : Linear<1>
    {};

    using Turning = Composite<Heading, Rate>;
    auto const model = ConstantVelocity<Heading, Rate>{0.01};
    auto const initial =
        GaussianParams<Turning>{Turning{}, Eigen::Vector2d{pi - 0.05, 0.2}, Eigen::Matrix2d{{0.01, 0}, {0, 0.001}}};
    auto rng = std::mt19937_64{7};
    auto filter = RbParticleFilter{model, drawParticleGaussian<Heading>(initial, 1000, rng)};
    filter.predict(rng, 1.0);
    auto const headings = filter.state().particles().states().row(0);
    EXPECT_GE(headings.minCoeff(), -pi);
    EXPECT_LT(headings.maxCoeff(), pi);
    // Most particles crossed the seam; the tangent-space mean sits just past it.
    EXPECT_GT((headings.array() < 0).count(), 800);
    EXPECT_NEAR(filter.state().toDensity()->collapsed().mean()(0), -pi + 0.15, 0.02);
}

// After one prediction of 40,000 particles the collapsed moments are within five standard errors of
// the affine Kalman prediction.
TEST(RbParticleFilter, PredictMatchesTheKalmanPredictionInDistribution)
{
    auto rng = std::mt19937_64{5};
    constexpr Eigen::Index count = 40000;
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), count, rng)};
    auto const exact = Emrys::predict(methods::linearized(), makeInitial(), makeModel(), timeStep);
    filter.predict(rng, timeStep);
    expectMomentsNear(filter.state().toDensity()->collapsed(), exact.mean(), exact.covarianceLower(), double(count), 5);
}

// Under noise of two components, a particle's update collapses its two branches, its likelihood is
// theirs together, and its NIS is that of the branch more probable given the measurement: the
// narrow one for a measurement near the prediction, the wide one for a measurement far from it.
TEST(RbParticleFilter, MixtureNoiseReportsTheMostProbableBranchsNis)
{
    for (std::size_t const outlier : {0u, 1u})
    {
        auto rng = std::mt19937_64{7};
        auto filter =
            RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 5, rng)};
        auto const prior = filter.state();
        auto const rate = makeRobustRangeRate(outlier == 0 ? 0.7 : 30.0, beacons[0]);
        auto const result = filter.update(rate);
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->accepted);
        for (Eigen::Index i = 0; i < 5; ++i)
        {
            Eigen::Vector2d const p = prior.particle(i);
            auto const expected = update(prior.conditional(i), rate, given<Position>(State{}, p));
            ASSERT_TRUE(expected) << i;
            ASSERT_EQ(expected->branches.size(), 2u);
            std::size_t const best = expected->branches[1].logWeight > expected->branches[0].logWeight ? 1 : 0;
            EXPECT_EQ(best, outlier) << i;
            EXPECT_NEAR(result->logLikelihoods()(i), expected->logLikelihood(), 1e-12) << i;
            EXPECT_NEAR(result->nis()(i), expected->branches[best].nis(), 1e-12) << i;
        }
    }
}

// A range-rate update with per-particle covariances gives each particle the Kalman update of its
// Gaussian given its position, with the matching log-likelihood and NIS, multiplies its weight by
// the likelihood, and returns the log-evidence.
TEST(RbParticleFilter, AFixWeighsEachParticleByItsLikelihood)
{
    auto rng = std::mt19937_64{6};
    auto filter =
        RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 5, rng)};
    auto const prior = filter.state();
    auto const rate = makeRangeRate(0.7, beacons[0]);
    auto const result = filter.update(rate);
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    Eigen::ArrayXd logWeights = prior.logWeights();
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        Eigen::Vector2d const p = prior.particle(i);
        auto const expected = update(prior.conditional(i), rate, given<Position>(State{}, p));
        ASSERT_TRUE(expected) << i;
        EXPECT_NEAR(result->logLikelihoods()(i), expected->logLikelihood(), 1e-12) << i;
        EXPECT_NEAR(result->nis()(i), expected->branches[0].nis(), 1e-12) << i;
        EXPECT_TRUE(filter.state().conditional(i).mean().isApprox(expected->branches[0].posterior.mean(), 1e-12)) << i;
        EXPECT_TRUE(filter.state().conditional(i).covarianceLower().isApprox(
            expected->branches[0].posterior.covarianceLower(),
            1e-12
        )) << i;
        EXPECT_EQ(filter.state().particle(i), prior.particle(i));
        logWeights(i) += expected->logLikelihood();
    }
    double const evidence = std::log(logWeights.exp().sum());
    EXPECT_NEAR(result->logLikelihood(), evidence, 1e-12);
    EXPECT_TRUE(filter.state().logWeights().isApprox(logWeights - evidence, 1e-12));
}

namespace
{

// The log-likelihood of a position fix z with Laplace noise of scale 0.5 on each axis at p.
double
laplaceLogLikelihood(Eigen::Vector2d const & z, Eigen::Vector2d const & p)
{
    return -2 * std::log(2 * 0.5) - (z - p).lpNorm<1>() / 0.5;
}

// A position fix with Laplace noise of scale 0.5 on each axis, known only by its likelihood.
auto
makeLaplaceFix(Eigen::Vector2d const & z)
{
    return observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Position>),
            Emrys::logLikelihood([](auto const & in, auto const & x, auto const & observed) {
                return laplaceLogLikelihood(observed, Emrys::block<Position>(in, x));
            })
        ),
        z
    );
}

template <typename Covariance>
void
checkLikelihoodOverTheParticles()
{
    auto rng = std::mt19937_64{7};
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position, Covariance>(makeInitial(), 20, rng)};
    filter.predict(rng, 1.0);
    auto const prior = filter.state();
    auto const z = Eigen::Vector2d{0.5, -0.5};
    auto const fix = makeLaplaceFix(z);
    auto const result = filter.update(fix);
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    EXPECT_EQ(result->innovationSize(), 0);
    EXPECT_TRUE(result->nis().isNaN().all());
    Eigen::ArrayXd logWeights = prior.logWeights();
    for (Eigen::Index i = 0; i < prior.size(); ++i)
    {
        Eigen::Vector2d const p = prior.particle(i);
        EXPECT_EQ(result->logLikelihoods()(i), laplaceLogLikelihood(z, p)) << i;
        EXPECT_EQ(filter.state().particle(i), prior.particle(i)) << i;
        EXPECT_EQ(filter.state().conditional(i).mean(), prior.conditional(i).mean()) << i;
        EXPECT_EQ(filter.state().conditional(i).covarianceLower(), prior.conditional(i).covarianceLower()) << i;
        logWeights(i) += laplaceLogLikelihood(z, p);
    }
    double const evidence = std::log(logWeights.exp().sum());
    EXPECT_NEAR(result->logLikelihood(), evidence, 1e-12);
    EXPECT_TRUE(filter.state().logWeights().isApprox(logWeights - evidence, 1e-12));
}

}  // namespace

// A likelihood over the sampled block alone multiplies each particle's weight by the likelihood at
// the particle and leaves every particle and its Gaussian exactly as they were, with NIS NaN and
// no innovation, under both covariance forms ([likelihood-measurements]).
TEST(RbParticleFilter, ALikelihoodOverTheParticlesLeavesTheRest)
{
    checkLikelihoodOverTheParticles<SharedCovariance>();
    checkLikelihoodOverTheParticles<PerPointCovariance>();
}

// With a linear measurement, SharedCovariance and PerPointCovariance filters driven by the same
// seed agree on every particle, weight, conditional mean, and covariance over five steps.
TEST(RbParticleFilter, SharedCovarianceMatchesPerParticleOnALinearProblem)
{
    auto sharedRng = std::mt19937_64{7};
    auto perRng = std::mt19937_64{7};
    auto shared = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), 50, sharedRng)};
    auto per =
        RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 50, perRng)};
    Vector4 truth = makeInitial().mean();
    for (int step = 1; step <= 5; ++step)
    {
        truth = transitionMatrix() * truth;
        shared.predict(sharedRng, timeStep);
        per.predict(perRng, timeStep);
        Eigen::Vector2d const z = truth.head<2>() + Eigen::Vector2d{0.3, -0.2};
        auto const a = shared.update(makeFix(z));
        auto const b = per.update(makeFix(z));
        ASSERT_TRUE(a && b);
        EXPECT_NEAR(a->logLikelihood(), b->logLikelihood(), 1e-9);
    }
    for (Eigen::Index i = 0; i < 50; ++i)
    {
        EXPECT_TRUE(shared.state().particle(i).isApprox(per.state().particle(i), 1e-9)) << i;
        EXPECT_NEAR(shared.state().logWeight(i), per.state().logWeight(i), 1e-9) << i;
        EXPECT_TRUE(shared.state().conditional(i).mean().isApprox(per.state().conditional(i).mean(), 1e-9)) << i;
        EXPECT_TRUE(shared.state().sharedCovariance().isApprox(per.state().conditional(i).covarianceLower(), 1e-9))
            << i;
    }
}

// Resampling a ParticleGaussian through WhenEss copies each ancestor's particle and its Gaussian
// together and equalizes the weights.
TEST(ParticleGaussian, ResampleCarriesEachParticlesGaussian)
{
    auto rng = std::mt19937_64{8};
    auto filter =
        RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 200, rng)};
    ASSERT_TRUE(filter.update(makeRangeRate(0.7, beacons[1])));
    auto const before = filter.state();
    auto const resampling = upkeep::whenEss(std::less{}, 0.99, upkeep::resample(rng));
    ASSERT_TRUE(resampling(filter.state()).value());
    auto const ancestors = filter.state().particles().ancestors();
    ASSERT_EQ(ancestors.size(), 200u);
    for (Eigen::Index i = 0; i < 200; ++i)
    {
        Eigen::Index const a = ancestors[std::size_t(i)];
        EXPECT_EQ(filter.state().particle(i), before.particle(a));
        EXPECT_EQ(filter.state().conditional(i).mean(), before.conditional(a).mean());
        EXPECT_EQ(filter.state().conditional(i).covarianceLower(), before.conditional(a).covarianceLower());
        EXPECT_NEAR(filter.state().logWeight(i), -std::log(200.0), 1e-12);
    }
}

// Roughen jitters the particles and leaves their Gaussians and weights alone.
TEST(ParticleGaussian, RoughenMovesTheParticlesAndKeepsTheirGaussians)
{
    auto rng = std::mt19937_64{9};
    auto state = drawParticleGaussian<Position>(makeInitial(), 100, rng);
    auto const before = state;
    ASSERT_TRUE(upkeep::roughen(rng, 0.5)(state).value());
    EXPECT_NE(state.particle(0), before.particle(0));
    EXPECT_EQ(state.conditional(0).mean(), before.conditional(0).mean());
    EXPECT_TRUE((state.logWeights() == before.logWeights()).all());
}

// Under a Gate, particles whose innovation fails get log-likelihood and log-weight minus infinity
// while the rest are updated, and a fix no particle can explain is rejected and changes nothing.
TEST(RbParticleFilter, AGateZeroesTheParticlesAMeasurementContradicts)
{
    auto rng = std::mt19937_64{10};
    auto filter =
        RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 400, rng)};
    // A fix 4 m off the prior mean: the particles near it pass the gate, the far ones don't.
    auto const result = filter.update(makeFix(Eigen::Vector2d{4, 0}), reductions::gate(0.99));
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    auto const rejected = (result->logLikelihoods() == -std::numeric_limits<double>::infinity()).count();
    EXPECT_GT(rejected, 0);
    EXPECT_LT(rejected, 400);
    EXPECT_EQ((filter.state().logWeights() == -std::numeric_limits<double>::infinity()).count(), rejected);

    // A fix no particle can explain is rejected and changes nothing.
    auto const before = filter.state();
    auto const far = filter.update(makeFix(Eigen::Vector2d{1000, 0}), reductions::gate(0.99));
    ASSERT_TRUE(far);
    EXPECT_FALSE(far->accepted);
    EXPECT_TRUE((filter.state().logWeights() == before.logWeights()).all());
    EXPECT_EQ(filter.state().conditional(3).mean(), before.conditional(3).mean());
}

namespace
{

// An accurate velocity fix, which a particle filter sees only through its velocity samples.
auto
makeVelocityFix(Eigen::Vector2d const & z)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Velocity>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 0.01 * Eigen::Matrix2d::Identity()})
        ),
        z
    );
}

// How far a resampling filter strays from the Kalman filter over a constant-velocity run with
// position fixes, and with velocity fixes if `velocity`, where the Kalman filter is exact: the
// worst error of any mean, in standard deviations, of any covariance entry, relative to the
// standard deviations, and of the log-likelihood. The fixes are fixed offsets from the truth, so
// only the particles are random.
struct LinearRun
{
    double mean = 0;
    double cov = 0;
    double logLikelihood = 0;
};

// The update a filter returned, taken out of its std::expected where it has one: the
// Rao-Blackwellized filter's Gaussian updates can fail, and a particle filter's update cannot.
template <typename Update>
auto
updateValue(Update const & update)
{
    if constexpr (requires { update.has_value(); })
    {
        return update.value();
    }
    else
    {
        return update;
    }
}

template <typename Filter, typename Moments>
LinearRun
runLinear(Filter & filter, Moments const & moments, bool velocity, std::mt19937_64 & rng)
{
    auto kalman = KalmanFilter{makeModel(), makeInitial()};
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    Vector4 truth = makeInitial().mean();
    auto run = LinearRun{};
    for (int step = 1; step <= 8; ++step)
    {
        truth = transitionMatrix() * truth;
        Eigen::Vector2d const z = truth.head<2>() + 0.8 * Eigen::Vector2d{std::cos(2.4 * step), std::sin(2.4 * step)};
        Eigen::Vector2d const w = truth.tail<2>() + 0.1 * Eigen::Vector2d{std::sin(1.7 * step), std::cos(1.7 * step)};
        kalman.predict(timeStep);
        filter.predict(rng, timeStep);
        double exact = kalman.update(makeFix(z)).value().logLikelihood();
        auto const fixed = updateValue(filter.update(makeFix(z)));
        EXPECT_TRUE(fixed.accepted);
        double sampled = fixed.logLikelihood();
        if (velocity)
        {
            exact += kalman.update(makeVelocityFix(w)).value().logLikelihood();
            sampled += updateValue(filter.update(makeVelocityFix(w))).logLikelihood();
        }
        run.logLikelihood = std::max(run.logLikelihood, std::abs(sampled - exact));
        auto const estimate = moments(filter.state());
        auto const & mean = kalman.state().mean();
        auto const & cov = kalman.state().covarianceLower();
        for (Eigen::Index i = 0; i < 4; ++i)
        {
            run.mean = std::max(run.mean, std::abs(estimate.mean()(i) - mean(i)) / std::sqrt(cov(i, i)));
            for (Eigen::Index j = 0; j < 4; ++j)
            {
                run.cov = std::max(
                    run.cov,
                    std::abs(estimate.covarianceLower()(i, j) - cov(i, j)) / std::sqrt(cov(i, i) * cov(j, j))
                );
            }
        }
        EXPECT_TRUE(resampling(filter.state()).has_value());
    }
    return run;
}

LinearRun
runRb(unsigned seed, Eigen::Index count, bool velocity)
{
    auto rng = std::mt19937_64{seed};
    auto filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), count, rng)};
    return runLinear(
        filter,
        [](auto const & state) {
            return state.toDensity()->collapsed().moments();
        },
        velocity,
        rng
    );
}

LinearRun
runPlain(unsigned seed, Eigen::Index count, bool velocity)
{
    auto rng = std::mt19937_64{seed};
    auto const initial = makeInitial();
    auto filter = ParticleFilter{
        makeModel(),
        ParticleSet<State>{State{}, Emrys::drawSamples(initial.toDensity().value(), count, rng)}
    };
    return runLinear(
        filter,
        [](auto const & state) {
            return state.moments().value();
        },
        velocity,
        rng
    );
}

}  // namespace

// Over eight steps with 5,000 particles and resampling, the worst errors against the exact Kalman
// filter in mean, covariance, and log-likelihood stay under 0.25, about twice the worst seen over
// 30 seeds; they shrink as one over root n, so there is no bias.
TEST(RbParticleFilter, MatchesTheKalmanFilterOnALinearProblem)
{
    // Errors carry over between steps through resampling, so the bounds are about twice the worst
    // seen over 30 seeds (0.13 in the mean), not a single step's standard error. They shrink as one
    // over the square root of the number of particles: no bias.
    LinearRun const run = runRb(19, 5000, false);
    EXPECT_LT(run.mean, 0.25);
    EXPECT_LT(run.cov, 0.25);
    EXPECT_LT(run.logLikelihood, 0.25);
}

// With accurate velocity fixes, summed over 30 seeds of 500 particles, the Rao-Blackwellized
// filter's mean and log-likelihood errors are under 0.7 of a plain particle filter's, whose weights
// collapse onto the few particles with agreeing velocities.
TEST(RbParticleFilter, StraysLessThanAParticleFilterWhenTheGaussianPartIsMeasured)
{
    // With accurate velocity fixes, a particle filter's weights collapse onto the few particles
    // whose velocities agree, while the Rao-Blackwellized filter updates each velocity exactly.
    // Over 30 seeds with 500 particles its errors are about half the particle filter's in the
    // means and under half in the log-likelihood, with each of two standard libraries; single
    // seeds vary, so the sums are compared. With position fixes alone, which only the particles
    // see, the two are about even.
    auto rb = LinearRun{};
    auto plain = LinearRun{};
    for (unsigned seed = 1; seed <= 30; ++seed)
    {
        LinearRun const a = runRb(seed, 500, true);
        LinearRun const b = runPlain(seed, 500, true);
        rb.mean += a.mean;
        rb.logLikelihood += a.logLikelihood;
        plain.mean += b.mean;
        plain.logLikelihood += b.logLikelihood;
    }
    EXPECT_LT(rb.mean, 0.7 * plain.mean) << plain.mean;
    EXPECT_LT(rb.logLikelihood, 0.7 * plain.logLikelihood) << plain.logLikelihood;
}

// Seen only through ranges to three beacons over 30 steps, the filter localizes the position within
// 1.5 units and learns the velocity through the particles' motion, its standard deviation falling
// from 2 to under 0.5 with the error within three of them.
TEST(RbParticleFilter, LearnsAVelocityFromRanges)
{
    // A target moving at constant velocity, seen only through ranges to three beacons, which say
    // nothing about the velocity. The velocity is learned through the particles' motion.
    auto rng = std::mt19937_64{11};
    auto normal = std::normal_distribution<double>{};
    auto truth = Vector4{5, -3, 1.5, -0.8};
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 25, 25, 4, 4;
    auto filter = RbParticleFilter{
        makeModel(),
        drawParticleGaussian<Position>(GaussianParams<State>{State{}, Vector4::Zero(), cov}, 500, rng)
    };
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    for (int step = 1; step <= 30; ++step)
    {
        truth = transitionMatrix() * truth;
        filter.predict(rng, timeStep);
        for (auto const & beacon : beacons)
        {
            double const z = (truth.head<2>() - beacon).norm() + 0.5 * normal(rng);
            ASSERT_TRUE(filter.update(makeRange(z, beacon)).value().accepted);
        }
        ASSERT_TRUE(resampling(filter.state()).has_value());
    }
    // The velocity's standard deviation falls from 2 m/s to about 0.33, and over 20 seeds its error
    // stays within 2.5 of them, with the position within 1 m.
    auto const estimate = filter.state().toDensity()->collapsed();
    Eigen::Matrix2d const velocityCov = estimate.covarianceLower().bottomRightCorner<2, 2>();
    EXPECT_LT((estimate.mean().head<2>() - truth.head<2>()).norm(), 1.5) << estimate.mean().transpose();
    EXPECT_LT(std::sqrt(velocityCov(0, 0)), 0.5);
    EXPECT_LT(std::sqrt(velocityCov(1, 1)), 0.5);
    EXPECT_LT((estimate.mean().tail<2>() - truth.tail<2>()).norm(), 3 * std::sqrt(velocityCov.trace()))
        << estimate.mean().transpose();
}

// A Rao-Blackwellized particle filter prints its name, with its state nested under it; the state
// prints the particle count and effective sample size on its first line, then the lines the
// collapsed Gaussian prints under its own first line.
TEST(RbParticleFilter, PrintsItsNameAndItsStateCollapsed)
{
    auto rng = std::mt19937_64{2};
    auto const filter = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeInitial(), 100, rng)};
    std::string const text = printed(filter);
    EXPECT_EQ(text.substr(0, text.find('\n')), "Rao-Blackwellized particle filter");
    std::string const state = printed(filter.state());
    EXPECT_EQ(
        state.substr(0, state.find('\n')),
        "Rao-Blackwellized particle set of 100 particles of dimension 4, effective sample size 100"
    );
    EXPECT_EQ(afterFirstLine(state), afterFirstLine(printed(filter.state().toDensity()->collapsed())));
}

// Hybrid particles: under a switching model, each particle holds a hybrid over the rest and is an
// IMM filter given its value [rb-particle-filter], [imm].

namespace
{

struct Steady
{};

struct Maneuvering
{};

// The target holds its velocity or maneuvers, and stays as it is with probability 0.9.
struct Mode : Emrys::Named<Steady, Maneuvering>
{};

using Rest = ParticleGaussianParams<State, Position>::RestSpace;

// The modes differ in their process noise, which moves the position, so a particle's draw of its
// position tells them apart.
auto
makeSwitching(double stay = 0.9)
{
    return Emrys::SwitchingModel{
        Emrys::MarkovChain<Mode>::persistent(stay),
        ConstantVelocity<Position, Velocity>{0.01},
        ConstantVelocity<Position, Velocity>{2.0}
    };
}

// The modes start with different velocities, so the mixing moves each.
Emrys::HybridGaussianParams<Rest, Mode>
initialHybrid()
{
    auto hybrid = Emrys::HybridGaussianParams<Rest, Mode>{
        Emrys::CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.7, 0.3}),
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.5}, Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}}
    };
    hybrid.setConditional(
        1,
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{-0.5, 1.5}, 3.0 * Eigen::Matrix2d::Identity()}
    );
    return hybrid;
}

// Six particles spread over the beacons' field.
Eigen::Matrix<double, 2, 6>
particlePositions()
{
    return Eigen::Matrix<double, 2, 6>{{-3, 0, 2, 5, -1, 4}, {1, -2, 0, 3, 4, -4}};
}

auto
makeHybridState()
{
    return Emrys::particleGaussian<Position>(State{}, ParticleSet<Position>{particlePositions()}, initialHybrid());
}

// The particle's position, measured with almost no noise: an IMM over the whole state given it.
auto
makeExactFix(Eigen::Vector2d const & z)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 1e-12 * Eigen::Matrix2d::Identity()})
        ),
        z
    );
}

// A range rate clear of obstruction, or biased by 0.5 with variance 0.25: one noise component per
// value of Mode, for the update by mode axis.
Emrys::GaussianMixtureParams<Linear<1>>
biasedNoise()
{
    auto noise = Emrys::GaussianMixtureParams<Linear<1>>{Linear<1>{}};
    noise.add(std::log(0.9), Vector1::Zero(), Vector1{0.01});
    noise.add(std::log(0.1), Vector1{0.5}, Vector1{0.25});
    return noise;
}

auto
makeBiasedRangeRate(double z, Eigen::Vector2d const & beacon)
{
    auto const rangeRate = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{
                d.dot(Emrys::block<Velocity>(in, x)) / Eigen::numext::sqrt(T(d.squaredNorm()))
            };
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        additiveNoise(biasedNoise()).byMode<Mode>()
    );
    return observe(methods::linearized(), rangeRate, Vector1{z});
}

// The range rate with the position known as p, a measurement of the rest alone; with the noise
// policy \a noise, the biased noise by mode or not.
template <typename NoisePolicy>
auto
makeRangeRateAt(double z, Eigen::Vector2d const & beacon, Eigen::Vector2d const & p, NoisePolicy noise)
{
    auto const rangeRate = Emrys::measurementModel(
        [u = ((p - beacon) / (p - beacon).norm()).eval()]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{u.cast<T>().dot(Emrys::block<Velocity>(in, x))};
        },
        Emrys::reads(Emrys::path<Velocity>),
        std::move(noise)
    );
    return observe(methods::linearized(), rangeRate, Vector1{z});
}

// A hybrid over the whole state with the position known as p, almost exactly, and the rest as
// `hybrid` has it.
Emrys::HybridGaussianParams<State, Mode>
lifted(Eigen::Vector2d const & p, Emrys::HybridGaussianParams<Rest, Mode> const & hybrid)
{
    auto const whole = [&](Eigen::Index m) {
        Vector4 mean;
        mean << p, hybrid.conditional(m).mean();
        Matrix4 cov = Matrix4::Zero();
        cov.topLeftCorner<2, 2>() = 1e-12 * Eigen::Matrix2d::Identity();
        cov.bottomRightCorner<2, 2>() = hybrid.conditional(m).covarianceLower();
        return GaussianParams<State>{State{}, mean, cov};
    };
    auto result = Emrys::HybridGaussianParams<State, Mode>{hybrid.modes(), whole(0)};
    result.setConditional(1, whole(1));
    return result;
}

// The marginal of a hybrid over the whole state on the rest, mode by mode.
Emrys::HybridGaussianParams<Rest, Mode>
restOf(Emrys::HybridGaussianParams<State, Mode> const & hybrid)
{
    auto const rest = [&](Eigen::Index m) {
        auto const & component = hybrid.conditional(m);
        return GaussianParams<Rest>{
            Rest{},
            component.mean().tail<2>(),
            component.covarianceLower().bottomRightCorner<2, 2>()
        };
    };
    auto result = Emrys::HybridGaussianParams<Rest, Mode>{hybrid.modes(), rest(0)};
    result.setConditional(1, rest(1));
    return result;
}

// Whether two hybrids have the same normalized mode probabilities and the same Gaussian per mode.
template <typename Hybrid>
void
expectSameHybrid(Hybrid const & actual, Hybrid const & expected, double tolerance)
{
    auto const probabilities = [](auto const & hybrid) {
        Eigen::ArrayXd const logP = hybrid.modes().logWeights() - hybrid.modes().logTotalWeight().value();
        return Eigen::ArrayXd{logP.exp()};
    };
    EXPECT_LE((probabilities(actual) - probabilities(expected)).matrix().norm(), tolerance);
    for (Eigen::Index m = 0; m < actual.size(); ++m)
    {
        EXPECT_LE((actual.conditional(m).mean() - expected.conditional(m).mean()).norm(), tolerance) << m;
        EXPECT_LE(
            (actual.conditional(m).covarianceLower() - expected.conditional(m).covarianceLower()).norm(),
            tolerance
        ) << m;
    }
}

}  // namespace

// With a switching model, every particle holds a hybrid and is an IMM filter given its value. Over
// four steps, each particle's prediction matches an ImmFilter over the whole state, from the
// particle's position known and its hybrid, predicted and then given the drawn position almost
// exactly: the mixing, each mode's prediction, the conditioning on the draw, and the modes
// reweighted by the draw's density. Its weight is unchanged. Each update, alternately plain and by
// mode axis, matches an IMM update of the predicted hybrid with the particle's position built into
// the measurement, in its log-likelihood, NIS, and hybrid, and reweights the particles by it.
TEST(RbParticleFilter, HybridParticlesMatchAnImmFilterPerParticle)
{
    auto rng = std::mt19937_64{31};
    auto filter = RbParticleFilter{makeSwitching(), makeHybridState()};
    static_assert(std::is_same_v<decltype(filter)::StateType::CovarianceType, PerPointCovariance>);
    static_assert(std::is_same_v<decltype(filter)::ConditionalType, Emrys::HybridGaussianParams<Rest, Mode>>);
    auto recursion = Emrys::ImmRecursion<Emrys::Gaussian<Rest>, decltype(makeSwitching())>{makeSwitching()};
    for (int k = 0; k < 4; ++k)
    {
        auto const before = filter.state();
        ASSERT_TRUE(filter.predict(rng, timeStep));
        for (Eigen::Index i = 0; i < before.size(); ++i)
        {
            EXPECT_EQ(filter.state().logWeight(i), before.logWeight(i)) << i;
            auto imm = Emrys::ImmFilter{makeSwitching(), lifted(before.particle(i), before.conditional(i))};
            imm.predict(timeStep);
            auto const given = imm.update(makeExactFix(filter.state().particle(i)));
            ASSERT_TRUE(given && given->accepted) << i;
            expectSameHybrid(filter.state().conditional(i), restOf(imm.state()), 1e-7);
        }

        auto const predicted = filter.state();
        std::size_t const b = std::size_t(k) % beacons.size();
        double const z = 0.6 - 0.4 * k;
        bool const byAxis = k % 2 == 1;
        auto const result =
            byAxis ? filter.update(makeBiasedRangeRate(z, beacons[b])) : filter.update(makeRangeRate(z, beacons[b]));
        ASSERT_TRUE(result && result->accepted);
        EXPECT_EQ(result->innovationSize(), 1);
        Eigen::ArrayXd logWeights = predicted.logWeights();
        for (Eigen::Index i = 0; i < predicted.size(); ++i)
        {
            auto hybrid = predicted.conditional(i);
            auto const p = predicted.particle(i);
            auto const known = byAxis
                                 ? recursion.update(
                                       hybrid,
                                       makeRangeRateAt(z, beacons[b], p, additiveNoise(biasedNoise()).byMode<Mode>())
                                   )
                                 : recursion.update(
                                       hybrid,
                                       makeRangeRateAt(
                                           z,
                                           beacons[b],
                                           p,
                                           additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.01}})
                                       )
                                   );
            ASSERT_TRUE(known) << i;
            EXPECT_NEAR(result->logLikelihoods()(i), known->logLikelihood(), 1e-9) << i;
            Eigen::Index mode = 0;
            hybrid.modes().logWeights().maxCoeff(&mode);
            EXPECT_NEAR(result->nis()(i), known->modes[std::size_t(mode)].branches[0].nis(), 1e-9) << i;
            expectSameHybrid(filter.state().conditional(i), hybrid, 1e-9);
            logWeights(i) += known->logLikelihood();
        }
        double const total = Emrys::logSumExp(logWeights);
        EXPECT_TRUE(filter.state().logWeights().isApprox((logWeights - total).matrix(), 1e-9));
    }
}

// Under a switching model whose modes share one model, with every mode starting from the same
// Gaussian, the modes never part: each step matches a filter of one Gaussian per particle from
// the same standard normals, in the particles, the weights, and each hybrid's collapse, and the
// modes move only by the chain. A switching model of fixed-step models [fixed-step-models] takes
// the same steps.
TEST(RbParticleFilter, HybridParticlesOfOneDynamicsMatchTheFilterOfOneGaussian)
{
    auto rng = std::mt19937_64{32};
    auto const initial = drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), 30, rng);
    auto const chain = Emrys::MarkovChain<Mode>{Eigen::Matrix2d{{0.9, 0.1}, {0.3, 0.7}}};
    auto const priorModes = Emrys::CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector2d{0.6, 0.4});
    auto hybrids = std::vector<Emrys::HybridGaussianParams<Rest, Mode>>{};
    for (Eigen::Index i = 0; i < initial.size(); ++i)
    {
        hybrids.emplace_back(priorModes, initial.conditional(i));
    }
    auto const hybridState = ParticleGaussianParams<
        State,
        Position,
        PerPointCovariance,
        double,
        Emrys::DefaultAllocator,
        Emrys::HybridGaussian<Rest, Mode>>{State{}, initial.particles(), hybrids};
    auto hybrid = RbParticleFilter{Emrys::SwitchingModel{chain, makeModel()}, hybridState};
    auto fixed = RbParticleFilter{
        Emrys::SwitchingModel{chain, Emrys::FixedStepModel{makeModel(), State{}, timeStep}},
        hybridState
    };
    auto plain = RbParticleFilter{makeModel(), initial};
    auto modes = priorModes;
    for (int k = 1; k <= 3; ++k)
    {
        auto plainRng = rng;
        auto fixedRng = rng;
        ASSERT_TRUE(hybrid.predict(rng, timeStep));
        ASSERT_TRUE(fixed.predict(fixedRng));
        plain.predict(plainRng, timeStep);
        Eigen::Vector2d const z{0.5 * k, -0.3 * k};
        ASSERT_TRUE(hybrid.update(makeFix(z)));
        ASSERT_TRUE(fixed.update(makeFix(z)));
        ASSERT_TRUE(plain.update(makeFix(z)));
        modes = Emrys::predict(modes, chain);
        Eigen::ArrayXd const expectedModes = (modes.logWeights() - modes.logTotalWeight().value()).exp();
        auto const expectMatches = [&](auto const & filter) {
            for (Eigen::Index i = 0; i < plain.state().size(); ++i)
            {
                auto const & cell = filter.state().conditional(i);
                EXPECT_TRUE(filter.state().particle(i).isApprox(plain.state().particle(i), 1e-9)) << i;
                EXPECT_NEAR(filter.state().logWeight(i), plain.state().logWeight(i), 1e-9) << i;
                auto const collapsed = cell.toDensity()->collapsed();
                EXPECT_TRUE(collapsed.mean().isApprox(plain.state().conditional(i).mean(), 1e-9)) << i;
                EXPECT_TRUE(collapsed.covarianceLower().isApprox(plain.state().conditional(i).covarianceLower(), 1e-9))
                    << i;
                Eigen::ArrayXd const p = (cell.modes().logWeights() - cell.modes().logTotalWeight().value()).exp();
                EXPECT_TRUE(p.isApprox(expectedModes, 1e-9)) << i;
            }
        };
        expectMatches(hybrid);
        expectMatches(fixed);
    }
}

// A state of hybrid particles sums its particles' modes in modes(), collapses each particle's
// hybrid in its density's collapse, carries each hybrid with its particle through resampling, and
// prints; a likelihood of the sampled block alone weighs the particles and leaves every hybrid.
TEST(ParticleGaussian, HoldsAHybridPerParticle)
{
    auto rng = std::mt19937_64{33};
    auto filter = RbParticleFilter{makeSwitching(), makeHybridState()};
    ASSERT_TRUE(filter.predict(rng, timeStep));
    ASSERT_TRUE(filter.update(makeRangeRate(0.4, beacons[0])));
    ASSERT_TRUE(filter.predict(rng, timeStep));
    ASSERT_TRUE(filter.update(makeBiasedRangeRate(-0.2, beacons[1])));
    auto const & state = filter.state();

    Eigen::Array2d expectedModes = Eigen::Array2d::Zero();
    Eigen::Vector2d restMean = Eigen::Vector2d::Zero();
    for (Eigen::Index i = 0; i < state.size(); ++i)
    {
        auto const & hybrid = state.conditional(i);
        double const weight = std::exp(state.logWeight(i));
        expectedModes += weight * (hybrid.modes().logWeights() - hybrid.modes().logTotalWeight().value()).exp();
        restMean += weight * hybrid.toDensity()->collapsed().mean();
    }
    auto const modes = state.modes();
    EXPECT_TRUE((modes.logWeights() - modes.logTotalWeight().value()).exp().isApprox(expectedModes, 1e-12));
    EXPECT_NEAR(expectedModes.sum(), 1.0, 1e-12);
    auto const collapsed = state.toDensity()->collapsed();
    EXPECT_TRUE(collapsed.mean().tail<2>().isApprox(restMean, 1e-12));
    EXPECT_FALSE(printed(state).empty());

    auto resampled = state;
    ASSERT_TRUE(resampled.resample(rng));
    for (Eigen::Index j = 0; j < resampled.size(); ++j)
    {
        Eigen::Index const ancestor = resampled.particles().ancestors()[std::size_t(j)];
        EXPECT_EQ(resampled.particle(j), state.particle(ancestor)) << j;
        expectSameHybrid(resampled.conditional(j), state.conditional(ancestor), 0.0);
    }

    auto const before = filter.state();
    auto const weighed = filter.update(makeLaplaceFix(Eigen::Vector2d{1, 0}));
    ASSERT_TRUE(weighed && weighed->accepted);
    for (Eigen::Index i = 0; i < before.size(); ++i)
    {
        expectSameHybrid(filter.state().conditional(i), before.conditional(i), 0.0);
    }
}

// A mode whose prediction gives the position a covariance that isn't positive definite has no
// density for the draw to be weighed by, so predict() returns NotPositiveDefinite and leaves the
// particles and their hybrids as they were. Here the second mode has no process noise and starts
// with its velocity known, and the identity chain keeps it from mixing with the first.
TEST(RbParticleFilter, AHybridPredictionWithoutADensityLeavesTheState)
{
    auto rng = std::mt19937_64{34};
    auto prior = initialHybrid();
    prior.setConditional(1, GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1, 0}, Eigen::Matrix2d::Zero()});
    auto filter = RbParticleFilter{
        Emrys::SwitchingModel{
            Emrys::MarkovChain<Mode>::persistent(1.0),
            ConstantVelocity<Position, Velocity>{0.01},
            ConstantVelocity<Position, Velocity>{0.0}
        },
        Emrys::particleGaussian<Position>(State{}, ParticleSet<Position>{particlePositions()}, prior)
    };
    auto const before = filter.state();
    auto const moved = filter.predict(rng, timeStep);
    ASSERT_FALSE(moved);
    EXPECT_EQ(moved.error(), Emrys::NumericError::NotPositiveDefinite);
    EXPECT_EQ(filter.state().particles().states(), before.particles().states());
    for (Eigen::Index i = 0; i < before.size(); ++i)
    {
        expectSameHybrid(filter.state().conditional(i), before.conditional(i), 0.0);
    }
}

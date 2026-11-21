// ParticleFilter::step(), auxiliaryStep(), and move() of Filters/ParticleFilter.h: the steps that
// use the measurement. A step with a measurement observed without a method is predict() then
// update(), bitwise. With a linear measurement and a method the proposal is the optimal one, so
// each weight is the likelihood of the observation given the state the particle moved from,
// computed here in plain Eigen, and an auxiliary step leaves every weight equal. On a linear
// Gaussian model the guided, auxiliary, and moved filters stay near the Kalman filter, and with a
// precise measurement the guided step keeps an effective sample size where the bootstrap step
// loses it. A noiseless block stays exactly as it was. A range measurement guides by sigma points.
// A move separates the copies a resampling made and leaves the moments where they were. The
// evidence for roadmap step 17zi, a prototype (docs/design/particles.md).

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <set>
#include <utility>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::observe;
using Emrys::ParticleFilter;
using Emrys::ParticleSet;

namespace
{

namespace upkeep = Emrys::upkeep;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Velocity>;
using Biased = Composite<Position, Velocity, Bias>;

double const accelDensity = 0.1;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

GaussianParams<State>
makeInitial()
{
    auto const variances = Eigen::Vector4d{4.0, 4.0, 1.0, 1.0};
    return {State{}, Eigen::Vector4d{0.0, 0.0, 1.0, 0.5}, Eigen::Matrix4d{variances.asDiagonal()}};
}

// A position fix with the noise variance given, as a measurement model.
auto
fixModel(double variance)
{
    return Emrys::measurementModel(
        Eigen::Matrix2d::Identity(),
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), variance * Eigen::Matrix2d::Identity()})
    );
}

// Range from a point off the track, with a standard deviation of 0.2.
auto
rangeModel()
{
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, 2, 1> const offset =
                Emrys::block<Position>(in, x) - Eigen::Vector2d{-20.0, 10.0}.template cast<T>();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(offset.squaredNorm()))};
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.04}})
    );
}

ParticleSet<State>
drawParticles(int count, std::mt19937_64 & rng)
{
    auto const initial = makeInitial();
    return ParticleSet<State>{Emrys::drawSamples(initial.toDensity().value(), count, rng)};
}

// The fix at step k of the runs below: a fixed offset from the mean track, so that no run meets a
// fix in the tail of its prediction.
Eigen::Vector2d
fixAt(int k)
{
    return Eigen::Vector2d{1.0 * k + 0.3, 0.5 * k - 0.2};
}

// The number of distinct first coordinates among the particles.
template <typename Set>
std::size_t
distinct(Set const & set)
{
    auto values = std::set<double>{};
    for (Eigen::Index i = 0; i < set.size(); ++i)
    {
        values.insert(set.states()(0, i));
    }
    return values.size();
}

}  // namespace

// A measurement observed without a method draws from the model alone, so the step is predict()
// then update(): the same states and log-likelihood from the same seed, bitwise. The measurement is
// a function's, since a model given its matrix is observed by Linearized [linear-sensors].
TEST(ParticleProposals, AStepWithoutAMethodIsPredictThenUpdate)
{
    auto seed = std::mt19937_64{1};
    auto const particles = drawParticles(500, seed);
    auto separate = ParticleFilter{makeModel(), particles};
    auto fused = ParticleFilter{makeModel(), particles};
    auto const range = observe(rangeModel(), Eigen::Matrix<double, 1, 1>{23.0});
    static_assert(!decltype(range)::HasMethod);
    auto a = std::mt19937_64{7};
    auto b = std::mt19937_64{7};
    separate.predict(1.0, a);
    auto const updated = separate.update(range);
    auto const stepped = fused.step(1.0, range, b);
    ASSERT_TRUE(updated.has_value());
    ASSERT_TRUE(stepped.has_value());
    EXPECT_EQ(fused.state().states(), separate.state().states());
    EXPECT_EQ(stepped->logLikelihood(), updated->logLikelihood());
    EXPECT_TRUE((fused.state().logWeights() == separate.state().logWeights()).all());
}

// With a measurement affine in the state and Gaussian noise, the Linearized proposal is the
// optimal one: a particle's weight is the likelihood of the observation given the state it moved
// from, N(z; H f(x), H Q H^T + R), whatever it drew. Here that is computed in plain Eigen from the
// particles before the step, Q from a FixedStepModel, and the log-likelihood is the log of its mean.
TEST(ParticleProposals, TheOptimalProposalWeightsByTheStateMovedFrom)
{
    auto seed = std::mt19937_64{2};
    auto const before = drawParticles(300, seed);
    auto filter = ParticleFilter{makeModel(), before};
    auto const z = fixAt(1);
    auto rng = std::mt19937_64{8};
    auto const stepped = filter.step(1.0, observe(Emrys::methods::linearized(), fixModel(0.25), z), rng);
    ASSERT_TRUE(stepped.has_value());
    ASSERT_TRUE(stepped->accepted);
    Eigen::Matrix4d const q = FixedStepModel{makeModel(), State{}, 1.0}.noise();
    Eigen::Matrix2d const s = q.topLeftCorner<2, 2>() + 0.25 * Eigen::Matrix2d::Identity();
    auto const llt = Eigen::LLT<Eigen::Matrix2d>{s};
    Eigen::ArrayXd expected(before.size());
    for (Eigen::Index i = 0; i < before.size(); ++i)
    {
        Eigen::Vector4d const moved = Emrys::propagate(State{}, Eigen::Vector4d{before.state(i)}, makeModel(), 1.0);
        Eigen::Vector2d const whitened = llt.matrixL().solve(z - moved.head<2>());
        expected(i) = -0.5 * whitened.squaredNorm() - std::log(llt.matrixLLT()(0, 0) * llt.matrixLLT()(1, 1))
                    - std::log(2 * std::numbers::pi);
    }
    double const logMean = std::log(expected.exp().mean());
    EXPECT_NEAR(stepped->logLikelihood(), logMean, 1e-9);
    Eigen::ArrayXd const weights = expected - logMean - std::log(double(before.size()));
    EXPECT_LT((filter.state().logWeights() - weights).abs().maxCoeff(), 1e-9);
}

// The auxiliary step with the optimal proposal is fully adapted: resampled by the look-ahead
// likelihood, which is the whole weight, every particle leaves with the same weight, and the
// log-likelihood is that of the optimal step from the same particles.
TEST(ParticleProposals, AFullyAdaptedAuxiliaryStepLeavesEqualWeights)
{
    auto seed = std::mt19937_64{3};
    auto const particles = drawParticles(300, seed);
    auto resampling = std::mt19937_64{9};
    auto auxiliary = ParticleFilter{makeModel(), particles, upkeep::resample(resampling)};
    auto plain = ParticleFilter{makeModel(), particles};
    auto const fix = observe(Emrys::methods::linearized(), fixModel(0.25), fixAt(1));
    auto a = std::mt19937_64{10};
    auto b = std::mt19937_64{10};
    auto const stepped = auxiliary.auxiliaryStep(1.0, fix, a);
    auto const reference = plain.step(1.0, fix, b);
    ASSERT_TRUE(stepped.has_value());
    ASSERT_TRUE(reference.has_value());
    EXPECT_EQ(auxiliary.state().resampleCount(), 1);
    EXPECT_LT((auxiliary.state().logWeights() + std::log(300.0)).abs().maxCoeff(), 1e-9);
    EXPECT_NEAR(stepped->logLikelihood(), reference->logLikelihood(), 1e-9);
}

// On a linear Gaussian model each way of stepping stays near the Kalman filter over five steps with
// 20000 particles: means within 0.15 standard deviations, covariance entries within 0.15 of the
// Kalman variances' scale, and the summed log-likelihood within 0.15, the tolerances of the
// bootstrap filter's own test in particle_filter.cpp.
TEST(ParticleProposals, EveryStepApproachesTheKalmanFilterOnALinearModel)
{
    enum class Way
    {
        guided,
        auxiliary,
        auxiliaryPrior,
        moved
    };
    for (Way const way : {Way::guided, Way::auxiliary, Way::auxiliaryPrior, Way::moved})
    {
        SCOPED_TRACE(int(way));
        auto seed = std::mt19937_64{4};
        auto resampling = std::mt19937_64{11};
        auto filter = ParticleFilter{
            makeModel(),
            drawParticles(20'000, seed),
            upkeep::whenEss(std::less{}, 0.5, upkeep::resample(resampling))
        };
        auto kalman = KalmanFilter{makeModel(), makeInitial()};
        auto rng = std::mt19937_64{12};
        double logLikelihood = 0;
        double kalmanLogLikelihood = 0;
        for (int k = 1; k <= 5; ++k)
        {
            auto const guided = observe(Emrys::methods::linearized(), fixModel(0.25), fixAt(k));
            auto const weighed = observe(fixModel(0.25), fixAt(k));
            kalman.predict(1.0);
            kalmanLogLikelihood += kalman.update(guided)->logLikelihood();
            if (way == Way::guided)
            {
                logLikelihood += filter.step(1.0, guided, rng)->logLikelihood();
                ASSERT_TRUE(filter.maintain().has_value());
            }
            else if (way == Way::auxiliary)
            {
                logLikelihood += filter.auxiliaryStep(1.0, guided, rng)->logLikelihood();
            }
            else if (way == Way::auxiliaryPrior)
            {
                logLikelihood += filter.auxiliaryStep(1.0, weighed, rng)->logLikelihood();
            }
            else
            {
                logLikelihood += filter.step(1.0, weighed, rng)->logLikelihood();
                ASSERT_TRUE(filter.maintain().has_value());
                EXPECT_GT(filter.move(1.0, weighed, rng), 0);
            }
        }
        auto const moments = filter.state().moments().value();
        Eigen::Vector4d const deviations = kalman.state().covarianceLower().diagonal().cwiseSqrt();
        Eigen::Vector4d const error = (moments.mean() - kalman.state().mean()).cwiseQuotient(deviations);
        EXPECT_LT(error.cwiseAbs().maxCoeff(), 0.15);
        Eigen::Matrix4d const kalmanCov = kalman.state().covarianceLower().selfadjointView<Eigen::Lower>();
        Eigen::Matrix4d const cov = moments.covarianceLower().template selfadjointView<Eigen::Lower>();
        Eigen::Matrix4d const scale = deviations * deviations.transpose();
        EXPECT_LT((cov - kalmanCov).cwiseQuotient(scale).cwiseAbs().maxCoeff(), 0.15);
        EXPECT_NEAR(logLikelihood, kalmanLogLikelihood, 0.15);
    }
}

// With a fix of standard deviation 0.01 against process noise that moves a position by 0.18, the
// bootstrap step leaves one or two effective particles of 2000. The guided step's weight is the
// likelihood of the fix given the state moved from, whose width is the process noise's, so it leaves
// the particles that started near enough: about 2000 (sqrt(2) 0.18 / 2.24)^2 = 26 from a prior
// position spread of 2.24, more than ten times the bootstrap step's.
TEST(ParticleProposals, AGuidedStepKeepsItsEffectiveSampleSizeUnderAPreciseMeasurement)
{
    auto seed = std::mt19937_64{5};
    auto const particles = drawParticles(2000, seed);
    auto bootstrap = ParticleFilter{makeModel(), particles};
    auto guided = ParticleFilter{makeModel(), particles};
    auto a = std::mt19937_64{13};
    auto b = std::mt19937_64{13};
    bootstrap.predict(1.0, a);
    ASSERT_TRUE(bootstrap.update(observe(fixModel(1e-4), fixAt(1))).has_value());
    ASSERT_TRUE(guided.step(1.0, observe(Emrys::methods::linearized(), fixModel(1e-4), fixAt(1)), b).has_value());
    EXPECT_LT(bootstrap.state().ess().value(), 3.0);
    EXPECT_GT(guided.state().ess().value(), 20.0);
    EXPECT_GT(guided.state().ess().value(), 10 * bootstrap.state().ess().value());
}

// A block with no noise has a zero column in the noise's factor, so a guided step leaves it
// exactly as it was in every particle, and the weights stay finite.
TEST(ParticleProposals, AGuidedStepLeavesANoiselessBlockAsItWas)
{
    auto const model = compose(makeModel(), Constant<Bias>{});
    Eigen::Matrix<double, 5, 5> cov = Eigen::Matrix<double, 5, 5>::Identity();
    auto const initial = GaussianParams<Biased>{Eigen::Matrix<double, 5, 1>{0.0, 0.0, 1.0, 0.5, 0.0}, cov};
    auto seed = std::mt19937_64{6};
    auto const before = ParticleSet<Biased>{Emrys::drawSamples(initial.toDensity().value(), 200, seed)};
    auto filter = ParticleFilter{model, before};
    auto rng = std::mt19937_64{14};
    auto const stepped = filter.step(1.0, observe(Emrys::methods::linearized(), fixModel(0.25), fixAt(1)), rng);
    ASSERT_TRUE(stepped.has_value());
    EXPECT_TRUE(stepped->accepted);
    EXPECT_EQ(filter.state().states().row(4), before.states().row(4));
    EXPECT_NE(filter.state().states().row(0), before.states().row(0));
    EXPECT_TRUE(filter.state().logWeights().isFinite().all());
}

// A range guides by sigma points spread by the noise's factor. With process noise wider than the
// range's, the guided step leaves more than twice the bootstrap step's effective sample size from
// the same particles (over 20 seeds of a scratch run the ratio was 4.7), and its mean lies on the
// observed range to within the bend of the arc the particles cover.
TEST(ParticleProposals, ARangeGuidesBySigmaPoints)
{
    auto seed = std::mt19937_64{7};
    auto const particles = drawParticles(2000, seed);
    auto const model = ConstantVelocity<Position, Velocity>{3.0};
    auto bootstrap = ParticleFilter{model, particles};
    auto guided = ParticleFilter{model, particles};
    auto const z = Eigen::Matrix<double, 1, 1>{(Eigen::Vector2d{1.0, 0.5} - Eigen::Vector2d{-20.0, 10.0}).norm()};
    auto a = std::mt19937_64{15};
    auto b = std::mt19937_64{15};
    ASSERT_TRUE(bootstrap.step(1.0, observe(rangeModel(), z), a).has_value());
    auto const stepped = guided.step(1.0, observe(Emrys::methods::unscented(), rangeModel(), z), b);
    ASSERT_TRUE(stepped.has_value());
    EXPECT_TRUE(stepped->accepted);
    EXPECT_GT(guided.state().ess().value(), 2 * bootstrap.state().ess().value());
    Eigen::Vector2d const mean = guided.state().moments().value().mean().head<2>();
    EXPECT_NEAR((mean - Eigen::Vector2d{-20.0, 10.0}).norm(), z(0), 0.5);
}

// After a resampling has left copies, a move separates them: more distinct particles than before
// it, with weights unchanged, and over three sweeps the mean and variances stay where they were to
// within the Monte Carlo error of 5000 particles, about five standard errors.
TEST(ParticleProposals, AMoveSeparatesCopiesAndKeepsTheMoments)
{
    auto seed = std::mt19937_64{8};
    auto resampling = std::mt19937_64{16};
    auto filter = ParticleFilter{makeModel(), drawParticles(5000, seed), upkeep::resample(resampling)};
    auto rng = std::mt19937_64{17};
    auto const fix = observe(Emrys::methods::linearized(), fixModel(0.25), fixAt(1));
    ASSERT_TRUE(filter.step(1.0, fix, rng).has_value());
    ASSERT_TRUE(filter.maintain().has_value());
    std::size_t const copies = distinct(filter.state());
    auto const before = filter.state().moments().value();
    Eigen::ArrayXd const weights = filter.state().logWeights();
    Eigen::Index moves = 0;
    for (int sweep = 0; sweep < 3; ++sweep)
    {
        moves += filter.move(1.0, fix, rng);
    }
    EXPECT_GT(moves, 5000);
    EXPECT_GT(distinct(filter.state()), copies + 1000);
    EXPECT_TRUE((filter.state().logWeights() == weights).all());
    auto const after = filter.state().moments().value();
    Eigen::Vector4d const deviations = before.covarianceLower().diagonal().cwiseSqrt();
    EXPECT_LT((after.mean() - before.mean()).cwiseQuotient(deviations).cwiseAbs().maxCoeff(), 0.1);
    Eigen::Vector4d const ratio = after.covarianceLower().diagonal().cwiseQuotient(before.covarianceLower().diagonal());
    EXPECT_LT((ratio.array() - 1.0).abs().maxCoeff(), 0.1);
}

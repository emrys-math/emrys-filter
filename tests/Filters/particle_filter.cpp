// ParticleFilter (steps 7, 9a, 13d; [upkeep], [fixed-step-models], [likelihood-measurements]): point-wise
// log-likelihoods match the likelihood, deduction and accessors, prediction matches the Kalman prediction in
// distribution, a fixed-step model moves the particles as the step does, a noiseless block moves only by its
// dynamics, periodic coordinates wrap, an update multiplies each weight by its likelihood and rejects what no particle
// can explain, a measurement known only by its likelihood, or one that supplies its own, is weighted by that
// likelihood, the filter matches KalmanFilter on a linear problem, mixture noise rides out injected outliers, and an
// ambiguity is carried until resolved.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <Eigen/LU>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::Constant;
using Emrys::ConstantVelocity;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::ParticleFilter;
using Emrys::ParticleSet;
using Emrys::RadiansPi;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace upkeep = Emrys::upkeep;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double accelDensity = 0.1;
constexpr double timeStep = 1.0;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

GaussianParams<State>
makeInitial()
{
    auto const variances = Vector4{4.0, 4.0, 1.0, 1.0};
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, Matrix4{variances.asDiagonal()}};
}

template <typename Space, typename G>
ParticleSet<Space>
drawParticles(GaussianParams<Space> const & gaussian, Eigen::Index count, G & rng)
{
    return ParticleSet<Space>{gaussian.space(), Emrys::drawSamples(gaussian.toDensity().value(), count, rng)};
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

template <typename Noise>
auto
makeFix(Eigen::Vector2d const & z, Noise noise)
{
    return observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(std::move(noise))
        ),
        z
    );
}

auto
makeFix(Eigen::Vector2d const & z)
{
    return makeFix(z, GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R});
}

// Range from the origin, which can't tell a position from its mirror image.
auto
makeRange(double z)
{
    auto const range = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(Emrys::block<Position>(in, x).squaredNorm()))};
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.04}})
    );
    return observe(Emrys::methods::unscented(), range, Eigen::Matrix<double, 1, 1>{z});
}

// The log-likelihood of a position fix z with Laplace noise of scale `b` on each axis at the
// position p; with `impossible`, no position can explain it.
double
laplaceLogLikelihood(Eigen::Vector2d const & z, Eigen::Vector2d const & p, double b, bool impossible)
{
    if (impossible)
    {
        return -std::numeric_limits<double>::infinity();
    }
    return -2 * std::log(2 * b) - (z - p).lpNorm<1>() / b;
}

// The Laplace likelihood as a model's own logLikelihood(g).
auto
laplaceLikelihood(double b, bool impossible)
{
    return Emrys::logLikelihood([b, impossible](auto const & in, auto const & x, auto const & observed) {
        return laplaceLogLikelihood(observed, Emrys::block<Position>(in, x), b, impossible);
    });
}

// A position fix with Laplace noise, known only by its likelihood: a model with no function, so no
// method, noise, or Kalman update.
auto
makeLaplaceFix(Eigen::Vector2d const & z, double b = 0.5, bool impossible = false)
{
    return observe(Emrys::measurementModel(Emrys::reads(Emrys::path<Position>), laplaceLikelihood(b, impossible)), z);
}

static_assert(Emrys::Measurement<decltype(makeLaplaceFix(Eigen::Vector2d{}))>);
static_assert(!decltype(makeLaplaceFix(Eigen::Vector2d{}))::ModelType::HasFunction);

// The same fix as a full measurement, with a Gaussian noise for a Kalman update and the Laplace
// likelihood for the filters that weight points, which weight by the likelihood.
auto
makeLaplaceOrGaussianFix(Eigen::Vector2d const & z, double b = 0.5)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(
            [](auto const & in, auto const &, auto const & x) {
                return Emrys::block<Position>(in, x);
            },
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}),
            laplaceLikelihood(b, false)
        ),
        z
    );
}

static_assert(Emrys::Measurement<decltype(makeLaplaceOrGaussianFix(Eigen::Vector2d{}))>);
static_assert(Emrys::internal::hasOwnLikelihood<decltype(makeLaplaceOrGaussianFix(Eigen::Vector2d{}))>);

// The specific force an accelerometer reads, an uncertain input of the call.
struct SpecificForce : Linear<2>
{};

// Velocity driven by a measured acceleration, which arrives as an uncertain input, with a little
// velocity noise of its own, so the process noise and the input both contribute.
template <typename P, typename V>
auto
accelerometerDriven()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Eigen::Matrix<T, 2, 1> const a = Emrys::block<SpecificForce>(in, x);
            Emrys::block<P>(out, y) = Emrys::block<P>(in, x) + T(dt) * Emrys::block<V>(in, x) + T(dt * dt / 2) * a;
            Emrys::block<V>(out, y) = Emrys::block<V>(in, x) + T(dt) * a;
            return y;
        },
        Emrys::reads(Emrys::path<P>, Emrys::path<V>, Emrys::path<SpecificForce>),
        Emrys::writes(Emrys::path<P>, Emrys::path<V>),
        Emrys::continuousNoise(Emrys::on<V>(0.05))
    );
}

template <typename Particles>
auto
weightedMoments(Particles const & particles)
{
    return particles.moments().value();
}

// Checks sample moments of `count` independent draws against a Gaussian's, to within `sigmas`
// standard errors of each entry.
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

}  // namespace

// logLikelihoods(measurement, space, states) over a sample matrix equals the Gaussian likelihood by
// hand, the mixture noise's density at each residual, and a hand-written measurement's own
// logLikelihood(), and agrees with the single-point logLikelihood(); the space named as a template
// argument gives what the space passed does.
TEST(Measurement, LogLikelihoodsMatchThePointwiseLikelihood)
{
    auto rng = std::mt19937_64{3};
    auto const particles = drawParticles(makeInitial(), 7, rng);
    auto const z = Eigen::Vector2d{0.5, -0.25};

    // From the additive noise: one Gaussian, then a mixture.
    auto const fix = makeFix(z);
    auto const gaussian = Emrys::logLikelihoods<State>(fix, particles.states());
    auto robust = GaussianMixtureParams<Linear<2>>{2};
    robust.add(std::log(0.9), Eigen::Vector2d::Zero(), R);
    robust.add(std::log(0.1), Eigen::Vector2d{1.0, 0.0}, 100 * R);
    auto const mixture = Emrys::logLikelihoods<State>(makeFix(z, robust), particles.states());
    // The measurement's own.
    auto const laplace = makeLaplaceFix(z);
    auto const custom = Emrys::logLikelihoods<State>(laplace, particles.states());

    ASSERT_EQ(gaussian.size(), 7);
    // The fixed-size space named as a template argument ([space-storage]) or passed gives the same values.
    EXPECT_EQ(gaussian.matrix(), Emrys::logLikelihoods(fix, State{}, particles.states()).matrix());
    EXPECT_EQ(
        gaussian.matrix(),
        Emrys::logLikelihoods<State>(Emrys::execution::sequential, fix, particles.states()).matrix()
    );
    EXPECT_EQ(gaussian(0), Emrys::logLikelihood<State>(fix, particles.state(0)));
    for (Eigen::Index i = 0; i < particles.size(); ++i)
    {
        Eigen::Vector2d const residual = z - particles.state(i).head<2>();
        double const byHand =
            -0.5 * (2 * std::log(2 * pi) + std::log(R.determinant()) + residual.dot(R.inverse() * residual));
        EXPECT_NEAR(gaussian(i), byHand, 1e-12);
        EXPECT_NEAR(gaussian(i), Emrys::logLikelihood(fix, State{}, particles.state(i)), 1e-12);
        EXPECT_NEAR(mixture(i), robust.toDensity()->logDensity(residual), 1e-12);
        EXPECT_NEAR(custom(i), laplaceLogLikelihood(z, particles.state(i).head<2>(), 0.5, false), 1e-12);
    }
}

// The deduction guide picks the space and model, the state is a ParticleSet, and model() and
// state() are writable.
TEST(ParticleFilter, DeducesItsTypeAndHoldsItsParts)
{
    auto rng = std::mt19937_64{1};
    auto filter = ParticleFilter{makeModel(), drawParticles(makeInitial(), 10, rng)};
    static_assert(std::is_same_v<decltype(filter), ParticleFilter<State, decltype(makeModel())>>);
    static_assert(std::is_same_v<decltype(filter)::StateType, ParticleSet<State>>);
    EXPECT_EQ(filter.state().size(), 10);
    filter.model() = ConstantVelocity<Position, Velocity>{0.5};
    EXPECT_EQ(std::as_const(filter).model().spectralDensity(), 0.5 * Eigen::Matrix2d::Identity());
    filter.state().setLogWeight(0, -1.0);
    EXPECT_EQ(filter.state().logWeight(0), -1.0);
}

// setState() replaces the particles with a set of another size: the steps that follow equal those
// of a filter made with the new set, from the same generator.
TEST(ParticleFilter, SetStateReplacesTheParticles)
{
    auto rng = std::mt19937_64{3};
    auto const replacement = drawParticles(makeInitial(), 40, rng);
    auto filter = ParticleFilter{makeModel(), drawParticles(makeInitial(), 10, rng)};
    auto reference = ParticleFilter{makeModel(), replacement};
    auto const fix = makeFix(Eigen::Vector2d{1.0, 0.5});
    filter.predict(rng, timeStep);
    static_cast<void>(filter.update(fix));

    filter.setState(replacement);
    ASSERT_EQ(filter.state().size(), 40);
    for (auto * const each : {&filter, &reference})
    {
        auto stepRng = std::mt19937_64{4};
        each->predict(stepRng, timeStep);
        EXPECT_TRUE(each->update(fix)->accepted);
    }
    EXPECT_EQ(filter.state().states(), reference.state().states());
    EXPECT_TRUE((filter.state().logWeights() == reference.state().logWeights()).all());
}

// 40,000 equal-weight particles moved through an accelerometer-driven model with an uncertain input
// and process noise have sample moments within five standard errors of the affine Kalman
// prediction.
TEST(ParticleFilter, PredictMatchesTheKalmanPredictionInDistribution)
{
    // Equal-weight particles drawn from a Gaussian, moved through a model driven by an uncertain
    // accelerometer reading and process noise of its own, are a draw from the Kalman prediction.
    constexpr Eigen::Index count = 40000;
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
    auto const model = accelerometerDriven<Position, Velocity>();
    auto rng = std::mt19937_64{11};
    auto filter = ParticleFilter{model, drawParticles(makeInitial(), count, rng)};
    filter.predict(rng, 2.0, accelerometer);

    auto const expected = Emrys::predict(methods::linearized(), makeInitial(), model, 2.0, accelerometer);
    expectMomentsNear(weightedMoments(filter.state()), expected.mean(), expected.covarianceLower(), double(count), 5.0);
}

// With a FixedStepModel [fixed-step-models], from the same particles, input, and seed, the filter moves every
// particle exactly as one predicting over the step does: the noise and the draws are the same.
TEST(ParticleFilter, AFixedStepModelMovesParticlesAsTheStepDoes)
{
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
    auto const model = accelerometerDriven<Position, Velocity>();
    auto rng = std::mt19937_64{19};
    auto const particles = drawParticles(makeInitial(), 100, rng);
    auto continuous = ParticleFilter{model, particles};
    auto fixed = ParticleFilter{Emrys::FixedStepModel{model, State{}, 2.0}, particles};
    static_assert(Emrys::isFixedStep<decltype(fixed)::ModelType>);

    auto continuousRng = std::mt19937_64{23};
    auto fixedRng = std::mt19937_64{23};
    for (int k = 0; k < 3; ++k)
    {
        continuous.predict(continuousRng, 2.0, accelerometer);
        fixed.predict(fixedRng, accelerometer);
    }
    for (Eigen::Index i = 0; i < fixed.state().size(); ++i)
    {
        EXPECT_EQ(fixed.state().state(i), continuous.state().state(i)) << "particle " << i;
    }
}

// A Constant block contributes no noise, making Q singular; the semidefinite factor still works and
// the bias coordinate does not move while the others do.
TEST(ParticleFilter, ANoiselessBlockMovesOnlyByItsDynamics)
{
    // A Constant block contributes no process noise, so Q is singular; the bias must not move.
    using Biased = Composite<Position, Velocity, Bias>;
    auto const model = compose(ConstantVelocity<Position, Velocity>{accelDensity}, Constant<Bias>{});
    auto const mean = Eigen::Matrix<double, 5, 1>{0.0, 0.0, 1.0, 0.5, 0.3};
    auto const initial = GaussianParams<Biased>{Biased{}, mean, Eigen::Matrix<double, 5, 5>::Identity()};
    auto rng = std::mt19937_64{5};
    auto filter = ParticleFilter{model, drawParticles(initial, 100, rng)};
    Eigen::MatrixXd const before = filter.state().states();
    filter.predict(rng, timeStep);
    EXPECT_EQ(filter.state().states().row(4), before.row(4));
    EXPECT_GT((filter.state().states().topRows(4) - before.topRows(4)).cwiseAbs().minCoeff(), 0.0);
}

// Predicting headings near pi keeps them in [-pi, pi), most having crossed the seam, with the
// tangent-space mean just past it.
TEST(ParticleFilter, PredictWrapsAPeriodicCoordinate)
{
    struct Heading : RadiansPi<1>
    {};

    struct Rate : Linear<1>
    {};

    using Turning = Composite<Heading, Rate>;
    auto const model = ConstantVelocity<Heading, Rate>{0.01};
    auto const initial =
        GaussianParams<Turning>{Turning{}, Eigen::Vector2d{pi - 0.05, 0.2}, Eigen::Matrix2d{{0.01, 0}, {0, 0.001}}};
    auto rng = std::mt19937_64{7};
    auto filter = ParticleFilter{model, drawParticles(initial, 1000, rng)};
    filter.predict(rng, 1.0);
    auto const headings = filter.state().states().row(0);
    EXPECT_GE(headings.minCoeff(), -pi);
    EXPECT_LT(headings.maxCoeff(), pi);
    // Most particles crossed the seam; the tangent-space mean sits just past it.
    EXPECT_GT((headings.array() < 0).count(), 800);
    EXPECT_NEAR(weightedMoments(filter.state()).mean()(0), -pi + 0.15, 0.02);
}

// update() multiplies each prior weight by its likelihood and normalizes, returns the log-evidence
// over the normalized prior, leaves the states alone, and records the weights' log total as zero.
TEST(ParticleFilter, UpdateMultipliesEachWeightByItsLikelihood)
{
    auto rng = std::mt19937_64{13};
    auto particles = drawParticles(makeInitial(), 50, rng);
    Eigen::ArrayXd prior = Eigen::ArrayXd::LinSpaced(50, -1.0, 1.0);
    particles.setLogWeights(prior);
    auto filter = ParticleFilter{makeModel(), particles};
    auto const fix = makeFix(Eigen::Vector2d{1.0, 0.5});
    auto const result = filter.update(fix);

    Eigen::ArrayXd const likelihoods = Emrys::logLikelihoods<State>(fix, particles.states());
    double const priorTotal = std::log(prior.exp().sum());
    Eigen::ArrayXd const joint = prior - priorTotal + likelihoods;
    double const evidence = std::log(joint.exp().sum());
    EXPECT_TRUE(result->accepted);
    EXPECT_NEAR(result->logLikelihood(), evidence, 1e-12);
    EXPECT_LT((filter.state().logWeights() - (joint - evidence)).abs().maxCoeff(), 1e-12);
    EXPECT_EQ(filter.state().states(), particles.states());
    // The update records the log total weight it normalized to, without summing the weights again.
    EXPECT_EQ(filter.state().logTotalWeight().value(), 0.0);
}

// When every likelihood is zero the update is rejected with log-likelihood minus infinity and the
// weights are unchanged.
TEST(ParticleFilter, RejectsAnObservationNoParticleCanExplain)
{
    auto rng = std::mt19937_64{17};
    auto particles = drawParticles(makeInitial(), 20, rng);
    particles.setLogWeights(Eigen::ArrayXd::LinSpaced(20, 0.0, 1.0));
    auto filter = ParticleFilter{makeModel(), particles};
    auto const result = filter.update(makeLaplaceFix(Eigen::Vector2d::Zero(), 0.5, true));
    ASSERT_TRUE(result);
    EXPECT_FALSE(result->accepted);
    EXPECT_EQ(result->logLikelihood(), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(filter.state().logWeights().matrix(), particles.logWeights().matrix());
}

// Prior weights that fail to define a distribution, as a caller can leave them through the mutable
// state(), are an error rather than a rejection (decisions.md [update-outcomes]): the update returns
// their WeightError and leaves them as they were.
TEST(ParticleFilter, ReturnsTheErrorOfWeightsThatCannotBeNormalized)
{
    auto rng = std::mt19937_64{19};
    auto particles = drawParticles(makeInitial(), 2, rng);
    particles.setLogWeights(Eigen::ArrayXd::Constant(2, -std::numeric_limits<double>::infinity()));
    auto filter = ParticleFilter{makeModel(), particles};
    auto const result = filter.update(makeFix(Eigen::Vector2d{1.0, 0.5}));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), Emrys::WeightError::ZeroSum);
    EXPECT_EQ(filter.state().logWeights().matrix(), particles.logWeights().matrix());
}

// A measurement known only by its likelihood weights the particles by it, and a full measurement
// that also supplies a logLikelihood() is weighted by that likelihood, not by its Gaussian noise
// ([likelihood-measurements]): both give the same weights, which differ from the Gaussian fix's.
TEST(ParticleFilter, WeightsByAMeasurementsOwnLikelihood)
{
    auto rng = std::mt19937_64{19};
    auto const particles = drawParticles(makeInitial(), 50, rng);
    auto const z = Eigen::Vector2d{1.0, 0.5};
    auto byLikelihood = ParticleFilter{makeModel(), particles};
    auto byOverride = ParticleFilter{makeModel(), particles};
    auto byNoise = ParticleFilter{makeModel(), particles};
    auto const likelihood = byLikelihood.update(makeLaplaceFix(z));
    auto const overridden = byOverride.update(makeLaplaceOrGaussianFix(z));
    auto const gaussian = byNoise.update(makeFix(z));

    Eigen::ArrayXd laplace = Eigen::ArrayXd{particles.size()};
    for (Eigen::Index i = 0; i < particles.size(); ++i)
    {
        laplace(i) = laplaceLogLikelihood(z, particles.state(i).head<2>(), 0.5, false);
    }
    double const evidence = std::log((particles.logWeights() + laplace).exp().sum());
    EXPECT_NEAR(likelihood->logLikelihood(), evidence, 1e-12);
    EXPECT_LT(
        (byLikelihood.state().logWeights() - (particles.logWeights() + laplace - evidence)).abs().maxCoeff(),
        1e-12
    );
    EXPECT_EQ(overridden->logLikelihood(), likelihood->logLikelihood());
    EXPECT_EQ(byOverride.state().logWeights().matrix(), byLikelihood.state().logWeights().matrix());
    // The weights, not the evidence: a sum over the particles, it can come close by chance.
    EXPECT_TRUE(gaussian->accepted);
    EXPECT_GT((byNoise.state().logWeights() - byLikelihood.state().logWeights()).abs().maxCoeff(), 0.1);
}

namespace
{

// How far a resampling particle filter strays from the Kalman filter over a constant-velocity run
// with position fixes, where the Kalman filter is exact: the worst error of any mean, in standard
// deviations, of any covariance entry, relative to the standard deviations, and of the
// log-likelihood. The fixes are a fixed offset from the truth with an ordinary NIS, so only the
// particles are random; a fix far in the tail would leave a few particles with all the weight.
struct LinearRun
{
    double mean = 0;
    double cov = 0;
    double logLikelihood = 0;
    int resampled = 0;
};

LinearRun
runLinear(unsigned seed, Eigen::Index count)
{
    auto rng = std::mt19937_64{seed};
    auto kalman = KalmanFilter{makeModel(), makeInitial()};
    auto particles = ParticleFilter{makeModel(), drawParticles(makeInitial(), count, rng)};
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    Vector4 truth = makeInitial().mean();
    auto run = LinearRun{};
    for (int step = 1; step <= 8; ++step)
    {
        truth = transitionMatrix() * truth;
        Eigen::Vector2d const z = truth.head<2>() + 0.8 * Eigen::Vector2d{std::cos(2.4 * step), std::sin(2.4 * step)};
        kalman.predict(timeStep);
        particles.predict(rng, timeStep);
        auto const exact = kalman.update(makeFix(z));
        auto const sampled = particles.update(makeFix(z));
        EXPECT_TRUE(sampled->accepted);
        run.logLikelihood =
            std::max(run.logLikelihood, std::abs(sampled->logLikelihood() - exact.value().logLikelihood()));
        auto const moments = weightedMoments(particles.state());
        auto const & mean = kalman.state().mean();
        auto const & cov = kalman.state().covarianceLower();
        for (Eigen::Index i = 0; i < 4; ++i)
        {
            run.mean = std::max(run.mean, std::abs(moments.mean()(i) - mean(i)) / std::sqrt(cov(i, i)));
            for (Eigen::Index j = 0; j < 4; ++j)
            {
                run.cov = std::max(
                    run.cov,
                    std::abs(moments.covarianceLower()(i, j) - cov(i, j)) / std::sqrt(cov(i, i) * cov(j, j))
                );
            }
        }
        run.resampled += resampling(particles.state()).value() ? 1 : 0;
    }
    return run;
}

}  // namespace

// Over eight steps with 20,000 particles and resampling, the worst errors against the exact Kalman
// filter in mean (in standard deviations), covariance (relative), and log-likelihood stay under
// 0.15, about twice the worst seen over 30 seeds.
TEST(ParticleFilter, MatchesTheKalmanFilterOnALinearProblem)
{
    // Errors carry over between steps through resampling, so the bounds are about twice the worst
    // seen over 30 seeds with two standard libraries, not a single step's standard error.
    LinearRun const run = runLinear(19, 20000);
    EXPECT_LT(run.mean, 0.15);
    EXPECT_LT(run.cov, 0.15);
    EXPECT_LT(run.logLikelihood, 0.15);
    EXPECT_GT(run.resampled, 0);
}

// The Kalman filter's outlier test: with inlier/outlier mixture noise the particle filter stays
// within 5 units of the truth, while with inlier noise alone it is dragged past 5.
TEST(ParticleFilter, MixtureNoiseRidesOutInjectedOutliers)
{
    // The Kalman filter's outlier test: every seventh fix is off by 60 m. With the inlier/outlier
    // mixture as its noise, the particle filter's likelihood ignores them; with the inlier noise
    // alone, it is dragged off, though less than a Kalman filter (15 m), since the weights can only
    // pick among the particles it has.
    constexpr int steps = 70;
    constexpr Eigen::Index count = 2000;
    Eigen::Matrix2d const inlierCov = Eigen::Matrix2d::Identity();
    auto robustNoise = GaussianMixtureParams<Linear<2>>{2};
    robustNoise.add(std::log(0.95), Eigen::Vector2d::Zero(), inlierCov);
    robustNoise.add(std::log(0.05), Eigen::Vector2d::Zero(), 1e4 * inlierCov);
    auto const plainNoise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), inlierCov};

    auto rng = std::mt19937_64{17};
    auto robust = ParticleFilter{makeModel(), drawParticles(makeInitial(), count, rng)};
    auto plain = ParticleFilter{makeModel(), drawParticles(makeInitial(), count, rng)};
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng), upkeep::roughen(rng, 0.25));

    auto gaussian = std::normal_distribution<double>{0.0, 1.0};
    Vector4 truth = makeInitial().mean();
    double robustError = 0;
    double plainError = 0;
    for (int step = 1; step <= steps; ++step)
    {
        truth = transitionMatrix() * truth;
        bool const outlier = step % 7 == 0;
        Eigen::Vector2d const z = truth.head<2>() + Eigen::Vector2d{gaussian(rng), gaussian(rng)}
                                + (outlier ? Eigen::Vector2d{60.0, 0.0} : Eigen::Vector2d::Zero());
        robust.predict(rng, timeStep);
        plain.predict(rng, timeStep);
        ASSERT_TRUE(robust.update(makeFix(z, robustNoise))->accepted);
        ASSERT_TRUE(plain.update(makeFix(z, plainNoise))->accepted);
        ASSERT_TRUE(resampling(robust.state()));
        ASSERT_TRUE(resampling(plain.state()));
        if (step > 5)
        {
            robustError =
                std::max(robustError, (weightedMoments(robust.state()).mean().head<2>() - truth.head<2>()).norm());
            plainError =
                std::max(plainError, (weightedMoments(plain.state()).mean().head<2>() - truth.head<2>()).norm());
        }
    }
    EXPECT_LT(robustError, 5.0);
    EXPECT_GT(plainError, 5.0);
}

// Particles split between a position and its mirror image keep about half the weight on each side
// under five range measurements, and one position fix puts over 99.9% on the right side.
TEST(ParticleFilter, CarriesAnAmbiguityUntilAMeasurementResolvesIt)
{
    // Particles around a position and its mirror image; ranges from the origin can't tell them
    // apart, so both halves keep their weight until a fix picks one.
    constexpr Eigen::Index count = 4000;
    auto rng = std::mt19937_64{23};
    auto const variances = Vector4{0.25, 0.25, 0.01, 0.01};
    auto const right = GaussianParams<State>{State{}, Vector4{3.0, 4.0, 0.5, 0.0}, Matrix4{variances.asDiagonal()}};
    auto const left = GaussianParams<State>{State{}, Vector4{-3.0, -4.0, -0.5, 0.0}, Matrix4{variances.asDiagonal()}};
    auto particles = drawParticles(right, count, rng);
    particles.append(drawParticles(left, count, rng));
    auto filter = ParticleFilter{makeModel(), particles};
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));

    auto const rightWeight = [&] {
        auto const weights = filter.state().normalizedWeights().value();
        double sum = 0;
        for (Eigen::Index i = 0; i < filter.state().size(); ++i)
        {
            if (filter.state().state(i)(0) > 0)
            {
                sum += weights(i);
            }
        }
        return sum;
    };
    auto truth = Eigen::Vector2d{3.0, 4.0};
    for (int step = 1; step <= 5; ++step)
    {
        truth += Eigen::Vector2d{0.5, 0.0};
        filter.predict(rng, timeStep);
        ASSERT_TRUE(filter.update(makeRange(truth.norm()))->accepted);
        ASSERT_TRUE(resampling(filter.state()));
        EXPECT_NEAR(rightWeight(), 0.5, 0.1) << "step " << step;
    }
    ASSERT_TRUE(filter.update(makeFix(truth))->accepted);
    EXPECT_GT(rightWeight(), 0.999);
}

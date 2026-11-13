// Every overload that takes an execution policy, run under std::execution::par (and par_unseq)
// against the same call under Sequential: ParticleSet, the resamplers, KernelDensity,
// GaussianMixtureParams, the upkeep actions and triggers, ParticleFilter, ParticleHistory and
// GaussianMixtureHistory (step 14, [smoothing]), the Rao-Blackwellized particle and point-mass filters,
// GaussianMixtureFilter and its predictJoint(), fitMixture(), and the Monte Carlo divergences, over
// the headers they live in, and the particle filter under policies of the user's own. Roadmap step
// 18; the evidence for [execution-policies]: a for-each (a prediction, a gather, a likelihood, a Monte Carlo term)
// gives Sequential's result bitwise, and a reduction (a total weight, an evidence, moments) matches
// it to rounding and is the same from run to run. Counts are several chunks (internal::chunkSize)
// so the reductions split.

#include <EmrysFilter/EmrysFilter.h>

#include "standard_policies.h"
#include "user_policies.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#if EMRYS_FILTER_TEST_STANDARD_POLICIES

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::ParticleSet;
using Emrys::PerPointCovariance;
using Emrys::ScalarOf;
using Emrys::SharedCovariance;

namespace execution = Emrys::execution;
namespace methods = Emrys::methods;
namespace resamplers = Emrys::resamplers;

namespace
{

using Index = Eigen::Index;

constexpr Index manyParticles = 5 * Emrys::internal::chunkSize + 123;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Heading : Emrys::RadiansPi<1>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

using State = Composite<Position, Velocity>;
using Pose = Composite<Heading, Position>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Vector1 = Eigen::Matrix<double, 1, 1>;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{0.1};
}

GaussianParams<State>
makeInitial()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    cov(1, 3) = cov(3, 1) = -0.8;
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, cov};
}

auto
makeFix(Eigen::Vector2d const & z)
{
    return observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}})
        ),
        z
    );
}

// A fix whose noise is a mixture of an inlier and an outlier Gaussian, whose density the
// likelihoods evaluate at every particle at once.
auto
makeMixtureFix(Eigen::Vector2d const & z)
{
    auto noise = GaussianMixtureParams<Linear<2>>{Linear<2>{}};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity() * 100.0);
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

// The specific force an accelerometer reads, an uncertain input of the call.
struct SpecificForce : Linear<2>
{};

// Velocity driven by a measured acceleration, which arrives as an uncertain input.
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
        Emrys::continuousNoise(0.05)  // on every coordinate it writes, so the Cramér-Rao bound's Q is invertible
    );
}

// Particles drawn from the initial Gaussian, with uneven log-weights.
ParticleSet<State>
makeParticles(Index count, unsigned seed)
{
    auto rng = std::mt19937_64{seed};
    auto const initial = makeInitial();
    Emrys::SampleSet<State, double> const states = Emrys::drawSamples(initial.toDensity().value(), count, rng);
    auto spread = std::normal_distribution<double>{0.0, 1.5};
    auto logWeights = Eigen::ArrayXd{count};
    for (Index i = 0; i < count; ++i)
    {
        logWeights(i) = spread(rng);
    }
    return ParticleSet<State>{State{}, states, logWeights};
}

// Particles in a space with a periodic heading, whose moments are tangent-space moments, with
// headings that straddle the seam at pi.
ParticleSet<Pose>
makePoses(Index count, unsigned seed)
{
    auto rng = std::mt19937_64{seed};
    auto normal = std::normal_distribution<double>{};
    auto states = Eigen::Matrix<double, 3, Eigen::Dynamic>{3, count};
    auto logWeights = Eigen::ArrayXd{count};
    for (Index i = 0; i < count; ++i)
    {
        states(0, i) = std::numbers::pi - 0.1 + 0.2 * normal(rng);
        states(1, i) = normal(rng);
        states(2, i) = 2.0 * normal(rng);
        logWeights(i) = normal(rng);
    }
    return ParticleSet<Pose>{Pose{}, states, logWeights};
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

// Expects \a a and \a b to agree to \a tolerance in every coefficient.
template <typename A, typename B>
void
expectNear(A const & a, B const & b, double tolerance)
{
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    EXPECT_LE((a.array() - b.array()).abs().maxCoeff(), tolerance);
}

// A position fix with Laplace noise, known only by its likelihood [likelihood-measurements].
auto
makeLaplaceFix(Eigen::Vector2d const & z)
{
    return observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Position>),
            Emrys::logLikelihood([](auto const & in, auto const & x, auto const & observed) {
                return -Eigen::Vector2d{observed - Emrys::block<Position>(in, x)}.lpNorm<1>();
            })
        ),
        z
    );
}

}  // namespace

// The reductions over a particle set (total weight, effective sample size, moments, and
// Silverman's bandwidth) agree with Sequential to rounding in a linear and a periodic space, and
// are the same from run to run and under par_unseq. The tolerances are a few hundred ulps of the
// values, which are of order one to a few thousand.
TEST(Parallel, ParticleSetReductionsMatchSequential)
{
    auto const check = [](auto const & particles) {
        auto const total = particles.logTotalWeight().value();
        EXPECT_EQ(particles.logTotalWeight(execution::sequential).value(), total);
        EXPECT_NEAR(particles.logTotalWeight(std::execution::par).value(), total, 1e-12);
        EXPECT_EQ(
            particles.logTotalWeight(std::execution::par).value(),
            particles.logTotalWeight(std::execution::par_unseq).value()
        );

        auto const ess = particles.ess().value();
        EXPECT_NEAR(particles.ess(std::execution::par).value(), ess, 1e-10 * ess);

        auto const moments = particles.moments().value();
        auto const parallel = particles.moments(std::execution::par).value();
        expectNear(parallel.mean(), moments.mean(), 1e-12);
        expectNear(parallel.covarianceLower(), moments.covarianceLower(), 1e-12);
        expectSame(parallel.covarianceLower(), parallel.covarianceLower().transpose());
        auto const again = particles.moments(std::execution::par_unseq).value();
        expectSame(again.mean(), parallel.mean());
        expectSame(again.covarianceLower(), parallel.covarianceLower());

        expectNear(particles.kdeBandwidth(std::execution::par).value(), particles.kdeBandwidth().value(), 1e-12);
    };
    check(makeParticles(manyParticles, 3));
    check(makePoses(manyParticles, 5));
}

// logSumExp() without a policy is Sequential's, bitwise, in its array and its callable form, and
// under par and par_unseq it agrees to rounding and is the same from run to run. The inputs span
// forty nats, so the sum depends on rescaling as the maximum moves.
TEST(Parallel, LogSumExpMatchesSequential)
{
    auto rng = std::mt19937{11};
    auto uniform = std::uniform_real_distribution<double>{-40.0, 0.0};
    auto logs = Eigen::ArrayXd(manyParticles);
    for (Index i = 0; i < logs.size(); ++i)
    {
        logs(i) = uniform(rng);
    }
    auto const logAt = [&](Index i) {
        return logs(i);
    };

    double const sum = Emrys::logSumExp(logs);
    EXPECT_EQ(Emrys::logSumExp(execution::sequential, logs), sum);
    EXPECT_NEAR(Emrys::logSumExp(std::execution::par, logs), sum, 1e-12);
    EXPECT_EQ(Emrys::logSumExp(std::execution::par, logs), Emrys::logSumExp(std::execution::par_unseq, logs));

    double const streamed = Emrys::logSumExp<double>(logs.size(), logAt);
    EXPECT_EQ(Emrys::logSumExp<double>(execution::sequential, logs.size(), logAt), streamed);
    EXPECT_NEAR(streamed, sum, 1e-12);
    EXPECT_NEAR(Emrys::logSumExp<double>(std::execution::par, logs.size(), logAt), streamed, 1e-12);
}

// reweight() is a for-each: the same states and weights as Sequential. removeIf() keeps the same
// particles in the same order, and its removed weight agrees to rounding.
TEST(Parallel, ParticleSetReweightAndRemoveIfMatchSequential)
{
    auto const body = [](auto & particle) {
        particle.setState(particle.state() * 0.5);
        return -0.25 * particle.state().squaredNorm();
    };
    auto sequential = makeParticles(manyParticles, 7);
    auto parallel = sequential;
    sequential.reweight(body);
    parallel.reweight(std::execution::par, body);
    expectSame(parallel.states(), sequential.states());
    expectSame(parallel.logWeights(), sequential.logWeights());

    auto const far = [](double, auto const & state) {
        return state(0) > 0.5;
    };
    auto const removed = sequential.removeIf<true>(far).value();
    auto const removedInParallel = parallel.removeIf<true>(std::execution::par, far).value();
    EXPECT_NEAR(removedInParallel, removed, 1e-12);
    ASSERT_LT(parallel.size(), manyParticles);
    expectSame(parallel.states(), sequential.states());
    expectSame(parallel.logWeights(), sequential.logWeights());
}

// Each resampler under a policy draws from the generator as the sequential form does and inverts
// the same cumulative weights, so the ancestors are Sequential's (resamplers::Multinomial's in the
// order of its draws, so compared sorted, as is the residual resampler's multinomial remainder).
// Random weights put no position within rounding of a boundary.
TEST(Parallel, ResamplersMatchSequential)
{
    auto weightRng = std::mt19937_64{11};
    auto uniform = std::uniform_real_distribution<double>{0.0, 1.0};
    auto weights = std::vector<double>(static_cast<std::size_t>(manyParticles));
    for (double & w : weights)
    {
        w = uniform(weightRng);
    }
    weights[10] = 0.0;
    weights.back() = 0.0;
    auto const check = [&](auto const & resampler, bool sort) {
        auto sequential = std::vector<Index>(static_cast<std::size_t>(manyParticles - 17));
        auto parallel = std::vector<Index>(sequential.size());
        auto first = std::mt19937_64{13};
        auto second = std::mt19937_64{13};
        resampler(std::span<double const>{weights}, std::span<Index>{sequential}, first);
        resampler(std::execution::par, std::span<double const>{weights}, std::span<Index>{parallel}, second);
        EXPECT_EQ(first(), second()) << "the same draws were taken";
        if (sort)
        {
            std::ranges::sort(parallel);
            std::ranges::sort(sequential);
        }
        EXPECT_EQ(parallel, sequential);
        for (Index const a : parallel)
        {
            EXPECT_NE(a, 10);
            EXPECT_NE(a, manyParticles - 1);
        }
    };
    check(Emrys::resamplers::systematic(), false);
    check(Emrys::resamplers::stratified(), false);
    check(Emrys::resamplers::multinomial(), true);
    check(Emrys::resamplers::residual(Emrys::resamplers::systematic()), false);
    check(Emrys::resamplers::residual(), true);
    static_assert(Emrys::ResamplerUnder<
                  Emrys::resamplers::Systematic,
                  decltype(std::execution::par) const &,
                  double,
                  std::mt19937_64>);
}

// A particle set resampled under a policy, systematically, takes the same ancestors' states as
// under Sequential, with equal weights.
TEST(Parallel, ParticleSetResampleMatchesSequential)
{
    auto sequential = makeParticles(manyParticles, 17);
    auto parallel = sequential;
    auto first = std::mt19937_64{19};
    auto second = std::mt19937_64{19};
    ASSERT_TRUE(sequential.resample(first));
    ASSERT_TRUE(parallel.resample(std::execution::par, second));
    expectSame(parallel.states(), sequential.states());
    expectSame(parallel.logWeights(), sequential.logWeights());
    EXPECT_TRUE(std::ranges::equal(parallel.ancestors(), sequential.ancestors()));
}

// A particle filter's prediction under a policy, with an uncertain input, moves every particle
// exactly as Sequential does from the same seed, also for a fixed-step model. Its update's
// likelihoods are a for-each, so the weights agree to the rounding of the evidence, with Gaussian
// and with mixture noise.
TEST(Parallel, ParticleFilterMatchesSequential)
{
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
    auto const model = accelerometerDriven<Position, Velocity>();
    auto sequential = Emrys::ParticleFilter{model, makeParticles(manyParticles, 23)};
    auto parallel = Emrys::ParticleFilter{model, sequential.state()};
    auto first = std::mt19937_64{29};
    auto second = std::mt19937_64{29};
    sequential.predict(first, 0.5, accelerometer);
    parallel.predict(std::execution::par, second, 0.5, accelerometer);
    expectSame(parallel.state().states(), sequential.state().states());

    auto const fix = makeFix(Eigen::Vector2d{0.5, 0.2});
    auto const update = sequential.update(fix);
    auto const updateInParallel = parallel.update(std::execution::par, fix);
    EXPECT_EQ(updateInParallel->accepted, update->accepted);
    EXPECT_NEAR(updateInParallel->logLikelihood(), update->logLikelihood(), 1e-12);
    expectNear(parallel.state().logWeights(), sequential.state().logWeights(), 1e-12);

    auto const outlying = makeMixtureFix(Eigen::Vector2d{8.0, -3.0});
    EXPECT_NEAR(
        parallel.update(std::execution::par, outlying)->logLikelihood(),
        sequential.update(outlying)->logLikelihood(),
        1e-12
    );
    expectNear(parallel.state().logWeights(), sequential.state().logWeights(), 1e-12);

    auto fixed = Emrys::ParticleFilter{Emrys::FixedStepModel{model, State{}, 0.5}, makeParticles(manyParticles, 31)};
    auto fixedInParallel = Emrys::ParticleFilter{fixed};
    fixed.predict(first, accelerometer);
    fixedInParallel.predict(std::execution::par_unseq, second, accelerometer);
    expectSame(fixedInParallel.state().states(), fixed.state().states());
}

// A batched range with a bias, an uncertain input it is affine in, weighs points by one call per
// chunk under par, with the Jacobian in the bias by autodiff over each chunk, and gives Sequential's
// weights bitwise: each entry is written by one chunk, which computes it as one call over every point
// does ([batched-functions]).
TEST(Parallel, BatchedLikelihoodsWithAnInputMatchSequential)
{
    struct Bias : Linear<1>
    {};

    auto const model = Emrys::measurementModel(
        Emrys::batched([](auto const & in, auto const &, auto const & x, auto & y) {
            y.values() = Emrys::block<Position>(in, x).colwise().norm() + Emrys::block<Bias>(in, x);
        }),
        Emrys::reads(Emrys::path<Position>, Emrys::path<Bias>),
        Emrys::additiveNoise(
            GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.25}}
        ),
        Emrys::affineIn(Emrys::path<Bias>)
    );
    auto const bias = GaussianParams<Bias>{Eigen::Matrix<double, 1, 1>{0.1}, Eigen::Matrix<double, 1, 1>{0.04}};
    auto const measurement = Emrys::observe(model, Eigen::Matrix<double, 1, 1>{2.5}, bias);
    auto const particles = makeParticles(manyParticles, 41);
    expectSame(
        Emrys::logLikelihoods(std::execution::par, measurement, State{}, particles.states()),
        Emrys::logLikelihoods(measurement, State{}, particles.states())
    );
}

// The same particle filter under policies of the user's own, one on threads of its own and one
// that visits the particles in reverse, gives std::execution::par's prediction, update, and
// moments bitwise: a user's policy runs the same chunked loops, so even the reductions agree.
TEST(Parallel, ParticleFilterUnderAUsersPolicyMatchesPar)
{
    auto const accelerometer =
        GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
    auto const model = accelerometerDriven<Position, Velocity>();
    auto const fix = makeFix(Eigen::Vector2d{0.5, 0.2});
    auto byPar = Emrys::ParticleFilter{model, makeParticles(manyParticles, 23)};
    auto rng = std::mt19937_64{29};
    byPar.predict(std::execution::par, rng, 0.5, accelerometer);
    auto const update = byPar.update(std::execution::par, fix);
    auto const moments = byPar.state().moments(std::execution::par).value();
    auto const check = [&](auto const & policy) {
        auto filter = Emrys::ParticleFilter{model, makeParticles(manyParticles, 23)};
        auto same = std::mt19937_64{29};
        filter.predict(policy, same, 0.5, accelerometer);
        expectSame(filter.state().states(), byPar.state().states());
        EXPECT_EQ(filter.update(policy, fix)->logLikelihood(), update->logLikelihood());
        expectSame(filter.state().logWeights(), byPar.state().logWeights());
        auto const own = filter.state().moments(policy).value();
        expectSame(own.mean(), moments.mean());
        expectSame(own.covarianceLower(), moments.covarianceLower());
    };
    check(EmrysTest::ThreadsPolicy{});
    check(EmrysTest::ReversedPolicy{});
}

// Ancestor tracing over a ParticleHistory is a for-each over the latest particles at each step,
// both the lineage lookup and the gather, so every step's smoothed states are Sequential's
// bitwise. Four records with a resample between each, and one before the latest.
TEST(Parallel, ParticleHistoryMatchesSequential)
{
    ParticleSet<State> set = makeParticles(manyParticles, 37);
    auto sequential = Emrys::ParticleHistory<State>{};
    auto rng = std::mt19937_64{41};
    for (int k = 0; k < 4; ++k)
    {
        sequential.push(set);
        set.setStates((set.states().array() + 0.25).matrix());
        set.setLogWeights(makeParticles(manyParticles, 43 + unsigned(k)).logWeights());
        ASSERT_TRUE(set.resample(rng));
    }
    auto parallel = Emrys::ParticleHistory<State>{sequential};
    sequential.smooth(set);
    parallel.smooth(std::execution::par, set);
    auto expected = sequential.begin();
    for (auto const & step : parallel)
    {
        expectSame(step.smoothed.states(), expected->smoothed.states());
        expectSame(step.smoothed.logWeights(), expected->smoothed.logWeights());
        ++expected;
    }
}

// A kernel density's log-density is a reduction over the particles, so agrees to rounding; its
// moments too; and its draws' log-densities are a for-each over the draws, so are Sequential's.
// The mixture's draws are the same.
TEST(Parallel, DensitiesMatchSequential)
{
    auto const particles = makeParticles(manyParticles, 37);
    auto const kde = particles.toDensity();
    ASSERT_TRUE(kde);
    auto const x = Eigen::Vector4d{0.5, -0.5, 1.0, 0.0};
    EXPECT_NEAR(kde->logDensity(std::execution::par, x), kde->logDensity(x), 1e-12);
    expectNear(kde->moments(std::execution::par).covarianceLower(), kde->moments().covarianceLower(), 1e-12);

    auto first = std::mt19937_64{41};
    auto second = std::mt19937_64{41};
    auto const draws = Emrys::drawSamplesWithLogDensity(*kde, 200, first);
    auto const drawsInParallel = Emrys::drawSamplesWithLogDensity(std::execution::par, *kde, 200, second);
    expectSame(drawsInParallel.samples, draws.samples);
    expectSame(drawsInParallel.logDensity, draws.logDensity);

    auto mixture = GaussianMixtureParams<State>{State{}};
    mixture.add(std::log(0.3), makeInitial().mean(), makeInitial().covarianceLower());
    mixture.add(std::log(0.7), Vector4{5, -2, 0, 1}, Matrix4::Identity());
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density);
    auto const mixtureDraws = Emrys::drawSamplesWithLogDensity(*density, manyParticles, first);
    auto const mixtureDrawsInParallel =
        Emrys::drawSamplesWithLogDensity(std::execution::par, *density, manyParticles, second);
    expectSame(mixtureDrawsInParallel.samples, mixtureDraws.samples);
    expectSame(mixtureDrawsInParallel.logDensity, mixtureDraws.logDensity);
}

// WhenEss made with a policy takes the effective sample size under it, and Resample and Roughen made
// with it move the particles as Sequential does, up to the rounding of the bandwidth.
TEST(Parallel, UpkeepMatchesSequential)
{
    auto sequential = makeParticles(manyParticles, 43);
    auto parallel = sequential;
    auto first = std::mt19937_64{47};
    auto second = std::mt19937_64{47};
    auto const resampling =
        Emrys::upkeep::whenEss(std::less{}, 2.0, Emrys::upkeep::resample(first), Emrys::upkeep::roughen(first, 0.5));
    auto const inParallel = Emrys::upkeep::whenEss(
        std::execution::par,
        std::less{},
        2.0,
        Emrys::upkeep::resample(std::execution::par, second),
        Emrys::upkeep::roughen(std::execution::par, second, 0.5)
    );
    ASSERT_TRUE(resampling(sequential).value());
    ASSERT_TRUE(inParallel(parallel).value());
    expectNear(parallel.states(), sequential.states(), 1e-10);
    expectSame(parallel.logWeights(), sequential.logWeights());
}

// Roughen in three dimensions, an odd number, where the standard library's normal distribution
// keeps a value from one pair of draws for the next: Sequential, drawing particle by particle, must
// still take the draws another policy takes all at once, so the two agree to the bandwidth's rounding.
TEST(Parallel, RoughenMatchesSequentialInAnOddDimension)
{
    auto rng = std::mt19937_64{59};
    auto normal = std::normal_distribution<double>{};
    auto states = Eigen::Matrix3Xd{3, manyParticles};
    std::generate(states.data(), states.data() + states.size(), [&] {
        return normal(rng);
    });
    auto sequential = ParticleSet<Linear<3>>{Linear<3>{}, states};
    auto parallel = sequential;
    auto first = std::mt19937_64{61};
    auto second = std::mt19937_64{61};
    ASSERT_TRUE(Emrys::upkeep::roughen(first, 0.5)(sequential).value());
    ASSERT_TRUE(Emrys::upkeep::roughen(std::execution::par, second, 0.5)(parallel).value());
    expectNear(parallel.states(), sequential.states(), 1e-10);
}

// The Rao-Blackwellized particle filter under a policy, with either covariance form: its
// prediction draws as Sequential does and moves each particle and Gaussian as a for-each, so is
// Sequential's; its update, by a Kalman update or a likelihood, agrees to the rounding of the
// evidence; and resampling takes the same
// ancestors and their Gaussians. A fixed-step model's prediction, through its cached transition,
// is Sequential's too.
TEST(Parallel, RbParticleFilterMatchesSequential)
{
    auto const check = [](auto initial) {
        auto sequential = Emrys::RbParticleFilter{makeModel(), std::move(initial)};
        auto parallel = sequential;
        auto first = std::mt19937_64{59};
        auto second = std::mt19937_64{59};
        sequential.predict(first, 1.0);
        parallel.predict(std::execution::par, second, 1.0);
        expectSame(parallel.state().particles().states(), sequential.state().particles().states());
        for (Index i : {Index{0}, Index{777}, sequential.state().size() - 1})
        {
            expectSame(parallel.state().conditional(i).mean(), sequential.state().conditional(i).mean());
            expectSame(
                parallel.state().conditional(i).covarianceLower(),
                sequential.state().conditional(i).covarianceLower()
            );
        }

        auto const fix = makeFix(Eigen::Vector2d{1.0, 0.5});
        auto const update = sequential.update(fix);
        auto const updateInParallel = parallel.update(std::execution::par, fix);
        ASSERT_TRUE(update);
        ASSERT_TRUE(updateInParallel);
        EXPECT_NEAR(updateInParallel->logLikelihood(), update->logLikelihood(), 1e-12);
        EXPECT_EQ(updateInParallel->innovationSize(), update->innovationSize());
        expectSame(updateInParallel->logLikelihoods(), update->logLikelihoods());
        expectSame(updateInParallel->nis(), update->nis());
        expectNear(parallel.state().logWeights(), sequential.state().logWeights(), 1e-12);

        // A likelihood over the particles is a for-each too.
        auto const laplace = makeLaplaceFix(Eigen::Vector2d{0.5, 1.0});
        auto const weighed = sequential.update(laplace);
        auto const weighedInParallel = parallel.update(std::execution::par, laplace);
        ASSERT_TRUE(weighed);
        ASSERT_TRUE(weighedInParallel);
        EXPECT_NEAR(weighedInParallel->logLikelihood(), weighed->logLikelihood(), 1e-12);
        expectSame(weighedInParallel->logLikelihoods(), weighed->logLikelihoods());
        expectNear(parallel.state().logWeights(), sequential.state().logWeights(), 1e-12);

        ASSERT_TRUE(sequential.state().resample(first));
        ASSERT_TRUE(parallel.state().resample(std::execution::par, second));
        expectSame(parallel.state().particles().states(), sequential.state().particles().states());
        expectSame(parallel.state().conditional(5).mean(), sequential.state().conditional(5).mean());

        auto fixed = Emrys::RbParticleFilter{Emrys::FixedStepModel{makeModel(), State{}, 1.0}, sequential.state()};
        auto fixedInParallel = fixed;
        fixed.predict(first);
        fixedInParallel.predict(std::execution::par, second);
        expectSame(fixedInParallel.state().particles().states(), fixed.state().particles().states());
        for (Index i : {Index{0}, fixed.state().size() - 1})
        {
            expectSame(fixedInParallel.state().conditional(i).mean(), fixed.state().conditional(i).mean());
            expectSame(
                fixedInParallel.state().conditional(i).covarianceLower(),
                fixed.state().conditional(i).covarianceLower()
            );
        }
    };
    auto rng = std::mt19937_64{61};
    check(Emrys::drawParticleGaussian<Position>(makeInitial(), manyParticles, rng));
    check(
        Emrys::drawParticleGaussian<Position, PerPointCovariance>(makeInitial(), Emrys::internal::chunkSize + 50, rng)
    );
}

// The point-mass filter under a policy, with either covariance form: its prediction is a for-each
// over the cells, so is Sequential's, and its update, by a Kalman update or a likelihood, agrees to
// the rounding of the evidence. The
// grid is offset by a quarter cell so that no point sits at the origin, where the range's
// Jacobian is undefined. A fixed-step model's prediction is Sequential's too.
TEST(Parallel, RbPointMassFilterMatchesSequential)
{
    using Grid = Composite<Position, Bias, Drift>;
    using Rest = Composite<Bias, Drift>;
    auto const model = ConstantVelocity<Bias, Drift>{0.01};
    auto const rest =
        GaussianParams<Rest>{Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Vector2d{4, 0.01}.asDiagonal().toDenseMatrix()};
    auto const grid =
        Emrys::regularGrid<Grid, Position>(Eigen::Vector2d{-19.75, -19.75}, Eigen::Vector2d{0.5, 0.5}, {80, 80});
    auto const rangeModel = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{
                Eigen::numext::sqrt(T(Emrys::block<Position>(in, x).squaredNorm())) + Emrys::block<Bias>(in, x)(0)
            };
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Bias>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.25}})
    );
    auto const range = observe(methods::linearized(), rangeModel, Vector1{12.0});
    auto const check = [&](auto covariance) {
        auto sequential = Emrys::RbPointMassFilter{model, grid, rest, covariance};
        auto parallel = sequential;
        sequential.predict(1.0);
        parallel.predict(std::execution::par, 1.0);
        auto const update = sequential.update(range);
        auto const updateInParallel = parallel.update(std::execution::par, range);
        ASSERT_TRUE(update);
        ASSERT_TRUE(updateInParallel);
        EXPECT_NEAR(updateInParallel->logLikelihood(), update->logLikelihood(), 1e-12);
        expectSame(updateInParallel->logLikelihoods(), update->logLikelihoods());
        ASSERT_EQ(parallel.state().size(), sequential.state().size());
        for (Index i : {Index{0}, Index{3000}, sequential.state().size() - 1})
        {
            expectSame(parallel.state().conditional(i).mean(), sequential.state().conditional(i).mean());
            expectSame(
                parallel.state().conditional(i).covarianceLower(),
                sequential.state().conditional(i).covarianceLower()
            );
        }

        // A likelihood over the grid is a for-each too.
        auto const laplace = makeLaplaceFix(Eigen::Vector2d{3.0, -4.0});
        auto const weighed = sequential.update(laplace);
        auto const weighedInParallel = parallel.update(std::execution::par, laplace);
        ASSERT_TRUE(weighed);
        ASSERT_TRUE(weighedInParallel);
        EXPECT_NEAR(weighedInParallel->logLikelihood(), weighed->logLikelihood(), 1e-12);
        expectSame(weighedInParallel->logLikelihoods(), weighed->logLikelihoods());
        ASSERT_EQ(parallel.state().size(), sequential.state().size());
        expectNear(parallel.state().logWeights(), sequential.state().logWeights(), 1e-12);

        auto fixed =
            Emrys::RbPointMassFilter{Emrys::FixedStepModel{model, grid.restSpace(), 1.0}, grid, rest, covariance};
        auto fixedInParallel = fixed;
        fixed.predict();
        fixedInParallel.predict(std::execution::par);
        for (Index i : {Index{0}, Index{3000}, fixed.state().size() - 1})
        {
            expectSame(fixedInParallel.state().conditional(i).mean(), fixed.state().conditional(i).mean());
            expectSame(
                fixedInParallel.state().conditional(i).covarianceLower(),
                fixed.state().conditional(i).covarianceLower()
            );
        }
    };
    check(SharedCovariance{});
    check(PerPointCovariance{});
}

// The Gaussian mixture filter under a policy predicts and updates each component as a for-each
// and reduces them in order, so its state and update are Sequential's exactly, with a fixed-step
// model too.
TEST(Parallel, GaussianMixtureFilterMatchesSequential)
{
    auto rng = std::mt19937_64{67};
    auto normal = std::normal_distribution<double>{};
    auto prior = GaussianMixtureParams<State>{State{}};
    for (int k = 0; k < 50; ++k)
    {
        prior.add(
            normal(rng),
            Vector4{5 * normal(rng), 5 * normal(rng), normal(rng), normal(rng)},
            makeInitial().covarianceLower()
        );
    }
    auto sequential = Emrys::GaussianMixtureFilter{makeModel(), prior};
    auto parallel = Emrys::GaussianMixtureFilter{makeModel(), prior};
    sequential.predict(1.0);
    parallel.predict(std::execution::par, 1.0);
    auto const fix = makeMixtureFix(Eigen::Vector2d{1.0, -1.0});
    auto const update = sequential.update(fix, Emrys::reductions::keep());
    auto const updateInParallel = parallel.update(std::execution::par, fix, Emrys::reductions::keep());
    ASSERT_TRUE(update);
    ASSERT_TRUE(updateInParallel);
    EXPECT_EQ(updateInParallel->logLikelihood(), update->logLikelihood());
    ASSERT_EQ(parallel.state().size(), sequential.state().size());
    for (Index k = 0; k < sequential.state().size(); ++k)
    {
        auto const & a = sequential.state()[k];
        auto const & b = parallel.state()[k];
        EXPECT_EQ(b.logWeight, a.logWeight);
        expectSame(b.component.mean(), a.component.mean());
        expectSame(b.component.covarianceLower(), a.component.covarianceLower());
    }
    for (Index i = 0; i < prior.size(); ++i)
    {
        EXPECT_TRUE(std::ranges::equal(updateInParallel->posteriorIndices(i), update->posteriorIndices(i)));
    }
    // Under noise of one Gaussian, Keep keeps each component's one branch.
    auto const plain = makeFix(Eigen::Vector2d{0.5, -0.5});
    auto const plainUpdate = sequential.update(plain, Emrys::reductions::keep());
    auto const plainInParallel = parallel.update(std::execution::par, plain, Emrys::reductions::keep());
    ASSERT_TRUE(plainUpdate);
    ASSERT_TRUE(plainInParallel);
    EXPECT_EQ(plainInParallel->logLikelihood(), plainUpdate->logLikelihood());
    for (Index k = 0; k < sequential.state().size(); ++k)
    {
        auto const & a = sequential.state()[k];
        auto const & b = parallel.state()[k];
        EXPECT_EQ(b.logWeight, a.logWeight);
        expectSame(b.component.mean(), a.component.mean());
        expectSame(b.component.covarianceLower(), a.component.covarianceLower());
    }

    auto fixed = Emrys::GaussianMixtureFilter{Emrys::FixedStepModel{makeModel(), State{}, 1.0}, prior};
    auto fixedInParallel = fixed;
    fixed.predict();
    fixedInParallel.predict(std::execution::par);
    for (Index k = 0; k < fixed.state().size(); ++k)
    {
        auto const & a = fixed.state()[k];
        auto const & b = fixedInParallel.state()[k];
        expectSame(b.component.mean(), a.component.mean());
        expectSame(b.component.covarianceLower(), a.component.covarianceLower());
    }
}

// The Gaussian mixture smoother under a policy: predictJoint() is a for-each over the components into
// the joint's slots, and each backward step's pairs and merges a for-each over the step's
// components, so the joints and the smoothed mixtures are Sequential's exactly, through Keep and
// merges back to 40 components, with a fixed-step model's predictJoint() too.
TEST(Parallel, GaussianMixtureSmoothingMatchesSequential)
{
    auto rng = std::mt19937_64{71};
    auto normal = std::normal_distribution<double>{};
    auto prior = GaussianMixtureParams<State>{State{}};
    for (int k = 0; k < 40; ++k)
    {
        prior.add(
            normal(rng),
            Vector4{5 * normal(rng), 5 * normal(rng), normal(rng), normal(rng)},
            makeInitial().covarianceLower()
        );
    }
    auto sequential = Emrys::GaussianMixtureFilter{makeModel(), prior};
    auto parallel = Emrys::GaussianMixtureFilter{makeModel(), prior};
    auto sequentialHistory = Emrys::GaussianMixtureHistory<State>{};
    auto parallelHistory = Emrys::GaussianMixtureHistory<State>{};
    for (int k = 1; k <= 3; ++k)
    {
        auto joint = sequential.predictJoint(1.0);
        auto jointInParallel = joint;  // storage that the parallel prediction refills
        parallel.predictJoint(std::execution::par, 1.0, jointInParallel);
        for (std::size_t n = 0; n < joint.joints.size(); ++n)
        {
            expectSame(jointInParallel.joints[n].predicted.mean(), joint.joints[n].predicted.mean());
            expectSame(jointInParallel.joints[n].cross, joint.joints[n].cross);
        }
        sequentialHistory.push(std::move(joint));
        parallelHistory.push(std::move(jointInParallel));
        auto const fix = makeMixtureFix(Eigen::Vector2d{1.0 * k, -1.0});
        EXPECT_TRUE(sequential.update(fix, Emrys::reductions::keep()));
        EXPECT_TRUE(parallel.update(fix, Emrys::reductions::keep()));
        for (auto * state : {&sequential.state(), &parallel.state()})
        {
            while (state->size() > 40)
            {
                static_cast<void>(state->merge(0, 1));
            }
        }
    }
    sequentialHistory.smooth(sequential.state());
    parallelHistory.smooth(std::execution::par, parallel.state());
    auto step = parallelHistory.begin();
    for (auto const & expected : sequentialHistory)
    {
        ASSERT_EQ(step->smoothed.size(), expected.smoothed.size());
        for (Index n = 0; n < expected.smoothed.size(); ++n)
        {
            auto const & a = expected.smoothed[n];
            auto const & b = step->smoothed[n];
            EXPECT_EQ(b.logWeight, a.logWeight);
            expectSame(b.component.mean(), a.component.mean());
            expectSame(b.component.covarianceLower(), a.component.covarianceLower());
        }
        ++step;
    }

    auto fixed = Emrys::GaussianMixtureFilter{Emrys::FixedStepModel{makeModel(), State{}, 1.0}, prior};
    auto fixedInParallel = Emrys::GaussianMixtureFilter{Emrys::FixedStepModel{makeModel(), State{}, 1.0}, prior};
    auto const fixedJoint = fixed.predictJoint();
    auto fixedJointInParallel = fixedJoint;
    fixedInParallel.predictJoint(std::execution::par, fixedJointInParallel);
    for (std::size_t n = 0; n < fixedJoint.joints.size(); ++n)
    {
        expectSame(
            fixedJointInParallel.joints[n].predicted.covarianceLower(),
            fixedJoint.joints[n].predicted.covarianceLower()
        );
        expectSame(fixedJointInParallel.joints[n].cross, fixedJoint.joints[n].cross);
    }
}

// fitMixture() under a policy seeds as Sequential does and runs EM with the E-step as a for-each
// and the M-step's moments as reductions, so the fit agrees to rounding, which EM's few
// iterations amplify only mildly.
TEST(Parallel, FitMixtureMatchesSequential)
{
    auto rng = std::mt19937_64{71};
    auto source = GaussianMixtureParams<State>{State{}};
    source.add(std::log(0.5), Vector4{-6, 0, 0, 0}, Matrix4::Identity());
    source.add(std::log(0.5), Vector4{6, 2, 0, 0}, Matrix4::Identity());
    auto const particles =
        ParticleSet<State>{State{}, Emrys::drawSamples(source.toDensity().value(), manyParticles, rng)};
    auto first = std::mt19937_64{73};
    auto second = std::mt19937_64{73};
    auto const fit = Emrys::fitMixture(particles, 2, first);
    auto const fitInParallel = Emrys::fitMixture(std::execution::par, particles, 2, second);
    ASSERT_TRUE(fit);
    ASSERT_TRUE(fitInParallel);
    EXPECT_EQ(fitInParallel->iterations, fit->iterations);
    EXPECT_NEAR(fitInParallel->logLikelihood, fit->logLikelihood, 1e-9 * std::abs(fit->logLikelihood));
    ASSERT_EQ(fitInParallel->mixture.size(), fit->mixture.size());
    for (Index k = 0; k < fit->mixture.size(); ++k)
    {
        expectNear(fitInParallel->mixture[k].component.mean(), fit->mixture[k].component.mean(), 1e-9);
    }
}

// The posterior Cramér-Rao bound (step 17g, [cramer-rao-bound]) under a policy, over several chunks of truths:
// its expectations are reductions over the columns, so a prediction by the model's Jacobian at
// each truth (a model driven by a sure input, which declares nothing affine, is differentiated at
// each) and an update by a range agree with Sequential's to rounding.
TEST(Parallel, PosteriorCramerRaoBoundMatchesSequential)
{
    auto const particles = makeParticles(manyParticles, 83);
    auto const & truths = particles.states();
    auto const anchor = Eigen::Vector2d{20.0, -15.0};
    auto const range = Emrys::measurementModel(
        [anchor]<typename T>(auto const & in, auto const &, auto const & x) {
            Eigen::Matrix<T, 2, 1> const offset = Emrys::block<Position>(in, x) - anchor.cast<T>();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(offset.squaredNorm()))};
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<1>>{Vector1::Zero(), Vector1{0.25}})
    );
    auto const driven = accelerometerDriven<Position, Velocity>();
    auto const force = Emrys::sure<SpecificForce>(Eigen::Vector2d{0.3, -0.1});
    auto sequential = Emrys::PosteriorCramerRaoBound<State>::create(makeInitial()).value();
    auto parallel = sequential;
    for (int k = 0; k < 3; ++k)
    {
        ASSERT_TRUE(sequential.predict(driven, truths, 1.0, force));
        ASSERT_TRUE(parallel.predict(std::execution::par, driven, truths, 1.0, force));
        ASSERT_TRUE(sequential.update(truths, range));
        ASSERT_TRUE(parallel.update(std::execution::par, truths, range));
        expectNear(parallel.info(), sequential.info(), 1e-10);
    }
}

// The Monte Carlo divergences under a policy draw as Sequential does and evaluate each draw as a
// for-each, so the estimates and their errors are Sequential's exactly.
TEST(Parallel, MonteCarloDivergencesMatchSequential)
{
    auto mixture = GaussianMixtureParams<State>{State{}};
    mixture.add(std::log(0.4), makeInitial().mean(), makeInitial().covarianceLower());
    mixture.add(std::log(0.6), Vector4{3, 1, 0, 0}, Matrix4::Identity());
    auto const gaussian = makeInitial();
    auto const p = mixture.toDensity();
    auto const q = gaussian.toDensity();
    ASSERT_TRUE(p && q);
    auto const method = Emrys::MonteCarlo{manyParticles};
    auto const compare = [](auto const & a, auto const & b) {
        EXPECT_EQ(b.value, a.value);
        EXPECT_EQ(b.standardError, a.standardError);
    };
    auto first = std::mt19937_64{79};
    auto second = std::mt19937_64{79};
    compare(
        Emrys::kullbackLeibler(method, *p, *q, first),
        Emrys::kullbackLeibler(std::execution::par, method, *p, *q, second)
    );
    compare(
        Emrys::bhattacharyya(method, *p, *q, first),
        Emrys::bhattacharyya(std::execution::par, method, *p, *q, second)
    );
    compare(Emrys::hellinger(method, *p, *q, first), Emrys::hellinger(std::execution::par, method, *p, *q, second));
    compare(
        Emrys::jensenShannon(method, *p, *q, first),
        Emrys::jensenShannon(std::execution::par, method, *p, *q, second)
    );
}

#endif  // EMRYS_FILTER_TEST_STANDARD_POLICIES

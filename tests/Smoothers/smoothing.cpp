// The Kalman, IMM, and particle smoothers (step 14; [smoothing]). predictJoint() returns the prior, the prediction,
// and their cross-covariance under every method and with a fixed-step model; the fixed-interval
// smoother over a GaussianHistory matches the batch least-squares posterior on an affine model, the
// unscented smoother matches the affine one there, and the fixed-lag window matches the
// fixed-interval pass over the same steps. The IMM's predictJoint() records mode-matched
// predictions; its smoother over a HybridGaussianHistory is the Kalman smoother with one mode,
// matches the backward equations of Nadarajah (2009) by hand, places a maneuver where the filter
// lags, and agrees between fixed-lag and fixed-interval. Ancestor tracing over a ParticleHistory
// follows a genealogy written by hand, refuses two resamples between records, and on the affine
// problem is near the Kalman smoother at short lags while its paths collapse at long ones. The
// hidden Markov model smoother over a
// CategoricalHistory (step 15; [emissions]), its transition posteriors, and its Viterbi path match
// enumeration of every value sequence, for a discrete chain and for a Markov-modulated Poisson
// process, are certain under identity emissions, and agree between fixed-lag and fixed-interval.
// Headers: Smoothers/GaussianHistory.h, Smoothers/HybridGaussianHistory.h,
// Smoothers/ParticleHistory.h, Smoothers/CategoricalHistory.h, Filters/KalmanFilter.h, Filters/ImmFilter.h,
// Filters/ParticleFilter.h, Filters/HiddenMarkovFilter.h, Distributions/ParticleSet.h,
// Models/ProcessModel.h. The quaternion case is in unequal_dimensions_filters.cpp, the allocation
// cases in allocation.cpp, allocation_mixtures.cpp, and allocation_particles.cpp, and the parallel
// tracing case in parallel.cpp.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Filters/HiddenMarkovFilter.h>
#include <EmrysFilter/Models/Emission.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Smoothers/CategoricalHistory.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Smoothers/HybridGaussianHistory.h>
#include <EmrysFilter/Smoothers/ParticleHistory.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory_resource>
#include <numbers>
#include <random>
#include <set>
#include <span>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::FixedStepModel;
using Emrys::GaussianHistory;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianHistory;
using Emrys::HybridParams;
using Emrys::ImmFilter;
using Emrys::KalmanFilter;
using Emrys::Lag;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::ParticleFilter;
using Emrys::ParticleHistory;
using Emrys::ParticleSet;
using Emrys::PredictedJoint;
using Emrys::ScalarOf;
using Emrys::SwitchingModel;

namespace methods = Emrys::methods;
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

constexpr double accelDensity = 0.1;  // white-noise acceleration spectral density
constexpr double timeStep = 1.0;
constexpr int stepCount = 8;

auto
makeModel(double density = accelDensity)
{
    return ConstantVelocity<Position, Velocity>{density};
}

GaussianParams<State>
makeInitial()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, cov};
}

Matrix4
transitionMatrix()
{
    Matrix4 f = Matrix4::Identity();
    f.block<2, 2>(0, 2) = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};
Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

// A position fix, z = p + v.
template <typename Method = methods::Linearized>
auto
makeFix(Eigen::Vector2d const & z, Method const & method = {})
{
    auto const position = [](auto const & in, auto const &, auto const & x) {
        return Emrys::block<Position>(in, x);
    };
    return observe(
        method,
        Emrys::measurementModel(
            position,
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), R})
        ),
        z
    );
}

// A target turning slowly, seen at step k.
Eigen::Vector2d
fixAt(int k)
{
    return Eigen::Vector2d{1.0, 0.5} * k + 0.8 * Eigen::Vector2d{std::cos(0.7 * k), std::sin(1.3 * k)};
}

// Runs the filter over fixes at steps 0 to stepCount, recording each prediction in history, and
// returns the final posterior.
template <typename Filter, typename History, typename Method = methods::Linearized>
GaussianParams<State>
runForward(Filter & filter, History & history, Method const & method = {})
{
    EXPECT_TRUE(filter.update(makeFix(fixAt(0), method)));
    for (int k = 1; k <= stepCount; ++k)
    {
        history.push(filter.predictJoint(timeStep));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k), method)));
    }
    return filter.state();
}

// The posterior of x_0 ... x_N given z_0 ... z_N by batch least squares in plain Eigen: the
// information matrix and vector of the prior on x_0, each step x_{k+1} - F x_k ~ N(0, Q), and each
// fix H x_k + v ~ N(z_k, R), solved for the mean and inverted for the covariance.
struct Batch
{
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov;
};

Batch
batchPosterior()
{
    int const n = 4 * (stepCount + 1);
    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd vector = Eigen::VectorXd::Zero(n);
    GaussianParams<State> const initial = makeInitial();
    Matrix4 const p0Inverse = initial.covarianceLower().inverse();
    info.topLeftCorner<4, 4>() += p0Inverse;
    vector.head<4>() += p0Inverse * initial.mean();

    Matrix4 const q = Emrys::FixedStepModel{makeModel(), State{}, timeStep}.noise();
    Matrix4 const qInverse = q.inverse();
    auto a = Eigen::Matrix<double, 4, 8>{};
    a << -transitionMatrix(), Matrix4::Identity();
    for (int k = 0; k < stepCount; ++k)
    {
        info.block<8, 8>(4 * k, 4 * k) += a.transpose() * qInverse * a;
    }
    Eigen::Matrix2d const rInverse = R.inverse();
    for (int k = 0; k <= stepCount; ++k)
    {
        info.block<4, 4>(4 * k, 4 * k) += H.transpose() * rInverse * H;
        vector.segment<4>(4 * k) += H.transpose() * rInverse * fixAt(k);
    }
    auto const llt = Eigen::LLT<Eigen::MatrixXd>{info};
    return {llt.solve(vector), llt.solve(Eigen::MatrixXd::Identity(n, n))};
}

}  // namespace

// predictJoint() under Linearized keeps the prior as given, predicts exactly as predict() does, and
// gives the cross-covariance P F^T of the hand transition; under Linearized and Unscented, on this
// affine model, it gives the same joint to rounding.
TEST(Smoothing, PredictJointHoldsThePriorThePredictionAndTheirCross)
{
    GaussianParams<State> const prior = makeInitial();
    auto const model = makeModel();
    PredictedJoint<State> const joint = Emrys::predictJoint(methods::linearized(), prior, model, timeStep);
    GaussianParams<State> const predicted = Emrys::predict(methods::linearized(), prior, model, timeStep);
    EXPECT_EQ(joint.filtered.mean(), prior.mean());
    EXPECT_EQ(joint.filtered.covarianceLower(), prior.covarianceLower());
    EXPECT_EQ(joint.predicted.mean(), predicted.mean());
    EXPECT_EQ(joint.predicted.covarianceLower(), predicted.covarianceLower());
    Matrix4 const cross = prior.covarianceLower() * transitionMatrix().transpose();
    EXPECT_TRUE(joint.cross.isApprox(cross, 1e-14));

    auto const linearized = Emrys::predictJoint(methods::linearized(), prior, model, timeStep);
    auto const unscented = Emrys::predictJoint(methods::unscented(), prior, model, timeStep);
    for (auto const * other : {&linearized, &unscented})
    {
        EXPECT_TRUE(other->predicted.mean().isApprox(predicted.mean(), 1e-12));
        EXPECT_TRUE(other->predicted.covarianceLower().isApprox(predicted.covarianceLower(), 1e-12));
        EXPECT_TRUE(other->cross.isApprox(cross, 1e-12));
    }
}

// A fixed-step model's predictJoint() gives the joint over its step, with the cross computed
// from its cached transition under Linearized and through the transform under Unscented, and a
// KalmanFilter's predictJoint() moves its state exactly as predict() does.
TEST(Smoothing, PredictJointWithAFixedStepModelAndAFilter)
{
    GaussianParams<State> const prior = makeInitial();
    auto const reference = Emrys::predictJoint(methods::linearized(), prior, makeModel(), timeStep);
    auto const cached = FixedStepModel{makeModel(), State{}, timeStep};
    static_assert(decltype(cached)::CachesTransition);
    auto const unscented = FixedStepModel{makeModel(), State{}, timeStep, methods::unscented()};
    for (auto const & joint : {Emrys::predictJoint(prior, cached), Emrys::predictJoint(prior, unscented)})
    {
        EXPECT_EQ(joint.filtered.mean(), prior.mean());
        EXPECT_TRUE(joint.predicted.mean().isApprox(reference.predicted.mean(), 1e-12));
        EXPECT_TRUE(joint.predicted.covarianceLower().isApprox(reference.predicted.covarianceLower(), 1e-12));
        EXPECT_TRUE(joint.cross.isApprox(reference.cross, 1e-12));
    }

    auto joint = KalmanFilter{makeModel(), prior};
    auto plain = KalmanFilter{makeModel(), prior};
    auto fixed = KalmanFilter{cached, prior};
    auto const recorded = joint.predictJoint(timeStep);
    auto const fixedRecorded = fixed.predictJoint();
    plain.predict(timeStep);
    EXPECT_EQ(joint.state().mean(), plain.state().mean());
    EXPECT_EQ(joint.state().covarianceLower(), plain.state().covarianceLower());
    EXPECT_EQ(recorded.predicted.mean(), plain.state().mean());
    EXPECT_EQ(recorded.filtered.covarianceLower(), prior.covarianceLower());
    EXPECT_EQ(fixedRecorded.predicted.mean(), fixed.state().mean());
    EXPECT_TRUE(fixed.state().covarianceLower().isApprox(plain.state().covarianceLower(), 1e-12));
}

// The fixed-interval smoother over eight affine steps with a fix at each gives the batch
// least-squares posterior's mean and marginal covariance at every step, the final filtered
// posterior included; the tolerance allows for the batch solve's rounding on a 36-coordinate
// system.
TEST(Smoothing, FixedIntervalMatchesBatchLeastSquares)
{
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    auto history = GaussianHistory<State>{};
    GaussianParams<State> const latest = runForward(filter, history);
    EXPECT_FALSE(history.lag());
    ASSERT_EQ(history.size(), stepCount);
    history.smooth(latest);

    Batch const batch = batchPosterior();
    int k = 0;
    for (auto const & step : history)
    {
        EXPECT_TRUE(step.smoothed.mean().isApprox(batch.mean.segment<4>(4 * k), 1e-9)) << "step " << k;
        EXPECT_TRUE(step.smoothed.covarianceLower().isApprox(batch.cov.block<4, 4>(4 * k, 4 * k), 1e-9))
            << "step " << k;
        // Smoothing never adds uncertainty.
        EXPECT_LE(step.smoothed.covarianceLower().trace(), step.joint.filtered.covarianceLower().trace())
            << "step " << k;
        ++k;
    }
    EXPECT_TRUE(latest.mean().isApprox(batch.mean.tail<4>(), 1e-9));
    EXPECT_TRUE(latest.covarianceLower().isApprox(batch.cov.bottomRightCorner<4, 4>(), 1e-9));
}

// On an affine model the unscented transform is exact, so the unscented smoother's estimates are
// the affine smoother's to rounding.
TEST(Smoothing, UnscentedMatchesAffineOnAnAffineModel)
{
    auto linearizedFilter = KalmanFilter{makeModel(), makeInitial()};
    auto unscentedFilter = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    auto linearized = GaussianHistory<State>{};
    auto unscented = GaussianHistory<State>{};
    linearized.smooth(runForward(linearizedFilter, linearized));
    unscented.smooth(runForward(unscentedFilter, unscented, methods::unscented()));
    auto expected = linearized.begin();
    for (auto const & step : unscented)
    {
        EXPECT_TRUE(step.smoothed.mean().isApprox(expected->smoothed.mean(), 1e-10));
        EXPECT_TRUE(step.smoothed.covarianceLower().isApprox(expected->smoothed.covarianceLower(), 1e-10));
        ++expected;
    }
}

// A fixed-lag history of three steps keeps the newest three, and its smoothed estimates are the
// fixed-interval pass's at the same steps, since the backward step at k reads only the steps after
// it; it is the same arithmetic, so they agree exactly.
TEST(Smoothing, FixedLagMatchesFixedIntervalOverTheWindow)
{
    auto fullFilter = KalmanFilter{makeModel(), makeInitial()};
    auto lagFilter = KalmanFilter{makeModel(), makeInitial()};
    auto full = GaussianHistory<State>{};
    auto window = GaussianHistory<State>{Lag{3}};
    GaussianParams<State> const latest = runForward(fullFilter, full);
    full.smooth(latest);
    window.smooth(runForward(lagFilter, window));
    ASSERT_EQ(window.lag(), 3);
    ASSERT_EQ(window.size(), 3);

    auto expected = full.begin();
    std::advance(expected, stepCount - 3);
    for (auto const & step : window)
    {
        EXPECT_EQ(step.joint.filtered.mean(), expected->joint.filtered.mean());
        EXPECT_EQ(step.smoothed.mean(), expected->smoothed.mean());
        EXPECT_EQ(step.smoothed.covarianceLower(), expected->smoothed.covarianceLower());
        ++expected;
    }
    // The lagged estimate after each step: x_{k-3|k} at the window's front.
    EXPECT_EQ(window.front().smoothed.mean(), std::next(full.begin(), stepCount - 3)->smoothed.mean());
}

// A history takes its nodes from the allocator it is given, keeps it when copied or moved, and
// copies into another allocator's nodes with the allocator-extended constructor.
TEST(Smoothing, HistoryUsesItsAllocator)
{
    using Pmr = std::pmr::polymorphic_allocator<>;
    auto first = std::pmr::monotonic_buffer_resource{};
    auto second = std::pmr::monotonic_buffer_resource{};
    auto history = GaussianHistory<State, double, Pmr>{Lag{4}, Pmr{&first}};
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    static_cast<void>(runForward(filter, history));
    EXPECT_EQ(history.get_allocator(), Pmr{&first});

    auto const copy = GaussianHistory<State, double, Pmr>{history, Pmr{&second}};
    EXPECT_EQ(copy.get_allocator(), Pmr{&second});
    EXPECT_EQ(copy.lag(), 4);
    EXPECT_EQ(copy.size(), 4);
    EXPECT_EQ(copy.back().joint.predicted.mean(), history.back().joint.predicted.mean());

    GaussianHistory<State, double, Pmr> const plain = history;
    EXPECT_EQ(plain.get_allocator(), Pmr{&first});
    EXPECT_EQ(plain.size(), 4);

    auto const moved = GaussianHistory<State, double, Pmr>{std::move(history)};
    EXPECT_EQ(moved.get_allocator(), Pmr{&first});
    EXPECT_EQ(moved.size(), 4);
}

namespace
{

using Line = Linear<1>;
using Row4 = Eigen::Matrix<double, 1, 4>;

// A resampler that picks the ancestors it is given, so that a genealogy can be written by hand.
struct FixedAncestors
{
    std::array<Eigen::Index, 4> indices;

    void
    operator()(std::span<double const> /*weights*/, std::span<Eigen::Index> ancestors, auto & /*rng*/) const
    {
        std::ranges::copy(indices, ancestors.begin());
    }
};

// Four particles on a line through three recorded steps: a prediction adds ten to every particle,
// a resample with the ancestors 2 2 0 1 falls before the second record, none before the third, and
// one with the ancestors 3 3 3 0 after it. The set stores a bandwidth throughout. Returns the latest
// set, with unequal weights.
ParticleSet<Line>
recordGenealogy(auto & histories)
{
    auto rng = std::mt19937_64{1};
    auto set = ParticleSet<Line>{Line{}, Row4{0, 1, 2, 3}};
    set.setBandwidth(Eigen::Matrix<double, 1, 1>{0.5});
    auto const record = [&] {
        for (auto * history : histories)
        {
            history->push(set);
        }
    };
    auto const predict = [&] {
        set.setStates((set.states().array() + 10.0).matrix());
    };
    record();  // step 0: 0 1 2 3
    predict();
    EXPECT_TRUE(set.resample(rng, FixedAncestors{{2, 2, 0, 1}}));
    record();  // step 1: 12 12 10 11
    predict();
    record();  // step 2: 22 22 20 21
    predict();
    EXPECT_TRUE(set.resample(rng, FixedAncestors{{3, 3, 3, 0}}));  // latest: 31 31 31 32
    set.setLogWeights(Eigen::Array4d{-1.0, -2.0, -3.0, -4.0});
    return set;
}

}  // namespace

// Each particle's smoothed states are its genealogy, traced by hand: latest particle 0 (31) came
// from 21, which was 11 before the second record, which was a copy of step 0's particle 1; the
// last (32) came from 22, 12, and 2. Every step takes the latest weights and drops the filtered
// set's bandwidth, which isn't the smoothed set's. A two-step window traces the same newest two
// steps.
TEST(Smoothing, ParticleTrajectoriesFollowTheirGenealogy)
{
    auto history = ParticleHistory<Line>{};
    auto window = ParticleHistory<Line>{Lag{2}};
    auto histories = std::array<ParticleHistory<Line> *, 2>{&history, &window};
    ParticleSet<Line> const latest = recordGenealogy(histories);
    ASSERT_TRUE(history.front().particles.bandwidth());
    history.smooth(latest);
    window.smooth(latest);

    auto const expected = std::array<Row4, 3>{Row4{1, 1, 1, 2}, Row4{11, 11, 11, 12}, Row4{21, 21, 21, 22}};
    auto const resampled = std::array<bool, 3>{false, true, false};
    ASSERT_EQ(history.size(), 3);
    std::size_t k = 0;
    for (auto const & step : history)
    {
        EXPECT_EQ(step.resampled, resampled[k]) << "step " << k;
        EXPECT_EQ(step.smoothed.states(), expected[k]) << "step " << k;
        EXPECT_TRUE((step.smoothed.logWeights() == latest.logWeights()).all()) << "step " << k;
        EXPECT_FALSE(step.smoothed.bandwidth()) << "step " << k;
        ++k;
    }
    ASSERT_EQ(window.size(), 2);
    EXPECT_EQ(window.front().smoothed.states(), expected[1]);
    EXPECT_EQ(window.back().smoothed.states(), expected[2]);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
// Two resamples between records leave only the second's ancestors, so the history refuses them.
TEST(SmoothingDeathTest, ParticleHistoryRefusesTwoResamplesBetweenRecords)
{
    auto rng = std::mt19937_64{1};
    auto set = ParticleSet<Line>{Line{}, Row4{0, 1, 2, 3}};
    auto history = ParticleHistory<Line>{};
    history.push(set);
    ASSERT_TRUE(set.resample(rng));
    ASSERT_TRUE(set.resample(rng));
    EXPECT_DEATH(history.push(set), "at most one resample");
}
#endif

namespace
{

// A particle filter's run over runForward()'s fixes: the history of the particles before each
// prediction, smoothed by the latest particles, and how many times the effective sample size fell
// below half and they were resampled.
struct ParticleRun
{
    ParticleHistory<State> history;
    int resampled = 0;
};

ParticleRun
runParticles(unsigned seed, Eigen::Index count)
{
    auto rng = std::mt19937_64{seed};
    auto const initial = makeInitial();
    auto filter = ParticleFilter{
        makeModel(),
        ParticleSet<State>{State{}, Emrys::drawSamples(initial.toDensity().value(), count, rng)}
    };
    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    auto run = ParticleRun{};
    static_cast<void>(filter.update(makeFix(fixAt(0))));
    run.resampled += resampling(filter.state()).value() ? 1 : 0;
    for (int k = 1; k <= stepCount; ++k)
    {
        run.history.push(filter.state());
        filter.predict(rng, timeStep);
        static_cast<void>(filter.update(makeFix(fixAt(k))));
        run.resampled += resampling(filter.state()).value() ? 1 : 0;
    }
    run.history.smooth(filter.state());
    return run;
}

// The number of distinct states in a smoothed set: the lineages that survive back to its step.
std::size_t
distinctStates(ParticleSet<State> const & smoothed)
{
    auto distinct = std::set<std::array<double, 4>>{};
    for (Eigen::Index j = 0; j < smoothed.size(); ++j)
    {
        auto const x = smoothed.state(j);
        distinct.insert({x(0), x(1), x(2), x(3)});
    }
    return distinct.size();
}

}  // namespace

// On the affine problem of runForward(), with 20,000 particles resampled when the effective sample
// size falls below half, the smoothed sets one to three steps back have the Kalman smoother's
// moments: each mean within 0.2 of its standard deviations and the covariance's trace within 10%,
// about twice the worst seen over 30 seeds (0.093 and 4.7%; Monte Carlo error carried through the
// resamples, not one step's standard error). Further back the paths collapse: the distinct states
// never grow going back, and eight steps back fewer than a tenth of the paths are distinct (650 to
// 750 over those seeds).
TEST(Smoothing, ParticleSmootherIsNearTheKalmanSmootherAtShortLags)
{
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    auto exact = GaussianHistory<State>{};
    exact.smooth(runForward(filter, exact));
    ParticleRun const run = runParticles(7, 20000);
    EXPECT_GT(run.resampled, 2);
    ASSERT_EQ(run.history.size(), stepCount);

    auto expected = exact.begin();
    std::size_t previous = 0;
    int k = 0;
    for (auto const & step : run.history)
    {
        std::size_t const distinct = distinctStates(step.smoothed);
        EXPECT_GE(distinct, previous) << "step " << k;
        previous = distinct;
        if (k >= stepCount - 3)
        {
            auto const moments = step.smoothed.moments();
            ASSERT_TRUE(moments);
            auto const & smoothed = expected->smoothed;
            Vector4 const error = (moments->mean() - smoothed.mean()).array().abs()
                                / smoothed.covarianceLower().diagonal().array().sqrt();
            EXPECT_LT(error.maxCoeff(), 0.2) << "step " << k;
            EXPECT_NEAR(moments->covarianceLower().trace() / smoothed.covarianceLower().trace(), 1.0, 0.1)
                << "step " << k;
        }
        ++expected;
        ++k;
    }
    EXPECT_LT(distinctStates(run.history.front().smoothed), std::size_t(2000));
}

namespace
{

struct Cruise
{};

struct Maneuvering
{};

struct Maneuver : Named<Cruise, Maneuvering>
{};

struct Alone : Named<Cruise>
{};

constexpr double quiet = 0.01;
constexpr double agile = 4.0;

// A chain that leaves cruise less readily than it leaves a maneuver, so that a swapped index in
// the backward transition shows.
MarkovChain<Maneuver> const chain{Maneuver{}, Eigen::Matrix2d{{0.9, 0.1}, {0.3, 0.7}}};

auto
makeSwitching()
{
    return SwitchingModel{
        chain,
        ConstantVelocity<Position, Velocity>{quiet},
        ConstantVelocity<Position, Velocity>{agile}
    };
}

auto
makeImm()
{
    auto const start = Eigen::Vector2d{0.7, 0.3};
    return ImmFilter{
        makeSwitching(),
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, start), makeInitial()}
    };
}

// Runs the IMM over fixes at steps 0 to stepCount through one kept joint, recording each
// prediction in history.
template <typename Filter, typename History>
void
runImm(Filter & filter, History & history)
{
    EXPECT_TRUE(filter.update(makeFix(fixAt(0))));
    auto joint = filter.predictJoint(timeStep);
    history.push(joint);
    EXPECT_TRUE(filter.update(makeFix(fixAt(1))));
    for (int k = 2; k <= stepCount; ++k)
    {
        filter.predictJoint(timeStep, joint);
        history.push(joint);
        EXPECT_TRUE(filter.update(makeFix(fixAt(k))));
    }
}

// The density of N(mean, cov) at x, by hand.
double
normalDensity(Vector4 const & x, Vector4 const & mean, Matrix4 const & cov)
{
    Vector4 const r = x - mean;
    double const squaredDistance = r.dot(cov.inverse() * r);
    return std::exp(-0.5 * squaredDistance) / std::sqrt(std::pow(2 * std::numbers::pi, 4) * cov.determinant());
}

// One mode's smoothed Gaussian and probability, by hand.
struct HandMode
{
    Vector4 mean;
    Matrix4 cov;
    double probability;
};

}  // namespace

// With one mode the IMM smoother is the Kalman smoother: the mixing is trivial and the mode keeps
// probability one, so the smoothed Gaussians agree to rounding.
TEST(Smoothing, OneModeImmSmootherIsTheKalmanSmoother)
{
    auto imm = ImmFilter{
        SwitchingModel{MarkovChain<Alone>{}, makeModel()},
        HybridParams{CategoricalParams<Alone>{}, makeInitial()}
    };
    auto kalman = KalmanFilter{makeModel(), makeInitial()};
    auto hybrid = HybridGaussianHistory<State, Alone>{};
    auto plain = GaussianHistory<State>{};
    runImm(imm, hybrid);
    GaussianParams<State> const latest = runForward(kalman, plain);
    hybrid.smooth(imm.state());
    plain.smooth(latest);

    auto expected = plain.begin();
    for (auto const & step : hybrid)
    {
        EXPECT_NEAR(step.smoothed.modes().toDensity().value().probability(0), 1.0, 1e-15);
        EXPECT_TRUE(step.smoothed.conditional(0).mean().isApprox(expected->smoothed.mean(), 1e-12));
        EXPECT_TRUE(
            step.smoothed.conditional(0).covarianceLower().isApprox(expected->smoothed.covarianceLower(), 1e-12)
        );
        ++expected;
    }
}

// A mode the filter never enters, under a chain that never switches, has no probability at any
// step: nothing smoothed comes from it, so it keeps its filtered Gaussian, and the other mode is
// the Kalman smoother's.
TEST(Smoothing, ImmSmootherKeepsAModeNeverEntered)
{
    auto filter = ImmFilter{
        SwitchingModel{
            MarkovChain<Maneuver>{Eigen::Matrix2d::Identity()},
            ConstantVelocity<Position, Velocity>{quiet},
            ConstantVelocity<Position, Velocity>{agile}
        },
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{1.0, 0.0}), makeInitial()}
    };
    auto kalman = KalmanFilter{ConstantVelocity<Position, Velocity>{quiet}, makeInitial()};
    auto hybrid = HybridGaussianHistory<State, Maneuver>{};
    auto plain = GaussianHistory<State>{};
    runImm(filter, hybrid);
    GaussianParams<State> const latest = runForward(kalman, plain);
    hybrid.smooth(filter.state());
    plain.smooth(latest);

    auto expected = plain.begin();
    for (auto const & step : hybrid)
    {
        EXPECT_EQ(step.smoothed.modes().toDensity().value().probability(1), 0.0);
        EXPECT_TRUE(step.smoothed.conditional(1).mean() == step.joint.joints[1].filtered.mean());
        EXPECT_TRUE(step.smoothed.conditional(0).mean().isApprox(expected->smoothed.mean(), 1e-12));
        ++expected;
    }

    // Under a chain that leaves the mode only toward the other, nothing reaches it either, so a
    // prediction mixes it from its own Gaussian alone and leaves its probability at zero.
    auto leaving = ImmFilter{
        SwitchingModel{
            MarkovChain<Maneuver>{Eigen::Matrix2d{{1.0, 0.0}, {0.5, 0.5}}},
            ConstantVelocity<Position, Velocity>{quiet},
            ConstantVelocity<Position, Velocity>{agile}
        },
        HybridParams{CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{1.0, 0.0}), makeInitial()}
    };
    leaving.predict(timeStep);
    EXPECT_EQ(leaving.state().modes().toDensity().value().probability(1), 0.0);
    EXPECT_TRUE(
        leaving.state().conditional(1).mean().isApprox(makeInitial().mean() + Vector4{1.0, 0.5, 0.0, 0.0}, 1e-12)
    );
}

// predictJoint() records the mode probabilities and chain before the step and, for each mode, its
// filtered Gaussian moved through its own model, F m and F P F^T + Q_j with cross P F^T, not the
// IMM's prediction from the mixed Gaussian; and it moves the state exactly as predict() does.
TEST(Smoothing, ImmPredictJointRecordsModeMatchedPredictions)
{
    auto recording = makeImm();
    auto plain = makeImm();
    EXPECT_TRUE(recording.update(makeFix(fixAt(0))));
    EXPECT_TRUE(plain.update(makeFix(fixAt(0))));
    auto const before = recording.state();
    auto const joint = recording.predictJoint(timeStep);
    plain.predict(timeStep);

    EXPECT_EQ(joint.modes.logWeights().matrix(), before.modes().logWeights().matrix());
    EXPECT_EQ(joint.transitions, chain.matrix());
    ASSERT_EQ(joint.joints.size(), 2);
    Matrix4 const f = transitionMatrix();
    for (Eigen::Index j = 0; j < 2; ++j)
    {
        auto const & modeJoint = joint.joints[std::size_t(j)];
        auto const & filtered = before.conditional(j);
        Matrix4 const q = Emrys::FixedStepModel{makeModel(j == 0 ? quiet : agile), State{}, timeStep}.noise();
        EXPECT_EQ(modeJoint.filtered.mean(), filtered.mean());
        EXPECT_TRUE(modeJoint.predicted.mean().isApprox(f * filtered.mean(), 1e-14));
        EXPECT_TRUE(
            modeJoint.predicted.covarianceLower().isApprox(f * filtered.covarianceLower() * f.transpose() + q, 1e-14)
        );
        EXPECT_TRUE(modeJoint.cross.isApprox(filtered.covarianceLower() * f.transpose(), 1e-14));
        EXPECT_EQ(recording.state().conditional(j).mean(), plain.state().conditional(j).mean());
        EXPECT_EQ(recording.state().conditional(j).covarianceLower(), plain.state().conditional(j).covarianceLower());
    }
    EXPECT_EQ(
        recording.state().modes().toDensity().value().logProbabilities().matrix(),
        plain.state().modes().toDensity().value().logProbabilities().matrix()
    );

    // Fixed-step models record the same joint over their step.
    auto fixed = ImmFilter{
        SwitchingModel{
            chain,
            FixedStepModel{makeModel(quiet), State{}, timeStep},
            FixedStepModel{makeModel(agile), State{}, timeStep}
        },
        before
    };
    auto const fixedJoint = fixed.predictJoint();
    for (std::size_t j = 0; j < 2; ++j)
    {
        EXPECT_TRUE(fixedJoint.joints[j].predicted.mean().isApprox(joint.joints[j].predicted.mean(), 1e-14));
        EXPECT_TRUE(fixedJoint.joints[j].cross.isApprox(joint.joints[j].cross, 1e-14));
    }
}

// The backward pass matches the equations of Nadarajah (2009), 8.11 to 8.17, written out in plain
// Eigen over two affine modes with an asymmetric chain: backward transition and mixing, the moment
// match, and the mode-matched RTS step from each filtered Gaussian (with the plus sign of the RTS
// covariance update); and the mode probabilities by the expectation correction of Barber (2006)
// under its mean approximation, b_ij times the density of mode i's smoothed mean under mode j's
// filtered Gaussian moved through mode i's model, normalized over j and summed over i.
TEST(Smoothing, ImmSmootherMatchesTheBackwardEquationsByHand)
{
    auto filter = makeImm();
    auto history = HybridGaussianHistory<State, Maneuver>{};
    runImm(filter, history);
    history.smooth(filter.state());

    Matrix4 const f = transitionMatrix();
    auto const q = std::array<Matrix4, 2>{
        Emrys::FixedStepModel{makeModel(quiet), State{}, timeStep}.noise(),
        Emrys::FixedStepModel{makeModel(agile), State{}, timeStep}.noise()
    };
    Eigen::Matrix2d const p = chain.matrix();

    auto next = std::array<HandMode, 2>{};
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        auto const & latest = filter.state();
        next[std::size_t(i)] = {
            latest.conditional(i).mean(),
            latest.conditional(i).covarianceLower(),
            latest.modes().toDensity().value().probability(i)
        };
    }
    auto expected = std::vector<std::array<HandMode, 2>>(std::size_t(history.size()));
    auto step = history.end();
    for (std::size_t t = expected.size(); t-- > 0;)
    {
        --step;
        auto const & joint = step->joint;
        Eigen::Vector2d const mu = joint.modes.logWeights().exp().matrix();
        Eigen::Vector2d const e = p.transpose() * mu;
        auto b = Eigen::Matrix2d{};  // b(i, j) = p_ji mu_j / e_i
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                b(i, j) = p(j, i) * mu(j) / e(i);
            }
        }
        auto smoothed = std::array<HandMode, 2>{};
        auto probabilities = Eigen::Vector2d{};
        for (int j = 0; j < 2; ++j)
        {
            auto mixing = Eigen::Vector2d{};
            for (int i = 0; i < 2; ++i)
            {
                mixing(i) = b(i, j) * next[std::size_t(i)].probability;
            }
            mixing /= mixing.sum();
            Vector4 m0 = Vector4::Zero();
            for (int i = 0; i < 2; ++i)
            {
                m0 += mixing(i) * next[std::size_t(i)].mean;
            }
            Matrix4 p0 = Matrix4::Zero();
            for (int i = 0; i < 2; ++i)
            {
                Vector4 const d = next[std::size_t(i)].mean - m0;
                p0 += mixing(i) * (next[std::size_t(i)].cov + d * d.transpose());
            }
            auto const & filtered = joint.joints[std::size_t(j)].filtered;
            Vector4 const predictedMean = f * filtered.mean();
            Matrix4 const predictedCov = f * filtered.covarianceLower() * f.transpose() + q[std::size_t(j)];
            Matrix4 const gain = filtered.covarianceLower() * f.transpose() * predictedCov.inverse();
            smoothed[std::size_t(j)].mean = filtered.mean() + gain * (m0 - predictedMean);
            smoothed[std::size_t(j)].cov = filtered.covarianceLower() + gain * (p0 - predictedCov) * gain.transpose();
        }
        probabilities.setZero();
        for (int i = 0; i < 2; ++i)
        {
            auto given = Eigen::Vector2d{};
            for (int j = 0; j < 2; ++j)
            {
                auto const & filtered = joint.joints[std::size_t(j)].filtered;
                Vector4 const mean = f * filtered.mean();
                Matrix4 const cov = f * filtered.covarianceLower() * f.transpose() + q[std::size_t(i)];
                given(j) = b(i, j) * normalDensity(next[std::size_t(i)].mean, mean, cov);
            }
            probabilities += next[std::size_t(i)].probability * given / given.sum();
        }
        for (int j = 0; j < 2; ++j)
        {
            smoothed[std::size_t(j)].probability = probabilities(j);
        }
        expected[t] = smoothed;
        next = smoothed;
    }

    std::size_t t = 0;
    for (auto const & recorded : history)
    {
        for (Eigen::Index j = 0; j < 2; ++j)
        {
            HandMode const & hand = expected[t][std::size_t(j)];
            EXPECT_TRUE(recorded.smoothed.conditional(j).mean().isApprox(hand.mean, 1e-10)) << t << " " << j;
            EXPECT_TRUE(recorded.smoothed.conditional(j).covarianceLower().isApprox(hand.cov, 1e-10)) << t << " " << j;
            EXPECT_NEAR(recorded.smoothed.modes().toDensity().value().probability(j), hand.probability, 1e-12)
                << t << " " << j;
        }
        ++t;
    }
}

// With the identity chain, an autonomous multiple model filter, the mode is fixed for all time, so
// its exact smoothed probability at every step is the final filtered one; the backward
// transition is the identity there, and the smoother keeps it to rounding.
TEST(Smoothing, ImmSmootherKeepsAFixedModesProbability)
{
    auto filter = ImmFilter{
        SwitchingModel{
            MarkovChain<Maneuver>{Eigen::Matrix2d::Identity()},
            ConstantVelocity<Position, Velocity>{quiet},
            ConstantVelocity<Position, Velocity>{agile}
        },
        HybridParams{CategoricalParams<Maneuver>{}, makeInitial()}
    };
    auto history = HybridGaussianHistory<State, Maneuver>{};
    runImm(filter, history);
    history.smooth(filter.state());
    double const last = filter.state().modes().toDensity().value().probability<Maneuvering>();
    for (auto const & step : history)
    {
        EXPECT_NEAR(step.smoothed.modes().toDensity().value().probability<Maneuvering>(), last, 1e-12);
    }
}

namespace
{

struct Turning
{};

struct ThreeManeuvers : Named<Cruise, Maneuvering, Turning>
{};

}  // namespace

// Under a chain that never stays in a mode, b_ij is zero for j = i, and mode j's Gaussian moved
// through every other mode's model is degenerate here, so no mode's prediction gives a smoothed
// mean a density: the mode probabilities are left at sum_i mu_{t+1|N}^i b_ij, unweighted. The
// same step with its predictions intact comes out elsewhere.
TEST(Smoothing, ImmSmootherLeavesTheBackwardTransitionWhereNoPredictionHasADensity)
{
    Eigen::Matrix3d const p{{0.0, 0.5, 0.5}, {0.6, 0.0, 0.4}, {0.3, 0.7, 0.0}};
    auto filter = ImmFilter{
        SwitchingModel{
            MarkovChain<ThreeManeuvers>{p},
            ConstantVelocity<Position, Velocity>{quiet},
            ConstantVelocity<Position, Velocity>{agile},
            ConstantVelocity<Position, Velocity>{1.0}
        },
        HybridParams{
            CategoricalParams<ThreeManeuvers>::fromWeights(ThreeManeuvers{}, Eigen::Vector3d{0.5, 0.3, 0.2}),
            makeInitial()
        }
    };
    // A step before the one recorded, so the modes' Gaussians differ and the densities matter.
    EXPECT_TRUE(filter.update(makeFix(fixAt(0))));
    filter.predict(timeStep);
    EXPECT_TRUE(filter.update(makeFix(fixAt(1))));
    auto const joint = filter.predictJoint(timeStep);
    auto degenerate = joint;
    for (auto & prediction : degenerate.throughOtherModels)
    {
        prediction.setCovariance(Matrix4::Zero());
    }
    auto history = HybridGaussianHistory<State, ThreeManeuvers>{};
    auto intact = HybridGaussianHistory<State, ThreeManeuvers>{};
    history.push(degenerate);
    intact.push(joint);
    EXPECT_TRUE(filter.update(makeFix(fixAt(2))));
    history.smooth(filter.state());
    intact.smooth(filter.state());

    Eigen::Vector3d const mu = joint.modes.toDensity().value().probabilities().matrix();
    Eigen::Vector3d const next = filter.state().modes().toDensity().value().probabilities().matrix();
    Eigen::Vector3d const e = p.transpose() * mu;
    auto expected = Eigen::Vector3d{Eigen::Vector3d::Zero()};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            expected(j) += next(i) * p(j, i) * mu(j) / e(i);
        }
    }
    Eigen::Vector3d const smoothed = history.front().smoothed.modes().toDensity().value().probabilities().matrix();
    Eigen::Vector3d const weighed = intact.front().smoothed.modes().toDensity().value().probabilities().matrix();
    EXPECT_LT((smoothed - expected).cwiseAbs().maxCoeff(), 1e-14);
    EXPECT_GT((weighed - expected).cwiseAbs().maxCoeff(), 1e-3);
}

// The smoothed mode probabilities against the exact posterior: every sequence of the mode, r_k the
// model of the step into k, over the nine fixes of runImm(), each with its own Kalman filter for
// the likelihood of the fixes, written out in plain Eigen. The expectation correction is
// approximate; a run's largest error is 0.024, at step 7, where the filtered probability the pass
// starts from is itself 0.010 from exact. Nadarajah's Lambda gave 0.054, and Kim's 0.10.
TEST(Smoothing, ImmSmootherModeProbabilitiesTrackEnumeration)
{
    auto filter = makeImm();
    auto history = HybridGaussianHistory<State, Maneuver>{};
    runImm(filter, history);
    history.smooth(filter.state());

    Matrix4 const f = transitionMatrix();
    auto const q = std::array<Matrix4, 2>{
        Emrys::FixedStepModel{makeModel(quiet), State{}, timeStep}.noise(),
        Emrys::FixedStepModel{makeModel(agile), State{}, timeStep}.noise()
    };
    Eigen::Matrix2d const p = chain.matrix();
    auto const start = Eigen::Vector2d{0.7, 0.3};
    constexpr int modes = stepCount + 1;
    auto const update = [&](Vector4 & mean, Matrix4 & cov, Eigen::Vector2d const & z) {
        Eigen::Vector2d const r = z - H * mean;
        Eigen::Matrix2d const s = H * cov * H.transpose() + R;
        Eigen::Matrix<double, 4, 2> const gain = cov * H.transpose() * s.inverse();
        mean += gain * r;
        cov = (Matrix4::Identity() - gain * H) * cov;
        return -0.5 * (r.dot(s.inverse() * r) + std::log(s.determinant()) + 2 * std::log(2 * std::numbers::pi));
    };
    auto logWeights = std::vector<double>{};
    for (int sequence = 0; sequence < (1 << modes); ++sequence)
    {
        auto const mode = [&](int k) {
            return (sequence >> k) & 1;
        };
        double logWeight = std::log(start(mode(0)));
        Vector4 mean = makeInitial().mean();
        Matrix4 cov = makeInitial().covarianceLower();
        logWeight += update(mean, cov, fixAt(0));
        for (int k = 1; k < modes; ++k)
        {
            logWeight += std::log(p(mode(k - 1), mode(k)));
            mean = f * mean;
            cov = f * cov * f.transpose() + q[std::size_t(mode(k))];
            logWeight += update(mean, cov, fixAt(k));
        }
        logWeights.push_back(logWeight);
    }
    double const maxLog = *std::ranges::max_element(logWeights);
    auto exact = std::vector<double>(std::size_t(modes), 0.0);
    double total = 0;
    for (int sequence = 0; sequence < (1 << modes); ++sequence)
    {
        double const weight = std::exp(logWeights[std::size_t(sequence)] - maxLog);
        total += weight;
        for (int k = 0; k < modes; ++k)
        {
            exact[std::size_t(k)] += ((sequence >> k) & 1) * weight;
        }
    }

    // The history's step k holds the smoothed state at time k, before the step's prediction; the
    // filter's state is the last.
    auto smoothed = std::vector<double>{};
    for (auto const & step : history)
    {
        smoothed.push_back(step.smoothed.modes().toDensity().value().probability<Maneuvering>());
    }
    smoothed.push_back(filter.state().modes().toDensity().value().probability<Maneuvering>());
    ASSERT_EQ(int(smoothed.size()), modes);
    double largest = 0;
    for (int k = 0; k < modes; ++k)
    {
        largest = std::max(largest, std::abs(smoothed[std::size_t(k)] - exact[std::size_t(k)] / total));
    }
    EXPECT_LT(largest, 0.03);
}

// On a target that cruises and then turns sharply at step 20, the smoothed probability of the
// maneuvering mode rises at the turn, where the filtered one rises only once later fixes show it.
// The fixes carry a deterministic error, the same on every compiler. The thresholds come from a
// run: the smoothed probability rises over steps 17 to 22 (0.70 at 19, 0.998 at 22) and is under
// 0.01 from 23 to 39, and the filtered one peaks at 0.52 at step 22.
TEST(Smoothing, ImmSmootherPlacesAManeuverWhereTheFilterLags)
{
    constexpr int turn = 20;
    constexpr int steps = 40;
    auto const truthAt = [&](int k) {
        auto const cruise = Eigen::Vector2d{1.0, 0.0};
        auto const turned = Eigen::Vector2d{0.0, 1.5};
        return k <= turn ? Eigen::Vector2d{cruise * k} : Eigen::Vector2d{cruise * turn + turned * (k - turn)};
    };
    auto const fix = [&](int k) {
        auto const error = Eigen::Vector2d{0.3 * std::sin(1.7 * k), 0.3 * std::cos(2.3 * k)};
        return makeFix(truthAt(k) + error);
    };
    auto filter = ImmFilter{
        SwitchingModel{MarkovChain<Maneuver>::persistent(0.95), makeModel(quiet), makeModel(agile)},
        HybridParams{
            CategoricalParams<Maneuver>{},
            GaussianParams<State>{State{}, Vector4{0.0, 0.0, 1.0, 0.0}, Matrix4{Matrix4::Identity()}}
        }
    };
    auto history = HybridGaussianHistory<State, Maneuver>{};
    auto filteredManeuver = std::vector<double>{};
    EXPECT_TRUE(filter.update(fix(0)));
    auto joint = filter.predictJoint(timeStep);
    for (int k = 1; k <= steps; ++k)
    {
        if (k > 1)
        {
            filter.predictJoint(timeStep, joint);
        }
        history.push(joint);
        EXPECT_TRUE(filter.update(fix(k)));
        filteredManeuver.push_back(filter.state().modes().toDensity().value().probability<Maneuvering>());
    }
    history.smooth(filter.state());

    // filteredManeuver[k - 1] and the history's step k are both at time k, the step's state
    // before its prediction being at time k - 1.
    auto smoothedManeuver = std::vector<double>{};
    for (auto const & step : history)
    {
        smoothedManeuver.push_back(step.smoothed.modes().toDensity().value().probability<Maneuvering>());
    }
    smoothedManeuver.erase(smoothedManeuver.begin());
    smoothedManeuver.push_back(filter.state().modes().toDensity().value().probability<Maneuvering>());

    auto const firstAbove = [](std::vector<double> const & probabilities) {
        for (std::size_t k = 0; k < probabilities.size(); ++k)
        {
            if (probabilities[k] > 0.5)
            {
                return int(k) + 1;
            }
        }
        return 0;
    };
    int const filteredAt = firstAbove(filteredManeuver);
    int const smoothedAt = firstAbove(smoothedManeuver);
    // The smoothed probability crosses one half within a step of the corner (at 19 here), the
    // filtered one at least two steps after the smoothed one (at 22 here).
    EXPECT_GE(smoothedAt, turn - 1);
    EXPECT_LE(smoothedAt, turn + 1);
    EXPECT_GE(filteredAt, smoothedAt + 2);
    // Away from the turn the smoother is surer of cruising than the filter.
    for (int k : {5, 10, turn - 5, turn + 6, turn + 10})
    {
        EXPECT_LT(smoothedManeuver[std::size_t(k - 1)], 0.05) << "step " << k;
        EXPECT_LT(smoothedManeuver[std::size_t(k - 1)], filteredManeuver[std::size_t(k - 1)]) << "step " << k;
    }
}

// A fixed-lag IMM history of four steps gives the fixed-interval pass's estimates at the same
// steps exactly, since the backward step at t reads only the steps after it.
TEST(Smoothing, ImmFixedLagMatchesFixedIntervalOverTheWindow)
{
    auto fullFilter = makeImm();
    auto lagFilter = makeImm();
    auto full = HybridGaussianHistory<State, Maneuver>{};
    auto window = HybridGaussianHistory<State, Maneuver>{Lag{4}};
    runImm(fullFilter, full);
    runImm(lagFilter, window);
    full.smooth(fullFilter.state());
    window.smooth(lagFilter.state());
    ASSERT_EQ(window.size(), 4);
    auto expected = std::next(full.begin(), stepCount - 4);
    for (auto const & step : window)
    {
        for (Eigen::Index j = 0; j < 2; ++j)
        {
            EXPECT_EQ(step.smoothed.conditional(j).mean(), expected->smoothed.conditional(j).mean());
            EXPECT_EQ(
                step.smoothed.conditional(j).covarianceLower(),
                expected->smoothed.conditional(j).covarianceLower()
            );
        }
        EXPECT_EQ(
            step.smoothed.modes().toDensity().value().logProbabilities().matrix(),
            expected->smoothed.modes().toDensity().value().logProbabilities().matrix()
        );
        ++expected;
    }
}

namespace
{

using Emrys::ArrivalInterval;
using Emrys::CategoricalHistory;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::EmissionTable;
using Emrys::Finite;
using Emrys::HiddenMarkovFilter;

using Hidden = Finite<3>;
using Seen = Finite<2>;

Eigen::Matrix3d const hiddenTransitions{{0.8, 0.15, 0.05}, {0.1, 0.6, 0.3}, {0.25, 0.25, 0.5}};
Eigen::Matrix<double, 3, 2> const hiddenEmissions{{0.9, 0.1}, {0.4, 0.6}, {0.15, 0.85}};
Eigen::Vector3d const hiddenInitial{0.5, 0.3, 0.2};

// The observations at each of five steps: one at most steps, two at step 2, and none at step 3, so
// that the history records predictions with any number of updates between them.
std::array<std::vector<Eigen::Index>, 5> const hiddenObservations{{{0}, {1}, {1, 0}, {}, {1}}};

// The joint probability of the values x and the observations, the product of the initial
// probability, the transitions, and every observation's emission probability.
double
hiddenJoint(std::array<Eigen::Index, 5> const & x)
{
    double p = hiddenInitial(x[0]);
    for (std::size_t k = 0; k < 5; ++k)
    {
        if (k > 0)
        {
            p *= hiddenTransitions(x[k - 1], x[k]);
        }
        for (Eigen::Index const z : hiddenObservations[k])
        {
            p *= hiddenEmissions(x[k], z);
        }
    }
    return p;
}

// Every sequence of five values in Hidden, the first varying slowest.
std::vector<std::array<Eigen::Index, 5>>
everyHiddenSequence()
{
    auto sequences = std::vector<std::array<Eigen::Index, 5>>{};
    for (int code = 0; code < 243; ++code)
    {
        auto x = std::array<Eigen::Index, 5>{};
        int rest = code;
        for (std::size_t k = 5; k > 0; --k)
        {
            x[k - 1] = rest % 3;
            rest /= 3;
        }
        sequences.push_back(x);
    }
    return sequences;
}

}  // namespace

// Over the five steps, the forward pass, the smoother, the pairwise transition posteriors, and
// Viterbi all match brute-force enumeration of the 243 value sequences: the log-evidence is the
// log of the total joint probability, each smoothed distribution is a marginal of the normalized
// joint, each transition posterior a pairwise marginal, and the most probable sequence is the
// joint's argmax with its normalized probability. The tolerances are rounding over a few dozen
// products and sums.
TEST(Smoothing, HiddenMarkovSmootherMatchesEnumeration)
{
    auto const hiddenChain = MarkovChain<Hidden>{hiddenTransitions};
    auto const table = EmissionTable{Hidden{}, Seen{}, hiddenEmissions};
    auto filter = HiddenMarkovFilter{hiddenChain, CategoricalParams<Hidden>::fromWeights(Hidden{}, hiddenInitial)};
    auto history = CategoricalHistory<Hidden>{};
    for (std::size_t k = 0; k < 5; ++k)
    {
        if (k > 0)
        {
            history.push(filter.predictJoint());
        }
        for (Eigen::Index const z : hiddenObservations[k])
        {
            ASSERT_TRUE(filter.update(table.observe(z)));
        }
    }
    history.smooth(filter.state());
    auto const path = mostProbableSequence(history, filter.state());

    double total = 0;
    double best = 0;
    auto argmax = std::array<Eigen::Index, 5>{};
    auto marginals = std::array<Eigen::Vector3d, 5>{};
    marginals.fill(Eigen::Vector3d::Zero());
    auto pairs = std::array<Eigen::Matrix3d, 4>{};
    pairs.fill(Eigen::Matrix3d::Zero());
    for (auto const & x : everyHiddenSequence())
    {
        double const p = hiddenJoint(x);
        total += p;
        if (p > best)
        {
            best = p;
            argmax = x;
        }
        for (std::size_t k = 0; k < 5; ++k)
        {
            marginals[k](x[k]) += p;
            if (k < 4)
            {
                pairs[k](x[k], x[k + 1]) += p;
            }
        }
    }

    EXPECT_NEAR(filter.logEvidence(), std::log(total), 1e-13);
    ASSERT_EQ(history.size(), 4);
    std::size_t k = 0;
    for (auto step = history.begin(); step != history.end(); ++step, ++k)
    {
        EXPECT_TRUE(step->smoothed.logWeights().exp().matrix().isApprox(marginals[k] / total, 1e-13)) << "step " << k;
        auto const & next = std::next(step) == history.end() ? filter.state() : std::next(step)->smoothed;
        EXPECT_TRUE(transitionPosterior(step->joint, next).isApprox(pairs[k] / total, 1e-13)) << "step " << k;
    }
    EXPECT_TRUE(filter.state().toDensity().value().probabilities().matrix().isApprox(marginals[4] / total, 1e-13));

    ASSERT_EQ(path.values.size(), 5U);
    for (std::size_t s = 0; s < 5; ++s)
    {
        EXPECT_EQ(path.values[s], argmax[s]) << "step " << s;
    }
    EXPECT_NEAR(path.logProbability, std::log(best / total), 1e-13);
}

// With an identity emission table every value is observed, so the smoothed distributions are
// certain of the true values, Viterbi returns them, and their sequence has probability one.
TEST(Smoothing, HiddenMarkovSmootherWithIdentityEmissionsIsCertain)
{
    using Space = Finite<4>;
    auto const fourChain = MarkovChain<Space>::persistent(0.6);
    auto const identity = EmissionTable{Space{}, Space{}, Eigen::Matrix4d::Identity()};
    auto filter = HiddenMarkovFilter{fourChain};
    auto history = CategoricalHistory<Space>{};
    auto const truth = std::array<Eigen::Index, 8>{0, 0, 2, 2, 2, 1, 3, 3};
    for (std::size_t k = 0; k < truth.size(); ++k)
    {
        if (k > 0)
        {
            history.push(filter.predictJoint());
        }
        ASSERT_TRUE(filter.update(identity.observe(truth[k])));
    }
    history.smooth(filter.state());
    std::size_t k = 0;
    for (auto const & step : history)
    {
        EXPECT_DOUBLE_EQ(step.smoothed.toDensity().value().probability(truth[k++]), 1.0);
    }
    auto const path = mostProbableSequence(history, filter.state());
    ASSERT_EQ(path.values.size(), truth.size());
    for (std::size_t s = 0; s < truth.size(); ++s)
    {
        EXPECT_EQ(path.values[s], truth[s]);
    }
    EXPECT_NEAR(path.logProbability, 0.0, 1e-15);
}

// A value the chain never enters, and which the run doesn't start in, has probability zero at every
// step, smoothed too, and the Viterbi path never passes through it.
TEST(Smoothing, AValueNeverEnteredIsNeverOnThePath)
{
    auto const closed = Eigen::Matrix3d{{0.7, 0.3, 0.0}, {0.4, 0.6, 0.0}, {0.5, 0.5, 0.0}};
    auto const table = EmissionTable{Hidden{}, Seen{}, hiddenEmissions};
    auto filter = HiddenMarkovFilter{
        MarkovChain<Hidden>{closed},
        CategoricalParams<Hidden>::fromWeights(Hidden{}, Eigen::Vector3d{0.5, 0.5, 0.0})
    };
    auto history = CategoricalHistory<Hidden>{};
    auto const observations = std::array<Eigen::Index, 4>{1, 1, 0, 1};
    for (std::size_t k = 0; k < observations.size(); ++k)
    {
        if (k > 0)
        {
            history.push(filter.predictJoint());
        }
        ASSERT_TRUE(filter.update(table.observe(observations[k])));
    }
    history.smooth(filter.state());
    for (auto const & step : history)
    {
        EXPECT_EQ(step.smoothed.toDensity().value().probability(2), 0.0);
    }
    auto const path = mostProbableSequence(history, filter.state());
    ASSERT_EQ(path.values.size(), observations.size());
    for (Eigen::Index const value : path.values)
    {
        EXPECT_NE(value, 2);
    }
    EXPECT_TRUE(std::isfinite(path.logProbability));
}

// A three-step fixed-lag window ends with the same steps as an unbounded history over the same
// run, smoothed the same, so their estimates are bit-identical: the window's backward pass starts
// from the same filtered state and reads the same joints. Its Viterbi path over the window is the
// most probable one of those values alone, with the earlier ones summed out, so it need not be the
// tail of the full path; the test checks only that it has the window's length.
TEST(Smoothing, HiddenMarkovFixedLagMatchesFixedIntervalOverTheWindow)
{
    auto const hiddenChain = MarkovChain<Hidden>{hiddenTransitions};
    auto const table = EmissionTable{Hidden{}, Seen{}, hiddenEmissions};
    auto filter = HiddenMarkovFilter{hiddenChain, CategoricalParams<Hidden>::fromWeights(Hidden{}, hiddenInitial)};
    auto full = CategoricalHistory<Hidden>{};
    auto window = CategoricalHistory<Hidden>{Lag{3}};
    auto const observations = std::array<Eigen::Index, 10>{0, 1, 1, 0, 0, 1, 0, 1, 1, 1};
    for (std::size_t k = 0; k < observations.size(); ++k)
    {
        if (k > 0)
        {
            auto joint = filter.predictJoint();
            full.push(joint);
            window.push(std::move(joint));
        }
        ASSERT_TRUE(filter.update(table.observe(observations[k])));
    }
    full.smooth(filter.state());
    window.smooth(filter.state());
    ASSERT_EQ(window.size(), 3);
    auto fullStep = std::prev(full.end(), 3);
    for (auto const & step : window)
    {
        EXPECT_TRUE((step.smoothed.logWeights() == fullStep->smoothed.logWeights()).all());
        ++fullStep;
    }
    EXPECT_EQ(mostProbableSequence(window, filter.state()).values.size(), 4U);
}

// The Markov-modulated Poisson process smoothed over a record of arrivals: the joints from
// predictJoint(dt, interval) carry the step e^{(Q - Λ)τ} Λ divided by its total, so the smoother
// and Viterbi over them match enumeration of the chain's value at each of the ten epochs (the
// start, eight arrivals, and the end of the record), whose joint probability is
// π(x_0) Π_k M_k(x_k, x_{k+1}) with M_k the unnormalized step, evaluated with plain Eigen.
TEST(Smoothing, ModulatedPoissonSmootherMatchesEnumeration)
{
    struct Quiet
    {};

    struct Busy
    {};

    struct Regime : Emrys::Named<Quiet, Busy>
    {};

    auto const generator = Eigen::Matrix2d{{-0.4, 0.4}, {1.1, -1.1}};
    auto const rates = Eigen::Array2d{0.5, 6.0};
    auto const gaps = std::array<double, 9>{0.37, 1.12, 0.05, 2.4, 0.81, 0.3, 1.9, 0.14, 0.63};
    auto const regimeChain = ContinuousTimeMarkovChain<Regime>{generator};
    auto filter =
        HiddenMarkovFilter{regimeChain, CategoricalParams<Regime>::fromWeights(Regime{}, Eigen::Vector2d{0.7, 0.3})};
    auto history = CategoricalHistory<Regime>{};
    auto steps = std::array<Eigen::Matrix2d, 9>{};
    for (std::size_t k = 0; k < gaps.size(); ++k)
    {
        bool const arrival = k + 1 < gaps.size();
        auto joint = filter.predictJoint(gaps[k], ArrivalInterval<Regime>{rates, arrival});
        ASSERT_TRUE(joint);
        history.push(std::move(*joint));
        steps[k] = (Eigen::Matrix2d{generator - Eigen::Matrix2d{rates.matrix().asDiagonal()}} * gaps[k]).exp();
        if (arrival)
        {
            steps[k] = steps[k] * rates.matrix().asDiagonal();
        }
    }
    history.smooth(filter.state());
    auto const path = mostProbableSequence(history, filter.state());

    double total = 0;
    double best = 0;
    int argmax = 0;
    auto marginals = std::array<Eigen::Vector2d, 10>{};
    marginals.fill(Eigen::Vector2d::Zero());
    for (int code = 0; code < 1024; ++code)
    {
        auto const value = [code](std::size_t k) {
            return Eigen::Index((code >> (9 - k)) & 1);
        };
        double p = value(0) == 0 ? 0.7 : 0.3;
        for (std::size_t k = 0; k < 9; ++k)
        {
            p *= steps[k](value(k), value(k + 1));
        }
        total += p;
        if (p > best)
        {
            best = p;
            argmax = code;
        }
        for (std::size_t k = 0; k < 10; ++k)
        {
            marginals[k](value(k)) += p;
        }
    }

    EXPECT_NEAR(filter.logEvidence(), std::log(total), 1e-11);
    std::size_t k = 0;
    for (auto const & step : history)
    {
        EXPECT_TRUE(step.smoothed.logWeights().exp().matrix().isApprox(marginals[k] / total, 1e-11)) << "epoch " << k;
        ++k;
    }
    EXPECT_TRUE(filter.state().toDensity().value().probabilities().matrix().isApprox(marginals[9] / total, 1e-11));
    ASSERT_EQ(path.values.size(), 10U);
    for (std::size_t s = 0; s < 10; ++s)
    {
        EXPECT_EQ(path.values[s], (argmax >> (9 - s)) & 1) << "epoch " << s;
    }
    EXPECT_NEAR(path.logProbability, std::log(best / total), 1e-11);
}

// A history with no steps has nothing to smooth, and the most probable sequence of no steps is the
// latest state's most probable value, with its log-probability.
TEST(Smoothing, AnEmptyHistoryHasNothingToSmooth)
{
    auto particles = ParticleHistory<State>{};
    particles.smooth(ParticleSet<State>{State{}, Matrix4::Identity()});
    EXPECT_TRUE(particles.empty());
    auto hybrid = HybridGaussianHistory<State, Maneuver>{};
    hybrid.smooth(makeImm().state());
    EXPECT_TRUE(hybrid.empty());

    auto const latest = CategoricalParams<Hidden>::fromWeights(Hidden{}, Eigen::Vector3d{0.2, 0.5, 0.3});
    auto const path = mostProbableSequence(CategoricalHistory<Hidden>{}, latest);
    ASSERT_EQ(path.values.size(), 1U);
    EXPECT_EQ(path.values[0], 1);
    EXPECT_NEAR(path.logProbability, std::log(0.5), 1e-15);
}

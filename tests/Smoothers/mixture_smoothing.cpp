// The Gaussian mixture smoother (step 14; [smoothing]): MixtureFilter::predictJoint() records each
// component's joint, the weights, and the lineage, and starts the lineage over; Keep gives each
// branch its prior's lineage and a merge sums the shares by weight. The smoother over a
// GaussianMixtureHistory is the Kalman smoother per component under Collapse, matches the exact
// Gaussian-sum smoother enumerated branch by branch under Keep, follows the normalized shares of
// Rahmathullah, Svensson, and Svensson (2014), eq. (15), through a merge, agrees between fixed-lag
// and fixed-interval, and refuses a record whose lineage doesn't follow the one before. Headers:
// Smoothers/GaussianMixtureHistory.h, Filters/MixtureFilter.h, Distributions/Mixture.h. The lineage of the mixture
// itself is in Distributions/gaussian_mixture.cpp, the allocation case in allocation_mixtures.cpp,
// and the parallel case in parallel.cpp.

#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Smoothers/GaussianMixtureHistory.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::Gaussian;
using Emrys::GaussianHistory;
using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureHistory;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::KalmanFilter;
using Emrys::Lag;
using Emrys::Linear;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

constexpr double accelDensity = 0.1;
constexpr double timeStep = 1.0;
double const logZero = -std::numeric_limits<double>::infinity();

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{accelDensity};
}

Matrix4
initialCov()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    return cov;
}

// Two components for where the target starts, weights 0.6 and 0.4.
GaussianMixtureParams<State>
makePrior()
{
    auto prior = GaussianMixtureParams<State>{State{}};
    prior.add(std::log(0.6), Vector4{0.0, 0.0, 1.0, 0.5}, initialCov());
    prior.add(std::log(0.4), Vector4{3.0, -2.0, 0.5, 1.0}, initialCov());
    return prior;
}

Eigen::Matrix2d const R{{1.0, 0.2}, {0.2, 0.8}};

// A position fix, z = p + v, with noise \a noise, a Gaussian or a mixture.
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

// The noise components of a fix that is an outlier one time in five: weights, offsets, and covariances.
std::array<double, 2> const noiseWeights{0.8, 0.2};

std::array<Eigen::Vector2d, 2> const noiseOffsets{Eigen::Vector2d::Zero(), Eigen::Vector2d{1.5, -1.0}};
std::array<Eigen::Matrix2d, 2> const noiseCovs{R, 25 * R};

GaussianMixtureParams<Linear<2>>
mixtureNoise()
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    for (std::size_t s = 0; s < 2; ++s)
    {
        noise.add(std::log(noiseWeights[s]), noiseOffsets[s], noiseCovs[s]);
    }
    return noise;
}

// The fix under noise component s alone.
auto
makeBranchFix(Eigen::Vector2d const & z, std::size_t s)
{
    return makeFix(z, GaussianParams<Linear<2>>{noiseOffsets[s], noiseCovs[s]});
}

// A target turning slowly, seen at step k.
Eigen::Vector2d
fixAt(int k)
{
    return Eigen::Vector2d{1.0, 0.5} * k + 0.8 * Eigen::Vector2d{std::cos(0.7 * k), std::sin(1.3 * k)};
}

// The normalized weights of \a mixture.
std::vector<double>
normalizedWeights(GaussianMixtureParams<State> const & mixture)
{
    double const logTotal = mixture.logTotalWeight().value();
    auto weights = std::vector<double>{};
    for (auto const & entry : mixture)
    {
        weights.push_back(std::exp(entry.logWeight - logTotal));
    }
    return weights;
}

// A weighted Gaussian, and the moments of several by plain moment matching.
struct Weighted
{
    double weight;
    Vector4 mean;
    Matrix4 cov;
};

Weighted
momentMatch(std::vector<Weighted> const & parts)
{
    auto result = Weighted{0.0, Vector4::Zero(), Matrix4::Zero()};
    for (Weighted const & part : parts)
    {
        result.weight += part.weight;
        result.mean += part.weight * part.mean;
    }
    result.mean /= result.weight;
    for (Weighted const & part : parts)
    {
        Vector4 const d = part.mean - result.mean;
        result.cov += part.weight * (part.cov + d * d.transpose());
    }
    result.cov /= result.weight;
    return result;
}

}  // namespace

// A history with no steps has nothing to smooth.
TEST(MixtureSmoothing, AnEmptyHistoryHasNothingToSmooth)
{
    auto history = GaussianMixtureHistory<State>{};
    auto latest = GaussianMixtureParams<State>{State{}};
    latest.add(0.0, Vector4::Zero(), Matrix4::Identity());
    history.smooth(latest);
    EXPECT_TRUE(history.empty());
}

// predictJoint() records the weights and, for each component, the KalmanFilter's joint, moves each
// component as the KalmanFilter does, and records the lineage the state had (none, before the first
// mark) before starting it over with each component its own parent.
TEST(MixtureSmoothing, PredictJointRecordsEachComponentAndStartsTheLineage)
{
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto const prior = filter.state();
    auto const joint = filter.predictJoint(timeStep);

    ASSERT_EQ(joint.joints.size(), 2u);
    ASSERT_EQ(joint.logWeights.size(), 2u);
    EXPECT_EQ(joint.lineage.count(), 0);
    EXPECT_EQ(joint.lineage.size(), 0);
    for (std::size_t n = 0; n < 2; ++n)
    {
        auto const & before = prior[Eigen::Index(n)];
        EXPECT_EQ(joint.logWeights[n], before.logWeight);
        auto single = KalmanFilter{makeModel(), before.component};
        auto const expected = single.predictJoint(timeStep);
        EXPECT_EQ(joint.joints[n].filtered.mean(), expected.filtered.mean());
        EXPECT_EQ(joint.joints[n].predicted.mean(), expected.predicted.mean());
        EXPECT_EQ(joint.joints[n].predicted.covarianceLower(), expected.predicted.covarianceLower());
        EXPECT_EQ(joint.joints[n].cross, expected.cross);
        EXPECT_EQ(filter.state()[Eigen::Index(n)].component.mean(), single.state().mean());
    }

    auto const & lineage = filter.state().lineage();
    EXPECT_EQ(lineage.count(), 1);
    ASSERT_EQ(lineage.parentCount(), 2);
    ASSERT_EQ(lineage.size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        for (Eigen::Index n = 0; n < 2; ++n)
        {
            EXPECT_EQ(lineage.logShare(i, n), i == n ? 0.0 : logZero);
        }
    }
}

// Keep gives each prior's two branches its lineage, so after one update the four components come
// from parents 0, 0, 1, 1; merging branches 1 and 2 by hand gives the merge each parent's share of
// its weight, and the next predictJoint() records that lineage.
TEST(MixtureSmoothing, KeepAndMergeCarryTheLineage)
{
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    static_cast<void>(filter.predictJoint(timeStep));
    EXPECT_TRUE(filter.update(makeFix(fixAt(1), mixtureNoise()), reductions::keep()));
    auto const & state = filter.state();
    ASSERT_EQ(state.size(), 4);
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        EXPECT_EQ(state.lineage().logShare(i, 0), i < 2 ? 0.0 : logZero) << "component " << i;
        EXPECT_EQ(state.lineage().logShare(i, 1), i < 2 ? logZero : 0.0) << "component " << i;
    }

    double const w1 = std::exp(state[1].logWeight);
    double const w2 = std::exp(state[2].logWeight);
    static_cast<void>(filter.state().merge(1, 2));
    ASSERT_EQ(state.size(), 3);
    EXPECT_NEAR(std::exp(state.lineage().logShare(1, 0)), w1 / (w1 + w2), 1e-14);
    EXPECT_NEAR(std::exp(state.lineage().logShare(1, 1)), w2 / (w1 + w2), 1e-14);
    // The last branch moved into the merged one's place, with its row.
    EXPECT_EQ(state.lineage().logShare(2, 0), logZero);
    EXPECT_EQ(state.lineage().logShare(2, 1), 0.0);

    auto const joint = filter.predictJoint(timeStep);
    EXPECT_EQ(joint.lineage.count(), 1);
    ASSERT_EQ(joint.lineage.size(), 3);
    EXPECT_NEAR(std::exp(joint.lineage.logShare(1, 0)), w1 / (w1 + w2), 1e-14);
    EXPECT_EQ(filter.state().lineage().count(), 2);
    EXPECT_EQ(filter.state().lineage().parentCount(), 3);
}

// Under Collapse each component keeps one parent, so the mixture smoother is a Kalman smoother per
// component, run here by a KalmanFilter and GaussianHistory from each prior component, and every
// step's smoothed weights are the final posterior's. The same arithmetic in another order, so the
// tolerance is rounding.
TEST(MixtureSmoothing, CollapseIsTheKalmanSmootherPerComponent)
{
    constexpr int steps = 6;
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto history = GaussianMixtureHistory<State>{};
    auto singles = std::vector<KalmanFilter<Gaussian<State>, decltype(makeModel())>>{};
    auto singleHistories = std::vector<GaussianHistory<State>>(2);
    for (auto const & entry : filter.state())
    {
        singles.emplace_back(makeModel(), entry.component);
    }
    for (int k = 1; k <= steps; ++k)
    {
        history.push(filter.predictJoint(timeStep));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse()));
        for (std::size_t n = 0; n < 2; ++n)
        {
            singleHistories[n].push(singles[n].predictJoint(timeStep));
            EXPECT_TRUE(singles[n].update(makeFix(fixAt(k))));
        }
    }
    history.smooth(filter.state());
    for (std::size_t n = 0; n < 2; ++n)
    {
        singleHistories[n].smooth(singles[n].state());
    }

    auto const latestWeights = normalizedWeights(filter.state());
    ASSERT_EQ(history.size(), steps);
    auto single =
        std::array<GaussianHistory<State>::const_iterator, 2>{singleHistories[0].begin(), singleHistories[1].begin()};
    for (auto const & step : history)
    {
        auto const weights = normalizedWeights(step.smoothed);
        ASSERT_EQ(weights.size(), 2u);
        for (std::size_t n = 0; n < 2; ++n)
        {
            auto const & smoothed = step.smoothed[Eigen::Index(n)].component;
            EXPECT_NEAR(weights[n], latestWeights[n], 1e-12);
            EXPECT_LT((smoothed.mean() - single[n]->smoothed.mean()).norm(), 1e-10);
            EXPECT_LT((smoothed.covarianceLower() - single[n]->smoothed.covarianceLower()).norm(), 1e-10);
            ++single[n];
        }
    }
}

// Under Keep with mixture noise and no merging, the smoothed component n at step k has the weight
// and moments of every exact branch through it: each sequence of a prior component and noise
// components, filtered and smoothed by its own KalmanFilter and GaussianHistory, weighted by its
// prior weight, noise weights, and likelihoods. Keep orders component i's branches i * 2 + s, so
// branch (i, s_0, ..., s_{T-1}) passes through component i * 2^k + (s_0 ... s_{k-1} in binary) at
// step k. The tolerance allows for summing 16 branches' moments by hand.
TEST(MixtureSmoothing, KeepMatchesTheExactSmootherBranchByBranch)
{
    constexpr int steps = 3;
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto history = GaussianMixtureHistory<State>{};
    for (int k = 1; k <= steps; ++k)
    {
        history.push(filter.predictJoint(timeStep));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k), mixtureNoise()), reductions::keep()));
    }
    history.smooth(filter.state());
    ASSERT_EQ(filter.state().size(), 2 << steps);

    struct Branch
    {
        double logWeight;
        int index;                                // the component the branch ends in
        std::vector<GaussianParams<State>> path;  // its smoothed Gaussian at each recorded step
    };

    auto branches = std::vector<Branch>{};
    auto const prior = makePrior();
    for (int i = 0; i < 2; ++i)
    {
        for (int sequence = 0; sequence < (1 << steps); ++sequence)
        {
            auto const & start = prior[i];
            auto single = KalmanFilter{makeModel(), start.component};
            auto singleHistory = GaussianHistory<State>{};
            double logWeight = start.logWeight;
            for (int k = 1; k <= steps; ++k)
            {
                std::size_t const s = std::size_t((sequence >> (steps - k)) & 1);
                singleHistory.push(single.predictJoint(timeStep));
                auto const update = single.update(makeBranchFix(fixAt(k), s));
                ASSERT_TRUE(update);
                logWeight += std::log(noiseWeights[s]) + update->logLikelihood();
            }
            singleHistory.smooth(single.state());
            auto branch = Branch{logWeight, i * (1 << steps) + sequence, {}};
            for (auto const & step : singleHistory)
            {
                branch.path.push_back(step.smoothed);
            }
            branches.push_back(std::move(branch));
        }
    }
    double maxLogWeight = logZero;
    for (Branch const & branch : branches)
    {
        maxLogWeight = std::max(maxLogWeight, branch.logWeight);
    }
    double total = 0;
    for (Branch const & branch : branches)
    {
        total += std::exp(branch.logWeight - maxLogWeight);
    }

    int k = 0;
    for (auto const & step : history)
    {
        int const shift = steps - k;
        auto const weights = normalizedWeights(step.smoothed);
        ASSERT_EQ(step.smoothed.size(), 2 << k);
        for (Eigen::Index n = 0; n < step.smoothed.size(); ++n)
        {
            auto through = std::vector<Weighted>{};
            for (Branch const & branch : branches)
            {
                if ((branch.index >> shift) == n)
                {
                    auto const & g = branch.path[std::size_t(k)];
                    through.push_back(
                        {std::exp(branch.logWeight - maxLogWeight) / total, g.mean(), g.covarianceLower()}
                    );
                }
            }
            Weighted const exact = momentMatch(through);
            auto const & smoothed = step.smoothed[n].component;
            EXPECT_NEAR(weights[std::size_t(n)], exact.weight, 1e-12) << "step " << k << ", component " << n;
            EXPECT_LT((smoothed.mean() - exact.mean).norm(), 1e-9) << "step " << k << ", component " << n;
            EXPECT_LT((smoothed.covarianceLower() - exact.cov).norm(), 1e-9) << "step " << k << ", component " << n;
        }
        ++k;
    }
}

// Through a merge, each pair's weight is the smoothed child's times its normalized share from the
// parent, eq. (15) of Rahmathullah, Svensson, and Svensson (2014), and the pairs that share a parent
// are moment-matched: one step from two components, updated under Keep into four, of which branches
// 1 (parent 0) and 2 (parent 1) are merged by hand. The expected values are that transcription, with
// smoothStep() for each pair, which the Kalman smoother's tests cover. Their eq. (16) as printed,
// without the division by the merged weight, would weight the merge's pairs by w1 and w2 rather
// than by its smoothed weight times w1/(w1 + w2) and w2/(w1 + w2), and the parents' weights would
// not sum to one.
TEST(MixtureSmoothing, MergedComponentsSplitByTheirShares)
{
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto history = GaussianMixtureHistory<State>{};
    history.push(filter.predictJoint(timeStep));
    EXPECT_TRUE(filter.update(makeFix(fixAt(1), mixtureNoise()), reductions::keep()));
    auto & state = filter.state();
    static_cast<void>(state.normalize());
    double const w1 = std::exp(state[1].logWeight);
    double const w2 = std::exp(state[2].logWeight);
    static_cast<void>(state.merge(1, 2));
    history.smooth(filter.state());

    // Children: 0 from parent 0; 1, the merge, from both; 2, the last branch moved into place, from 1.
    auto const & joint = history.front().joint;
    auto const child = [&](Eigen::Index p) -> auto const & {
        return filter.state()[p];
    };
    double const a = w1 / (w1 + w2);

    struct Pair
    {
        Eigen::Index child;
        Eigen::Index parent;
        double share;
    };

    auto const pairs = std::array<Pair, 4>{{{0, 0, 1.0}, {1, 0, a}, {1, 1, 1 - a}, {2, 1, 1.0}}};
    auto groups = std::array<std::vector<Weighted>, 2>{};
    for (Pair const & pair : pairs)
    {
        auto const g = Emrys::smoothStep(joint.joints[std::size_t(pair.parent)], child(pair.child).component);
        groups[std::size_t(pair.parent)].push_back(
            {std::exp(child(pair.child).logWeight) * pair.share, g.mean(), g.covarianceLower()}
        );
    }
    auto const weights = normalizedWeights(history.front().smoothed);
    ASSERT_EQ(weights.size(), 2u);
    for (std::size_t n = 0; n < 2; ++n)
    {
        Weighted const expected = momentMatch(groups[n]);
        auto const & smoothed = history.front().smoothed[Eigen::Index(n)].component;
        EXPECT_NEAR(weights[n], expected.weight, 1e-12) << "parent " << n;
        EXPECT_LT((smoothed.mean() - expected.mean).norm(), 1e-12) << "parent " << n;
        EXPECT_LT((smoothed.covarianceLower() - expected.cov).norm(), 1e-12) << "parent " << n;
    }
    EXPECT_NEAR(weights[0] + weights[1], 1.0, 1e-14);
}

// Under Keep branches 2 and 3 descend from parent 1; with no weight, as if pruned, nothing at the
// end descends from it, so it keeps its filtered Gaussian with no weight, and parent 0 has it all.
TEST(MixtureSmoothing, AComponentNothingDescendsFromKeepsItsFilteredGaussian)
{
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto history = GaussianMixtureHistory<State>{};
    history.push(filter.predictJoint(timeStep));
    EXPECT_TRUE(filter.update(makeFix(fixAt(1), mixtureNoise()), reductions::keep()));
    auto & state = filter.state();
    ASSERT_EQ(state.size(), 4);
    state.setLogWeight(2, logZero);
    state.setLogWeight(3, logZero);
    history.smooth(state);

    auto const & step = history.front();
    auto const & orphan = step.smoothed[1];
    auto const & filtered = step.joint.joints[1].filtered;
    EXPECT_EQ(orphan.logWeight, logZero);
    EXPECT_EQ(orphan.component.mean(), filtered.mean());
    EXPECT_EQ(orphan.component.covarianceLower(), filtered.covarianceLower());
    EXPECT_NEAR(normalizedWeights(step.smoothed)[0], 1.0, 1e-15);
}

// A three-step fixed-lag window over a run with Keep and merges keeps the newest three steps,
// and its smoothed mixtures are the fixed-interval pass's there, since the backward step at k reads
// only the steps after it; it is the same arithmetic, so they agree exactly.
TEST(MixtureSmoothing, FixedLagMatchesFixedIntervalOverTheWindow)
{
    constexpr int steps = 8;
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto all = GaussianMixtureHistory<State>{};
    auto window = GaussianMixtureHistory<State>{Lag{3}};
    for (int k = 1; k <= steps; ++k)
    {
        auto joint = filter.predictJoint(timeStep);
        all.push(joint);
        window.push(std::move(joint));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k), mixtureNoise()), reductions::keep()));
        auto & state = filter.state();
        while (state.size() > 3)
        {
            static_cast<void>(state.merge(0, 1));
        }
    }
    all.smooth(filter.state());
    window.smooth(filter.state());
    ASSERT_EQ(all.size(), steps);
    ASSERT_EQ(window.size(), 3);
    auto expected = std::next(all.begin(), steps - 3);
    for (auto const & step : window)
    {
        ASSERT_EQ(step.smoothed.size(), expected->smoothed.size());
        for (Eigen::Index n = 0; n < step.smoothed.size(); ++n)
        {
            auto const & got = step.smoothed[n];
            auto const & want = expected->smoothed[n];
            EXPECT_EQ(got.logWeight, want.logWeight);
            EXPECT_EQ(got.component.mean(), want.component.mean());
            EXPECT_EQ(got.component.covarianceLower(), want.component.covarianceLower());
        }
        ++expected;
    }
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
// A state replaced by a copy has no lineage, so the next record does not follow the one before,
// and the history refuses it; so does smoothing to such a state.
TEST(MixtureSmoothingDeathTest, HistoryRefusesALineageThatDoesNotFollow)
{
    auto filter = GaussianMixtureFilter{makeModel(), makePrior()};
    auto history = GaussianMixtureHistory<State>{};
    history.push(filter.predictJoint(timeStep));
    history.push(filter.predictJoint(timeStep));
    GaussianMixtureParams<State> const copy = filter.state();
    filter.state() = copy;
    EXPECT_DEATH(history.smooth(filter.state()), "the lineage must start");
    EXPECT_DEATH(history.push(filter.predictJoint(timeStep)), "the lineage must start");
}
#endif

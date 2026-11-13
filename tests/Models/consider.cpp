// The consider, or Schmidt-Kalman, update: consider() of Models/Policies.h, an argument of observe()
// (Models/MeasurementModel.h), carried out by kalmanConditionInPlace() of Models/Condition.h and by the
// iterated update of Models/IteratedUpdate.h (roadmap step 17v; decisions.md [consider-states]). The
// considered block keeps the prior's mean and covariance block, and every other row of the posterior,
// its cross-covariance with the considered block included, is the full update's; the innovation, so
// the NIS and the log-likelihood, is unchanged. That holds under Linearized, Unscented, and
// Iterated<Linearized>, in each covariance update form, with mixture noise, for a considered block the
// model doesn't read, on a unit quaternion, whose reset leaves the considered block alone, and in each
// mode of an ImmFilter (Filters/ImmFilter.h). With the considered block unread and uncorrelated, the
// update is the update of the rest alone.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

#include <gtest/gtest.h>

#include "unequal_dimensions.h"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Composite;
using Emrys::consider;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::path;
using Emrys::reads;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;
namespace updates = Emrys::updates;

namespace
{

struct Position : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

using State = Composite<Position, Bias, Drift>;

using Vector1 = Eigen::Matrix<double, 1, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

// The difference rows of each block of State.
constexpr Eigen::Index biasRow = 2;
constexpr Eigen::Index driftRow = 3;

// A prior whose blocks are all correlated, so that every block of the posterior moves under the
// full update.
GaussianParams<State>
makePrior()
{
    auto const cov =
        Matrix4{{1.0, 0.2, 0.3, 0.1}, {0.2, 2.0, -0.2, 0.25}, {0.3, -0.2, 0.5, 0.05}, {0.1, 0.25, 0.05, 0.8}};
    return {State{}, Vector4{3.0, 4.0, 0.2, -0.1}, cov};
}

// A range from the origin read with an additive bias, h(p, b) = |p| + b: nonlinear in the position
// and linear in the bias, so that an Iterated update's iterates of the position are the same whether
// the bias moves or not.
template <typename Noise>
auto
makeRangeModel(Noise noise)
{
    return measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{
                Eigen::numext::sqrt(T(block<Position>(in, x).squaredNorm())) + block<Bias>(in, x)(0)
            };
        },
        reads(path<Position>, path<Bias>),
        additiveNoise(std::move(noise))
    );
}

auto const rangeNoise = GaussianParams<Linear<1>>{Vector1{0.0}, Vector1{0.04}};

// The rows of a 4-vector or the rows and columns of a 4 by 4 matrix outside the considered row c.
Eigen::Matrix<double, 3, 1>
rest(Vector4 const & v, Eigen::Index c)
{
    auto result = Eigen::Matrix<double, 3, 1>{};
    for (Eigen::Index i = 0, k = 0; i < 4; ++i)
    {
        if (i != c)
        {
            result(k++) = v(i);
        }
    }
    return result;
}

// The posterior of a consider update must equal the full update's outside the considered row c,
// cross-covariance included, and the prior's at c: its mean and variance exactly.
template <typename Considered, typename Full>
void
expectConsiderPosterior(
    Considered const & considered,
    Full const & full,
    GaussianParams<State> const & prior,
    Eigen::Index c,
    double tolerance
)
{
    Vector4 const mean = considered.mean();
    Matrix4 const cov = considered.covarianceLower();
    Matrix4 const fullCov = full.covarianceLower();
    EXPECT_LT((rest(mean, c) - rest(Vector4{full.mean()}, c)).norm(), tolerance);
    EXPECT_EQ(mean(c), prior.mean()(c));
    EXPECT_EQ(cov(c, c), prior.covarianceLower()(c, c));
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        for (Eigen::Index j = 0; j <= i; ++j)
        {
            if (i != c || j != c)
            {
                EXPECT_NEAR(cov(i, j), fullCov(i, j), tolerance) << i << ", " << j;
            }
        }
    }
    EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-14));
}

}  // namespace

// consider() is an argument of observe(): the measurement names its blocks by type, and one that
// considers none is the full update.
TEST(Consider, IsAnArgumentOfObserve)
{
    auto const model = makeRangeModel(rangeNoise);
    auto const full = observe(methods::linearized(), model, Vector1{5.1});
    auto const considered = observe(methods::linearized(), model, Vector1{5.1}, consider(path<Bias>));
    auto const both =
        observe(methods::linearized(), model, Vector1{5.1}, updates::joseph(), consider(path<Bias>, Drift{}));
    static_assert(std::is_same_v<decltype(full)::ConsiderType, Emrys::Consider<>>);
    static_assert(std::is_same_v<decltype(considered)::ConsiderType, Emrys::Consider<Emrys::TagPath<Bias>>>);
    static_assert(
        std::is_same_v<decltype(both)::ConsiderType, Emrys::Consider<Emrys::TagPath<Bias>, Emrys::TagPath<Drift>>>
    );
    static_assert(std::is_same_v<decltype(both)::FormType, updates::Joseph>);
    static_assert(!Emrys::internal::hasConsider<decltype(full)>);
    static_assert(Emrys::internal::hasConsider<decltype(considered)>);
}

// Under each method and each covariance update form, considering the bias the model reads keeps the
// bias's mean and variance exactly, and gives every other entry of the posterior and the NIS and
// log-likelihood of the full update. The full update is the reference: [consider-states] shows the
// two agree there in exact arithmetic, so the tolerance is rounding, larger for Iterated, whose last
// linearization differs from the full update's in the last digits of the iterate.
TEST(Consider, KeepsTheConsideredBlockAndMatchesTheFullUpdateElsewhere)
{
    GaussianParams<State> const prior = makePrior();
    auto const model = makeRangeModel(rangeNoise);
    auto const z = Vector1{5.6};
    auto const check = [&](auto const & method, auto const & form, double tolerance) {
        auto const full = Emrys::update(prior, observe(method, model, z, form));
        auto const considered = Emrys::update(prior, observe(method, model, z, form, consider(path<Bias>)));
        ASSERT_TRUE(full && considered);
        expectConsiderPosterior(
            considered->branches[0].posterior,
            full->branches[0].posterior,
            prior,
            biasRow,
            tolerance
        );
        EXPECT_NEAR(considered->logLikelihood(), full->logLikelihood(), 1e-14);
        EXPECT_NEAR(considered->branches[0].nis(), full->branches[0].nis(), 1e-14);
        // The full update does move the bias, so the test sees a difference.
        EXPECT_GT(std::abs(full->branches[0].posterior.mean()(biasRow) - prior.mean()(biasRow)), 1e-3);
    };
    auto const iterated = methods::iterated(methods::linearized(), {.tolerance = 1e-12, .maxIterations = 20});
    auto const forms = [&](auto const & method, double tolerance) {
        check(method, updates::standard(), tolerance);
        check(method, updates::symmetric(), tolerance);
        check(method, updates::joseph(), tolerance);
    };
    forms(methods::linearized(), 1e-14);
    forms(methods::unscented(), 1e-13);
    forms(iterated, 1e-10);
}

// Through a KalmanFilter, the update in place gives the posterior that update() returns.
TEST(Consider, AKalmanFilterUpdatesInPlaceAsUpdateDoes)
{
    GaussianParams<State> const prior = makePrior();
    auto const measurement =
        observe(methods::unscented(), makeRangeModel(rangeNoise), Vector1{5.6}, consider(path<Bias>));
    auto const model =
        Emrys::compose(Emrys::RandomWalk<Position>{0.1}, Emrys::RandomWalk<Bias>{0.1}, Emrys::RandomWalk<Drift>{0.1});
    auto filter = Emrys::KalmanFilter{model, prior};
    auto const updated = filter.update(measurement);
    auto const reference = Emrys::update(prior, measurement);
    ASSERT_TRUE(updated && reference);
    EXPECT_EQ(Vector4{filter.state().mean()}, Vector4{reference->branches[0].posterior.mean()});
    EXPECT_LT((filter.state().covarianceLower() - reference->branches[0].posterior.covarianceLower()).norm(), 1e-15);
    EXPECT_EQ(filter.state().mean()(biasRow), prior.mean()(biasRow));
}

// With mixture noise every branch is the full update's branch outside the bias and keeps the
// bias's prior, so the branches share the bias's mean and variance, and the branch weights and the
// mixture's log-likelihood are the full update's.
TEST(Consider, EveryBranchOfMixtureNoiseKeepsTheConsideredBlock)
{
    GaussianParams<State> const prior = makePrior();
    auto noise = GaussianMixtureParams<Linear<1>>{1};
    noise.add(std::log(0.9), Vector1{0.0}, Vector1{0.04});
    noise.add(std::log(0.1), Vector1{0.5}, Vector1{2.0});
    auto const model = makeRangeModel(noise);
    auto const z = Vector1{5.6};
    auto const check = [&](auto const & form) {
        auto const full = Emrys::update(prior, observe(methods::unscented(), model, z, form));
        auto const considered =
            Emrys::update(prior, observe(methods::unscented(), model, z, form, consider(path<Bias>)));
        ASSERT_TRUE(full && considered);
        ASSERT_EQ(considered->branches.size(), 2u);
        for (std::size_t i = 0; i < 2; ++i)
        {
            expectConsiderPosterior(
                considered->branches[i].posterior,
                full->branches[i].posterior,
                prior,
                biasRow,
                1e-13
            );
            EXPECT_NEAR(considered->branches[i].logWeight, full->branches[i].logWeight, 1e-13) << i;
        }
        EXPECT_NEAR(considered->logLikelihood(), full->logLikelihood(), 1e-13);
    };
    check(updates::standard());
    check(updates::symmetric());
    check(updates::joseph());
}

// A considered block the model doesn't read is reached through its covariance with the blocks read:
// the drift's mean and variance stay, and its covariance with the position and the bias, which the
// full update changes, is the full update's.
TEST(Consider, AConsideredBlockTheModelDoesNotRead)
{
    GaussianParams<State> const prior = makePrior();
    auto const model = makeRangeModel(rangeNoise);
    auto const z = Vector1{5.6};
    auto const check = [&](auto const & form) {
        auto const full = Emrys::update(prior, observe(methods::linearized(), model, z, form));
        auto const considered =
            Emrys::update(prior, observe(methods::linearized(), model, z, form, consider(path<Drift>)));
        ASSERT_TRUE(full && considered);
        auto const & posterior = considered->branches[0].posterior;
        expectConsiderPosterior(posterior, full->branches[0].posterior, prior, driftRow, 1e-14);
        // The drift's covariance with the position did change.
        EXPECT_GT(std::abs(posterior.covarianceLower()(driftRow, 0) - prior.covarianceLower()(driftRow, 0)), 1e-3);
    };
    check(updates::standard());
    check(updates::symmetric());
    check(updates::joseph());
}

// With the considered block neither read nor correlated with the rest, H_c = 0 and P_xc = 0, the
// update of the rest is the update of a filter over the rest alone.
TEST(Consider, AnUnreadUncorrelatedBlockLeavesTheUpdateOfTheRest)
{
    using Rest = Composite<Position, Bias>;
    Matrix4 cov = makePrior().covarianceLower();
    cov.row(driftRow).head<3>().setZero();
    cov.col(driftRow).head<3>().setZero();
    auto const prior = GaussianParams<State>{State{}, makePrior().mean(), cov};
    auto const restPrior = GaussianParams<Rest>{
        Rest{},
        Eigen::Vector3d{prior.mean().head<3>()},
        Eigen::Matrix3d{cov.topLeftCorner<3, 3>()}
    };
    auto const z = Vector1{5.6};
    auto const considered =
        Emrys::update(prior, observe(methods::linearized(), makeRangeModel(rangeNoise), z, consider(path<Drift>)));
    auto const alone = Emrys::update(restPrior, observe(methods::linearized(), makeRangeModel(rangeNoise), z));
    ASSERT_TRUE(considered && alone);
    auto const & posterior = considered->branches[0].posterior;
    EXPECT_LT((posterior.mean().head<3>() - alone->branches[0].posterior.mean()).norm(), 1e-14);
    EXPECT_LT(
        (Matrix4{posterior.covarianceLower()}.topLeftCorner<3, 3>() - alone->branches[0].posterior.covarianceLower())
            .norm(),
        1e-14
    );
    EXPECT_EQ(Matrix4{posterior.covarianceLower()}.row(driftRow), cov.row(driftRow));
    EXPECT_EQ(posterior.mean()(driftRow), prior.mean()(driftRow));
}

// On a pose, a unit quaternion and a position, a measured body direction moves the attitude, whose
// reset carries the covariance to the posterior's tangent space. Considering the position, which
// the model doesn't read, the attitude's mean and covariance and its covariance with the position are
// the full update's, reset included, and the position keeps its mean and covariance exactly.
TEST(Consider, TheResetOfAUnitQuaternionLeavesTheConsideredBlock)
{
    namespace ud = EmrysTest::UnequalDimensions;
    GaussianParams<ud::Pose> const prior = ud::pose();
    Eigen::Vector3d const body = Eigen::Vector3d::UnitX();
    auto const model = measurementModel(
        [body]<typename T>(auto const & in, auto const &, auto const & x) {
            return EmrysTest::rotate(Eigen::Matrix<T, 4, 1>{block<ud::Orientation>(in, x)}, body).eval();
        },
        reads(path<ud::Orientation>),
        additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 0.001 * Eigen::Matrix3d::Identity()})
    );
    Eigen::Vector3d const observed = EmrysTest::rotate(
        EmrysTest::multiply(ud::turned, EmrysTest::exponential(Eigen::Vector3d{0.2, -0.1, 0.15})),
        body
    );
    auto const check = [&](auto const & form) {
        auto const full = Emrys::update(prior, observe(methods::linearized(), model, observed, form));
        auto const considered =
            Emrys::update(prior, observe(methods::linearized(), model, observed, form, consider(path<ud::Position>)));
        ASSERT_TRUE(full && considered);
        auto const & a = considered->branches[0].posterior;
        auto const & b = full->branches[0].posterior;
        using Matrix5 = Eigen::Matrix<double, 5, 5>;
        Matrix5 const cov = a.covarianceLower();
        Matrix5 const fullCov = b.covarianceLower();
        EXPECT_LT((a.mean().template head<4>() - b.mean().template head<4>()).norm(), 1e-14);
        EXPECT_EQ(Eigen::Vector2d{a.mean().template tail<2>()}, Eigen::Vector2d{prior.mean().tail<2>()});
        EXPECT_LT((cov.leftCols<3>() - fullCov.leftCols<3>()).norm(), 1e-14);
        Eigen::Matrix2d const position = cov.bottomRightCorner<2, 2>();
        Eigen::Matrix2d const priorPosition = Matrix5{prior.covarianceLower()}.bottomRightCorner<2, 2>();
        EXPECT_EQ(position, priorPosition);
        // The attitude moved enough for the reset to matter, and the full update moved the position.
        EXPECT_GT(Emrys::minus(prior.space(), b.mean(), prior.mean()).template head<3>().norm(), 0.05);
        EXPECT_GT((b.mean().template tail<2>() - prior.mean().tail<2>()).norm(), 1e-3);
    };
    check(updates::standard());
    check(updates::symmetric());
    check(updates::joseph());
}

// An ImmFilter's update conditions each mode's Gaussian, so each keeps the bias the measurement
// considers, while its position moves.
TEST(Consider, EachModeOfAnImmFilterKeepsTheConsideredBlock)
{
    struct Slow
    {};

    struct Fast
    {};

    struct Mode : Emrys::Named<Slow, Fast>
    {};

    auto const walk = [](double density) {
        return Emrys::compose(
            Emrys::RandomWalk<Position>{density},
            Emrys::RandomWalk<Bias>{0.01},
            Emrys::RandomWalk<Drift>{0.01}
        );
    };
    GaussianParams<State> const prior = makePrior();
    auto filter = Emrys::ImmFilter{
        Emrys::SwitchingModel{Emrys::MarkovChain<Mode>::persistent(0.9), walk(0.1), walk(1.0)},
        Emrys::HybridParams{
            Emrys::CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Array2d{0.5, 0.5}),
            std::vector<GaussianParams<State>>{prior, prior}
        }
    };
    ASSERT_TRUE(
        filter.update(observe(methods::linearized(), makeRangeModel(rangeNoise), Vector1{5.6}, consider(path<Bias>)))
    );
    for (Eigen::Index m = 0; m < 2; ++m)
    {
        auto const & mode = filter.state().conditional(m);
        EXPECT_EQ(mode.mean()(biasRow), prior.mean()(biasRow)) << m;
        EXPECT_EQ(mode.covarianceLower()(biasRow, biasRow), prior.covarianceLower()(biasRow, biasRow)) << m;
        EXPECT_GT(std::abs(mode.mean()(0) - prior.mean()(0)), 1e-3) << m;
    }
}

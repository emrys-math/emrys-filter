// RbPointMassFilter, GridGaussian, and PointMassFilter (steps 6, 8a, 9, 9a, 13d; [rb-point-mass],
// [rao-blackwellized-core], [fixed-step-models], [likelihood-measurements]): an update given a block matches the Kalman
// update,
// grids order their cells, shared and per-point covariances match a Kalman filter per cell, a fixed-step model matches
// the step, collapsing over cells, localization on a fine grid, cell-dependent Jacobians, gates reject contradicted
// cells, cells with covariances of their own, subdivision keeps the collapsed state and each cell's Gaussian, and
// PointMassFilter matches the Bayes posterior under both covariance forms, takes a measurement known only by its
// likelihood, which leaves a Rao-Blackwellized grid's Gaussians alone, deduces its defaults, and covers periodic and
// bounded coordinates. With a switching model, each cell holds a hybrid and is an IMM filter given its
// center, through ImmRecursion's update given a block.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Models/Reduction.h>

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <sstream>
#include <span>
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
using Emrys::Gaussian;
using Emrys::GaussianParams;
using Emrys::given;
using Emrys::GridAxis;
using Emrys::GridGaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::PerPointCovariance;
using Emrys::pointGrid;
using Emrys::PointMassFilter;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::RbPointMassFilter;
using Emrys::regularGrid;
using Emrys::ScalarOf;
using Emrys::SharedCovariance;

namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;
namespace upkeep = Emrys::upkeep;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Drift : Linear<1>
{};

struct Scale : Linear<1>
{};

using State = Composite<Position, Bias, Drift>;
using Rest = Composite<Bias, Drift>;

using Vector1 = Eigen::Matrix<double, 1, 1>;
constexpr double rangeVariance = 0.25;
constexpr double biasDensity = 0.01;
constexpr double timeStep = 1.0;

std::array<Eigen::Vector2d, 4> const beacons{
    {Eigen::Vector2d{-40, -30}, Eigen::Vector2d{60, -20}, Eigen::Vector2d{10, 70}, Eigen::Vector2d{-50, 50}}
};

auto
makeModel()
{
    return ConstantVelocity<Bias, Drift>{biasDensity};
}

GaussianParams<Linear<1>>
rangeNoise()
{
    return {Vector1::Zero(), Vector1{rangeVariance}};
}

// A range to a beacon plus the clock bias: nonlinear in position, affine in the bias, by method,
// with any further arguments of observe(), such as a covariance update form.
template <typename Method = methods::Linearized, typename... Args>
auto
makeRange(double z, Eigen::Vector2d const & beacon, Method const & method = {}, Args const &... args)
{
    auto const range = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(d.squaredNorm())) + Emrys::block<Bias>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Bias>),
        additiveNoise(rangeNoise())
    );
    return observe(method, range, Vector1{z}, args...);
}

// The same range for a known position, as a measurement of the bias alone.
auto
makeRangeAt(double z, Eigen::Vector2d const & beacon, Eigen::Vector2d const & p)
{
    auto const range = Emrys::measurementModel(
        [r = (p - beacon).norm()]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{T(r) + Emrys::block<Bias>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Bias>),
        additiveNoise(rangeNoise())
    );
    return observe(methods::linearized(), range, Vector1{z});
}

GaussianParams<Rest>
initialRest()
{
    return {Rest{}, Eigen::Vector2d{0.5, 0}, Eigen::Vector2d{4, 0.01}.asDiagonal().toDenseMatrix()};
}

double
rangeFrom(Eigen::Vector2d const & p, std::size_t beacon, double bias)
{
    return (p - beacons[beacon]).norm() + bias;
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

// update(prior, measurement, given<Position>(space, p)) is the Kalman update of the rest with the
// position known, and with a Gaussian position the innovation covariance widens by the range's
// Jacobian times its covariance.
TEST(RbPointMassFilter, UpdateGivenABlockMatchesTheKalmanUpdate)
{
    auto const prior = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.1}, Eigen::Matrix2d{{2.0, 0.3}, {0.3, 0.5}}};
    auto const p = Eigen::Vector2d{12, -7};
    double const z = 55.0;
    auto const updated = update(prior, makeRange(z, beacons[0]), given<Position>(State{}, p));

    auto const h = Eigen::RowVector2d{1, 0};
    double const s = h * prior.covarianceLower() * h.transpose() + rangeVariance;
    double const innovation = z - (p - beacons[0]).norm() - prior.mean()(0);
    Eigen::Vector2d const k = prior.covarianceLower() * h.transpose() / s;
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 1u);
    EXPECT_TRUE(updated->branches[0].posterior.mean().isApprox(prior.mean() + k * innovation, 1e-12));
    EXPECT_TRUE(updated->branches[0].posterior.covarianceLower().isApprox(
        prior.covarianceLower() - k * s * k.transpose(),
        1e-12
    ));
    EXPECT_NEAR(updated->logLikelihood(), -0.5 * (std::log(2 * pi * s) + innovation * innovation / s), 1e-12);

    // A Gaussian position widens the innovation by the range's sensitivity to it, and the rest is
    // updated as before with the wider innovation.
    auto const c = Eigen::Matrix2d{{0.5, 0.1}, {0.1, 0.3}};
    auto const position = GaussianParams<Position>{Position{}, p, c};
    auto const widened =
        update(prior, makeRange(z, beacons[0], methods::linearized()), given<Position>(State{}, position));
    Eigen::RowVector2d const j = (p - beacons[0]).normalized().transpose();
    double const sw = s + j * c * j.transpose();
    Eigen::Vector2d const kw = prior.covarianceLower() * h.transpose() / sw;
    ASSERT_TRUE(widened);
    EXPECT_TRUE(widened->branches[0].posterior.mean().isApprox(prior.mean() + kw * innovation, 1e-12));
    EXPECT_TRUE(widened->branches[0].posterior.covarianceLower().isApprox(
        prior.covarianceLower() - kw * sw * kw.transpose(),
        1e-12
    ));
    EXPECT_NEAR(widened->logLikelihood(), -0.5 * (std::log(2 * pi * sw) + innovation * innovation / sw), 1e-12);
}

// A range that returns its Jacobian with its value (withJacobian(), [returned-jacobians]), one matrix over
// the blocks read, updates given a block as the range differentiated automatically does, with the
// position known or Gaussian: the position's columns go to the given block and the bias's to the rest.
TEST(RbPointMassFilter, UpdateGivenABlockTakesAReturnedJacobian)
{
    auto const prior = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.1}, Eigen::Matrix2d{{2.0, 0.3}, {0.3, 0.5}}};
    auto const p = Eigen::Vector2d{12, -7};
    double const z = 55.0;
    auto const returningRange = Emrys::measurementModel(
        [beacon = beacons[0]](auto const & in, auto const &, Eigen::Vector3d const & x) {
            Eigen::Vector2d const d = Emrys::block<Position>(in, x) - beacon;
            double const r = d.norm();
            Eigen::RowVector3d jacobian;
            jacobian << d.transpose() / r, 1.0;
            return Emrys::withJacobian(Vector1{r + Emrys::block<Bias>(in, x)(0)}, jacobian);
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Bias>),
        additiveNoise(rangeNoise())
    );
    auto const returning = observe(methods::linearized(), returningRange, Vector1{z});
    auto const expectSame = [](auto const & actual, auto const & expected) {
        ASSERT_TRUE(actual && expected);
        ASSERT_EQ(actual->branches.size(), 1u);
        EXPECT_TRUE(actual->branches[0].posterior.mean().isApprox(expected->branches[0].posterior.mean(), 1e-12));
        EXPECT_TRUE(actual->branches[0].posterior.covarianceLower().isApprox(
            expected->branches[0].posterior.covarianceLower(),
            1e-12
        ));
        EXPECT_NEAR(actual->logLikelihood(), expected->logLikelihood(), 1e-12);
    };
    expectSame(
        update(prior, returning, given<Position>(State{}, p)),
        update(prior, makeRange(z, beacons[0]), given<Position>(State{}, p))
    );
    auto const uncertain = GaussianParams<Position>{Position{}, p, Eigen::Matrix2d{{0.5, 0.1}, {0.1, 0.3}}};
    expectSame(
        update(prior, returning, given<Position>(State{}, uncertain)),
        update(prior, makeRange(z, beacons[0]), given<Position>(State{}, uncertain))
    );
}

// regularGrid<Position>() names the rest space and the cells type, tagged by the block with an axis
// per coordinate, numbers its cells with the last axis fastest, maps coordinates to centers from
// its first, and reports the pinned rows and rest indices.
TEST(RbPointMassFilter, GridsHaveTheirCellsInOrder)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-10, 5}, Eigen::Vector2d{2, 0.5}, {3, 4});
    static_assert(std::is_same_v<decltype(grid)::RestSpace, Rest>);
    static_assert(std::is_same_v<decltype(grid)::CellsType, Emrys::GridCells<Position>>);
    static_assert(std::is_same_v<
                  Emrys::internal::DiscreteProductAxes<Emrys::GridCells<Position>>,
                  std::tuple<GridAxis<Position, 0>, GridAxis<Position, 1>>>);
    EXPECT_EQ(Emrys::valueOf(grid.cells(), Emrys::at<Emrys::GridCells<Position>>(2, 3)), 11);
    EXPECT_EQ(
        Emrys::valueOf(grid.cells(), Emrys::at<GridAxis<Position, 1>>(3), Emrys::at<GridAxis<Position, 0>>(2)),
        11
    );
    EXPECT_EQ(grid.size(), 12);
    EXPECT_EQ(grid.first(), (Eigen::Vector2d{-10, 5}));
    EXPECT_EQ(grid.center(0), (Eigen::Vector2d{-10, 5}));
    EXPECT_EQ(grid.center(1), (Eigen::Vector2d{-10, 5.5}));
    EXPECT_EQ(grid.center(4), (Eigen::Vector2d{-8, 5}));
    EXPECT_EQ(grid.center(grid.cells().value({2, 3})), (Eigen::Vector2d{-6, 6.5}));
    EXPECT_EQ(grid.pinnedRows().start, 0);
    EXPECT_EQ(grid.restIndices(), (Eigen::Vector2<Eigen::Index>{2, 3}));
}

// Every cell of the filter is checked against a KalmanFilter on the bias alone, with the cell's
// position built into the measurement, and the cell probabilities against the sums of those
// filters' log-likelihoods.
template <typename Covariance>
void
expectMatchesKalmanFiltersPerCell()
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), Covariance{}};
    auto kalman = std::vector<KalmanFilter<Gaussian<Rest>, decltype(makeModel())>>{};
    for (Eigen::Index c = 0; c < grid.size(); ++c)
    {
        kalman.emplace_back(makeModel(), initialRest());
    }
    Eigen::ArrayXd logLikelihoods = Eigen::ArrayXd::Zero(grid.size());

    auto const truth = Eigen::Vector2d{1.5, -2.0};
    for (int k = 0; k < 6; ++k)
    {
        std::size_t const b = std::size_t(k) % beacons.size();
        double const z = rangeFrom(truth, b, 1.0 + 0.1 * k) + 0.3 * std::sin(k);
        filter.predict(timeStep);
        auto const result = filter.update(makeRange(z, beacons[b]));
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->accepted);
        ASSERT_TRUE(std::ranges::equal(result->cells(), filter.state().cells()));
        for (Eigen::Index c = 0; c < grid.size(); ++c)
        {
            kalman[std::size_t(c)].predict(timeStep);
            auto const cell = kalman[std::size_t(c)].update(makeRangeAt(z, beacons[b], grid.center(c)));
            ASSERT_TRUE(cell) << c;
            EXPECT_NEAR(result->logLikelihoods()(c), cell->logLikelihood(), 1e-9) << c;
            EXPECT_NEAR(result->nis()(c), cell->branches[0].nis(), 1e-9) << c;
            logLikelihoods(c) += cell->logLikelihood();
        }
    }
    Eigen::ArrayXd const expected =
        (logLikelihoods - logLikelihoods.maxCoeff()).exp() / (logLikelihoods - logLikelihoods.maxCoeff()).exp().sum();
    EXPECT_TRUE(filter.state().toCategorical().logWeights().exp().isApprox(expected, 1e-9));
    if constexpr (std::is_same_v<Covariance, SharedCovariance>)
    {
        EXPECT_EQ(filter.state().sharedCovariance(), filter.state().conditional(0).covarianceLower());
    }
    for (Eigen::Index c = 0; c < grid.size(); ++c)
    {
        EXPECT_TRUE(filter.state().conditional(c).mean().isApprox(kalman[std::size_t(c)].state().mean(), 1e-9)) << c;
        EXPECT_TRUE(filter.state().conditional(c).covarianceLower().isApprox(
            kalman[std::size_t(c)].state().covarianceLower(),
            1e-9
        )) << c;
    }
}

// With a shared covariance, over six range updates every cell's log-likelihood, NIS, and Gaussian
// match a KalmanFilter on the bias with the cell's position built in, and the cell probabilities
// are the normalized products of those likelihoods.
TEST(RbPointMassFilter, SharedCovarianceMatchesAKalmanFilterPerCell)
{
    expectMatchesKalmanFiltersPerCell<SharedCovariance>();
}

// The same check as the shared-covariance test with a covariance per cell.
TEST(RbPointMassFilter, PerPointCovarianceMatchesAKalmanFilterPerCell)
{
    expectMatchesKalmanFiltersPerCell<PerPointCovariance>();
}

template <typename Covariance>
void
expectFixedStepMatchesTheStep()
{
    using Emrys::FixedStepModel;

    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto cached = Emrys::RbPointMassFilter{
        FixedStepModel{makeModel(), grid.restSpace(), timeStep},
        grid,
        initialRest(),
        Covariance{}
    };
    static_assert(std::is_same_v<typename decltype(cached)::MethodType, methods::Linearized>);
    auto unscented = Emrys::RbPointMassFilter{
        FixedStepModel{makeModel(), grid.restSpace(), timeStep, Emrys::methods::unscented<0.5>()},
        grid,
        initialRest(),
        Covariance{}
    };
    static_assert(std::is_same_v<typename decltype(unscented)::MethodType, Emrys::methods::Unscented<0.5>>);
    EXPECT_EQ(unscented.method().alpha, 0.5);
    auto continuousCached = RbPointMassFilter{makeModel(), grid, initialRest(), Covariance{}};
    auto continuousUnscented =
        RbPointMassFilter{makeModel(), grid, initialRest(), Covariance{}, Emrys::methods::unscented<0.5>()};

    auto const expectSame = [&](auto const & filter, auto const & expected, double tolerance) {
        Eigen::ArrayXd const probabilities = filter.state().toCategorical().logWeights().exp();
        EXPECT_LE((probabilities - expected.state().toCategorical().logWeights().exp()).matrix().norm(), tolerance);
        for (Eigen::Index c = 0; c < grid.size(); ++c)
        {
            auto const & cell = filter.state().conditional(c);
            auto const & other = expected.state().conditional(c);
            EXPECT_LE((cell.mean() - other.mean()).norm(), tolerance) << c;
            EXPECT_LE((cell.covarianceLower() - other.covarianceLower()).norm(), tolerance) << c;
        }
    };
    auto const truth = Eigen::Vector2d{1.5, -2.0};
    for (int k = 0; k < 4; ++k)
    {
        std::size_t const b = std::size_t(k) % beacons.size();
        double const z = rangeFrom(truth, b, 1.0 + 0.1 * k) + 0.3 * std::sin(k);
        cached.predict();
        continuousCached.predict(timeStep);
        unscented.predict();
        continuousUnscented.predict(timeStep);
        ASSERT_TRUE(cached.update(makeRange(z, beacons[b])));
        ASSERT_TRUE(continuousCached.update(makeRange(z, beacons[b])));
        ASSERT_TRUE(unscented.update(makeRange(z, beacons[b])));
        ASSERT_TRUE(continuousUnscented.update(makeRange(z, beacons[b])));
        expectSame(cached, continuousCached, 1e-12);
        expectSame(unscented, continuousUnscented, 0.0);
    }
}

// With a FixedStepModel [fixed-step-models] over the grid's rest, under both covariance forms, the filter deduces
// through the abbreviation, takes the model's method, and over four ranges matches the same filter
// predicting over the step: exactly under Unscented, which takes the same path with the same noise,
// and to rounding under Linearized, which predicts by the cached transition.
TEST(RbPointMassFilter, AFixedStepModelMatchesTheStep)
{
    expectFixedStepMatchesTheStep<SharedCovariance>();
    expectFixedStepMatchesTheStep<PerPointCovariance>();
}

// With a shared covariance, a range observed with a covariance update form updates the one covariance in
// its form: the symmetric and Joseph forms equal the standard form to rounding, which is the same
// update in exact arithmetic, and their shared covariance is exactly symmetric.
TEST(RbPointMassFilter, SharedCovarianceUsesTheMeasurementsForm)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto const run = [&](auto const & form) {
        auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), SharedCovariance{}};
        auto const truth = Eigen::Vector2d{1.5, -2.0};
        for (int k = 0; k < 4; ++k)
        {
            std::size_t const b = std::size_t(k) % beacons.size();
            filter.predict(timeStep);
            EXPECT_TRUE(filter.update(makeRange(rangeFrom(truth, b, 1.0), beacons[b], methods::linearized(), form)));
        }
        return Eigen::Matrix2d{filter.state().conditional(0).covarianceLower()};
    };
    Eigen::Matrix2d const standard = run(Emrys::updates::standard());
    Eigen::Matrix2d const symmetric = run(Emrys::updates::symmetric());
    Eigen::Matrix2d const joseph = run(Emrys::updates::joseph());
    EXPECT_TRUE(symmetric.isApprox(standard, 1e-12));
    EXPECT_TRUE(joseph.isApprox(standard, 1e-12));
    EXPECT_EQ(symmetric, symmetric.transpose());
    EXPECT_EQ(joseph, joseph.transpose());
}

// collapsed() of two cells with covariances about their centers has the probability-weighted mean,
// the spread of the centers plus the cell covariance in the position block, and the shared rest
// covariance.
TEST(RbPointMassFilter, CollapsesOverCells)
{
    // Two cells with a covariance about their centers.
    auto const centers = Eigen::Matrix<double, 2, 2>{{0, 10}, {0, 0}};
    Eigen::Matrix2d const c = Eigen::Vector2d{1.0 / 12, 1.0 / 12}.asDiagonal();
    auto const grid = pointGrid<State, Position>(centers, c);
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), PerPointCovariance{}};
    ASSERT_TRUE(filter.state().setWeights(Emrys::CategoricalParams{grid.cells(), Eigen::Array2d{0.25, 0.75}.log()}));
    auto const collapsed = filter.state().toDensity()->collapsed();
    EXPECT_TRUE(collapsed.mean().isApprox(Eigen::Vector4d{7.5, 0, 0.5, 0}, 1e-14));
    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    expected(0, 0) = 0.25 * 0.75 * 100;
    expected.topLeftCorner<2, 2>() += c;
    expected.bottomRightCorner<2, 2>() = initialRest().covarianceLower();
    EXPECT_TRUE(collapsed.covarianceLower().isApprox(expected, 1e-14));
}

// A static receiver on a 101-by-101 grid with a drifting clock bias is located within 1 unit after
// 40 ranges, the axis marginals peak at the truth and match the dense categorical's, PruneBelow leaves
// under 200 sorted cells with their Gaussians and the estimate unchanged, and the filter runs on
// with them.
TEST(RbPointMassFilter, LocalizesOnAFineGrid)
{
    // A static receiver somewhere in a 100 m square, with ranges to four beacons and a drifting
    // clock bias, on a 1 m grid of 10201 cells.
    auto rng = std::mt19937_64{3};
    auto normal = std::normal_distribution<double>{};
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{1, 1}, {101, 101});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest()};
    static_assert(std::is_same_v<decltype(filter)::GridType, std::remove_const_t<decltype(grid)>>);

    auto const truth = Eigen::Vector2d{17.3, -8.6};
    double bias = 1.0;
    for (int k = 0; k < 40; ++k)
    {
        bias += 0.05;
        std::size_t const b = std::size_t(k) % beacons.size();
        filter.predict(timeStep);
        ASSERT_TRUE(
            filter.update(makeRange(rangeFrom(truth, b, bias) + std::sqrt(rangeVariance) * normal(rng), beacons[b]))
        );
    }
    auto const & state = filter.state();
    auto const estimate = state.toDensity()->collapsed();
    EXPECT_LT((estimate.mean().head<2>() - truth).norm(), 1.0) << estimate.mean().transpose();
    EXPECT_LT((state.center(state.mostProbable()) - truth).norm(), 1.5);
    EXPECT_NEAR(estimate.mean()(2), bias, 0.5);

    // Each axis's marginal peaks at the truth's row and column, and matches the dense one.
    auto const xWeights = state.marginal<GridAxis<Position, 0>>();
    auto const x = xWeights.toDensity().value();
    EXPECT_NEAR(double(x.mostProbable()) - 50, truth(0), 1.0);
    EXPECT_TRUE(
        x.probabilities().isApprox(state.toCategorical().marginal<GridAxis<Position, 0>>().logWeights().exp(), 1e-12)
    );

    // Pruning removes all but the cells that matter, and the estimate barely moves. The cells
    // left keep their Gaussians.
    Eigen::Index const best = state.cell(state.mostProbable());
    auto const bestMean = state.conditional(state.mostProbable()).mean().eval();
    auto const pruned = upkeep::pruneBelow(1e-12)(filter.state());
    ASSERT_TRUE(pruned.has_value());
    EXPECT_TRUE(*pruned);
    EXPECT_LT(state.size(), 200);
    EXPECT_TRUE(std::is_sorted(state.cells().begin(), state.cells().end()));
    EXPECT_NEAR(state.logWeights().exp().sum(), 1.0, 1e-12);
    EXPECT_TRUE(state.toDensity()->collapsed().mean().isApprox(estimate.mean(), 1e-6));
    ASSERT_TRUE(state.find(best).has_value());
    EXPECT_EQ(state.conditional(state.find(best).value()).mean(), bestMean);
    EXPECT_FALSE(state.find(0).has_value());

    // The filter goes on with the cells left.
    filter.predict(timeStep);
    auto const result = filter.update(makeRange(rangeFrom(truth, 0, bias + 0.05), beacons[0]));
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->accepted);
    EXPECT_EQ(result->logLikelihoods().size(), Eigen::Index(result->cells().size()));
}

// A range with an unknown scale factor, whose Jacobian in the scale is each cell's range, localizes
// within 1 unit and leaves the cells with different covariances.
TEST(RbPointMassFilter, PerPointCovariancesFollowACellDependentJacobian)
{
    // A range with an unknown scale factor: its Jacobian in the scale is the cell's true range,
    // so the cells' covariances differ.
    using ScaledState = Composite<Position, Scale>;
    auto rng = std::mt19937_64{5};
    auto normal = std::normal_distribution<double>{};
    auto const grid = regularGrid<ScaledState, Position>(Eigen::Vector2d{-20, -20}, Eigen::Vector2d{1, 1}, {41, 41});
    auto filter = RbPointMassFilter{
        RandomWalk<Scale>{1e-8},
        grid,
        GaussianParams<Scale>{Scale{}, Vector1{0}, Vector1{1e-4}},
        PerPointCovariance{}
    };
    auto const truth = Eigen::Vector2d{6.2, -11.7};
    constexpr double scale = 0.004;
    for (int k = 0; k < 24; ++k)
    {
        std::size_t const b = std::size_t(k) % beacons.size();
        double const z = (1 + scale) * (truth - beacons[b]).norm() + std::sqrt(rangeVariance) * normal(rng);
        filter.predict(timeStep);
        auto const scaledRange = Emrys::measurementModel(
            [beacon = beacons[b]]<typename T>(auto const & in, auto const &, auto const & x) {
                return Eigen::Matrix<T, 1, 1>{
                    (T(1) + Emrys::block<Scale>(in, x)(0))
                    * Eigen::numext::sqrt(T((Emrys::block<Position>(in, x) - beacon.cast<T>()).squaredNorm()))
                };
            },
            Emrys::reads(Emrys::path<Position>, Emrys::path<Scale>),
            additiveNoise(rangeNoise())
        );
        ASSERT_TRUE(filter.update(observe(methods::linearized(), scaledRange, Vector1{z})));
    }
    EXPECT_LT((filter.state().toDensity()->collapsed().mean().head<2>() - truth).norm(), 1.0);
    EXPECT_NE(
        filter.state().conditional(0).covarianceLower()(0, 0),
        filter.state().conditional(grid.size() - 1).covarianceLower()(0, 0)
    );
}

// Under a Gate, cells whose innovation fails get log-likelihood minus infinity and are removed, the
// survivor renormalizes to one, and a measurement no cell can explain is rejected and changes
// nothing.
TEST(RbPointMassFilter, AGateRejectsCellsAMeasurementContradicts)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-20, 0}, Eigen::Vector2d{20, 1}, {3, 1});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), PerPointCovariance{}};
    // The truth is at the middle cell; the outer cells are 20 m off, which the bias can't explain.
    double const z = rangeFrom(Eigen::Vector2d{0, 0}, 1, 0.5);
    auto const result = filter.update(makeRange(z, beacons[1]), reductions::gate(0.997));
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    EXPECT_TRUE(std::ranges::equal(result->cells(), std::vector<Eigen::Index>{0, 1, 2}));
    EXPECT_FALSE(std::isfinite(result->logLikelihoods()(0)));
    EXPECT_FALSE(std::isfinite(result->logLikelihoods()(2)));
    // The rejected cells are gone.
    ASSERT_EQ(filter.state().size(), 1);
    EXPECT_EQ(filter.state().cell(0), 1);
    EXPECT_EQ(filter.state().logWeight(0), 0.0);

    // A measurement no cell can explain is rejected and changes nothing.
    auto const before = filter.state().conditional(0).mean();
    auto const far = filter.update(makeRange(z + 1000, beacons[1]), reductions::gate(0.997));
    ASSERT_TRUE(far);
    EXPECT_FALSE(far->accepted);
    EXPECT_EQ(filter.state().conditional(0).mean(), before);
    EXPECT_EQ(filter.state().size(), 1);
    EXPECT_EQ(filter.state().logWeight(0), 0.0);
}

// pointGrid() with a covariance per cell makes each cell's update the Kalman update given a
// Gaussian position with that covariance, the probabilities follow, and collapsing weights each
// cell's covariance by its probability.
TEST(RbPointMassFilter, CellsMayHaveCovariancesOfTheirOwn)
{
    // Three cells of different sizes; each widens its own innovation by its own covariance.
    auto const centers = Eigen::Matrix<double, 2, 3>{{-5, 0, 8}, {2, 0, -3}};
    auto const covariances = std::vector<Eigen::Matrix2d>{
        Eigen::Matrix2d{{0.1, 0.0}, {0.0, 0.1}},
        Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.5}},
        Eigen::Matrix2d{{4.0, -1.0}, {-1.0, 2.0}}
    };
    auto const grid = pointGrid<State, Position>(centers, covariances);
    static_assert(std::is_same_v<decltype(grid)::CellCovarianceType, PerPointCovariance>);
    EXPECT_EQ(grid.cellCovariance(2), covariances[2]);

    // A regular grid takes covariances per cell in flat-index order too.
    auto const regular = regularGrid<State, Position>(
        Eigen::Vector2d{-5, 0},
        Eigen::Vector2d{1, 1},
        {3, 1},
        std::span<Eigen::Matrix2d const>{covariances}
    );
    static_assert(std::is_same_v<decltype(regular)::CellCovarianceType, PerPointCovariance>);
    EXPECT_EQ(regular.center(2), (Eigen::Vector2d{-3, 0}));
    EXPECT_EQ(regular.cellCovariance(1), covariances[1]);
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), PerPointCovariance{}};

    double const z = 60.0;
    auto const result = filter.update(makeRange(z, beacons[1], methods::linearized()));
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->accepted);
    auto joint = Eigen::Array3d{};
    for (Eigen::Index c = 0; c < 3; ++c)
    {
        auto const pinned = GaussianParams<Position>{Position{}, centers.col(c), covariances[std::size_t(c)]};
        auto const expected =
            update(initialRest(), makeRange(z, beacons[1], methods::linearized()), given<Position>(State{}, pinned));
        ASSERT_TRUE(expected) << c;
        EXPECT_NEAR(result->logLikelihoods()(c), expected->logLikelihood(), 1e-12) << c;
        EXPECT_TRUE(filter.state().conditional(c).mean().isApprox(expected->branches[0].posterior.mean(), 1e-12)) << c;
        EXPECT_TRUE(filter.state().conditional(c).covarianceLower().isApprox(
            expected->branches[0].posterior.covarianceLower(),
            1e-12
        )) << c;
        joint(c) = std::exp(expected->logLikelihood()) / 3;
    }
    Eigen::Array3d const p = joint / joint.sum();
    EXPECT_TRUE(filter.state().toCategorical().logWeights().exp().isApprox(p, 1e-12));

    // Collapsing weights each cell's covariance by its probability.
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (Eigen::Index c = 0; c < 3; ++c)
    {
        mean += p(c) * centers.col(c);
    }
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (Eigen::Index c = 0; c < 3; ++c)
    {
        Eigen::Vector2d const d = centers.col(c) - mean;
        cov += p(c) * (covariances[std::size_t(c)] + d * d.transpose());
    }
    auto const collapsed = filter.state().toDensity()->collapsed();
    EXPECT_TRUE(collapsed.mean().head<2>().isApprox(mean, 1e-12));
    EXPECT_TRUE((collapsed.covarianceLower().topLeftCorner<2, 2>().isApprox(cov, 1e-12)));
}

namespace
{

template <typename State>
concept Subdividable = requires(State & state) { state.subdivide(Eigen::Index{2}); };

}  // namespace

// subdivide() exists for a regular grid with a shared cell covariance and not for an irregular
// point grid, whose cells are PointCells, or a regular grid with a covariance per cell.
TEST(GridGaussian, OnlyARegularGridWithOneCellCovarianceSubdivides)
{
    using Regular = decltype(regularGrid<State, Position>(Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {2, 2}));
    using Irregular = decltype(pointGrid<State, Position>(Eigen::Matrix2d::Zero()));
    using RegularPerCell = decltype(regularGrid<State, Position>(
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Ones(),
        {2, 2},
        std::vector<Eigen::Matrix2d>{}
    ));
    static_assert(Regular::Regular && !Irregular::Regular && RegularPerCell::Regular);
    static_assert(std::is_same_v<Irregular::CellsType, Emrys::PointCells<Position>>);
    static_assert(Subdividable<GridGaussianParams<Regular>>);
    static_assert(Subdividable<GridGaussianParams<Regular, PerPointCovariance>>);
    static_assert(!Subdividable<GridGaussianParams<Irregular>>);
    static_assert(!Subdividable<GridGaussianParams<RegularPerCell, PerPointCovariance>>);
}

// Splitting cells whose covariance is that of a uniform box leaves the collapsed state unchanged:
// the children's spread and covariances add up to the parent's covariance.
template <typename Covariance>
void
expectSubdivisionKeepsTheCollapsedState(Eigen::Index factor)
{
    auto const spacing = Eigen::Vector2d{2, 0.5};
    Eigen::Matrix2d const box = (spacing.array().square() / 12).matrix().asDiagonal();
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-10, 5}, spacing, {3, 4}, box);
    auto state = GridGaussianParams{grid, initialRest(), Covariance{}};
    Eigen::ArrayXd weights = Eigen::ArrayXd::LinSpaced(12, 1, 12);
    weights(5) = 0;  // a dead cell has no children
    ASSERT_TRUE(state.setWeights(Emrys::CategoricalParams{grid.cells(), weights.log()}));
    ASSERT_EQ(state.size(), 11);
    // Weights that give every live cell weight zero are refused, and the weights stay as they were
    // ([update-outcomes]).
    Eigen::ArrayXd const zero = Eigen::ArrayXd::Zero(12);
    auto const refused = state.setWeights(Emrys::CategoricalParams{grid.cells(), zero.log()});
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error(), Emrys::WeightError::ZeroSum);
    ASSERT_EQ(state.size(), 11);
    auto const before = state.toDensity()->collapsed();

    state.subdivide(factor);
    Eigen::Index const children = factor * factor;
    ASSERT_EQ(state.size(), 11 * children);
    EXPECT_TRUE(std::is_sorted(state.cells().begin(), state.cells().end()));
    auto const & refined = state.grid();
    EXPECT_EQ(refined.size(), 12 * children);
    EXPECT_TRUE(refined.spacing().isApprox(spacing / double(factor), 1e-15));
    EXPECT_TRUE(refined.cellCovariance(0).isApprox(box / double(children), 1e-15));
    auto const after = state.toDensity()->collapsed();
    EXPECT_TRUE(after.mean().isApprox(before.mean(), 1e-12));
    EXPECT_TRUE(after.covarianceLower().isApprox(before.covarianceLower(), 1e-12));

    // Cell (2, 3) of the old grid becomes (2 k + a, 3 k + b), each with a share of its
    // probability, around its center.
    Eigen::Vector2d const parentCenter = grid.center(grid.cells().value({2, 3}));
    double const parentProbability = weights(grid.cells().value({2, 3})) / weights.sum();
    Eigen::Vector2d meanOfChildren = Eigen::Vector2d::Zero();
    for (Eigen::Index a = 0; a < factor; ++a)
    {
        for (Eigen::Index b = 0; b < factor; ++b)
        {
            auto const i = state.find(refined.cells().value({2 * factor + a, 3 * factor + b}));
            ASSERT_TRUE(i.has_value());
            EXPECT_NEAR(std::exp(state.logWeight(*i)), parentProbability / double(children), 1e-14);
            EXPECT_EQ(state.conditional(*i).mean(), initialRest().mean());
            meanOfChildren += state.center(*i) / double(children);
        }
    }
    EXPECT_TRUE(meanOfChildren.isApprox(parentCenter, 1e-14));
    EXPECT_FALSE(state.find(refined.cells().value({1 * factor, 1 * factor})).has_value());
}

// Splitting each live cell into factor-squared children with the uniform-box covariance leaves the
// collapsed state unchanged, keeps the cells sorted, shares each parent's probability among its
// children about its center, and gives a dead cell no children, for factors 2 and 3 and both
// covariance forms.
TEST(GridGaussian, SubdivisionKeepsTheCollapsedState)
{
    expectSubdivisionKeepsTheCollapsedState<SharedCovariance>(2);
    expectSubdivisionKeepsTheCollapsedState<SharedCovariance>(3);
    expectSubdivisionKeepsTheCollapsedState<PerPointCovariance>(2);
}

// upkeep::whenSize(std::less{}, 4, upkeep::subdivide(2)) subdivides when fewer than four cells are live, not after, and
// each child gets a copy of its parent's Gaussian.
TEST(GridGaussian, SubdivisionCarriesEachCellsGaussian)
{
    // Cells with Gaussians of their own: each child gets a copy of its parent's.
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{0, 0}, Eigen::Vector2d{20, 20}, {3, 1});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), PerPointCovariance{}};
    ASSERT_TRUE(filter.update(makeRange(rangeFrom(Eigen::Vector2d{20, 0}, 1, 0.5), beacons[1])));
    auto parents = std::vector<GaussianParams<Rest>>{};
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        parents.push_back(filter.state().conditional(i));
    }
    ASSERT_NE(parents[0].mean(), parents[1].mean());

    auto const refine = upkeep::whenSize(std::less{}, 4, upkeep::subdivide(2));
    ASSERT_TRUE(refine(filter.state()).value());
    ASSERT_EQ(filter.state().size(), 12);
    EXPECT_FALSE(refine(filter.state()).value());  // 12 cells are enough
    auto const & refined = filter.state().grid();
    for (Eigen::Index i = 0; i < filter.state().size(); ++i)
    {
        auto const coordinates = refined.cells().coordinates(filter.state().cell(i));
        auto const & parent = parents[std::size_t(coordinates[0] / 2)];
        EXPECT_EQ(filter.state().conditional(i).mean(), parent.mean()) << i;
        EXPECT_EQ(filter.state().conditional(i).covarianceLower(), parent.covarianceLower()) << i;
    }
}

// Starting on a 5-unit grid and pruning and subdividing whenever fewer than 100 cells are live
// reaches a 0.625-unit grid with under 1000 cells alive at once, the position within 1 unit, and
// the bias within three standard deviations.
TEST(RbPointMassFilter, RefinesACoarseGrid)
{
    // The receiver of LocalizesOnAFineGrid on a 5 m grid of 441 cells, each widened by its own
    // box. Pruning and subdividing whenever fewer than 100 cells are left reach a 0.625 m grid.
    auto rng = std::mt19937_64{3};
    auto normal = std::normal_distribution<double>{};
    double const spacing = 5;
    Eigen::Matrix2d const box = Eigen::Matrix2d::Identity() * spacing * spacing / 12;
    auto const grid =
        regularGrid<State, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{spacing, spacing}, {21, 21}, box);
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest()};
    auto const prune = upkeep::pruneBelow(1e-9);
    auto const refine = upkeep::whenSize(std::less{}, 100, upkeep::subdivide(2));

    auto const truth = Eigen::Vector2d{17.3, -8.6};
    double bias = 1.0;
    Eigen::Index mostCells = 0;
    for (int k = 0; k < 60; ++k)
    {
        bias += 0.05;
        std::size_t const b = std::size_t(k) % beacons.size();
        filter.predict(timeStep);
        auto const result = filter.update(makeRange(
            rangeFrom(truth, b, bias) + std::sqrt(rangeVariance) * normal(rng),
            beacons[b],
            methods::linearized()
        ));
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->accepted);
        ASSERT_TRUE(prune(filter.state()).has_value());
        if (filter.state().grid().spacing()(0) > 0.7)
        {
            ASSERT_TRUE(refine(filter.state()).has_value());
        }
        mostCells = std::max(mostCells, filter.state().size());
    }
    // Over 20 seeds the position error is at most 0.48 m, at most 324 cells are alive at once,
    // and the bias error is within twice its standard deviation.
    auto const & state = filter.state();
    EXPECT_DOUBLE_EQ(state.grid().spacing()(0), 0.625);
    EXPECT_LT(mostCells, 1000);
    auto const estimate = state.toDensity()->collapsed();
    EXPECT_LT((estimate.mean().head<2>() - truth).norm(), 1.0) << estimate.mean().transpose();
    EXPECT_LT((state.center(state.mostProbable()) - truth).norm(), 1.0);
    EXPECT_LT(std::abs(estimate.mean()(2) - bias), 3 * std::sqrt(estimate.covarianceLower()(2, 2)));
}

// A range to a beacon from a position alone, for a grid over the whole state.
auto
makePositionRange(double z, Eigen::Vector2d const & beacon)
{
    auto const range = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(d.squaredNorm()))};
        },
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(rangeNoise())
    );
    return observe(methods::linearized(), range, Vector1{z});
}

template <typename Covariance>
void
checkPointMassMatchesBayes()
{
    // With nothing left Gaussian, each cell's probability is the normalized product of the range
    // likelihoods at its center, and predict changes nothing.
    using Whole = Composite<Position>;
    auto const grid = regularGrid<Whole, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{5, 5}, {21, 21});
    auto filter = PointMassFilter{grid, Covariance{}};
    static_assert(std::is_same_v<
                  decltype(filter),
                  RbPointMassFilter<
                      std::remove_const_t<decltype(grid)>,
                      Emrys::NoProcessModel,
                      Covariance,
                      methods::Linearized>>);
    static_assert(decltype(filter)::StateType::ConditionalType::SpaceType::FixedValueDimension == 0);

    auto const truth = Eigen::Vector2d{17.3, -8.6};
    auto const errors = std::array<double, 4>{0.3, -0.5, 0.1, 0.7};
    Eigen::ArrayXd expected = Eigen::ArrayXd::Zero(grid.size());
    for (std::size_t b = 0; b < beacons.size(); ++b)
    {
        double const z = (truth - beacons[b]).norm() + errors[b];
        filter.predict(timeStep);
        auto const result = filter.update(makePositionRange(z, beacons[b]));
        ASSERT_TRUE(result);
        EXPECT_TRUE(result->accepted);
        EXPECT_EQ(result->innovationSize(), 1);
        for (Eigen::Index c = 0; c < grid.size(); ++c)
        {
            double const r = z - (grid.center(c) - beacons[b]).norm();
            expected(c) += -0.5 * r * r / rangeVariance;
        }
    }
    expected -= expected.maxCoeff();
    expected = expected.exp() / expected.exp().sum();

    auto const & state = filter.state();
    ASSERT_EQ(state.size(), grid.size());
    for (Eigen::Index i = 0; i < state.size(); ++i)
    {
        EXPECT_NEAR(std::exp(state.logWeights()(i)), expected(state.cell(i)), 1e-12);
    }
    auto const estimate = state.toDensity()->collapsed();
    static_assert(std::is_same_v<typename decltype(estimate)::SpaceType, Whole>);
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (Eigen::Index c = 0; c < grid.size(); ++c)
    {
        mean += expected(c) * grid.center(c);
    }
    EXPECT_TRUE(estimate.mean().isApprox(mean, 1e-12));
    EXPECT_LT((estimate.mean() - truth).norm(), 5.0);
    EXPECT_TRUE(upkeep::pruneBelow(1e-9)(filter.state()).has_value());
    EXPECT_LT(state.size(), grid.size());
}

// With the grid covering the whole state, each cell's probability is the normalized product of the
// range likelihoods at its center, predict() changes nothing, the collapsed mean is the
// probability-weighted center, and PruneBelow reduces the cell count.
TEST(PointMassFilter, MatchesTheBayesPosteriorWithSharedCovariance)
{
    checkPointMassMatchesBayes<SharedCovariance>();
}

// The same check as the shared-covariance test with a covariance per cell.
TEST(PointMassFilter, MatchesTheBayesPosteriorWithPerPointCovariance)
{
    checkPointMassMatchesBayes<PerPointCovariance>();
}

// PointMassFilter{grid} deduces the default model, covariance form, and method, the same as
// RbPointMassFilter{grid}, and starts uniform.
TEST(PointMassFilter, DeducesTheDefaultsWithoutACovarianceForm)
{
    auto const grid =
        regularGrid<Composite<Position>, Position>(Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {2, 2});
    auto const filter = PointMassFilter{grid};
    static_assert(std::is_same_v<decltype(filter), PointMassFilter<std::remove_const_t<decltype(grid)>> const>);
    auto const full = RbPointMassFilter{grid};
    static_assert(std::is_same_v<decltype(full), decltype(filter)>);
    EXPECT_NEAR(filter.state().logWeights().exp().sum(), 1.0, 1e-15);
    auto copy = full;
    EXPECT_EQ(&copy.model(), &std::as_const(copy).model());
}

// On a grid over a heading and a speed, a heading fix near the wrap gathers the probability
// on both ends of the axis and the collapsed heading is their circular mean.
TEST(PointMassFilter, CoversPeriodicAndLinearCoordinates)
{
    // A heading and a speed, gridded over the full circle and [0, 10): a heading fix near the wrap
    // gathers the probability on both sides of it, and the estimate's heading is their circular
    // mean rather than the average of the grid's ends.
    struct Heading : RadiansPi<>
    {};

    struct Speed : Linear<1>
    {};

    struct Motion : Composite<Heading, Speed>
    {};

    double const step = 2 * pi / 72;
    auto const grid =
        regularGrid<Composite<Motion>, Motion>(Eigen::Vector2d{-pi, 0.25}, Eigen::Vector2d{step, 0.5}, {72, 20});
    auto filter = PointMassFilter{grid, PerPointCovariance{}};
    double const heading = pi - 0.02;
    // The heading is measured in a periodic space, which wraps the residual.
    auto const motion = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 2, 1>{Emrys::block<Heading>(in, x)(0), Emrys::block<Speed>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Heading>, Emrys::path<Speed>),
        Emrys::outputSpace(Motion{}),
        additiveNoise(
            GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Vector2d{0.01, 0.04}.asDiagonal().toDenseMatrix()}
        )
    );
    auto const fix = observe(methods::linearized(), motion, Eigen::Vector2d{heading, 3.1});
    ASSERT_TRUE(filter.update(fix));
    ASSERT_TRUE(filter.update(fix));

    auto const & state = filter.state();
    auto const hWeights = state.marginal<GridAxis<Heading>>();
    auto const h = hWeights.toDensity().value();
    EXPECT_TRUE(h.mostProbable() == 0 || h.mostProbable() == 71);
    EXPECT_GT(h.probabilities()(0) + h.probabilities()(71), 0.5);
    auto const estimate = state.toDensity()->collapsed();
    EXPECT_NEAR(std::abs(estimate.mean()(0)), pi, 0.05);
    EXPECT_NEAR(estimate.mean()(1), 3.1, 0.2);
    EXPECT_LT(estimate.covarianceLower()(0, 0), 0.05);
}

namespace
{

constexpr double quantum = 2.0;
constexpr double timingDeviation = 0.5;

// The integral of the standard normal distribution function up to t, phi(t) + t Phi(t).
double
integratedNormal(double t)
{
    return std::exp(-0.5 * t * t) / std::sqrt(2 * pi) + t * 0.5 * std::erfc(-t / std::numbers::sqrt2);
}

// The log-density at e of a triangle on [-quantum, quantum] convolved with a Gaussian of standard
// deviation timingDeviation, in closed form: the triangle is a second difference of ramps, and a
// ramp convolved with the Gaussian is sigma * integratedNormal(x / sigma). The density is even, so
// it is evaluated at -|e|, where the terms do not cancel.
double
triangleGaussianLogDensity(double e)
{
    double const s = timingDeviation;
    double const t = -std::abs(e) / s;
    double const h = quantum / s;
    double const density =
        s * (integratedNormal(t + h) - 2 * integratedNormal(t) + integratedNormal(t - h)) / (quantum * quantum);
    return density > 0 ? std::log(density) : -std::numeric_limits<double>::infinity();
}

// The same density as the integral of the triangle times the Gaussian, by Simpson's rule on 2000
// intervals with the triangle's peak at a node, for the hand computation.
double
triangleGaussianDensityByQuadrature(double e)
{
    int const intervals = 2000;
    double const h = 2 * quantum / intervals;
    double sum = 0;
    for (int k = 0; k <= intervals; ++k)
    {
        double const t = -quantum + k * h;
        double const weight = k == 0 || k == intervals ? 1 : (k % 2 == 1 ? 4 : 2);
        double const r = (e - t) / timingDeviation;
        double const gaussian = std::exp(-0.5 * r * r) / (std::sqrt(2 * pi) * timingDeviation);
        sum += weight * (quantum - std::abs(t)) / (quantum * quantum) * gaussian;
    }
    return sum * h / 3;
}

// A range to a beacon from a position alone, its error a quantization triangle convolved with
// Gaussian noise, known only by its likelihood: its measurement() is a model with logLikelihood(g)
// and no function, so no noise for a Kalman update.
struct QuantizedRange
{
    double z;
    Eigen::Vector2d beacon;

    [[nodiscard]] double
    logLikelihood(Eigen::Vector2d const & p) const
    {
        return triangleGaussianLogDensity(z - (p - beacon).norm());
    }

    [[nodiscard]] auto
    measurement() const
    {
        return observe(
            Emrys::measurementModel(
                Emrys::reads(Emrys::path<Position>),
                Emrys::logLikelihood([beacon = beacon](auto const & in, auto const & x, auto const & observed) {
                    return triangleGaussianLogDensity(observed(0) - (Emrys::block<Position>(in, x) - beacon).norm());
                })
            ),
            Vector1{z}
        );
    }
};

static_assert(Emrys::Measurement<decltype(QuantizedRange{}.measurement())>);
static_assert(!decltype(QuantizedRange{}.measurement())::ModelType::HasFunction);

// The ranges from the truth with errors from the triangle's support and beyond it.
std::array<QuantizedRange, 4>
quantizedRanges(Eigen::Vector2d const & truth)
{
    auto const errors = std::array<double, 4>{1.3, -0.5, 0.1, -2.4};
    auto ranges = std::array<QuantizedRange, 4>{};
    for (std::size_t b = 0; b < beacons.size(); ++b)
    {
        ranges[b] = QuantizedRange{(truth - beacons[b]).norm() + errors[b], beacons[b]};
    }
    return ranges;
}

template <typename Covariance>
Eigen::ArrayXd
checkPointMassTakesALikelihood(Eigen::ArrayXd const & expected, double evidence)
{
    using Whole = Composite<Position>;
    auto const grid = regularGrid<Whole, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{5, 5}, {21, 21});
    auto filter = PointMassFilter{grid, Covariance{}};
    double logLikelihood = 0;
    for (auto const & range : quantizedRanges(Eigen::Vector2d{17.3, -8.6}))
    {
        auto const result = filter.update(range.measurement());
        EXPECT_TRUE(result);
        if (!result)
        {
            return {};
        }
        EXPECT_TRUE(result->accepted);
        EXPECT_EQ(result->innovationSize(), 0);
        EXPECT_TRUE(result->nis().isNaN().all());
        logLikelihood += result->logLikelihood();
    }
    EXPECT_NEAR(logLikelihood, evidence, 1e-9);
    // The cells whose likelihood underflows to zero are removed, and count as probability zero.
    auto const & state = filter.state();
    EXPECT_LT(state.size(), grid.size());
    Eigen::ArrayXd weights = Eigen::ArrayXd::Zero(grid.size());
    for (Eigen::Index i = 0; i < state.size(); ++i)
    {
        weights(state.cell(i)) = std::exp(state.logWeights()(i));
    }
    EXPECT_LT((weights - expected).abs().maxCoeff(), 1e-9);
    return weights;
}

}  // namespace

// A range known only by its likelihood, a quantization triangle convolved with Gaussian noise,
// takes a PointMassFilter to the Bayes posterior over the grid: each cell's probability is the
// normalized product of the likelihoods at its center, by quadrature of the convolution, and the
// evidence the sum of the updates'. The cells far from every range, where the likelihood
// underflows to zero, are removed. The result has NIS NaN and no innovation, and both covariance
// forms give the same weights to rounding.
TEST(PointMassFilter, TakesAMeasurementKnownOnlyByItsLikelihood)
{
    auto const grid =
        regularGrid<Composite<Position>, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{5, 5}, {21, 21});
    Eigen::ArrayXd logPosterior = Eigen::ArrayXd::Constant(grid.size(), -std::log(double(grid.size())));
    for (auto const & range : quantizedRanges(Eigen::Vector2d{17.3, -8.6}))
    {
        for (Eigen::Index c = 0; c < grid.size(); ++c)
        {
            logPosterior(c) +=
                std::log(triangleGaussianDensityByQuadrature(range.z - (grid.center(c) - range.beacon).norm()));
        }
    }
    double const evidence = std::log(logPosterior.exp().sum());
    Eigen::ArrayXd const expected = (logPosterior - evidence).exp();
    auto const shared = checkPointMassTakesALikelihood<SharedCovariance>(expected, evidence);
    auto const perPoint = checkPointMassTakesALikelihood<PerPointCovariance>(expected, evidence);
    EXPECT_LT((shared - perPoint).abs().maxCoeff(), 1e-14);
}

namespace
{

template <typename Covariance>
void
checkLikelihoodLeavesTheRest()
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-50, -50}, Eigen::Vector2d{5, 5}, {21, 21});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest(), Covariance{}};
    auto const truth = Eigen::Vector2d{17.3, -8.6};
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeRange(rangeFrom(truth, 0, 0.5), beacons[0])));
    auto const before = filter.state();
    auto const range = quantizedRanges(truth)[1];
    auto const result = filter.update(range.measurement());
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->accepted);
    EXPECT_EQ(result->innovationSize(), 0);

    Eigen::ArrayXd joint = Eigen::ArrayXd{before.size()};
    for (Eigen::Index i = 0; i < before.size(); ++i)
    {
        joint(i) = before.logWeights()(i) + range.logLikelihood(before.center(i));
    }
    double const evidence = std::log(joint.exp().sum());
    EXPECT_NEAR(result->logLikelihood(), evidence, 1e-12);

    // The cells whose likelihood underflows to zero are removed; the others keep their Gaussians.
    auto const & after = filter.state();
    EXPECT_EQ(after.size(), joint.isFinite().count());
    EXPECT_LT(after.size(), before.size());
    for (Eigen::Index i = 0; i < after.size(); ++i)
    {
        auto const k = before.find(after.cell(i));
        ASSERT_TRUE(k);
        EXPECT_NEAR(after.logWeights()(i), joint(*k) - evidence, 1e-12);
        EXPECT_EQ(after.conditional(i).mean(), before.conditional(*k).mean());
        EXPECT_EQ(after.conditional(i).covarianceLower(), before.conditional(*k).covarianceLower());
    }
}

}  // namespace

// In a Rao-Blackwellized grid over the position, with a clock bias and drift as the rest, a
// likelihood over the position alone multiplies each cell's probability by the likelihood at its
// center, removes the cells where it underflows to zero, and leaves every other cell's Gaussian
// exactly as it was, under both covariance forms.
TEST(RbPointMassFilter, ALikelihoodOverTheGridLeavesTheRest)
{
    checkLikelihoodLeavesTheRest<SharedCovariance>();
    checkLikelihoodLeavesTheRest<PerPointCovariance>();
}

// A Rao-Blackwellized grid prints its cell count and the dimension on the first line, then the
// lines the collapsed Gaussian prints under its own first line.
TEST(RbPointMassFilter, PrintsItsStateCollapsed)
{
    auto const spacing = Eigen::Vector2d{2, 0.5};
    Eigen::Matrix2d const box = (spacing.array().square() / 12).matrix().asDiagonal();
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-10, 5}, spacing, {3, 4}, box);
    auto const state = GridGaussianParams{grid, initialRest(), SharedCovariance{}};
    std::string const text = printed(state);
    EXPECT_EQ(text.substr(0, text.find('\n')), "Rao-Blackwellized grid of 12 cells of dimension 4");
    EXPECT_EQ(afterFirstLine(text), afterFirstLine(printed(state.toDensity()->collapsed())));
}

// The collapse is the density's (decisions.md [params-and-density]): toDensity() factorizes the
// cells' Gaussians, so a covariance that isn't positive definite fails it, shared or per cell, and
// prints as its error. A filter's step changes the state, so a density made before it is stale
// [view-stamp].
TEST(GridGaussian, ItsDensityChecksTheCells)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-10, 5}, Eigen::Vector2d{2, 0.5}, {3, 4});
    auto const state = GridGaussianParams{grid, initialRest(), SharedCovariance{}};
    auto const density = state.toDensity();
    ASSERT_TRUE(density);
    EXPECT_EQ(density->moments().mean(), density->collapsed().mean());
    EXPECT_EQ(density->size(), 12);

    auto broken = initialRest();
    broken.setCovariance(-broken.covarianceLower());
    auto const shared = GridGaussianParams{grid, broken, SharedCovariance{}};
    EXPECT_EQ(shared.toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
    auto const perCell = GridGaussianParams{grid, broken, PerPointCovariance{}};
    EXPECT_EQ(perCell.toDensity().error(), Emrys::GaussianError::NotPositiveDefinite);
    EXPECT_NE(printed(perCell).find("not positive definite"), std::string::npos) << printed(perCell);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
TEST(GridGaussianDeathTest, ADensityReadAfterAStepFails)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-10, 5}, Eigen::Vector2d{2, 0.5}, {3, 4});
    auto filter = RbPointMassFilter{makeModel(), grid, initialRest()};
    auto const before = filter.state().toDensity();
    ASSERT_TRUE(before);
    EXPECT_DEATH(
        {
            filter.predict(0.1);
            std::ignore = before->collapsed();
        },
        "changed while a view"
    );
}
#endif

namespace
{

struct Steady
{};

struct Wander
{};

// The clock either holds its drift or wanders, and stays as it is with probability 0.9.
struct Clock : Emrys::Named<Steady, Wander>
{};

auto
makeSwitching()
{
    return Emrys::SwitchingModel{
        Emrys::MarkovChain<Clock>::persistent(0.9),
        ConstantVelocity<Bias, Drift>{biasDensity},
        ConstantVelocity<Bias, Drift>{1.0}
    };
}

Emrys::HybridGaussianParams<Rest, Clock>
initialHybrid()
{
    return {Emrys::CategoricalParams<Clock>::fromWeights(Clock{}, Eigen::Vector2d{0.7, 0.3}), initialRest()};
}

// A range clear of obstruction, or delayed by 2 m with variance 4, one component per value of Clock,
// for the noise by mode axis.
Emrys::GaussianMixtureParams<Linear<1>>
delayedNoise()
{
    auto noise = Emrys::GaussianMixtureParams<Linear<1>>{Linear<1>{}};
    noise.add(std::log(0.9), rangeNoise());
    noise.add(std::log(0.1), GaussianParams<Linear<1>>{Vector1{2.0}, Vector1{4.0}});
    return noise;
}

// makeRange() and makeRangeAt() with the delayed noise, by the mode axis Clock.
auto
makeDelayedRange(double z, Eigen::Vector2d const & beacon)
{
    auto const range = Emrys::measurementModel(
        [beacon]<typename T>(auto const & in, auto const &, auto const & x) {
            auto const d = (Emrys::block<Position>(in, x) - beacon.cast<T>()).eval();
            return Eigen::Matrix<T, 1, 1>{Eigen::numext::sqrt(T(d.squaredNorm())) + Emrys::block<Bias>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Bias>),
        additiveNoise(delayedNoise()).byMode<Clock>()
    );
    return observe(methods::linearized(), range, Vector1{z});
}

auto
makeDelayedRangeAt(double z, Eigen::Vector2d const & beacon, Eigen::Vector2d const & p)
{
    auto const range = Emrys::measurementModel(
        [r = (p - beacon).norm()]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 1, 1>{T(r) + Emrys::block<Bias>(in, x)(0)};
        },
        Emrys::reads(Emrys::path<Bias>),
        additiveNoise(delayedNoise()).byMode<Clock>()
    );
    return observe(methods::linearized(), range, Vector1{z});
}

// Whether two hybrids have the same normalized mode probabilities and the same Gaussian per mode.
void
expectSameHybrid(
    Emrys::HybridGaussianParams<Rest, Clock> const & actual,
    Emrys::HybridGaussianParams<Rest, Clock> const & expected,
    double tolerance
)
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

// ImmRecursion's update given a block is the IMM update of the hybrid over the rest with the block
// built into the measurement, plainly and by mode axis.
TEST(RbPointMassFilter, AnImmUpdateGivenABlockMatchesTheUpdateWithTheBlockKnown)
{
    auto recursion = Emrys::ImmRecursion<Gaussian<Rest>, decltype(makeSwitching())>{makeSwitching()};
    auto const p = Eigen::Vector2d{12, -7};
    double const z = 55.0;

    auto viaGiven = initialHybrid();
    auto viaKnown = initialHybrid();
    auto const givenResult = recursion.update(viaGiven, makeRange(z, beacons[0]), given<Position>(State{}, p));
    auto const knownResult = recursion.update(viaKnown, makeRangeAt(z, beacons[0], p));
    ASSERT_TRUE(givenResult && knownResult);
    EXPECT_TRUE(givenResult->accepted);
    EXPECT_NEAR(givenResult->logLikelihood(), knownResult->logLikelihood(), 1e-12);
    expectSameHybrid(viaGiven, viaKnown, 1e-12);

    auto axisGiven = initialHybrid();
    auto axisKnown = initialHybrid();
    auto const axisGivenResult = recursion.update(
        axisGiven,
        makeDelayedRange(z + 1.5, beacons[0]),
        given<Position>(State{}, p),
        reductions::collapse()
    );
    auto const axisKnownResult = recursion.update(axisKnown, makeDelayedRangeAt(z + 1.5, beacons[0], p));
    ASSERT_TRUE(axisGivenResult && axisKnownResult);
    EXPECT_NEAR(axisGivenResult->logLikelihood(), axisKnownResult->logLikelihood(), 1e-12);
    ASSERT_EQ(axisGivenResult->modes.size(), 2u);
    EXPECT_EQ(axisGivenResult->modes[1].branches[0].component, 1);
    expectSameHybrid(axisGiven, axisKnown, 1e-12);
}

// With a switching model, every cell holds a hybrid and is an IMM filter given its center: over six
// steps of ranges, alternately plain and delayed by mode axis, each cell's hybrid, log-likelihood, and
// NIS match an ImmFilter on the rest alone with the cell's position built into the measurement, and
// the cell probabilities match the sums of those filters' log-likelihoods.
TEST(RbPointMassFilter, HybridCellsMatchAnImmFilterPerCell)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto filter = RbPointMassFilter{makeSwitching(), grid, initialHybrid()};
    static_assert(std::is_same_v<decltype(filter)::StateType::CovarianceType, PerPointCovariance>);
    static_assert(std::is_same_v<decltype(filter)::ConditionalType, Emrys::HybridGaussianParams<Rest, Clock>>);
    auto imm = std::vector<decltype(Emrys::ImmFilter{makeSwitching(), initialHybrid()})>{};
    for (Eigen::Index c = 0; c < grid.size(); ++c)
    {
        imm.emplace_back(makeSwitching(), initialHybrid());
    }
    Eigen::ArrayXd logLikelihoods = Eigen::ArrayXd::Zero(grid.size());

    auto const truth = Eigen::Vector2d{1.5, -2.0};
    for (int k = 0; k < 6; ++k)
    {
        std::size_t const b = std::size_t(k) % beacons.size();
        double const z = rangeFrom(truth, b, 1.0 + 0.1 * k) + 0.3 * std::sin(k) + (k == 3 ? 2.5 : 0.0);
        bool const byAxis = k % 2 == 1;
        filter.predict(timeStep);
        auto const result =
            byAxis ? filter.update(makeDelayedRange(z, beacons[b])) : filter.update(makeRange(z, beacons[b]));
        ASSERT_TRUE(result);
        ASSERT_TRUE(result->accepted);
        EXPECT_EQ(result->innovationSize(), 1);
        for (Eigen::Index c = 0; c < grid.size(); ++c)
        {
            auto & cellFilter = imm[std::size_t(c)];
            cellFilter.predict(timeStep);
            auto const cell = byAxis ? cellFilter.update(makeDelayedRangeAt(z, beacons[b], grid.center(c)))
                                     : cellFilter.update(makeRangeAt(z, beacons[b], grid.center(c)));
            ASSERT_TRUE(cell) << c;
            EXPECT_NEAR(result->logLikelihoods()(c), cell->logLikelihood(), 1e-9) << c;
            Eigen::Index mode = 0;
            cellFilter.state().modes().logWeights().maxCoeff(&mode);
            EXPECT_NEAR(result->nis()(c), cell->modes[std::size_t(mode)].branches[0].nis(), 1e-9) << c;
            logLikelihoods(c) += cell->logLikelihood();
        }
    }
    Eigen::ArrayXd const expected =
        (logLikelihoods - logLikelihoods.maxCoeff()).exp() / (logLikelihoods - logLikelihoods.maxCoeff()).exp().sum();
    EXPECT_TRUE(filter.state().toCategorical().logWeights().exp().isApprox(expected, 1e-9));
    for (Eigen::Index c = 0; c < grid.size(); ++c)
    {
        expectSameHybrid(filter.state().conditional(c), imm[std::size_t(c)].state(), 1e-9);
    }
}

// A grid of hybrids sums its cells' modes in modes(), collapses each cell's hybrid in collapsed(),
// deduces PerPointCovariance from a hybrid conditional, and subdivides with each child holding its
// parent's hybrid.
TEST(GridGaussian, HoldsAHybridPerCell)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto filter = RbPointMassFilter{makeSwitching(), grid, initialHybrid()};
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeRange(30.0, beacons[0])));
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeDelayedRange(64.0, beacons[1])));
    auto const & state = filter.state();

    Eigen::Array2d expectedModes = Eigen::Array2d::Zero();
    Eigen::Vector2d restMean = Eigen::Vector2d::Zero();
    for (Eigen::Index i = 0; i < state.size(); ++i)
    {
        auto const & cell = state.conditional(i);
        double const weight = std::exp(state.logWeight(i));
        expectedModes += weight * (cell.modes().logWeights() - cell.modes().logTotalWeight().value()).exp();
        restMean += weight * cell.toDensity()->collapsed().mean();
    }
    auto const modes = state.modes();
    EXPECT_TRUE((modes.logWeights() - modes.logTotalWeight().value()).exp().isApprox(expectedModes, 1e-12));
    EXPECT_NEAR(expectedModes.sum(), 1.0, 1e-12);
    auto const collapsed = state.toDensity()->collapsed();
    EXPECT_TRUE(collapsed.mean().tail<2>().isApprox(restMean, 1e-12));

    auto const fromHybrid = GridGaussianParams{grid, initialHybrid()};
    static_assert(std::is_same_v<decltype(fromHybrid)::CovarianceType, PerPointCovariance>);
    static_assert(std::is_same_v<decltype(fromHybrid)::ConditionalType, Emrys::HybridGaussianParams<Rest, Clock>>);

    auto refined = state;
    EXPECT_TRUE(upkeep::subdivide(2)(refined));
    ASSERT_EQ(refined.size(), 4 * state.size());
    for (Eigen::Index i = 0; i < refined.size(); ++i)
    {
        auto const parent = state.find(state.grid().cells().value([&] {
            auto const child = refined.grid().cells().coordinates(refined.cell(i));
            return std::array<Eigen::Index, 2>{child[0] / 2, child[1] / 2};
        }()));
        ASSERT_TRUE(parent) << i;
        expectSameHybrid(refined.conditional(i), state.conditional(*parent), 0.0);
    }
    EXPECT_TRUE(refined.toDensity()->collapsed().mean().tail<2>().isApprox(restMean, 1e-12));
}

// setState() replaces the state with one over another grid: the steps that follow equal those of a
// filter made on that grid.
template <typename Model, typename Conditional, typename... Covariance>
void
expectSetStateReplacesTheState(Model const & model, Conditional const & conditional, Covariance const &... covariance)
{
    auto const coarse = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{6, 8}, {2, 3});
    auto const fine = regularGrid<State, Position>(Eigen::Vector2d{-4, -6}, Eigen::Vector2d{3, 4}, {4, 5});
    auto filter = RbPointMassFilter{model, coarse, conditional, covariance...};
    auto reference = RbPointMassFilter{model, fine, conditional, covariance...};
    auto const range = makeRange(rangeFrom(Eigen::Vector2d{1.5, -2.0}, 0, 1.0), beacons[0]);
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(range));

    filter.setState(typename decltype(filter)::StateType{fine, conditional});
    ASSERT_EQ(filter.state().size(), fine.size());
    for (auto * const each : {&filter, &reference})
    {
        each->predict(timeStep);
        ASSERT_TRUE(each->update(range));
    }
    EXPECT_TRUE((filter.state().logWeights() == reference.state().logWeights()).all());
    auto const replaced = filter.state().toDensity()->collapsed();
    auto const expected = reference.state().toDensity()->collapsed();
    EXPECT_EQ(replaced.mean(), expected.mean());
    EXPECT_EQ(replaced.covarianceLower(), expected.covarianceLower());
}

// Under both covariance forms and with a hybrid per cell, and on a grid that covers the whole
// state, which has no process model to check.
TEST(RbPointMassFilter, SetStateReplacesTheState)
{
    expectSetStateReplacesTheState(makeModel(), initialRest(), SharedCovariance{});
    expectSetStateReplacesTheState(makeModel(), initialRest(), PerPointCovariance{});
    expectSetStateReplacesTheState(makeSwitching(), initialHybrid());

    using Whole = Composite<Position>;
    auto whole =
        PointMassFilter{regularGrid<Whole, Position>(Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {2, 2})};
    auto const larger = regularGrid<Whole, Position>(Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {3, 3});
    whole.setState(decltype(whole)::StateType{larger, decltype(whole)::ConditionalType{}});
    EXPECT_EQ(whole.state().size(), 9);
    EXPECT_NEAR(whole.state().logWeights().exp().sum(), 1.0, 1e-15);
}

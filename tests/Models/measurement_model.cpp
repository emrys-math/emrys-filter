// Measurement models as functions with reads and writes ([model-policies], [sensors]): a model built by
// measurementModel() from a function or a matrix, observed with observe(method, model, z, args...), updates a
// Gaussian as the textbook step does; it reads blocks wherever they sit in the state, renames one with
// on<Name>(path), takes the call's uncertain and sure inputs by tag, and gives the likelihood at a point, NaN where
// the Jacobian that widens it is not finite.

#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <tuple>
#include <type_traits>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <EmrysFilter/Distributions.h>
#include <EmrysFilter/Models.h>
#include <EmrysFilter/Spaces.h>

using Emrys::additiveNoise;
using Emrys::affineIn;
using Emrys::block;
using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::logLikelihood;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::path;
using Emrys::reads;
using Emrys::ScalarOf;
using Emrys::sure;
using Emrys::update;
using Emrys::writes;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<1>
{};

struct Range : Linear<1>
{};

struct ScaleNoise : Linear<1>
{};

struct Gravity : Linear<2>
{};

using State = Composite<Position, Velocity, Bias>;

using Vector5 = Eigen::Matrix<double, 5, 1>;
using Matrix5 = Eigen::Matrix<double, 5, 5>;
using Vector1 = Eigen::Matrix<double, 1, 1>;
using Matrix1 = Eigen::Matrix<double, 1, 1>;

[[nodiscard]] GaussianParams<State>
makePrior()
{
    Vector5 mean;
    mean << 3.0, 4.0, 0.5, -0.2, 0.1;
    Matrix5 cov = Matrix5::Identity();
    cov(0, 1) = cov(1, 0) = 0.3;
    cov(2, 2) = 0.5;
    cov(4, 4) = 0.04;
    return {State{}, mean, cov};
}

// The range to the origin, a scalar function of Position, returned as the input's scalar: an
// expression of automatic-differentiation scalars would refer to the function's temporaries.
constexpr auto range = []<typename T>(auto const & in, auto const &, auto const & x) {
    return T{block<Position>(in, x).norm()};
};

// The range with its Jacobian over the input, Position alone.
constexpr auto rangeWithJacobian = [](auto const & in, auto const &, auto const & x) {
    auto const p = block<Position>(in, x);
    Eigen::Matrix<double, 1, 2> const jacobian = p.transpose() / p.norm();
    return Emrys::withJacobian(Eigen::Matrix<double, 1, 1>{p.norm()}, jacobian);
};

[[nodiscard]] GaussianParams<Linear<1>>
rangeNoise(double variance = 0.25)
{
    return {Vector1{0.0}, Matrix1{variance}};
}

}  // namespace

// A linear model given its matrix over the blocks read updates as the Kalman filter does on the whole
// state: the gain reaches every coordinate through the cross-covariance with the blocks read.
TEST(MeasurementModel, ALinearModelUpdatesAsTheKalmanFilterDoes)
{
    auto const prior = makePrior();
    Eigen::Matrix2d const h = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d const r = 0.1 * Eigen::Matrix2d::Identity();
    auto const gps = measurementModel(
        h,
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), r})
    );
    Eigen::Vector2d const z{3.5, 3.8};
    auto const updated = update(prior, observe(methods::linearized(), gps, z));
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 1u);

    Eigen::Matrix<double, 2, 5> fullH = Eigen::Matrix<double, 2, 5>::Zero();
    fullH.leftCols<2>() = h;
    Matrix5 const p = prior.covarianceLower().template selfadjointView<Eigen::Lower>();
    Eigen::Matrix2d const s = fullH * p * fullH.transpose() + r;
    Eigen::Matrix<double, 5, 2> const k = p * fullH.transpose() * s.inverse();
    Vector5 const mean = prior.mean() + k * (z - fullH * prior.mean());
    Matrix5 const cov = p - k * s * k.transpose();
    auto const & posterior = updated->branches.front().posterior;
    EXPECT_LT((posterior.mean() - mean).norm(), 1e-12);
    Matrix5 const posteriorCov = posterior.covarianceLower().template selfadjointView<Eigen::Lower>();
    EXPECT_LT((posteriorCov - cov).norm(), 1e-12);
}

// additiveNoise() and on<>() take a covariance as a diagonal, its variances, and hold the dense
// matrix it is, so a model given the diagonal updates exactly as one given that matrix.
TEST(MeasurementModel, NoiseTakesADiagonalCovariance)
{
    auto const variances = Eigen::Vector2d{0.1, 0.4};
    Eigen::Matrix2d const r = variances.asDiagonal();
    auto const diagonal = additiveNoise(variances.asDiagonal());
    using Noise = std::remove_cvref_t<decltype(diagonal)>;
    static_assert(Noise::ZeroMean);
    static_assert(std::is_same_v<Noise::Distribution, Eigen::Matrix2d>);
    EXPECT_EQ(diagonal.distribution(), r);
    auto const placed = additiveNoise(on<Position>(variances.asDiagonal()));
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(std::get<0>(placed.parts()).value)>, Eigen::Matrix2d>);

    auto const prior = makePrior();
    Eigen::Vector2d const z{3.5, 3.8};
    auto const dense = measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(r));
    auto const fromDiagonal =
        measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(variances.asDiagonal()));
    auto const expected = update(prior, observe(dense, z));
    auto const actual = update(prior, observe(fromDiagonal, z));
    ASSERT_TRUE(expected && actual);
    EXPECT_EQ(actual->branches.front().posterior.mean(), expected->branches.front().posterior.mean());
    EXPECT_EQ(
        actual->branches.front().posterior.covarianceLower(),
        expected->branches.front().posterior.covarianceLower()
    );
}

// A function generic in its scalar is differentiated; one that returns its Jacobian is not, and the two
// updates agree. A scalar return stands for a measurement of one coordinate.
TEST(MeasurementModel, AReturnedJacobianMatchesAutoDiff)
{
    auto const prior = makePrior();
    auto const byAutoDiff = measurementModel(range, reads(path<Position>), additiveNoise(rangeNoise()));
    auto const byJacobian = measurementModel(rangeWithJacobian, reads(path<Position>), additiveNoise(rangeNoise()));
    Vector1 const z{5.2};
    auto const a = update(prior, observe(methods::linearized(), byAutoDiff, z));
    auto const b = update(prior, observe(methods::linearized(), byJacobian, z));
    ASSERT_TRUE(a && b);
    EXPECT_LT((a->branches.front().posterior.mean() - b->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (a->branches.front().posterior.covarianceLower() - b->branches.front().posterior.covarianceLower()).norm(),
        1e-12
    );
    // The posterior moved toward the observation along the range direction and left Bias alone.
    EXPECT_GT(a->branches.front().posterior.mean().head<2>().norm(), 5.0);
    EXPECT_NEAR(a->branches.front().posterior.mean()(4), prior.mean()(4), 1e-15);
}

// writes() names the output's block, and the function fills it by name in zeros<T>(out).
TEST(MeasurementModel, WritesNamesTheOutput)
{
    auto const prior = makePrior();
    auto const named = measurementModel(
        []<typename T>(auto const & in, auto const & out, auto const & x) {
            auto y = Emrys::zeros<T>(out);
            block<Range>(out, y) = Eigen::Matrix<T, 1, 1>{block<Position>(in, x).norm()};
            return y;
        },
        reads(path<Position>),
        writes(path<Range>),
        additiveNoise(on<Range>(rangeNoise()))
    );
    auto const plain = measurementModel(range, reads(path<Position>), additiveNoise(rangeNoise()));
    Vector1 const z{5.2};
    auto const a = update(prior, observe(methods::unscented(), named, z));
    auto const b = update(prior, observe(methods::unscented(), plain, z));
    ASSERT_TRUE(a && b);
    EXPECT_LT((a->branches.front().posterior.mean() - b->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_EQ(a->branches.front().innovation.residual().size(), 1);
}

// An uncertain input of the call passes through the function by its tag, and where the model adds it, the
// update equals the one with the same noise declared additive; a sure input is a point with no spread.
TEST(MeasurementModel, InputsAreNamedByTag)
{
    auto const prior = makePrior();
    auto const scaled = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Position>(in, x).norm() + block<ScaleNoise>(in, x)(0)};
        },
        reads(path<Position>, path<ScaleNoise>),
        affineIn(path<ScaleNoise>)
    );
    auto const additive = measurementModel(range, reads(path<Position>), additiveNoise(rangeNoise()));
    Vector1 const z{5.2};
    auto const noise = GaussianParams<ScaleNoise>{Vector1{0.0}, Matrix1{0.25}};
    auto const a = update(prior, observe(methods::linearized(), scaled, z, noise));
    auto const b = update(prior, observe(methods::linearized(), additive, z));
    ASSERT_TRUE(a && b);
    EXPECT_LT((a->branches.front().posterior.mean() - b->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (a->branches.front().posterior.covarianceLower() - b->branches.front().posterior.covarianceLower()).norm(),
        1e-12
    );
    // The same noise known exactly at zero shifts nothing and adds no spread: the no-noise update.
    auto const exact = update(prior, observe(methods::linearized(), scaled, z, sure<ScaleNoise>(Vector1{0.0})));
    ASSERT_TRUE(exact);
    EXPECT_LT(
        exact->branches.front().innovation.covarianceLower()(0, 0),
        a->branches.front().innovation.covarianceLower()(0, 0) - 0.2
    );
    // A sigma-point method spreads its points over the input too until the affine declaration factors
    // it out (roadmap 17z, the Jacobians item), so it is checked there, not here.
}

// A model reads blocks wherever they sit: on a two-vehicle state, path<Ownship, Position> read under the
// name SensorPosition and path<Target, Position> under its own, and the function sees only the names.
TEST(MeasurementModel, ReadsRenameABlockByItsName)
{
    struct SensorPosition : Linear<2>
    {};

    struct Vehicle : Composite<Position, Velocity>
    {};

    struct Ownship : Vehicle
    {};

    struct Target : Vehicle
    {};

    using Pair = Composite<Ownship, Target>;
    auto const relative = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{(block<Position>(in, x) - block<SensorPosition>(in, x)).norm()};
        },
        reads(on<SensorPosition>(path<Ownship, Position>), path<Target, Position>),
        additiveNoise(rangeNoise(0.01))
    );
    using Vector8 = Eigen::Matrix<double, 8, 1>;
    Vector8 mean;
    mean << 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 1.0;
    auto const prior = GaussianParams<Pair>{Pair{}, mean, Eigen::Matrix<double, 8, 8>::Identity()};
    auto const updated = update(prior, observe(methods::linearized(), relative, Vector1{5.5}));
    ASSERT_TRUE(updated);
    EXPECT_NEAR(updated->branches.front().innovation.residual()(0), 0.5, 1e-12);
    // Both positions moved apart along the line between them, and the velocities did not.
    auto const & posterior = updated->branches.front().posterior.mean();
    EXPECT_LT(posterior(0), 0.0);
    EXPECT_GT(posterior(4), 3.0);
    EXPECT_NEAR(posterior(2), 1.0, 1e-12);
}

// Two items of reads() that would share a name are refused at compile time; here the check that the
// function's input space is the Composite of the names, in order.
TEST(MeasurementModel, TheInputSpaceIsTheCompositeOfTheNames)
{
    auto const model = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            using In = std::remove_cvref_t<decltype(in)>;
            static_assert(std::is_same_v<In, Composite<Bias, Position>>);
            return T{block<Position>(in, x).norm() + block<Bias>(in, x)(0)};
        },
        reads(path<Bias>, path<Position>),
        additiveNoise(rangeNoise())
    );
    auto const prior = makePrior();
    auto const updated = update(prior, observe(methods::linearized(), model, Vector1{5.1}));
    ASSERT_TRUE(updated);
    EXPECT_NEAR(updated->branches.front().innovation.residual()(0), 5.1 - 5.0 - 0.1, 1e-12);
}

// The likelihood at a point: the additive noise's density at the residual, the model's own
// logLikelihood(g) when it has one, and for an affine input the density widened by the input's spread.
TEST(MeasurementModel, TheLikelihoodAtAPoint)
{
    Eigen::Matrix<double, 5, 2> points;
    points.col(0) << 3.0, 4.0, 0.0, 0.0, 0.0;
    points.col(1) << 6.0, 8.0, 0.0, 0.0, 0.0;
    Vector1 const z{6.0};
    auto const additive = measurementModel(range, reads(path<Position>), additiveNoise(rangeNoise()));
    Eigen::ArrayXd const direct = Emrys::logLikelihoods(observe(additive, z), State{}, points);
    for (Eigen::Index j = 0; j < 2; ++j)
    {
        double const residual = 6.0 - points.col(j).head<2>().norm();
        double const expected = -0.5 * (std::log(2 * std::numbers::pi) + std::log(0.25) + residual * residual / 0.25);
        EXPECT_NEAR(direct(j), expected, 1e-12) << j;
    }
    // The same density written as the model's own likelihood, a function of the input, the point, and z.
    auto const own = measurementModel(
        reads(path<Position>),
        logLikelihood([](auto const & in, auto const & x, auto const & observed) {
            double const residual = observed(0) - block<Position>(in, x).norm();
            return -0.5 * (std::log(2 * std::numbers::pi) + std::log(0.25) + residual * residual / 0.25);
        })
    );
    Eigen::ArrayXd const byOwn = Emrys::logLikelihoods(observe(own, z), State{}, points);
    EXPECT_LT((byOwn - direct).abs().maxCoeff(), 1e-12);
    // Noise through the function, affine, with no additive noise: the same density.
    auto const scaled = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Position>(in, x).norm() + block<ScaleNoise>(in, x)(0)};
        },
        reads(path<Position>, path<ScaleNoise>),
        affineIn(path<ScaleNoise>),
        additiveNoise(rangeNoise(0.0))
    );
    auto const noise = GaussianParams<ScaleNoise>{Vector1{0.0}, Matrix1{0.25}};
    Eigen::ArrayXd const byInput = Emrys::logLikelihoods(observe(scaled, z, noise), State{}, points);
    EXPECT_LT((byInput - direct).abs().maxCoeff(), 1e-12);
}

// The likelihood at a point where the Jacobian in an affine input is not finite is NaN: a range offset by an uncertain
// sensor position has the Jacobian 0/0 at the sensor, where the widening is not defined, and elsewhere the additive
// noise widened by the offset's variance along the line of sight.
TEST(MeasurementModel, TheLikelihoodIsNaNWhereTheWideningIsNot)
{
    struct Offset : Linear<2>
    {};

    Eigen::Matrix<double, 5, 2> points;
    points.col(0) << 0.0, 0.0, 0.0, 0.0, 0.0;
    points.col(1) << 3.0, 4.0, 0.0, 0.0, 0.0;
    auto const offsetRange = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{(block<Position>(in, x) + block<Offset>(in, x)).norm()};
        },
        reads(path<Position>, path<Offset>),
        affineIn(path<Offset>),
        additiveNoise(rangeNoise())
    );
    auto const offset = GaussianParams<Offset>{Eigen::Vector2d::Zero(), 0.09 * Eigen::Matrix2d::Identity()};
    Eigen::ArrayXd const result = Emrys::logLikelihoods(observe(offsetRange, Vector1{6.0}, offset), State{}, points);
    EXPECT_TRUE(std::isnan(result(0)));
    double const variance = 0.25 + 0.09;
    double const expected = -0.5 * (std::log(2 * std::numbers::pi) + std::log(variance) + 1.0 / variance);
    EXPECT_NEAR(result(1), expected, 1e-12);
}

// simulate() draws the inputs and the noise in a fixed order, so a seeded draw is reproducible and its
// residual from the noiseless measurement is the draw.
TEST(MeasurementModel, SimulateDrawsTheMeasurement)
{
    auto const additive = measurementModel(range, reads(path<Position>), additiveNoise(rangeNoise()));
    Vector5 truth;
    truth << 3.0, 4.0, 0.0, 0.0, 0.0;
    auto rng = std::mt19937_64{7};
    auto const drawn = Emrys::simulate(State{}, truth, additive, rng);
    static_assert(std::is_same_v<decltype(drawn), Eigen::Matrix<double, 1, 1> const>);
    auto again = std::mt19937_64{7};
    EXPECT_EQ(drawn, Emrys::simulate(State{}, truth, additive, again));
    EXPECT_LT(std::abs(drawn(0) - 5.0), 5.0);
    // A sure input reaches the function as given, and the sure gravity shifts the draw by itself.
    auto const shifted = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Position>(in, x).norm() + block<Gravity>(in, x)(1)};
        },
        reads(path<Position>, path<Gravity>),
        additiveNoise(rangeNoise())
    );
    auto once = std::mt19937_64{7};
    auto const withGravity = Emrys::simulate(State{}, truth, shifted, once, sure<Gravity>(Eigen::Vector2d{0.0, -9.8}));
    EXPECT_NEAR(withGravity(0), drawn(0) - 9.8, 1e-12);
}

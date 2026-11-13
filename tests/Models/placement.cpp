// Models placed in a larger state, and measurement models side by side ([model-placement]): two
// placements of one vehicle model, one with its input renamed, predict a two-vehicle state as the model written out
// by hand does; a fleet placed by onEach() with an input per repetition does too; and compose(range, bearing) updates
// as one radar model written as a function.

#include <cmath>
#include <tuple>
#include <type_traits>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <EmrysFilter/Distributions.h>
#include <EmrysFilter/Models.h>
#include <EmrysFilter/Spaces.h>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::compose;
using Emrys::Composite;
using Emrys::continuousNoise;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::onEach;
using Emrys::path;
using Emrys::processModel;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::ScalarOf;
using Emrys::sure;
using Emrys::update;
using Emrys::writes;
using Emrys::zeros;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Bias : Linear<2>
{};

struct SpecificForce : Linear<2>
{};

struct TargetForce : Linear<2>
{};

struct Vehicle : Composite<Position, Velocity, Bias>
{};

struct Ownship : Vehicle
{};

struct Target : Vehicle
{};

struct Targets : Repeated<Vehicle, 4>
{
    using Repeated::Repeated;
};

struct TargetForces : Repeated<SpecificForce, 4>
{
    using Repeated::Repeated;
};

using Pair = Composite<Ownship, Target>;
using Fleet = Composite<Ownship, Targets>;

constexpr double timeStep = 0.1;
constexpr double velocityDensity = 0.3;
constexpr double biasDensity = 0.01;

// The kinematics of one vehicle driven by a measured specific force less its bias.
[[nodiscard]] auto
kinematics()
{
    return processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto const v = block<Velocity>(in, x);
            auto const a = block<SpecificForce>(in, x) - block<Bias>(in, x);
            auto y = zeros<T>(out);
            block<Position>(out, y) = block<Position>(in, x) + dt * v + (dt * dt / 2) * a;
            block<Velocity>(out, y) = v + dt * a;
            return y;
        },
        reads(path<Position>, path<Velocity>, path<Bias>, path<SpecificForce>),
        writes(path<Position>, path<Velocity>),
        continuousNoise(on<Velocity>(velocityDensity))
    );
}

[[nodiscard]] auto
vehicle()
{
    return compose(kinematics(), RandomWalk<Bias>{biasDensity});
}

// One vehicle's step at its blocks (position, velocity, bias) and force, as a vector of six.
template <typename P, typename V, typename B, typename F>
[[nodiscard]] auto
vehicleStep(P const & p, V const & v, B const & b, F const & f, double dt)
{
    using T = ScalarOf<P>;
    Eigen::Matrix<T, 6, 1> y;
    auto const a = (f - b).eval();
    y.template segment<2>(0) = p + dt * v + (dt * dt / 2) * a;
    y.template segment<2>(2) = v + dt * a;
    y.template segment<2>(4) = b;
    return y;
}

// One vehicle's process noise over a step, over its six differences.
[[nodiscard]] Eigen::Matrix<double, 6, 6>
vehicleNoise(double dt)
{
    Eigen::Matrix<double, 6, 6> q = Eigen::Matrix<double, 6, 6>::Zero();
    q.block<2, 2>(2, 2).diagonal().setConstant(velocityDensity * dt);
    q.block<2, 2>(4, 4).diagonal().setConstant(biasDensity * dt);
    return q;
}

// The block-diagonal matrix of \a count copies of \a block.
[[nodiscard]] Eigen::MatrixXd
blockDiagonal(Eigen::Matrix<double, 6, 6> const & block, Eigen::Index count)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(6 * count, 6 * count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        result.block<6, 6>(6 * i, 6 * i) = block;
    }
    return result;
}

[[nodiscard]] Eigen::VectorXd
statePoint(Eigen::Index size)
{
    Eigen::VectorXd mean(size);
    for (Eigen::Index i = 0; i < size; ++i)
    {
        mean(i) = 0.3 * double(i) - 0.7 * std::sin(double(i));
    }
    return mean;
}

[[nodiscard]] Eigen::MatrixXd
stateCovariance(Eigen::Index size)
{
    Eigen::MatrixXd root = Eigen::MatrixXd::Identity(size, size);
    for (Eigen::Index i = 0; i + 1 < size; ++i)
    {
        root(i + 1, i) = 0.1;
    }
    return root * root.transpose();
}

}  // namespace

// Two placements of one vehicle model, the target's force renamed and uncertain, predict as the model written out
// by hand over the pair does, bitwise: the placement changes where the model reads, never what it computes.
TEST(Placement, TwoPlacementsOfOneModelMatchTheModelWrittenOut)
{
    auto const pair = compose(on<Ownship>(vehicle()), on<Target>(vehicle(), on<TargetForce>(path<SpecificForce>)));
    auto const byHand = processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = zeros<T>(out);
            auto const own = block<Ownship>(in, x);
            auto const target = block<Target>(in, x);
            y.template segment<6>(0) = vehicleStep(
                own.template segment<2>(0),
                own.template segment<2>(2),
                own.template segment<2>(4),
                block<SpecificForce>(in, x),
                dt
            );
            y.template segment<6>(6) = vehicleStep(
                target.template segment<2>(0),
                target.template segment<2>(2),
                target.template segment<2>(4),
                block<TargetForce>(in, x),
                dt
            );
            return y;
        },
        reads(path<Ownship>, path<Target>, path<SpecificForce>, path<TargetForce>),
        writes(path<Ownship>, path<Target>),
        continuousNoise(
            on<Ownship>(Eigen::Matrix<double, 6, 6>{vehicleNoise(1.0)}),
            on<Target>(Eigen::Matrix<double, 6, 6>{vehicleNoise(1.0)})
        )
    );
    auto const prior = GaussianParams<Pair>{Pair{}, statePoint(12), stateCovariance(12)};
    Eigen::Vector2d const ownForce{0.4, -0.1};
    auto const targetForce = GaussianParams<TargetForce>{Eigen::Vector2d{0.2, 0.3}, 0.05 * Eigen::Matrix2d::Identity()};

    auto const placed =
        Emrys::predict(methods::linearized(), prior, pair, timeStep, sure<SpecificForce>(ownForce), targetForce);
    auto const written =
        Emrys::predict(methods::linearized(), prior, byHand, timeStep, sure<SpecificForce>(ownForce), targetForce);
    EXPECT_EQ(placed.mean(), written.mean());
    EXPECT_EQ(placed.covarianceLower(), written.covarianceLower());

    auto const placedUnscented =
        Emrys::predict(methods::unscented(), prior, pair, timeStep, sure<SpecificForce>(ownForce), targetForce);
    auto const writtenUnscented =
        Emrys::predict(methods::unscented(), prior, byHand, timeStep, sure<SpecificForce>(ownForce), targetForce);
    EXPECT_LT((placedUnscented.mean() - writtenUnscented.mean()).norm(), 1e-12);
    EXPECT_LT((placedUnscented.covarianceLower() - writtenUnscented.covarianceLower()).norm(), 1e-12);
}

// A placement without a rename shares the input with the other placement: both vehicles read one force.
TEST(Placement, AnInputNotRenamedIsShared)
{
    auto const pair = compose(on<Ownship>(vehicle()), on<Target>(vehicle()));
    Eigen::VectorXd const x = statePoint(12);
    Eigen::Vector2d const force{0.4, -0.1};
    auto const y = Emrys::propagate(Pair{}, x, pair, timeStep, sure<SpecificForce>(force));
    Eigen::VectorXd expected(12);
    expected << vehicleStep(x.segment<2>(0).eval(), x.segment<2>(2).eval(), x.segment<2>(4).eval(), force, timeStep),
        vehicleStep(x.segment<2>(6).eval(), x.segment<2>(8).eval(), x.segment<2>(10).eval(), force, timeStep);
    EXPECT_EQ(y, expected);
}

// A fleet placed by onEach(), each target with its own force from one argument over a Repeated tag, predicts as the
// model written out by hand does.
TEST(Placement, AFleetByOnEachMatchesTheModelWrittenOut)
{
    constexpr Eigen::Index count = 3;
    auto const fleet =
        compose(on<Ownship>(vehicle()), onEach<Targets>(vehicle(), on<TargetForces>(path<SpecificForce>)));
    auto const byHand = processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = zeros<T>(out);
            auto const own = block<Ownship>(in, x);
            y.template segment<6>(0) = vehicleStep(
                own.template segment<2>(0),
                own.template segment<2>(2),
                own.template segment<2>(4),
                block<SpecificForce>(in, x),
                dt
            );
            auto const targets = block<Targets>(in, x);
            auto const forces = block<TargetForces>(in, x);
            for (Eigen::Index i = 0; i < targets.size() / 6; ++i)
            {
                y.segment(6 + 6 * i, 6) = vehicleStep(
                    targets.template segment<2>(6 * i),
                    targets.template segment<2>(6 * i + 2),
                    targets.template segment<2>(6 * i + 4),
                    forces.template segment<2>(2 * i),
                    dt
                );
            }
            return y;
        },
        reads(path<Ownship>, path<Targets>, path<SpecificForce>, path<TargetForces>),
        writes(path<Ownship>, path<Targets>),
        continuousNoise(
            on<Ownship>(Eigen::Matrix<double, 6, 6>{vehicleNoise(1.0)}),
            on<Targets>(blockDiagonal(vehicleNoise(1.0), count))
        )
    );
    auto const space = Fleet{Ownship{}, Targets{Vehicle{}, count}};
    Eigen::Index const n = 6 + 6 * count;
    auto const prior = GaussianParams<Fleet>{space, statePoint(n), stateCovariance(n)};
    Eigen::Vector2d const ownForce{0.4, -0.1};
    Eigen::VectorXd forces(2 * count);
    forces << 0.2, 0.3, -0.1, 0.0, 0.5, -0.4;
    auto const targetForces = GaussianParams<TargetForces>{
        TargetForces{SpecificForce{}, count},
        forces,
        0.05 * Eigen::MatrixXd::Identity(2 * count, 2 * count)
    };

    auto const placed =
        Emrys::predict(methods::linearized(), prior, fleet, timeStep, sure<SpecificForce>(ownForce), targetForces);
    auto const written =
        Emrys::predict(methods::linearized(), prior, byHand, timeStep, sure<SpecificForce>(ownForce), targetForces);
    EXPECT_LT((placed.mean() - written.mean()).norm(), 1e-12);
    EXPECT_LT((placed.covarianceLower() - written.covarianceLower()).norm(), 1e-12);
}

namespace
{

struct Range : Linear<1>
{};

struct Bearing : Linear<1>
{};

using Plane = Composite<Position, Velocity>;

}  // namespace

// compose(range, bearing) measures Composite<Range, Bearing> and updates as one radar model written as a function
// over both, with the noise of each on its block.
TEST(Placement, ComposedMeasurementsMatchOneModel)
{
    auto const range = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{block<Position>(in, x).norm()};
        },
        reads(path<Position>),
        writes(path<Range>),
        additiveNoise(GaussianParams<Range>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.04}})
    );
    auto const bearing = measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            auto const p = block<Position>(in, x);
            using std::atan2;
            return T{atan2(p(1), p(0))};
        },
        reads(path<Position>),
        writes(path<Bearing>),
        additiveNoise(
            on<Bearing>(GaussianParams<Bearing>{Eigen::Matrix<double, 1, 1>::Zero(), Eigen::Matrix<double, 1, 1>{0.01}})
        )
    );
    auto const radar = measurementModel(
        []<typename T>(auto const & in, auto const & out, auto const & x) {
            auto const p = block<Position>(in, x);
            auto y = zeros<T>(out);
            using std::atan2;
            block<Range>(out, y)(0) = p.norm();
            block<Bearing>(out, y)(0) = atan2(p(1), p(0));
            return y;
        },
        reads(path<Position>),
        writes(path<Range>, path<Bearing>),
        additiveNoise(
            Emrys::GaussianParams<Composite<Range, Bearing>>{
                Eigen::Vector2d::Zero(),
                Eigen::Vector2d{0.04, 0.01}.asDiagonal().toDenseMatrix()
            }
        )
    );
    Eigen::Vector4d mean{3.0, 4.0, 0.5, -0.2};
    auto const prior = GaussianParams<Plane>{Plane{}, mean, stateCovariance(4)};
    Eigen::Vector2d const z{5.1, 0.95};

    auto const composed = update(prior, observe(methods::unscented(), compose(range, bearing), z));
    auto const one = update(prior, observe(methods::unscented(), radar, z));
    ASSERT_TRUE(composed && one);
    EXPECT_LT((composed->branches.front().posterior.mean() - one->branches.front().posterior.mean()).norm(), 1e-12);
    EXPECT_LT(
        (composed->branches.front().posterior.covarianceLower() - one->branches.front().posterior.covarianceLower())
            .norm(),
        1e-12
    );

    auto const composedLinearized = update(prior, observe(methods::linearized(), compose(range, bearing), z));
    auto const oneLinearized = update(prior, observe(methods::linearized(), radar, z));
    ASSERT_TRUE(composedLinearized && oneLinearized);
    EXPECT_LT(
        (composedLinearized->branches.front().posterior.mean() - oneLinearized->branches.front().posterior.mean())
            .norm(),
        1e-12
    );
}

// A measurement model placed on a block reads the block's paths under the prefix.
TEST(Placement, APlacedMeasurementModelReadsUnderThePrefix)
{
    auto const gps = measurementModel(
        Eigen::Matrix2d::Identity(),
        reads(path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 0.1 * Eigen::Matrix2d::Identity()})
    );
    auto const prior = GaussianParams<Pair>{Pair{}, statePoint(12), stateCovariance(12)};
    Eigen::Vector2d const z{0.5, 0.2};
    auto const placed = update(prior, observe(methods::linearized(), on<Target>(gps), z));
    auto const direct = update(
        prior,
        observe(
            methods::linearized(),
            measurementModel(
                Eigen::Matrix2d::Identity(),
                reads(path<Target, Position>),
                additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 0.1 * Eigen::Matrix2d::Identity()})
            ),
            z
        )
    );
    ASSERT_TRUE(placed && direct);
    EXPECT_EQ(placed->branches.front().posterior.mean(), direct->branches.front().posterior.mean());
    EXPECT_EQ(
        placed->branches.front().posterior.covarianceLower(),
        direct->branches.front().posterior.covarianceLower()
    );
}

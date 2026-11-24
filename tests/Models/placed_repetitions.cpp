// Models placed on repetitions of a repeated block by a path value (roadmap step 17zu, decisions.md
// [placed-repetitions], [model-placement]): on(path<Fleet>[k], model) on one repetition, and
// onEach(path<Fleet>[repetitions], model) on each of a sequence or a list. A state written with one tag per
// repetition, Composite<First, Second, Third>, has the coordinates of a Fleet of three vehicles, in the same
// order, so a model placed on a repetition must predict or update as the model placed on its tagged block;
// the two differ only in the sizes Eigen fixes at compile time, hence the tolerance. Also covered: models
// placed on one repetition that each write a block of it, an input with a value per repetition selected,
// and the selections the placed model's policies hold.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <type_traits>

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
using Emrys::reads;
using Emrys::Repeated;
using Emrys::writes;
using Emrys::zeros;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Vehicle : Composite<Position, Velocity>
{};

struct Fleet : Repeated<Vehicle, 5>
{
    using Repeated::Repeated;
};

struct First : Vehicle
{};

struct Second : Vehicle
{};

struct Third : Vehicle
{};

/// the coordinates of a Fleet of three, a tag per vehicle
using Tagged = Composite<First, Second, Third>;

/// the push a vehicle reads as an input, a value per vehicle pushed, and the names of the tagged ones
struct Push : Linear<2>
{};

struct Pushes : Repeated<Push, 5>
{
    using Repeated::Repeated;
};

struct FirstPush : Linear<2>
{};

struct ThirdPush : Linear<2>
{};

/// what a fix of one vehicle measures: its position and its speed
struct Fix : Linear<3>
{};

using Vector12 = Eigen::Matrix<double, 12, 1>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;

// Three vehicles, each a position and a velocity, every coordinate distinct.
Vector12
fleetMean()
{
    Vector12 mean;
    mean << 1.0, 2.0, 0.3, -0.4, -3.0, 8.0, 1.1, 0.2, 6.0, -7.0, -0.6, 0.9;
    return mean;
}

// A covariance with every block correlated with every other: A A^T + I / 2 for a fixed full A.
Matrix12
fleetCovariance()
{
    Matrix12 a;
    for (Eigen::Index i = 0; i < 12; ++i)
    {
        for (Eigen::Index j = 0; j < 12; ++j)
        {
            a(i, j) = 0.3 * std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
        }
    }
    return a * a.transpose() + 0.5 * Matrix12::Identity();
}

GaussianParams<Fleet>
fleetPrior()
{
    return {Fleet{Vehicle{}, 3}, fleetMean(), fleetCovariance()};
}

GaussianParams<Tagged>
taggedPrior()
{
    return {fleetMean(), fleetCovariance()};
}

// One vehicle, its position and velocity, turns its velocity by a quarter of the step and moves by it.
auto const turn = []<typename T>(auto const & x, double dt) {
    T const c = T(std::cos(0.25 * dt));
    T const s = T(std::sin(0.25 * dt));
    Eigen::Vector<T, 2> const v = x.template segment<2>(2);
    Eigen::Vector<T, 2> const turned{c * v(0) - s * v(1), s * v(0) + c * v(1)};
    Eigen::Vector<T, 4> y;
    y << x.template head<2>() + dt * turned + T(0.1) * v.cwiseProduct(v), turned;
    return y;
};

// A vehicle that turns, with spectral density 0.3 on its velocity.
auto
turning()
{
    return processModel(
        turn,
        reads(path<Position>, path<Velocity>),
        writes(path<Position>, path<Velocity>),
        continuousNoise(on<Velocity>(0.3))
    );
}

// A vehicle that holds its course, with spectral density 0.05 on its velocity.
auto
cruising()
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            Eigen::Vector<T, 4> y;
            y << x.template head<2>() + dt * x.template tail<2>(), x.template tail<2>();
            return y;
        },
        reads(path<Position>, path<Velocity>),
        writes(path<Position>, path<Velocity>),
        continuousNoise(on<Velocity>(0.05))
    );
}

// A vehicle pushed by the input Push, which moves its velocity and, through the square of its
// velocity, its position.
auto
pushed()
{
    return processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = zeros<T>(out);
            auto const v = block<Velocity>(in, x);
            block<Position>(out, y) = block<Position>(in, x) + dt * v + T(0.1) * v.cwiseProduct(v);
            block<Velocity>(out, y) = v + dt * block<Push>(in, x);
            return y;
        },
        reads(path<Position>, path<Velocity>, path<Push>),
        writes(path<Position>, path<Velocity>),
        continuousNoise(on<Velocity>(0.2))
    );
}

// A velocity that turns by a quarter of the step, written alone.
auto
spinning()
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            T const c = T(std::cos(0.25 * dt));
            T const s = T(std::sin(0.25 * dt));
            return Eigen::Vector<T, 2>{c * x(0) - s * x(1), s * x(0) + c * x(1)};
        },
        reads(path<Velocity>),
        writes(path<Velocity>),
        continuousNoise(0.3)
    );
}

// A position that moves by the velocity, written alone.
auto
moving()
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            return Eigen::Vector<T, 2>{
                x.template head<2>() + dt * x.template tail<2>()
                + T(0.1) * x.template tail<2>().cwiseProduct(x.template tail<2>())
            };
        },
        reads(path<Position>, path<Velocity>),
        writes(path<Position>),
        continuousNoise(0.05)
    );
}

// A fix of one vehicle: its position and its speed.
auto
fixOf()
{
    return measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            auto const v = block<Velocity>(in, x);
            Eigen::Vector<T, 3> z;
            z << block<Position>(in, x), T(Eigen::numext::sqrt(T(v.squaredNorm())));
            return z;
        },
        reads(path<Position>, path<Velocity>),
        writes(path<Fix>),
        additiveNoise(GaussianParams<Fix>{Eigen::Vector3d::Zero(), Eigen::Vector3d{0.04, 0.04, 0.01}.asDiagonal()})
    );
}

// Two predictions agree in their means and in their covariances' lower triangles, and the
// prediction moved the state from its prior mean.
template <typename Selected, typename ByTag>
void
expectSamePrediction(Selected const & selected, ByTag const & byTag)
{
    EXPECT_LT((selected.mean() - byTag.mean()).norm(), 1e-12);
    Eigen::MatrixXd const lowerA = selected.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = byTag.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
    EXPECT_GT((selected.mean() - fleetMean()).norm(), 1e-3);
}

// The prediction of \a selected on the fleet against that of \a byTag on the tagged state, under
// linearization and two sigma-point methods, with the inputs \a inputs and \a taggedInputs.
template <typename Selected, typename ByTag, typename... Inputs, typename... TaggedInputs>
void
expectSamePredictions(
    Selected const & selected,
    ByTag const & byTag,
    std::tuple<Inputs...> const & inputs = {},
    std::tuple<TaggedInputs...> const & taggedInputs = {}
)
{
    auto const predictBoth = [&](auto const & method) {
        expectSamePrediction(
            std::apply(
                [&](auto const &... in) {
                    return Emrys::predict(method, fleetPrior(), selected, 0.5, in...);
                },
                inputs
            ),
            std::apply(
                [&](auto const &... in) {
                    return Emrys::predict(method, taggedPrior(), byTag, 0.5, in...);
                },
                taggedInputs
            )
        );
    };
    predictBoth(methods::linearized());
    predictBoth(methods::unscented());
    predictBoth(methods::cubature());
}

}  // namespace

// A model placed on one repetition reads and writes each of its blocks through the repetition,
// under a name of its own, and holds the repetition in each item.
TEST(PlacedRepetitions, OnOneRepetitionHoldsIt)
{
    auto const placed = on(path<Fleet>[1], turning());
    auto const & placedReads = placed.policy<Emrys::internal::PolicyKind::reads>();
    using Item = std::remove_cvref_t<decltype(placedReads)>::Item<0>;
    static_assert(std::is_same_v<Item::PathType, Emrys::TagPath<Fleet, Position>>);
    static_assert(std::is_same_v<Item::SelectionType, Eigen::Index>);
    EXPECT_EQ(placedReads.selection<0>(), 1);
    EXPECT_EQ(placedReads.selection<1>(), 1);
    auto const & placedWrites = placed.policy<Emrys::internal::PolicyKind::writes>();
    EXPECT_EQ(placedWrites.selection<0>(), 1);
    EXPECT_EQ(placedWrites.selection<1>(), 1);
}

// A model placed on vehicle 1 beside a part that carries the others predicts as the model placed
// on the tagged block Second.
TEST(PlacedRepetitions, OnOneRepetitionIsOnItsTaggedBlock)
{
    expectSamePredictions(
        compose(on(path<Fleet>[1], turning()), onEach(path<Fleet>[std::array{0, 2}], cruising())),
        compose(on<First>(cruising()), on<Second>(turning()), on<Third>(cruising()))
    );
}

// A model placed on each of a list of vehicles, in an order that is not the state's, and on each of
// a sequence, predicts as the model placed on each tagged block.
TEST(PlacedRepetitions, OnEachOfASelectionIsOnEachTaggedBlock)
{
    expectSamePredictions(
        compose(onEach(path<Fleet>[std::array{2, 0}], turning()), on(path<Fleet>[1], cruising())),
        compose(on<First>(turning()), on<Second>(cruising()), on<Third>(turning()))
    );
    expectSamePredictions(
        compose(onEach(path<Fleet>[Eigen::seqN(1, 2)], turning()), on(path<Fleet>[0], cruising())),
        compose(on<First>(cruising()), on<Second>(turning()), on<Third>(turning()))
    );
    expectSamePredictions(
        compose(onEach(path<Fleet>[Eigen::placeholders::lastN(2)], turning()), on(path<Fleet>[0], cruising())),
        compose(on<First>(cruising()), on<Second>(turning()), on<Third>(turning()))
    );
}

// Two models placed on one repetition, each writing one of its blocks, predict as the two placed on
// the tagged block.
TEST(PlacedRepetitions, ModelsOnOneRepetitionWriteItsBlocks)
{
    expectSamePredictions(
        compose(
            on(path<Fleet>[1], spinning()),
            on(path<Fleet>[1], moving()),
            onEach(path<Fleet>[std::array{0, 2}], cruising())
        ),
        compose(on<First>(cruising()), on<Second>(spinning()), on<Second>(moving()), on<Third>(cruising()))
    );
}

// An input renamed to a Repeated tag has a value per repetition selected, in their order: a model
// placed on vehicles 2 and 0 reads the first push for vehicle 2 and the second for vehicle 0.
TEST(PlacedRepetitions, AnInputHasAValuePerRepetitionSelected)
{
    Eigen::Vector4d const pushMean{0.4, -0.2, -0.3, 0.6};
    Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
    covariance.topLeftCorner<2, 2>() = Eigen::Vector2d{0.05, 0.02}.asDiagonal();
    covariance.bottomRightCorner<2, 2>() = Eigen::Vector2d{0.03, 0.04}.asDiagonal();
    auto const pushes = GaussianParams<Pushes>{Pushes{Push{}, 2}, pushMean, covariance};
    auto const thirdPush = GaussianParams<ThirdPush>{pushMean.head<2>(), covariance.topLeftCorner<2, 2>()};
    auto const firstPush = GaussianParams<FirstPush>{pushMean.tail<2>(), covariance.bottomRightCorner<2, 2>()};
    expectSamePredictions(
        compose(
            onEach(path<Fleet>[std::array{2, 0}], pushed(), on<Pushes>(path<Push>)),
            on(path<Fleet>[1], cruising())
        ),
        compose(
            on<First>(pushed(), on<FirstPush>(path<Push>)),
            on<Second>(cruising()),
            on<Third>(pushed(), on<ThirdPush>(path<Push>))
        ),
        std::tuple{pushes},
        std::tuple{firstPush, thirdPush}
    );
}

// A measurement model placed on one repetition updates as the model placed on its tagged block.
TEST(PlacedRepetitions, AMeasurementOnOneRepetitionIsOnItsTaggedBlock)
{
    Eigen::Vector3d const observed{-2.5, 7.6, 1.4};
    auto const expectSamePosterior = [&](auto const & method) {
        auto const selected = Emrys::update(fleetPrior(), observe(method, on(path<Fleet>[1], fixOf()), observed));
        auto const byTag = Emrys::update(taggedPrior(), observe(method, on<Second>(fixOf()), observed));
        ASSERT_TRUE(selected);
        ASSERT_TRUE(byTag);
        auto const & a = selected->branches[0].posterior;
        auto const & b = byTag->branches[0].posterior;
        EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
        Eigen::MatrixXd const lowerA = a.covarianceLower().template triangularView<Eigen::Lower>();
        Eigen::MatrixXd const lowerB = b.covarianceLower().template triangularView<Eigen::Lower>();
        EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
        EXPECT_GT((a.mean() - fleetMean()).norm(), 1e-3);
    };
    expectSamePosterior(methods::linearized());
    expectSamePosterior(methods::unscented());
}

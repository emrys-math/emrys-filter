// Process models whose writes() select repetitions of a repeated block (roadmap step 17zu, decisions.md
// [selected-writes], [composed-selections]): one repetition, a sequence, and a list written by parts of a
// composition. A state written with one tag per repetition, Composite<First, Second, Third>, has the
// coordinates of a Fleet of three vehicles, in the same order, so a prediction whose model writes a
// selection must equal the prediction whose model writes the tagged blocks, under every method; the two
// differ only in the sizes Eigen fixes at compile time, hence the tolerance. Also covered: a returned
// Jacobian whose rows reach a written list, a composition placed on a block, the policy's names and
// selections, and the assertion on writes that do not partition the state.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Placement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>

using Emrys::compose;
using Emrys::Composite;
using Emrys::continuousNoise;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::on;
using Emrys::path;
using Emrys::processModel;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::writes;

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

/// a clock beside the fleet, so that the fleet's rows are not the state's first
struct Clock : Linear<1>
{};

struct Convoy : Composite<Fleet>
{
    using Composite::Composite;
};

struct TaggedConvoy : Composite<First, Second, Third>
{};

using Site = Composite<Clock, Convoy>;
using TaggedSite = Composite<Clock, TaggedConvoy>;

using Vector12 = Eigen::Matrix<double, 12, 1>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;
using Vector13 = Eigen::Matrix<double, 13, 1>;
using Matrix13 = Eigen::Matrix<double, 13, 13>;

// A clock at 0.5, then three vehicles, each a position and a velocity, every coordinate distinct.
Vector13
siteMean()
{
    Vector13 mean;
    mean << 0.5, 1.0, 2.0, 0.3, -0.4, -3.0, 8.0, 1.1, 0.2, 6.0, -7.0, -0.6, 0.9;
    return mean;
}

// A covariance with every block correlated with every other: A A^T + I / 2 for a fixed full A.
Matrix13
siteCovariance()
{
    Matrix13 a;
    for (Eigen::Index i = 0; i < 13; ++i)
    {
        for (Eigen::Index j = 0; j < 13; ++j)
        {
            a(i, j) = 0.3 * std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
        }
    }
    return a * a.transpose() + 0.5 * Matrix13::Identity();
}

Vector12
fleetMean()
{
    return siteMean().tail<12>();
}

Matrix12
fleetCovariance()
{
    return siteCovariance().bottomRightCorner<12, 12>();
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

GaussianParams<Site>
sitePrior()
{
    return {Site{Clock{}, Convoy{Fleet{Vehicle{}, 3}}}, siteMean(), siteCovariance()};
}

GaussianParams<TaggedSite>
taggedSitePrior()
{
    return {siteMean(), siteCovariance()};
}

// Each vehicle of the input, four coordinates each, turns its velocity by a quarter of the step and
// moves by it, whatever the blocks the vehicles are read as.
auto const turnAll = []<typename T>(auto const & x, double dt) {
    Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 20, 1> y = x;
    T const c = T(std::cos(0.25 * dt));
    T const s = T(std::sin(0.25 * dt));
    for (Eigen::Index i = 0; i < x.size() / 4; ++i)
    {
        Eigen::Vector<T, 2> const v = x.template segment<2>(4 * i + 2);
        Eigen::Vector<T, 2> const turned{c * v(0) - s * v(1), s * v(0) + c * v(1)};
        y.template segment<2>(4 * i) = x.template segment<2>(4 * i) + dt * turned + T(0.1) * v.cwiseProduct(v);
        y.template segment<2>(4 * i + 2) = turned;
    }
    return y;
};

// The vehicles of the input held as they are.
auto const hold = []<typename T>(auto const & x, double) {
    return Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 20, 1>{x};
};

// A part of a composition that reads and writes \a items by \a function, with spectral density
// \a density on each coordinate it writes.
template <typename Function, typename... Items>
auto
partOf(Function const & function, double density, Items const &... items)
{
    return processModel(function, reads(items...), writes(items...), continuousNoise(density));
}

// Two predictions agree in their means and in their covariances' lower triangles, and the
// prediction moved the state from the mean \a prior.
template <typename Selected, typename ByTag, typename Prior>
void
expectSamePrediction(Selected const & selected, ByTag const & byTag, Prior const & prior)
{
    EXPECT_LT((selected.mean() - byTag.mean()).norm(), 1e-12);
    Eigen::MatrixXd const lowerA = selected.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = byTag.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), 1e-12);
    EXPECT_GT((selected.mean() - prior).norm(), 1e-3);
}

// The prediction of \a selected on the fleet against that of \a byTag on the tagged state, under
// linearization and two sigma-point methods.
template <typename Selected, typename ByTag>
void
expectSamePredictions(Selected const & selected, ByTag const & byTag)
{
    expectSamePrediction(
        Emrys::predict(methods::linearized(), fleetPrior(), selected, 0.5),
        Emrys::predict(methods::linearized(), taggedPrior(), byTag, 0.5),
        fleetMean()
    );
    expectSamePrediction(
        Emrys::predict(methods::unscented(), fleetPrior(), selected, 0.5),
        Emrys::predict(methods::unscented(), taggedPrior(), byTag, 0.5),
        fleetMean()
    );
    expectSamePrediction(
        Emrys::predict(methods::cubature(), fleetPrior(), selected, 0.5),
        Emrys::predict(methods::cubature(), taggedPrior(), byTag, 0.5),
        fleetMean()
    );
}

}  // namespace

// writes() of whole blocks is a Writes, which holds nothing; one with a selection or a name of its
// own is a WritesItems, which names each item as reads() does and holds each selection.
TEST(SelectedWrites, ItemsNamesAndSelections)
{
    static_assert(std::is_same_v<
                  decltype(writes(path<Position>, path<Velocity>)),
                  Emrys::Writes<Emrys::TagPath<Position>, Emrys::TagPath<Velocity>>>);
    static_assert(std::is_empty_v<decltype(writes(path<Position>))>);
    auto const one = writes(path<Fleet>[1]);
    static_assert(std::is_same_v<decltype(one)::OutputSpace, Composite<Vehicle>>);
    EXPECT_EQ(one.selection<0>(), 1);
    auto const several = writes(path<Fleet>[std::array{2, 0}], on<First>(path<Fleet>[1]));
    static_assert(std::is_same_v<decltype(several)::OutputSpace, Composite<Fleet, First>>);
    EXPECT_EQ(several.selection<0>(), (Eigen::Matrix<Eigen::Index, 2, 1>{2, 0}));
    EXPECT_EQ(several.selection<1>(), 1);
    auto const renamed = writes(on<Vehicle>(path<Second>));
    static_assert(std::is_same_v<decltype(renamed)::OutputSpace, Composite<Vehicle>>);
    static_assert(!decltype(renamed)::Selected && decltype(several)::Selected);
}

// One part writes the vehicle at repetition 1 and another the list of the rest: the composition
// predicts as the parts that write the tagged blocks.
TEST(SelectedWrites, OneRepetitionAndAListAreTheirTaggedBlocks)
{
    auto const selected =
        compose(partOf(turnAll, 0.3, path<Fleet>[1]), partOf(hold, 0.05, path<Fleet>[std::array{0, 2}]));
    auto const byTag =
        compose(partOf(turnAll, 0.3, on<Vehicle>(path<Second>)), partOf(hold, 0.05, path<First>, path<Third>));
    expectSamePredictions(selected, byTag);
}

// One part writes a sequence of two vehicles, as one run of rows, and another the first: the
// composition predicts as the parts that write the tagged blocks, whether the sequence's bounds are
// integers or count from the block's end.
TEST(SelectedWrites, ASequenceIsItsTaggedBlocks)
{
    auto const byTag =
        compose(partOf(turnAll, 0.3, path<Second>, path<Third>), partOf(hold, 0.05, on<Vehicle>(path<First>)));
    expectSamePredictions(
        compose(partOf(turnAll, 0.3, path<Fleet>[Eigen::seqN(1, 2)]), partOf(hold, 0.05, path<Fleet>[0])),
        byTag
    );
    expectSamePredictions(
        compose(partOf(turnAll, 0.3, path<Fleet>[Eigen::placeholders::lastN(2)]), partOf(hold, 0.05, path<Fleet>[0])),
        byTag
    );
}

// A list written in an order that is not the state's writes each vehicle's rows where the list puts
// them: the part moves the vehicles listed, third then first, as the part that writes Third and
// First in that order.
TEST(SelectedWrites, AListIsWrittenInItsOrder)
{
    auto const selected =
        compose(partOf(turnAll, 0.3, path<Fleet>[std::array{2, 0}]), partOf(hold, 0.05, path<Fleet>[1]));
    auto const byTag =
        compose(partOf(turnAll, 0.3, path<Third>, path<First>), partOf(hold, 0.05, on<Vehicle>(path<Second>)));
    expectSamePredictions(selected, byTag);
}

// Parts that return their Jacobians, whose rows a written list scatters into the state's rows,
// predict by linearization as the same parts differentiated automatically.
TEST(SelectedWrites, AReturnedJacobianIsWrittenToTheSelection)
{
    auto const turnReturning = Emrys::writesJacobians(
        [](Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 20, 1> const & x, double dt, auto & output) {
            Eigen::Index const n = x.size();
            Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(n, n);
            double const c = std::cos(0.25 * dt);
            double const s = std::sin(0.25 * dt);
            Eigen::Matrix2d rotation;
            rotation << c, -s, s, c;
            for (Eigen::Index i = 0; i < n / 4; ++i)
            {
                Eigen::Vector2d const v = x.segment<2>(4 * i + 2);
                jacobian.block<2, 2>(4 * i, 4 * i).setIdentity();
                jacobian.block<2, 2>(4 * i, 4 * i + 2) =
                    dt * rotation + Eigen::Matrix2d{Eigen::Vector2d{0.2 * v(0), 0.2 * v(1)}.asDiagonal()};
                jacobian.block<2, 2>(4 * i + 2, 4 * i + 2) = rotation;
            }
            Emrys::value(output) = turnAll.template operator()<double>(x, dt);
            if (Emrys::wantsJacobians(output))
            {
                Emrys::jacobian(output) = jacobian;
            }
        }
    );
    auto const holdReturning = Emrys::writesJacobians(
        [](Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 20, 1> const & x, double, auto & output) {
            Emrys::value(output) = x;
            if (Emrys::wantsJacobians(output))
            {
                Emrys::jacobian(output) = Eigen::MatrixXd::Identity(x.size(), x.size());
            }
        }
    );
    auto const supplied =
        compose(partOf(turnReturning, 0.3, path<Fleet>[std::array{2, 0}]), partOf(holdReturning, 0.05, path<Fleet>[1]));
    auto const automatic =
        compose(partOf(turnAll, 0.3, path<Fleet>[std::array{2, 0}]), partOf(hold, 0.05, path<Fleet>[1]));
    expectSamePrediction(
        Emrys::predict(methods::linearized(), fleetPrior(), supplied, 0.5),
        Emrys::predict(methods::linearized(), fleetPrior(), automatic, 0.5),
        fleetMean()
    );
}

// Continuous noise placed on one repetition and on a list of the others, as a density or as a
// matrix over the repetitions listed, is the noise placed on the tagged blocks ([placed-repetitions]).
TEST(SelectedWrites, NoiseIsPlacedOnSelectedRepetitions)
{
    auto const fleet = [](auto const &... noise) {
        return processModel(turnAll, reads(path<Fleet>), writes(path<Fleet>), continuousNoise(noise...));
    };
    auto const tagged = [](auto const &... noise) {
        return processModel(
            turnAll,
            reads(path<First>, path<Second>, path<Third>),
            writes(path<First>, path<Second>, path<Third>),
            continuousNoise(noise...)
        );
    };
    expectSamePredictions(
        fleet(on(path<Fleet>[1], 0.3), on(path<Fleet>[std::array{2, 0}], 0.05)),
        tagged(on<Second>(0.3), on<First>(0.05), on<Third>(0.05))
    );
    Eigen::Vector4d const first{0.1, 0.2, 0.3, 0.4};
    Eigen::Vector4d const third{0.5, 0.6, 0.7, 0.8};
    Eigen::Matrix<double, 8, 1> listed;
    listed << third, first;
    expectSamePredictions(
        fleet(
            on(
                path<Fleet>[std::array{2, 0}],
                Eigen::Matrix<double, 8, 8> {
                    listed.asDiagonal()
                }
            ),
            on(path<Fleet>[1], 0.3)
        ),
        tagged(
            on<First>(Eigen::Matrix4d{first.asDiagonal()}),
            on<Second>(0.3),
            on<Third>(Eigen::Matrix4d{third.asDiagonal()})
        )
    );
}

// A composition that writes selections, placed on a block of a larger state, predicts as the
// tagged composition placed on the tagged block.
TEST(SelectedWrites, APlacedCompositionWritesItsSelections)
{
    auto const fleetModel =
        compose(partOf(turnAll, 0.3, path<Fleet>[1]), partOf(hold, 0.05, path<Fleet>[std::array{2, 0}]));
    auto const taggedModel =
        compose(partOf(turnAll, 0.3, on<Vehicle>(path<Second>)), partOf(hold, 0.05, path<Third>, path<First>));
    auto const selected = compose(RandomWalk<Clock>{0.02}, on<Convoy>(fleetModel));
    auto const byTag = compose(RandomWalk<Clock>{0.02}, on<TaggedConvoy>(taggedModel));
    expectSamePrediction(
        Emrys::predict(methods::linearized(), sitePrior(), selected, 0.5),
        Emrys::predict(methods::linearized(), taggedSitePrior(), byTag, 0.5),
        siteMean()
    );
    expectSamePrediction(
        Emrys::predict(methods::unscented(), sitePrior(), selected, 0.5),
        Emrys::predict(methods::unscented(), taggedSitePrior(), byTag, 0.5),
        siteMean()
    );
}

// Writes whose selections overlap, or leave a repetition unwritten, do not partition the state,
// which is asserted when the model meets it.
TEST(SelectedWrites, WritesThatDoNotPartitionAssert)
{
    auto const overlapping =
        compose(partOf(turnAll, 0.3, path<Fleet>[1]), partOf(hold, 0.05, path<Fleet>[Eigen::seqN(0, 2)]));
    EXPECT_DEATH(std::ignore = Emrys::predict(methods::linearized(), fleetPrior(), overlapping, 0.5), "partitions");
    auto const missing = compose(partOf(turnAll, 0.3, path<Fleet>[1]), partOf(hold, 0.05, on<First>(path<Fleet>[0])));
    EXPECT_DEATH(std::ignore = Emrys::predict(methods::linearized(), fleetPrior(), missing, 0.5), "partitions");
}

// A batched function that writes a list of repetitions, a column per point, predicts as its
// per-point form under a sigma-point method (tests/Filters/selected_filters.cpp moves particles by one).
TEST(SelectedWrites, ABatchedFunctionWritesAList)
{
    auto const shrink = []<typename T>(auto const & x, double dt) {
        Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 20, 1> y = (1.0 - 0.1 * dt) * x;
        y += T(0.05) * x.cwiseProduct(x);
        return y;
    };
    auto const batchedShrink = Emrys::batched([](auto const &, auto const &, auto const & x, double dt, auto & y) {
        Emrys::values(y) = (1.0 - 0.1 * dt) * x + 0.05 * x.cwiseProduct(x);
    });
    auto const list = path<Fleet>[std::array{2, 0}];
    auto const byPoint = compose(partOf(shrink, 0.3, list), partOf(hold, 0.05, path<Fleet>[1]));
    auto const byBatch = compose(partOf(batchedShrink, 0.3, list), partOf(hold, 0.05, path<Fleet>[1]));
    expectSamePrediction(
        Emrys::predict(methods::unscented(), fleetPrior(), byBatch, 0.5),
        Emrys::predict(methods::unscented(), fleetPrior(), byPoint, 0.5),
        fleetMean()
    );
}

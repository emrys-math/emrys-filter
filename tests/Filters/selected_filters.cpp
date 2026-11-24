// Selections of a repeated block through the filters (roadmap step 17zu, decisions.md [selected-reads],
// [selected-writes], [placed-selected-values]). A state written with one tag per repetition,
// Composite<Position, Heading, First, Second, Third>, has the coordinates of Composite<Position, Heading,
// Landmarks> with three landmarks, in the same order, so each filter's step through a model that reads or
// writes path<Landmarks>[k] must equal its step through the model that reads or writes the tagged block:
// the Kalman filter in each of its four forms, the Gaussian mixture filter, the particle filter, drawn from
// the same generator, the Rao-Blackwellized particle filter, and the IMM's update of a hybrid over the
// rest given a landmark. The two differ only in the sizes Eigen fixes at compile time, hence the tolerance.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::byMode;
using Emrys::compose;
using Emrys::Composite;
using Emrys::continuousNoise;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::path;
using Emrys::processModel;
using Emrys::RadiansPi;
using Emrys::RandomWalk;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::when;
using Emrys::writes;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Heading : RadiansPi<>
{};

struct Landmark : Linear<2>
{};

struct Landmarks : Repeated<Landmark, 6>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Heading, Landmarks>;

struct First : Linear<2>
{};

struct Second : Linear<2>
{};

struct Third : Linear<2>
{};

using Tagged = Composite<Position, Heading, First, Second, Third>;

/// the vehicle's rest, given its position, in the Rao-Blackwellized filter
using Vector9 = Eigen::Matrix<double, 9, 1>;
using Matrix9 = Eigen::Matrix<double, 9, 9>;

// A vehicle at (1, 2) heading 0.3, and landmarks at (10, 4), (-3, 8), and (6, -7).
Vector9
mapMean()
{
    Vector9 mean;
    mean << 1.0, 2.0, 0.3, 10.0, 4.0, -3.0, 8.0, 6.0, -7.0;
    return mean;
}

// A covariance with every block correlated with every other: A A^T + I / 2 for a fixed full A.
Matrix9
mapCovariance()
{
    Matrix9 a;
    for (Eigen::Index i = 0; i < 9; ++i)
    {
        for (Eigen::Index j = 0; j < 9; ++j)
        {
            a(i, j) = 0.3 * std::sin(1.0 + 3.0 * double(i) + 7.0 * double(j));
        }
    }
    return a * a.transpose() + 0.5 * Matrix9::Identity();
}

GaussianParams<Map>
mapPrior()
{
    return {Map{Position{}, Heading{}, Landmarks{Landmark{}, 3}}, mapMean(), mapCovariance()};
}

GaussianParams<Tagged>
taggedPrior()
{
    return {mapMean(), mapCovariance()};
}

/// the noise of a sighting, 0.2 m in range and 0.05 rad in bearing, scaled in variance by \a scale
auto
sightingNoise(double scale = 1.0)
{
    return GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), scale * Eigen::Vector2d{0.04, 0.0025}.asDiagonal()};
}

// The range and the bearing from the vehicle to the landmark that \a item reads under Landmark, with
// noise \a noise.
template <typename Item, typename Noise = decltype(additiveNoise(sightingNoise()))>
auto
sightingOf(Item const & item, Noise const & noise = additiveNoise(sightingNoise()))
{
    return measurementModel(
        []<typename T>(auto const & in, auto const & x) {
            Eigen::Vector<T, 2> const d = block<Landmark>(in, x) - block<Position>(in, x);
            return Eigen::Vector<T, 2>{
                T(Eigen::numext::sqrt(T(d.squaredNorm()))),
                T(Eigen::numext::atan2(d(1), d(0)) - block<Heading>(in, x)(0))
            };
        },
        reads(path<Position>, path<Heading>, item),
        noise
    );
}

auto const selectedSighting = [] {
    return sightingOf(path<Landmarks>[1]);
};

auto const taggedSighting = [] {
    return sightingOf(on<Landmark>(path<Second>));
};

Eigen::Vector2d const observed{9.5, 0.1};

// The landmarks that \a items write each drift toward the origin by a tenth a second, with density
// 0.01 on each coordinate.
template <typename... Items>
auto
driftOf(Items const &... items)
{
    return processModel(
        []<typename T>(auto const & x, double dt) {
            return Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, 12, 1>{(1.0 - 0.1 * dt) * x};
        },
        reads(items...),
        writes(items...),
        continuousNoise(0.01)
    );
}

// The vehicle wanders, landmark 1 drifts, and the others drift more slowly, selected or tagged.
auto
selectedMotion()
{
    return compose(
        RandomWalk<Position>{0.2},
        RandomWalk<Heading>{0.01},
        driftOf(path<Landmarks>[1]),
        driftOf(path<Landmarks>[std::array{0, 2}])
    );
}

auto
taggedMotion()
{
    return compose(
        RandomWalk<Position>{0.2},
        RandomWalk<Heading>{0.01},
        driftOf(on<Landmark>(path<Second>)),
        driftOf(path<First>, path<Third>)
    );
}

struct Cruise
{};

struct Maneuvering
{};

/// a target that cruises or maneuvers, for the IMM's modes
struct Maneuver : Emrys::Named<Cruise, Maneuvering>
{};

// Two Gaussian states agree in their means and covariances, and the step moved the state.
template <typename A, typename B>
void
expectSameGaussian(A const & a, B const & b, double tolerance)
{
    EXPECT_LT((a.mean() - b.mean()).norm(), tolerance);
    Eigen::MatrixXd const lowerA = a.covarianceLower().template triangularView<Eigen::Lower>();
    Eigen::MatrixXd const lowerB = b.covarianceLower().template triangularView<Eigen::Lower>();
    EXPECT_LT((lowerA - lowerB).norm(), tolerance);
    EXPECT_GT((a.mean() - mapMean()).norm(), 1e-3);
}

}  // namespace

// The Kalman filter in each form predicts and updates through the selections as through the tagged blocks.
TEST(SelectedFilters, EachGaussianFormStepsAsTheTaggedBlocks)
{
    {
        auto selected = Emrys::KalmanFilter{selectedMotion(), mapPrior(), methods::unscented()};
        auto byTag = Emrys::KalmanFilter{taggedMotion(), taggedPrior(), methods::unscented()};
        selected.predict(0.5);
        byTag.predict(0.5);
        ASSERT_TRUE(selected.update(observe(methods::unscented(), selectedSighting(), observed)));
        ASSERT_TRUE(byTag.update(observe(methods::unscented(), taggedSighting(), observed)));
        expectSameGaussian(selected.state(), byTag.state(), 1e-12);
    }
    {
        auto selected = Emrys::SqrtKalmanFilter{selectedMotion(), Emrys::toSqrt(mapPrior()).value()};
        auto byTag = Emrys::SqrtKalmanFilter{taggedMotion(), Emrys::toSqrt(taggedPrior()).value()};
        ASSERT_TRUE(selected.predict(0.5));
        ASSERT_TRUE(byTag.predict(0.5));
        ASSERT_TRUE(selected.update(observe(methods::linearized(), selectedSighting(), observed)));
        ASSERT_TRUE(byTag.update(observe(methods::linearized(), taggedSighting(), observed)));
        expectSameGaussian(*Emrys::toGaussian(selected.state()), *Emrys::toGaussian(byTag.state()), 1e-11);
    }
    {
        auto selected = Emrys::InfoFilter{selectedMotion(), Emrys::toInfo(mapPrior()).value()};
        auto byTag = Emrys::InfoFilter{taggedMotion(), Emrys::toInfo(taggedPrior()).value()};
        ASSERT_TRUE(selected.predict(0.5));
        ASSERT_TRUE(byTag.predict(0.5));
        ASSERT_TRUE(selected.update(observe(methods::linearized(), selectedSighting(), observed)));
        ASSERT_TRUE(byTag.update(observe(methods::linearized(), taggedSighting(), observed)));
        expectSameGaussian(*Emrys::toGaussian(selected.state()), *Emrys::toGaussian(byTag.state()), 1e-10);
    }
    {
        auto selected = Emrys::SqrtInfoFilter{selectedMotion(), Emrys::toSqrtInfo(mapPrior()).value()};
        auto byTag = Emrys::SqrtInfoFilter{taggedMotion(), Emrys::toSqrtInfo(taggedPrior()).value()};
        ASSERT_TRUE(selected.predict(0.5));
        ASSERT_TRUE(byTag.predict(0.5));
        ASSERT_TRUE(selected.update(observe(methods::linearized(), selectedSighting(), observed)));
        ASSERT_TRUE(byTag.update(observe(methods::linearized(), taggedSighting(), observed)));
        expectSameGaussian(*Emrys::toGaussian(selected.state()), *Emrys::toGaussian(byTag.state()), 1e-10);
    }
}

// The Gaussian mixture filter steps each component as the Kalman filter does, and weights them alike.
TEST(SelectedFilters, AGaussianMixtureStepsAsTheTaggedBlocks)
{
    auto selectedMixture = Emrys::GaussianMixtureParams<Map>{mapPrior().space()};
    auto taggedMixture = Emrys::GaussianMixtureParams<Tagged>{Tagged{}};
    Vector9 shifted = mapMean();
    shifted.head<2>() += Eigen::Vector2d{0.5, -0.4};
    for (auto const & [logWeight, mean] : {std::pair{0.0, mapMean()}, std::pair{std::log(0.5), shifted}})
    {
        selectedMixture.add(logWeight, mean, mapCovariance());
        taggedMixture.add(logWeight, mean, mapCovariance());
    }
    auto selected = Emrys::GaussianMixtureFilter{selectedMotion(), std::move(selectedMixture)};
    auto byTag = Emrys::GaussianMixtureFilter{taggedMotion(), std::move(taggedMixture)};
    selected.predict(0.5);
    byTag.predict(0.5);
    ASSERT_TRUE(selected.update(observe(methods::unscented(), selectedSighting(), observed)));
    ASSERT_TRUE(byTag.update(observe(methods::unscented(), taggedSighting(), observed)));
    ASSERT_EQ(selected.state().size(), byTag.state().size());
    for (Eigen::Index i = 0; i < selected.state().size(); ++i)
    {
        auto const & a = selected.state().begin()[i];
        auto const & b = byTag.state().begin()[i];
        EXPECT_NEAR(a.logWeight, b.logWeight, 1e-12);
        expectSameGaussian(a.component, b.component, 1e-12);
    }
}

// The particle filter draws the same process noise from the same generator and weights the same
// particles alike through the selections as through the tagged blocks.
TEST(SelectedFilters, AParticleFilterStepsAsTheTaggedBlocks)
{
    auto draws = std::mt19937_64{5};
    auto const prior = mapPrior();
    Eigen::MatrixXd const states = Emrys::drawSamples(prior.toDensity().value(), 50, draws);
    auto selected = Emrys::ParticleFilter{selectedMotion(), Emrys::ParticleSet<Map>{mapPrior().space(), states}};
    auto byTag = Emrys::ParticleFilter{taggedMotion(), Emrys::ParticleSet<Tagged>{Tagged{}, states}};
    auto selectedRng = std::mt19937_64{7};
    auto taggedRng = std::mt19937_64{7};
    selected.predict(selectedRng, 0.5);
    byTag.predict(taggedRng, 0.5);
    EXPECT_LT((selected.state().states() - byTag.state().states()).norm(), 1e-12);
    ASSERT_TRUE(selected.update(observe(selectedSighting(), observed)));
    ASSERT_TRUE(byTag.update(observe(taggedSighting(), observed)));
    EXPECT_LT((selected.state().logWeights() - byTag.state().logWeights()).matrix().norm(), 1e-10);
}

// The Rao-Blackwellized particle filter, sampling the vehicle's position, predicts each particle's
// Gaussian over the rest through the selected writes, and updates it through a sighting of one
// landmark, as through the tagged blocks.
TEST(SelectedFilters, ARaoBlackwellizedFilterStepsAsTheTaggedBlocks)
{
    auto selectedDraws = std::mt19937_64{9};
    auto taggedDraws = std::mt19937_64{9};
    auto selected =
        Emrys::RbParticleFilter{selectedMotion(), Emrys::drawParticleGaussian<Position>(mapPrior(), 20, selectedDraws)};
    auto byTag =
        Emrys::RbParticleFilter{taggedMotion(), Emrys::drawParticleGaussian<Position>(taggedPrior(), 20, taggedDraws)};
    selected.predict(selectedDraws, 0.5);
    byTag.predict(taggedDraws, 0.5);
    ASSERT_TRUE(selected.update(observe(methods::unscented(), selectedSighting(), observed)));
    ASSERT_TRUE(byTag.update(observe(methods::unscented(), taggedSighting(), observed)));
    auto const & a = selected.state();
    auto const & b = byTag.state();
    EXPECT_LT((a.particles().logWeights() - b.particles().logWeights()).matrix().norm(), 1e-10);
    for (Eigen::Index i = 0; i < a.particles().size(); ++i)
    {
        EXPECT_LT((a.conditional(i).mean() - b.conditional(i).mean()).norm(), 1e-10);
    }
}

// A batched function that writes a list of landmarks moves the particles as its per-point form does,
// from the same draws.
TEST(SelectedFilters, AParticleFilterMovesParticlesByABatchedFunctionThatWritesAList)
{
    auto const batchedDrift = processModel(
        Emrys::batched([](auto const &, auto const &, auto const & x, double dt, auto & y) {
            y.values() = (1.0 - 0.1 * dt) * x;
        }),
        reads(path<Landmarks>[std::array{0, 2}]),
        writes(path<Landmarks>[std::array{0, 2}]),
        continuousNoise(0.01)
    );
    auto const byBatch =
        compose(RandomWalk<Position>{0.2}, RandomWalk<Heading>{0.01}, driftOf(path<Landmarks>[1]), batchedDrift);
    auto draws = std::mt19937_64{5};
    auto const prior = mapPrior();
    Eigen::MatrixXd const states = Emrys::drawSamples(prior.toDensity().value(), 50, draws);
    auto byPoint = Emrys::ParticleFilter{selectedMotion(), Emrys::ParticleSet<Map>{prior.space(), states}};
    auto batched = Emrys::ParticleFilter{byBatch, Emrys::ParticleSet<Map>{prior.space(), states}};
    auto first = std::mt19937_64{11};
    auto second = std::mt19937_64{11};
    byPoint.predict(first, 0.5);
    batched.predict(second, 0.5);
    EXPECT_LT((batched.state().states() - byPoint.state().states()).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_GT((batched.state().states() - states).cwiseAbs().maxCoeff(), 1e-3);
}

// The Rao-Blackwellized point-mass filter, its grid over the vehicle's position, predicts each cell's
// Gaussian over the rest through the selected writes, and updates it and the cells' weights through a
// sighting of one landmark, as through the tagged blocks.
TEST(SelectedFilters, ARaoBlackwellizedPointMassFilterStepsAsTheTaggedBlocks)
{
    auto const still = [](auto const & landmark, auto const &... others) {
        return compose(RandomWalk<Heading>{0.01}, driftOf(landmark), driftOf(others...));
    };
    auto const restMean = Eigen::Matrix<double, 7, 1>{mapMean().tail<7>()};
    auto const restCovariance = Eigen::Matrix<double, 7, 7>{mapCovariance().bottomRightCorner<7, 7>()};
    auto const space = mapPrior().space();
    auto selected = Emrys::RbPointMassFilter{
        still(path<Landmarks>[1], path<Landmarks>[std::array{0, 2}]),
        Emrys::regularGrid<Position>(space, Eigen::Vector2d{0.0, 1.0}, Eigen::Vector2d{1.0, 1.0}, {3, 3}),
        GaussianParams<Composite<Heading, Landmarks>>{
            Composite<Heading, Landmarks>{Heading{}, Landmarks{Landmark{}, 3}},
            restMean,
            restCovariance
        }
    };
    auto byTag = Emrys::RbPointMassFilter{
        still(on<Landmark>(path<Second>), path<First>, path<Third>),
        Emrys::regularGrid<Position>(Tagged{}, Eigen::Vector2d{0.0, 1.0}, Eigen::Vector2d{1.0, 1.0}, {3, 3}),
        GaussianParams<Composite<Heading, First, Second, Third>>{restMean, restCovariance}
    };
    selected.predict(0.5);
    byTag.predict(0.5);
    ASSERT_TRUE(selected.update(observe(methods::unscented(), selectedSighting(), observed)));
    ASSERT_TRUE(byTag.update(observe(methods::unscented(), taggedSighting(), observed)));
    auto const & a = selected.state();
    auto const & b = byTag.state();
    EXPECT_LT((a.toCategorical().logWeights() - b.toCategorical().logWeights()).matrix().norm(), 1e-10);
    EXPECT_LT((a.conditional(4).mean() - b.conditional(4).mean()).norm(), 1e-10);
}

// The IMM's update of a hybrid over the rest of the map given landmark 1 separately, as a
// Rao-Blackwellized filter's point gives it (decisions.md [placed-selected-values]), is the update
// given the tagged block Second: for a sighting the same in every mode, a sighting per mode, and a
// sighting with noise by mode.
TEST(SelectedFilters, AnImmUpdateGivenARepetitionIsGivenItsTaggedBlock)
{
    using Rest = Composite<Position, Heading, First, Third>;
    auto const rows = std::vector<Eigen::Index>{0, 1, 2, 3, 4, 7, 8};
    Eigen::VectorXd const mean = mapMean()(rows);
    Eigen::MatrixXd const covariance = mapCovariance()(rows, rows);
    Eigen::VectorXd shifted = mean;
    shifted.head<2>() += Eigen::Vector2d{0.5, -0.4};
    auto const initial = Emrys::HybridParams{
        Emrys::CategoricalParams<Maneuver>::fromWeights(0.7, 0.3),
        std::vector{GaussianParams<Rest>{mean, covariance}, GaussianParams<Rest>{shifted, 2.0 * covariance}}
    };
    auto const chain = Emrys::ContinuousTimeMarkovChain<Maneuver>::persistent(0.1);
    auto selected =
        Emrys::ImmRecursion<Emrys::Gaussian<Rest>, decltype(Emrys::SwitchingModel{chain, selectedMotion()})>{
            Emrys::SwitchingModel{chain, selectedMotion()}
        };
    auto byTag = Emrys::ImmRecursion<Emrys::Gaussian<Rest>, decltype(Emrys::SwitchingModel{chain, taggedMotion()})>{
        Emrys::SwitchingModel{chain, taggedMotion()}
    };
    Eigen::Vector2d const landmark{-2.8, 8.3};
    auto const space = mapPrior().space();
    auto const expectSame = [&](auto const & throughSelection, auto const & throughTag) {
        auto a = initial;
        auto b = initial;
        auto const updatedA = selected.update(a, throughSelection, Emrys::given<Landmarks>(space, landmark, 1));
        auto const updatedB = byTag.update(b, throughTag, Emrys::given<Second>(Tagged{}, landmark));
        ASSERT_TRUE(updatedA && updatedB);
        EXPECT_NEAR(updatedA->logLikelihood(), updatedB->logLikelihood(), 1e-10);
        EXPECT_LT((a.modes().logWeights() - b.modes().logWeights()).matrix().norm(), 1e-10);
        for (Eigen::Index m = 0; m < 2; ++m)
        {
            EXPECT_LT((a.conditional(m).mean() - b.conditional(m).mean()).norm(), 1e-10) << m;
            Eigen::MatrixXd const lowerA = a.conditional(m).covarianceLower().template triangularView<Eigen::Lower>();
            Eigen::MatrixXd const lowerB = b.conditional(m).covarianceLower().template triangularView<Eigen::Lower>();
            EXPECT_LT((lowerA - lowerB).norm(), 1e-10) << m;
        }
        EXPECT_GT((a.conditional(0).mean() - mean).norm(), 1e-3);
    };
    auto const byManeuver = [](auto const & item) {
        return byMode<Maneuver>(
            when<Cruise>(sightingOf(item)),
            when<Maneuvering>(sightingOf(item, additiveNoise(sightingNoise(4.0))))
        );
    };
    auto const noiseByManeuver = [](auto const & item) {
        return sightingOf(
            item,
            additiveNoise(byMode<Maneuver>(when<Cruise>(sightingNoise()), when<Maneuvering>(sightingNoise(4.0))))
        );
    };
    expectSame(
        observe(methods::unscented(), selectedSighting(), observed),
        observe(methods::unscented(), taggedSighting(), observed)
    );
    expectSame(
        observe(methods::unscented(), byManeuver(path<Landmarks>[1]), observed),
        observe(methods::unscented(), byManeuver(on<Landmark>(path<Second>)), observed)
    );
    expectSame(
        observe(methods::unscented(), noiseByManeuver(path<Landmarks>[1]), observed),
        observe(methods::unscented(), noiseByManeuver(on<Landmark>(path<Second>)), observed)
    );
}

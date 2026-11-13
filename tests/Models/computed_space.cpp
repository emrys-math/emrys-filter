// Measurement spaces computed from the state's space, and additive noise placed by tag path, of
// Models/MeasurementModel.h, Models/Policies.h, and Models/MeasurementReads.h (decisions.md
// [measurement-space-functions], [product-factors]): a model whose output space is a function of the state's, such
// as CountFrom, updates, weights points, simulates, and conditions every Gaussian form exactly as the same
// measurement built by hand for one count does, follows a new count without being rebuilt, and passes its function
// the computed space as out; noise placed in a fixed measurement space is the Gaussian of its parts. Its compile
// errors are in tests/Models/compile_errors.cpp.

#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Models/FormSteps.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>

#include <gtest/gtest.h>

#include <random>

using Emrys::additiveNoise;
using Emrys::block;
using Emrys::Composite;
using Emrys::CountFrom;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::measurementModel;
using Emrys::observe;
using Emrys::on;
using Emrys::onEach;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

namespace
{

struct Pose : Linear<2>
{};

struct Landmark : Linear<2>
{};

struct Landmarks : Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using Map = Composite<Pose, Landmarks>;

struct Range : Linear<1>
{};

struct Bearing : RadiansPi<>
{};

struct RangeBearing : Composite<Range, Bearing>
{};

struct RangeBearings : Repeated<RangeBearing, 8>
{
    using Repeated::Repeated;
};

// the noise of one range and bearing
Eigen::Matrix2d const rangeBearingNoise{{0.01, 0.0}, {0.0, 0.0004}};

// the range and bearing of landmark from pose
auto const rangeBearing = [](auto const & landmark, auto const & pose) {
    using T = ScalarOf<decltype(landmark)>;
    T const dx = T(landmark(0)) - T(pose(0));
    T const dy = T(landmark(1)) - T(pose(1));
    return Eigen::Matrix<T, 2, 1>{Eigen::numext::sqrt(T(dx * dx + dy * dy)), T(Eigen::numext::atan2(dy, dx))};
};

// the ranges and bearings of the landmarks, one per repetition of the output space out; a function
// another calls, so it deduces its scalar rather than taking it as a template parameter
auto const rangesAndBearings = [](auto const & in, auto const & out, auto const & x) {
    auto const landmarks = block<Landmarks>(in, x);
    auto const pose = block<Pose>(in, x);
    auto result = Emrys::zeros<ScalarOf<decltype(x)>>(out);
    for (Eigen::Index k = 0; k < out.count(); ++k)
    {
        result.template segment<2>(2 * k) = rangeBearing(landmarks.template segment<2>(2 * k).eval(), pose);
    }
    return result;
};

// Returns the model of every landmark's range and bearing, sized by the state's space
auto
makeComputed()
{
    return measurementModel(
        rangesAndBearings,
        reads(path<Landmarks>, path<Pose>),
        outputSpace(CountFrom<Landmarks, RangeBearings>{}),
        additiveNoise(onEach<RangeBearings>(GaussianParams<RangeBearing>{Eigen::Vector2d::Zero(), rangeBearingNoise}))
    );
}

// Returns the same model built by hand for count landmarks, with its space and I (x) R given
auto
makeByHand(Eigen::Index count)
{
    auto const space = RangeBearings{RangeBearing{}, count};
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(2 * count, 2 * count);
    for (Eigen::Index k = 0; k < count; ++k)
    {
        r.block<2, 2>(2 * k, 2 * k) = rangeBearingNoise;
    }
    return measurementModel(
        rangesAndBearings,
        reads(path<Landmarks>, path<Pose>),
        outputSpace(space),
        additiveNoise(GaussianParams<Linear<Eigen::Dynamic, 16>>{Eigen::VectorXd::Zero(2 * count), r})
    );
}

// Returns a prior over a pose and count landmarks, the last of them behind the pose across
// the bearing's seam
GaussianParams<Map>
makePrior(Eigen::Index count)
{
    Eigen::VectorXd mean(2 + 2 * count);
    mean.head<2>().setZero();
    for (Eigen::Index k = 0; k < count; ++k)
    {
        mean.segment<2>(2 + 2 * k) =
            Eigen::Vector2d{10.0 * std::cos(2.0 * double(k) + 0.5), 10.0 * std::sin(2.0 * double(k) + 0.5)};
    }
    mean.tail<2>() = Eigen::Vector2d{-10.0, 0.01};
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2 + 2 * count, 2 + 2 * count);
    cov(0, 1) = cov(1, 0) = 0.2;
    return GaussianParams<Map>{Map{Pose{}, Landmarks{Landmark{}, count}}, mean, cov};
}

// Returns a draw of the measurement of prior's mean by the model of every landmark
Eigen::VectorXd
makeObservation(GaussianParams<Map> const & prior, unsigned seed)
{
    auto rng = std::mt19937_64{seed};
    return Emrys::simulate(prior.space(), prior.mean(), makeComputed(), rng);
}

}  // namespace

// The update by a model whose space is computed is the update by the same measurement with its
// space and noise built by hand for the state's count, to rounding, by Linearized and Unscented.
TEST(ComputedSpace, UpdatesAsTheModelBuiltForTheCount)
{
    auto const prior = makePrior(3);
    auto const z = makeObservation(prior, 3);
    ASSERT_EQ(z.size(), 6);

    auto const computed = Emrys::update(prior, observe(methods::linearized(), makeComputed(), z));
    auto const byHand = Emrys::update(prior, observe(methods::linearized(), makeByHand(3), z));
    ASSERT_TRUE(computed && byHand);
    EXPECT_TRUE(computed->branches[0].posterior.mean().isApprox(byHand->branches[0].posterior.mean(), 1e-12));
    EXPECT_TRUE(computed->branches[0].posterior.covarianceLower().isApprox(
        byHand->branches[0].posterior.covarianceLower(),
        1e-12
    ));
    EXPECT_NEAR(computed->logLikelihood(), byHand->logLikelihood(), 1e-10);

    auto const unscented = Emrys::update(prior, observe(methods::unscented(), makeComputed(), z));
    auto const unscentedByHand = Emrys::update(prior, observe(methods::unscented(), makeByHand(3), z));
    ASSERT_TRUE(unscented && unscentedByHand);
    EXPECT_TRUE(unscented->branches[0].posterior.mean().isApprox(unscentedByHand->branches[0].posterior.mean(), 1e-12));
}

// One model serves states of any count: built once, it updates a state of three landmarks and
// one of five, each as the model built by hand for that count does, and its function sees each
// count in the space it is passed.
TEST(ComputedSpace, FollowsTheCountWithoutBeingRebuilt)
{
    auto const modelOfAll = makeComputed();
    for (Eigen::Index count : {3, 5})
    {
        auto const prior = makePrior(count);
        auto const z = makeObservation(prior, unsigned(count));
        ASSERT_EQ(z.size(), 2 * count);
        auto const computed = Emrys::update(prior, observe(methods::linearized(), modelOfAll, z));
        auto const byHand = Emrys::update(prior, observe(methods::linearized(), makeByHand(count), z));
        ASSERT_TRUE(computed && byHand);
        EXPECT_TRUE(computed->branches[0].posterior.mean().isApprox(byHand->branches[0].posterior.mean(), 1e-12));
    }

    auto seen = Eigen::Index{-1};
    auto const counting = measurementModel(
        [&seen](auto const & in, auto const & out, auto const & x) {
            seen = out.count();
            return rangesAndBearings(in, out, x);
        },
        reads(path<Landmarks>, path<Pose>),
        outputSpace(CountFrom<Landmarks, RangeBearings>{}),
        additiveNoise(onEach<RangeBearings>(GaussianParams<RangeBearing>{Eigen::Vector2d::Zero(), rangeBearingNoise}))
    );
    auto const prior = makePrior(4);
    ASSERT_TRUE(Emrys::update(prior, observe(methods::linearized(), counting, makeObservation(prior, 4))));
    EXPECT_EQ(seen, 4);
}

// The likelihood at points and a simulated draw come from the computed space and placed noise as
// they come from those built by hand: the same log-likelihoods, and the same draw from one seed.
TEST(ComputedSpace, WeightsPointsAndSimulatesAsTheModelBuiltForTheCount)
{
    auto const prior = makePrior(3);
    auto const z = makeObservation(prior, 7);
    auto rng = std::mt19937_64{8};
    auto const density = prior.toDensity();
    ASSERT_TRUE(density.has_value());
    Eigen::MatrixXd const states = Emrys::drawSamples(*density, 5, rng);
    Eigen::ArrayXd const computed = Emrys::logLikelihoods(observe(makeComputed(), z), prior.space(), states);
    Eigen::ArrayXd const byHand = Emrys::logLikelihoods(observe(makeByHand(3), z), prior.space(), states);
    EXPECT_TRUE(computed.isApprox(byHand, 1e-12));

    auto simulated = std::mt19937_64{9};
    auto simulatedByHand = std::mt19937_64{9};
    Eigen::VectorXd const draw = Emrys::simulate(prior.space(), prior.mean(), makeComputed(), simulated);
    Eigen::VectorXd const drawByHand = Emrys::simulate(prior.space(), prior.mean(), makeByHand(3), simulatedByHand);
    EXPECT_TRUE(draw.isApprox(drawByHand, 1e-14));
}

// The square-root, information, and square-root information forms place the measurement as the
// covariance form does, and update as they do by the model built by hand.
TEST(ComputedSpace, EveryGaussianFormTakesIt)
{
    auto const prior = makePrior(3);
    auto const z = makeObservation(prior, 11);
    auto const expected = Emrys::update(prior, observe(methods::linearized(), makeByHand(3), z));
    ASSERT_TRUE(expected);
    auto const & mean = expected->branches[0].posterior.mean();

    auto sqrt = Emrys::toSqrt(prior);
    ASSERT_TRUE(sqrt.has_value());
    ASSERT_TRUE(Emrys::internal::updateInPlace<false>(*sqrt, observe(methods::linearized(), makeComputed(), z)));
    EXPECT_TRUE(sqrt->mean().isApprox(mean, 1e-10));

    auto info = Emrys::toInfo(prior);
    ASSERT_TRUE(info.has_value());
    ASSERT_TRUE(Emrys::internal::updateInPlace<false>(*info, observe(methods::linearized(), makeComputed(), z)));
    auto const infoMean = info->mean();
    ASSERT_TRUE(infoMean.has_value());
    EXPECT_TRUE(infoMean->isApprox(mean, 1e-10));

    auto sqrtInfo = Emrys::toSqrtInfo(prior);
    ASSERT_TRUE(sqrtInfo.has_value());
    ASSERT_TRUE(Emrys::internal::updateInPlace<false>(*sqrtInfo, observe(methods::linearized(), makeComputed(), z)));
    auto const sqrtInfoMean = sqrtInfo->mean();
    ASSERT_TRUE(sqrtInfoMean.has_value());
    EXPECT_TRUE(sqrtInfoMean->isApprox(mean, 1e-10));
}

// By a sigma-point method, which linearizes over the factor, each of those forms places the
// measurement too, and updates as it does by the model built by hand.
TEST(ComputedSpace, EveryGaussianFormTakesItBySigmaPoints)
{
    auto const prior = makePrior(3);
    auto const z = makeObservation(prior, 11);
    auto const expectSameUpdate = [&](auto form) {
        auto byHand = form;
        ASSERT_TRUE(Emrys::internal::updateInPlace<false>(form, observe(methods::unscented(), makeComputed(), z)));
        ASSERT_TRUE(Emrys::internal::updateInPlace<false>(byHand, observe(methods::unscented(), makeByHand(3), z)));
        auto const placed = Emrys::toGaussian(form);
        auto const expected = Emrys::toGaussian(byHand);
        ASSERT_TRUE(placed.has_value());
        ASSERT_TRUE(expected.has_value());
        EXPECT_TRUE(placed->mean().isApprox(expected->mean(), 1e-10));
        EXPECT_TRUE(placed->covarianceLower().isApprox(expected->covarianceLower(), 1e-10));
    };
    expectSameUpdate(Emrys::toSqrt(prior).value());
    expectSameUpdate(Emrys::toInfo(prior).value());
    expectSameUpdate(Emrys::toSqrtInfo(prior).value());
}

// An update given the pose, as a Rao-Blackwellized filter's, computes the space from the whole
// state's, and conditions the landmarks as the model built by hand does.
TEST(ComputedSpace, UpdatesGivenABlock)
{
    using Rest = Composite<Landmarks>;
    auto const full = makePrior(2);
    auto const prior = GaussianParams<Rest>{
        Rest{Landmarks{Landmark{}, 2}},
        Eigen::VectorXd{full.mean().tail<4>()},
        Eigen::MatrixXd::Identity(4, 4)
    };
    auto const z = makeObservation(full, 13);
    auto const pose = Eigen::Vector2d{0.5, -0.5};
    static_assert(
        std::is_same_v<decltype(Emrys::given<Pose>(full.space(), pose)), Emrys::GivenBlock<Map, Pose, Eigen::Vector2d>>
    );
    auto const computed =
        Emrys::update(prior, observe(methods::linearized(), makeComputed(), z), Emrys::given<Pose>(full.space(), pose));
    auto const byHand =
        Emrys::update(prior, observe(methods::linearized(), makeByHand(2), z), Emrys::given<Pose>(full.space(), pose));
    ASSERT_TRUE(computed && byHand);
    EXPECT_TRUE(computed->branches[0].posterior.mean().isApprox(byHand->branches[0].posterior.mean(), 1e-12));
}

// CountFrom gives a resizable coordinate the count of a repeated block, one coordinate per
// repetition, whether or not the model reads the block.
TEST(ComputedSpace, CountFromSizesAResizableCoordinate)
{
    auto const space = Map{Pose{}, Landmarks{Landmark{}, 5}};
    auto const ranges = CountFrom<Landmarks, Linear<Eigen::Dynamic, 8>>{}(space);
    EXPECT_EQ(Emrys::valueDimension(ranges), 5);
    auto const rangeBearings = CountFrom<Landmarks, RangeBearings>{}(space);
    EXPECT_EQ(rangeBearings.count(), 5);
}

// Noise placed by tag path in a fixed measurement space is the Gaussian of its parts: a radar whose
// range and bearing noise are two parts updates as one with their covariance on the diagonal, and
// its bearing residual wraps across the seam in either.
TEST(ComputedSpace, PlacedNoiseInAFixedSpaceIsTheGaussianOfItsParts)
{
    struct Radar : Composite<Range, Bearing>
    {};

    auto const fromRadar = []<typename T>(auto const & in, auto const &, auto const & x) {
        return rangeBearing(block<Pose>(in, x), Eigen::Matrix<T, 2, 1>{T(5.0), T(0.0)});
    };
    auto const byParts = measurementModel(
        fromRadar,
        reads(path<Pose>),
        outputSpace(Radar{}),
        additiveNoise(
            on<Range>(GaussianParams<Range>{
                Eigen::Matrix<double, 1, 1>{0.0},
                Eigen::Matrix<double, 1, 1>{rangeBearingNoise(0, 0)}
            }),
            on<Bearing>(GaussianParams<Bearing>{
                Eigen::Matrix<double, 1, 1>{0.0},
                Eigen::Matrix<double, 1, 1>{rangeBearingNoise(1, 1)}
            })
        )
    );
    auto const dense = measurementModel(
        fromRadar,
        reads(path<Pose>),
        outputSpace(Radar{}),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), rangeBearingNoise})
    );
    auto const prior = GaussianParams<Pose>{Eigen::Vector2d{0.0, 0.01}, Eigen::Matrix2d::Identity()};
    auto const z = Eigen::Vector2d{5.0, -3.14};  // across the seam from the predicted bearing near pi
    auto const fromParts = Emrys::update(prior, observe(methods::linearized(), byParts, z));
    auto const fromDense = Emrys::update(prior, observe(methods::linearized(), dense, z));
    ASSERT_TRUE(fromParts && fromDense);
    EXPECT_TRUE(fromParts->branches[0].posterior.mean().isApprox(fromDense->branches[0].posterior.mean(), 1e-14));
    EXPECT_LT(fromParts->branches[0].nis(), 1.0);  // the residual took the short way round
}

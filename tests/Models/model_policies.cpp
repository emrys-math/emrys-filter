// The policies of Models/Policies.h (roadmap step 17z, decisions.md [model-policies], [tag-addressing],
// [additive-noise]): reads() and the input space it names, with items renamed by on<Name>(); writes() and its output
// space; affineIn(); additiveNoise() unplaced, by reference, and placed by tag path, and its placing in an output
// space as one Gaussian; continuousNoise(); logLikelihood(); outputSpace() given an instance or a function; the
// internal PolicySet that finds each in a pack; and sure(), the mark on a known input. Expected covariances are
// written out by hand. The compile errors are in tests/Models/compile_errors.cpp.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Models/MeasurementModel.h>
#include <EmrysFilter/Models/Policies.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>

using Emrys::additiveNoise;
using Emrys::affineIn;
using Emrys::Composite;
using Emrys::continuousNoise;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::logLikelihood;
using Emrys::on;
using Emrys::onEach;
using Emrys::outputSpace;
using Emrys::path;
using Emrys::RadiansPi;
using Emrys::reads;
using Emrys::Repeated;
using Emrys::sure;
using Emrys::TagPath;
using Emrys::writes;

namespace internal = Emrys::internal;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct SensorPosition : Linear<2>
{};

struct ScaleNoise : Linear<1>
{};

struct Heading : RadiansPi<>
{};

struct Ownship : Composite<Position, Velocity>
{};

struct Target : Composite<Position, Velocity>
{};

using Pair = Composite<Ownship, Target>;

struct Range : Linear<1>
{};

struct Bearing : RadiansPi<>
{};

using RangeBearing = Composite<Range, Bearing>;

struct Offset : Linear<2>
{};

struct Offsets : Repeated<Offset, 8>
{
    using Repeated::Repeated;
};

struct Landmark : Linear<2>
{};

struct Landmarks : Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using Map = Composite<Position, Landmarks>;

}  // namespace

// reads() keeps each item's path and reads it under its last tag, or under the tag on<Name>() gives;
// a bare tag's value is a path of one tag; and the input space is the Composite of the names, in order.
TEST(Policies, ReadsNamesItsItems)
{
    auto const r = reads(on<SensorPosition>(path<Ownship, Position>), path<Target, Position>, ScaleNoise{});
    using R = decltype(r);
    static_assert(R::Count == 3);
    static_assert(std::is_same_v<R::Item<0>::PathType, TagPath<Ownship, Position>>);
    static_assert(std::is_same_v<R::Item<0>::NameType, SensorPosition>);
    static_assert(!R::Item<0>::SingleTag);
    static_assert(std::is_same_v<R::Item<1>::PathType, TagPath<Target, Position>>);
    static_assert(std::is_same_v<R::Item<1>::NameType, Position>);
    static_assert(std::is_same_v<R::Item<2>::PathType, TagPath<ScaleNoise>>);
    static_assert(std::is_same_v<R::Item<2>::NameType, ScaleNoise>);
    static_assert(R::Item<2>::SingleTag);
    static_assert(std::is_same_v<R::InputSpace, Composite<SensorPosition, Position, ScaleNoise>>);
    static_assert(Emrys::valueDimensionAtCompileTime<R::InputSpace> == 5);
    static_assert(std::is_empty_v<R>);
}

// writes() keeps the blocks written, and the output space is their Composite in order.
TEST(Policies, WritesNamesItsBlocks)
{
    auto const w = writes(path<Range>, Bearing{});
    using W = decltype(w);
    static_assert(W::Count == 2);
    static_assert(std::is_same_v<W::Path<0>, TagPath<Range>>);
    static_assert(std::is_same_v<W::Path<1>, TagPath<Bearing>>);
    static_assert(std::is_same_v<W::OutputSpace, Composite<Range, Bearing>>);
    static_assert(std::is_same_v<decltype(writes(path<Ownship, Position>))::OutputSpace, Composite<Position>>);
}

// affineIn() keeps the names, each the tag an input is read under.
TEST(Policies, AffineInNamesInputs)
{
    auto const a = affineIn(path<ScaleNoise>, Position{});
    static_assert(std::is_same_v<decltype(a), Emrys::AffineIn<ScaleNoise, Position> const>);
    static_assert(decltype(a)::Count == 2);
}

// additiveNoise() of one distribution holds a copy, or a reference for std::cref(), and a mixture as
// it is; placed parts are kept as placed, and placing them in the output space gives one Gaussian
// over its differences with each part's covariance on its block.
TEST(Policies, AdditiveNoiseHoldsAndPlacesItsParts)
{
    auto const r = GaussianParams<Linear<2>>{Eigen::Vector2d{0.1, -0.1}, Eigen::Matrix2d{{0.5, 0.1}, {0.1, 0.4}}};
    auto const copied = additiveNoise(r);
    static_assert(!decltype(copied)::Placed);
    static_assert(std::is_same_v<decltype(copied)::Distribution, GaussianParams<Linear<2>>>);
    EXPECT_NE(&copied.distribution(), &r);
    EXPECT_EQ(copied.distribution().mean(), r.mean());

    auto const referred = additiveNoise(std::cref(r));
    static_assert(std::is_same_v<decltype(referred), Emrys::AdditiveNoise<GaussianParams<Linear<2>> const &> const>);
    EXPECT_EQ(&referred.distribution(), &r);

    auto mixture = GaussianMixtureParams<Linear<2>>{2};
    mixture.add(std::log(0.9), r);
    mixture.add(std::log(0.1), Eigen::Vector2d::Zero(), 25 * r.covarianceLower());
    auto const robust = additiveNoise(std::cref(mixture));
    static_assert(internal::isGaussianMixture<decltype(robust)::Distribution>);
    EXPECT_EQ(robust.distribution().size(), 2);

    // Placed parts, in a fixed output space: the unplaced Gaussian is the one given, by reference.
    auto const placed = additiveNoise(
        on<Range>(GaussianParams<Range>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.01}}),
        on<Bearing>(GaussianParams<Bearing>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{0.0004}})
    );
    static_assert(decltype(placed)::Placed);
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(placed.parts())>> == 2);
    auto const inRangeBearing = internal::placeNoise(RangeBearing{}, placed);
    static_assert(!decltype(inRangeBearing)::Placed);
    static_assert(std::is_same_v<decltype(inRangeBearing)::Distribution, GaussianParams<Linear<2>>>);
    EXPECT_EQ(inRangeBearing.distribution().mean(), Eigen::Vector2d::Zero());
    EXPECT_EQ(inRangeBearing.distribution().covarianceLower(), (Eigen::Matrix2d{{0.01, 0.0}, {0.0, 0.0004}}));
    auto const unplaced = internal::placeNoise(Linear<2>{}, copied);
    static_assert(std::is_same_v<decltype(unplaced), Emrys::AdditiveNoise<GaussianParams<Linear<2>> const &> const>);
    EXPECT_EQ(&unplaced.distribution(), &copied.distribution());

    // A part on each repetition of a repeated block follows the space instance's count.
    auto const perOffset =
        additiveNoise(onEach<Offsets>(GaussianParams<Offset>{Eigen::Vector2d{1.0, 2.0}, r.covarianceLower()}));
    auto const inOffsets = internal::placeNoise(Offsets{Offset{}, 3}, perOffset);
    EXPECT_EQ(inOffsets.distribution().valueDimension(), 6);
    EXPECT_EQ(inOffsets.distribution().mean().segment<2>(2), Eigen::Vector2d(1.0, 2.0));
    EXPECT_EQ((inOffsets.distribution().covarianceLower().block<2, 2>(4, 4)), r.covarianceLower());
    EXPECT_EQ((inOffsets.distribution().covarianceLower().block<2, 2>(2, 0)), Eigen::Matrix2d::Zero());
}

// additiveNoise() of a covariance alone is zero-mean noise: a matrix expression is held evaluated,
// it has no space of its own, and placing it gives the zero-mean Gaussian over the output's
// differences, unplaced, on a block, on each repetition, or beside a placed distribution.
TEST(Policies, AdditiveNoiseOfACovarianceIsZeroMean)
{
    Eigen::Matrix2d const r{{0.5, 0.1}, {0.1, 0.4}};
    auto const zeroMean = additiveNoise(2.0 * r);
    static_assert(std::is_same_v<decltype(zeroMean)::Distribution, Eigen::Matrix2d>);
    static_assert(decltype(zeroMean)::ZeroMean && !decltype(zeroMean)::HasOwnSpace && !decltype(zeroMean)::Placed);
    static_assert(decltype(additiveNoise(GaussianParams<Linear<2>>{}))::HasOwnSpace);
    auto const inLinear = internal::placeNoise(Linear<2>{}, zeroMean);
    static_assert(std::is_same_v<decltype(inLinear)::Distribution, GaussianParams<Linear<2>>>);
    EXPECT_EQ(inLinear.distribution().mean(), Eigen::Vector2d::Zero());
    EXPECT_EQ(Emrys::internal::symmetricFromLower<Eigen::Matrix2d>(inLinear.distribution().covarianceLower()), 2.0 * r);

    // In a periodic output space, over its differences.
    auto const inRangeBearing = internal::placeNoise(RangeBearing{}, zeroMean);
    EXPECT_EQ(inRangeBearing.distribution().mean(), Eigen::Vector2d::Zero());

    // Placed: a covariance on a block, beside a distribution on another, and on each repetition.
    auto const placed = additiveNoise(
        on<Range>(Eigen::Matrix<double, 1, 1>{0.01}),
        on<Bearing>(GaussianParams<Bearing>{Eigen::Matrix<double, 1, 1>{0.002}, Eigen::Matrix<double, 1, 1>{0.0004}})
    );
    static_assert(decltype(placed)::Placed);
    auto const placedIn = internal::placeNoise(RangeBearing{}, placed);
    EXPECT_EQ(placedIn.distribution().mean(), Eigen::Vector2d(0.0, 0.002));
    EXPECT_EQ(placedIn.distribution().covarianceLower(), (Eigen::Matrix2d{{0.01, 0.0}, {0.0, 0.0004}}));

    auto const perOffset = internal::placeNoise(Offsets{Offset{}, 3}, additiveNoise(onEach<Offsets>(r)));
    auto const perOffsetGaussian = internal::placeNoise(
        Offsets{Offset{}, 3},
        additiveNoise(onEach<Offsets>(GaussianParams<Offset>{Eigen::Vector2d::Zero(), r}))
    );
    EXPECT_EQ(perOffset.distribution().mean(), perOffsetGaussian.distribution().mean());
    EXPECT_EQ(perOffset.distribution().covarianceLower(), perOffsetGaussian.distribution().covarianceLower());
}

// continuousNoise() keeps its parts, placed or one for the whole output; logLikelihood() keeps its
// function; outputSpace() keeps an instance or a function of the state's space, and gives the
// space in a state's space either way.
TEST(Policies, TheOtherPoliciesHoldWhatTheyAreGiven)
{
    auto const q = continuousNoise(on<Velocity>(0.1), on<Heading>(Eigen::Matrix<double, 1, 1>{0.01}));
    static_assert(decltype(q)::Placed);
    EXPECT_EQ(std::get<0>(q.parts()).value, 0.1);
    EXPECT_EQ(std::get<1>(q.parts()).value(0, 0), 0.01);
    auto const whole = continuousNoise(Eigen::Matrix2d::Identity());
    static_assert(!decltype(whole)::Placed);
    EXPECT_EQ(std::get<0>(whole.parts()), Eigen::Matrix2d::Identity());

    auto const g = logLikelihood([](auto const &, auto const & x, auto const & z) {
        return -(z - x).squaredNorm();
    });
    EXPECT_EQ(g.function(Linear<2>{}, Eigen::Vector2d{1.0, 1.0}, Eigen::Vector2d{1.0, 2.0}), -1.0);

    auto const fixed = outputSpace(RangeBearing{});
    static_assert(!decltype(fixed)::Computed);
    static_assert(std::is_same_v<decltype(fixed.in(Map{})), RangeBearing>);
    auto const counted = outputSpace(Emrys::CountFrom<Landmarks, Offsets>{});
    static_assert(decltype(counted)::Computed);
    EXPECT_EQ(counted.in(Map{Position{}, Landmarks{Landmark{}, 5}}).count(), 5);
}

// PolicySet finds each policy of a pack by its kind, in any order, and says which are missing.
TEST(Policies, PolicySetFindsEachPolicyByKind)
{
    using internal::PolicyKind;
    auto const r = reads(path<Position>);
    auto const w = writes(path<Range>);
    auto const n =
        additiveNoise(GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}});
    auto const policies = std::tuple{n, w, r};
    using Set = internal::PolicySet<decltype(n), decltype(w), decltype(r)>;
    static_assert(Set::has<PolicyKind::reads> && Set::has<PolicyKind::writes> && Set::has<PolicyKind::additiveNoise>);
    static_assert(!Set::has<PolicyKind::affineIn> && !Set::has<PolicyKind::logLikelihood>);
    static_assert(std::is_same_v<Set::Type<PolicyKind::reads>, decltype(r)>);
    static_assert(std::is_same_v<Set::Type<PolicyKind::affineIn>, internal::NoPolicy>);
    EXPECT_EQ(&Set::get<PolicyKind::additiveNoise>(policies), &std::get<0>(policies));
    EXPECT_EQ(&Set::get<PolicyKind::reads>(policies), &std::get<2>(policies));
    static_assert(
        internal::ModelPolicy<decltype(r)> && !internal::ModelPolicy<int> && !internal::ModelPolicy<Position>
    );
}

// sure<Tag>(value) holds a copy of the value as a point of the tagged space, with its size checked
// against the space, and sure(space, value) takes a resizable instance.
TEST(Policies, SureMarksAKnownInput)
{
    auto const gravity = sure<Position>(Eigen::Vector2d{0.0, -9.8});
    static_assert(std::is_same_v<decltype(gravity), Emrys::Sure<Position, Eigen::Vector2d> const>);
    EXPECT_EQ(gravity.value, Eigen::Vector2d(0.0, -9.8));
    Eigen::VectorXd const two = Eigen::VectorXd::Ones(2);
    auto const dynamic = sure<Position>(two);
    static_assert(std::is_same_v<decltype(dynamic)::VectorType, Eigen::VectorXd>);
    EXPECT_EQ(dynamic.value, two);
    auto const landmarks = sure(Landmarks{Landmark{}, 2}, Eigen::Vector4d::Ones());
    EXPECT_EQ(landmarks.space.count(), 2);
    EXPECT_EQ(landmarks.value.size(), 4);
    static_assert(internal::isSure<std::remove_cvref_t<decltype(landmarks)>> && !internal::isSure<Eigen::Vector2d>);
    EXPECT_DEATH(std::ignore = sure<Position>(Eigen::VectorXd::Ones(3).eval()), "value in its space");
}

// The names a model reads a block under must retract as the block does: a tag over the block's own
// library space, the block's tag, or a tag derived from it; a composite or repeated tag by its parts.
TEST(Policies, NamesMatchBlocksByTheirLeaf)
{
    struct Derived : Position
    {};

    struct Fleet : Repeated<Target, 4>
    {};

    struct Convoy : Repeated<Target, 4>
    {};

    struct Shorter : Repeated<Target, 2>
    {};

    static_assert(internal::sameLeaf<SensorPosition, Position> && internal::sameLeaf<Derived, Position>);
    static_assert(internal::sameLeaf<Position, Position> && internal::sameLeaf<Bearing, Heading>);
    static_assert(!internal::sameLeaf<Range, Position> && !internal::sameLeaf<Heading, Range>);
    static_assert(internal::sameLeaf<Ownship, Target> && !internal::sameLeaf<Ownship, Position>);
    static_assert(internal::sameLeaf<Fleet, Convoy> && !internal::sameLeaf<Fleet, Shorter>);
    static_assert(internal::checkReadNames<decltype(reads(on<SensorPosition>(path<Ownship, Position>))), Pair>());
    static_assert(internal::checkReadNames<decltype(reads(path<ScaleNoise>)), Pair>());
}

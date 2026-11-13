// ParticleSet, KernelDensity, GaussianMixtureParams, and fitMixture() over non-linear spaces (steps
// 5a, 7; [discrete-spaces]): stored states are normalized, moments are tangent-space moments, kernels and
// conditionals wrap residuals, tag marginals keep the subspace, and a cluster that straddles a
// periodic seam is fitted and reduced as one.

#include <EmrysFilter/Distributions/Fitting.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/KernelDensity.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <type_traits>

using Emrys::Composite;
using Emrys::GaussianMixtureParams;
using Emrys::Linear;
using Emrys::ParticleSet;
using Emrys::RadiansPi;
using Emrys::Repeated;
using Emrys::TagPath;

namespace
{

constexpr double pi = std::numbers::pi;

struct Position : Linear<1>
{};

struct Heading : RadiansPi<>
{};

using State = Composite<Position, Heading>;
using Particles = ParticleSet<State>;
using MixtureParams = GaussianMixtureParams<State>;

// Distance around the circle between two headings.
double
angularGap(double a, double b)
{
    return std::abs(std::remainder(a - b, 2 * pi));
}

// Particles at headings π - 0.1 and -π + 0.1: a tight cloud straddling the seam.
Particles
makeStraddlingParticles()
{
    auto states = Eigen::Matrix<double, 2, 4>{{1.0, 2.0, 1.0, 2.0}, {pi - 0.1, pi - 0.1, -pi + 0.1, -pi + 0.1}};
    return Particles{State{}, states};
}

}  // namespace

// Every way of storing a state (construction, setState, add, forEach, setStates) wraps a periodic
// coordinate into [-pi, pi).
TEST(SpaceParticles, StatesAreNormalizedWhenStored)
{
    auto set = Particles{State{}, Eigen::Matrix<double, 2, 2>{{0.0, 0.0}, {4.0, -4.0}}};
    EXPECT_NEAR(set.state(0)(1), 4.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(set.state(1)(1), -4.0 + 2 * pi, 1e-15);

    set.setState(0, Eigen::Vector2d{0.0, 3.5});
    EXPECT_NEAR(set.state(0)(1), 3.5 - 2 * pi, 1e-15);
    set.add(0.0, Eigen::Vector2d{0.0, 7.0});
    EXPECT_NEAR(set.state(2)(1), 7.0 - 2 * pi, 1e-15);
    set.forEach([](auto & p) {
        p.setState(Eigen::Vector2d{p.state()(0), p.state()(1) + 2 * pi});
    });
    EXPECT_NEAR(set.state(2)(1), 7.0 - 2 * pi, 1e-12);
    set.setStates(Eigen::Matrix<double, 2, 3>::Constant(4.0));
    EXPECT_NEAR(set.state(1)(1), 4.0 - 2 * pi, 1e-15);
}

// Particles straddling the seam have mean heading pi and heading variance 0.01, the tangent-space
// values, where a linear average would give near zero and pi squared.
TEST(SpaceParticles, MomentsAreTangentSpaceMoments)
{
    auto const set = makeStraddlingParticles();
    auto const moments = set.moments().value();
    EXPECT_LT(angularGap(moments.mean()(1), pi), 1e-12);
    EXPECT_NEAR(moments.covarianceLower()(1, 1), 0.01, 1e-12);  // not ~π², as a linear average would give
    EXPECT_NEAR(moments.mean()(0), 1.5, 1e-12);
}

// The kernel density of a cloud straddling the seam is symmetric about pi, much higher there than
// at heading zero, and its samples stay in range.
TEST(SpaceParticles, KernelDensityWrapsResiduals)
{
    auto set = makeStraddlingParticles();
    set.setBandwidth(Eigen::Matrix2d{{0.1, 0.0}, {0.0, 0.02}});
    auto const kde = set.toDensity();
    ASSERT_TRUE(kde.has_value());
    // The cloud is symmetric about π, so points the same angular distance either side match.
    EXPECT_NEAR(
        kde->logDensity(Eigen::Vector2d{1.5, pi - 0.05}),
        kde->logDensity(Eigen::Vector2d{1.5, -pi + 0.05}),
        1e-12
    );
    EXPECT_GT(kde->logDensity(Eigen::Vector2d{1.5, -pi + 0.05}), kde->logDensity(Eigen::Vector2d{1.5, 0.0}) + 10.0);

    auto rng = std::mt19937_64{1};
    auto const samples = Emrys::drawSamples(*kde, 500, rng);
    EXPECT_TRUE((samples.row(1).array() >= -pi).all() && (samples.row(1).array() < pi).all());
}

// Conditioning on a heading just past the seam weights the particles as the equivalent heading
// does, the result is over the remaining block, and headings given a position stay in range.
TEST(SpaceParticles, ConditionalWrapsAndKeepsTheTargetSpace)
{
    struct Swapped : Composite<Heading, Position>
    {};

    auto states = Eigen::Matrix<double, 2, 4>{{pi - 0.1, pi - 0.1, -pi + 0.1, -pi + 0.1}, {1.0, 2.0, 1.0, 2.0}};
    auto set = ParticleSet<Swapped>{Swapped{}, states};
    set.setBandwidth(Eigen::Matrix2d{{0.02, 0.0}, {0.0, 0.1}});
    // Conditioning on heading -π + 0.1 weights both particle groups the same way as π - 0.1 would.
    auto const a = set.conditional<Heading>(Eigen::Matrix<double, 1, 1>{-pi + 0.1});
    auto const b = set.conditional<Heading>(Eigen::Matrix<double, 1, 1>{pi + 0.1});
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    static_assert(std::is_same_v<decltype(a)::value_type::SpaceType, Position>);
    EXPECT_TRUE((a->logWeights() - b->logWeights()).abs().maxCoeff() < 1e-12);

    auto const headingGivenPosition = set.conditional<Position>(Eigen::Matrix<double, 1, 1>{1.5});
    ASSERT_TRUE(headingGivenPosition);
    static_assert(std::is_same_v<decltype(headingGivenPosition)::value_type::SpaceType, Heading>);
    EXPECT_TRUE((headingGivenPosition->states().array() >= -pi).all());
    EXPECT_TRUE((headingGivenPosition->states().array() < pi).all());
}

// A tag marginal of the heading is a particle set over Heading, so its moments wrap too.
TEST(SpaceParticles, TagMarginalKeepsTheSubspace)
{
    auto const set = makeStraddlingParticles();
    auto const heading = set.marginal<Heading>().copy();
    static_assert(std::is_same_v<decltype(heading)::SpaceType, Heading>);
    auto const moments = heading.moments().value();
    EXPECT_LT(angularGap(moments.mean()(0), pi), 1e-12);
}

// add() and setMean() wrap a component's periodic coordinate.
TEST(SpaceMixture, ComponentMeansAreNormalized)
{
    auto mixture = MixtureParams{};
    mixture.add(0.0, Eigen::Vector2d{0.0, 4.0}, Eigen::Matrix2d::Identity() * 0.01);
    EXPECT_NEAR(mixture.begin()->component.mean()(1), 4.0 - 2 * pi, 1e-15);
    mixture.setMean(0, Eigen::Vector2d{0.0, -4.0});
    EXPECT_NEAR(mixture.begin()->component.mean()(1), -4.0 + 2 * pi, 1e-15);
}

// Two components either side of the seam have moment-matched mean pi, the within-component plus
// between-component variance, a density symmetric about pi, in-range samples, and conditionals and
// marginals over the named block spaces.
TEST(SpaceMixture, MomentsAndDensityWrap)
{
    auto mixture = MixtureParams{};
    mixture.add(0.0, Eigen::Vector2d{1.0, pi - 0.1}, Eigen::Matrix2d::Identity() * 0.01);
    mixture.add(0.0, Eigen::Vector2d{2.0, -pi + 0.1}, Eigen::Matrix2d::Identity() * 0.01);
    auto const g = mixture.toDensity();
    auto const moments = g->moments();
    EXPECT_LT(angularGap(moments.mean()(1), pi), 1e-12);
    EXPECT_NEAR(moments.covarianceLower()(1, 1), 0.01 + 0.01, 1e-12);
    EXPECT_NEAR(g->logDensity(Eigen::Vector2d{1.5, pi - 0.05}), g->logDensity(Eigen::Vector2d{1.5, -pi + 0.05}), 1e-12);

    auto rng = std::mt19937_64{2};
    auto const samples = Emrys::drawSamples(*g, 500, rng);
    EXPECT_TRUE((samples.row(1).array() >= -pi).all() && (samples.row(1).array() < pi).all());

    auto const conditional = mixture.conditional<Position>(Eigen::Matrix<double, 1, 1>{1.5});
    static_assert(std::is_same_v<decltype(conditional)::value_type::SpaceType, Heading>);
    auto const heading = mixture.marginal<Heading>().copy();
    static_assert(std::is_same_v<decltype(heading)::SpaceType, Heading>);
}

// Fitting two components to samples of a cluster centered on the seam and one at heading zero
// recovers both headings to 0.01 and the seam cluster's heading variance to 10%, which a linear fit
// would smear across the circle.
TEST(SpaceFitting, FitsAClusterAcrossTheSeam)
{
    // One cluster centered at heading π, and one at 0; samples straddle the seam.
    auto truth = MixtureParams{};
    truth.add(std::log(0.6), Eigen::Vector2d{0.0, pi}, Eigen::Matrix2d{{1.0, 0.0}, {0.0, 0.04}});
    truth.add(std::log(0.4), Eigen::Vector2d{5.0, 0.0}, Eigen::Matrix2d{{1.0, 0.0}, {0.0, 0.04}});
    auto rng = std::mt19937_64{3};
    auto const particles = Particles{State{}, Emrys::drawSamples(truth.toDensity().value(), 20000, rng)};
    auto const fit = Emrys::fitMixture(particles, 2, rng);
    ASSERT_TRUE(fit.has_value());
    static_assert(std::is_same_v<decltype(fit->mixture), MixtureParams>);
    ASSERT_EQ(fit->mixture.size(), 2);
    for (auto const & component : fit->mixture)
    {
        bool const seamCluster = std::abs(component.component.mean()(0)) < 2.5;
        double const expectedHeading = seamCluster ? pi : 0.0;
        EXPECT_LT(angularGap(component.component.mean()(1), expectedHeading), 0.01);
        EXPECT_NEAR(
            component.component.covarianceLower()(1, 1),
            0.04,
            0.004
        );  // a linear fit would smear the seam cluster
    }
}

// Reducing twenty components alternating either side of the seam to one gives mean heading pi and
// variance 0.01 + 0.05^2: within plus between, measured around the circle.
TEST(SpaceFitting, ReducesAMixtureAcrossTheSeam)
{
    auto stacked = MixtureParams{};
    for (int i = 0; i < 20; ++i)
    {
        double const heading = i % 2 == 0 ? pi - 0.05 : -pi + 0.05;
        stacked.add(0.0, Eigen::Vector2d{1.0, heading}, Eigen::Matrix2d{{0.5, 0.0}, {0.0, 0.01}});
    }
    auto rng = std::mt19937_64{4};
    auto const fit = Emrys::fitMixture(stacked.toDensity().value(), 1, rng);
    ASSERT_TRUE(fit.has_value());
    auto const first = fit->mixture.begin();
    auto const & component = *first;
    EXPECT_LT(angularGap(component.component.mean()(1), pi), 1e-9);
    EXPECT_NEAR(component.component.covarianceLower()(1, 1), 0.01 + 0.0025, 1e-6);
}

// marginal<Fleet>(i) of a mixture or a particle set over a Repeated space is over the element
// space, and a path into one repetition gives the leaf space.
TEST(SpaceDistributions, MarginalOfOneRepetitionIsInTheBlockSpace)
{
    struct Vehicle : Composite<Position, Heading>
    {};

    struct Fleet : Repeated<Vehicle>
    {
        using Repeated::Repeated;
    };

    auto const fleet = Fleet{Vehicle{}, 2};

    auto mixture = GaussianMixtureParams<Fleet>{fleet, 4};
    mixture.add(0.0, Eigen::Vector4d{1.0, 0.5, 2.0, pi - 0.1}, Eigen::Matrix4d::Identity());
    auto const second = mixture.marginal<Fleet>(1).copy();
    static_assert(std::is_same_v<decltype(second), GaussianMixtureParams<Vehicle> const>);
    EXPECT_EQ(second.begin()->component.mean(), Eigen::Vector2d(2.0, pi - 0.1));
    auto const density = mixture.toDensity();
    ASSERT_TRUE(density.has_value());
    static_assert(std::is_same_v<decltype(density->marginal<TagPath<Fleet, Heading>>(0))::SpaceType, Heading>);

    auto const particles =
        ParticleSet<Fleet>{fleet, Eigen::Matrix<double, 4, 2>{{1.0, 2.0}, {0.5, 0.6}, {3.0, 4.0}, {0.1, 0.2}}};
    auto const headings = particles.marginal<TagPath<Fleet, Heading>>(1).copy();
    static_assert(std::is_same_v<decltype(headings), ParticleSet<Heading> const>);
    EXPECT_EQ(headings.states(), (Eigen::Matrix<double, 1, 2>{0.1, 0.2}));
}

// An update whose innovation covariance is not positive definite (roadmap step 17k, [innovation-cholesky]): a
// measurement with no noise of a block the prior knows exactly makes S = 0, and every update
// returns GaussianError::NotPositiveDefinite with the state unchanged, through Emrys::update(),
// KalmanFilter in the covariance, square-root, and information forms, mixture noise,
// GaussianMixtureFilter, ImmFilter, RbParticleFilter, and RbPointMassFilter; a particle set's
// conditional on a block in which every particle coincides returns the same error. A measurement
// observed with roots::ridge() repairs an indefinite S instead, as
// if its noise were larger by the ridge, and keeps the ridged S in its innovation, and one with a repair of
// the user's own keeps that repair's S ([innovation-cholesky], [covariance-repair]). Headers under test:
// Models/Measurement.h, Models/Linearization.h, Models/FormSteps.h, Filters/KalmanFilter.h, Filters/MixtureFilter.h,
// Filters/ImmFilter.h, Filters/RbUpdate.h, Filters/RbParticleFilter.h, Filters/RbPointMassFilter.h,
// Distributions/ParticleSet.h.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

#include "user_axes.h"
#include "user_policies.h"

using Emrys::additiveNoise;
using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::drawParticleGaussian;
using Emrys::GaussianError;
using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Hybrid;
using Emrys::HybridParams;
using Emrys::ImmFilter;
using Emrys::InfoFilter;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MarkovChain;
using Emrys::Named;
using Emrys::ParticleSet;
using Emrys::PerPointCovariance;
using Emrys::RandomWalk;
using Emrys::RbParticleFilter;
using Emrys::RbPointMassFilter;
using Emrys::regularGrid;
using Emrys::SqrtGaussianParams;
using Emrys::SqrtKalmanFilter;
using Emrys::SwitchingModel;
using Emrys::toInfo;

namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;

auto
makeModel()
{
    return ConstantVelocity<Position, Velocity>{0.1};
}

// A prior that knows its velocity exactly: the velocity block of the covariance is zero.
GaussianParams<State>
makeKnownVelocity()
{
    auto const variances = Vector4{4.0, 4.0, 0.0, 0.0};
    return GaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, Matrix4{variances.asDiagonal()}};
}

// A prior uncertain in every coordinate.
GaussianParams<State>
makeUncertain()
{
    auto const variances = Vector4{4.0, 4.0, 1.0, 1.0};
    return GaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, Matrix4{variances.asDiagonal()}};
}

// The velocity, as a model's function.
auto const velocityOf = [](auto const & in, auto const &, auto const & x) {
    return Emrys::block<Velocity>(in, x);
};

// A measurement of the velocity with noise of covariance noiseVariance times the identity, by
// method, with any further arguments of observe(), such as a covariance update form and a repair.
template <typename Method, typename... Args>
auto
makeVelocityFix(Method const & method, double noiseVariance, Args const &... args)
{
    auto noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), noiseVariance * Eigen::Matrix2d::Identity()};
    return observe(
        method,
        Emrys::measurementModel(velocityOf, Emrys::reads(Emrys::path<Velocity>), additiveNoise(std::move(noise))),
        Eigen::Vector2d{0.5, -0.5},
        args...
    );
}

// The noise of an indefinite fix: with the uncertain prior's unit velocity variances it makes
// S = [[-1, 0.5], [0.5, 2]], whose Gershgorin bound is -1.5, so the ridge is 1.5 + 2 sqrt(eps) and
// the ridged S is well conditioned.
Eigen::Matrix2d
indefiniteNoise()
{
    return Eigen::Matrix2d{{-2.0, 0.5}, {0.5, 1.0}};
}

// A measurement of the velocity with noise of covariance noiseCov, by method, with any further
// arguments of observe().
template <typename Method, typename... Args>
auto
makeVelocityFixWith(Method const & method, Eigen::Matrix2d const & noiseCov, Args const &... args)
{
    auto noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), noiseCov};
    return observe(
        method,
        Emrys::measurementModel(velocityOf, Emrys::reads(Emrys::path<Velocity>), additiveNoise(std::move(noise))),
        Eigen::Vector2d{0.3, -0.2},
        args...
    );
}

// The indefinite fix by method, repairing its S by the ridge.
template <typename Method>
auto
makeRidgedFix(Method const & method)
{
    return makeVelocityFixWith(method, indefiniteNoise(), Emrys::updates::standard(), Emrys::roots::ridge());
}

// The ridge roots::ridge() adds to the indefinite fix's S, from the library's own computation.
double
indefiniteRidge()
{
    Eigen::Matrix2d const s = Eigen::Matrix2d::Identity() + indefiniteNoise();
    return Emrys::internal::ridgedCovariance(s).value()(0, 0) - s(0, 0);
}

// Whether two Gaussians have the same mean and covariance, bit for bit.
template <typename G>
bool
same(G const & a, G const & b)
{
    return a.mean() == b.mean() && a.covarianceLower() == b.covarianceLower();
}

// Whether two Rao-Blackwellized states hold the same weights and Gaussians, and for particles the
// same particles: compared element by element, since a state whose Gaussians know a coordinate
// exactly has no density to collapse (decisions.md [params-and-density]).
template <typename S>
[[nodiscard]] bool
sameState(S const & a, S const & b)
{
    if (a.size() != b.size() || !(a.logWeights() == b.logWeights()).all())
    {
        return false;
    }
    for (Eigen::Index i = 0; i < a.size(); ++i)
    {
        if (!same(a.conditional(i), b.conditional(i)))
        {
            return false;
        }
        if constexpr (requires { a.particle(i); })
        {
            if (a.particle(i) != b.particle(i))
            {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

// The measurement with no noise of the velocity the prior knows exactly has S = 0, which has no
// Cholesky factor, by a linear and by a sigma-point method; with noise the same update succeeds.
TEST(UpdateFailure, AnUpdateWithASingularInnovationReturnsTheError)
{
    auto const prior = makeKnownVelocity();
    auto const linear = Emrys::update(prior, makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(linear);
    EXPECT_EQ(linear.error(), GaussianError::NotPositiveDefinite);
    auto const sigmaPoints = Emrys::update(prior, makeVelocityFix(methods::unscented(), 0.0));
    ASSERT_FALSE(sigmaPoints);
    EXPECT_EQ(sigmaPoints.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(Emrys::update(prior, makeVelocityFix(methods::linearized(), 0.01)));
}

// A KalmanFilter's update returns the error and leaves its state as it was.
TEST(UpdateFailure, AKalmanFilterKeepsItsState)
{
    auto filter = KalmanFilter{makeModel(), makeKnownVelocity()};
    auto const before = filter.state();
    auto const updated = filter.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(same(filter.state(), before));
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
// A NaN or infinite observation, which would make the posterior NaN with the update accepted,
// fails an assertion in the covariance, square-root, and information forms.
TEST(UpdateFailureDeathTest, ANonFiniteObservationFails)
{
    auto const fixAt = [](Eigen::Vector2d const & observation) {
        auto noise = GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), 0.01 * Eigen::Matrix2d::Identity()};
        return observe(
            methods::linearized(),
            Emrys::measurementModel(
                Eigen::Matrix2d::Identity(),
                Emrys::reads(Emrys::path<Velocity>),
                additiveNoise(std::move(noise))
            ),
            observation
        );
    };
    auto const nan = fixAt(Eigen::Vector2d{std::nan(""), 0.0});
    auto const infinite = fixAt(Eigen::Vector2d{0.0, std::numeric_limits<double>::infinity()});
    EXPECT_DEATH((void)Emrys::update(makeUncertain(), nan), "the observation must be finite");
    EXPECT_DEATH((void)Emrys::update(makeUncertain(), infinite), "the observation must be finite");
    auto const factor = Matrix4{Vector4{2.0, 2.0, 1.0, 1.0}.asDiagonal()};
    auto sqrt = SqrtKalmanFilter{makeModel(), SqrtGaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, factor}};
    EXPECT_DEATH((void)sqrt.update(nan), "the observation must be finite");
    auto info = InfoFilter{makeModel(), toInfo(makeUncertain()).value()};
    EXPECT_DEATH((void)info.update(nan), "the observation must be finite");
}
#endif

// Mixture noise whose inlier component has no noise fails the whole update, though the outlier
// component's S is positive definite, and a filter's state is left as it was, as is a mixture
// filter's that keeps the branches into a result it reuses.
TEST(UpdateFailure, OneSingularBranchFailsMixtureNoise)
{
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero());
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    auto const fix = observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Velocity>),
            additiveNoise(std::move(noise))
        ),
        Eigen::Vector2d{0.5, -0.5}
    );
    auto const measured = Emrys::update(makeKnownVelocity(), fix);
    ASSERT_FALSE(measured);
    EXPECT_EQ(measured.error(), GaussianError::NotPositiveDefinite);

    auto filter = KalmanFilter{makeModel(), makeKnownVelocity()};
    auto const before = filter.state();
    auto const updated = filter.update(fix);
    ASSERT_FALSE(updated);
    EXPECT_TRUE(same(filter.state(), before));

    auto uncertain = GaussianMixtureParams<State>{State{}};
    uncertain.add(0.0, makeUncertain().mean(), makeUncertain().covarianceLower());
    auto result = GaussianMixtureFilter{makeModel(), uncertain}.update(fix, Emrys::reductions::keep()).value();
    auto known = GaussianMixtureParams<State>{State{}};
    known.add(0.0, makeKnownVelocity().mean(), makeKnownVelocity().covarianceLower());
    auto mixture = GaussianMixtureFilter{makeModel(), known};
    EXPECT_EQ(mixture.update(fix, Emrys::reductions::keep(), result).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(mixture.state().size(), 1);
}

// The square-root form never forms S: its update fails on the zero that the triangularization
// leaves on the diagonal of the innovation's factor, and the factor and mean stay as they were.
TEST(UpdateFailure, TheSquareRootFormFailsOnAZeroInTheInnovationFactor)
{
    auto const factor = Matrix4{Vector4{2.0, 2.0, 0.0, 0.0}.asDiagonal()};
    auto filter = SqrtKalmanFilter{makeModel(), SqrtGaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, factor}};
    auto const before = filter.state();
    auto const updated = filter.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(filter.state().mean() == before.mean());
    EXPECT_TRUE(filter.state().root() == before.root());
}

// A mixture filter fails the whole step when one component's update fails, here the component
// that knows its velocity exactly, and keeps every component and weight: the components are
// updated in place, so the one before it is updated and the failed one keeps its prior.
TEST(UpdateFailure, AMixtureFilterFailsTheStepForOneComponent)
{
    auto prior = GaussianMixtureParams<State>{State{}};
    prior.add(std::log(0.5), Vector4{1.0, 2.0, 0.5, -0.5}, Matrix4{Vector4{4.0, 4.0, 1.0, 1.0}.asDiagonal()});
    prior.add(std::log(0.5), makeKnownVelocity().mean(), makeKnownVelocity().covarianceLower());
    auto filter = GaussianMixtureFilter{makeModel(), prior};
    auto const updated = filter.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    ASSERT_EQ(filter.state().size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i)
    {
        EXPECT_EQ(filter.state()[i].logWeight, prior[i].logWeight);
    }
    auto const first = Emrys::update(prior.begin()->component, makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_TRUE(first);
    EXPECT_TRUE(filter.state().begin()->component.mean().isApprox(first->branches.front().posterior.mean(), 1e-12));
    EXPECT_TRUE(same(filter.state()[1].component, prior[1].component));
}

// The failure stops the mixture filter's other routes the same way: the components after a
// failing one are not updated, and keeping each component's branches, or updating them under a
// parallel policy, fails the step.
TEST(UpdateFailure, AMixtureFilterFailsOnEveryRoute)
{
    auto prior = GaussianMixtureParams<State>{State{}};
    prior.add(std::log(0.5), makeKnownVelocity().mean(), makeKnownVelocity().covarianceLower());
    prior.add(std::log(0.5), Vector4{1.0, 2.0, 0.5, -0.5}, Matrix4{Vector4{4.0, 4.0, 1.0, 1.0}.asDiagonal()});
    auto const fix = makeVelocityFix(methods::linearized(), 0.0);
    auto filter = GaussianMixtureFilter{makeModel(), prior};
    EXPECT_EQ(filter.update(fix).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(filter.update(fix, Emrys::reductions::keep()).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(
        filter.update(EmrysTest::ThreadsPolicy{}, fix, Emrys::reductions::keep()).error(),
        GaussianError::NotPositiveDefinite
    );
    EXPECT_EQ(filter.state().size(), 2);
}

// A KalmanFilter fails the same way when a policy other than Collapse reads its one branch, and
// when it updates into a result it reuses, in the covariance and the square-root forms.
TEST(UpdateFailure, AKalmanFilterFailsUnderAnyPolicyAndIntoAResult)
{
    auto const fix = makeVelocityFix(methods::linearized(), 0.0);
    auto filter = KalmanFilter{makeModel(), makeKnownVelocity()};
    EXPECT_EQ(filter.update(fix, Emrys::reductions::gate(0.99)).error(), GaussianError::NotPositiveDefinite);
    // A result of one branch holds a whole update, so it comes from one that succeeded.
    auto const noisy = makeVelocityFix(methods::linearized(), 0.01);
    auto result = KalmanFilter{makeModel(), makeUncertain()}.update(noisy).value();
    EXPECT_EQ(filter.update(fix, Emrys::reductions::collapse(), result).error(), GaussianError::NotPositiveDefinite);

    auto const factor = Matrix4{Vector4{2.0, 2.0, 0.0, 0.0}.asDiagonal()};
    auto root = SqrtKalmanFilter{makeModel(), SqrtGaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, factor}};
    auto rootResult = SqrtKalmanFilter{makeModel(), Emrys::toSqrt(makeUncertain()).value()}.update(noisy).value();
    EXPECT_EQ(root.update(fix, Emrys::reductions::collapse(), rootResult).error(), GaussianError::NotPositiveDefinite);
}

// An IMM filter fails the step when a mode's update fails and keeps its components and mode
// probabilities, in the square-root form too, and a mixture of IMMs fails the same way.
TEST(UpdateFailure, AnImmFilterKeepsItsState)
{
    struct Only
    {};

    struct Single : Named<Only>
    {};

    auto filter = ImmFilter{
        SwitchingModel{MarkovChain<Single>{}, makeModel()},
        HybridParams{CategoricalParams<Single>{}, makeKnownVelocity()}
    };
    auto const updated = filter.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(same(filter.state().conditional(0), makeKnownVelocity()));
    EXPECT_EQ(filter.state().modes().toDensity().value().probability(0), 1.0);

    auto const factor = Matrix4{Vector4{2.0, 2.0, 0.0, 0.0}.asDiagonal()};
    auto const knownRoot = SqrtGaussianParams<State>{Vector4{1.0, 2.0, 0.5, -0.5}, factor};
    auto root = ImmFilter{
        SwitchingModel{MarkovChain<Single>{}, makeModel()},
        Emrys::HybridParams<Emrys::SqrtGaussian<State>, Single>{CategoricalParams<Single>{}, knownRoot}
    };
    EXPECT_EQ(root.update(makeVelocityFix(methods::linearized(), 0.0)).error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(root.state().conditional(0).root(), factor);

    // So does an update whose noise components are the values of a mode axis.
    auto noise = GaussianMixtureParams<Linear<2>>{2};
    noise.add(0.0, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero());
    auto const byMode = observe(
        methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d::Identity(),
            Emrys::reads(Emrys::path<Velocity>),
            additiveNoise(std::move(noise)).byMode<Single>()
        ),
        Eigen::Vector2d{0.5, -0.5}
    );
    EXPECT_EQ(filter.update(byMode).error(), GaussianError::NotPositiveDefinite);

    auto hybrids = Emrys::MixtureParams<Emrys::HybridGaussian<State, Single>>{State{}};
    hybrids.add(0.0, HybridParams{CategoricalParams<Single>{}, makeKnownVelocity()});
    auto mixture = Emrys::MixtureFilter{SwitchingModel{MarkovChain<Single>{}, makeModel()}, hybrids};
    EXPECT_EQ(mixture.update(byMode).error(), GaussianError::NotPositiveDefinite);
}

// A Rao-Blackwellized particle filter whose particles' velocity Gaussians know the velocity
// exactly fails the step on a measurement with no noise of it, under a shared covariance (one S
// for every particle) and under a covariance per particle (an S each), with or without a policy,
// and keeps its particles, weights, and Gaussians.
TEST(UpdateFailure, ARbParticleFilterKeepsItsState)
{
    auto rng = std::mt19937_64{3};
    auto shared = RbParticleFilter{makeModel(), drawParticleGaussian<Position>(makeKnownVelocity(), 10, rng)};
    auto perPoint =
        RbParticleFilter{makeModel(), drawParticleGaussian<Position, PerPointCovariance>(makeKnownVelocity(), 10, rng)};
    auto const sharedBefore = shared.state();
    auto const perPointBefore = perPoint.state();
    auto const fromShared = shared.update(makeVelocityFix(methods::linearized(), 0.0));
    auto const fromPerPoint = perPoint.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(fromShared);
    ASSERT_FALSE(fromPerPoint);
    EXPECT_EQ(fromShared.error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(fromPerPoint.error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(
        perPoint.update(makeVelocityFix(methods::linearized(), 0.0), Emrys::reductions::collapse()).error(),
        GaussianError::NotPositiveDefinite
    );
    EXPECT_TRUE(sameState(shared.state(), sharedBefore));
    EXPECT_TRUE(sameState(perPoint.state(), perPointBefore));
}

// A Rao-Blackwellized point-mass filter whose cells' velocity Gaussian knows the velocity exactly
// fails the step on a measurement with no noise of it, under either covariance form, with or
// without a policy, and keeps its cells' probabilities and Gaussians.
TEST(UpdateFailure, ARbPointMassFilterKeepsItsState)
{
    auto const grid = regularGrid<State, Position>(Eigen::Vector2d{-1.0, -1.0}, Eigen::Vector2d{1.0, 1.0}, {3, 3});
    auto const velocity = GaussianParams<Velocity>{Velocity{}, Eigen::Vector2d{0.5, -0.5}, Eigen::Matrix2d::Zero()};
    auto shared = RbPointMassFilter{RandomWalk<Velocity>{0.01}, grid, velocity};
    auto perPoint = RbPointMassFilter{RandomWalk<Velocity>{0.01}, grid, velocity, PerPointCovariance{}};
    auto const sharedBefore = shared.state();
    auto const perPointBefore = perPoint.state();
    auto const fromShared = shared.update(makeVelocityFix(methods::linearized(), 0.0));
    auto const fromPerPoint = perPoint.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(fromShared);
    ASSERT_FALSE(fromPerPoint);
    EXPECT_EQ(fromShared.error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(fromPerPoint.error(), GaussianError::NotPositiveDefinite);
    EXPECT_EQ(
        perPoint.update(makeVelocityFix(methods::linearized(), 0.0), Emrys::reductions::collapse()).error(),
        GaussianError::NotPositiveDefinite
    );
    EXPECT_TRUE(sameState(shared.state(), sharedBefore));
    EXPECT_TRUE(sameState(perPoint.state(), perPointBefore));
}

// The information form already refused an S_0 without a Cholesky factor: with no noise on the
// velocity, S_0 is zero; it returns the same error and keeps its information.
TEST(UpdateFailure, TheInformationFormKeepsItsState)
{
    auto filter = InfoFilter{makeModel(), toInfo(makeUncertain()).value()};
    auto const before = filter.state();
    auto const updated = filter.update(makeVelocityFix(methods::linearized(), 0.0));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(filter.state().infoVector() == before.infoVector());
    EXPECT_TRUE(filter.state().infoMatrix() == before.infoMatrix());
}

// A particle set whose particles coincide in the given block has a Silverman bandwidth of zero
// there, so its conditional on that block has no kernel density and returns the error.
TEST(UpdateFailure, AParticleSetConditionalOnCoincidentParticlesReturnsTheError)
{
    auto states = Eigen::Matrix<double, 4, 3>{};
    states << 1.0, 1.0, 1.0,  //
        2.0, 2.0, 2.0,        //
        0.1, 0.4, -0.3,       //
        0.2, -0.1, 0.5;
    auto const set = ParticleSet<State>{states};
    auto const conditional = set.conditional<Position>(Eigen::Vector2d{1.0, 2.0});
    ASSERT_FALSE(conditional);
    EXPECT_EQ(conditional.error(), GaussianError::NotPositiveDefinite);
}

// An indefinite S fails the plain update. With the ridge, the update goes on as if the noise were
// larger by the ridge: its posterior is the plain update's with that noise, to rounding, and its
// innovation holds the ridged S, whose NIS and log-likelihood are then computed without failing.
TEST(UpdateFailure, TheRidgeRepairsAnIndefiniteInnovation)
{
    auto const prior = makeUncertain();
    auto const plain = Emrys::update(prior, makeVelocityFixWith(methods::linearized(), indefiniteNoise()));
    ASSERT_FALSE(plain);
    EXPECT_EQ(plain.error(), GaussianError::NotPositiveDefinite);

    double const ridge = indefiniteRidge();
    EXPECT_NEAR(ridge, 1.5, 1e-7);
    Eigen::Matrix2d const inflated = indefiniteNoise() + ridge * Eigen::Matrix2d::Identity();
    auto const reference = Emrys::update(prior, makeVelocityFixWith(methods::linearized(), inflated));
    auto const repaired = Emrys::update(prior, makeRidgedFix(methods::linearized()));
    ASSERT_TRUE(reference);
    ASSERT_TRUE(repaired);
    auto const & expected = reference->branches[0];
    auto const & branch = repaired->branches[0];
    EXPECT_TRUE(branch.posterior.mean().isApprox(expected.posterior.mean(), 1e-12));
    EXPECT_TRUE(branch.posterior.covarianceLower().isApprox(expected.posterior.covarianceLower(), 1e-12));
    Eigen::Matrix2d const ridged = Eigen::Matrix2d::Identity() + inflated;
    EXPECT_TRUE(branch.innovation.covarianceLower().isApprox(ridged, 1e-12));
    EXPECT_NEAR(branch.nis(), expected.nis(), 1e-12);
    EXPECT_NEAR(branch.logLikelihood(), expected.logLikelihood(), 1e-12);
}

// A ridge can't repair an innovation covariance that is not finite: it returns NonFinite.
TEST(UpdateFailure, TheRidgeFailsOnANonFiniteInnovation)
{
    Eigen::Matrix2d noise = indefiniteNoise();
    noise(0, 0) = -std::numeric_limits<double>::infinity();
    auto const fix =
        makeVelocityFixWith(methods::linearized(), noise, Emrys::updates::standard(), Emrys::roots::ridge());
    EXPECT_EQ(Emrys::update(makeUncertain(), fix).error(), GaussianError::NonFinite);
}

// A KalmanFilter conditions in place on the ridged S, which its update's innovation keeps, and
// reaches the free update's posterior bit for bit; so does Iterated, whose every iterate's S is
// the same indefinite matrix for this linear measurement and is repaired each time.
TEST(UpdateFailure, AKalmanFilterRepairsInPlace)
{
    auto const prior = makeUncertain();
    auto const expected = Emrys::update(prior, makeRidgedFix(methods::linearized()));
    ASSERT_TRUE(expected);
    auto filter = KalmanFilter{makeModel(), prior};
    auto const updated = filter.update(makeRidgedFix(methods::linearized()));
    ASSERT_TRUE(updated);
    EXPECT_TRUE(same(filter.state(), expected->branches[0].posterior));
    EXPECT_TRUE(
        updated->branches[0].innovation.covarianceLower() == expected->branches[0].innovation.covarianceLower()
    );
    EXPECT_EQ(updated->branches[0].nis(), expected->branches[0].nis());

    auto iterated = KalmanFilter{makeModel(), prior};
    ASSERT_TRUE(iterated.update(makeRidgedFix(Emrys::methods::iterated(methods::linearized()))));
    EXPECT_TRUE(iterated.state().mean().isApprox(filter.state().mean(), 1e-12));
    EXPECT_TRUE(iterated.state().covarianceLower().isApprox(filter.state().covarianceLower(), 1e-12));
}

// A repair of the user's own, an eigenvalue floor, is a measurement's repair as the ridge is: the update
// conditions on the S it returns, which its innovation keeps, bit for bit the floor's repair of S.
TEST(UpdateFailure, AMeasurementTakesARepairOfTheUsersOwn)
{
    auto const fix = makeVelocityFixWith(
        methods::linearized(),
        indefiniteNoise(),
        Emrys::updates::standard(),
        EmrysTest::EigenvalueFloor<1.0>{}
    );
    auto const updated = Emrys::update(makeUncertain(), fix);
    ASSERT_TRUE(updated);
    Eigen::Matrix2d const s = Eigen::Matrix2d::Identity() + indefiniteNoise();
    Eigen::Matrix2d const expected = EmrysTest::EigenvalueFloor<1.0>{}.repaired(s).value();
    Eigen::Matrix2d const kept = updated->branches[0].innovation.covarianceLower();
    EXPECT_EQ(
        kept.triangularView<Eigen::Lower>().toDenseMatrix(),
        expected.triangularView<Eigen::Lower>().toDenseMatrix()
    );
}

// A zero S has no ridge, since the margin scales with its diagonal, so the repaired update still
// fails and keeps the state.
TEST(UpdateFailure, TheRidgeLeavesAZeroInnovationFailing)
{
    auto filter = KalmanFilter{makeModel(), makeKnownVelocity()};
    auto const before = filter.state();
    auto const updated =
        filter.update(makeVelocityFix(methods::linearized(), 0.0, Emrys::updates::joseph(), Emrys::roots::ridge()));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(same(filter.state(), before));
}

// A Rao-Blackwellized particle filter with a shared covariance factors one S for every particle,
// which the ridge repairs too: the plain update fails and the ridged one succeeds.
TEST(UpdateFailure, TheRidgeRepairsASharedInnovation)
{
    auto rng = std::mt19937_64{5};
    auto const particles = drawParticleGaussian<Position>(makeUncertain(), 10, rng);
    auto plain = RbParticleFilter{makeModel(), particles};
    auto repaired = RbParticleFilter{makeModel(), particles};
    auto const failed = plain.update(makeVelocityFixWith(methods::linearized(), indefiniteNoise()));
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error(), GaussianError::NotPositiveDefinite);
    EXPECT_TRUE(repaired.update(makeRidgedFix(methods::linearized())));
}

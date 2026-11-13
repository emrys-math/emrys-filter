// Allocation control [allocation]: on bounded spaces, the Kalman filter's steps allocate nothing once
// warmed up, with a fixed-step model too [fixed-step-models] and a linear sensor [linear-sensors], the hidden Markov
// filter included (step 15, [emissions]),
// and the smoothers' histories (step 14, [smoothing]) take no more than a node per step, the covariance
// reset on a unit quaternion allocates nothing (step 13e, [covariance-reset]), and neither do the closed-form
// divergences [divergences], single draws with their log-densities, or the Monte Carlo divergences under
// Sequential. The mixture filters are in allocation_mixtures.cpp and the particle filters in
// allocation_particles.cpp. These files are their own test target, built with
// EIGEN_RUNTIME_NO_MALLOC so that every translation unit sees the same Eigen, and this one
// replaces the global operator new to count what the standard containers ask of the heap.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Divergence.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Filters/HiddenMarkovFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Smoothers/CategoricalHistory.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Models/CramerRao.h>
#include <EmrysFilter/Models/Emission.h>

#include <gtest/gtest.h>

#include "Spaces/unit_quaternion.h"
#include "allocation.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <random>

void *
operator new(std::size_t size)
{
    if (EmrysTest::Allocation::countingHeap)
    {
        ++EmrysTest::Allocation::heapAllocations;
    }
    if (void * p{std::malloc(size == 0 ? 1 : size)})
    {
        return p;
    }
#if defined(__cpp_exceptions)
    throw std::bad_alloc{};
#else
    std::abort();
#endif
}

void
operator delete(void * p) noexcept
{
    std::free(p);
}

void
operator delete(void * p, std::size_t /*size*/) noexcept
{
    std::free(p);
}

using namespace EmrysTest::Allocation;

using Emrys::CategoricalParams;
using Emrys::FixedStepModel;
using Emrys::GaussianHistory;
using Emrys::KalmanFilter;
using Emrys::Lag;
using Emrys::MarkovChain;

namespace methods = Emrys::methods;

// KalmanFilter predict() and update() under Linearized and Unscented, with a fixed-step model
// under Linearized and Unscented, and an update in the Joseph form, touch neither the heap nor
// Eigen's allocator after the first step.
TEST(Allocation, KalmanFilterStepsWithoutAllocating)
{
    auto joseph = KalmanFilter{makeModel(), makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        joseph.predict(timeStep);
        EXPECT_TRUE(joseph.update(makeFix(fixAt(k), methods::linearized(), Emrys::updates::joseph())));
    });
    auto linearized = KalmanFilter{makeModel(), makeInitial(), methods::linearized()};
    expectStepsAllocateNothing([&](int k) {
        linearized.predict(timeStep);
        EXPECT_TRUE(linearized.update(makeFix(fixAt(k), methods::linearized())));
    });
    auto unscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    expectStepsAllocateNothing([&](int k) {
        unscented.predict(timeStep);
        EXPECT_TRUE(unscented.update(makeFix(fixAt(k), methods::unscented())));
    });
    auto fixedAffine = KalmanFilter{FixedStepModel{makeModel(), State{}, timeStep}, makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        fixedAffine.predict();
        EXPECT_TRUE(fixedAffine.update(makeFix(fixAt(k))));
    });
    auto fixedUnscented =
        KalmanFilter{FixedStepModel{makeModel(), State{}, timeStep, methods::unscented()}, makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        fixedUnscented.predict();
        EXPECT_TRUE(fixedUnscented.update(makeFix(fixAt(k), methods::unscented())));
    });
}

// An update by a linear model, whose Jacobian is its matrix, allocates nothing either
// [linear-sensors].
TEST(Allocation, KalmanFilterUpdatesByALinearModelWithoutAllocating)
{
    auto const gps = Emrys::measurementModel(
        Eigen::Matrix2d::Identity(),
        Emrys::reads(Emrys::path<Position>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}})
    );
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        filter.predict(timeStep);
        EXPECT_TRUE(filter.update(observe(methods::linearized(), gps, fixAt(k))));
    });
}

// Under mixture noise an update has a run-time number of branches, held with their posteriors in
// the result's scratch (ReusableFilterUpdate): refilled through update(measurement, policy, result),
// under Collapse and under Select, which moves the posterior it keeps out of the scratch, the
// steps allocate nothing once the result has held the branches ([allocation], [innovation-statistics] as amended).
TEST(Allocation, KalmanFilterWithMixtureNoiseRefillsItsUpdate)
{
    auto z = Eigen::Vector2d{};
    auto const fix = makeMixtureFix(z);
    auto collapsing = KalmanFilter{makeModel(), makeInitial()};
    using Update = decltype(collapsing.update(fix))::value_type;
    static_assert(Update::BranchCount == std::dynamic_extent);
    auto result = Update{};
    expectStepsAllocateNothing([&](int k) {
        z = fixAt(k);
        collapsing.predict(timeStep);
        EXPECT_TRUE(collapsing.update(fix, Emrys::reductions::collapse(), result));
    });
    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(result.branches.size(), 2);

    auto selecting = KalmanFilter{makeModel(), makeInitial()};
    auto selected = Update{};
    expectStepsAllocateNothing([&](int k) {
        z = fixAt(k);
        selecting.predict(timeStep);
        EXPECT_TRUE(selecting.update(fix, Emrys::reductions::select(), selected));
    });
    EXPECT_TRUE(selected.accepted);
}

// The same under the sigma-point sets of step 17e [sigma-point-sets], whose points and outputs are sized by
// the state's bound, for Gauss-Hermite as for the others.
TEST(Allocation, KalmanFilterStepsWithoutAllocatingUnderEachPointSet)
{
    auto const check = [](auto const & method) {
        auto filter = KalmanFilter{makeModel(), makeInitial(), method};
        expectStepsAllocateNothing([&](int k) {
            filter.predict(timeStep);
            EXPECT_TRUE(filter.update(makeFix(fixAt(k), method)));
        });
    };
    check(Emrys::methods::cubature());
    check(Emrys::methods::sphericalSimplex<0.2>());
    check(Emrys::methods::gaussHermite<3>());
}

// A KalmanFilter recording each step in a fixed-lag GaussianHistory and smoothing over the window
// after each update, under Linearized and Unscented and with a fixed-step model, allocates nothing
// once the window is full: a full window reuses its oldest node [allocation], [smoothing].
TEST(Allocation, FixedLagSmoothingAllocatesNothingOnceFull)
{
    static constexpr Eigen::Index lag = 5;
    auto const expectSmoothingAllocatesNothing = [](auto & filter, auto const & fix, auto const & predictJoint) {
        auto window = GaussianHistory<State>{Lag{lag}};
        auto const step = [&](int k) {
            window.push(predictJoint(filter));
            EXPECT_TRUE(filter.update(fix(k)));
            window.smooth(filter.state());
        };
        for (int k = 0; k < lag; ++k)
        {
            step(k);
        }
        expectStepsAllocateNothing(step);
        EXPECT_EQ(window.size(), lag);
    };
    auto const linearizedFix = [](int k) {
        return makeFix(fixAt(k));
    };
    auto const unscentedFix = [](int k) {
        return makeFix(fixAt(k), methods::unscented());
    };
    auto const overTheStep = [](auto & filter) {
        return filter.predictJoint(timeStep);
    };
    auto const overTheFixedStep = [](auto & filter) {
        return filter.predictJoint();
    };

    auto linearized = KalmanFilter{makeModel(), makeInitial()};
    expectSmoothingAllocatesNothing(linearized, linearizedFix, overTheStep);
    auto unscented = KalmanFilter{makeModel(), makeInitial(), methods::unscented()};
    expectSmoothingAllocatesNothing(unscented, unscentedFix, overTheStep);
    auto fixed = KalmanFilter{FixedStepModel{makeModel(), State{}, timeStep}, makeInitial()};
    expectSmoothingAllocatesNothing(fixed, linearizedFix, overTheFixedStep);
}

// A Gaussian and a categorical history over a pmr resource, each keeping the newest steps, copy and
// move as allocator-aware containers do (expectHistoryIsAllocatorAware()).
TEST(Allocation, GaussianAndCategoricalHistoriesAreAllocatorAware)
{
    auto resource = std::pmr::monotonic_buffer_resource{};
    auto const allocator = Pmr{&resource};
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    auto gaussian = GaussianHistory<State, double, Pmr>{Lag{3}, allocator};
    for (int k = 0; k < 4; ++k)
    {
        gaussian.push(filter.predictJoint(timeStep));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k))));
    }
    expectHistoryIsAllocatorAware(gaussian);

    using Space = Emrys::Finite<3>;
    auto hidden = Emrys::HiddenMarkovFilter{MarkovChain<Space>::persistent(0.8)};
    auto categorical = Emrys::CategoricalHistory<Space, double, Pmr>{Lag{2}, allocator};
    for (int k = 0; k < 3; ++k)
    {
        categorical.push(hidden.predictJoint());
    }
    EXPECT_EQ(categorical.size(), 2);
    EXPECT_EQ(categorical.lag(), 2);
    expectHistoryIsAllocatorAware(categorical);
}

namespace
{

struct Attitude : EmrysTest::AutoDiffQuaternion
{};

struct ClosedFormAttitude : EmrysTest::UnitQuaternion
{};

// Random-walk noise on the rotation vector of the quaternion at Path, which does not move: a
// model of the user's own, differentiated automatically.
template <typename Path>
auto
attitudeWalk(double spectralDensity)
{
    return Emrys::processModel(
        [](auto const &, auto const &, auto const & x, auto) {
            return x;
        },
        Emrys::reads(Emrys::path<Path>),
        Emrys::writes(Emrys::path<Path>),
        Emrys::continuousNoise(Emrys::on<Path>(spectralDensity))
    );
}

// A direction in the body frame seen from a slowly turning attitude at step k.
template <typename A, typename Method = methods::Linearized>
auto
makeDirection(int k, Method const & method = {})
{
    Eigen::Vector3d const body = k % 2 == 0 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitZ();
    Eigen::Vector3d const observed =
        EmrysTest::rotate(EmrysTest::exponential(Eigen::Vector3d{0.01 * k, 0.3, -0.2}), body);
    auto const direction = Emrys::measurementModel(
        [body]<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 3, 1>{EmrysTest::rotate(Eigen::Matrix<T, 4, 1>{Emrys::block<A>(in, x)}, body)};
        },
        Emrys::reads(Emrys::path<A>),
        additiveNoise(GaussianParams<Linear<3>>{Eigen::Vector3d::Zero(), 0.01 * Eigen::Matrix3d::Identity()})
    );
    return observe(method, direction, observed);
}

// A KalmanFilter over attitude, position, and velocity that predicts, updates by a direction and a
// position fix, records each step in a fixed-lag window, and smooths it.
template <typename A>
void
expectAttitudeStepsAllocateNothing()
{
    using Body = Composite<A, Position, Velocity>;
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Identity();
    cov.topLeftCorner<3, 3>() *= 0.5;
    auto filter = KalmanFilter{
        compose(attitudeWalk<A>(0.01), ConstantVelocity<Position, Velocity>{0.1}),
        GaussianParams<Body>{mean, cov}
    };
    auto window = GaussianHistory<Body>{Lag{5}};
    auto const step = [&](int k) {
        window.push(filter.predictJoint(timeStep));
        EXPECT_TRUE(filter.update(makeDirection<A>(k)));
        EXPECT_TRUE(filter.update(makeFix(fixAt(k))));
        window.smooth(filter.state());
    };
    for (int k = 0; k < 5; ++k)
    {
        step(k);
    }
    expectStepsAllocateNothing(step);
}

}  // namespace

// Every root of a bounded covariance allocates nothing (step 17t): each of the library's axes, from the covariance and
// from a wide factor, the ridge, and a list with its report.
TEST(Allocation, RootsOfABoundedCovarianceAllocateNothing)
{
    namespace roots = Emrys::roots;
    using Bounded = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;
    using Wide = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 8>;
    Wide factor = Wide::Zero(4, 6);
    factor.leftCols<4>() = Eigen::Matrix4d::Identity();
    factor(1, 0) = 0.5;
    factor(3, 5) = 0.25;
    Bounded const cov = factor * factor.transpose();
    Bounded semidefinite = cov;
    semidefinite.row(2).setZero();
    semidefinite.col(2).setZero();
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 4, 1> const direction =
        Eigen::Vector4d{1.0, 2.0, 0.0, -1.0};
    double total = 0.0;
    expectStepsAllocateNothing([&](int /*k*/) {
        total += roots::squareRoot(roots::cholesky(roots::pivoted()), cov).sum();
        total += roots::squareRoot(roots::cholesky(roots::ridge<1.0, 1e-9>(), roots::pivoted()), semidefinite).sum();
        total += roots::squareRoot(roots::principal(), cov).sum();
        total += roots::squareRoot(roots::symmetric(), cov).sum();
        total += roots::squareRoot(roots::along(direction, roots::pivoted()), cov).sum();
        total += Emrys::internal::rootFromFactor(roots::cholesky(roots::pivoted()), factor).sum();
        total += Emrys::internal::rootFromFactor(roots::principal(), factor).sum();
        total += Emrys::internal::rootFromFactor(roots::along(direction, roots::pivoted()), factor).sum();
        total +=
            roots::squareRootWithReport(roots::cholesky(roots::ridge(), roots::pivoted()), semidefinite).root.sum();
    });
    EXPECT_TRUE(std::isfinite(total));
}

// The root form's solve with its root ([linearization-forms], roadmap step 17u) allocates nothing on a bounded
// size, whether the root is a Cholesky factor, solved by triangular substitution, or a pivoted root of
// a semidefinite matrix or a principal root, solved by a pivoted LDLT of S S^T.
TEST(Allocation, SolvesWithARootAllocateNothing)
{
    namespace roots = Emrys::roots;
    using Bounded = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;
    using Whitened = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 2>;
    Bounded cov = Bounded::Identity(4, 4);
    cov(1, 0) = 0.5;
    cov(0, 1) = 0.5;
    Bounded semidefinite = cov;
    semidefinite.row(2).setZero();
    semidefinite.col(2).setZero();
    Whitened whitened = Whitened::Ones(4, 2);
    whitened.row(2).setZero();
    Bounded const lower = roots::squareRoot(roots::cholesky(), cov).value();
    Bounded const pivoted = roots::squareRoot(roots::pivoted(), semidefinite);
    Bounded const principal = roots::squareRoot(roots::principal(), cov);
    double total = 0.0;
    expectStepsAllocateNothing([&](int /*k*/) {
        total += Emrys::internal::solveWithRoot<Whitened>(lower, whitened).sum();
        total += Emrys::internal::solveWithRoot<Whitened>(pivoted, whitened).sum();
        total += Emrys::internal::solveWithRoot<Whitened>(principal, whitened).sum();
    });
    EXPECT_TRUE(std::isfinite(total));
}

// On a state with a unit quaternion, the covariance reset of the update and the smoother's resets
// both ways [covariance-reset] allocate nothing, whether the quaternion's reset Jacobian is differentiated
// automatically or written in closed form.
TEST(Allocation, CovarianceResetAllocatesNothing)
{
    expectAttitudeStepsAllocateNothing<Attitude>();
    expectAttitudeStepsAllocateNothing<ClosedFormAttitude>();
}

namespace
{

// A KalmanFilter over attitude, position, and velocity that updates by a direction under Iterated
// with Inner, with no tolerance so that every update makes all its linearizations.
template <typename A, typename Inner = methods::Linearized>
void
expectIteratedAttitudeStepsAllocateNothing()
{
    using Body = Composite<A, Position, Velocity>;
    auto mean = Eigen::Matrix<double, 8, 1>{};
    mean << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5;
    Eigen::Matrix<double, 7, 7> cov = Eigen::Matrix<double, 7, 7>::Identity();
    cov.topLeftCorner<3, 3>() *= 0.5;
    auto filter = KalmanFilter{
        compose(attitudeWalk<A>(0.01), ConstantVelocity<Position, Velocity>{0.1}),
        GaussianParams<Body>{mean, cov}
    };
    auto const method = Emrys::methods::iterated(Inner{}, {.tolerance = 0.0, .maxIterations = 4});
    expectStepsAllocateNothing([&](int k) {
        filter.predict(timeStep);
        EXPECT_TRUE(filter.update(makeDirection<A>(k, method)));
    });
}

}  // namespace

// The iterated update (step 17b, [iterated-update]) allocates nothing on a bounded state, in place in a
// KalmanFilter and into a new posterior, nor on a unit quaternion, where each iterate's Jacobian is
// carried to the prior's chart by the reset Jacobian, automatic or closed form.
TEST(Allocation, IteratedUpdateAllocatesNothing)
{
    auto const method = Emrys::methods::iterated(methods::linearized(), {.tolerance = 0.0, .maxIterations = 4});
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        filter.predict(timeStep);
        EXPECT_TRUE(filter.update(makeFix(fixAt(k), method)));
    });
    GaussianParams<State> const prior = makeInitial();
    expectStepsAllocateNothing([&](int k) {
        auto const updated = Emrys::update(prior, makeFix(fixAt(k), method));
        ASSERT_TRUE(updated);
        // The fix is affine, so the second step is zero and stops the iteration even at no tolerance.
        EXPECT_GE(updated->branches.front().iterations, 2u);
    });
}

// The posterior Cramér-Rao bound (step 17g, [cramer-rao-bound]) steps over three truths held in a fixed-size
// matrix without allocating: by the model's transition, and through a position fix at each truth.
TEST(Allocation, PosteriorCramerRaoBoundStepsWithoutAllocating)
{
    auto bound = Emrys::PosteriorCramerRaoBound<State>::create(makeInitial()).value();
    auto truths = Eigen::Matrix<double, 4, 3>{};
    truths << 0.0, 1.0, -2.0,  //
        0.0, 0.5, 1.0,         //
        1.0, 0.0, 0.5,         //
        0.5, 1.0, -0.5;
    expectStepsAllocateNothing([&](int) {
        static_cast<void>(bound.predict(makeModel(), truths, timeStep));
        EXPECT_TRUE(bound.update(truths, makeFixModel()));
        static_cast<void>(bound.covariance());
    });
}

// The same on a unit quaternion.
TEST(Allocation, IteratedUpdateOnAQuaternionAllocatesNothing)
{
    expectIteratedAttitudeStepsAllocateNothing<Attitude>();
}

// The same on a unit quaternion with a closed-form reset.
TEST(Allocation, IteratedUpdateOnAClosedFormQuaternionAllocatesNothing)
{
    expectIteratedAttitudeStepsAllocateNothing<ClosedFormAttitude>();
}

// The iterated posterior linearization update (step 17e, [sigma-point-sets]) allocates nothing either: the
// posterior covariance of the blocks read that spreads each iterate's points is one buffer of their
// bounded size, in place and into a new posterior, and on a unit quaternion, where it is carried to
// each iterate's chart by the reset Jacobian.
TEST(Allocation, IteratedPosteriorLinearizationAllocatesNothing)
{
    auto const method = Emrys::methods::iterated(Emrys::methods::cubature(), {.tolerance = 0.0, .maxIterations = 4});
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    expectStepsAllocateNothing([&](int k) {
        filter.predict(timeStep);
        EXPECT_TRUE(filter.update(makeFix(fixAt(k), method)));
    });
    GaussianParams<State> const prior = makeInitial();
    expectStepsAllocateNothing([&](int k) {
        auto const updated = Emrys::update(prior, makeFix(fixAt(k), method));
        ASSERT_TRUE(updated);
        EXPECT_GE(updated->branches.front().iterations, 2u);
    });
    expectIteratedAttitudeStepsAllocateNothing<Attitude, Emrys::methods::Cubature>();
    expectIteratedAttitudeStepsAllocateNothing<ClosedFormAttitude, Emrys::methods::GaussHermite<2>>();
}

// An unbounded GaussianHistory over a pmr resource asks the resource for exactly one node per
// recorded step and the global heap for nothing, and smoothing it allocates nothing at all.
TEST(Allocation, UnboundedHistoryAllocatesOneNodePerStep)
{
    auto resource = CountingResource{arenaBytes};
    auto history = GaussianHistory<State, double, Pmr>{Pmr{&resource}};
    auto filter = KalmanFilter{makeModel(), makeInitial()};
    history.push(filter.predictJoint(timeStep));
    EXPECT_TRUE(filter.update(makeFix(fixAt(0))));
    int const before = resource.allocations();
    int heap = 0;
    {
        auto const watch = HeapWatch{};
        for (int k = 1; k <= 100; ++k)
        {
            history.push(filter.predictJoint(timeStep));
            EXPECT_TRUE(filter.update(makeFix(fixAt(k))));
        }
        history.smooth(filter.state());
        heap = watch.allocations();
    }
    EXPECT_EQ(heap, 0) << "allocations from the global heap";
    EXPECT_EQ(resource.allocations() - before, 100) << "one node per step";
    EXPECT_EQ(history.size(), 101);
}

// The closed-form divergences [divergences] are sized by the space, so on a bounded one they allocate
// nothing at all, the ground metric's singular-value decomposition included, nor between two forms,
// such as an information form's density and a square-root form's, and neither does the
// collapse bracket of a mixture, which factorizes one fixed-size covariance and loops over the
// components. The mixture allocates its components when built, before the steps.
TEST(Allocation, ClosedFormDivergencesAllocateNothing)
{
    GaussianParams<State> const p = makeInitial();
    GaussianParams<State> q = makeInitial();
    q.setMean(Vector4{1.0, -0.5, 0.8, 0.7});
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    ASSERT_TRUE(pDensity && qDensity);
    auto const pInfo = Emrys::toInfo(p).value();
    auto const qSqrt = Emrys::toSqrt(q).value();
    auto const pInfoDensity = pInfo.toDensity();
    auto const qSqrtDensity = qSqrt.toDensity();
    ASSERT_TRUE(pInfoDensity && qSqrtDensity);
    auto metric = Emrys::GroundMetric{State{}};
    metric.combine<Position, Velocity>(2.0);
    metric.scale<Velocity>(0);
    auto const firstWeights = CategoricalParams<Mode>::fromWeights(Mode{}, Vector4{0.4, 0.3, 0.2, 0.1});
    auto const secondWeights = CategoricalParams<Mode>::fromWeights(Mode{}, Vector4{0.1, 0.2, 0.3, 0.4});
    auto const first = firstWeights.toDensity().value();
    auto const second = secondWeights.toDensity().value();
    auto mixtureParams = Emrys::GaussianMixtureParams<State>{State{}};
    mixtureParams.add(std::log(0.4), p.mean(), p.covarianceLower());
    mixtureParams.add(std::log(0.6), q.mean(), q.covarianceLower());
    auto const mixture = mixtureParams.toDensity();
    ASSERT_TRUE(mixture);
    double total = 0;
    expectStepsAllocateNothing([&](int) {
        auto const bounds = Emrys::kullbackLeiblerCollapseBounds(*mixture);
        total += bounds.lower + bounds.upper;
        total += Emrys::kullbackLeibler(*pDensity, *qDensity) + Emrys::bhattacharyya(*pDensity, *qDensity)
               + Emrys::hellinger(*pDensity, *qDensity) + Emrys::wasserstein(*pDensity, *qDensity)
               + Emrys::wasserstein(*pDensity, *qDensity, metric) + Emrys::expectedSquaredError(p, q.mean(), metric)
               + Emrys::kullbackLeibler(first, second) + Emrys::hellinger(first, second)
               + Emrys::jensenShannon(first, second);
        total += Emrys::kullbackLeibler(*pInfoDensity, *qSqrtDensity)
               + Emrys::bhattacharyya(*pInfoDensity, *qSqrtDensity)
               + Emrys::wasserstein(*qSqrtDensity, *pInfoDensity, metric);
    });
    EXPECT_GT(total, 0);
}

// On a bounded space a draw with its log-density, from a Gaussian or a mixture, allocates nothing,
// and so do the Monte Carlo divergences under Sequential, which take their draws one at a time and
// add each term to a running mean [allocation]. The mixture allocates its components when built,
// before the steps.
TEST(Allocation, SingleDrawsAndMonteCarloDivergencesAllocateNothing)
{
    GaussianParams<State> const p = makeInitial();
    GaussianParams<State> q = makeInitial();
    q.setMean(Vector4{1.0, -0.5, 0.8, 0.7});
    auto mixtureParams = Emrys::GaussianMixtureParams<State>{State{}};
    mixtureParams.add(std::log(0.4), p.mean(), p.covarianceLower());
    mixtureParams.add(std::log(0.6), q.mean(), q.covarianceLower());
    auto const pDensity = p.toDensity();
    auto const qDensity = q.toDensity();
    auto const mixture = mixtureParams.toDensity();
    ASSERT_TRUE(pDensity && qDensity && mixture);
    auto const method = Emrys::MonteCarlo{200};
    auto rng = std::mt19937_64{11};
    double total = 0;
    expectStepsAllocateNothing([&](int) {
        total += Emrys::drawSampleWithLogDensity(*pDensity, rng).logDensity
               + Emrys::drawSampleWithLogDensity(*mixture, rng).logDensity;
        total += Emrys::kullbackLeibler(method, *mixture, *qDensity, rng).value
               + Emrys::bhattacharyya(method, *pDensity, *qDensity, rng).value
               + Emrys::hellinger(method, *mixture, *pDensity, rng).value
               + Emrys::jensenShannon(method, *mixture, *qDensity, rng).value;
    });
    EXPECT_TRUE(std::isfinite(total));
}

// HiddenMarkovFilter on a bounded space touches neither the heap nor Eigen's allocator after the
// first step: predict() and update() by an emission table with a MarkovChain; predict(dt), the
// arrival interval, and a Poisson count with a ContinuousTimeMarkovChain; and a fixed-lag
// CategoricalHistory recording predictJoint() and smoothing after every update, once full.
TEST(Allocation, HiddenMarkovFilterAndSmootherStepWithoutAllocating)
{
    using Space = Emrys::Finite<3>;
    auto const emissions = Eigen::Matrix<double, 3, 2>{{0.9, 0.1}, {0.4, 0.6}, {0.15, 0.85}};
    auto const table = Emrys::EmissionTable{Space{}, Emrys::Finite<2>{}, emissions};
    auto const chain = MarkovChain<Space>::persistent(0.8);

    auto discrete = Emrys::HiddenMarkovFilter{chain};
    expectStepsAllocateNothing([&](int k) {
        discrete.predict();
        EXPECT_TRUE(discrete.update(table.observe(k % 2)));
    });

    auto const rates = Eigen::Array3d{0.5, 2.0, 6.0};
    auto continuous = Emrys::HiddenMarkovFilter{Emrys::ContinuousTimeMarkovChain<Space>::persistent(0.7)};
    expectStepsAllocateNothing([&](int k) {
        continuous.predict(0.1);
        EXPECT_TRUE(continuous.update(Emrys::PoissonEmission<Space>{rates, k % 3, 0.1}));
        static_cast<void>(continuous.predict(0.2, Emrys::ArrivalInterval<Space>{rates}));
    });

    auto smoothed = Emrys::HiddenMarkovFilter{chain};
    auto window = Emrys::CategoricalHistory<Space>{Lag{5}};
    auto const step = [&](int k) {
        window.push(smoothed.predictJoint());
        EXPECT_TRUE(smoothed.update(table.observe(k % 2)));
        window.smooth(smoothed.state());
    };
    for (int k = 0; k < 5; ++k)
    {
        step(k);
    }
    expectStepsAllocateNothing(step);
    EXPECT_EQ(window.size(), 5);
}

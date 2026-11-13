// Allocation control [allocation] for the particle filters: the particle filter, its fixed-lag smoother
// (step 14, [smoothing]), and the Rao-Blackwellized particle and point-mass filters over pmr states take
// their allocators and step within their resources after warm-up, with fixed-step models too
// [fixed-step-models], the particle filter under a user's policy that brings its memory, off the global heap
// [execution-policies], its prediction with an uncertain input from construction on, Roughen, the kernel density
// estimate, drawSample() and drawSampleWithLogDensity() between steps, and a Monte Carlo divergence from the
// estimate, every resampler, the prunes of the particle states, and the Rao-Blackwellized states'
// collapsed(). Part of the allocation test target; the counters and fixtures are in allocation.h.

#include <EmrysFilter/Distributions/Conditionals.h>
#include <EmrysFilter/Distributions/Divergence.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/ParticleGaussian.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Filters/RbParticleFilter.h>
#include <EmrysFilter/Filters/RbPointMassFilter.h>
#include <EmrysFilter/Smoothers/ParticleHistory.h>

#include <gtest/gtest.h>

#include "allocation.h"
#include "user_policies.h"

#include <cmath>
#include <functional>
#include <memory_resource>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>

using namespace EmrysTest::Allocation;

using Emrys::FixedStepModel;
using Emrys::Lag;
using Emrys::ParticleFilter;
using Emrys::ParticleHistory;
using Emrys::ParticleSet;
using Emrys::PerPointCovariance;
using Emrys::RandomWalk;
using Emrys::RbParticleFilter;
using Emrys::RbPointMassFilter;
using Emrys::regularGrid;
using Emrys::SharedCovariance;

namespace resamplers = Emrys::resamplers;
namespace upkeep = Emrys::upkeep;

template <typename Model, typename Predict>
void
expectParticleFilterStepsWithinItsResource(Model const & model, Predict const & predict)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto rng = std::mt19937_64{1};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        allocator
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{model, std::move(particles)};
    EXPECT_EQ(filter.get_allocator(), allocator);

    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    expectStepsAllocateNothing(
        [&](int k) {
            predict(filter, rng);
            static_cast<void>(filter.update(makeFix(fixAt(k))));
            static_cast<void>(resampling(filter.state()));
        },
        &resource
    );
    EXPECT_EQ(filter.state().size(), particleCount);
}

// A ParticleFilter over a pmr ParticleSet with a set capacity takes its allocator, and predict,
// update, and resampling allocate nothing after warm-up, with a fixed-step model too.
TEST(Allocation, ParticleFilterStepsWithinItsResource)
{
    expectParticleFilterStepsWithinItsResource(makeModel(), [](auto & filter, auto & rng) {
        filter.predict(rng, timeStep);
    });
    expectParticleFilterStepsWithinItsResource(
        FixedStepModel{makeModel(), State{}, timeStep},
        [](auto & filter, auto & rng) {
            filter.predict(rng);
        }
    );
}

namespace
{

// Constant velocity as a batched function, with continuous noise on the velocity.
auto
batchedModel()
{
    return Emrys::processModel(
        Emrys::batched([](auto const & in, auto const & out, auto const & x, double dt, auto & y) {
            using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
            Emrys::block<Position>(out, y.values()) =
                Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(out, y.values()) = Emrys::block<Velocity>(in, x);
        }),
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(0.1))
    );
}

// The position fix of makeFix() as a batched function.
auto
batchedFix(Eigen::Vector2d const & z)
{
    return observe(
        Emrys::measurementModel(
            Emrys::batched([](auto const & in, auto const &, auto const & x, auto & y) {
                y.values() = Emrys::block<Position>(in, x);
            }),
            Emrys::reads(Emrys::path<Position>),
            additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}})
        ),
        z
    );
}

}  // namespace

// A particle filter whose process and measurement models are batched keeps each batch's input and
// output in scratch from its own resource, so its predict and update allocate nothing after warm-up
// ([batched-functions], [allocation]).
TEST(Allocation, ParticleFilterWithBatchedModelsStepsWithinItsResource)
{
    auto resource = CountingResource{arenaBytes};
    auto rng = std::mt19937_64{3};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        Pmr{&resource}
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{batchedModel(), std::move(particles)};
    expectStepsAllocateNothing(
        [&](int k) {
            filter.predict(rng, timeStep);
            static_cast<void>(filter.update(batchedFix(fixAt(k))));
        },
        &resource
    );
}

// A particle filter given particles from another resource by setState() keeps its own, as a pmr
// assignment does. Given a larger set whose allocator propagates on assignment and differs from
// its own, it takes that allocator, and its scratch follows: the steps after allocate nothing
// through the allocator it had.
TEST(Allocation, ParticleFilterSetStateFollowsTheAllocator)
{
    auto rng = std::mt19937_64{1};
    auto const initial = makeInitial();
    auto const draw = [&](auto const & allocator, Eigen::Index count) {
        return ParticleSet<State, double, std::remove_cvref_t<decltype(allocator)>>{
            State{},
            Emrys::drawSamples(initial.toDensity().value(), count, rng),
            allocator
        };
    };
    auto resource = CountingResource{arenaBytes};
    auto other = std::pmr::monotonic_buffer_resource{};
    auto kept = ParticleFilter{makeModel(), draw(Pmr{&resource}, particleCount)};
    kept.setState(draw(Pmr{&other}, 2 * particleCount));
    EXPECT_EQ(kept.get_allocator(), Pmr{&resource});
    EXPECT_EQ(kept.state().size(), 2 * particleCount);
    kept.predict(rng, timeStep);
    EXPECT_TRUE(kept.update(makeFix(fixAt(1)))->accepted);

    using Allocator = Propagating<std::byte>;
    auto filter = ParticleFilter{makeModel(), draw(Allocator{1}, particleCount)};
    filter.predict(rng, timeStep);
    EXPECT_TRUE(filter.update(makeFix(fixAt(0)))->accepted);
    filter.setState(draw(Allocator{2}, 2 * particleCount));
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    auto const before = propagatingAllocations;
    filter.predict(rng, timeStep);
    EXPECT_TRUE(filter.update(makeFix(fixAt(1)))->accepted);
    EXPECT_TRUE(upkeep::resample(rng)(filter.state()));
    EXPECT_EQ(propagatingAllocations[1], before[1]);
}

namespace
{

// A policy of the user's own that brings its memory: it visits the indices in reverse on the
// calling thread, and its get_allocator() hands out pmr memory, which the library's primitives
// take their scratch from under it [execution-policies].
struct PolicyWithMemory : EmrysTest::ReversedPolicy
{
    std::pmr::memory_resource * resource = nullptr;

    [[nodiscard]] Pmr
    get_allocator() const
    {
        return Pmr{resource};
    }
};

}  // namespace

// Under a user's policy with an allocator, a particle filter of several chunks predicts, updates,
// and resamples when its effective sample size falls without the global heap, as under Sequential:
// its own buffers come from its resource after warm-up, and the primitives' scratch from the
// policy's memory, which a pool reuses after warm-up.
TEST(Allocation, ParticleFilterUnderAUsersPolicyStaysOffTheHeap)
{
    constexpr Eigen::Index count = 3 * Emrys::internal::chunkSize + 5;
    auto resource = CountingResource{arenaBytes};
    auto scratch = CountingResource{arenaBytes};
    // Pools well past the biggest scratch, a few Eigen::Index per particle, so that the pool
    // reuses every block whatever the standard library's default largest block. libc++ pools
    // blocks below the largest only loosely: at 64 KiB it sent a 24 KiB scratch upstream every
    // step.
    auto const options = std::pmr::pool_options{.max_blocks_per_chunk = 0, .largest_required_pool_block = 1 << 20};
    auto pool = std::pmr::unsynchronized_pool_resource{options, &scratch};
    auto const policy = PolicyWithMemory{{}, &pool};
    auto rng = std::mt19937_64{1};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), count, rng),
        Pmr{&resource}
    };
    particles.setCapacity(count);
    auto filter = ParticleFilter{makeModel(), std::move(particles)};
    auto const resampling = upkeep::whenEss(policy, std::less{}, 0.5, upkeep::resample(policy, rng));
    auto const step = [&](int k) {
        filter.predict(policy, rng, timeStep);
        static_cast<void>(filter.update(policy, makeFix(fixAt(k))));
        static_cast<void>(resampling(filter.state()));
    };
    step(0);
    int const warmed = scratch.allocations();
    EXPECT_GT(warmed, 0) << "the scratch came from the policy's memory";
    expectStepsAllocateNothing(step, &resource);
    EXPECT_EQ(scratch.allocations(), warmed) << "allocations from the policy's memory after warm-up";
}

namespace
{

// The specific force an accelerometer reads, an uncertain input of the call that a particle filter
// draws once per particle, two more rows of each particle's normals.
struct SpecificForce : Linear<2>
{};

// Velocity driven by the measured acceleration.
template <typename P, typename V>
auto
accelerometerDriven()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Eigen::Matrix<T, 2, 1> const a = Emrys::block<SpecificForce>(in, x);
            Emrys::block<P>(out, y) = Emrys::block<P>(in, x) + T(dt) * Emrys::block<V>(in, x) + T(dt * dt / 2) * a;
            Emrys::block<V>(out, y) = Emrys::block<V>(in, x) + T(dt) * a;
            return y;
        },
        Emrys::reads(Emrys::path<P>, Emrys::path<V>, Emrys::path<SpecificForce>),
        Emrys::writes(Emrys::path<P>, Emrys::path<V>),
        Emrys::continuousNoise(Emrys::on<V>(0.05))
    );
}

auto
accelerated()
{
    return accelerometerDriven<Position, Velocity>();
}

GaussianParams<SpecificForce>
makeAccelerometer()
{
    return GaussianParams<SpecificForce>{Eigen::Vector2d{0.3, -0.1}, Eigen::Matrix2d{{0.2, 0.05}, {0.05, 0.1}}};
}

// A position fix with Laplace noise, known only by its likelihood [likelihood-measurements].
auto
makeLaplaceFix(Eigen::Vector2d const & z)
{
    return observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Position>),
            Emrys::logLikelihood([](auto const & in, auto const & x, auto const & observed) {
                return -Eigen::Vector2d{observed - Emrys::block<Position>(in, x)}.lpNorm<1>();
            })
        ),
        z
    );
}

}  // namespace

// A particle filter predicting with an uncertain input under Sequential draws one particle's
// normals at a time, so from its first prediction on it asks neither its resource nor the heap for
// memory: no warm-up is excused, since nothing grows to hold the input's rows [allocation].
TEST(Allocation, ParticleFilterPredictsWithAnUncertainInputFromConstructionOn)
{
    auto resource = CountingResource{arenaBytes};
    auto rng = std::mt19937_64{7};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        Pmr{&resource}
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{accelerated(), std::move(particles)};
    auto const accelerometer = makeAccelerometer();
    int const constructed = resource.allocations();
    int heap = 0;
    {
        auto const watch = HeapWatch{};
        for (int k = 0; k < 100; ++k)
        {
            filter.predict(rng, timeStep, accelerometer);
        }
        heap = watch.allocations();
    }
    EXPECT_EQ(heap, 0) << "allocations from the global heap";
    EXPECT_EQ(resource.allocations(), constructed) << "allocations from the resource after construction";
}

// Under a user's policy with an allocator, the same prediction draws every particle's normals,
// the input's rows among them, into the filter's own scratch, which the first prediction grows
// past the noise's rows it was reserved for; after that one, neither the filter's resource, the
// policy's memory, nor the heap gives anything [allocation], [execution-policies].
TEST(Allocation, ParticleFilterUnderAUsersPolicyReusesItsDrawsForAnUncertainInput)
{
    constexpr Eigen::Index count = 3 * Emrys::internal::chunkSize + 5;
    auto resource = CountingResource{arenaBytes};
    auto scratch = CountingResource{arenaBytes};
    auto const policy = PolicyWithMemory{{}, &scratch};
    auto rng = std::mt19937_64{9};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), count, rng),
        Pmr{&resource}
    };
    particles.setCapacity(count);
    auto filter = ParticleFilter{accelerated(), std::move(particles)};
    auto const accelerometer = makeAccelerometer();
    auto const step = [&](int) {
        filter.predict(policy, rng, timeStep, accelerometer);
    };
    int const constructed = resource.allocations();
    step(0);
    EXPECT_GT(resource.allocations(), constructed) << "the first prediction grows the draws";
    int const warmed = scratch.allocations();
    expectStepsAllocateNothing(step, &resource);
    EXPECT_EQ(scratch.allocations(), warmed) << "allocations from the policy's memory after warm-up";
}

// What a system with no heap runs on a particle set between filter steps allocates nothing [allocation]:
// the set's moments and Silverman's bandwidth from them, taken one particle at a time; Roughen
// under Sequential, which draws and moves one particle at a time; the kernel density estimate's
// log-density, one particle at a time; drawSample() and drawSampleWithLogDensity() of it, and of a
// Gaussian, which return one draw in a vector sized by the space; and a Monte Carlo divergence from
// it under Sequential, which takes its draws that way and stores none.
TEST(Allocation, RoughenKernelDensityAndSingleDrawsAllocateNothing)
{
    auto resource = CountingResource{arenaBytes};
    auto rng = std::mt19937_64{3};
    auto const initial = makeInitial();
    auto const gaussian = initial.toDensity().value();
    auto particles =
        ParticleSet<State, double, Pmr>{State{}, Emrys::drawSamples(gaussian, particleCount, rng), Pmr{&resource}};
    particles.setCapacity(particleCount);
    auto const x = Vector4{0.5, -0.5, 0.1, 0.0};
    double total = 0;
    expectStepsAllocateNothing(
        [&](int) {
            auto const moments = particles.moments();
            ASSERT_TRUE(moments);
            total += moments->covarianceLower().trace();
            ASSERT_TRUE(Emrys::upkeep::roughen(rng, 0.5)(particles));
            auto const kde = particles.toDensity();
            ASSERT_TRUE(kde);
            total += kde->logDensity(x);
            total += Emrys::drawSample(*kde, rng).sum();
            total += Emrys::drawSample(gaussian, rng).sum();
            total += Emrys::drawSampleWithLogDensity(*kde, rng).logDensity;
            total += Emrys::drawSampleWithLogDensity(gaussian, rng).logDensity;
            total += Emrys::kullbackLeibler(Emrys::MonteCarlo{50}, *kde, gaussian, rng).value;
        },
        &resource
    );
    EXPECT_TRUE(std::isfinite(total));
}

// Every resampler resamples a pmr ParticleSet within its resource after warm-up [allocation]: the
// stratified and systematic resamplers need no scratch, and the multinomial and residual ones
// take the set's (ResamplerWithScratch), where their three-argument forms would allocate. The
// weights are made unequal first, so that the residual resamplers have a remainder to draw.
template <typename R>
void
expectResamplingAllocatesNothing(R const & resampler)
{
    auto resource = CountingResource{arenaBytes};
    auto rng = std::mt19937_64{5};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        Pmr{&resource}
    };
    particles.setCapacity(particleCount);
    expectStepsAllocateNothing(
        [&](int) {
            particles.setLogWeights(-0.5 * particles.states().row(0).transpose().array().square());
            ASSERT_TRUE(particles.resample(rng, resampler));
        },
        &resource
    );
    EXPECT_EQ(particles.size(), particleCount);
}

TEST(Allocation, EveryResamplerStaysWithinTheResource)
{
    expectResamplingAllocatesNothing(resamplers::multinomial());
    expectResamplingAllocatesNothing(resamplers::stratified());
    expectResamplingAllocatesNothing(resamplers::systematic());
    expectResamplingAllocatesNothing(resamplers::residual());
    expectResamplingAllocatesNothing(resamplers::residual(resamplers::systematic()));
}

// A copy of a pmr ParticleSet keeps its source's resource and capacity, not the default resource
// select_on_container_copy_construction() would give, so a filter built from a copy of another's
// state, and a copy of a whole filter with its scratch, step without a single allocation from the
// global heap or the resource, the first step included [allocation].
TEST(Allocation, CopiesKeepTheirSourcesResourceAndCapacity)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto rng = std::mt19937_64{1};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        allocator
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{makeModel(), std::move(particles)};

    ParticleSet<State, double, Pmr> const copied = filter.state();
    EXPECT_EQ(copied.get_allocator(), allocator);
    EXPECT_EQ(copied.capacity(), particleCount);

    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    auto const expectNoAllocations = [&](auto & stepped) {
        int const reserved = resource.allocations();
        int heap = 0;
        {
            auto const watch = HeapWatch{};
            for (int k = 0; k < 10; ++k)
            {
                stepped.predict(rng, timeStep);
                static_cast<void>(stepped.update(makeFix(fixAt(k))));
                static_cast<void>(resampling(stepped.state()));
            }
            heap = watch.allocations();
        }
        EXPECT_EQ(heap, 0) << "allocations from the global heap";
        EXPECT_EQ(resource.allocations(), reserved) << "allocations from the resource";
    };

    auto fromCopy = ParticleFilter{makeModel(), filter.state()};
    EXPECT_EQ(fromCopy.get_allocator(), allocator);
    expectNoAllocations(fromCopy);

    auto wholeCopy = filter;
    EXPECT_EQ(wholeCopy.get_allocator(), allocator);
    expectNoAllocations(wholeCopy);
}

// A ParticleFilter recording its particles in a fixed-lag ParticleHistory over the same resource
// and tracing the window after each step, resampling when the effective sample size falls, takes
// nothing from the resource or the heap once the window is full: a full window copies into its
// oldest node's sets and parents [allocation], [smoothing].
TEST(Allocation, FixedLagParticleSmoothingStepsWithinItsResource)
{
    static constexpr Eigen::Index lag = 5;
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto rng = std::mt19937_64{3};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        allocator
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{makeModel(), std::move(particles)};
    auto window = ParticleHistory<State, double, Pmr>{Lag{lag}, allocator};
    EXPECT_EQ(window.get_allocator(), allocator);

    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    auto const step = [&](int k) {
        window.push(filter.state());
        filter.predict(rng, timeStep);
        static_cast<void>(filter.update(makeFix(fixAt(k))));
        static_cast<void>(resampling(filter.state()));
        window.smooth(filter.state());
    };
    for (int k = 0; k < lag; ++k)
    {
        step(k);
    }
    expectStepsAllocateNothing(step, &resource);
    EXPECT_EQ(window.size(), lag);
    EXPECT_GT(filter.state().resampleCount(), lag);
    expectHistoryIsAllocatorAware(window);
}

template <typename Covariance, typename Model>
void
expectRbpFilterStepsWithinItsResource(Model const & model)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto rng = std::mt19937_64{2};
    auto state = Emrys::drawParticleGaussian<Position>(makeInitial(), particleCount, rng, Covariance{}, allocator);
    state.setCapacity(particleCount);
    auto filter = RbParticleFilter{model, std::move(state)};
    EXPECT_EQ(filter.get_allocator(), allocator);
    // The abbreviation deduces the same filter.
    static_assert(std::is_same_v<decltype(Emrys::RbParticleFilter{model, filter.state()}), decltype(filter)>);
    // The updates refill one result, which owns its arrays; update() would make a new one.
    auto result = typename decltype(filter)::UpdateType{filter.get_allocator()};
    EXPECT_EQ(result.get_allocator(), allocator);
    auto const updateInto = [&](auto const & measurement) {
        if constexpr (std::is_same_v<Covariance, SharedCovariance>)
        {
            return filter.update(measurement, result);
        }
        else
        {
            return filter.update(measurement, Emrys::reductions::collapse(), result);
        }
    };

    auto const resampling = upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng));
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter, rng);
            EXPECT_TRUE(updateInto(makeFix<Position>(fixAt(k))));
            EXPECT_TRUE(updateInto(makeFix<Velocity>(Eigen::Vector2d{1.0, 0.5})));
            EXPECT_TRUE(filter.update(makeLaplaceFix(fixAt(k)), result));
            static_cast<void>(resampling(filter.state()));
            // The whole-state readout streams the particles [allocation].
            static_cast<void>(filter.state().toDensity()->collapsed());
        },
        &resource
    );
    EXPECT_EQ(filter.state().size(), particleCount);
}

// An RbParticleFilter over a pmr ParticleGaussian, under both covariance forms, takes its allocator and runs
// predict, two updates and a likelihood into one result, and resampling without allocating after warm-up,
// with a fixed-step model too, whose prediction given each particle takes the cached transition.
TEST(Allocation, RbpFilterStepsWithinItsResource)
{
    auto const fixedStep = FixedStepModel{makeModel(), State{}, timeStep};
    expectRbpFilterStepsWithinItsResource<SharedCovariance>(makeModel());
    expectRbpFilterStepsWithinItsResource<PerPointCovariance>(makeModel());
    expectRbpFilterStepsWithinItsResource<SharedCovariance>(fixedStep);
    expectRbpFilterStepsWithinItsResource<PerPointCovariance>(fixedStep);
}

// The three prunes of a particle filter's state and of a Rao-Blackwellized particle filter's, with
// a set capacity, allocate nothing after warm-up, each round followed by a resample back to the
// capacity: the order of the weights takes the set's scratch, and the Rao-Blackwellized state's
// list of the particles kept its own.
TEST(Allocation, ParticlePrunesStayWithinTheResource)
{
    auto const prune = [](auto & state, auto & rng) {
        static_cast<void>(state.pruneBelow(std::log(1e-6)));
        static_cast<void>(state.pruneTail(std::log(1e-3)));
        static_cast<void>(state.pruneToSize(particleCount / 2));
        EXPECT_TRUE(state.resample(rng, particleCount));
    };

    auto resource = CountingResource{arenaBytes};
    auto rng = std::mt19937_64{1};
    auto const initial = makeInitial();
    auto particles = ParticleSet<State, double, Pmr>{
        State{},
        Emrys::drawSamples(initial.toDensity().value(), particleCount, rng),
        Pmr{&resource}
    };
    particles.setCapacity(particleCount);
    auto filter = ParticleFilter{makeModel(), std::move(particles)};
    expectStepsAllocateNothing(
        [&](int k) {
            filter.predict(rng, timeStep);
            static_cast<void>(filter.update(makeFix(fixAt(k))));
            prune(filter.state(), rng);
        },
        &resource
    );
    EXPECT_EQ(filter.state().size(), particleCount);

    auto rbResource = CountingResource{arenaBytes};
    auto state =
        Emrys::drawParticleGaussian<Position>(initial, particleCount, rng, SharedCovariance{}, Pmr{&rbResource});
    state.setCapacity(particleCount);
    auto rbFilter = RbParticleFilter{makeModel(), std::move(state)};
    auto result = typename decltype(rbFilter)::UpdateType{rbFilter.get_allocator()};
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(rbFilter, rng);
            EXPECT_TRUE(rbFilter.update(makeFix<Position>(fixAt(k)), result));
            prune(rbFilter.state(), rng);
        },
        &rbResource
    );
    EXPECT_EQ(rbFilter.state().size(), particleCount);
}

// A Rao-Blackwellized particle filter given a larger state whose allocator propagates on assignment
// and differs from its own takes it by setState(), and its scratch follows: the steps after
// allocate nothing through the allocator it had.
template <typename Covariance>
void
expectRbpFilterScratchFollowsAPropagatingAllocator()
{
    using Allocator = Propagating<std::byte>;
    auto rng = std::mt19937_64{2};
    auto const draw = [&](int id, Eigen::Index count) {
        return Emrys::drawParticleGaussian<Position>(makeInitial(), count, rng, Covariance{}, Allocator{id});
    };
    auto filter = RbParticleFilter{makeModel(), draw(1, particleCount)};
    filter.predict(rng, timeStep);
    ASSERT_TRUE(filter.update(makeFix<Position>(fixAt(0))));
    filter.setState(draw(2, 2 * particleCount));
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    EXPECT_EQ(filter.state().size(), 2 * particleCount);
    auto const before = propagatingAllocations;
    filter.predict(rng, timeStep);
    ASSERT_TRUE(filter.update(makeFix<Position>(fixAt(1))));
    ASSERT_TRUE(filter.update(makeFix<Velocity>(Eigen::Vector2d{1.0, 0.5})));
    EXPECT_TRUE(upkeep::resample(rng)(filter.state()));
    EXPECT_EQ(propagatingAllocations[1], before[1]);
}

TEST(Allocation, RbpFilterScratchFollowsAPropagatingAllocator)
{
    expectRbpFilterScratchFollowsAPropagatingAllocator<SharedCovariance>();
    expectRbpFilterScratchFollowsAPropagatingAllocator<PerPointCovariance>();
}

template <typename Covariance, bool FixedStep>
void
expectRbpmFilterStepsWithinItsResource()
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto grid =
        regularGrid<State, Position>(Eigen::Vector2d{-4, -4}, Eigen::Vector2d{2, 2}, {5, 5}, std::nullopt, allocator);
    using Rest = typename decltype(grid)::RestSpace;
    auto const velocity = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.5}, Eigen::Matrix2d::Identity()};
    // Moved in: a copy would take the allocator select_on_container_copy_construction() gives.
    auto const model = RandomWalk<Velocity>{0.01};
    auto const chosen = [&] {
        if constexpr (FixedStep)
        {
            return FixedStepModel{model, Rest{}, timeStep};
        }
        else
        {
            return model;
        }
    }();
    auto filter = RbPointMassFilter{chosen, std::move(grid), velocity, Covariance{}};
    EXPECT_EQ(filter.get_allocator(), allocator);
    // The updates refill one result, which owns its arrays; update() would make a new one.
    auto result = typename decltype(filter)::UpdateType{filter.get_allocator()};
    EXPECT_EQ(result.get_allocator(), allocator);
    auto const updateInto = [&](auto const & measurement) {
        if constexpr (std::is_same_v<Covariance, SharedCovariance>)
        {
            return filter.update(measurement, result);
        }
        else
        {
            return filter.update(measurement, Emrys::reductions::collapse(), result);
        }
    };

    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            EXPECT_TRUE(updateInto(makeFix<Position>(Eigen::Vector2d{0.5, -0.5} + 0.1 * fixAt(k) / (k + 1))));
            EXPECT_TRUE(updateInto(makeFix<Velocity>(Eigen::Vector2d{1.0, 0.5})));
            EXPECT_TRUE(filter.update(makeLaplaceFix(Eigen::Vector2d{0.5, -0.5}), result));
            static_cast<void>(upkeep::pruneBelow(1e-12)(filter.state()));
            static_cast<void>(filter.state().toDensity()->collapsed());
        },
        &resource
    );
}

// An RbPointMassFilter over a pmr grid moved in, under both covariance forms, takes its allocator and runs
// predict, two updates and a likelihood into one result, and PruneBelow without allocating after warm-up, with a
// fixed-step model over the rest too.
TEST(Allocation, RbpmFilterStepsWithinItsResource)
{
    expectRbpmFilterStepsWithinItsResource<SharedCovariance, false>();
    expectRbpmFilterStepsWithinItsResource<PerPointCovariance, false>();
    expectRbpmFilterStepsWithinItsResource<SharedCovariance, true>();
    expectRbpmFilterStepsWithinItsResource<PerPointCovariance, true>();
}

// A Rao-Blackwellized point-mass filter given a state over a larger grid whose allocator propagates
// on assignment and differs from its own takes it by setState(), and its scratch follows: the
// steps after allocate nothing through the allocator it had.
template <typename Covariance>
void
expectRbpmFilterScratchFollowsAPropagatingAllocator()
{
    using Allocator = Propagating<std::byte>;
    auto const makeGrid = [](int id, Eigen::Index side) {
        return regularGrid<State, Position>(
            Eigen::Vector2d{-4, -4},
            Eigen::Vector2d{2, 2},
            {side, side},
            std::nullopt,
            Allocator{id}
        );
    };
    using Rest = typename decltype(makeGrid(1, 3))::RestSpace;
    auto const velocity = GaussianParams<Rest>{Rest{}, Eigen::Vector2d{1.0, 0.5}, Eigen::Matrix2d::Identity()};
    auto filter = RbPointMassFilter{RandomWalk<Velocity>{0.01}, makeGrid(1, 3), velocity, Covariance{}};
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeFix<Position>(Eigen::Vector2d{0.5, -0.5})));
    filter.setState(typename decltype(filter)::StateType{makeGrid(2, 5), velocity});
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    EXPECT_EQ(filter.state().size(), 25);
    auto const before = propagatingAllocations;
    filter.predict(timeStep);
    ASSERT_TRUE(filter.update(makeFix<Position>(Eigen::Vector2d{0.5, -0.5})));
    ASSERT_TRUE(filter.update(makeFix<Velocity>(Eigen::Vector2d{1.0, 0.5})));
    EXPECT_EQ(propagatingAllocations[1], before[1]);
}

TEST(Allocation, RbpmFilterScratchFollowsAPropagatingAllocator)
{
    expectRbpmFilterScratchFollowsAPropagatingAllocator<SharedCovariance>();
    expectRbpmFilterScratchFollowsAPropagatingAllocator<PerPointCovariance>();
}

// Allocation control [allocation] for the mixture filters: the Gaussian mixture filter, the IMM, the
// autonomous multiple model filter with its mode floor (step 17c, [autonomous-multiple-model]), and a mixture of IMMs
// (step 15c, [component-first], [filters-and-recursions]) over pmr states take their allocators and step within their
// resources after warm-up, with fixed-step models too [fixed-step-models], and so do fixed-lag IMM and Gaussian mixture
// smoothing (step 14, [smoothing]). Part of
// the allocation test target; the counters and fixtures are in allocation.h.

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Filters/AmmFilter.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/ImmFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Smoothers/GaussianMixtureHistory.h>
#include <EmrysFilter/Smoothers/HybridGaussianHistory.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/SwitchingModel.h>

#include <gtest/gtest.h>

#include "allocation.h"

#include <cmath>
#include <optional>
#include <type_traits>
#include <utility>

using namespace EmrysTest::Allocation;

using Emrys::AmmFilter;
using Emrys::Capacity;
using Emrys::CategoricalParams;
using Emrys::FixedStepModel;
using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureHistory;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianMixturePredictedJoint;
using Emrys::HybridGaussian;
using Emrys::HybridGaussianHistory;
using Emrys::HybridGaussianParams;
using Emrys::HybridPredictedJoint;
using Emrys::ImmFilter;
using Emrys::Lag;
using Emrys::MarkovChain;
using Emrys::MixtureFilter;
using Emrys::MixtureParams;
using Emrys::switchingModel;

namespace reductions = Emrys::reductions;
namespace upkeep = Emrys::upkeep;

template <typename Model>
void
expectGaussianMixtureFilterStepsWithinItsResource(Model const & model)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto prior = GaussianMixtureParams<State, double, Pmr>{Capacity{2}, allocator};
    prior.add(std::log(0.6), makeInitial().moments());
    prior.add(std::log(0.4), Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower());
    auto filter = GaussianMixtureFilter{model, std::move(prior)};
    EXPECT_EQ(filter.get_allocator(), allocator);

    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    EXPECT_EQ(result.get_allocator(), allocator);
    result.components.reserve(2);
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
            EXPECT_EQ(result.posteriorIndices(1).back() + 1, filter.state().size());
            ASSERT_TRUE(filter.state().pruneToSize(2));
        },
        &resource
    );
    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(filter.state().size(), 2);
}

// A GaussianMixtureFilter over a pmr mixture takes the mixture's allocator, as does a result made
// from it, and with a reserved result and truncation its steps ask neither the heap nor the
// resource for memory after warm-up, with a fixed-step model too; the result's posterior indices,
// reserved by the state's capacity, are read at each step.
TEST(Allocation, GaussianMixtureFilterStepsWithinItsResource)
{
    expectGaussianMixtureFilterStepsWithinItsResource(makeModel());
    expectGaussianMixtureFilterStepsWithinItsResource(FixedStepModel{makeModel(), State{}, timeStep});
}

// Under mixture noise each component's update has a run-time number of branches, kept with their
// posteriors in its slot of the reused result; under Collapse the components are updated where
// they are, and under Keep the policy moves every branch into the posterior mixture through the
// filter's output, with no mixture built per component. With a reserved result and truncation the
// steps ask neither the heap nor the resource for memory after warm-up ([allocation], [innovation-statistics] as
// amended), with the result's posterior indices, which Keep doubles, reserved by the state's capacity.
TEST(Allocation, GaussianMixtureFilterWithMixtureNoiseStepsWithinItsResource)
{
    auto const run = [](auto const & policy, Eigen::Index capacity) {
        auto resource = CountingResource{arenaBytes};
        auto const allocator = Pmr{&resource};
        auto prior = GaussianMixtureParams<State, double, Pmr>{Capacity{capacity}, allocator};
        prior.add(std::log(0.6), makeInitial().moments());
        prior.add(std::log(0.4), Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower());
        auto filter = GaussianMixtureFilter{makeModel(), std::move(prior)};
        auto z = Eigen::Vector2d{};
        auto const fix = makeMixtureFix(z);
        using Update = typename decltype(filter.update(fix, policy))::value_type;
        auto result = Update{allocator};
        result.components.reserve(2);
        expectStepsAllocateNothing(
            [&](int k) {
                z = fixAt(k);
                predictOverTheStep(filter);
                ASSERT_TRUE(filter.update(fix, policy, result));
                EXPECT_EQ(result.posteriorIndices(1).back() + 1, filter.state().size());
                ASSERT_TRUE(filter.state().pruneToSize(2));
            },
            &resource
        );
        EXPECT_TRUE(result.accepted);
        EXPECT_EQ(filter.state().size(), 2);
        EXPECT_EQ(result.components.front().branches.size(), 2);
    };
    run(reductions::collapse(), 2);
    run(reductions::keep(), 4);
}

// A gate that passes some of a run-time number of branches hands its inner policy their indices
// from the stack, so reducing them allocates nothing: here the outlier branch alone passes, its
// NIS below the inlier's [allocation], [reduction-policies].
TEST(Allocation, AGateOverARunTimeNumberOfBranchesAllocatesNothing)
{
    auto const updated = Emrys::update(makeInitial(), makeMixtureFix(fixAt(1)));
    ASSERT_TRUE(updated);
    ASSERT_EQ(updated->branches.size(), 2u);
    double const inlier = updated->branches[0].nis();
    double const outlier = updated->branches[1].nis();
    ASSERT_LT(outlier, inlier);
    auto const gate = reductions::gate(Emrys::NisThreshold{(inlier + outlier) / 2});
    auto kept = std::optional<GaussianParams<State>>{};
    expectStepsAllocateNothing([&](int) {
        kept = Emrys::reduceToOne<GaussianParams<State>>(gate, updated->branches);
        ASSERT_TRUE(kept);
    });
    EXPECT_EQ(kept->mean(), updated->branches[1].posterior.mean());
}

// An IMM under mixture noise keeps each mode's branches in its slot of the reused result, and its
// steps allocate nothing after warm-up [allocation].
TEST(Allocation, ImmFilterWithMixtureNoiseStepsWithinItsResource)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    auto initial = HybridGaussianParams<State, Mode, double, Pmr>{CategoricalParams<Mode>{}, makeInitial(), allocator};
    auto filter = ImmFilter{switchingModel<Maneuver>(chain, makeModel(0.01), makeModel(4.0)), std::move(initial)};
    auto z = Eigen::Vector2d{};
    auto const fix = makeMixtureFix(z);
    using Update = typename decltype(filter.update(fix))::value_type;
    auto result = Update{allocator};
    EXPECT_EQ(result.get_allocator(), allocator);
    result.modes.reserve(4);
    expectStepsAllocateNothing(
        [&](int k) {
            z = fixAt(k);
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(fix, reductions::collapse(), result));
        },
        &resource
    );
    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(result.modes.front().branches.size(), 2);
}

// A copy of a pmr mixture keeps its source's resource and capacity, and a filter built from it
// steps within that resource [allocation].
TEST(Allocation, MixtureCopyKeepsItsSourcesResourceAndCapacity)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto prior = GaussianMixtureParams<State, double, Pmr>{Capacity{2}, allocator};
    prior.add(std::log(0.6), makeInitial().moments());
    prior.add(std::log(0.4), Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower());

    GaussianMixtureParams<State, double, Pmr> const copied = prior;
    EXPECT_EQ(copied.get_allocator(), allocator);
    EXPECT_EQ(copied.capacity(), 2);

    auto filter = GaussianMixtureFilter{makeModel(), copied};
    EXPECT_EQ(filter.get_allocator(), allocator);
    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    result.components.reserve(2);
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
            ASSERT_TRUE(filter.state().pruneToSize(2));
        },
        &resource
    );
}

template <typename Quiet, typename Agile>
void
expectImmFilterStepsWithinItsResource(Quiet const & quiet, Agile const & agile)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    auto initial = HybridGaussianParams<State, Mode, double, Pmr>{CategoricalParams<Mode>{}, makeInitial(), allocator};
    auto filter = ImmFilter{switchingModel<Maneuver>(chain, quiet, agile), std::move(initial)};
    EXPECT_EQ(filter.get_allocator(), allocator);
    // The abbreviation deduces the same filter.
    auto const abbreviated = Emrys::ImmFilter{switchingModel<Maneuver>(chain, quiet, agile), filter.state()};
    static_assert(std::is_same_v<decltype(abbreviated), const decltype(filter)>);

    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    result.modes.reserve(4);
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
        },
        &resource
    );
    EXPECT_TRUE(result.accepted);

    // A state from another resource moves into the filter's, as a pmr assignment does.
    auto other = std::pmr::monotonic_buffer_resource{};
    filter.setState(
        HybridGaussianParams<State, Mode, double, Pmr>{CategoricalParams<Mode>{}, makeInitial(), Pmr{&other}}
    );
    EXPECT_EQ(filter.get_allocator(), allocator);
    EXPECT_EQ(filter.recursion().get_allocator(), allocator);
    EXPECT_EQ(&filter.recursion().model(), &std::as_const(filter).model());
}

// An IMM over a pmr HybridGaussian takes its allocator, as does a result made from it, and with a
// reserved result its steps allocate nothing after warm-up, with fixed-step models too, and
// setState() keeps its resource.
TEST(Allocation, ImmFilterStepsWithinItsResource)
{
    expectImmFilterStepsWithinItsResource(makeModel(0.01), makeModel(4.0));
    expectImmFilterStepsWithinItsResource(
        FixedStepModel{makeModel(0.01), State{}, timeStep},
        FixedStepModel{makeModel(4.0), State{}, timeStep}
    );
}

// An IMM given a state whose allocator propagates on assignment and differs from its own takes it,
// and its scratch follows, so the steps after run with the new allocator throughout.
TEST(Allocation, ImmFilterScratchFollowsAPropagatingAllocator)
{
    using Allocator = Propagating<std::byte>;
    using HybridState = HybridGaussianParams<State, Mode, double, Allocator>;
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    auto filter = ImmFilter{
        switchingModel<Maneuver>(chain, makeModel(0.01), makeModel(4.0)),
        HybridState{CategoricalParams<Mode>{}, makeInitial(), Allocator{1}}
    };
    EXPECT_EQ(filter.recursion().get_allocator(), Allocator{1});
    filter.setState(HybridState{CategoricalParams<Mode>{}, makeInitial(), Allocator{2}});
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    EXPECT_EQ(filter.recursion().get_allocator(), Allocator{2});
    predictOverTheStep(filter);
    auto const updated = filter.update(makeFix(fixAt(1)));
    ASSERT_TRUE(updated);
    EXPECT_TRUE(updated->accepted);
}

// A Gaussian mixture filter given a state from another resource by setState() keeps its own, as a
// pmr assignment does. Given a larger state whose allocator propagates on assignment and differs
// from its own, it takes that allocator, and its scratch and the mixture's follow: prediction, an
// update under Keep, which exchanges the state with the scratch, and truncation then allocate
// nothing through the allocator it had.
TEST(Allocation, GaussianMixtureFilterSetStateFollowsTheAllocator)
{
    auto const makePrior = [](auto const & allocator, int count) {
        auto prior = GaussianMixtureParams<State, double, std::remove_cvref_t<decltype(allocator)>>{allocator};
        for (int i = 0; i < count; ++i)
        {
            prior.add(std::log(1.0 / count), Vector4{1.0 * i, -1.0, 0.5, 0.5}, makeInitial().covarianceLower());
        }
        return prior;
    };
    auto resource = CountingResource{arenaBytes};
    auto other = std::pmr::monotonic_buffer_resource{};
    auto kept = GaussianMixtureFilter{makeModel(), makePrior(Pmr{&resource}, 2)};
    kept.setState(makePrior(Pmr{&other}, 4));
    EXPECT_EQ(kept.get_allocator(), Pmr{&resource});
    EXPECT_EQ(kept.state().size(), 4);
    predictOverTheStep(kept);
    ASSERT_TRUE(kept.update(makeFix(fixAt(1)), reductions::keep()));

    using Allocator = Propagating<std::byte>;
    auto filter = GaussianMixtureFilter{makeModel(), makePrior(Allocator{1}, 2)};
    predictOverTheStep(filter);
    ASSERT_TRUE(filter.update(makeFix(fixAt(0)), reductions::keep()));
    ASSERT_TRUE(filter.state().pruneToSize(2));
    filter.setState(makePrior(Allocator{2}, 8));
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    auto const before = propagatingAllocations;
    predictOverTheStep(filter);
    auto const updated = filter.update(makeFix(fixAt(1)), reductions::keep());
    ASSERT_TRUE(updated);
    EXPECT_TRUE(updated->accepted);
    ASSERT_TRUE(filter.state().pruneToSize(4));
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    EXPECT_EQ(propagatingAllocations[1], before[1]);
    EXPECT_GT(propagatingAllocations[2], before[2]);
}

// A mixture of IMMs given a state whose allocator propagates on assignment and differs from its
// own takes it by setState(), and the IMM recursion's scratch follows, so the steps after run with
// the new allocator throughout.
TEST(Allocation, MixtureOfImmsScratchFollowsAPropagatingAllocator)
{
    using Allocator = Propagating<std::byte>;
    using HybridState = HybridGaussianParams<State, Mode, double, Allocator>;
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    auto const makePrior = [](Allocator const & allocator) {
        auto prior = MixtureParams<HybridGaussian<State, Mode, double, Allocator>, Allocator>{allocator};
        prior.add(std::log(0.6), HybridState{CategoricalParams<Mode>{}, makeInitial(), allocator});
        auto const offset =
            GaussianParams<State>{State{}, Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower()};
        prior.add(std::log(0.4), HybridState{CategoricalParams<Mode>{}, offset, allocator});
        return prior;
    };
    auto filter =
        MixtureFilter{switchingModel<Maneuver>(chain, makeModel(0.01), makeModel(4.0)), makePrior(Allocator{1})};
    EXPECT_EQ(filter.recursion().get_allocator(), Allocator{1});
    filter.setState(makePrior(Allocator{2}));
    EXPECT_EQ(filter.get_allocator(), Allocator{2});
    EXPECT_EQ(filter.recursion().get_allocator(), Allocator{2});
    auto const before = propagatingAllocations;
    predictOverTheStep(filter);
    auto const updated = filter.update(makeFix(fixAt(1)));
    ASSERT_TRUE(updated);
    EXPECT_TRUE(updated->accepted);
    EXPECT_EQ(propagatingAllocations[1], before[1]);
}

template <typename Quiet, typename Agile>
void
expectAmmFilterStepsWithinItsResource(Quiet const & quiet, Agile const & agile)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto initial =
        HybridGaussianParams<State, Maneuver, double, Pmr>{CategoricalParams<Maneuver>{}, makeInitial(), allocator};
    auto filter = AmmFilter{std::move(initial), quiet, agile};
    EXPECT_EQ(filter.get_allocator(), allocator);

    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    result.modes.reserve(2);
    auto const floor = upkeep::floorModes(1e-3);
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
            ASSERT_TRUE(floor(filter.state()));
        },
        &resource
    );
    EXPECT_TRUE(result.accepted);
}

// An AMM over a pmr HybridGaussian takes its allocator, and with a reserved result its steps, the
// prediction that skips the mixing and the mode floor included, allocate nothing after warm-up,
// with fixed-step models too.
TEST(Allocation, AmmFilterStepsWithinItsResource)
{
    expectAmmFilterStepsWithinItsResource(makeModel(0.01), makeModel(4.0));
    expectAmmFilterStepsWithinItsResource(
        FixedStepModel{makeModel(0.01), State{}, timeStep},
        FixedStepModel{makeModel(4.0), State{}, timeStep}
    );
}

template <typename Quiet, typename Agile>
void
expectMixtureOfImmsStepsWithinItsResource(Quiet const & quiet, Agile const & agile)
{
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    using HybridState = HybridGaussianParams<State, Mode, double, Pmr>;
    auto prior = MixtureParams<HybridGaussian<State, Mode, double, Pmr>, Pmr>{Capacity{2}, allocator};
    prior.add(std::log(0.6), HybridState{CategoricalParams<Mode>{}, makeInitial(), allocator});
    auto const offset = GaussianParams<State>{State{}, Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower()};
    prior.add(std::log(0.4), HybridState{CategoricalParams<Mode>{}, offset, allocator});
    auto filter = MixtureFilter{switchingModel<Maneuver>(chain, quiet, agile), std::move(prior)};
    EXPECT_EQ(filter.get_allocator(), allocator);

    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    expectStepsAllocateNothing(
        [&](int k) {
            predictOverTheStep(filter);
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
        },
        &resource
    );
    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(filter.state().size(), 2);
    for (auto const & entry : filter.state())
    {
        EXPECT_EQ(entry.component.get_allocator(), allocator);
    }

    // A copy into another resource builds each hybrid there.
    auto other = std::pmr::monotonic_buffer_resource{};
    auto const elsewhere = MixtureParams<HybridGaussian<State, Mode, double, Pmr>, Pmr>{filter.state(), Pmr{&other}};
    ASSERT_EQ(elsewhere.size(), 2);
    for (auto const & entry : elsewhere)
    {
        EXPECT_EQ(entry.component.get_allocator(), Pmr{&other});
    }
}

// A MixtureFilter over a pmr mixture of pmr HybridGaussians, a mixture of IMMs [component-first],
// [filters-and-recursions], takes the mixture's allocator, and its steps ask neither the heap nor the resource for
// memory after warm-up, with fixed-step models too: the IMM recursion's scratch, the posterior's hybrids, and each
// component's result in the reused result keep their storage. A copy of the state into another resource builds
// each hybrid there.
TEST(Allocation, MixtureOfImmsStepsWithinItsResource)
{
    expectMixtureOfImmsStepsWithinItsResource(makeModel(0.01), makeModel(4.0));
    expectMixtureOfImmsStepsWithinItsResource(
        FixedStepModel{makeModel(0.01), State{}, timeStep},
        FixedStepModel{makeModel(4.0), State{}, timeStep}
    );
}

// Records the step in joint with filter's predictJoint() over timeStep, or over its models' fixed step.
template <typename Filter, typename Joint>
void
predictJointOverTheStep(Filter & filter, Joint & joint)
{
    if constexpr (Emrys::isFixedStep<typename Filter::ModelType>)
    {
        filter.predictJoint(joint);
    }
    else
    {
        filter.predictJoint(timeStep, joint);
    }
}

template <typename Quiet, typename Agile>
void
expectImmSmoothingWithinItsResource(Quiet const & quiet, Agile const & agile)
{
    static constexpr Eigen::Index lag = 5;
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto const chain = MarkovChain<Mode>{MarkovChain<Maneuver>::persistent(0.95), MarkovChain<Sensor>::persistent(0.8)};
    auto initial = HybridGaussianParams<State, Mode, double, Pmr>{CategoricalParams<Mode>{}, makeInitial(), allocator};
    auto filter = ImmFilter{switchingModel<Maneuver>(chain, quiet, agile), std::move(initial)};
    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    result.modes.reserve(4);
    using Joint = HybridPredictedJoint<State, Mode, double, Pmr>;
    auto joint = Joint{
        CategoricalParams<Mode>{},
        typename Joint::JointsType(allocator),
        chain.matrix(),
        typename Joint::PredictionsType(allocator)
    };
    joint.joints.reserve(4);
    joint.throughOtherModels.reserve(12);

    auto window = HybridGaussianHistory<State, Mode, double, Pmr>{Lag{lag}, allocator};
    auto const step = [&](int k) {
        predictJointOverTheStep(filter, joint);
        window.push(joint);
        ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
        window.smooth(filter.state());
    };
    for (int k = 0; k < lag; ++k)
    {
        step(k);
    }
    expectStepsAllocateNothing(step, &resource);
    EXPECT_EQ(window.size(), lag);
    for (auto const & recorded : window)
    {
        EXPECT_EQ(recorded.joint.joints.get_allocator(), allocator);
        EXPECT_EQ(recorded.smoothed.get_allocator(), allocator);
    }

    // Unbounded, every step takes the same few allocations from the resource as the first (its
    // node, its joints, and its smoothed Gaussians, and with MSVC's debug iterators a proxy for
    // each container), and nothing from the heap; smoothing takes none.
    auto unbounded = HybridGaussianHistory<State, Mode, double, Pmr>{allocator};
    int heap = 0;
    int first = 0;
    int total = 0;
    {
        auto const watch = HeapWatch{};
        for (int k = 0; k < 10; ++k)
        {
            predictJointOverTheStep(filter, joint);
            int const before = resource.allocations();
            unbounded.push(joint);
            first = k == 0 ? resource.allocations() - before : first;
            total += resource.allocations() - before;
            ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
        }
        int const beforeSmoothing = resource.allocations();
        unbounded.smooth(filter.state());
        EXPECT_EQ(resource.allocations(), beforeSmoothing) << "smoothing allocates nothing";
        heap = watch.allocations();
    }
    EXPECT_EQ(heap, 0) << "allocations from the global heap";
    EXPECT_GE(first, 3) << "a node, its joints, and its Gaussians";
    EXPECT_EQ(total, 10 * first) << "the same allocations at every step";

    expectHistoryIsAllocatorAware(window);
    expectHistoryIsAllocatorAware(unbounded);
}

template <typename Model>
void
expectGaussianMixtureSmoothingWithinItsResource(Model const & model)
{
    static constexpr Eigen::Index lag = 5;
    auto resource = CountingResource{arenaBytes};
    auto const allocator = Pmr{&resource};
    auto prior = GaussianMixtureParams<State, double, Pmr>{Capacity{2}, allocator};
    prior.add(std::log(0.6), makeInitial().moments());
    prior.add(std::log(0.4), Vector4{1.0, -1.0, 0.5, 0.5}, makeInitial().covarianceLower());
    auto filter = GaussianMixtureFilter{model, std::move(prior)};
    using Update = typename decltype(filter.update(makeFix(fixAt(0))))::value_type;
    auto result = Update{allocator};
    result.components.reserve(2);
    using Joint = GaussianMixturePredictedJoint<State, double, Pmr>;
    auto joint = Joint{
        typename Joint::LogWeightsType(allocator),
        typename Joint::JointsType(allocator),
        typename Joint::LineageType(allocator)
    };

    auto window = GaussianMixtureHistory<State, double, Pmr>{Lag{lag}, allocator};
    auto const step = [&](int k) {
        predictJointOverTheStep(filter, joint);
        window.push(joint);
        ASSERT_TRUE(filter.update(makeFix(fixAt(k)), reductions::collapse(), result));
        window.smooth(filter.state());
    };
    for (int k = 0; k < lag; ++k)
    {
        step(k);
    }
    expectStepsAllocateNothing(step, &resource);
    EXPECT_EQ(window.size(), lag);
    for (auto const & recorded : window)
    {
        EXPECT_EQ(recorded.joint.joints.get_allocator(), allocator);
        EXPECT_EQ(recorded.joint.lineage.get_allocator(), allocator);
        EXPECT_EQ(recorded.smoothed.get_allocator(), allocator);
    }
    expectHistoryIsAllocatorAware(window);
}

// A Gaussian mixture filter over a pmr mixture recording each step through one kept joint into a
// fixed-lag GaussianMixtureHistory, and smoothing the window after each update, asks neither the
// heap nor the resource for memory once the window is full, with a fixed-step model too: the
// state's lineage is reserved by its capacity, and the history's joints, lineages, smoothed
// mixtures, and pair scratch come from the resource and are reused.
TEST(Allocation, GaussianMixtureSmoothingStepsWithinItsResource)
{
    expectGaussianMixtureSmoothingWithinItsResource(makeModel());
    expectGaussianMixtureSmoothingWithinItsResource(FixedStepModel{makeModel(), State{}, timeStep});
}

// An IMM recording each step through one kept joint into a fixed-lag HybridGaussianHistory over a
// pmr resource, and smoothing the window after each update, asks neither the heap nor the
// resource for memory once the window is full, with fixed-step models too; the history's joints
// and smoothed states come from the resource by uses-allocator construction.
TEST(Allocation, ImmSmoothingStepsWithinItsResource)
{
    expectImmSmoothingWithinItsResource(makeModel(0.01), makeModel(4.0));
    expectImmSmoothingWithinItsResource(
        FixedStepModel{makeModel(0.01), State{}, timeStep},
        FixedStepModel{makeModel(4.0), State{}, timeStep}
    );
}

// The policy namespaces of decisions.md [policy-factories] (roadmap step 17s): every type and factory in
// Emrys::methods, updates, reductions, resamplers, upkeep, execution, and jacobians meets its kind's concept, which
// tools/check_policies.py checks is asserted here for every member the Doxygen XML lists, since C++ can't
// enumerate a namespace. Also that iterated<float>() holds a float and iterated() a double [embedded-builds],
// that a factory given a run-time value holds it in the scalar it was written in, an integer as double, and
// that a factory's result is the type beside it. Headers under test: Models/Methods.h, Models/Condition.h,
// Models/Reduction.h, Core/Resampling.h, Distributions/Upkeep.h, Core/Execution.h, Models/Batched.h.

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GridGaussian.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/Upkeep.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/Reduction.h>
#include <EmrysFilter/Models/Transform.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/TagPath.h>

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <concepts>
#include <functional>
#include <random>
#include <utility>

namespace execution = Emrys::execution;
namespace jacobians = Emrys::jacobians;
namespace methods = Emrys::methods;
namespace reductions = Emrys::reductions;
namespace resamplers = Emrys::resamplers;
namespace updates = Emrys::updates;
namespace upkeep = Emrys::upkeep;

using Emrys::CovarianceUpdate;
using Emrys::ExecutionPolicy;
using Emrys::JacobianMark;
using Emrys::MixtureReductionPolicy;
using Emrys::ReductionPolicy;
using Emrys::Resampler;
using Emrys::TransformMethod;
using Emrys::UpkeepOperation;

namespace
{

struct Position : Emrys::Linear<2>
{};

using State = Emrys::Linear<2>;
using Posterior = Emrys::GaussianParams<State>;
using Component = Emrys::Gaussian<State>;
using Update = Emrys::MeasurementUpdate<State, Emrys::Linear<1>, 1>;
using Jacobian = Eigen::Matrix<double, 1, 2> (*)(Eigen::Vector2d const &);
using Generator = std::mt19937_64;
using Particles = Emrys::ParticleSet<Emrys::Linear<1>>;

// A state no upkeep operation applies to, but None.
struct AnyState
{};

struct Velocity : Emrys::Linear<2>
{};

// The states the actions apply to beside Particles: a regular grid subdivides, a hybrid floors its modes,
// and a Gaussian repairs its covariance.
using Grid = Emrys::GridGaussianParams<
    Emrys::PointGrid<Emrys::Composite<Position, Velocity>, Position, Emrys::GridCells<Position>>>;
using Hybrid = Emrys::HybridGaussianParams<State, Emrys::Finite<2>>;

// A caller's predicate for When.
struct SizeAbove
{
    [[nodiscard]] bool
    operator()(Particles const & particles) const
    {
        return particles.size() > 16;
    }
};

// methods: the types, then the factories, each a TransformMethod.
static_assert(TransformMethod<methods::Linearized>);
static_assert(TransformMethod<methods::Unscented<>>);
static_assert(TransformMethod<methods::FourthMomentUnscented>);
static_assert(TransformMethod<methods::Cubature>);
static_assert(TransformMethod<methods::SphericalSimplex<>>);
static_assert(TransformMethod<methods::GaussHermite<>>);
static_assert(TransformMethod<methods::Iterated<>>);
static_assert(TransformMethod<methods::Spread<methods::Cubature, Emrys::roots::Principal>>);
static_assert(TransformMethod<decltype(methods::linearized())>);
static_assert(TransformMethod<decltype(methods::unscented<0.5>())>);
static_assert(TransformMethod<decltype(methods::fourthMomentUnscented())>);
static_assert(TransformMethod<decltype(methods::cubature())>);
static_assert(TransformMethod<decltype(methods::sphericalSimplex<0.5>())>);
static_assert(TransformMethod<decltype(methods::gaussHermite<3>())>);
static_assert(TransformMethod<decltype(methods::iterated(methods::cubature()))>);
static_assert(TransformMethod<decltype(methods::spread(methods::cubature(), Emrys::roots::principal()))>);

// updates: each a CovarianceUpdate.
static_assert(CovarianceUpdate<updates::Standard>);
static_assert(CovarianceUpdate<updates::Symmetric>);
static_assert(CovarianceUpdate<updates::Joseph>);
static_assert(CovarianceUpdate<decltype(updates::standard())>);
static_assert(CovarianceUpdate<decltype(updates::symmetric())>);
static_assert(CovarianceUpdate<decltype(updates::joseph())>);

// reductions: a ReductionPolicy to a Gaussian, or a MixtureReductionPolicy for Keep and a gate over it.
static_assert(ReductionPolicy<reductions::Collapse, Update, Posterior>);
static_assert(ReductionPolicy<reductions::Select, Update, Posterior>);
static_assert(MixtureReductionPolicy<reductions::Keep, Update, Component>);
static_assert(ReductionPolicy<reductions::Gate<>, Update, Posterior>);
static_assert(ReductionPolicy<decltype(reductions::collapse()), Update, Posterior>);
static_assert(ReductionPolicy<decltype(reductions::select()), Update, Posterior>);
static_assert(MixtureReductionPolicy<decltype(reductions::keep()), Update, Component>);
static_assert(ReductionPolicy<decltype(reductions::gate(0.997)), Update, Posterior>);
static_assert(
    ReductionPolicy<decltype(reductions::gate(Emrys::NisThreshold{9.21}, reductions::select())), Update, Posterior>
);
static_assert(MixtureReductionPolicy<decltype(reductions::gate(0.997, reductions::keep())), Update, Component>);

// resamplers: each a Resampler.
static_assert(Resampler<resamplers::Multinomial, double, Generator>);
static_assert(Resampler<resamplers::Stratified, double, Generator>);
static_assert(Resampler<resamplers::Systematic, double, Generator>);
static_assert(Resampler<resamplers::Residual<>, double, Generator>);
static_assert(Resampler<decltype(resamplers::multinomial()), double, Generator>);
static_assert(Resampler<decltype(resamplers::stratified()), double, Generator>);
static_assert(Resampler<decltype(resamplers::systematic()), double, Generator>);
static_assert(Resampler<decltype(resamplers::residual(resamplers::stratified())), double, Generator>);

// upkeep: each an UpkeepOperation of the states it acts on, called with the state alone; Resample and Roughen
// hold their generator.
static_assert(UpkeepOperation<upkeep::Resample<Generator>, Particles>);
static_assert(UpkeepOperation<upkeep::Roughen<Generator>, Particles>);
static_assert(UpkeepOperation<upkeep::PruneBelow<>, Particles>);
static_assert(UpkeepOperation<upkeep::PruneTail<>, Particles>);
static_assert(UpkeepOperation<upkeep::PruneToSize, Particles>);
static_assert(UpkeepOperation<upkeep::Subdivide, Grid>);
static_assert(UpkeepOperation<upkeep::FloorModes<>, Hybrid>);
static_assert(UpkeepOperation<upkeep::RepairCovariance<>, Posterior>);
static_assert(UpkeepOperation<upkeep::When<SizeAbove>, Particles>);
static_assert(UpkeepOperation<upkeep::WhenEss<std::less<>, double>, Particles>);
static_assert(UpkeepOperation<upkeep::WhenSize<std::greater<>>, Particles>);
static_assert(UpkeepOperation<decltype(upkeep::resample(std::declval<Generator &>())), Particles>);
static_assert(
    UpkeepOperation<decltype(upkeep::resample(std::declval<Generator &>(), resamplers::stratified())), Particles>
);
static_assert(
    UpkeepOperation<decltype(upkeep::resample(execution::sequential, std::declval<Generator &>())), Particles>
);
static_assert(UpkeepOperation<decltype(upkeep::roughen(std::declval<Generator &>(), 0.25)), Particles>);
static_assert(
    UpkeepOperation<decltype(upkeep::roughen(execution::sequential, std::declval<Generator &>(), 0.25)), Particles>
);
static_assert(UpkeepOperation<decltype(upkeep::pruneBelow(1e-6)), Particles>);
static_assert(UpkeepOperation<decltype(upkeep::pruneTail(1e-3)), Particles>);
static_assert(UpkeepOperation<decltype(upkeep::pruneToSize(16)), Particles>);
static_assert(UpkeepOperation<decltype(upkeep::subdivide()), Grid>);
static_assert(UpkeepOperation<decltype(upkeep::floorModes(1e-3)), Hybrid>);
static_assert(UpkeepOperation<decltype(upkeep::repairCovariance()), Posterior>);
static_assert(UpkeepOperation<decltype(upkeep::repairCovariance(Emrys::roots::ridge<1.0, 1e-12>())), Posterior>);
static_assert(UpkeepOperation<
              decltype(upkeep::whenEss(
                  std::less{},
                  0.5,
                  upkeep::resample(std::declval<Generator &>()),
                  upkeep::roughen(std::declval<Generator &>())
              )),
              Particles>);
static_assert(
    UpkeepOperation<
        decltype(upkeep::
                     whenEss(execution::sequential, std::less{}, 0.5, upkeep::resample(std::declval<Generator &>()))),
        Particles>
);
static_assert(UpkeepOperation<decltype(upkeep::when(SizeAbove{}, upkeep::pruneToSize(16))), Particles>);
static_assert(UpkeepOperation<decltype(upkeep::whenSize(std::greater{}, 32, upkeep::pruneToSize(16))), Particles>);
static_assert(UpkeepOperation<decltype(upkeep::whenSize(std::less{}, 64, upkeep::subdivide(2))), Grid>);

static_assert(UpkeepOperation<upkeep::Sequence<upkeep::PruneBelow<>>, Particles>);
static_assert(UpkeepOperation<decltype(upkeep::sequence(upkeep::pruneBelow(1e-6))), Particles>);
static_assert(UpkeepOperation<upkeep::None, AnyState>);
static_assert(UpkeepOperation<decltype(upkeep::none()), AnyState>);

// Each operation is constrained on what it calls, so it answers truthfully whether it applies to a state
// (decisions.md [upkeep]): AnyState has none of the members, and a trigger holding Subdivide doesn't apply to
// Particles, which can't subdivide.
static_assert(!UpkeepOperation<upkeep::Resample<Generator>, AnyState>);
static_assert(!UpkeepOperation<upkeep::Roughen<Generator>, AnyState>);
static_assert(!UpkeepOperation<upkeep::PruneBelow<>, AnyState>);
static_assert(!UpkeepOperation<upkeep::PruneTail<>, AnyState>);
static_assert(!UpkeepOperation<upkeep::PruneToSize, AnyState>);
static_assert(!UpkeepOperation<upkeep::Subdivide, AnyState>);
static_assert(!UpkeepOperation<upkeep::FloorModes<>, AnyState>);
static_assert(!UpkeepOperation<upkeep::RepairCovariance<>, AnyState>);
static_assert(!UpkeepOperation<upkeep::WhenSize<std::greater<>>, AnyState>);
static_assert(!UpkeepOperation<upkeep::WhenEss<std::less<>, double>, AnyState>);
static_assert(!UpkeepOperation<decltype(upkeep::whenSize(std::less{}, 64, upkeep::subdivide(2))), Particles>);

// execution: the constant and its type.
static_assert(ExecutionPolicy<execution::Sequential>);
static_assert(ExecutionPolicy<decltype(execution::sequential)>);

// The marks of a batched function's Jacobians [batched-functions].
static_assert(JacobianMark<jacobians::PerColumn>);
static_assert(JacobianMark<jacobians::Shared>);
static_assert(JacobianMark<decltype(jacobians::perColumn())>);
static_assert(JacobianMark<decltype(jacobians::shared())>);

// A factory's result is the type beside it, with its defaults.
static_assert(std::same_as<decltype(methods::linearized()), methods::Linearized>);
static_assert(std::same_as<decltype(methods::unscented()), methods::Unscented<>>);
static_assert(std::same_as<decltype(methods::unscented<0.5>()), methods::Unscented<0.5>>);
static_assert(std::same_as<decltype(methods::fourthMomentUnscented()), methods::FourthMomentUnscented>);
static_assert(std::same_as<decltype(methods::iterated()), methods::Iterated<>>);
static_assert(std::same_as<decltype(methods::iterated(methods::cubature())), methods::Iterated<methods::Cubature>>);
static_assert(std::same_as<decltype(reductions::gate(0.997)), reductions::Gate<>>);
static_assert(std::same_as<decltype(resamplers::residual()), resamplers::Residual<resamplers::Multinomial>>);
static_assert(std::same_as<
              decltype(upkeep::resample(std::declval<Generator &>())),
              upkeep::Resample<Generator, resamplers::Systematic, execution::Sequential>>);
static_assert(std::same_as<decltype(upkeep::whenSize(std::greater{}, 32)), upkeep::WhenSize<std::greater<>>>);

// iterated() holds its tolerance in double unless the scalar is named, since a braced argument deduces
// nothing, and a float filter names it [embedded-builds].
static_assert(std::same_as<decltype(methods::iterated().tolerance), double>);
static_assert(std::same_as<decltype(methods::iterated<float>().tolerance), float>);
static_assert(std::same_as<
              decltype(methods::iterated<float>(methods::cubature(), {.tolerance = 1e-6f})),
              methods::Iterated<methods::Cubature, float>>);

// A factory given a run-time value holds it in the scalar it is written in, an integer as double.
static_assert(std::same_as<decltype(reductions::gate(0.997f)), reductions::Gate<reductions::Collapse, float>>);
static_assert(std::same_as<
              decltype(reductions::gate(Emrys::NisThreshold{9.21}, reductions::select())),
              reductions::Gate<reductions::Select, double>>);
static_assert(std::same_as<decltype(upkeep::roughen(std::declval<Generator &>(), 0)), upkeep::Roughen<Generator>>);
static_assert(
    std::same_as<decltype(upkeep::roughen(std::declval<Generator &>(), 0.25f)), upkeep::Roughen<Generator, float>>
);
static_assert(std::same_as<
              decltype(upkeep::whenEss(std::less{}, 0.5f, upkeep::resample(std::declval<Generator &>()))),
              upkeep::WhenEss<std::less<>, float, upkeep::Resample<Generator>>>);
static_assert(std::same_as<decltype(upkeep::pruneBelow(1e-4f)), upkeep::PruneBelow<float>>);
static_assert(std::same_as<decltype(upkeep::pruneTail(1e-3)), upkeep::PruneTail<double>>);
static_assert(std::same_as<decltype(upkeep::floorModes(1e-3)), upkeep::FloorModes<double>>);

// repairCovariance() holds the ridge of decisions.md [covariance-repair] unless given another repair, and refuses
// axes, which are not a Repair, by its constraint.
template <typename R>
concept RepairsCovarianceBy = requires(R repair) { upkeep::repairCovariance(repair); };
static_assert(std::same_as<decltype(upkeep::repairCovariance()), upkeep::RepairCovariance<Emrys::roots::Ridge<>>>);
static_assert(RepairsCovarianceBy<Emrys::roots::Ridge<1.0, 1e-12>>);
static_assert(!RepairsCovarianceBy<Emrys::roots::Cholesky>);

// iterated() keeps the options it was given, and without them a tolerance of 1e-3 and 10 linearizations.
TEST(Policies, IteratedHoldsItsOptions)
{
    auto const defaults = methods::iterated();
    EXPECT_EQ(defaults.tolerance, 1e-3);
    EXPECT_EQ(defaults.maxIterations, 10u);
    auto const given = methods::iterated(methods::linearized(), {.tolerance = 1e-6, .maxIterations = 20});
    EXPECT_EQ(given.tolerance, 1e-6);
    EXPECT_EQ(given.maxIterations, 20u);
    auto const inFloat = methods::iterated<float>(methods::cubature(), {.tolerance = 1e-4f});
    EXPECT_EQ(inFloat.tolerance, 1e-4f);
    EXPECT_EQ(inFloat.maxIterations, 10u);
}

// The factories that take a value hold it: a gate's threshold, a roughening scale and generator, an action's
// probability or factor, and a trigger's limit.
TEST(Policies, FactoriesHoldTheirValues)
{
    auto rng = Generator{1};
    EXPECT_EQ(reductions::gate(Emrys::NisThreshold{9.21}).threshold(2), 9.21);
    EXPECT_EQ(upkeep::roughen(rng, 0.25).scale(), 0.25);
    EXPECT_EQ(&upkeep::roughen(rng).generator(), &rng);
    EXPECT_EQ(&upkeep::resample(rng).generator(), &rng);
    EXPECT_EQ(upkeep::pruneBelow(1e-6).probability, 1e-6);
    EXPECT_EQ(upkeep::pruneTail(1e-3).fraction, 1e-3);
    EXPECT_EQ(upkeep::pruneToSize(16).maxSize, 16);
    EXPECT_EQ(upkeep::subdivide(3).factor, 3);
    EXPECT_EQ(upkeep::floorModes(1e-3).probability, 1e-3);
    EXPECT_EQ(upkeep::whenEss(std::less{}, 0.5, upkeep::resample(rng)).predicate.fraction, 0.5);
    EXPECT_EQ(upkeep::whenSize(std::greater{}, 32).predicate.limit, 32);
}

}  // namespace

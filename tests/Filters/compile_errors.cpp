// The compile errors that the filters claim (Filters/*.h, and the Rao-Blackwellized update's checks
// in Models/Measurement.h): each case below is a misuse that must fail to compile with the
// static_assert or constraint it names, checked by tests/compile_errors.cmake. Without a case the
// file compiles, which checks what the cases share. The evidence for [coverage]'s negative compile
// tests, for the policies and methods of [filters-and-recursions] and [gaussian-forms], and for
// the filters that take a likelihood and those that refuse one [likelihood-measurements], and for the
// refusals of [consider-states].

#include <EmrysFilter/EmrysFilter.h>

#include <Eigen/Core>

#include <optional>
#include <random>
#include <type_traits>

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

struct Cruise
{};

struct Turn
{};

struct Good
{};

struct Bad
{};

struct Maneuver : Emrys::Named<Cruise, Turn>
{};

struct Sensor : Emrys::Named<Good, Bad>
{};

using State = Emrys::Composite<Position, Velocity>;

[[maybe_unused]] std::mt19937 rng{1};

[[maybe_unused]] auto
makeModel()
{
    return Emrys::ConstantVelocity<Position, Velocity>{0.1};
}

using Model = decltype(makeModel());
using Fixed = decltype(Emrys::FixedStepModel{makeModel(), State{}, 1.0});

[[maybe_unused]] Emrys::GaussianParams<State>
makeInitial()
{
    return {State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
}

[[maybe_unused]] Emrys::GaussianParams<Emrys::Linear<2>>
makeNoise()
{
    return {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
}

[[maybe_unused]] Emrys::GaussianMixtureParams<Emrys::Linear<2>>
makeMixtureNoise()
{
    auto noise = Emrys::GaussianMixtureParams<Emrys::Linear<2>>{Emrys::Linear<2>{}};
    noise.add(0.0, makeNoise());
    return noise;
}

auto const readPosition = []<typename T>(auto const & in, auto const &, auto const & x) {
    return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x)};
};

[[maybe_unused]] auto
makeFix(auto method, auto noise)
{
    return Emrys::observe(
        method,
        Emrys::measurementModel(
            readPosition,
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(std::move(noise))
        ),
        Eigen::Vector2d::Zero()
    );
}

[[maybe_unused]] auto
makeImm()
{
    return Emrys::ImmFilter{
        Emrys::SwitchingModel{Emrys::MarkovChain<Maneuver>::persistent(0.9), makeModel(), makeModel()},
        Emrys::HybridParams{
            Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.5, 0.5}),
            makeInitial()
        }
    };
}

[[maybe_unused]] auto
makeRbParticle()
{
    return Emrys::RbParticleFilter{makeModel(), Emrys::drawParticleGaussian<Position>(makeInitial(), 10, rng)};
}

// A measurement known only by its likelihood, over the block named by Path.
template <typename Path>
[[maybe_unused]] auto
makeLikelihood()
{
    return Emrys::observe(
        Emrys::measurementModel(
            Emrys::reads(Emrys::path<Path>),
            Emrys::logLikelihood([](auto const & in, auto const & x, auto const &) {
                return -Emrys::block<Path>(in, x).squaredNorm();
            })
        ),
        Eigen::Vector2d::Zero()
    );
}

[[maybe_unused]] auto
makeRbParticlePerPoint()
{
    return Emrys::RbParticleFilter<State, Position, Model, Emrys::PerPointCovariance>{
        makeModel(),
        Emrys::drawParticleGaussian<Position, Emrys::PerPointCovariance>(makeInitial(), 10, rng)
    };
}

[[maybe_unused]] auto
makeRbPointMassPerPoint()
{
    auto const grid = Emrys::regularGrid<Position>(State{}, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {3, 4});
    using Rest = std::remove_cvref_t<decltype(grid)>::RestSpace;
    return Emrys::RbPointMassFilter{
        Emrys::RandomWalk<Velocity>{0.1},
        grid,
        Emrys::GaussianParams<Rest>{Rest{}, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()},
        Emrys::PerPointCovariance{}
    };
}

[[maybe_unused]] void
everyCaseSharesThis()
{
    auto kalman = Emrys::KalmanFilter{makeModel(), makeInitial()};
    [[maybe_unused]] auto const kalmanUpdate = kalman.update(makeFix(Emrys::methods::linearized(), makeNoise()));
    auto imm = makeImm();
    [[maybe_unused]] auto const immUpdate = imm.update(makeFix(Emrys::methods::linearized(), makeNoise()));
    auto rb = makeRbParticle();
    [[maybe_unused]] auto const rbUpdate = rb.update(makeFix(Emrys::methods::linearized(), makeNoise()));
    [[maybe_unused]] auto const mixture = makeMixtureNoise();
    auto rbPerPoint = makeRbParticlePerPoint();
    [[maybe_unused]] auto const rbLikelihood = rbPerPoint.update(makeLikelihood<Position>());
    auto grid = makeRbPointMassPerPoint();
    [[maybe_unused]] auto const gridLikelihood = grid.update(makeLikelihood<Position>());
}

// A filter with a fixed-step model predicts by the model's method, so another is an error.

#if defined(KALMAN_FIXED_STEP_METHOD)  // static_assert: KalmanFilter.h: a filter with a fixed-step model predicts by
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::KalmanFilter<Emrys::Gaussian<State>, Fixed, Emrys::methods::Unscented<>>);
#endif

#if defined(IMM_FIXED_STEP_METHOD)  // static_assert: ImmFilter.h: a filter with fixed-step models predicts each mode
// Each fixed-step mode predicts by its own method, so the filter's is ModelsMethod, and any other
// method named in its type is an error, even the one every mode's model carries.
[[maybe_unused]] constexpr auto size = sizeof(Emrys::ImmRecursion<
                                              Emrys::Gaussian<State>,
                                              Emrys::SwitchingModel<Maneuver, double, Maneuver, Fixed, Fixed>,
                                              Emrys::methods::Linearized>);
#endif

#if defined(MIXTURE_FIXED_STEP_METHOD)  // static_assert: MixtureFilter.h: a filter with a fixed-step model predicts by
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::MixtureFilter<Emrys::Gaussian<State>, Fixed, Emrys::methods::Unscented<>>);
#endif

#if defined(RB_PARTICLE_FIXED_STEP_METHOD)  // static_assert: RbParticleFilter.h: a filter with a fixed-step model
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::RbParticleFilter<State, Position, Fixed, Emrys::SharedCovariance, Emrys::methods::Unscented<>>);
#endif

#if defined(RB_POINT_MASS_FIXED_STEP_METHOD)  // static_assert: RbPointMassFilter.h: a filter with a fixed-step model
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::RbPointMassFilter<
           decltype(Emrys::regularGrid<Position>(State{}, Eigen::Vector2d{}, Eigen::Vector2d{}, {3, 4})),
           Fixed,
           Emrys::SharedCovariance,
           Emrys::methods::Unscented<>>);
#endif

// Iterated is a measurement update's method, so a filter or a fixed-step model that predicts by it is
// an error at its type, before any prediction.

#if defined(KALMAN_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to relinearize
[[maybe_unused]] void
kalmanIteratedPrediction()
{
    [[maybe_unused]] auto filter = Emrys::KalmanFilter{makeModel(), makeInitial(), Emrys::methods::iterated()};
}
#endif

#if defined(IMM_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to relinearize
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::ImmRecursion<Emrys::Gaussian<State>, Model, Emrys::methods::Iterated<>>);
#endif

#if defined(MIXTURE_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::MixtureFilter<Emrys::Gaussian<State>, Model, Emrys::methods::Iterated<>>);
#endif

#if defined(RB_PARTICLE_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::RbParticleFilter<State, Position, Model, Emrys::SharedCovariance, Emrys::methods::Iterated<>>);
#endif

#if defined(RB_POINT_MASS_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::RbPointMassFilter<
           decltype(Emrys::regularGrid<Position>(State{}, Eigen::Vector2d{}, Eigen::Vector2d{}, {3, 4})),
           Model,
           Emrys::SharedCovariance,
           Emrys::methods::Iterated<>>);
#endif

#if defined(FIXED_STEP_ITERATED_PREDICTION)  // static_assert: ProcessModel.h: a prediction has no posterior to
[[maybe_unused]] constexpr auto size = sizeof(Emrys::FixedStepModel<State, Model, Emrys::methods::Iterated<>>);
#endif

#if defined(POINT_MASS_PREDICT_INPUT)  // static_assert: RbPointMassFilter.h: a point-mass filter over the whole state
[[maybe_unused]] void
pointMassPredictInput()
{
    using Whole = Emrys::Composite<Position>;
    auto filter = Emrys::PointMassFilter{
        Emrys::regularGrid<Position>(Whole{}, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {3, 4})
    };
    filter.predict(1.0, Emrys::sure<Velocity>(Eigen::Vector2d::Zero()));
}
#endif

#if defined(HIDDEN_MARKOV_OTHER_CHAIN)  // static_assert: HiddenMarkovFilter.h: the chain must be on the filter's
[[maybe_unused]] constexpr auto size = sizeof(Emrys::HiddenMarkovFilter<Maneuver, Emrys::MarkovChain<Sensor>>);
#endif

// Policies that do not reduce the branches as the filter keeps them.

#if defined(KALMAN_KEEPS_BRANCHES)  // constraint: KalmanFilter.h: ReductionPolicy
[[maybe_unused]] void
kalmanKeeps()
{
    auto filter = Emrys::KalmanFilter{makeModel(), makeInitial()};
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()), Emrys::reductions::keep());
}
#endif

#if defined(MIXTURE_POLICY_RETURNS_OTHER)  // constraint: MixtureFilter.h: MixtureReductionPolicy
[[maybe_unused]] void
mixturePolicyReturnsOther()
{
    auto mixture = Emrys::GaussianMixtureParams<State>{State{}};
    mixture.add(0.0, makeInitial());
    auto filter = Emrys::GaussianMixtureFilter{makeModel(), std::move(mixture)};
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()), [](auto const & branches) {
            return branches[0].posterior.mean();
        });
}
#endif

#if defined(SQUARE_ROOT_USER_POLICY)  // static_assert: KalmanFilter.h: a square-root or information form has one branch
[[maybe_unused]] void
squareRootUserPolicy()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeModel(),
        Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    auto const rejectAll = [](auto const &) {
        return std::optional<Emrys::GaussianParams<State>>{};
    };
    [[maybe_unused]] auto const updated = filter.update(makeFix(Emrys::methods::linearized(), makeNoise()), rejectAll);
}
#endif

#if defined(IMM_KEEPS_BRANCHES)  // static_assert: ImmFilter.h: an IMM filter keeps one component per mode, so its
[[maybe_unused]] void
immKeeps()
{
    auto filter = makeImm();
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()), Emrys::reductions::keep());
}
#endif

#if defined(RB_KEEPS_BRANCHES)  // static_assert: RbUpdate.h: a Rao-Blackwellized filter keeps one Gaussian per point
[[maybe_unused]] void
rbKeeps()
{
    auto filter = Emrys::RbParticleFilter<State, Position, Model, Emrys::PerPointCovariance>{
        makeModel(),
        Emrys::drawParticleGaussian<Position, Emrys::PerPointCovariance>(makeInitial(), 10, rng)
    };
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()), Emrys::reductions::keep());
}
#endif

#if defined(MIXTURE_OF_IMMS_IN_PARALLEL)  // static_assert: MixtureFilter.h: a recursion with scratch, such as the
// A policy of the user's own, since libc++ has no std::execution::par.
struct InOrder
{
    template <typename F>
    void
    forEachIndex(Eigen::Index count, F const & f) const
    {
        for (Eigen::Index i = 0; i < count; ++i)
        {
            f(i);
        }
    }
};

[[maybe_unused]] void
mixtureOfImmsInParallel()
{
    auto bank = Emrys::MixtureFilter{
        Emrys::SwitchingModel{Emrys::MarkovChain<Maneuver>::persistent(0.9), makeModel(), makeModel()},
        Emrys::MixtureParams<Emrys::Hybrid<Emrys::Gaussian<State>, Maneuver>>{}
    };
    bank.predict(InOrder{}, 1.0);
}
#endif

#if defined(RB_POINT_MASS_HYBRID_IN_PARALLEL)  // static_assert: RbPointMassFilter.h: a cell's IMM step writes the
// A policy of the user's own, since libc++ has no std::execution::par.
struct InOrder
{
    template <typename F>
    void
    forEachIndex(Eigen::Index count, F const & f) const
    {
        for (Eigen::Index i = 0; i < count; ++i)
        {
            f(i);
        }
    }
};

[[maybe_unused]] void
rbPointMassHybridInParallel()
{
    auto const grid = Emrys::regularGrid<Position>(State{}, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones(), {3, 4});
    using Rest = typename decltype(grid)::RestSpace;
    auto filter = Emrys::RbPointMassFilter{
        Emrys::SwitchingModel{
            Emrys::MarkovChain<Maneuver>::persistent(0.9),
            Emrys::RandomWalk<Velocity>{0.1},
            Emrys::RandomWalk<Velocity>{1.0}
        },
        grid,
        Emrys::HybridGaussianParams<Rest, Maneuver>{
            Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.5, 0.5}),
            Emrys::GaussianParams<Rest>{Rest{}, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}
        }
    };
    filter.predict(InOrder{}, 1.0);
}
#endif

#if defined(RB_PARTICLE_HYBRID_IN_PARALLEL)  // static_assert: RbParticleFilter.h: a particle's IMM step writes the
// A policy of the user's own, since libc++ has no std::execution::par.
struct InOrder
{
    template <typename F>
    void
    forEachIndex(Eigen::Index count, F const & f) const
    {
        for (Eigen::Index i = 0; i < count; ++i)
        {
            f(i);
        }
    }
};

[[maybe_unused]] void
rbParticleHybridInParallel()
{
    using Rest = typename Emrys::ParticleGaussianParams<State, Position>::RestSpace;
    auto filter = Emrys::RbParticleFilter{
        Emrys::SwitchingModel{
            Emrys::MarkovChain<Maneuver>::persistent(0.9),
            Emrys::ConstantVelocity<Position, Velocity>{0.1},
            Emrys::ConstantVelocity<Position, Velocity>{1.0}
        },
        Emrys::particleGaussian<Position>(
            State{},
            Emrys::ParticleSet<Position>{Eigen::Matrix<double, 2, 3>::Zero()},
            Emrys::HybridGaussianParams<Rest, Maneuver>{
                Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.5, 0.5}),
                Emrys::GaussianParams<Rest>{Rest{}, Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}
            }
        )
    };
    auto rng = std::mt19937_64{1};
    [[maybe_unused]] auto const moved = filter.predict(InOrder{}, rng, 1.0);
}
#endif

#if defined(REUSED_UPDATE_OTHER_INNOVATION)  // static_assert: MeasurementReads.h: the update reused must be over the
                                             // state and
[[maybe_unused]] void
reusedUpdateOtherInnovation()
{
    auto filter = Emrys::KalmanFilter{makeModel(), makeInitial()};
    auto result = Emrys::ReusableFilterUpdate<State, Emrys::Linear<3>>{};
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()), Emrys::reductions::collapse(), result);
}
#endif

// The Rao-Blackwellized update, given the pinned block.

#if defined(RB_SHARED_MIXTURE_NOISE)  // static_assert: RbUpdate.h: a shared covariance needs additive Gaussian noise
[[maybe_unused]] void
rbSharedMixtureNoise()
{
    auto filter = makeRbParticle();
    [[maybe_unused]] auto const updated = filter.update(makeFix(Emrys::methods::linearized(), makeMixtureNoise()));
}
#endif

#if defined(RB_UPDATE_WITHOUT_METHOD)  // static_assert: RbUpdate.h: a Kalman update pushes the prior through the model
[[maybe_unused]] void
rbWithoutMethod()
{
    auto filter = makeRbParticle();
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::observe(
            Emrys::measurementModel(
                readPosition,
                Emrys::reads(Emrys::path<Position>),
                Emrys::additiveNoise(makeNoise())
            ),
            Eigen::Vector2d::Zero()
        )
    );
}
#endif

#if defined(RB_ITERATED_UPDATE)  // static_assert: RbUpdate.h: an update given a block does not support Iterated yet
[[maybe_unused]] void
rbIterated()
{
    auto filter = makeRbParticle();
    [[maybe_unused]] auto const updated = filter.update(makeFix(Emrys::methods::iterated(), makeNoise()));
}
#endif

#if defined(IMM_NOISE_AXIS_MISSING)  // static_assert: ImmFilter.h: the measurement's noise is by a mode axis
[[maybe_unused]] void
immNoiseAxisMissing()
{
    auto filter = makeImm();
    auto const fix = Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            readPosition,
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(makeMixtureNoise()).byMode<Sensor>()
        ),
        Eigen::Vector2d::Zero()
    );
    [[maybe_unused]] auto const updated = filter.update(fix);
}
#endif

#if defined(IMM_SWITCHING_RESULT_TYPES)  // static_assert: ImmFilter.h: the measurements of a switching measurement
[[maybe_unused]] void
immSwitchingResultTypes()
{
    auto filter = makeImm();
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::switchingMeasurement<Maneuver>(
            makeFix(Emrys::methods::linearized(), makeNoise()),
            makeFix(Emrys::methods::linearized(), makeMixtureNoise())
        )
    );
}
#endif

#if defined(IMM_SWITCHING_AXIS_MISSING)  // static_assert: ImmFilter.h: a switching measurement is selected by a mode
[[maybe_unused]] void
immSwitchingAxisMissing()
{
    // The filter's mode space is Maneuver alone.
    auto filter = makeImm();
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::switchingMeasurement<Sensor>(
            makeFix(Emrys::methods::linearized(), makeNoise()),
            makeFix(Emrys::methods::linearized(), makeNoise())
        )
    );
}
#endif

#if defined(IMM_SWITCHING_BY_MODE)  // static_assert: ImmFilter.h: a measurement of a switching measurement has noise
[[maybe_unused]] void
immSwitchingByMode()
{
    auto filter = makeImm();
    auto const byMode = [] {
        return Emrys::observe(
            Emrys::methods::linearized(),
            Emrys::measurementModel(
                readPosition,
                Emrys::reads(Emrys::path<Position>),
                Emrys::additiveNoise(makeMixtureNoise()).byMode<Maneuver>()
            ),
            Eigen::Vector2d::Zero()
        );
    };
    [[maybe_unused]] auto const updated = filter.update(Emrys::switchingMeasurement<Maneuver>(byMode(), byMode()));
}
#endif

#if defined(RB_PER_POINT_ITERATED_UPDATE)  // static_assert: Measurement.h: a block does not support Iterated yet
[[maybe_unused]] void
rbPerPointIterated()
{
    auto filter = Emrys::RbParticleFilter<State, Position, Model, Emrys::PerPointCovariance>{
        makeModel(),
        Emrys::drawParticleGaussian<Position, Emrys::PerPointCovariance>(makeInitial(), 10, rng)
    };
    [[maybe_unused]] auto const updated = filter.update(makeFix(Emrys::methods::iterated(), makeNoise()));
}
#endif

// A measurement known only by its likelihood, which a Kalman update can't take, nor a policy, and
// which a Rao-Blackwellized filter takes only over its sampled or gridded block.

#if defined(KALMAN_TAKES_NO_LIKELIHOOD)  // static_assert: Measurement.h: a Kalman update needs the model's function
[[maybe_unused]] void
kalmanTakesNoLikelihood()
{
    auto filter = Emrys::KalmanFilter{makeModel(), makeInitial()};
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::observe(
            Emrys::methods::linearized(),
            Emrys::measurementModel(
                Emrys::reads(Emrys::path<Position>),
                Emrys::logLikelihood([](auto const & in, auto const & x, auto const &) {
                    return -Emrys::block<Position>(in, x).squaredNorm();
                })
            ),
            Eigen::Vector2d::Zero()
        )
    );
}
#endif

#if defined(RB_LIKELIHOOD_READS_THE_REST)  // static_assert: RbUpdate.h: logLikelihood() may read only the gridded or
[[maybe_unused]] void
rbLikelihoodReadsTheRest()
{
    auto filter = makeRbParticle();
    [[maybe_unused]] auto const updated = filter.update(makeLikelihood<Velocity>());
}
#endif

#if defined(RB_PARTICLE_LIKELIHOOD_POLICY)  // static_assert: RbParticleFilter.h: has no branches to reduce, so it
[[maybe_unused]] void
rbParticleLikelihoodPolicy()
{
    auto filter = makeRbParticlePerPoint();
    [[maybe_unused]] auto const updated = filter.update(makeLikelihood<Position>(), Emrys::reductions::collapse());
}
#endif

#if defined(RB_POINT_MASS_LIKELIHOOD_POLICY)  // static_assert: RbPointMassFilter.h: has no branches to reduce, so it
[[maybe_unused]] void
rbPointMassLikelihoodPolicy()
{
    auto filter = makeRbPointMassPerPoint();
    [[maybe_unused]] auto const updated = filter.update(makeLikelihood<Position>(), Emrys::reductions::collapse());
}
#endif

// A fixed-step filter takes no step, and a square-root IMM takes no gate.

#if defined(FIXED_STEP_GIVEN_A_STEP)  // static_assert: Time.h: a fixed-step filter takes its own step
[[maybe_unused]] void
fixedStepGivenAStep()
{
    auto filter = Emrys::KalmanFilter{Emrys::FixedStepModel{makeModel(), State{}, 1.0}, makeInitial()};
    filter.predict(1.0);
}
#endif

#if defined(SQRT_IMM_GATES)  // static_assert: ImmFilter.h: a square-root form has one branch and takes the Collapse
[[maybe_unused]] void
sqrtImmGates()
{
    auto filter = Emrys::ImmFilter{
        Emrys::SwitchingModel{Emrys::MarkovChain<Maneuver>::persistent(0.9), makeModel(), makeModel()},
        Emrys::HybridParams{
            Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.5, 0.5}),
            Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
        }
    };
    [[maybe_unused]] auto const updated =
        filter.update(makeFix(Emrys::methods::linearized(), makeNoise()), Emrys::reductions::gate(0.99));
}
#endif

#if defined(OBSERVE_OF_TEMPORARY_TABLE)  // deleted: Emission.h: observe(Eigen::Index observed) const &&
[[maybe_unused]] auto const observation =
    Emrys::EmissionTable<Maneuver, Sensor>{Maneuver{}, Sensor{}, Eigen::Matrix2d::Identity()}.observe(0);
#endif

// The consider update's refusals [consider-states].

#if defined(RB_SHARED_CONSIDER)  // static_assert: RbUpdate.h: an update given a block, as a Rao-Blackwellized
[[maybe_unused]] void
rbSharedConsider()
{
    auto filter = makeRbParticle();
    auto const velocityOf = []<typename T>(auto const & in, auto const &, auto const & x) {
        return Eigen::Matrix<T, 2, 1>{Emrys::block<Velocity>(in, x)};
    };
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::observe(
            Emrys::methods::linearized(),
            Emrys::measurementModel(velocityOf, Emrys::reads(Emrys::path<Velocity>), Emrys::additiveNoise(makeNoise())),
            Eigen::Vector2d::Zero(),
            Emrys::consider(Emrys::path<Velocity>)
        )
    );
}
#endif

#if defined(RB_LIKELIHOOD_CONSIDER)  // static_assert: RbUpdate.h: consider() is a Kalman update's
[[maybe_unused]] void
rbLikelihoodConsider()
{
    auto filter = makeRbParticle();
    [[maybe_unused]] auto const updated = filter.update(
        Emrys::observe(
            Emrys::measurementModel(
                Emrys::reads(Emrys::path<Position>),
                Emrys::logLikelihood([](auto const & in, auto const & x, auto const &) {
                    return -Emrys::block<Position>(in, x).squaredNorm();
                })
            ),
            Eigen::Vector2d::Zero(),
            Emrys::consider(Emrys::path<Velocity>)
        )
    );
}
#endif

}  // namespace

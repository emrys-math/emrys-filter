// The compile errors that the models claim (Models/Transform.h, MeasurementModel.h, MeasurementReads.h, Measurement.h,
// Likelihood.h, Simulation.h, ProcessModel.h, SwitchingModel.h, Consistency.h, CramerRao.h, FormSteps.h, Policies.h):
// each case below is a misuse that must fail to compile with the static_assert it names, checked by
// tests/compile_errors.cmake. Without a case the file compiles, which checks what the cases share. The evidence for
// [coverage]'s negative compile tests, and for the method parameters of [sigma-point-sets] and [iterated-update] and
// the refusals of [consider-states].

#include <EmrysFilter/EmrysFilter.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <Eigen/Core>

#include <concepts>
#include <random>
#include <tuple>
#include <type_traits>

namespace methods = Emrys::methods;

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<>
{};

struct Speed : Emrys::Linear<1>
{};

struct Attitude : Emrys::UnitQuaternion
{};

struct Spin : Emrys::Linear<4>
{};

// The noise of a position fix that passes through the function, an input of the call.
struct Offset : Emrys::Linear<2>
{};

// A measurement's block, written by a model's writes(), alone and repeated.
struct Fix : Emrys::Linear<2>
{};

struct Fixes : Emrys::Repeated<Fix, 4>
{};

struct Cruise
{};

struct Turn
{};

struct Maneuver : Emrys::Named<Cruise, Turn>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Emrys::Named<Inlier, Outlier>
{};

using State = Emrys::Composite<Position, Velocity>;
using Swapped = Emrys::Composite<Velocity, Position>;
using Pose = Emrys::Composite<Position, Heading>;

[[maybe_unused]] auto
makeModel()
{
    return Emrys::ConstantVelocity<Position, Velocity>{0.1};
}

[[maybe_unused]] auto
makeFixed()
{
    return Emrys::FixedStepModel{makeModel(), State{}, 1.0};
}

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

[[maybe_unused]] Emrys::GaussianParams<Offset>
makeOffset()
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

// A function of transform()'s: the position, and the position plus an offset, read by tag.
auto const readPosition = []<typename T>(auto const & x) {
    return Eigen::Matrix<T, 2, 1>{x};
};

auto const sum = []<typename T>(auto const & in, auto const & x) {
    return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x) + Emrys::block<Offset>(in, x)};
};

[[maybe_unused]] Emrys::GaussianParams<Position>
makePosition()
{
    return {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
}

// A whole-state model of each kind: the identity step, and the position of a Composite<Position, Velocity>.
[[maybe_unused]] auto
makeWholeStep()
{
    return Emrys::processModel([](auto const & x, auto) {
        return x.eval();
    });
}

[[maybe_unused]] auto
makeWholeFix()
{
    return Emrys::measurementModel(
        [](auto const & x) {
            return x.template head<2>().eval();
        },
        Emrys::additiveNoise(makeNoise())
    );
}

// A measurement model's function: the position of its input, and the position plus the offset it reads from the call.
auto const positionOf = []<typename T>(auto const & in, auto const &, auto const & x) {
    return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x)};
};

auto const offsetPosition = []<typename T>(auto const & in, auto const &, auto const & x) {
    return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x) + Emrys::block<Offset>(in, x)};
};

// A likelihood of the position alone, g(in, x, z).
auto const laplace = [](auto const & in, auto const & x, auto const & z) {
    return -(z - Emrys::block<Position>(in, x)).norm();
};

// A draw of the position itself, g(in, x, rng).
[[maybe_unused]] auto const drawPosition = [](auto const & in, auto const & x, auto &) {
    return Emrys::block<Position>(in, x).eval();
};

[[maybe_unused]] auto
makeFixModel()
{
    return Emrys::measurementModel(positionOf, Emrys::reads(Emrys::path<Position>), Emrys::additiveNoise(makeNoise()));
}

// The fix with its noise read from the call as an input.
[[maybe_unused]] auto
makeOffsetModel()
{
    return Emrys::measurementModel(offsetPosition, Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>));
}

[[maybe_unused]] auto
makeLikelihoodModel()
{
    return Emrys::measurementModel(Emrys::reads(Emrys::path<Position>), Emrys::logLikelihood(laplace));
}

[[maybe_unused]] auto
makeFix(auto method)
{
    return Emrys::observe(method, makeFixModel(), Eigen::Vector2d::Zero());
}

[[maybe_unused]] void
everyCaseSharesThis()
{
    [[maybe_unused]] auto const predicted = Emrys::predict(makeInitial(), makeFixed());
    [[maybe_unused]] auto const updated = Emrys::update(makeInitial(), makeFix(Emrys::methods::linearized()));
    [[maybe_unused]] auto const moved = Emrys::transform(
        Emrys::methods::unscented(),
        sum,
        Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
        makePosition(),
        makeOffset()
    );
    [[maybe_unused]] auto const read = Emrys::transform(Emrys::methods::unscented(), readPosition, makeNoise());
    [[maybe_unused]] auto const wholeStep =
        Emrys::predict(Emrys::methods::linearized(), makeInitial(), makeWholeStep(), 1.0);
    [[maybe_unused]] auto const wholeFix = Emrys::update(
        makeInitial(),
        Emrys::observe(Emrys::methods::linearized(), makeWholeFix(), Eigen::Vector2d::Zero())
    );
    [[maybe_unused]] auto const monitor = Emrys::NisMonitor<3>{};
    [[maybe_unused]] auto const chain = Emrys::Kinematic<double, Position, Velocity>{0.1};
    [[maybe_unused]] auto const mixture = makeMixtureNoise();
    [[maybe_unused]] auto const offset = Emrys::update(
        makeInitial(),
        Emrys::observe(Emrys::methods::linearized(), makeOffsetModel(), Eigen::Vector2d::Zero(), makeOffset())
    );
    [[maybe_unused]] auto const weights = Emrys::logLikelihoods(
        Emrys::observe(makeLikelihoodModel(), Eigen::Vector2d::Zero()),
        State{},
        Eigen::Matrix4d::Zero()
    );
}

// Methods and their parameters.

#if defined(UNSCENTED_ALPHA)  // static_assert: Methods.h: the unscented transform's alpha must be positive
[[maybe_unused]] constexpr auto size = sizeof(Emrys::methods::Unscented<0.0>);
#endif

#if defined(SPHERICAL_SIMPLEX_WEIGHT)  // static_assert: Methods.h: a spherical simplex's center weight must be in
[[maybe_unused]] constexpr auto size = sizeof(Emrys::methods::SphericalSimplex<1.0>);
#endif

#if defined(GAUSS_HERMITE_ORDER)  // static_assert: Methods.h: a Gauss-Hermite rule has at least one point per
[[maybe_unused]] constexpr auto size = sizeof(Emrys::methods::GaussHermite<0>);
#endif

#if defined(ITERATED_OTHER_METHOD)  // static_assert: Methods.h: Iterated relinearizes Linearized or a sigma-point
                                    // method
[[maybe_unused]] constexpr auto size = sizeof(Emrys::methods::Iterated<int>);
#endif

#if defined(AFFINE_NOT_DIFFERENTIABLE)  // static_assert: Transform.h: a function that a sigma-point method factors
[[maybe_unused]] void
affineNotDifferentiable()
{
    // The function takes plain doubles and returns no Jacobian, so its Jacobian in the affine
    // coordinates, the velocity, can be found neither way. A model reaches the same refusal
    // only where its own function fails on dual numbers first.
    auto const prior = Emrys::GaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
    auto const drift = [](std::same_as<Eigen::Vector4d> auto const & x) {
        return Eigen::Vector2d{x.template head<2>() + x.template tail<2>()};
    };
    auto const method = Emrys::internal::FactoredAt<2, 2, Emrys::methods::Unscented<>>{
        Emrys::methods::unscented(),
        Eigen::Matrix<Eigen::Index, 2, 1>{2, 3}
    };
    [[maybe_unused]] auto const moved = Emrys::transform(method, drift, prior);
}
#endif

// transform() and its inputs.

#if defined(TRANSFORM_OF_CERTAIN_INPUTS)  // static_assert: TransformArgs.h: transform() needs at least one uncertain
                                          // input
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    Emrys::sure<Position>(Eigen::Vector2d{Eigen::Vector2d::Zero()}),
    Emrys::sure<Offset>(Eigen::Vector2d{Eigen::Vector2d::Zero()})
);
#endif

#if defined(TRANSFORM_OF_TWO_SCALARS)  // static_assert: TransformArgs.h: every uncertain input must have the same
                                       // scalar
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    makePosition(),
    Emrys::GaussianParams<Offset, float>{Eigen::Vector2f::Zero(), Eigen::Matrix2f::Identity()}
);
#endif

#if defined(TRANSFORM_SPACE_AS_INPUT)  // static_assert: Transform.h: an input of transform() is an uncertain input
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented(), readPosition, Position{}, makeNoise());
#endif

#if defined(TRANSFORM_TWO_INPUTS_ONE_TAG)  // static_assert: Transform.h: two inputs of transform() have one tag
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    makePosition(),
    makePosition(),
    makeOffset()
);
#endif

#if defined(TRANSFORM_SEVERAL_INPUTS_WITHOUT_READS)  // static_assert: Transform.h: transform() of a function without
                                                     // reads() takes one uncertain input
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    [](auto const & x) {
        return x.eval();
    },
    makeNoise(),
    Emrys::GaussianParams<Emrys::Linear<1>>{
        Eigen::Matrix<double, 1, 1>::Zero(),
        Eigen::Matrix<double, 1, 1>::Identity()
    }
);
#endif

#if defined(TRANSFORM_READS_A_PATH)  // static_assert: Transform.h: transform() matches each item of reads() to an input
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    readPosition,
    Emrys::reads(Emrys::path<State, Position>),
    makePosition()
);
#endif

#if defined(TRANSFORM_INPUT_MISSING)  // static_assert: Transform.h: the function reads an input that transform() was
                                      // not
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    makePosition()
);
#endif

#if defined(TRANSFORM_INPUT_NOT_READ)  // static_assert: Transform.h: transform() was given an input the function does
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    makePosition(),
    makeOffset(),
    makeNoise()
);
#endif

#if defined(TRANSFORM_TUPLE_JACOBIAN)  // static_assert: Transform.h: a transformed function returns its Jacobian as one
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::linearized(),
    [](auto const &, Eigen::Vector4d const & x) {
        return Emrys::withJacobian(
            Eigen::Vector2d{x.head<2>() + x.tail<2>()},
            std::tuple{Eigen::Matrix2d{Eigen::Matrix2d::Identity()}, Eigen::Matrix2d{Eigen::Matrix2d::Identity()}}
        );
    },
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    makePosition(),
    makeOffset()
);
#endif

#if defined(TRANSFORM_MIXTURE_NOISE)  // static_assert: Transform.h: transform() adds one Gaussian noise
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented(), readPosition, Emrys::additiveNoise(makeMixtureNoise()), makeNoise());
#endif

#if defined(TRANSFORM_LIKELIHOOD_MODEL)  // static_assert: Transform.h: a model known by its likelihood alone has
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented(), makeLikelihoodModel(), makePosition());
#endif

#if defined(TRANSFORM_POLICY_AFTER_INPUT)  // static_assert: Transform.h: transform() takes its policies before its
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented(), readPosition, makeNoise(), Emrys::outputSpace(Position{}));
#endif

#if defined(TRANSFORM_PROCESS_MODEL)  // static_assert: Transform.h: transform() pushes its inputs through a measurement
[[maybe_unused]] auto const moved = Emrys::transform(Emrys::methods::unscented(), makeModel(), makeInitial());
#endif

#if defined(TRANSFORM_MODEL_WITH_POLICIES)  // static_assert: Transform.h: transform() of a model takes the model's own
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented(), makeFixModel(), Emrys::outputSpace(Position{}), makePosition());
#endif

#if defined(TRANSFORM_TO_MATRIX)  // static_assert: Transform.h: a transformed function must return a column vector
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::unscented(),
    []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 2, 2>{x * x.transpose()};
    },
    makeNoise()
);
#endif

#if defined(TRANSFORM_LINEARIZED_PLAIN_ARGUMENT)  // static_assert: Transform.h: methods::linearized() differentiates f
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::linearized(),
    [](Eigen::Vector2d const & x) {
        return Eigen::Vector2d{2.0 * x};
    },
    makeNoise()
);
#endif

#if defined(TRANSFORM_LINEARIZED_PLAIN_RESULT)  // static_assert: Transform.h: methods::linearized() differentiates f
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::linearized(),
    [](auto const & x) {
        Eigen::Vector2d const plain = x;
        return Eigen::Vector2d{2.0 * plain};
    },
    makeNoise()
);
#endif

#if defined(TRANSFORM_ITERATED)  // static_assert: Transform.h: Iterated is a measurement update's method
[[maybe_unused]] auto const moved = Emrys::transform(Emrys::methods::iterated(), readPosition, makeNoise());
#endif

#if defined(TRANSFORM_SPREAD)  // static_assert: Transform.h: the sigma-point spread must be positive
[[maybe_unused]] auto const moved =
    Emrys::transform(Emrys::methods::unscented<1.0, 2.0, -3.0>(), readPosition, makeNoise());
#endif

#if defined(TRANSFORM_GAUSS_HERMITE_POINTS)  // static_assert: Transform.h: a point set over this many coordinates
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::gaussHermite<3>(),
    []<typename T>(auto const & x) {
        return Eigen::Matrix<T, 9, 1>{x};
    },
    Emrys::GaussianParams<Emrys::Linear<9>>{Eigen::Vector<double, 9>::Zero(), Eigen::Matrix<double, 9, 9>::Identity()}
);
#endif

// Process models.

#if defined(KINEMATIC_ONE_BLOCK)  // static_assert: ProcessModel.h: a kinematic chain needs at least two blocks
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Kinematic<double, Position>);
#endif

#if defined(RANDOM_WALK_RESIZABLE)  // static_assert: ProcessModel.h: the block of a random walk must have a fixed size
struct Landmarks : Emrys::Linear<Eigen::Dynamic, 8>
{};

[[maybe_unused]] constexpr auto size = sizeof(Emrys::RandomWalk<Landmarks>);
#endif

#if defined(KINEMATIC_SIZES)  // static_assert: ProcessModel.h: a kinematic chain must have the same fixed size
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Kinematic<double, Position, Speed>);
#endif

#if defined(KINEMATIC_QUATERNION)  // static_assert: ProcessModel.h: differences of the same size as their values
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Kinematic<double, Attitude, Spin>);
#endif

// A process model's function of the position alone, the identity.
[[maybe_unused]] auto const holdPosition = []<typename T>(auto const & in, auto const &, auto const & x, auto) {
    return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x)};
};

#if defined(PROCESS_MODEL_WITHOUT_READS)  // static_assert: ProcessModel.h: a process model that writes blocks with
[[maybe_unused]] auto const model = Emrys::processModel(holdPosition, Emrys::writes(Emrys::path<Position>));
#endif

#if defined(PROCESS_MODEL_WITHOUT_WRITES)  // static_assert: ProcessModel.h: a process model that reads blocks with
[[maybe_unused]] auto const model = Emrys::processModel(holdPosition, Emrys::reads(Emrys::path<Position>));
#endif

#if defined(PROCESS_MODEL_AFFINE_WITHOUT_READS)  // static_assert: ProcessModel.h: affineIn() names items of reads()
[[maybe_unused]] auto const model = Emrys::processModel(
    [](auto const & x, auto) {
        return x.eval();
    },
    Emrys::affineIn(Emrys::path<Position>)
);
#endif

#if defined(MATRIX_PROCESS_MODEL_WITH_READS)  // static_assert: ProcessModel.h: a matrix process model is over the whole
[[maybe_unused]] auto const model = Emrys::processModel(
    Eigen::Matrix2d{Eigen::Matrix2d::Identity()},
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Position>)
);
#endif

#if defined(MATRIX_PROCESS_MODEL_NOT_SQUARE)  // static_assert: ProcessModel.h: a matrix process model's matrix is
                                              // square
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    makeInitial(),
    Emrys::processModel(Eigen::Matrix<double, 2, 4>{Eigen::Matrix<double, 2, 4>::Zero()}),
    1.0
);
#endif

#if defined(MATRIX_PROCESS_MODEL_OF_A_QUATERNION)  // static_assert: ProcessModel.h: a matrix process model moves a
                                                   // state
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    Emrys::GaussianParams<Attitude>{Eigen::Vector4d{1.0, 0.0, 0.0, 0.0}, Eigen::Matrix3d::Identity()},
    Emrys::processModel(Eigen::Matrix4d{Eigen::Matrix4d::Identity()}),
    1.0
);
#endif

#if defined(ON_WHOLE_STATE_MODEL)  // static_assert: ResolvedReads.h: a model without reads() and writes() reads and
[[maybe_unused]] auto const placed = Emrys::on<Position>(makeWholeStep());
#endif

#if defined(ON_EACH_WHOLE_STATE_MODEL)  // static_assert: ResolvedReads.h: a model without reads() and writes() reads
[[maybe_unused]] auto const placed = Emrys::onEach<Position>(makeWholeStep());
#endif

#if defined(COMPOSE_WHOLE_STATE_MODELS)  // static_assert: ResolvedReads.h: a model without reads() and writes() reads
[[maybe_unused]] auto const composed = Emrys::compose(makeWholeFix(), makeFixModel());
#endif

#if defined(FUNCTION_IN_NO_FORM)  // static_assert: Methods.h: a model's function takes h(x), h(in, x), or h(in, out, x)
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            [](auto const &, auto const &, auto const &, auto const &) {
                return Eigen::Vector2d{Eigen::Vector2d::Zero()};
            },
            Emrys::additiveNoise(makeNoise())
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(PROCESS_MODEL_LOG_LIKELIHOOD)  // static_assert: ProcessModel.h: logLikelihood(g) is a measurement model's
[[maybe_unused]] auto const model = Emrys::processModel(
    holdPosition,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Position>),
    Emrys::logLikelihood(laplace)
);
#endif

#if defined(PROCESS_MODEL_SAMPLE)  // static_assert: ProcessModel.h: sample(g) is a measurement model's
[[maybe_unused]] auto const model = Emrys::processModel(
    holdPosition,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Position>),
    Emrys::sample(drawPosition)
);
#endif

#if defined(MEASUREMENT_FUNCTION_SAMPLE)  // static_assert: MeasurementModel.h: so sample(g) is for a model
[[maybe_unused]] auto const model =
    Emrys::measurementModel(positionOf, Emrys::reads(Emrys::path<Position>), Emrys::sample(drawPosition));
#endif

#if defined(PROCESS_MODEL_OUTPUT_SPACE)  // static_assert: ProcessModel.h: a process model's output is the blocks of the
[[maybe_unused]] auto const model = Emrys::processModel(
    holdPosition,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Position>),
    Emrys::outputSpace(Position{})
);
#endif

#if defined(PROCESS_MODEL_BOTH_NOISES)  // static_assert: ProcessModel.h: a process model's noise is continuousNoise()
[[maybe_unused]] auto const model = Emrys::processModel(
    holdPosition,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Position>),
    Emrys::continuousNoise(0.1),
    Emrys::additiveNoise(makeNoise())
);
#endif

#if defined(PROCESS_MODEL_MIXTURE_NOISE)  // static_assert: ProcessModel.h: a process model's additive noise is one
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    Emrys::GaussianParams<Emrys::Composite<Position>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()},
    Emrys::processModel(
        holdPosition,
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Position>),
        Emrys::additiveNoise(makeMixtureNoise())
    ),
    1.0
);
#endif

#if defined(PROCESS_JACOBIAN_AS_BLOCKS)  // static_assert: ProcessModel.h: a process model's function returns its
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    Emrys::GaussianParams<Emrys::Composite<Position>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()},
    Emrys::processModel(
        [](auto const & in, auto const &, Eigen::Vector2d const & x, double) {
            return Emrys::withJacobian(
                Eigen::Vector2d{Emrys::block<Position>(in, x)},
                std::tuple{Eigen::Matrix2d{Eigen::Matrix2d::Identity()}}
            );
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Position>)
    ),
    1.0
);
#endif

#if defined(CONTINUOUS_NOISE_ON_EACH)  // static_assert: ProcessModel.h: continuousNoise() takes no onEach<Path>(q)
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    Emrys::GaussianParams<Emrys::Composite<Position>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()},
    Emrys::processModel(
        holdPosition,
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Position>),
        Emrys::continuousNoise(Emrys::onEach<Position>(0.1))
    ),
    1.0
);
#endif

#if defined(COMPOSE_NOTHING)  // static_assert: ProcessModel.h: a composition needs a process model
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Composition<>);
#endif

#if defined(COMPOSE_WRITES_TWICE)  // static_assert: ProcessModel.h: two parts of a composition write the same block
[[maybe_unused]] auto const model = Emrys::compose(Emrys::RandomWalk<Position>{0.1}, Emrys::Constant<Position>{});
#endif

#if defined(COMPOSE_TWO_SCALARS)  // static_assert: ProcessModel.h: the parts of a composition must have one scalar
[[maybe_unused]] auto const model =
    Emrys::compose(Emrys::RandomWalk<Position>{0.1}, Emrys::RandomWalk<Velocity, float>{0.1f});
#endif

#if defined(COMPOSE_NESTED_TWO_SCALARS)  // static_assert: ProcessModel.h: the parts of a composition must have one
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::compose(Emrys::Constant<Position>{}, Emrys::RandomWalk<Velocity, float>{0.1f}),
    Emrys::RandomWalk<Speed>{0.1}
);
#endif

#if defined(COMPOSITION_POLICY_MISSING)  // static_assert: ProcessModel.h: the composition was not given this policy
[[maybe_unused]] auto const & noise = Emrys::compose(Emrys::Constant<Position>{}, Emrys::Constant<Velocity>{})
                                          .policy<Emrys::internal::PolicyKind::continuousNoise>();
#endif

#if defined(FIXED_STEP_NOT_A_MODEL)  // static_assert: ProcessModel.h: a fixed-step model is a process model taken
[[maybe_unused]] constexpr auto size = sizeof(Emrys::FixedStepModel<State, int>);
#endif

#if defined(FIXED_STEP_BOUND_TO_DYNAMIC_SPACE)  // static_assert: ProcessModel.h: binds to a fixed-size state space
[[maybe_unused]] auto const model =
    Emrys::processModel(Eigen::MatrixXd::Identity(4, 4), Emrys::additiveNoise(Eigen::MatrixXd::Identity(4, 4)));
[[maybe_unused]] Emrys::FixedStepModel<Emrys::Linear<Eigen::Dynamic>, std::remove_cvref_t<decltype(model)>> const
    bound = Emrys::fixedStep(model, 1.0);
#endif

#if defined(FIXED_STEP_OTHER_SPACE)  // static_assert: ProcessModel.h: the model must be over the prior's space
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::GaussianParams<Swapped>{Swapped{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()},
    makeFixed()
);
#endif

#if defined(FIXED_STEP_OTHER_SCALAR)  // static_assert: ProcessModel.h: the model's scalar must be the prior's
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::GaussianParams<State, float>{State{}, Eigen::Vector4f::Zero(), Eigen::Matrix4f::Identity()},
    makeFixed()
);
#endif

#if defined(SQUARE_ROOT_FIXED_STEP_OTHER_SPACE)  // static_assert: FormSteps.h: the model must be over the state's space
[[maybe_unused]] void
squareRootOtherSpace()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeFixed(),
        Emrys::SqrtGaussianParams<Swapped>{Swapped{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    [[maybe_unused]] auto const predicted = filter.predict();
}
#endif

#if defined(SQUARE_ROOT_FIXED_STEP_OTHER_SCALAR)  // static_assert: FormSteps.h: the model's scalar must be the state's
[[maybe_unused]] void
squareRootOtherScalar()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeFixed(),
        Emrys::SqrtGaussianParams<State, float>{State{}, Eigen::Vector4f::Zero(), Eigen::Matrix4f::Identity()}
    };
    [[maybe_unused]] auto const predicted = filter.predict();
}
#endif

#if defined(SWITCHING_WITHOUT_MODELS)  // static_assert: SwitchingModel.h: a switching model needs a process model
[[maybe_unused]] constexpr auto size = sizeof(Emrys::SwitchingModel<Maneuver, double, Maneuver>);
#endif

#if defined(SWITCHING_FIXED_AND_NOT)  // static_assert: SwitchingModel.h: a switching model's process models are all
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::SwitchingModel<Maneuver, double, Maneuver, decltype(makeFixed()), decltype(makeModel())>);
#endif

#if defined(SWITCHING_MODEL_COUNT)  // static_assert: SwitchingModel.h: give one process model per value of the axis
[[maybe_unused]] auto const switching =
    Emrys::SwitchingModel{Emrys::MarkovChain<Maneuver>::persistent(0.9), makeModel(), makeModel(), makeModel()};
#endif

#if defined(SWITCHING_AXIS_ONE_MODEL)  // static_assert: SwitchingModel.h: or no path for one model shared by every mode
// One model where the path names a two-value axis is a forgotten model, not one shared by every mode.
[[maybe_unused]] auto const switching = Emrys::switchingModel<Maneuver>(
    Emrys::MarkovChain<Emrys::DiscreteProduct<Maneuver, Sensor>>::persistent(0.9),
    makeModel()
);
#endif

#if defined(WITH_METHOD_FIXED_STEP)  // static_assert: SwitchingModel.h: a fixed-step model carries its own method
// A fixed-step model takes its method from fixedStep() or its constructor, not from withMethod().
[[maybe_unused]] auto const model = Emrys::withMethod(makeFixed(), methods::unscented());
#endif

#if defined(SWITCHING_WITHOUT_MEASUREMENTS)  // static_assert: SwitchingModel.h: a switching measurement needs a
[[maybe_unused]] constexpr auto size = sizeof(Emrys::SwitchingMeasurement<Maneuver>);
#endif

#if defined(SWITCHING_MEASUREMENT_COUNT)  // static_assert: SwitchingModel.h: give one measurement per value of the
[[maybe_unused]] auto const switching = Emrys::switchingMeasurement<Maneuver>(
    Emrys::observe(makeWholeFix(), Eigen::Vector2d::Zero()),
    Emrys::observe(makeWholeFix(), Eigen::Vector2d::Zero()),
    Emrys::observe(makeWholeFix(), Eigen::Vector2d::Zero())
);
#endif

// Consistency monitors.

#if defined(NIS_WINDOW_EMPTY)  // static_assert: Consistency.h: a window holds at least one statistic
[[maybe_unused]] auto const monitor = Emrys::NisMonitor<0>{};
#endif

#if defined(NIS_WINDOW_BOUND)  // static_assert: Consistency.h: a window's bound is positive, and a fixed window is its
[[maybe_unused]] auto const monitor = Emrys::NisMonitor<5, 6>{};
#endif

#if defined(NIS_CUMULATIVE_BOUND)  // static_assert: Consistency.h: a cumulative window has no bound
[[maybe_unused]] auto const monitor = Emrys::NisMonitor<Emrys::Cumulative, 5>{};
#endif

#if defined(NEES_CUMULATIVE_BOUND)  // static_assert: Consistency.h: a cumulative window has no bound
[[maybe_unused]] auto const monitor = Emrys::NeesMonitor<Emrys::Cumulative, 5>{};
#endif

// Measurement models and their observations.

#if defined(OBSERVATION_OF_NO_MODEL)  // static_assert: MeasurementModel.h: an observation is of a model
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Observed<Emrys::methods::Linearized, int, Eigen::Vector2d>);
#endif

#if defined(MODEL_AFFINE_WITHOUT_READS)  // static_assert: MeasurementModel.h: affineIn() names items of reads()
[[maybe_unused]] auto const model =
    Emrys::measurementModel(readPosition, Emrys::affineIn(Emrys::path<Position>), Emrys::additiveNoise(makeNoise()));
#endif

#if defined(MODEL_CONTINUOUS_NOISE)  // static_assert: MeasurementModel.h: continuousNoise() is a process model's
[[maybe_unused]] auto const model =
    Emrys::measurementModel(positionOf, Emrys::reads(Emrys::path<Position>), Emrys::continuousNoise(0.1));
#endif

#if defined(NOISE_COVARIANCE_SIZE)  // static_assert: Policies.h: additiveNoise(R) is over the output's differences
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d{Eigen::Matrix2d::Identity()},
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(Eigen::Matrix3d{Eigen::Matrix3d::Identity()})
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(NOISE_COVARIANCE_ON_EACH_UNREPEATED)  // static_assert: Policies.h: onEach<Path>(R) in additiveNoise()
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix2d{Eigen::Matrix2d::Identity()},
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(Emrys::onEach<>(Eigen::Matrix2d{Eigen::Matrix2d::Identity()}))
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(LIKELIHOOD_MODEL_WITHOUT_G)  // static_assert: MeasurementModel.h: a measurement model without a function is
[[maybe_unused]] auto const model = Emrys::measurementModel(Emrys::reads(Emrys::path<Position>));
#endif

#if defined(LIKELIHOOD_MODEL_WITH_NOISE)  // static_assert: MeasurementModel.h: a measurement model known by its
[[maybe_unused]] auto const model = Emrys::measurementModel(
    Emrys::reads(Emrys::path<Position>),
    Emrys::logLikelihood(laplace),
    Emrys::additiveNoise(makeNoise())
);
#endif

#if defined(ARGUMENT_NOT_AN_INPUT)  // static_assert: MeasurementModel.h: an argument after the observation is an
[[maybe_unused]] auto const fix =
    Emrys::observe(Emrys::methods::linearized(), makeFixModel(), Eigen::Vector2d::Zero(), 3);
#endif

#if defined(INPUT_UNTAGGED)  // static_assert: MeasurementModel.h: an uncertain input is named by the tag of its space
[[maybe_unused]] auto const fix =
    Emrys::observe(Emrys::methods::linearized(), makeFixModel(), Eigen::Vector2d::Zero(), makeNoise());
#endif

#if defined(FORM_TWICE)  // static_assert: MeasurementModel.h: give the covariance update form once
[[maybe_unused]] auto const fix = Emrys::observe(
    Emrys::methods::linearized(),
    makeFixModel(),
    Eigen::Vector2d::Zero(),
    Emrys::updates::joseph(),
    Emrys::updates::symmetric()
);
#endif

#if defined(REPAIR_TWICE)  // static_assert: MeasurementModel.h: give the repair of the innovation covariance once
[[maybe_unused]] auto const fix = Emrys::observe(
    Emrys::methods::linearized(),
    makeFixModel(),
    Eigen::Vector2d::Zero(),
    Emrys::roots::ridge(),
    Emrys::roots::ridge()
);
#endif

#if defined(INPUT_TAG_TWICE)  // static_assert: MeasurementModel.h: two inputs of the call have one tag
[[maybe_unused]] auto const fix = Emrys::observe(
    Emrys::methods::linearized(),
    makeOffsetModel(),
    Eigen::Vector2d::Zero(),
    makeOffset(),
    Emrys::sure<Offset>(Eigen::Vector2d::Zero())
);
#endif

#if defined(KALMAN_UPDATE_WITHOUT_METHOD)  // static_assert: Measurement.h: a Kalman update pushes the prior through the
[[maybe_unused]] auto const updated =
    Emrys::update(makeInitial(), Emrys::observe(makeFixModel(), Eigen::Vector2d::Zero()));
#endif

#if defined(KALMAN_UPDATE_OF_LIKELIHOOD_MODEL)  // static_assert: Measurement.h: a Kalman update needs the model's
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(Emrys::methods::linearized(), makeLikelihoodModel(), Eigen::Vector2d::Zero())
);
#endif

#if defined(SQUARE_ROOT_UPDATE_WITHOUT_METHOD)  // static_assert: FormSteps.h: a Kalman update pushes the prior through
[[maybe_unused]] void
squareRootWithoutMethod()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeModel(),
        Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    [[maybe_unused]] auto const updated = filter.update(Emrys::observe(makeFixModel(), Eigen::Vector2d::Zero()));
}
#endif

#if defined(READS_NO_STATE_BLOCK)  // static_assert: ResolvedReads.h: a model reads at least one block of the state
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            []<typename T>(auto const & in, auto const &, auto const & x) {
                return Eigen::Matrix<T, 2, 1>{Emrys::block<Offset>(in, x)};
            },
            Emrys::reads(Emrys::path<Offset>)
        ),
        Eigen::Vector2d::Zero(),
        makeOffset()
    )
);
#endif

#if defined(READS_PATH_NOT_IN_STATE)  // static_assert: ResolvedReads.h: an item of reads() that names no block of
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(positionOf, Emrys::reads(Emrys::path<Position>, Emrys::path<Maneuver, Heading>)),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(READS_INPUT_NOT_GIVEN)  // static_assert: ResolvedReads.h: the model reads an input that the state lacks
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(Emrys::methods::linearized(), makeOffsetModel(), Eigen::Vector2d::Zero())
);
#endif

#if defined(INPUT_NOT_READ)  // static_assert: ResolvedReads.h: the call was given an input the model does not read
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(Emrys::methods::linearized(), makeFixModel(), Eigen::Vector2d::Zero(), makeOffset())
);
#endif

#if defined(FUNCTION_RETURNS_MATRIX)  // static_assert: MeasurementReads.h: a measurement model's function returns a
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::unscented(),
        Emrys::measurementModel(
            []<typename T>(auto const & in, auto const &, auto const & x) {
                auto const p = Emrys::block<Position>(in, x);
                return Eigen::Matrix<T, 2, 2>{p * p.transpose()};
            },
            Emrys::reads(Emrys::path<Position>)
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

// A function that returns a scalar as an expression of automatic-differentiation scalars, which refers
// to the function's own temporaries, is refused: it returns T{...}, its input's scalar, or a vector.
#if defined(FUNCTION_RETURNS_EXPRESSION)  // static_assert: ResolvedReads.h: a function that returns a scalar returns
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            [](auto const & in, auto const &, auto const & x) {
                return Emrys::block<Position>(in, x).norm() + Emrys::block<Position>(in, x)(0);
            },
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(
                Emrys::GaussianParams<Emrys::Linear<1>>{
                    Eigen::Matrix<double, 1, 1>::Zero(),
                    Eigen::Matrix<double, 1, 1>::Identity()
                }
            )
        ),
        Eigen::Matrix<double, 1, 1>::Zero()
    )
);
#endif

#if defined(FUNCTION_RETURNS_DYNAMIC_ROWS)  // static_assert: MeasurementReads.h: a function that returns a vector of
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::unscented(),
        Emrys::measurementModel(
            []<typename T>(auto const & in, auto const &, auto const & x) {
                return Eigen::Matrix<T, Eigen::Dynamic, 1>{Emrys::block<Position>(in, x)};
            },
            Emrys::reads(Emrys::path<Position>)
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(JACOBIAN_AS_BLOCKS)  // static_assert: MeasurementReads.h: a measurement model's function returns its
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            [](auto const & in, auto const &, Eigen::Vector2d const & x) {
                return Emrys::withJacobian(
                    Eigen::Vector2d{Emrys::block<Position>(in, x)},
                    std::tuple{Eigen::Matrix2d{Eigen::Matrix2d::Identity()}}
                );
            },
            Emrys::reads(Emrys::path<Position>)
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(LIKELIHOOD_WITHOUT_NOISE)  // static_assert: Likelihood.h: a measurement model with no additive noise has
[[maybe_unused]] auto const weights = Emrys::logLikelihoods(
    Emrys::observe(Emrys::measurementModel(positionOf, Emrys::reads(Emrys::path<Position>)), Eigen::Vector2d::Zero()),
    State{},
    Eigen::Matrix4d::Zero()
);
#endif

#if defined(LIKELIHOOD_INPUT_NOT_AFFINE)  // static_assert: Likelihood.h: the likelihood at a point of noise that
[[maybe_unused]] auto const weights = Emrys::logLikelihoods(
    Emrys::observe(
        Emrys::measurementModel(
            offsetPosition,
            Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
            Emrys::additiveNoise(makeNoise())
        ),
        Eigen::Vector2d::Zero(),
        makeOffset()
    ),
    State{},
    Eigen::Matrix4d::Zero()
);
#endif

#if defined(SIMULATE_LIKELIHOOD_MODEL)  // static_assert: Simulation.h: a measurement model known by its likelihood
[[maybe_unused]] void
simulateLikelihoodModel()
{
    auto rng = std::mt19937_64{1};
    [[maybe_unused]] auto const drawn = Emrys::simulate(State{}, Eigen::Vector4d::Zero(), makeLikelihoodModel(), rng);
}
#endif

#if defined(SQUARE_ROOT_MIXTURE_NOISE)  // static_assert: FormSteps.h: a square-root or information form takes
[[maybe_unused]] void
squareRootMixtureNoise()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeModel(),
        Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    auto const fix = Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            positionOf,
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(makeMixtureNoise())
        ),
        Eigen::Vector2d::Zero()
    );
    [[maybe_unused]] auto const updated = filter.update(fix);
}
#endif

#if defined(SQUARE_ROOT_REPAIR)  // static_assert: FormSteps.h: repairing the innovation covariance is the covariance
[[maybe_unused]] void
squareRootRepair()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeModel(),
        Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    auto const fix =
        Emrys::observe(Emrys::methods::linearized(), makeFixModel(), Eigen::Vector2d::Zero(), Emrys::roots::ridge());
    [[maybe_unused]] auto const updated = filter.update(fix);
}
#endif

#if defined(LINEAR_MODEL_PLUS)  // static_assert: MeasurementModel.h: a linear measurement model reads blocks with no
[[maybe_unused]] auto const updated = Emrys::update(
    Emrys::GaussianParams<Attitude>{Attitude{}, Eigen::Vector4d{0.0, 0.0, 0.0, 1.0}, Eigen::Matrix3d::Identity()},
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix<double, 2, 4>::Zero(),
            Emrys::reads(Emrys::path<Attitude>),
            Emrys::additiveNoise(makeNoise())
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

#if defined(LINEAR_MODEL_COLUMNS)  // static_assert: MeasurementModel.h: a linear measurement model's matrix must have
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::linearized(),
        Emrys::measurementModel(
            Eigen::Matrix<double, 2, 3>::Zero(),
            Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
            Emrys::additiveNoise(makeNoise())
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

// What measurementModel() checks before the model meets a state ([sensors]): the
// count of a function's parameters where its type gives it, and the noise's rows against an output
// space known without the state. None of these cases updates anything.

#if defined(MODEL_FUNCTION_ARITY)  // static_assert: MeasurementModel.h: a measurement model's function takes one to
                                   // three arguments
[[maybe_unused]] auto const model = Emrys::measurementModel(
    [](Eigen::Vector2d const & x, double, double, double) {
        return x;
    },
    Emrys::reads(Emrys::path<Position>),
    Emrys::additiveNoise(makeNoise())
);
#endif

#if defined(MODEL_NOISE_SIZE_WRITES)  // static_assert: MeasurementModel.h: a measurement model's additive noise has a
                                      // row for each difference coordinate of the output space
[[maybe_unused]] auto const model = Emrys::measurementModel(
    positionOf,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Fix>),
    Emrys::additiveNoise(Emrys::GaussianParams<Emrys::Linear<3>>{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()})
);
#endif

#if defined(MODEL_NOISE_SIZE_PLACED)  // static_assert: MeasurementModel.h: a measurement model's additive noise has a
                                      // row for each difference coordinate of the output space
[[maybe_unused]] auto const model = Emrys::measurementModel(
    positionOf,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Fix>),
    Emrys::additiveNoise(Emrys::on<Fix>(Eigen::Matrix3d::Identity()))
);
#endif

#if defined(MODEL_NOISE_SIZE_EACH)  // static_assert: MeasurementModel.h: a measurement model's additive noise has a
                                    // row for each difference coordinate of the output space
[[maybe_unused]] auto const model = Emrys::measurementModel(
    positionOf,
    Emrys::reads(Emrys::path<Position>),
    Emrys::writes(Emrys::path<Fixes>),
    Emrys::additiveNoise(Emrys::onEach<Fixes>(Eigen::Matrix3d::Identity()))
);
#endif

#if defined(MODEL_NOISE_SIZE_MATRIX)  // static_assert: MeasurementModel.h: a measurement model's additive noise has a
                                      // row for each difference coordinate of the output space
[[maybe_unused]] auto const model = Emrys::measurementModel(
    Eigen::Matrix2d::Identity(),
    Emrys::reads(Emrys::path<Position>),
    Emrys::additiveNoise(Emrys::GaussianParams<Emrys::Linear<3>>{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()})
);
#endif

// A generic function in no form whose output space comes from its result fails at the update with
// callForm()'s message, not with one about the output space (Q10).
#if defined(FUNCTION_IN_NO_FORM_INFERRED)  // static_assert: Methods.h: a model's function takes h(x), h(in, x), or
                                           // h(in, out, x)
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(
        Emrys::methods::unscented(),
        Emrys::measurementModel(
            [](auto const &, auto const &, auto const &, auto const &) {
                return Eigen::Vector2d{Eigen::Vector2d::Zero()};
            },
            Emrys::reads(Emrys::path<Position>)
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

// The posterior Cramer-Rao bound.

#if defined(BOUND_UNCERTAIN_INPUT)  // static_assert: CramerRao.h: noise that the transition density of the bound
[[maybe_unused]] void
boundUncertainInput()
{
    auto bound = Emrys::PosteriorCramerRaoBound<State>::create(makeInitial()).value();
    Eigen::Matrix<double, 4, 3> const truths = Eigen::Matrix<double, 4, 3>::Zero();
    [[maybe_unused]] auto const predicted = bound.predict(makeModel(), truths, 1.0, makeNoise());
}
#endif

#if defined(BOUND_UNCERTAIN_MEASUREMENT_INPUT)  // static_assert: CramerRao.h: noise that the measurement density
[[maybe_unused]] void
boundUncertainMeasurementInput()
{
    auto bound = Emrys::PosteriorCramerRaoBound<State>::create(makeInitial()).value();
    Eigen::Matrix<double, 4, 3> const truths = Eigen::Matrix<double, 4, 3>::Zero();
    [[maybe_unused]] auto const updated = bound.update(truths, makeOffsetModel(), makeOffset());
}
#endif

#if defined(BOUND_MIXTURE_NOISE)  // static_assert: CramerRao.h: the bound takes a measurement model with additive
[[maybe_unused]] void
boundMixtureNoise()
{
    auto bound = Emrys::PosteriorCramerRaoBound<State>::create(makeInitial()).value();
    Eigen::Matrix<double, 4, 3> const truths = Eigen::Matrix<double, 4, 3>::Zero();
    [[maybe_unused]] auto const updated = bound.update(
        truths,
        Emrys::measurementModel(
            positionOf,
            Emrys::reads(Emrys::path<Position>),
            Emrys::additiveNoise(makeMixtureNoise())
        )
    );
}
#endif

// Measurement spaces computed from the state's space.

struct Landmark : Emrys::Linear<2>
{};

struct Landmarks : Emrys::Repeated<Landmark, 4>
{
    using Repeated::Repeated;
};

struct Ranges : Emrys::Repeated<Emrys::Linear<1>, 4>
{
    using Repeated::Repeated;
};

using Map = Emrys::Composite<Position, Landmarks>;

[[maybe_unused]] Emrys::GaussianParams<Map>
makeMap()
{
    auto const space = Map{Position{}, Landmarks{Landmark{}, 1}};
    return {space, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()};
}

#if defined(COUNT_FROM_FIXED_MEASUREMENT)  // static_assert: MeasurementModel.h: CountFrom's measurement space must have
[[maybe_unused]] constexpr auto size = sizeof(Emrys::CountFrom<Landmarks, Emrys::Linear<2>>);
#endif

#if defined(COUNT_FROM_FIXED_BLOCK)  // static_assert: MeasurementModel.h: CountFrom's block of the state must have
[[maybe_unused]] auto const ranges = Emrys::CountFrom<Position, Ranges>{}(Map{});
#endif

#if defined(COUNT_FROM_SMALLER_MEASUREMENT)  // static_assert: MeasurementModel.h: must hold a block per repetition
[[maybe_unused]] auto const ranges = Emrys::CountFrom<Landmarks, Emrys::Repeated<Emrys::Linear<1>, 2>>{}(Map{});
#endif

// Linear models by their columns.

// Updates makeMap() by a model with the columns placed, one measurement of makeNoise() per landmark.
template <typename... Placed>
[[maybe_unused]] void
updateByColumns(Placed... placed)
{
    auto const relative =
        Emrys::measurementModel(Emrys::Columns{placed...}, Emrys::additiveNoise(Emrys::onEach<>(makeNoise())));
    [[maybe_unused]] auto const updated =
        Emrys::update(makeMap(), Emrys::observe(Emrys::methods::linearized(), relative, Eigen::Vector2d::Zero()));
}

#if defined(COLUMNS_NONE)  // static_assert: MeasurementModel.h: a model's matrix needs at least one block of columns
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Columns<>);
#endif

#if defined(COLUMNS_NOT_PLACED)  // static_assert: MeasurementModel.h: each block of a model's columns is a matrix
                                 // placed
[[maybe_unused]] auto const columns = Emrys::Columns{Eigen::Matrix2d::Identity().eval()};
#endif

#if defined(COLUMNS_SAME_BLOCK)  // static_assert: MeasurementModel.h: a model's columns name each block of the state
[[maybe_unused]] auto const columns =
    Emrys::Columns{Emrys::on<Position>(Eigen::Matrix2d::Identity()), Emrys::on<Position>(Eigen::Matrix2d::Identity())};
#endif

#if defined(COLUMNS_ROWS)  // static_assert: MeasurementModel.h: every block of a model's columns has the rows of one
[[maybe_unused]] auto const columns = Emrys::Columns{
    Emrys::on<Position>(Eigen::Matrix2d::Identity()),
    Emrys::on<Velocity>(Eigen::Matrix<double, 3, 2>::Zero())
};
#endif

// The same, into the output space given.
template <typename Space, typename... Placed>
[[maybe_unused]] void
updateByColumnsInto(Space const & space, Placed... placed)
{
    auto const relative = Emrys::measurementModel(
        Emrys::Columns{placed...},
        Emrys::outputSpace(space),
        Emrys::additiveNoise(Emrys::onEach<>(makeNoise()))
    );
    [[maybe_unused]] auto const updated =
        Emrys::update(makeMap(), Emrys::observe(Emrys::methods::linearized(), relative, Eigen::Vector2d::Zero()));
}

#if defined(COLUMNS_EACH_FIXED_BLOCK)  // static_assert: MeasurementModel.h: repeats over a resizable block of the state
[[maybe_unused]] auto const run = (updateByColumns(Emrys::onEach<Position>(Eigen::Matrix2d::Identity())), 0);
#endif

#if defined(COLUMNS_EACH_FIXED_BLOCK_INTO)  // static_assert: MeasurementModel.h: a model's onEach<Path>() names a
                                            // repeated
[[maybe_unused]] auto const run =
    (updateByColumnsInto(
         Emrys::Repeated<Emrys::Linear<2>, 4>{Emrys::Linear<2>{}, 1},
         Emrys::onEach<Position>(Eigen::Matrix2d::Identity())
     ),
     0);
#endif

#if defined(COLUMNS_EACH_COLUMNS)  // static_assert: MeasurementModel.h: a model's onEach<Path>() matrix has a column
[[maybe_unused]] auto const run = (updateByColumns(Emrys::onEach<Landmarks>(Eigen::Matrix<double, 2, 3>::Zero())), 0);
#endif

#if defined(COLUMNS_DYNAMIC_ROWS)  // static_assert: MeasurementModel.h: a model whose columns repeat infers its
[[maybe_unused]] auto const model = Emrys::measurementModel(
    Emrys::Columns{Emrys::onEach<Landmarks>(Eigen::MatrixXd::Identity(2, 2))},
    Emrys::additiveNoise(makeNoise())
);
#endif

#if defined(COLUMNS_WITH_READS)  // static_assert: MeasurementModel.h: a model given Columns reads the blocks its
                                 // columns
[[maybe_unused]] auto const model = Emrys::measurementModel(
    Emrys::Columns{Emrys::on<Position>(Eigen::Matrix2d::Identity())},
    Emrys::reads(Emrys::path<Position>),
    Emrys::additiveNoise(makeNoise())
);
#endif

// A model that misses a block of the state, and marks of temporaries, which would dangle.

#if defined(MODEL_MISSES_A_BLOCK)  // static_assert: ProcessModel.h: the process model must write every coordinate of
[[maybe_unused]] void
modelMissesABlock()
{
    using Wider = Emrys::Composite<Position, Velocity, Speed>;
    auto const prior = Emrys::GaussianParams<Wider>{
        Wider{},
        Eigen::Matrix<double, 5, 1>::Zero(),
        Eigen::Matrix<double, 5, 5>::Identity()
    };
    [[maybe_unused]] auto const predicted = Emrys::predict(Emrys::methods::linearized(), prior, makeModel(), 1.0);
}
#endif

#if defined(GIVEN_OF_TEMPORARY)  // deleted: Measurement.h: given(Space const & space, Value const && value)
[[maybe_unused]] auto const mark = Emrys::given<Position>(State{}, Eigen::Vector2d{});
#endif

#if defined(WITH_CROSS_OF_TEMPORARY)  // deleted: Transform.h: withCross(Input const && input)
[[maybe_unused]] auto const mark = Emrys::withCross(makeNoise());
#endif

// Steps linearized about a reference.

#if defined(LINEARIZED_ABOUT_TEMPORARY)  // deleted: Transform.h: linearizedAbout(Distribution const && distribution)
[[maybe_unused]] auto const about = Emrys::linearizedAbout(makeNoise());
#endif

#if defined(LINEARIZED_ABOUT_TWICE)  // static_assert: MeasurementModel.h: give the linearizedAbout() reference once
[[maybe_unused]] void
linearizedAboutTwice()
{
    auto const about = makeInitial();
    [[maybe_unused]] auto const fix = Emrys::observe(
        Emrys::methods::linearized(),
        makeFixModel(),
        Eigen::Vector2d::Zero(),
        Emrys::linearizedAbout(about),
        Emrys::linearizedAbout(about)
    );
}
#endif

#if defined(PREDICT_REFERENCE_TWICE)  // static_assert: ProcessModel.h: a prediction's linearizedAbout() reference comes
[[maybe_unused]] auto const predicted = Emrys::predict(
    Emrys::methods::linearized(),
    makeInitial(),
    makeModel(),
    1.0,
    Emrys::linearizedAbout(makeInitial().mean()),
    Emrys::linearizedAbout(makeInitial().mean())
);
#endif

#if defined(LINEARIZED_ABOUT_MATRIX)  // static_assert: Transform.h: a point to linearize at is a column vector
[[maybe_unused]] auto const about = Emrys::linearizedAbout(Eigen::Matrix2d{Eigen::Matrix2d::Identity()});
#endif

#if defined(LINEARIZED_ABOUT_KNOWN)  // static_assert: Transform.h: linearizedAbout() is followed by the uncertain input
auto const reference = makePosition();
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::linearized(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    Emrys::linearizedAbout(reference),
    Emrys::sure<Position>(Eigen::Vector2d{Eigen::Vector2d::Zero()}),
    makeOffset()
);
#endif

#if defined(LINEARIZED_ABOUT_OTHER_CROSS)  // static_assert: Transform.h: withCross() may mark only the input
                                           // linearizedAbout()
auto const reference = makePosition();
auto const other = makeOffset();
[[maybe_unused]] auto const moved = Emrys::transform(
    Emrys::methods::linearized(),
    sum,
    Emrys::reads(Emrys::path<Position>, Emrys::path<Offset>),
    Emrys::linearizedAbout(reference),
    reference,
    withCross(other)
);
#endif

#if defined(LINEARIZED_ABOUT_ITERATED)  // static_assert: Transform.h: an Iterated update chooses its own points
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    makeFix(Emrys::methods::iterated()),
    Emrys::linearizedAbout(Eigen::Vector4d{Eigen::Vector4d::Zero()})
);
#endif

#if defined(LINEARIZED_ABOUT_OTHER_SPACE)  // static_assert: Transform.h: linearizedAbout() takes a point or a
                                           // distribution
auto const reference = makeNoise();
[[maybe_unused]] auto const updated =
    Emrys::update(makeInitial(), makeFix(Emrys::methods::linearized()), Emrys::linearizedAbout(reference));
#endif

#if defined(LINEARIZED_ABOUT_POINT_FOR_SIGMA_POINTS)  // static_assert: Transform.h: a sigma-point method linearizes
                                                      // over
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    makeFix(Emrys::methods::cubature()),
    Emrys::linearizedAbout(Eigen::Vector4d{Eigen::Vector4d::Zero()})
);
#endif

#if defined(SPREAD_NOT_A_RULE)  // static_assert: Methods.h: spread() takes a sigma-point rule
[[maybe_unused]] auto const spreadLinearized = methods::spread(methods::linearized(), Emrys::roots::principal());
#endif

#if defined(SPREAD_NOT_AXES)  // static_assert: Methods.h: spread() takes axes
[[maybe_unused]] auto const spreadByRidge = methods::spread(methods::cubature(), Emrys::roots::ridge());
#endif

#if defined(TRANSFORM_AXES_CAN_FAIL)  // static_assert: Methods.h: a transform returns no std::expected
[[maybe_unused]] auto const failingAxes = Emrys::transform(
    methods::spread(methods::cubature(), Emrys::roots::cholesky()),
    [](Eigen::Vector2d const & p) {
        return p;
    },
    Emrys::GaussianParams<Position>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}
);
#endif

// The policies of a model (Policies.h) and the marks on a step's arguments.

struct Ownship : Emrys::Composite<Position, Velocity>
{};

struct Target : Emrys::Composite<Position, Velocity>
{};

#if defined(COMPOSE_SHARED_NAME)  // static_assert: ProcessModel.h: two parts of a composition read different blocks
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::RandomWalk<Emrys::TagPath<Ownship, Position>>{0.1},
    Emrys::RandomWalk<Emrys::TagPath<Target, Position>>{0.1}
);
#endif

#if defined(READS_NOT_A_PATH)  // static_assert: Policies.h: reads() takes paths, path<Tags...> or a tag's value
[[maybe_unused]] auto const r = Emrys::reads(3);
#endif

#if defined(READS_SHARED_NAME)  // static_assert: Policies.h: two items of reads() share a name
[[maybe_unused]] auto const r = Emrys::reads(Emrys::path<Ownship, Position>, Emrys::path<Target, Position>);
#endif

#if defined(WRITES_NOTHING)  // static_assert: Policies.h: writes() names at least one block
[[maybe_unused]] auto const w = Emrys::writes();
#endif

#if defined(WRITES_A_BLOCK_TWICE)  // static_assert: Policies.h: writes() names each block once
[[maybe_unused]] auto const w = Emrys::writes(Emrys::path<Position>, Position{});
#endif

#if defined(AFFINE_IN_A_PATH)  // static_assert: Policies.h: affineIn() names an input of the model by the one tag
[[maybe_unused]] auto const a = Emrys::affineIn(Emrys::path<Ownship, Position>);
#endif

#if defined(AFFINE_IN_AN_INPUT_TWICE)  // static_assert: Policies.h: affineIn() names each input once
[[maybe_unused]] auto const a = Emrys::affineIn(Position{}, Emrys::path<Position>);
#endif

#if defined(ADDITIVE_NOISE_NOTHING)  // static_assert: Policies.h: additiveNoise() takes a distribution, or parts placed
[[maybe_unused]] auto const n = Emrys::additiveNoise();
#endif

#if defined(ADDITIVE_NOISE_UNPLACED_BESIDE_PLACED)  // static_assert: Policies.h: additiveNoise() with no placement
                                                    // covers
[[maybe_unused]] auto const n = Emrys::additiveNoise(makeNoise(), Emrys::on<Position>(makeNoise()));
#endif

#if defined(ADDITIVE_NOISE_PLACED_MIXTURE)  // static_assert: Policies.h: a mixture is additive noise over the whole
[[maybe_unused]] auto const n = Emrys::additiveNoise(Emrys::on<Position>(makeMixtureNoise()));
#endif

#if defined(CONTINUOUS_NOISE_NOTHING)  // static_assert: Policies.h: continuousNoise() takes a spectral density
[[maybe_unused]] auto const q = Emrys::continuousNoise();
#endif

#if defined(CONTINUOUS_NOISE_UNPLACED_BESIDE_PLACED)  // static_assert: Policies.h: continuousNoise() with no placement
[[maybe_unused]] auto const q = Emrys::continuousNoise(0.1, Emrys::on<Position>(0.2));
#endif

#if defined(OUTPUT_SPACE_OF_ANOTHER_ARGUMENT)  // static_assert: Policies.h: a model's outputSpace() function takes
[[maybe_unused]] auto const space = Emrys::outputSpace([](int) {
                                        return Pose{};
                                    }).in(State{});
#endif

#if defined(OUTPUT_SPACE_RETURNS_NO_SPACE)  // static_assert: Policies.h: a model's outputSpace() function returns
[[maybe_unused]] auto const space = Emrys::outputSpace([](State const &) {
                                        return 1;
                                    }).in(State{});
#endif

#if defined(POLICY_NOT_A_POLICY)  // static_assert: Policies.h: a model takes its function and then its policies
[[maybe_unused]] constexpr auto size = sizeof(Emrys::internal::PolicySet<int>);
#endif

#if defined(POLICY_TWICE)  // static_assert: Policies.h: a model takes each policy at most once
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::internal::PolicySet<decltype(Emrys::writes(Position{})), decltype(Emrys::writes(Velocity{}))>);
#endif

#if defined(POLICY_MISSING)  // static_assert: Policies.h: the model was not given this policy
[[maybe_unused]] auto const & missing =
    Emrys::internal::PolicySet<decltype(Emrys::writes(Position{}))>::get<Emrys::internal::PolicyKind::reads>(
        std::tuple{Emrys::writes(Position{})}
    );
#endif

#if defined(READ_NAME_OF_ANOTHER_LEAF)  // static_assert: Policies.h: a name given by on<Name>() in reads() must have
[[maybe_unused]] constexpr bool checked =
    Emrys::internal::checkReadNames<decltype(Emrys::reads(Emrys::on<Heading>(Emrys::path<Position>))), State>();
#endif

#if defined(SURE_UNTAGGED)  // static_assert: Policies.h: a sure input is named by a tag
[[maybe_unused]] auto const g = Emrys::sure<Emrys::Linear<2>>(Eigen::Vector2d::Zero());
#endif

#if defined(SURE_MATRIX)  // static_assert: Policies.h: a sure input is a column vector
[[maybe_unused]] auto const g = Emrys::sure<Position>(Eigen::Matrix2d::Identity());
#endif

#if defined(SURE_ROWS)  // static_assert: Policies.h: a sure input's rows do not match its space's dimension
[[maybe_unused]] auto const g = Emrys::sure<Position>(Eigen::Vector3d::Zero());
#endif

// Placement and composition (Placement.h, roadmap 17z).
struct Vehicle : Emrys::Composite<Position, Velocity>
{};

struct Other : Vehicle
{};

struct Fleet : Emrys::Repeated<Vehicle, 4>
{
    using Repeated::Repeated;
};

struct Force : Emrys::Linear<2>
{};

struct Forces : Emrys::Repeated<Force, 4>
{
    using Repeated::Repeated;
};

struct Range : Emrys::Linear<1>
{};

struct Bearing : Emrys::Linear<1>
{};

struct Many : Emrys::Linear<Eigen::Dynamic, 4>
{};

struct OwnPosition : Emrys::Linear<2>
{};

struct OtherPosition : Emrys::Linear<2>
{};

// A vehicle pushed by a force it reads as an input.
[[maybe_unused]] auto
pushed()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + dt * Emrys::block<Force>(in, x);
            Emrys::block<Velocity>(out, y) = Emrys::block<Velocity>(in, x);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>, Emrys::path<Force>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>)
    );
}

// A measurement of one scalar block from the position.
template <typename Out>
[[maybe_unused]] auto
scalarOf(auto... policies)
{
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return T{Emrys::block<Position>(in, x).norm()};
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Out>),
        policies...
    );
}

#if defined(ON_RENAME_NOT_A_RENAME)  // static_assert: Placement.h: on<Path>(model, renames...) renames an input
[[maybe_unused]] auto const model = Emrys::on<Ownship>(pushed(), 3);
#endif

#if defined(ON_RENAME_NOT_AN_INPUT)  // static_assert: Placement.h: a rename in on<Path>(model, on<Name>(path<Tag>))
[[maybe_unused]] auto const model = Emrys::on<Ownship>(pushed(), Emrys::on<OwnPosition>(Emrys::path<Position>));
#endif

#if defined(ON_RENAME_OTHER_LEAF)  // static_assert: Placement.h: a renamed input has the same leaf beneath it
[[maybe_unused]] auto const model = Emrys::on<Ownship>(pushed(), Emrys::on<Speed>(Emrys::path<Force>));
#endif

#if defined(ON_WRITES_OUTSIDE)  // static_assert: Placement.h: on<Path>(model) places a model whose writes() are
[[maybe_unused]] auto const model = Emrys::on<Ownship>(Emrys::RandomWalk<Speed>{0.1});
#endif

#if defined(EACH_NOT_REPEATED)  // static_assert: Placement.h: onEach<Path>(model) places a model on a repeated block
[[maybe_unused]] auto const model = Emrys::onEach<Ownship>(pushed());
#endif

#if defined(EACH_RENAME_NOT_A_RENAME)  // static_assert: Placement.h: onEach<Path>(model, renames...) renames an
[[maybe_unused]] auto const model = Emrys::onEach<Fleet>(pushed(), 3);
#endif

#if defined(EACH_RENAME_NOT_AN_INPUT)  // static_assert: Placement.h: a rename in onEach<Path>(model, on<Name>(path
[[maybe_unused]] auto const model = Emrys::onEach<Fleet>(pushed(), Emrys::on<Forces>(Emrys::path<Position>));
#endif

#if defined(EACH_REPEATED_LEAF)  // static_assert: Placement.h: an input renamed to a Repeated tag repeats the leaf
[[maybe_unused]] auto const model = Emrys::onEach<Fleet>(pushed(), Emrys::on<Fleet>(Emrys::path<Force>));
#endif

#if defined(EACH_RESIZABLE_INPUT)  // static_assert: Placement.h: onEach<Path>(model) repeats a model whose input and
[[maybe_unused]] auto const model = Emrys::onEach<Fleet>(Emrys::processModel(
    []<typename T>(auto const & in, auto const & out, auto const & x, double) {
        auto y = Emrys::zeros<T>(out);
        Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + Emrys::block<Many>(in, x).template head<2>();
        Emrys::block<Velocity>(out, y) = Emrys::block<Velocity>(in, x);
        return y;
    },
    Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>, Emrys::path<Many>),
    Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>)
));
#endif

#if defined(EACH_PARTIAL_WRITES)  // static_assert: Placement.h: onEach<Path>(model) places a model that writes every
[[maybe_unused]] auto const model = Emrys::onEach<Fleet>(Emrys::RandomWalk<Position>{0.1});
#endif

#if defined(COMPOSE_WRITES_ONE_NAME)  // static_assert: ProcessModel.h: two parts of a composition write blocks under
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::processModel(
        [](auto const &, auto const &, auto const & x, double) {
            return x.eval();
        },
        Emrys::reads(Emrys::on<OwnPosition>(Emrys::path<Ownship, Position>)),
        Emrys::writes(Emrys::path<Ownship, Position>)
    ),
    Emrys::processModel(
        [](auto const &, auto const &, auto const & x, double) {
            return x.eval();
        },
        Emrys::reads(Emrys::on<OtherPosition>(Emrys::path<Other, Position>)),
        Emrys::writes(Emrys::path<Other, Position>)
    )
);
#endif

#if defined(COMPOSE_MEASUREMENT_NO_WRITES)  // static_assert: Placement.h: compose() takes measurement models that
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::measurementModel(Eigen::Matrix2d::Identity(), Emrys::reads(Emrys::path<Position>)),
    scalarOf<Range>()
);
#endif

#if defined(COMPOSE_LIKELIHOOD_ALONE)  // static_assert: Placement.h: compose() takes measurement models with a
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::measurementModel(
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Bearing>),
        Emrys::logLikelihood([](auto const &, auto const &, auto const &) {
            return 0.0;
        })
    ),
    scalarOf<Range>()
);
#endif

#if defined(COMPOSE_OUTPUT_SPACE)  // static_assert: Placement.h: compose() takes measurement models whose output
[[maybe_unused]] auto const model =
    Emrys::compose(scalarOf<Range>(Emrys::outputSpace(Emrys::Composite<Range>{})), scalarOf<Bearing>());
#endif

#if defined(COMPOSE_SOME_LIKELIHOODS)  // static_assert: Placement.h: in a composition of measurement models, every
[[maybe_unused]] auto const model = Emrys::compose(
    scalarOf<Range>(Emrys::logLikelihood([](auto const &, auto const &, auto const &) {
        return 0.0;
    })),
    scalarOf<Bearing>()
);
#endif

#if defined(COMPOSE_WHOLE_NOISE)  // static_assert: Placement.h: a part of a composition whose additiveNoise() covers
[[maybe_unused]] auto const model = Emrys::compose(
    Emrys::measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return Emrys::block<Position>(in, x).eval();
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Range>, Emrys::path<Bearing>),
        Emrys::additiveNoise(
            Emrys::GaussianParams<Emrys::Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()}
        )
    ),
    scalarOf<Speed>()
);
#endif

// The log-likelihoods of a batched likelihood marked by mark, whose body writes y as write does, at two
// states.
template <typename Mark, typename Write>
[[maybe_unused]] auto
batchedLikelihoods([[maybe_unused]] Mark mark, Write write)
{
    auto const function = [write](auto const &, auto const &, auto const &, auto & y) {
        y.values().setZero();
        write(y);
    };
    auto const batched = [&] {
        if constexpr (std::is_same_v<Mark, Emrys::internal::NoJacobians>)
        {
            return Emrys::batched(function);
        }
        else
        {
            return Emrys::batched(function, mark);
        }
    }();
    auto const measurement = observe(
        Emrys::measurementModel(Emrys::reads(Emrys::path<Position>), Emrys::logLikelihood(batched)),
        Eigen::Matrix<double, 1, 1>{1.0}
    );
    return Emrys::logLikelihoods<State>(measurement, Eigen::Matrix<double, 4, 2>::Zero());
}

#if defined(BATCHED_JACOBIAN_WITHOUT_MARK)  // static_assert: Batched.h: a batched function marked without Jacobians
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::internal::NoJacobians{}, [](auto & y) {
    y.template jacobian<Position>(0).setZero();
});
#endif

#if defined(BATCHED_JACOBIAN_NOT_READ)  // static_assert: Batched.h: a batched function writes a Jacobian only for a
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::jacobians::perColumn(), [](auto & y) {
    y.template jacobian<Velocity>(0).setZero();
});
#endif

#if defined(BATCHED_NOT_GENERIC)  // static_assert: Batched.h: a batched function whose Jacobian a method uses is
struct BatchOffset : Emrys::Linear<1>
{};

// The likelihood at a state widens the noise by the offset's covariance through the function's
// Jacobian in it, which a batched function that takes X as a reference to a matrix of doubles can't give.
[[maybe_unused]] auto const values = Emrys::logLikelihoods<State>(
    observe(
        Emrys::measurementModel(
            Emrys::batched([](auto const &, auto const &, Eigen::Ref<Eigen::MatrixXd const> x, auto & y) {
                y.values() = x.topRows(1);
            }),
            Emrys::reads(Emrys::path<Position>, Emrys::path<BatchOffset>),
            Emrys::additiveNoise(
                Emrys::GaussianParams<Emrys::Linear<1>>{
                    Eigen::Matrix<double, 1, 1>{0.0},
                    Eigen::Matrix<double, 1, 1>{1.0}
                }
            ),
            Emrys::affineIn(Emrys::path<BatchOffset>)
        ),
        Eigen::Matrix<double, 1, 1>{1.0},
        Emrys::GaussianParams<BatchOffset>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}}
    ),
    Eigen::Matrix<double, 4, 2>::Zero()
);
#endif

#if defined(BATCHED_SHARED_PER_COLUMN)  // static_assert: Batched.h: a batched function marked jacobians::shared()
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::jacobians::shared(), [](auto & y) {
    y.template jacobian<Position>(0).setZero();
});
#endif

#if defined(BATCHED_PER_COLUMN_SHARED)  // static_assert: Batched.h: a batched function marked jacobians::perColumn()
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::jacobians::perColumn(), [](auto & y) {
    y.template jacobian<Position>().setZero();
});
#endif

#if defined(BATCHED_WHOLE_SHARED_PER_COLUMN)  // static_assert: Batched.h: under jacobians::shared(), a batched
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::jacobians::shared(), [](auto & y) {
    y.jacobian(0).setZero();
});
#endif

#if defined(BATCHED_WHOLE_PER_COLUMN_SHARED)  // static_assert: Batched.h: under jacobians::perColumn(), a batched
[[maybe_unused]] auto const values = batchedLikelihoods(Emrys::jacobians::perColumn(), [](auto & y) {
    y.jacobian().setZero();
});
#endif

#if defined(BATCHED_WITHOUT_OUTPUT_SPACE)  // static_assert: Batched.h: a batched function returns nothing to size its
[[maybe_unused]] auto const updated = Emrys::update(
    Emrys::GaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()},
    observe(
        methods::linearized(),
        Emrys::measurementModel(
            Emrys::batched([](auto const &, auto const &, auto const &, auto &) {}),
            Emrys::reads(Emrys::path<Position>)
        ),
        Eigen::Vector2d::Zero()
    )
);
#endif

// The consider update [consider-states].

#if defined(CONSIDER_NOTHING)  // static_assert: Policies.h: consider() names at least one block of the state
[[maybe_unused]] auto const considered = Emrys::consider();
#endif

#if defined(CONSIDER_TWICE)  // static_assert: Policies.h: consider() names each block once
[[maybe_unused]] auto const considered = Emrys::consider(Emrys::path<Velocity>, Velocity{});
#endif

#if defined(CONSIDER_TWO_ARGUMENTS)  // static_assert: MeasurementModel.h: give the blocks to consider in one consider()
[[maybe_unused]] auto const fix = Emrys::observe(
    methods::linearized(),
    makeFixModel(),
    Eigen::Vector2d::Zero(),
    Emrys::consider(Emrys::path<Velocity>),
    Emrys::consider(Emrys::path<Position>)
);
#endif

#if defined(CONSIDER_NO_BLOCK)  // static_assert: MeasurementReads.h: a path of consider() names no block of the state
[[maybe_unused]] auto const updated = Emrys::update(
    makeInitial(),
    Emrys::observe(methods::linearized(), makeFixModel(), Eigen::Vector2d::Zero(), Emrys::consider(Emrys::path<Speed>))
);
#endif

#if defined(CONSIDER_NESTED_BLOCKS)  // static_assert: MeasurementReads.h: a block of consider() lies in another it
                                     // names
[[maybe_unused]] void
considerNestedBlocks()
{
    using Nested = Emrys::Composite<State, Speed>;
    auto const prior = Emrys::GaussianParams<Nested>{
        Nested{},
        Eigen::Matrix<double, 5, 1>::Zero(),
        Eigen::Matrix<double, 5, 5>::Identity()
    };
    [[maybe_unused]] auto const updated = Emrys::update(
        prior,
        Emrys::observe(
            methods::linearized(),
            makeFixModel(),
            Eigen::Vector2d::Zero(),
            Emrys::consider(Emrys::path<State>, Emrys::path<State, Velocity>)
        )
    );
}
#endif

#if defined(SQUARE_ROOT_CONSIDER)  // static_assert: FormSteps.h: a consider() update is the covariance form's
[[maybe_unused]] void
squareRootConsider()
{
    auto filter = Emrys::SqrtKalmanFilter{
        makeModel(),
        Emrys::SqrtGaussianParams<State>{State{}, Eigen::Vector4d::Zero(), Eigen::Matrix4d::Identity()}
    };
    auto const fix = Emrys::observe(
        Emrys::methods::linearized(),
        makeFixModel(),
        Eigen::Vector2d::Zero(),
        Emrys::consider(Emrys::path<Velocity>)
    );
    [[maybe_unused]] auto const updated = filter.update(fix);
}
#endif

#if defined(LIKELIHOOD_CONSIDER)  // static_assert: Likelihood.h: consider() is a Kalman update's
[[maybe_unused]] auto const weights = Emrys::logLikelihoods(
    Emrys::observe(makeLikelihoodModel(), Eigen::Vector2d::Zero(), Emrys::consider(Emrys::path<Velocity>)),
    State{},
    Eigen::Matrix4d::Zero()
);
#endif

#if defined(GIVEN_BLOCK_CONSIDER)  // static_assert: Measurement.h: an update given a block, as a Rao-Blackwellized
[[maybe_unused]] void
givenBlockConsider()
{
    auto const position = Eigen::Vector2d::Zero().eval();
    auto const velocity = Emrys::GaussianParams<Velocity>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    auto const velocityOf = Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Eigen::Matrix<T, 2, 1>{Emrys::block<Position>(in, x) + Emrys::block<Velocity>(in, x)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::additiveNoise(makeNoise())
    );
    [[maybe_unused]] auto const updated = Emrys::update(
        velocity,
        Emrys::observe(
            methods::linearized(),
            velocityOf,
            Eigen::Vector2d::Zero(),
            Emrys::consider(Emrys::path<Velocity>)
        ),
        Emrys::given<Position>(State{}, position)
    );
}
#endif

#if defined(BY_MODE_NOT_A_MIXTURE)  // static_assert: Policies.h: byMode() takes additive noise that is one mixture
struct Clear
{};

struct Blocked
{};

struct Outliers : Emrys::Named<Clear, Blocked>
{};

[[maybe_unused]] auto const byModeNoise = Emrys::additiveNoise(makeNoise()).byMode<Outliers>();
#endif

}  // namespace

#pragma once

/** \file
 * simulate(): a draw of a process model's next true state, the ground truth that a simulation
 * runs a filter against and that nees() and NeesMonitor test its estimates by, and a draw of a
 * measurement model's observation of a true state, which observe() turns into the measurement the
 * filter takes, and a draw of a Markov chain's next value, the true mode of a switching or hidden
 * Markov model (decisions.md [printing-and-simulation]). Read Consistency.h next.
 */

#include <random>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

namespace internal
{

/** \returns \a moved, a propagated state in \a space, with a draw from \f$ \mathcal{N}(0, Q) \f$
 * added by \f$ \boxplus \f$, for the process noise \a noise: the body of simulate(). The draw is
 * \f$ F z \f$ for standard normals \f$ z \f$ drawn in order from \a rng and the semidefinite
 * factor \f$ F F^T = Q \f$, as a particle filter's prediction moves a particle, so a noiseless
 * block stays exact.
 */
template <typename Space_, typename Vector, typename Noise, typename G>
[[nodiscard]] Vector
drawProcessNoise(Space_ const & space, Vector const & moved, Eigen::MatrixBase<Noise> const & noise, G & rng)
{
    using Scalar = typename Vector::Scalar;
    using Difference = DifferenceVector<Space_, Scalar>;
    Difference z = Difference::Zero(differenceDimension(space));
    fillStandardNormal(z, rng);
    auto const factor = defaultRoot(noise);
    return Emrys::plus(space, moved, Difference{multiply(factor, z)});
}

/** \returns a draw from \a distribution, a measurement model's noise or an uncertain input of the
 * call, by drawSample() of its density: the density its toDensity() returns, which must exist, or
 * itself when it has none
 */
template <typename Distribution, typename G>
[[nodiscard]] auto
drawMeasurementNoise(Distribution const & distribution, G & rng)
{
    if constexpr (requires { distribution.toDensity(); })
    {
        auto const density = distribution.toDensity();
        EMRYS_FILTER_ASSERT(density && "the measurement's noise must factorize");
        return drawSample(*density, rng);
    }
    else
    {
        return drawSample(distribution, rng);
    }
}

/** \returns the step of the resolved process model \a resolved over \a step from the true state
 * \a x, with each uncertain input of the call drawn from its distribution, in the order the model
 * reads them, and each sure input as given: the body of simulate() before the process noise
 */
template <typename R, typename Derived, typename Scalar, typename G>
[[nodiscard]] typename Moments<typename R::SpaceType, Scalar>::VectorType
drawStep(R const & resolved, Eigen::MatrixBase<Derived> const & x, Scalar step, G & rng)
{
    using Vector = typename Moments<typename R::SpaceType, Scalar>::VectorType;
    auto const draws = std::apply(
        [&](auto const &... inputs) {
            return std::tuple{drawMeasurementNoise(inputOf(inputs), rng)...};
        },
        resolved.inputs()
    );
    return std::apply(
        [&](auto const &... drawn) {
            return Vector{valueOf(resolved.propagate(x.derived(), step, drawn...))};
        },
        draws
    );
}

}  // namespace internal

/** \ingroup Models
 * \returns a draw of the true state a time step \a dt after the true state \a x in \a space,
 * under the process model \a model with the call's \a args: \f$ f(x) \boxplus w \f$, where \f$ f \f$ is
 * the model's step and \f$ w \sim \mathcal{N}(0, Q) \f$ its process noise over \a dt.
 *
 * This is the discrete-time model that a filter over \a model assumes, the one whose noise the
 * blocks discretize as \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)"
 * describe, so a filter run on its draws, with measurements drawn from the true states, should
 * pass NeesMonitor's and NisMonitor's tests. simulate(space, x, model, rng) with a measurement
 * model draws those measurements.
 *
 * \code
 * auto rng = std::mt19937_64{42};
 * auto truth = Eigen::Vector4d{0, 0, 1, 0};
 * for (int k = 0; k < steps; ++k)
 * {
 *     truth = simulate(space, truth, model, rng, 0.1);
 *     filter.predict(0.1);
 *     filter.update(observe(methods::linearized(), gps, simulate(space, truth, gps, rng)));
 * }
 * \endcode
 *
 * \a dt is any TimeStep, converted to the state's scalar (see \ref Time), which must be a standard
 * floating-point type. \a args are the call's inputs as predict() takes them: a sure input, such
 * as a control, reaches the model as it is, and an uncertain one is drawn from its distribution,
 * the inputs in the order the model reads them and the process noise last, so that a seed fixes
 * the run within one standard library: the standard normals come from \c std::normal_distribution,
 * whose algorithm the standard leaves to the implementation, so one seed draws another run under
 * libstdc++ than under MSVC's library (decisions.md [random-draws]). A filter's uncertain input is
 * the noisy measurement of a true one, so give the simulation the true value as
 * <tt>sure<Tag>(value)</tt>.
 *
 * \sa NeesMonitor, FixedStepModel
 */
template <Space Space_, typename Derived, ProcessModel Model, std::uniform_random_bit_generator G, typename... Args>
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
simulate(
    Space_ const & space,
    Eigen::MatrixBase<Derived> const & x,
    Model const & model,
    G & rng,
    TimeStep<typename Derived::Scalar> auto const & dt,
    Args const &... args
)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(x.size() == valueDimension(space) && "the state does not fit the space");
    internal::assertPartitions(model, space);
    Scalar const step = internal::toScalarTime<Scalar>(dt);
    auto const call = internal::modelWith(std::cref(model), args...);
    auto const resolved = internal::resolveProcess(call, space);
    return internal::drawProcessNoise(
        space,
        internal::drawStep(resolved, x, step, rng),
        resolved.template noise<Scalar>(step),
        rng
    );
}

/** \ingroup Models
 * \returns a draw of the true state one step of the fixed-step model \a model after the true
 * state \a x in \a space, with the call's \a args: simulate(space, x, model, rng, dt, args...) over
 * the model's step, with its cached process noise.
 */
template <Space Space_, typename Derived, typename Model, std::uniform_random_bit_generator G, typename... Args>
    requires isFixedStep<Model>
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
simulate(Space_ const & space, Eigen::MatrixBase<Derived> const & x, Model const & model, G & rng, Args const &... args)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(x.size() == valueDimension(space) && "the state does not fit the space");
    internal::assertPartitions(model, space, "the model must fit the state space");
    auto const call = internal::modelWith(std::cref(model.model()), args...);
    auto const resolved = internal::resolveProcess(call, space);
    return internal::drawProcessNoise(
        space,
        internal::drawStep(resolved, x, Scalar(model.timeStep()), rng),
        model.noise().template cast<Scalar>(),
        rng
    );
}

/** \ingroup Models
 * \returns a draw of the true state one step of \a step, made by fixedStep(model, dt), after the
 * true state \a x in \a space, with the call's \a args: simulate(space, x, model, rng, dt, args...).
 */
template <Space Space_, typename Derived, typename Step, std::uniform_random_bit_generator G, typename... Args>
    requires isUnboundStep<Step>
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
simulate(Space_ const & space, Eigen::MatrixBase<Derived> const & x, Step const & step, G & rng, Args const &... args)
{
    return simulate(space, x, step.model(), rng, step.timeStep(), args...);
}

/** \ingroup Models
 * \returns a draw of the measurement that \a model makes of the true state \a x in \a space, with
 * the call's \a args as observe() takes them after the observation: \f$ h(x, v) \boxplus w \f$,
 * formed in the measurement space, for each uncertain input \f$ v \f$ drawn from its distribution
 * and \f$ w \f$ drawn from the model's additive noise, a value in the measurement space that
 * observe() turns into the measurement a filter takes.
 *
 * This is the measurement model that a filter given \a model assumes, so a filter run on its
 * draws from simulate()'s true states should pass NeesMonitor's and NisMonitor's tests. Each
 * distribution is drawn by drawSample() of its density, the inputs in the order the model reads
 * them and the noise last, in a fixed order from \a rng, so a seed fixes the run. A model known by
 * its likelihood alone has no function, and is drawn by its sample(g), <tt>g(in, x, rng)</tt>,
 * after the inputs; without one it fails to compile.
 *
 * \code
 * auto const gps = measurementModel(Eigen::Matrix2d::Identity(), reads(path<Position>), additiveNoise(noise));
 * filter.update(observe(methods::linearized(), gps, simulate(space, truth, gps, rng)));
 * \endcode
 *
 * \sa observe(), NeesMonitor
 */
template <Space Space_, typename Derived, MeasurementModel M, std::uniform_random_bit_generator G, typename... Args>
[[nodiscard]] auto
simulate(Space_ const & space, Eigen::MatrixBase<Derived> const & x, M const & model, G & rng, Args const &... args)
{
    using Scalar = typename Derived::Scalar;
    EMRYS_FILTER_ASSERT(x.size() == valueDimension(space) && "the state does not fit the space");
    auto const observed = internal::modelWith(std::cref(model), args...);
    auto const resolved = internal::resolveMeasurement(observed, space);
    using R = std::remove_cvref_t<decltype(resolved)>;
    static_assert(
        R::HasFunction || R::HasSample,
        "a measurement model known by its likelihood alone has no function to draw a measurement from; give it "
        "sample(g), a draw of the observation at a state"
    );
    // Every input drawn first, in the model's order, then the model's own draw, or the function at
    // the state and the draws and then the noise.
    auto const draws = std::apply(
        [&](auto const &... inputs) {
            return std::tuple{internal::drawMeasurementNoise(inputs, rng)...};
        },
        resolved.inputs()
    );
    if constexpr (!R::HasFunction)
    {
        return std::apply(
            [&](auto const &... drawn) {
                return resolved.sampleAtState(x, rng, drawn...);
            },
            draws
        );
    }
    else
    {
        auto const & measurementSpace = resolved.space();
        using Vector = typename Moments<typename R::Out, Scalar>::VectorType;
        Vector predicted = std::apply(
            [&](auto const &... drawn) {
                return internal::valueOf(resolved.evaluateAtState(x, drawn...)).eval();
            },
            draws
        );
        normalizeValues(measurementSpace, predicted);
        if constexpr (R::HasAdditiveNoise)
        {
            auto const noise = internal::drawMeasurementNoise(resolved.noise().distribution(), rng);
            return Vector{Emrys::plus(measurementSpace, predicted, noise.template cast<Scalar>())};
        }
        else
        {
            return predicted;
        }
    }
}

/** \ingroup Models
 * \returns simulate(space, x, model, rng, dt, inputs...) in the fixed-size space \a Space_,
 * given as a template argument: <tt>simulate<State>(truth, model, rng, dt)</tt>.
 */
template <
    FixedSizeSpace Space_,
    typename Derived,
    typename Model,
    std::uniform_random_bit_generator G,
    typename... Inputs>
    requires(Space<Space_> && !isFixedStep<Model> && !isUnboundStep<Model>)
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
simulate(
    Eigen::MatrixBase<Derived> const & x,
    Model const & model,
    G & rng,
    TimeStep<typename Derived::Scalar> auto const & dt,
    Inputs const &... inputs
)
{
    return simulate(Space_{}, x, model, rng, dt, inputs...);
}

/** \ingroup Models
 * \returns simulate(space, x, model, rng, inputs...) for the fixed-step model \a model, or
 * fixedStep()'s result, in the fixed-size space \a Space_, given as a template argument:
 * <tt>simulate<State>(truth, model, rng)</tt>.
 */
template <
    FixedSizeSpace Space_,
    typename Derived,
    typename Model,
    std::uniform_random_bit_generator G,
    typename... Inputs>
    requires(Space<Space_> && (isFixedStep<Model> || isUnboundStep<Model>))
[[nodiscard]] typename Moments<Space_, typename Derived::Scalar>::VectorType
simulate(Eigen::MatrixBase<Derived> const & x, Model const & model, G & rng, Inputs const &... inputs)
{
    return simulate(Space_{}, x, model, rng, inputs...);
}

/** \ingroup Models
 * \returns simulate(space, x, model, rng, args...) in the fixed-size space \a Space_, given as a
 * template argument: <tt>simulate<State>(truth, gps, rng)</tt>.
 */
template <
    FixedSizeSpace Space_,
    typename Derived,
    MeasurementModel M,
    std::uniform_random_bit_generator G,
    typename... Args>
    requires Space<Space_>
[[nodiscard]] auto
simulate(Eigen::MatrixBase<Derived> const & x, M const & model, G & rng, Args const &... args)
{
    return simulate(Space_{}, x, model, rng, args...);
}

/** \ingroup Models
 * \returns a draw of the value one step of \a chain after the value \a from, as a flat index: a
 * draw from <tt>chain.next(from)</tt>, the true mode's next value in a simulation of a switching
 * or hidden Markov model, as simulate() of a process model draws the true state's.
 *
 * \code
 * mode = simulate(mode, switching.chain(), rng);
 * \endcode
 *
 * It makes one uniform draw against the running sum of the row, as drawSample() of a Categorical
 * does, and allocates nothing.
 */
template <typename Space, typename Scalar, std::uniform_random_bit_generator G>
[[nodiscard]] Eigen::Index
simulate(Eigen::Index from, MarkovChain<Space, Scalar> const & chain, G & rng)
{
    EMRYS_FILTER_ASSERT(from >= 0 && from < chain.space().cardinality() && "the value is not in the chain's space");
    return internal::drawIndex<Scalar>(
        chain.space().cardinality(),
        [&](Eigen::Index to) {
            return chain.transition(from, to);
        },
        rng
    );
}

/** \ingroup Models
 * \returns a draw of the value an interval of \a dt, any TimeStep, after the value \a from under
 * the continuous-time chain \a chain: simulate() of its transition() over \a dt, which computes
 * the matrix exponential at each call.
 */
template <typename Space, typename Scalar, std::uniform_random_bit_generator G>
[[nodiscard]] Eigen::Index
simulate(
    Eigen::Index from,
    ContinuousTimeMarkovChain<Space, Scalar> const & chain,
    G & rng,
    TimeStep<Scalar> auto const & dt
)
{
    return simulate(from, chain.transition(dt), rng);
}

}  // namespace Emrys

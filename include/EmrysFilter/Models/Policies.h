#pragma once

/** \file
 * The policies a model is built from, and the marks on a step's arguments (decisions.md [model-policies],
 * [tag-addressing]): reads(), writes(), affineIn(), additiveNoise(), continuousNoise(), logLikelihood(), and
 * outputSpace(), each a call returning an object of its own type, which processModel() and measurementModel() take
 * after the function in any order; the internal PolicySet, which finds each in an argument pack; the placing of
 * additive noise in an output space; and sure(), which marks a known input of predict() or observe() by the tag of
 * its space. Read MeasurementModel.h next.
 */

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

namespace internal
{

/** One item of reads(): the block that the path \a Path (a TagPath) names, read under the name
 * \a Name, a tag, which is the path's last tag unless on<Name>() gave another.
 */
template <typename Path, typename Name>
struct ReadItem
{
    using PathType = Path;  ///< the path read, a TagPath
    using NameType = Name;  ///< the name it is read under, a tag

    /// whether the path has one tag, the only form that can name an input of the call
    static constexpr bool SingleTag = std::is_same_v<Path, TagPath<Name>>;
};

/** What an argument of reads() of type \a T denotes: nothing, for anything but a path value or an
 * on<Name>() of one.
 */
template <typename T>
struct ReadItemOf
{
    static constexpr bool valid = false;  ///< not an item
};

/** A path value: the path, read under its last tag. */
template <PathValue T>
struct ReadItemOf<T>
{
    static constexpr bool valid = true;                    ///< an item
    using type = ReadItem<PathOf<T>, PathTag<PathOf<T>>>;  ///< the item
};

/** on<Name>(path): the path, read under \a Name. */
template <PathValue X, typename Name>
struct ReadItemOf<OnPath<X, Name>>
{
    static constexpr bool valid = true;      ///< an item
    using type = ReadItem<PathOf<X>, Name>;  ///< the item
};

/** Whether the types \a Ts are pairwise distinct. */
template <typename... Ts>
consteval bool
distinctTypes()
{
    if constexpr (sizeof...(Ts) < 2)
    {
        return true;
    }
    else
    {
        return []<typename First, typename... Rest>(std::type_identity<First>, std::type_identity<Rest>...) {
            return (!std::is_same_v<First, Rest> && ...) && distinctTypes<Rest...>();
        }(std::type_identity<Ts>{}...);
    }
}

/** The library space that the tag \a T derives from, reconstructed from what it exposes: a Linear or
 * Periodic of its sizes, a Composite of its blocks, a Repeated of its block, UnitQuaternion for a tag
 * derived from it, and the type itself for a space written from scratch. Two tags with one library
 * space beneath them retract the same way, so one may stand for the other in a model's input.
 */
template <typename T>
struct LibraryBaseOf
{
    /** The type itself. */
    using type = T;
};

/// \cond INTERNAL_SPECIALIZATIONS
template <IsComposite T>
struct LibraryBaseOf<T>
{
    template <typename... Blocks>
    static Composite<Blocks...> *
    of(std::tuple<Blocks...> *);

    using type = std::remove_pointer_t<decltype(of(static_cast<CompositeBlocks<T> *>(nullptr)))>;
};

template <IsRepeated T>
struct LibraryBaseOf<T>
{
    using type = Repeated<RepeatedBlock<T>, T::MaxCount>;
};

template <IsUniformLeaf T>
    requires(!IsPeriodic<T>)
struct LibraryBaseOf<T>
{
    using type = Linear<T::ValueDimensionStep == 0 ? T::FixedValueDimension : Eigen::Dynamic, T::MaxValueDimension>;
};

template <IsPeriodic T>
struct LibraryBaseOf<T>
{
    template <PeriodicInterval Interval, int Dimension, int MaxDimension>
    static Periodic<Interval, Dimension, MaxDimension> *
    of(Periodic<Interval, Dimension, MaxDimension> const *);

    using type = std::remove_pointer_t<decltype(of(static_cast<T const *>(nullptr)))>;
};

/// \endcond

/** The library space beneath the tag \a T (LibraryBaseOf). */
template <typename T>
using LibraryBase = typename LibraryBaseOf<T>::type;

/** Whether the tags \a Name and \a Block have the same leaf beneath them, so that \a Name can stand
 * for \a Block in a model's input: one derives from the other, or both derive from one library space.
 * Two tags over a space written from scratch, which exposes nothing to rebuild it from, match only
 * where one derives from the other.
 */
template <typename Name, typename Block>
inline constexpr bool sameLeaf = std::derived_from<Name, Block> || std::derived_from<Block, Name>
                              || std::is_same_v<LibraryBase<Name>, LibraryBase<Block>>;

}  // namespace internal

/** \ingroup Models
 * \brief What a model reads, as reads() declares it: the items, each a path into the state or the tag
 * of an input of the call, under a name (decisions.md [model-policies]).
 *
 * The model's input space is the Composite of the items' names, in the order given, and the
 * function reads its input vector with <tt>block<Name>(in, x)</tt>. An item is a path value,
 * <tt>path<Ownship, Position></tt> or a bare tag's value, read under its last tag, or
 * <tt>on<Name>(path<...>)</tt>, read under \a Name, so that two items whose paths end in one tag
 * can be told apart. A path of one tag that the state lacks names an input of the call, given as an
 * uncertain input over that tag's space or as sure<Tag>().
 *
 * \tparam Items  the items, as internal::ReadItem
 *
 * \sa reads(), writes()
 */
template <typename... Items>
struct Reads
{
    static constexpr std::size_t Count = sizeof...(Items);  ///< the number of items

    /** The input space: the Composite of the names, in order. */
    using InputSpace = Composite<typename Items::NameType...>;

    /** The item at position \a I. */
    template <std::size_t I>
    using Item = std::tuple_element_t<I, std::tuple<Items...>>;
};

/** \ingroup Models
 * \returns the reads() policy of \a items: each a path value, <tt>path<Tags...></tt> or a tag's
 * value, read under its last tag, or <tt>on<Name>(path)</tt>, read under \a Name. Two items that
 * would share a name are refused, with a message that points at on<Name>().
 *
 * \code
 * reads(path<Position>, path<Velocity>, path<SpecificForce>)
 * reads(on<SensorPosition>(path<Ownship, Position>), path<Target, Position>, path<ScaleNoise>)
 * \endcode
 */
template <typename... Items>
[[nodiscard]] auto
reads(Items const &...)
{
    static_assert(
        (internal::ReadItemOf<std::remove_cvref_t<Items>>::valid && ...),
        "reads() takes paths, path<Tags...> or a tag's value, and on<Name>(path) for a path read under another "
        "name"
    );
    using Result = Reads<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type...>;
    static_assert(
        internal::distinctTypes<typename internal::ReadItemOf<std::remove_cvref_t<Items>>::type::NameType...>(),
        "two items of reads() share a name, their paths' last tag; read one under another name with on<Name>(path)"
    );
    return Result{};
}

/** \ingroup Models
 * \brief What a model writes, as writes() declares it: the blocks its function's result fills
 * (decisions.md [model-policies]). For a process model they are paths into the state, for a
 * measurement model blocks of the measurement space, and the output space is their Composite, in
 * the order given.
 *
 * \tparam Paths  the paths, each a TagPath
 *
 * \sa writes(), reads()
 */
template <typename... Paths>
struct Writes
{
    static constexpr std::size_t Count = sizeof...(Paths);  ///< the number of blocks written

    /** The output space: the Composite of the blocks, in order. */
    using OutputSpace = Composite<internal::PathTag<Paths>...>;

    /** The paths, as a TagPaths. */
    using PathList = TagPaths<Paths...>;

    /** The path at position \a I. */
    template <std::size_t I>
    using Path = std::tuple_element_t<I, std::tuple<Paths...>>;
};

namespace internal
{

/** What a composition or a placed model writes: the paths \a Paths_, a TagPaths, into the state,
 * under the names of the output space \a Out_, a Composite with a block per path in order. A block's
 * name is its path's last tag, as in writes(), except where a placement gives it one of its own,
 * so that two placements of one model write blocks the output tells apart.
 *
 * \tparam Paths_  the paths, a TagPaths
 * \tparam Out_    the output space
 */
template <typename Paths_, typename Out_>
struct WritesAs
{
    using PathList = Paths_;   ///< the paths, as a TagPaths
    using OutputSpace = Out_;  ///< the output space: the Composite of the blocks' names, in order
};

}  // namespace internal

/** \ingroup Models
 * \returns the writes() policy of \a paths, each a path value, distinct
 */
template <PathValue... Paths>
[[nodiscard]] auto
writes(Paths const &...)
{
    static_assert(sizeof...(Paths) > 0, "writes() names at least one block");
    static_assert(internal::distinctTypes<internal::PathOf<Paths>...>(), "writes() names each block once");
    return Writes<internal::PathOf<Paths>...>{};
}

/** \ingroup Models
 * \brief The input names a model's function is affine in, given the rest, as affineIn() declares
 * them (decisions.md [model-policies], [affine-coordinates]): a sigma-point method factors those
 * coordinates out and spreads its points over the others alone.
 *
 * \tparam Names  the names, each a tag of the model's reads()
 */
template <typename... Names>
struct AffineIn
{
    static constexpr std::size_t Count = sizeof...(Names);  ///< the number of names
};

/** \ingroup Models
 * \returns the affineIn() policy of \a names, each the one tag an item of reads() is read under, as
 * a path value of one tag
 */
template <PathValue... Names>
[[nodiscard]] auto
affineIn(Names const &...)
{
    static_assert(
        (std::is_same_v<internal::PathOf<Names>, TagPath<internal::PathTag<internal::PathOf<Names>>>> && ...),
        "affineIn() names an input of the model by the one tag it is read under in reads(), never by a path into the "
        "state"
    );
    static_assert(
        internal::distinctTypes<internal::PathTag<internal::PathOf<Names>>...>(),
        "affineIn() names each input once"
    );
    return AffineIn<internal::PathTag<internal::PathOf<Names>>...>{};
}

namespace internal
{

/** Whether \a T is a GaussianMixtureParams, so additive noise has several components. */
template <typename T>
inline constexpr bool isGaussianMixture = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Scalar, typename Space, typename Allocator>
inline constexpr bool isGaussianMixture<GaussianMixtureParams<Space, Scalar, Allocator>>{true};
/// \endcond

/** Whether \a T, a part of additiveNoise() as held, is a covariance alone: zero-mean %Gaussian
 * noise over the differences of what it covers, whose space the model's output gives.
 */
template <typename T>
inline constexpr bool isNoiseCovariance = MatrixExpression<T>;

/** Whether \a T places a covariance alone by tag path, <tt>on<Path>(R)</tt> or <tt>onEach<Path>(R)</tt>. */
template <typename T>
inline constexpr bool placesCovariance = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path>
inline constexpr bool placesCovariance<OnPath<X, Path>> = isNoiseCovariance<X>;

template <typename X, typename Path>
inline constexpr bool placesCovariance<OnEach<X, Path>> = isNoiseCovariance<X>;
/// \endcond

/** Whether \a T, a part of additiveNoise(), is placed by tag path with on() or onEach(): a
 * distribution or a covariance.
 */
template <typename T>
inline constexpr bool isPlacedNoise =
    isPlacedFactor<std::remove_cvref_t<T>> || placesCovariance<std::remove_cvref_t<T>>;

/** Whether the part \a T of additiveNoise() places a mixture, which a product of factors cannot
 * hold.
 */
template <typename T>
inline constexpr bool placesMixture = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename X, typename Path>
inline constexpr bool placesMixture<OnPath<X, Path>> = isGaussianMixture<std::remove_cvref_t<X>>;

template <typename X, typename Path>
inline constexpr bool placesMixture<OnEach<X, Path>> = isGaussianMixture<std::remove_cvref_t<X>>;
/// \endcond

}  // namespace internal

template <typename Axis, typename Part>
class ModeNoise;

/** \ingroup Models
 * \brief Noise added to a model's output, \f$ z = h(x) \boxplus v \f$, as additiveNoise() declares
 * it (decisions.md [additive-noise], [model-policies]): the function is transformed once, and the
 * noise only widens and shifts the result.
 *
 * One unplaced part covers the whole output space: a covariance matrix \f$ R \f$ over the output's
 * differences, zero-mean %Gaussian noise, as additive noise nearly always is; a GaussianParams or a
 * GaussianMixtureParams over them, such as a biased sensor's or a narrow inlier and a wide outlier
 * component; or a const reference to one that outlives the model. Parts placed by tag path,
 * <tt>on<Range>(r)</tt> and <tt>onEach<Offsets>(r)</tt>, each a covariance or a GaussianParams, are
 * independent Gaussians on blocks of the output, placed in its space instance when the model meets
 * a state, as the factors of a Product are (decisions.md [product-factors]); a placed mixture is
 * refused. A covariance has no space of its own, so a model whose output space would come from its
 * noise measures in a Linear sized by its function's result instead.
 *
 * \tparam Parts  the parts, as given
 *
 * \sa additiveNoise(), ContinuousNoise
 */
template <typename... Parts>
class AdditiveNoise
{
    static_assert(sizeof...(Parts) > 0, "additiveNoise() takes a distribution, or parts placed by tag path");
    static_assert(
        (internal::isPlacedNoise<Parts> && ...) || (sizeof...(Parts) == 1 && !(internal::isPlacedNoise<Parts> || ...)),
        "additiveNoise() with no placement covers the whole output space, so it takes one distribution alone; place "
        "each part with on<Path>() or onEach<Path>() to give several"
    );
    static_assert(
        (!internal::placesMixture<std::remove_cvref_t<Parts>> && ...),
        "a mixture is additive noise over the whole output space, unplaced; a placed part is one Gaussian"
    );

public:

    /// whether the parts are placed by tag path, and so become one Gaussian in the output space
    static constexpr bool Placed = (internal::isPlacedNoise<Parts> || ...);

    /** The one unplaced distribution's type, unqualified, or the covariance's. */
    using Distribution = std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Parts...>>>;

    /// whether the one unplaced part is a covariance alone, zero-mean noise with no space of its own
    static constexpr bool ZeroMean = !Placed && internal::isNoiseCovariance<Distribution>;

    /// whether the one unplaced part is a distribution, whose space a model's output may take
    static constexpr bool HasOwnSpace = !Placed && !ZeroMean;

    /** The noise of \a parts. */
    explicit AdditiveNoise(Parts... parts)
        : m_parts{std::forward<Parts>(parts)...}
    {}

    /** \returns the parts, as given */
    [[nodiscard]] std::tuple<Parts...> const &
    parts() const
    {
        return m_parts;
    }

    /** \returns the one unplaced distribution, or covariance, over the whole output space */
    [[nodiscard]] Distribution const &
    distribution() const &
        requires(!Placed)
    {
        return std::get<0>(m_parts);
    }

    /** \returns the one unplaced distribution of a temporary noise: moved out where it is held by
     * copy, the reference where it is held by one, so that a reference bound to
     * <tt>model.noise().distribution()</tt> lives as long as the result and never dangles
     */
    [[nodiscard]] std::tuple_element_t<0, std::tuple<Parts...>>
    distribution() &&
        requires(!Placed)
    {
        return std::get<0>(std::move(m_parts));
    }

    /** \returns this noise, a mixture, with its components as the values of the mode axis named by
     * the path \a Axis (a tag or a TagPath), in order (see ModeNoise)
     */
    template <typename Axis>
    [[nodiscard]] auto
    byMode() const &
    {
        checkByMode();
        return ModeNoise<Axis, std::tuple_element_t<0, std::tuple<Parts...>>>{*this};
    }

    /** byMode() const &, moving the mixture out of a temporary noise. */
    template <typename Axis>
    [[nodiscard]] auto
    byMode() &&
    {
        checkByMode();
        return ModeNoise<Axis, std::tuple_element_t<0, std::tuple<Parts...>>>{std::move(*this)};
    }

private:

    /** Checks that this noise is one mixture over the whole output, as byMode() requires. */
    static constexpr void
    checkByMode()
    {
        static_assert(
            !Placed && internal::isGaussianMixture<Distribution>,
            "byMode() takes additive noise that is one mixture over the whole output, with one component per value "
            "of the mode axis"
        );
    }

    std::tuple<Parts...> m_parts;  ///< the parts
};

/** \ingroup Models
 * \brief Additive mixture noise whose components are the values of a mode axis, as
 * <tt>additiveNoise(mixture).byMode<Axis>()</tt> declares it (decisions.md [imm]): the Markov
 * outlier model, in which a sensor's outlier state is a mode of the target that persists as the
 * mode chain says.
 *
 * A filter whose state has the mode axis \a Axis, such as an ImmFilter whose mode space has it,
 * updates each mode with the one component that the mode's value in \a Axis names, in the order of
 * the axis's values, and reweights the mode by that component's likelihood; the components'
 * weights are not used, since the chain gives the probabilities. A filter whose state has no mode
 * space, such as a KalmanFilter or a GaussianMixtureFilter, reads it as the AdditiveNoise it
 * derives from, a mixture whose weights are the outliers' rate at each measurement, independent of
 * the last: the memoryless outlier model. A filter whose state has a mode space without \a Axis
 * refuses it at compile time. So one measurement serves the outlier-robust filter and the IMM, and
 * an IMM always updates the axis the measurement names. The weights the memoryless reading uses
 * should match the chain's stationary probabilities on \a Axis, for the two readings to describe
 * one sensor.
 *
 * \tparam Axis  the path to the mode axis, a tag or a TagPath
 * \tparam Part  the mixture, as held
 *
 * \sa AdditiveNoise::byMode(), ImmFilter
 */
template <typename Axis, typename Part>
class ModeNoise : public AdditiveNoise<Part>
{
public:

    using ModeAxisType = Axis;  ///< the path to the mode axis whose values the components are

    /** \a noise, with its components as the values of the mode axis. */
    explicit ModeNoise(AdditiveNoise<Part> noise)
        : AdditiveNoise<Part>{std::move(noise)}
    {}
};

/** \ingroup Models
 * \returns the additiveNoise() policy of \a parts (see AdditiveNoise): one covariance, zero-mean
 * noise, or one distribution, either held by copy, or for <tt>std::cref(noise)</tt> by reference to
 * one that must outlive the model, or parts placed by tag path with on<Path>() and onEach<Path>()
 *
 * \code
 * additiveNoise(R)  // zero-mean, R over the output's differences
 * additiveNoise(on<Range>(r), on<Bearing>(b))
 * additiveNoise(onEach<Offsets>(r))
 * additiveNoise(GaussianParams<Linear<2>>{bias, R})
 * additiveNoise(std::cref(mixture))
 * \endcode
 */
template <typename... Parts>
[[nodiscard]] AdditiveNoise<typename internal::HeldOf<std::unwrap_ref_decay_t<Parts>>::type...>
additiveNoise(Parts &&... parts)
{
    return AdditiveNoise<typename internal::HeldOf<std::unwrap_ref_decay_t<Parts>>::type...>{
        std::forward<Parts>(parts)...
    };
}

/** \ingroup Models
 * \brief A process model's white-noise spectral density, as continuousNoise() declares it, placed
 * by tag path on blocks of the output and discretized over the step by the model (decisions.md
 * [continuous-time-noise], [model-policies]): \f$ Q_c \, \Delta t \f$ on each block's differences.
 *
 * Each part is <tt>on<Velocity>(q)</tt> or <tt>onEach<Fleet>(q)</tt>, with \f$ Q_c \f$ a scalar,
 * the density of every axis, or a matrix over the block's differences; one unplaced \f$ Q_c \f$
 * covers the whole output space.
 *
 * \tparam Parts  the parts, as given
 *
 * \sa continuousNoise(), AdditiveNoise
 */
template <typename... Parts>
class ContinuousNoise
{
    static_assert(sizeof...(Parts) > 0, "continuousNoise() takes a spectral density, or parts placed by tag path");
    static_assert(
        (internal::isPlacedByPath<std::remove_cvref_t<Parts>> && ...)
            || (sizeof...(Parts) == 1 && !(internal::isPlacedByPath<std::remove_cvref_t<Parts>> || ...)),
        "continuousNoise() with no placement covers the whole output space, so it takes one spectral density alone; "
        "place each part with on<Path>() or onEach<Path>() to give several"
    );

public:

    /// whether the parts are placed by tag path
    static constexpr bool Placed = (internal::isPlacedByPath<std::remove_cvref_t<Parts>> || ...);

    /** The noise of \a parts. */
    explicit ContinuousNoise(Parts... parts)
        : m_parts{std::move(parts)...}
    {}

    /** \returns the parts, as given */
    [[nodiscard]] std::tuple<Parts...> const &
    parts() const
    {
        return m_parts;
    }

private:

    std::tuple<Parts...> m_parts;  ///< the parts
};

/** \ingroup Models
 * \returns the continuousNoise() policy of \a parts (see ContinuousNoise): spectral densities placed
 * by tag path, <tt>continuousNoise(on<Velocity>(q))</tt>, or one for the whole output
 */
template <typename... Parts>
[[nodiscard]] ContinuousNoise<std::remove_cvref_t<Parts>...>
continuousNoise(Parts &&... parts)
{
    return ContinuousNoise<std::remove_cvref_t<Parts>...>{std::forward<Parts>(parts)...};
}

namespace internal
{

/** A built-in process model's noise over a step, as its own closed form gives it: a function
 * <tt>g(out, dt)</tt> of the output space instance and the step, returning \f$ Q \f$ over the
 * output's differences, where continuousNoise() would give only \f$ Q_c \, \Delta t \f$. Internal:
 * the built-in models carry one, and processModel() builds none.
 *
 * \tparam G  the function
 */
template <typename G>
struct DiscreteNoise
{
    G function;  ///< \f$ Q \f$ over the step, as <tt>g(out, dt)</tt>
};

/** Whether \a T is a model, a measurement or a process model: what observe(), predict(), and the
 * calls that read a model with arguments take. A model type declares <tt>static constexpr bool
 * IsModel = true</tt>.
 */
template <typename T>
inline constexpr bool isModel = requires { requires std::remove_cvref_t<T>::IsModel; };

}  // namespace internal

/** \ingroup Models
 * \brief A measurement model's own \f$ \log p(z \mid x) \f$ at a state point, as logLikelihood()
 * declares it, for the filters that weight points (decisions.md [likelihood-measurements],
 * [model-policies]): a function <tt>g(in, x, z)</tt> of the model's input space instance, its
 * input vector, and the observation.
 *
 * \tparam G  the function
 *
 * \sa logLikelihood(G)
 */
template <typename G>
struct LogLikelihood
{
    G function;  ///< \f$ \log p(z \mid x) \f$ as <tt>g(in, x, z)</tt>
};

/** \ingroup Models
 * \returns the logLikelihood() policy of \a function, <tt>g(in, x, z)</tt> (see LogLikelihood):
 * what a particle or grid filter weights by when the model's noise is not additive, or is not
 * %Gaussian
 */
template <typename G>
    requires(!MatrixExpression<G> && !Space<std::remove_cvref_t<G>>)
[[nodiscard]] LogLikelihood<std::remove_cvref_t<G>>
logLikelihood(G && function)
{
    return {std::forward<G>(function)};
}

/** \ingroup Models
 * \brief A draw of the observation that a measurement model known by its likelihood alone makes of
 * a state, as sample() declares it, for simulate() (decisions.md [likelihood-measurements],
 * [model-policies]): a function <tt>g(in, x, rng)</tt> of the model's input space instance, its
 * input vector, and the caller's generator, returning the observation.
 *
 * \tparam G  the function
 *
 * \sa sample(G)
 */
template <typename G>
struct Sample
{
    G function;  ///< a draw of \f$ z \sim p(z \mid x) \f$ as <tt>g(in, x, rng)</tt>
};

/** \ingroup Models
 * \returns the sample() policy of \a function, <tt>g(in, x, rng)</tt> (see Sample): what simulate()
 * draws an observation by for a model known by its likelihood alone, whose logLikelihood() it must
 * agree with
 */
template <typename G>
    requires(!MatrixExpression<G> && !Space<std::remove_cvref_t<G>>)
[[nodiscard]] Sample<std::remove_cvref_t<G>>
sample(G && function)
{
    return {std::forward<G>(function)};
}

/** \ingroup Models
 * \brief A model's output space given by its instance, or computed from the state's space, as
 * outputSpace() declares it (decisions.md [measurement-space-functions], [model-policies]): for an
 * output with a resizable block, whose size the type does not fix.
 *
 * \tparam X  a Space instance, or a function from the state's space instance to the output's
 *
 * \sa outputSpace(), CountFrom
 */
template <typename X>
struct OutputSpace
{
    /// whether the output space is a function of the state's space
    static constexpr bool Computed = !Space<X>;

    X value;  ///< the instance, or the function

    /** \returns the output space in the state's space \a stateSpace: the instance, or the function's
     * result there
     */
    template <typename StateSpace>
    [[nodiscard]] auto
    in([[maybe_unused]] StateSpace const & stateSpace) const
    {
        if constexpr (Computed)
        {
            static_assert(
                std::invocable<X const &, StateSpace const &>,
                "a model's outputSpace() function takes the state's space instance"
            );
            using Result = std::remove_cvref_t<std::invoke_result_t<X const &, StateSpace const &>>;
            static_assert(Space<Result>, "a model's outputSpace() function returns the output space instance");
            return value(stateSpace);
        }
        else
        {
            return value;
        }
    }
};

/** \ingroup Models
 * \returns the outputSpace() policy of \a x (see OutputSpace): the output space instance, or a
 * function of the state's space instance that computes it, such as CountFrom
 */
template <typename X>
    requires(!MatrixExpression<X>)
[[nodiscard]] OutputSpace<std::remove_cvref_t<X>>
outputSpace(X && x)
{
    return {std::forward<X>(x)};
}

namespace internal
{

/** The kinds of policy a model takes. */
enum class PolicyKind
{
    reads,
    writes,
    affineIn,
    additiveNoise,
    continuousNoise,
    logLikelihood,
    sample,
    outputSpace,
    discreteNoise  ///< internal: a built-in process model's closed-form noise over the step
};

/** The kind of the policy type \a T, or nothing for a type that is no policy. */
template <typename T>
inline constexpr std::optional<PolicyKind> policyKind = std::nullopt;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Items>
inline constexpr std::optional<PolicyKind> policyKind<Reads<Items...>>{PolicyKind::reads};

template <typename... Paths>
inline constexpr std::optional<PolicyKind> policyKind<Writes<Paths...>>{PolicyKind::writes};

template <typename Paths, typename Out>
inline constexpr std::optional<PolicyKind> policyKind<WritesAs<Paths, Out>>{PolicyKind::writes};

template <typename... Names>
inline constexpr std::optional<PolicyKind> policyKind<AffineIn<Names...>>{PolicyKind::affineIn};

template <typename... Parts>
inline constexpr std::optional<PolicyKind> policyKind<AdditiveNoise<Parts...>>{PolicyKind::additiveNoise};

template <typename Axis, typename Part>
inline constexpr std::optional<PolicyKind> policyKind<ModeNoise<Axis, Part>>{PolicyKind::additiveNoise};

template <typename... Parts>
inline constexpr std::optional<PolicyKind> policyKind<ContinuousNoise<Parts...>>{PolicyKind::continuousNoise};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<LogLikelihood<G>>{PolicyKind::logLikelihood};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<Sample<G>>{PolicyKind::sample};

template <typename X>
inline constexpr std::optional<PolicyKind> policyKind<OutputSpace<X>>{PolicyKind::outputSpace};

template <typename G>
inline constexpr std::optional<PolicyKind> policyKind<DiscreteNoise<G>>{PolicyKind::discreteNoise};
/// \endcond

/** Whether \a T is a policy a model takes. */
template <typename T>
concept ModelPolicy = policyKind<std::remove_cvref_t<T>>.has_value();

/** Whether \a T is an AdditiveNoise. */
template <typename T>
inline constexpr bool isAdditiveNoise = policyKind<std::remove_cvref_t<T>> == PolicyKind::additiveNoise;

/** The path to the mode axis whose values the components of the additive noise \a Noise are, as
 * \c type: a ModeNoise's axis, or \c void for other noise.
 */
template <typename Noise>
struct NoiseModeAxisOf
{
    using type = void;  ///< no mode axis
};

/// \cond INTERNAL_SPECIALIZATIONS
template <typename Axis, typename Part>
struct NoiseModeAxisOf<ModeNoise<Axis, Part>>
{
    using type = Axis;  ///< the noise's mode axis
};

/// \endcond

/** The path to the mode axis whose values the components of the additive noise \a Noise are, or
 * \c void (see NoiseModeAxisOf).
 */
template <typename Noise>
using NoiseModeAxis = typename NoiseModeAxisOf<Noise>::type;

/** What holds a policy's place in a PolicySet that lacks it. */
struct NoPolicy
{};

/** The policies \a Policies given to a model, each at most once, found by kind. */
template <typename... Policies>
struct PolicySet
{
    static_assert(
        (ModelPolicy<Policies> && ...),
        "a model takes its function and then its policies, in any order: reads(), writes(), affineIn(), "
        "additiveNoise(), continuousNoise(), logLikelihood(), and outputSpace()"
    );

    /** The kinds given, in order. */
    static constexpr std::array<PolicyKind, sizeof...(Policies)> kinds{*policyKind<std::remove_cvref_t<Policies>>...};

    /** \returns the position of the policy of kind \a kind, or the number of policies if none is */
    [[nodiscard]] static consteval std::size_t
    position(PolicyKind kind)
    {
        for (std::size_t i = 0; i < kinds.size(); ++i)
        {
            if (kinds[i] == kind)
            {
                return i;
            }
        }
        return kinds.size();
    }

    /** \returns whether every kind is given at most once */
    [[nodiscard]] static consteval bool
    distinct()
    {
        for (std::size_t i = 0; i < kinds.size(); ++i)
        {
            for (std::size_t j = i + 1; j < kinds.size(); ++j)
            {
                if (kinds[i] == kinds[j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    static_assert(distinct(), "a model takes each policy at most once");

    /** Whether the policy of kind \a K is given. */
    template <PolicyKind K>
    static constexpr bool has = position(K) < sizeof...(Policies);

    /** The policy of kind \a K, or NoPolicy. */
    template <PolicyKind K>
    using Type = std::conditional_t<
        has<K>,
        std::tuple_element_t<has<K> ? position(K) : 0, std::tuple<Policies..., NoPolicy>>,
        NoPolicy>;

    /** \returns the policy of kind \a K in \a policies, a tuple of the pack, which must hold it */
    template <PolicyKind K, typename Tuple>
    [[nodiscard]] static auto const &
    get(Tuple const & policies)
    {
        static_assert(has<K>, "the model was not given this policy");
        return std::get<has<K> ? position(K) : 0>(policies);
    }
};

/** \returns the placed part \a part of additive noise as a factor of a Product over \a space:
 * itself, a distribution placed by tag path
 */
template <typename Space, typename Part>
[[nodiscard]] Part const &
noiseFactor(Space const &, Part const & part)
{
    return part;
}

/** The zero mean of a covariance of type \a X, as a column vector. */
template <typename X>
using ZeroMeanOf =
    Eigen::Matrix<typename X::Scalar, X::RowsAtCompileTime, 1, Eigen::ColMajor, X::MaxRowsAtCompileTime, 1>;

/** \returns the covariance \a part placed on a block as a zero-mean %Gaussian in the block's space */
template <typename Space, MatrixExpression X, typename Path>
[[nodiscard]] auto
noiseFactor(Space const & space, OnPath<X, Path> const & part)
{
    auto block = pathSubspace<Path>(space);
    using Noise = GaussianParams<decltype(block), typename X::Scalar>;
    return OnPath<Noise, Path>{Noise{std::move(block), ZeroMeanOf<X>::Zero(part.value.rows()), part.value}};
}

/** \returns the covariance \a part placed on each repetition of a block as a zero-mean %Gaussian in
 * one repetition's space
 */
template <typename Space, MatrixExpression X, typename Path>
[[nodiscard]] auto
noiseFactor(Space const &, OnEach<X, Path> const & part)
{
    using Block = typename EachBlockOf<Space, Path>::type;
    if constexpr (requires { typename Block::BlockType; })
    {
        using Noise = GaussianParams<typename Block::BlockType, typename X::Scalar>;
        return OnEach<Noise, Path>{Noise{ZeroMeanOf<X>::Zero(part.value.rows()), part.value}};
    }
    else
    {
        static_assert(
            std::is_void_v<X>,
            "onEach<Path>(R) in additiveNoise() places R on each repetition of a repeated block"
        );
    }
}

/** \returns the additive noise \a noise placed in \a space, the output space instance, as one
 * AdditiveNoise over the whole space: the unplaced distribution itself, by reference; a covariance
 * alone as a zero-mean %Gaussian over the space's differences in a Linear space; or the Product of
 * the placed parts, each a distribution or a zero-mean %Gaussian of a covariance in its block's
 * space, as one %Gaussian over the space's differences in a Linear space
 */
template <typename Space, typename... Parts>
[[nodiscard]] auto
placeNoise(Space const & space, AdditiveNoise<Parts...> const & noise)
{
    if constexpr (AdditiveNoise<Parts...>::ZeroMean)
    {
        auto const & cov = noise.distribution();
        using Cov = typename AdditiveNoise<Parts...>::Distribution;
        using Differences = DifferenceSpace<Space>;
        constexpr int Rows = valueDimensionAtCompileTime<Differences>;
        static_assert(
            Rows == Eigen::Dynamic || Cov::RowsAtCompileTime == Eigen::Dynamic || Cov::RowsAtCompileTime == Rows,
            "additiveNoise(R) is over the output's differences, so R has a row for each"
        );
        EMRYS_FILTER_ASSERT(
            cov.rows() == differenceDimension(space) && cov.cols() == cov.rows()
            && "additiveNoise(R) is over the output's differences, so R is square with a row for each"
        );
        using Noise = GaussianParams<Linear<Rows, maxValueDimensionAtCompileTime<Differences>>, typename Cov::Scalar>;
        return AdditiveNoise<Noise>{Noise{ZeroMeanOf<Cov>::Zero(cov.rows()), cov}};
    }
    else if constexpr (AdditiveNoise<Parts...>::Placed)
    {
        auto const product = std::apply(
            [&](auto const &... parts) {
                return Product{space, noiseFactor(space, parts)...};
            },
            noise.parts()
        );
        using Scalar = typename decltype(product)::Scalar;
        using Noise =
            GaussianParams<Linear<valueDimensionAtCompileTime<Space>, maxValueDimensionAtCompileTime<Space>>, Scalar>;
        auto const moments = product.moments();
        if constexpr (isExpected<std::remove_cvref_t<decltype(moments)>>)
        {
            EMRYS_FILTER_ASSERT(moments && "the noise's parts must have moments");
            return AdditiveNoise<Noise>{Noise{moments->mean(), moments->covarianceLower()}};
        }
        else
        {
            return AdditiveNoise<Noise>{Noise{moments.mean(), moments.covarianceLower()}};
        }
    }
    else
    {
        return AdditiveNoise<typename AdditiveNoise<Parts...>::Distribution const &>{noise.distribution()};
    }
}

/** Checks that each item of \a R, a Reads, that names a block of the state space \a Space is read
 * under a name with the same leaf as the block (sameLeaf), so that the model's input retracts as the
 * state does there. An item the state lacks names an input of the call, whose own space is its name's.
 *
 * \returns true, when it compiles
 */
template <typename R, typename Space>
[[nodiscard]] consteval bool
checkReadNames()
{
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        auto const check = []<typename Item>(std::type_identity<Item>) {
            using Path = typename Item::PathType;
            using Name = typename Item::NameType;
            constexpr bool inState = [] {
                return []<typename First, typename... Rest>(TagPath<First, Rest...> *) {
                    return countTag<First, Space>() > 0;
                }(static_cast<Path *>(nullptr));
            }();
            if constexpr (inState)
            {
                static_assert(
                    sameLeaf<Name, PathTag<Path>>,
                    "a name given by on<Name>() in reads() must have the same leaf space beneath it as the block it "
                    "names, such as Linear<2> for a Position : Linear<2>"
                );
            }
            return true;
        };
        return (check(std::type_identity<typename R::template Item<I>>{}) && ...);
    }(std::make_index_sequence<R::Count>{});
}

}  // namespace internal

/** \ingroup Models
 * \brief A sure input of predict() or observe(): a known value in the tagged space \a S, as sure()
 * marks it (decisions.md [prediction-inputs], [tag-addressing]). It names the item of the model's
 * reads() that the state lacks and has the tag of \a S, and reaches the function as a point, with no
 * spread.
 *
 * \tparam S       the input's space, a tag
 * \tparam Vector  the value's plain vector type
 */
template <Space S, typename Vector>
struct Sure
{
    /** the input's space, a tag */
    using SpaceType = S;
    /** the value's type */
    using VectorType = Vector;

    EMRYS_FILTER_NO_UNIQUE_ADDRESS S space;  ///< the space, with its sizes
    Vector value;                            ///< the value, a point of the space
};

/** \ingroup Models
 * \returns \a value, a vector, as the sure input in the fixed-size tagged space \a S:
 * <tt>sure<Gravity>(g)</tt>. Its size is checked against the space, at compile time where the
 * vector's is fixed.
 */
template <FixedSizeSpace S, MatrixExpression V>
    requires Space<S>
[[nodiscard]] auto
sure(V && value)
{
    using Plain = typename internal::CrtpDerived<V>::PlainObject;
    static_assert(!internal::IsUntagged<S>, "a sure input is named by a tag, a type derived from a space");
    static_assert(Plain::ColsAtCompileTime == 1, "a sure input is a column vector");
    static_assert(
        Plain::RowsAtCompileTime == Eigen::Dynamic || Plain::RowsAtCompileTime == valueDimensionAtCompileTime<S>,
        "a sure input's rows do not match its space's dimension at compile time"
    );
    EMRYS_FILTER_ASSERT(value.rows() == valueDimension(S{}) && "a sure input must be a value in its space");
    return Sure<S, Plain>{S{}, Plain{std::forward<V>(value)}};
}

/** \ingroup Models
 * \returns \a value as the sure input in \a space, a tagged space instance, for a resizable space,
 * whose size is checked against the instance
 */
template <Space S, MatrixExpression V>
[[nodiscard]] auto
sure(S space, V && value)
{
    using Plain = typename internal::CrtpDerived<V>::PlainObject;
    static_assert(!internal::IsUntagged<S>, "a sure input is named by a tag, a type derived from a space");
    static_assert(Plain::ColsAtCompileTime == 1, "a sure input is a column vector");
    EMRYS_FILTER_ASSERT(value.rows() == valueDimension(space) && "a sure input must be a value in its space");
    return Sure<S, Plain>{std::move(space), Plain{std::forward<V>(value)}};
}

namespace internal
{

/** Whether \a T is a Sure. */
template <typename T>
inline constexpr bool isSure = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename S, typename Vector>
inline constexpr bool isSure<Sure<S, Vector>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Models
 * \brief The blocks of the state that a Kalman update considers rather than estimates, as consider()
 * names them (decisions.md [consider-states]): the Schmidt-Kalman update of
 * \ref Schmidt1966 "Schmidt (1966)". The update keeps their mean and covariance block as the prior's,
 * and conditions the rest of the state, and its covariance with them, as the full update would.
 * <tt>Consider<></tt>, naming no block, is the full update, what an observation without a consider()
 * argument does.
 *
 * \tparam Paths  the blocks considered, each a TagPath looked up in the state's space
 */
template <typename... Paths>
struct Consider
{
    static constexpr std::size_t Count = sizeof...(Paths);  ///< the number of blocks considered
};

/** \ingroup Models
 * \returns the blocks named by \a paths, each a path value such as <tt>path<GyroBias></tt>, as the
 * argument of observe() that considers them rather than estimating them:
 * <tt>observe(methods::linearized(), model, z, consider(path<GyroBias>))</tt> (decisions.md
 * [consider-states]). Each path names a block of the state, which need not be one the model reads,
 * and no block lies in another.
 */
template <PathValue... Paths>
[[nodiscard]] constexpr Consider<internal::PathOf<Paths>...>
consider(Paths const &...)
{
    static_assert(sizeof...(Paths) > 0, "consider() names at least one block of the state");
    static_assert(internal::distinctTypes<internal::PathOf<Paths>...>(), "consider() names each block once");
    return {};
}

namespace internal
{

/** Whether \a T is a Consider. */
template <typename T>
inline constexpr bool isConsider = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Paths>
inline constexpr bool isConsider<Consider<Paths...>>{true};
/// \endcond

}  // namespace internal

}  // namespace Emrys

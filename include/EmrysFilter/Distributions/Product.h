#pragma once

/** \file
 * Product: the product of independent densities, each on its own block of a space, built
 * with on<Path>() or onEach<Path>() or from Factors (decisions.md [product-factors]), with marginals
 * and conditionals that work factor by factor, and its draws; EachRepetition; the HasLogDensity and
 * ProductFactor concepts. Read Categorical.h next.
 */

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/SampleSet.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/Subspace.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \ingroup Distributions
 * A density: a type that evaluates its log-density at a value in its space, such as a
 * Gaussian, a KernelDensity, or a Uniform.
 */
template <typename T>
concept HasLogDensity = requires(
    T const & density,
    Eigen::Matrix<
        typename T::Scalar,
        T::ValueDimensionAtCompileTime,
        1,
        Eigen::ColMajor,
        T::MaxValueDimensionAtCompileTime,
        1> const & x
) {
    typename T::SpaceType;
    { Emrys::valueDimension(density.space()) } -> std::convertible_to<Eigen::Index>;
    { density.logDensity(x) } -> std::convertible_to<typename T::Scalar>;
};

namespace internal
{

/** Parameters whose toDensity() checks them once and then hands out cheap density views, such
 * as a GaussianParams, a GaussianMixtureParams, or another form of the %Gaussian.
 */
template <typename T>
concept CheckedParams = requires(T const & params) {
    { params.toDensity().has_value() } -> std::convertible_to<bool>;
    requires HasLogDensity<typename decltype(params.toDensity())::value_type>;
};

/** Whether \a T has moments(), plain or in a std::expected. */
template <typename T>
concept HasMoments = requires(T const & distribution) { distribution.moments(); };

}  // namespace internal

/** \ingroup Distributions
 * What a Product takes as a factor: a density (HasLogDensity), or parameters that toDensity()
 * checks, such as a GaussianParams, which the product holds and checks itself. Either way the
 * density must have moments(), from which the product's are made.
 */
template <typename T>
concept ProductFactor = (HasLogDensity<T> && internal::HasMoments<T>)
                     || (internal::CheckedParams<T>
                         && internal::HasMoments<typename decltype(std::declval<T const &>().toDensity())::value_type>);

namespace internal
{

/** Whether a factor in the space \a Own fits a block of the space \a Block: the same type, a tag
 * derived from it or it from a tag, or Linear in both.
 */
template <typename Own, typename Block>
inline constexpr bool factorFits = std::is_same_v<Own, Block> || std::derived_from<Block, Own>
                                || std::derived_from<Own, Block> || (isLinear<Own> && isLinear<Block>);

/** How a Product places a factor given as \a Placed: on() or onEach() of a factor, or a FactorAt.
 * Each says the factor's type in a product over a space, the factor as the product holds it, the
 * coordinates it covers, and whether it fits them. Anything else has none of these.
 */
template <typename Placed>
struct PlacedFactor
{};

/** A factor on the subspace named by the path \a Path. */
template <ProductFactor F, typename Path>
struct PlacedFactor<OnPath<F, Path>>
{
    /** The factor's type in a product over \a Space: the factor's own. */
    template <typename Space>
    using TypeIn = F;

    /** \returns the factor of \a placed, as a product over \a Space holds it */
    template <typename Space>
    [[nodiscard]] static F
    factorIn(OnPath<F, Path> && placed, Space const &)
    {
        return std::move(placed.value);
    }

    /** \returns the coordinates of \a space that \a Path names, as an index set */
    template <typename Space>
    [[nodiscard]] static auto
    indicesIn(OnPath<F, Path> const &, Space const & space)
    {
        return asIndexSet(idx<Path>(space));
    }

    /** Whether the factor's space is the space of the block \a Path names in \a Space. */
    template <typename Space>
    static constexpr bool fitsIn = factorFits<typename F::SpaceType, PathSubspaceType<Space, Path>>;
};

/** A factor for a Product on the coordinates \a indices, which it is known to fit: a
 * product places its own factors this way when it builds a marginal or conditional.
 */
template <ProductFactor F, typename Indices>
struct FactorAt
{
    F factor;         ///< the factor
    Indices indices;  ///< the coordinates it covers
};

/** A factor on coordinates it is known to fit. */
template <ProductFactor F, typename Indices>
struct PlacedFactor<FactorAt<F, Indices>>
{
    /** The factor's type in a product over \a Space: the factor's own. */
    template <typename Space>
    using TypeIn = F;

    /** \returns the factor of \a placed, as a product over \a Space holds it */
    template <typename Space>
    [[nodiscard]] static F
    factorIn(FactorAt<F, Indices> && placed, Space const &)
    {
        return std::move(placed.factor);
    }

    /** \returns the stored coordinates, as an index set */
    template <typename Space>
    [[nodiscard]] static auto
    indicesIn(FactorAt<F, Indices> const & placed, Space const &)
    {
        return asIndexSet(placed.indices);
    }

    /** Always true: the product placed it on coordinates it fits. */
    template <typename Space>
    static constexpr bool fitsIn = true;
};

/** \returns \a factor on the coordinates \a indices, which it is known to fit. */
template <typename Indices, ProductFactor F>
[[nodiscard]] FactorAt<F, Indices>
placedAt(Indices indices, F factor)
{
    return {std::move(factor), std::move(indices)};
}

/** \returns the density of \a factor: itself, or a view of checked parameters */
template <typename F>
[[nodiscard]] decltype(auto)
densityOf(F const & factor)
{
    if constexpr (HasLogDensity<F>)
    {
        return (factor);
    }
    else
    {
        auto const density = factor.toDensity();
        auto view = *density;
        return view;
    }
}

/** Whether the moments() of the density of \a F returns a std::expected. */
template <typename F>
inline constexpr bool momentsMayFail =
    isExpected<std::remove_cvref_t<decltype(densityOf(std::declval<F const &>()).moments())>>;

/** A list of types, for passing a tag path's tags around as one argument. */
template <typename... T>
struct TypeList
{};

/** The types \a T after the first \a N. */
template <std::size_t N, typename... T>
struct DropFrontImpl
{
    /** The remaining types. */
    using type = TypeList<T...>;
};

/** Drops \a First and recurses. */
template <std::size_t N, typename First, typename... Rest>
    requires(N > 0)
struct DropFrontImpl<N, First, Rest...> : DropFrontImpl<N - 1, Rest...>
{};

/** \a T without its first \a N types, as a TypeList. */
template <std::size_t N, typename... T>
using DropFront = typename DropFrontImpl<N, T...>::type;

/** The first \a N of the types \a T, accumulated in \a Done. */
template <std::size_t N, typename Done, typename... T>
struct TakeFrontImpl
{
    /** The types taken so far. */
    using type = Done;
};

/** Takes \a First and recurses. */
template <std::size_t N, typename... Done, typename First, typename... Rest>
    requires(N > 0)
struct TakeFrontImpl<N, TypeList<Done...>, First, Rest...> : TakeFrontImpl<N - 1, TypeList<Done..., First>, Rest...>
{};

/** The first \a N of the types \a T. */
template <std::size_t N, typename... T>
using TakeFront = typename TakeFrontImpl<N, TypeList<>, T...>::type;

/** \returns the position within \a target of each entry of \a indices, as an index vector of
 * the given size; every entry must be in \a target
 */
template <int Size, int MaxSize, typename Indices, typename Target>
[[nodiscard]] IndexVector<Size, MaxSize>
positionsIn(Indices const & indices, Target const & target)
{
    auto result = IndexVector<Size, MaxSize>{};
    result.resize(Eigen::Index(indices.size()));
    for (Eigen::Index i = 0; i < result.size(); ++i)
    {
        auto const found = std::ranges::find(target, indices[i]);
        EMRYS_FILTER_ASSERT(found != std::ranges::end(target) && "a factor's coordinate is missing from the result");
        result(i) = Eigen::Index(found - std::ranges::begin(target));
    }
    return result;
}

/** The value inside a factor's conditional, unwrapped from a std::expected if need be. */
template <typename R>
struct ConditionalValueOf
{
    /** The value: \a R itself. */
    using type = R;
};

/** The value inside a std::expected. */
template <typename T, typename E>
struct ConditionalValueOf<std::expected<T, E>>
{
    /** The expected value. */
    using type = T;
};

/** The value inside a factor's conditional returning \a R, unwrapped from a std::expected if need
 * be.
 */
template <typename R>
using ConditionalValue = typename ConditionalValueOf<R>::type;

/** The error a factor's conditional can report: from its std::expected, or from checking the
 * parameters it returns; \c void if it cannot fail.
 */
template <typename R>
struct ConditionalErrorOf
{
    /** No error: the conditional cannot fail. */
    using type = void;
};

/** The error of a std::expected. */
template <typename T, typename E>
struct ConditionalErrorOf<std::expected<T, E>>
{
    /** The expected's error type. */
    using type = E;
};

/** The error of checking the parameters that a conditional returns. */
template <CheckedParams R>
struct ConditionalErrorOf<R>
{
    /** The parameters' toDensity() error type. */
    using type = typename decltype(std::declval<R const &>().toDensity())::error_type;
};

/** The error a factor's conditional returning \a R can report; \c void if it cannot fail. */
template <typename R>
using ConditionalError = typename ConditionalErrorOf<R>::type;

}  // namespace internal

template <Space Space_, RealScalar Scalar_, ProductFactor... Factors_>
class Product;

namespace internal
{

/** The Product in \a Space of \a Factors_ with factor \a K replaced by \a Replacement. */
template <
    typename Scalar,
    typename Space,
    std::size_t K,
    typename Replacement,
    typename Factors_,
    typename Sequence = std::make_index_sequence<std::tuple_size_v<Factors_>>>
struct ReplacedProductOf;

/** Substitutes \a Replacement at position \a K of \a Factors_. */
template <typename Scalar, typename Space, std::size_t K, typename Replacement, typename... Factors_, std::size_t... J>
struct ReplacedProductOf<Scalar, Space, K, Replacement, std::tuple<Factors_...>, std::index_sequence<J...>>
{
    /** The product with the replacement. */
    using type = Product<Space, Scalar, std::conditional_t<J == K, Replacement, Factors_>...>;
};

/** The Product in \a Space of \a Factors_ with factor \a K replaced by \a Replacement. */
template <typename Scalar, typename Space, std::size_t K, typename Replacement, typename Factors_>
using ReplacedProduct = typename ReplacedProductOf<Scalar, Space, K, Replacement, Factors_>::type;

}  // namespace internal

/** \ingroup Distributions
 * \brief One factor on every repetition of a repeated block: the product of independent copies
 * of a density, one per repetition, as a density over the whole block (decisions.md [product-factors]).
 *
 * onEach<Path>() makes one when a Product is placed, from the block's space there, so its count
 * is the space instance's: a %Gaussian factor with covariance \f$ R \f$ gives the block the
 * covariance \f$ I \otimes R \f$. The factor is a density or parameters (see ProductFactor), in
 * the space of the repeated block's one repetition, and its parameters are checked once, here.
 *
 * \tparam F           the factor's type, over one repetition
 * \tparam BlockSpace  the repeated block's space, a Repeated or a tag derived from one
 *
 * \sa onEach(), class Product
 */
template <ProductFactor F, Space BlockSpace>
class EachRepetition
{
    static_assert(
        requires { typename BlockSpace::BlockType; } && BlockSpace::FixedValueDimension == 0,
        "onEach<Path>() places a factor on a repeated block"
    );

    /// the coordinates of one repetition
    static constexpr int Step = BlockSpace::ValueDimensionStep;

    /// whether the factor's moments() returns a std::expected, so this one's does too
    static constexpr bool MomentsMayFail = internal::momentsMayFail<F>;

public:

    using Scalar = typename F::Scalar;                    ///< the scalar type
    using Index = Eigen::Index;                           ///< Eigen's index type
    using SpaceType = BlockSpace;                         ///< the space the values live in
    using MomentsType = Moments<BlockSpace, Scalar>;      ///< a mean and covariance together
    using VectorType = typename MomentsType::VectorType;  ///< a value

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /** \a factor on each repetition of \a space. */
    EachRepetition(F factor, BlockSpace space)
        : m_factor{std::move(factor)}
        , m_space{std::move(space)}
    {
        EMRYS_FILTER_ASSERT(
            Emrys::valueDimension(m_factor.space()) == Step && "the factor must cover one repetition of the block"
        );
        if constexpr (!HasLogDensity<F>)
        {
            [[maybe_unused]] bool const checked = m_factor.toDensity().has_value();
            EMRYS_FILTER_ASSERT(checked && "a factor failed its check; call its toDensity() first to handle this");
        }
    }

    /** \returns the factor on one repetition, as stored */
    [[nodiscard]] F const &
    factor() const
    {
        return m_factor;
    }

    /** \returns the block's space, with its count */
    [[nodiscard]] BlockSpace const &
    space() const
    {
        return m_space;
    }

    /** \returns the number of repetitions */
    [[nodiscard]] Index
    count() const
    {
        return m_space.count();
    }

    /** \returns the number of coordinates of the block */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the log of the density at \a x: the sum of the factor's log-densities at the
     * repetitions
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "logDensity() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        decltype(auto) density = internal::densityOf(m_factor);
        Scalar result = 0;
        for (Index k = 0; k < count(); ++k)
        {
            result += Scalar(density.logDensity(x.derived().template segment<Step>(k * Step)));
        }
        return result;
    }

    /** \returns the mean and covariance: the factor's moments at each repetition, and zero
     * covariance between repetitions. If the factor's moments() can fail, so can this.
     */
    [[nodiscard]] auto
    moments() const -> std::conditional_t<MomentsMayFail, std::expected<MomentsType, MomentsError>, MomentsType>
    {
        decltype(auto) density = internal::densityOf(m_factor);
        auto const & one = density.moments();
        auto const place = [&](auto const & moments) {
            auto result = MomentsType{valueDimension(), valueDimension()};
            result.mean().setZero();
            result.covarianceLower().setZero();
            for (Index k = 0; k < count(); ++k)
            {
                result.mean().template segment<Step>(k * Step) = moments.mean();
                result.covarianceLower().template block<Step, Step>(k * Step, k * Step) =
                    moments.covarianceLower().template selfadjointView<Eigen::Lower>();
            }
            return result;
        };
        if constexpr (MomentsMayFail)
        {
            if (!one)
            {
                return std::unexpected{one.error()};
            }
            return place(*one);
        }
        else
        {
            return place(one);
        }
    }

private:

    F m_factor;                                         ///< the factor on one repetition
    EMRYS_FILTER_NO_UNIQUE_ADDRESS BlockSpace m_space;  ///< the block's space, with its count
};

/** \ingroup Distributions
 * \returns \a n independent draws from \a each, one per column: the factor's draws at each
 * repetition, repetition by repetition.
 */
template <typename F, typename BlockSpace, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<BlockSpace, typename F::Scalar>
drawSamples(EachRepetition<F, BlockSpace> const & each, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    constexpr int Step = BlockSpace::ValueDimensionStep;
    auto samples = SampleSet<BlockSpace, typename F::Scalar>{each.valueDimension(), n};
    decltype(auto) density = internal::densityOf(each.factor());
    for (Eigen::Index k = 0; k < each.count(); ++k)
    {
        samples.template middleRows<Step>(k * Step) = drawSamples(density, n, rng);
    }
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a each, the factor's drawSample() at each repetition, in order.
 */
template <typename F, typename BlockSpace, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<BlockSpace, typename F::Scalar>::VectorType
drawSample(EachRepetition<F, BlockSpace> const & each, G & rng)
{
    constexpr int Step = BlockSpace::ValueDimensionStep;
    auto sample = typename Moments<BlockSpace, typename F::Scalar>::VectorType{};
    sample.resize(each.valueDimension());
    decltype(auto) density = internal::densityOf(each.factor());
    for (Eigen::Index k = 0; k < each.count(); ++k)
    {
        sample.template segment<Step>(k * Step) = drawSample(density, rng);
    }
    return sample;
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a each, one per column, with the log-density at each:
 * the sum of the factor's log-densities at its draws.
 */
template <typename F, typename BlockSpace, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<BlockSpace, typename F::Scalar>
drawSamplesWithLogDensity(EachRepetition<F, BlockSpace> const & each, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    using Scalar = typename F::Scalar;
    constexpr int Step = BlockSpace::ValueDimensionStep;
    auto result = SamplesWithLogDensity<BlockSpace, Scalar>{
        SampleSet<BlockSpace, Scalar>{each.valueDimension(), n},
        Eigen::Array<Scalar, Eigen::Dynamic, 1>::Zero(n)
    };
    decltype(auto) density = internal::densityOf(each.factor());
    for (Eigen::Index k = 0; k < each.count(); ++k)
    {
        auto draws = drawSamplesWithLogDensity(density, n, rng);
        result.samples.template middleRows<Step>(k * Step) = draws.samples;
        result.logDensity += draws.logDensity;
    }
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a each, the factor's drawSampleWithLogDensity() at each repetition, in
 * order, with the sum of their log-densities.
 */
template <typename F, typename BlockSpace, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<BlockSpace, typename F::Scalar>
drawSampleWithLogDensity(EachRepetition<F, BlockSpace> const & each, G & rng)
{
    constexpr int Step = BlockSpace::ValueDimensionStep;
    auto result = SampleWithLogDensity<BlockSpace, typename F::Scalar>{};
    result.sample.resize(each.valueDimension());
    decltype(auto) density = internal::densityOf(each.factor());
    for (Eigen::Index k = 0; k < each.count(); ++k)
    {
        auto const draw = drawSampleWithLogDensity(density, rng);
        result.sample.template segment<Step>(k * Step) = draw.sample;
        result.logDensity += draw.logDensity;
    }
    return result;
}

namespace internal
{

/** The path \a Path in \a Space, with WholeSpace the whole space. */
template <typename Space, typename Path>
struct EachBlockOf
{
    /** The block. */
    using type = PathSubspaceType<Space, Path>;

    /** \returns the block in \a space */
    [[nodiscard]] static type
    in(Space const & space)
    {
        return pathSubspace<Path>(space);
    }

    /** \returns its coordinates in \a space, as an index set */
    [[nodiscard]] static auto
    indicesIn(Space const & space)
    {
        return asIndexSet(idx<Path>(space));
    }
};

/** The whole space. */
template <typename Space>
struct EachBlockOf<Space, WholeSpace>
{
    /** The block. */
    using type = Space;

    /** \returns \a space */
    [[nodiscard]] static Space const &
    in(Space const & space)
    {
        return space;
    }

    /** \returns every coordinate of \a space, as an index set */
    [[nodiscard]] static auto
    indicesIn(Space const & space)
    {
        return asIndexSet(Eigen::seqN(Eigen::Index{0}, valueDimension(space)));
    }
};

/** A factor on every repetition of the repeated block named by the path \a Path: an
 * EachRepetition of it once the product's space gives the block's count.
 */
template <ProductFactor F, typename Path>
struct PlacedFactor<OnEach<F, Path>>
{
    /** The factor's type in a product over \a Space: an EachRepetition over the block. */
    template <typename Space>
    using TypeIn = EachRepetition<F, typename EachBlockOf<Space, Path>::type>;

    /** \returns the factor of \a placed on each repetition of the block in \a space */
    template <typename Space>
    [[nodiscard]] static TypeIn<Space>
    factorIn(OnEach<F, Path> && placed, Space const & space)
    {
        return {std::move(placed.value), EachBlockOf<Space, Path>::in(space)};
    }

    /** \returns the coordinates of \a space that \a Path names, as an index set */
    template <typename Space>
    [[nodiscard]] static auto
    indicesIn(OnEach<F, Path> const &, Space const & space)
    {
        return EachBlockOf<Space, Path>::indicesIn(space);
    }

    /** Whether \a Path names a repeated block in \a Space whose one repetition's space is the
     * factor's.
     */
    template <typename Space>
    static constexpr bool fitsIn = [] {
        using Block = typename EachBlockOf<Space, Path>::type;
        if constexpr (requires { typename Block::BlockType; })
        {
            return factorFits<typename F::SpaceType, typename Block::BlockType>;
        }
        else
        {
            return false;
        }
    }();
};

/** Whether \a T places a factor by tag path: on() or onEach() of a ProductFactor. */
template <typename T>
inline constexpr bool isPlacedFactor = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <ProductFactor F, typename Path>
inline constexpr bool isPlacedFactor<OnPath<F, Path>>{true};

template <ProductFactor F, typename Path>
inline constexpr bool isPlacedFactor<OnEach<F, Path>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Distributions
 * \brief The factors of a product, each placed by tag path with on() or onEach(), in no space
 * yet: a Product once placed in one (decisions.md [product-factors]).
 *
 * Paths name blocks by type, so the factors fit every instance of a space type, whatever its
 * counts. <tt>Product{space, factors}</tt> places them in one instance, resolving each path's
 * coordinates and each onEach()'s count there. It is not a distribution, since without sizes it
 * has no density. Additive measurement noise may be given as one (see additiveNoise()), and an
 * update places it in each measurement's space.
 *
 * \code
 * auto const landmarks = Factors{onEach<Landmarks>(GaussianParams<Landmark>{zero, P})};
 * auto const prior = Product{space, landmarks};  // I (x) P over space's count of landmarks
 * \endcode
 *
 * \tparam Placed  the factors, as on() and onEach() return them, in order
 *
 * \sa class Product, on(), onEach()
 */
template <typename... Placed>
class Factors
{
    static_assert(sizeof...(Placed) > 0, "a product needs at least one factor");
    static_assert(
        (internal::isPlacedFactor<Placed> && ...),
        "each factor is placed by tag path, with on<Path>() or onEach<Path>()"
    );

public:

    /** The factors \a placed. */
    explicit Factors(Placed... placed)
        : m_placed{std::move(placed)...}
    {}

    /** \returns the factors, as placed by path */
    [[nodiscard]] std::tuple<Placed...> const &
    placed() const
    {
        return m_placed;
    }

private:

    std::tuple<Placed...> m_placed;  ///< the factors, as placed by path
};

namespace internal
{

/** Whether \a T is a Factors. */
template <typename T>
inline constexpr bool isFactors = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename... Placed>
inline constexpr bool isFactors<Factors<Placed...>>{true};
/// \endcond

}  // namespace internal

/** \ingroup Distributions
 * \brief The product of independent densities, each on its own coordinates of a space.
 *
 * Represents \f$ p(x) = \prod_k p_k(x_{S_k}) \f$, where the index sets \f$ S_k \f$ partition
 * the coordinates: the joint density of
 * [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)) blocks. A prior that is %Gaussian in
 * position, uniform in heading, and Gamma in speed is one product:
 * \code
 * auto const prior = Emrys::Product{State{}, on<Position>(position),  // a GaussianParams<Position>
 *                                          on<Heading>(Uniform<Heading>{}),
 *                                          on<Speed>(Gamma<Speed>::create(4.0, 0.5).value())};
 * \endcode
 * Each factor's space must be the space of the block it covers: a factor placed by
 * on<Path>() is built in that block's space, a tag derived from it, or with both linear.
 * Sampling fills each factor's rows; log-densities add.
 *
 * A factor is a density or parameters (see ProductFactor), held by value. The product checks
 * parameters with toDensity() when it is built, and asserts that this succeeds; call their
 * toDensity() first to handle a failure. A factor that is a view, such as a Gaussian, checks
 * its parameters, which then must not change while the product is read, so prefer passing
 * the parameters.
 *
 * marginal<Path>() and conditional<GivenPath>() work on the factors' blocks, matched by
 * the factors' space types:
 *  - a path through a factor's space type, or to a block inside it, is that factor's own
 *    marginal or conditional, and the other factors are unchanged;
 *  - a path to a block made of whole factors selects those factors, or drops them when given.
 * A marginal or conditional with one factor left returns that factor alone. Marginals return
 * parameters or values, never views of this product. A conditional that may fail, such as a
 * Student-t's or a %Gaussian's that must factorize, returns a std::expected.
 *
 * \tparam Space_    space of the whole vector (see \ref Spaces)
 * \tparam Scalar_   scalar type, shared by every factor
 * \tparam Factors_  the factors' types, in order
 *
 * \sa on(), drawSamples(Product const &, Eigen::Index, G &)
 */
template <Space Space_, RealScalar Scalar_, ProductFactor... Factors_>
class Product
{
    static_assert(sizeof...(Factors_) > 0, "a product needs at least one factor");
    static_assert(
        (std::is_same_v<typename Factors_::Scalar, Scalar_> && ...),
        "every factor must have the same scalar"
    );

    /// whether any factor's moments() returns a std::expected, so this product's does too
    static constexpr bool MomentsMayFail = (internal::momentsMayFail<Factors_> || ...);

public:

    using Scalar = Scalar_;                               ///< the scalar type
    using Index = Eigen::Index;                           ///< Eigen's index type
    using SpaceType = Space_;                             ///< the space the values live in
    using MomentsType = Moments<Space_, Scalar_>;         ///< a mean and covariance together
    using VectorType = typename MomentsType::VectorType;  ///< a value
    using MatrixType = typename MomentsType::MatrixType;  ///< a matrix over differences

    /// the length of a value, or Eigen::Dynamic
    static constexpr int ValueDimensionAtCompileTime = MomentsType::ValueDimensionAtCompileTime;

    /// a bound on the length of a value, or Eigen::Dynamic
    static constexpr int MaxValueDimensionAtCompileTime = MomentsType::MaxValueDimensionAtCompileTime;

    /// the length of a difference and the rows of a covariance, or Eigen::Dynamic
    static constexpr int DifferenceDimensionAtCompileTime = MomentsType::DifferenceDimensionAtCompileTime;

    /// a bound on the length of a difference, or Eigen::Dynamic
    static constexpr int MaxDifferenceDimensionAtCompileTime = MomentsType::MaxDifferenceDimensionAtCompileTime;

    static constexpr std::size_t FactorCount = sizeof...(Factors_);  ///< the number of factors

    static_assert(
        internal::hasEqualSizes<Space_>,
        "a product places its factors by coordinate, so its space's values and differences must have the "
        "same coordinates"
    );

    /** The type of factor \a K. */
    template <std::size_t K>
    using FactorType = std::tuple_element_t<K, std::tuple<Factors_...>>;

    /** The product of the factors \a placed, made by on() or onEach(), in \a space. Their indices
     * must partition the space, and each factor's space must be that of the block it covers.
     */
    template <typename... Placed>
        requires(sizeof...(Placed) == sizeof...(Factors_)
                 && (std::is_same_v<typename internal::PlacedFactor<Placed>::template TypeIn<Space_>, Factors_> && ...))
    explicit Product(Space_ space, Placed... placed)
        : m_space{std::move(space)}
        , m_indices{resolve<Factors_>(placed)...}
        , m_factors{internal::PlacedFactor<Placed>::factorIn(std::move(placed), m_space)...}
    {
        static_assert(
            (internal::PlacedFactor<Placed>::template fitsIn<Space_> && ...),
            "a factor's space must be the space of the block it covers"
        );
        EMRYS_FILTER_ASSERT(sizesMatch() && "a factor's dimension must match the number of coordinates it covers");
        EMRYS_FILTER_ASSERT(partitionsSpace() && "the factors' indices must partition the space");
        std::apply(
            [](auto const &... factors) {
                (checkParams(factors), ...);
            },
            m_factors
        );
    }

    /** The product of \a factors placed in \a space (see Factors). */
    template <typename... Placed>
    explicit Product(Space_ space, Factors<Placed...> const & factors)
        : Product{std::move(space), factors, std::index_sequence_for<Placed...>{}}
    {}

    /** \returns the number of coordinates of the space */
    [[nodiscard]] Index
    valueDimension() const
    {
        return Emrys::valueDimension(m_space);
    }

    /** \returns the same: a product's space has plain differences */
    [[nodiscard]] Index
    differenceDimension() const
    {
        return Emrys::differenceDimension(m_space);
    }

    /** \returns the space of the whole vector, with its sizes */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns factor \a K, as stored */
    template <std::size_t K>
    [[nodiscard]] FactorType<K> const &
    factor() const
    {
        return std::get<K>(m_factors);
    }

    /** \returns the coordinates factor \a K covers */
    template <std::size_t K>
    [[nodiscard]] auto const &
    indices() const
    {
        return std::get<K>(m_indices);
    }

    /** Calls \a fn(density, indices) for each factor in order, with its density (a view, for
     * parameters) and the coordinates it covers.
     */
    template <typename Fn>
    void
    forEachFactor(Fn && fn) const
    {
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            (fn(internal::densityOf(std::get<K>(m_factors)), std::get<K>(m_indices)), ...);
        }(std::index_sequence_for<Factors_...>{});
    }

    /** \returns the log of the density at \a x: the sum of the factors' log-densities */
    template <typename Derived>
    [[nodiscard]] Scalar
    logDensity(Eigen::MatrixBase<Derived> const & x) const
    {
        static_assert(Derived::IsVectorAtCompileTime, "logDensity() expects a vector");
        EMRYS_FILTER_ASSERT(x.size() == valueDimension());
        Scalar result = 0;
        forEachFactor([&](auto const & density, auto const & indices) {
            result += Scalar(density.logDensity(x.derived()(indices)));
        });
        return result;
    }

    /** \returns the density at \a x
     *
     * \warning Underflows easily in high dimensions; prefer logDensity().
     */
    template <typename Derived>
    [[nodiscard]] Scalar
    density(Eigen::MatrixBase<Derived> const & x) const
    {
        return Eigen::numext::exp(logDensity(x));
    }

    /** \returns the mean and covariance: each factor's moments in its coordinates, and zero
     * covariance between factors. If any factor's moments() can fail, so can this, with the
     * first factor's error.
     */
    [[nodiscard]] auto
    moments() const -> std::conditional_t<MomentsMayFail, std::expected<MomentsType, MomentsError>, MomentsType>
    {
        auto result = MomentsType{valueDimension(), differenceDimension()};
        result.mean().setZero();
        result.covarianceLower().setZero();
        auto error = std::optional<MomentsError>{};
        auto const place = [&](auto const & moments, auto const & indices) {
            using FactorMatrix = std::remove_cvref_t<decltype(moments.covarianceLower())>;
            result.mean()(indices) = moments.mean();
            result.covarianceLower()(indices, indices) =
                FactorMatrix{moments.covarianceLower().template selfadjointView<Eigen::Lower>()};
        };
        forEachFactor([&](auto const & density, auto const & indices) {
            if (error)
            {
                return;
            }
            auto const & factorMoments = density.moments();
            if constexpr (internal::isExpected<std::remove_cvref_t<decltype(factorMoments)>>)
            {
                if (!factorMoments)
                {
                    error = factorMoments.error();
                }
                else
                {
                    place(*factorMoments, indices);
                }
            }
            else
            {
                place(factorMoments, indices);
            }
        });
        if constexpr (MomentsMayFail)
        {
            if (error)
            {
                return std::unexpected{*error};
            }
        }
        return result;
    }

    /** \returns the distribution of the subspace named by the tag path \a Path (see idx()), read
     * from this product rather than copied where it can be (decisions.md [element-marginals]):
     *  - for a factor's whole block, a reference to that factor;
     *  - for a block inside a factor, that factor's own marginal, a view for a factor that gives
     *    one, such as a GaussianParams, and a value for one that gives a value;
     *  - for a block made of several whole factors, the product of those factors in that
     *    subspace, which copies them.
     */
    template <typename Path>
    [[nodiscard]] decltype(auto)
    marginal() const &
    {
        return internal::withPathTags<Path>([&]<typename... Tags>() -> decltype(auto) {
            return marginalAlong<Tags...>();
        });
    }

    /** A marginal of a temporary would outlive it. */
    template <typename Path>
    void
    marginal() const && = delete;

    /** \returns the distribution of the other coordinates given that the subspace named by the
     * path \a GivenPath (see idx()) equals \a values:
     *  - for a path to a block inside a factor's space, that factor's conditional in place of
     *    the factor, and the other factors unchanged;
     *  - for a factor's whole block, or a block made of whole factors, the remaining factors,
     *    since the factors are independent.
     *
     * The result is in the space of the remaining blocks (see
     * GaussianParams::conditional<GivenPath>()); with one factor left, it is that factor (or its
     * conditional) alone. When the conditioned factor's conditional can fail, the result is a
     * std::expected with its error, including a failure to factorize a %Gaussian it returns.
     */
    template <typename GivenPath, typename Derived>
    [[nodiscard]] auto
    conditional(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::withPathTags<GivenPath>([&]<typename... Tags>() {
            return conditionalAlong<Tags...>(values);
        });
    }

private:

    /** The product of \a factors in \a space, each taken from its position \a K. */
    template <typename... Placed, std::size_t... K>
    Product(Space_ space, Factors<Placed...> const & factors, std::index_sequence<K...>)
        : Product{std::move(space), std::get<K>(factors.placed())...}
    {}

    /** marginal() along the tags of a path. */
    template <typename... Path>
    [[nodiscard]] decltype(auto)
    marginalAlong() const
    {
        using Match = MatchType<Path...>;
        if constexpr (Match::value.factor != None)
        {
            assertOnOwnBlock<Match::value.factor>(OwnBlockPath<Match::value, Path...>{});
            if constexpr (Match::value.whole)
            {
                return std::get<Match::value.factor>(m_factors);
            }
            else
            {
                return factorMarginal(
                    std::get<Match::value.factor>(m_factors),
                    internal::DropFront<Match::value.start, Path...>{}
                );
            }
        }
        else
        {
            static_assert(
                std::ranges::count(withinOf<Path...>(), true) > 0,
                "the path neither reaches into a factor nor contains one"
            );
            auto const target = internal::resolveIndices<ValueDimensionAtCompileTime, MaxValueDimensionAtCompileTime>(
                valueDimension(),
                internal::asIndexSet(idx<TagPath<Path...>>(m_space))
            );
            EMRYS_FILTER_ASSERT(
                coversExactly(withinOf<Path...>(), target) && "the path's block must be made of whole factors"
            );
            return keepOnly<withinOf<Path...>()>(target, internal::subspace<Space_, Path...>(m_space));
        }
    }

    /** conditional() along the tags of a path. */
    template <typename... GivenPath, typename Derived>
    [[nodiscard]] auto
    conditionalAlong(Eigen::MatrixBase<Derived> const & values) const
    {
        return internal::tagConditional<TagPath<GivenPath...>>(
            m_space,
            [&](auto const & targetCoordinates, auto const & givenCoordinates, auto targetSpace, auto const &) {
                // The values and the differences have the same coordinates.
                auto const & target = targetCoordinates.values;
                auto const & given = givenCoordinates.values;
                EMRYS_FILTER_ASSERT(values.size() == given.size());
                using Match = MatchType<GivenPath...>;
                if constexpr (Match::value.factor != None && !Match::value.whole)
                {
                    assertOnOwnBlock<Match::value.factor>(OwnBlockPath<Match::value, GivenPath...>{});
                    return replaceFactor<Match::value.factor>(
                        target,
                        given,
                        std::move(targetSpace),
                        internal::DropFront<Match::value.start, GivenPath...>{},
                        values
                    );
                }
                else
                {
                    static_assert(
                        std::ranges::count(droppedOf<GivenPath...>(), true) > 0,
                        "the path neither reaches into a factor nor contains one"
                    );
                    static_assert(
                        std::ranges::count(droppedOf<GivenPath...>(), true) < std::ptrdiff_t(FactorCount),
                        "conditioning on every factor leaves nothing"
                    );
                    EMRYS_FILTER_ASSERT(
                        coversExactly(droppedOf<GivenPath...>(), given)
                        && "the given block must be made of whole factors"
                    );
                    return keepOnly<complement(droppedOf<GivenPath...>())>(target, std::move(targetSpace));
                }
            }
        );
    }

    static constexpr std::size_t None = FactorCount;  ///< no factor: one past the last

    /** The factor a tag path reaches into, if any: the path from \c start on lies within it, and
     * \c whole means the path names exactly the factor's block.
     */
    struct PathMatch
    {
        std::size_t factor = None;  ///< the factor, or None
        std::size_t start = 0;      ///< where in the path the factor's space begins
        bool whole = false;         ///< the path names exactly the factor's block
        bool named = false;         ///< the path names the factor's space type at \c start
    };

    /** The tag path of the block of the factor that \a Match found in \a Path: the path up to the
     * factor's space type, or that type alone if the path only reaches inside it.
     */
    template <PathMatch Match, typename... Path>
    using OwnBlockPath = std::conditional_t<
        Match.named,
        internal::TakeFront<Match.start + 1, Path...>,
        internal::TypeList<typename FactorType<Match.factor>::SpaceType>>;

    /** \returns the factor whose space type is \a T, or None */
    template <typename T>
    static consteval std::size_t
    factorNamed()
    {
        constexpr auto same = std::array<bool, FactorCount>{std::is_same_v<T, typename Factors_::SpaceType>...};
        static_assert(std::ranges::count(same, true) <= 1, "two factors share this space type, so it names neither");
        auto const found = std::ranges::find(same, true);
        return found == same.end() ? None : std::size_t(found - same.begin());
    }

    /** \returns the factor whose space contains the tag \a T, or None */
    template <typename T>
    static consteval std::size_t
    factorContaining()
    {
        constexpr auto inside =
            std::array<bool, FactorCount>{(internal::countTag<T, typename Factors_::SpaceType>() > 0)...};
        static_assert(std::ranges::count(inside, true) <= 1, "this tag lies in more than one factor's space");
        auto const found = std::ranges::find(inside, true);
        return found == inside.end() ? None : std::size_t(found - inside.begin());
    }

    /** \returns which factors' spaces lie within the block named by the tag \a T */
    template <typename T>
    static consteval std::array<bool, FactorCount>
    factorsWithin()
    {
        return {(internal::countTag<typename Factors_::SpaceType, T>() > 0)...};
    }

    /** \returns the PathMatch of \a Path: the first tag that names a factor's space type, else the
     * factor containing the first tag
     */
    template <typename... Path>
    static consteval PathMatch
    matchPath()
    {
        constexpr auto named = std::array<std::size_t, sizeof...(Path)>{factorNamed<Path>()...};
        for (std::size_t j = 0; j < named.size(); ++j)
        {
            if (named[j] != None)
            {
                return {named[j], j, j + 1 == named.size(), true};
            }
        }
        return {factorContaining<std::tuple_element_t<0, std::tuple<Path...>>>(), 0, false};
    }

    /** \returns the factors that conditioning on a path ending in the tag \a Last drops, when the
     * path does not reach inside a factor: the factor it names, or those within its block
     */
    template <PathMatch Match, typename Last>
    static consteval std::array<bool, FactorCount>
    droppedBy()
    {
        if constexpr (Match.factor != None)
        {
            auto result = std::array<bool, FactorCount>{};
            result[Match.factor] = true;
            return result;
        }
        else
        {
            return factorsWithin<Last>();
        }
    }

    // The path analyses are types and consteval functions, not local constexpr variables: GCC can
    // conflate a local used as a template argument across instantiations of a generic lambda.

    /** The PathMatch of \a Path, as a type. */
    template <typename... Path>
    using MatchType = std::integral_constant<PathMatch, matchPath<Path...>()>;

    /** The last tag of \a Path. */
    template <typename... Path>
    using LastTag = std::tuple_element_t<sizeof...(Path) - 1, std::tuple<Path...>>;

    /** The factors within the block named by \a Path. */
    template <typename... Path>
    static consteval std::array<bool, FactorCount>
    withinOf()
    {
        return factorsWithin<LastTag<Path...>>();
    }

    /** The factors that conditioning on \a Path drops, when it does not reach inside a factor. */
    template <typename... Path>
    static consteval std::array<bool, FactorCount>
    droppedOf()
    {
        return droppedBy<MatchType<Path...>::value, LastTag<Path...>>();
    }

    /** \returns \a mask inverted */
    static consteval std::array<bool, FactorCount>
    complement(std::array<bool, FactorCount> mask)
    {
        for (bool & b : mask)
        {
            b = !b;
        }
        return mask;
    }

    /** \returns the positions of the factors \a Mask selects, in order */
    template <std::array<bool, FactorCount> Mask>
    static consteval auto
    maskedFactors()
    {
        auto result = std::array<std::size_t, std::size_t(std::ranges::count(Mask, true))>{};
        std::size_t out = 0;
        for (std::size_t k = 0; k < FactorCount; ++k)
        {
            if (Mask[k])
            {
                result[out++] = k;
            }
        }
        return result;
    }

    /** Checks \a factor with toDensity() if it is parameters, which factorizes those that cache a
     * factorization; a failure is a programmer error.
     */
    template <typename Factor>
    static void
    checkParams(Factor const & factor)
    {
        if constexpr (!HasLogDensity<Factor>)
        {
            [[maybe_unused]] bool const checked = factor.toDensity().has_value();
            EMRYS_FILTER_ASSERT(checked && "a factor failed its check; call its toDensity() first to handle this");
        }
    }

    /** \returns the coordinates that \a placed covers in the space, resolved to an index vector
     * with \a Factor's size
     */
    template <typename Factor, typename Placed>
    [[nodiscard]] internal::IndexVector<Factor::ValueDimensionAtCompileTime, Factor::MaxValueDimensionAtCompileTime>
    resolve(Placed const & placed) const
    {
        return internal::IndexVector<Factor::ValueDimensionAtCompileTime, Factor::MaxValueDimensionAtCompileTime>{
            internal::resolveIndices<ValueDimensionAtCompileTime, MaxValueDimensionAtCompileTime>(
                valueDimension(),
                internal::PlacedFactor<Placed>::indicesIn(placed, m_space)
            )
        };
    }

    /** Calls \a fn(factor, indices) for each stored factor, parameters or density. */
    template <typename Fn>
    void
    forEachStored(Fn && fn) const
    {
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            (fn(std::get<K>(m_factors), std::get<K>(m_indices)), ...);
        }(std::index_sequence_for<Factors_...>{});
    }

    /** \returns whether each factor's dimension equals the number of coordinates it covers */
    [[nodiscard]] bool
    sizesMatch() const
    {
        bool result = true;
        forEachStored([&](auto const & factor, auto const & indices) {
            result = result && Emrys::valueDimension(factor.space()) == indices.size();
        });
        return result;
    }

    /** \returns whether the factors' coordinates are distinct and cover the whole space */
    [[nodiscard]] bool
    partitionsSpace() const
    {
        Index total = 0;
        forEachStored([&](auto const &, auto const & indices) {
            total += indices.size();
        });
        bool const distinct = std::apply(
            [](auto const &... indices) {
                return internal::distinctIndices(indices...);
            },
            m_indices
        );
        return total == valueDimension() && distinct;
    }

    /** Asserts that factor \a K covers the block named by \a Path, in order, as a factor placed
     * by on<Path>() does, so paths within it line up with paths within the product.
     */
    template <std::size_t K, typename... Path>
    void
    assertOnOwnBlock(internal::TypeList<Path...>) const
    {
        [[maybe_unused]] auto const block =
            internal::resolveIndices<ValueDimensionAtCompileTime, MaxValueDimensionAtCompileTime>(
                valueDimension(),
                internal::asIndexSet(idx<TagPath<Path...>>(m_space))
            );
        EMRYS_FILTER_ASSERT(
            std::equal(block.begin(), block.end(), std::get<K>(m_indices).begin(), std::get<K>(m_indices).end())
            && "a factor reached by a tag path must be placed on that tag's block"
        );
    }

    /** \returns whether the factors in \a mask cover exactly the coordinates in \a block, in any
     * order; the factors' coordinates are disjoint, and \a block holds none twice
     */
    template <typename Block>
    [[nodiscard]] bool
    coversExactly(std::array<bool, FactorCount> const & mask, Block const & block) const
    {
        std::size_t k = 0;
        std::size_t covered = 0;
        bool inBlock = true;
        forEachStored([&](auto const &, auto const & indices) {
            if (mask[k++])
            {
                covered += std::size_t(indices.size());
                for (Index const i : indices)
                {
                    inBlock = inBlock && std::ranges::find(block, i) != std::ranges::end(block);
                }
            }
        });
        return inBlock && covered == std::size_t(std::ranges::size(block));
    }

    /** \returns the marginal of \a factor on \a Path, which lies within it: the factor's own, a view
     * or a value
     */
    template <typename Factor, typename... Path>
    [[nodiscard]] static auto
    factorMarginal(Factor const & factor, internal::TypeList<Path...>)
    {
        static_assert(
            requires { factor.template marginal<TagPath<Path...>>(); },
            "this factor has no marginal<Path>()"
        );
        return factor.template marginal<TagPath<Path...>>();
    }

    /** \returns the factors in \a Keep, on their coordinates' positions within \a target, which
     * form \a space: their product, or a reference to the one factor alone
     */
    template <std::array<bool, FactorCount> Keep, typename Target, typename SubSpace>
    [[nodiscard]] decltype(auto)
    keepOnly(Target const & target, SubSpace space) const
    {
        constexpr auto kept = maskedFactors<Keep>();
        if constexpr (kept.size() == 1)
        {
            return std::get<kept[0]>(m_factors);
        }
        else
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                return Product<SubSpace, Scalar, FactorType<kept[J]>...>{
                    std::move(space),
                    internal::placedAt(positionsOf<kept[J]>(target), std::get<kept[J]>(m_factors))...
                };
            }(std::make_index_sequence<kept.size()>{});
        }
    }

    /** \returns the positions within \a target of factor \a K's coordinates */
    template <std::size_t K, typename Target>
    [[nodiscard]] auto
    positionsOf(Target const & target) const
    {
        using Factor = FactorType<K>;
        return internal::positionsIn<Factor::ValueDimensionAtCompileTime, Factor::MaxValueDimensionAtCompileTime>(
            std::get<K>(m_indices),
            target
        );
    }

    /** \returns this product with factor \a K replaced by its conditional on the relative path
     * \a Path equal to \a values, in \a space, the space of the coordinates \a target
     */
    template <std::size_t K, typename Target, typename Given, typename SubSpace, typename... Path, typename Derived>
    [[nodiscard]] auto
    replaceFactor(
        Target const & target,
        Given const & given,
        SubSpace space,
        internal::TypeList<Path...>,
        Eigen::MatrixBase<Derived> const & values
    ) const
    {
        auto const & factor = std::get<K>(m_factors);
        static_assert(
            requires { factor.template conditional<TagPath<Path...>>(values); },
            "this factor has no conditional<Path>()"
        );
        if constexpr (FactorCount == 1)
        {
            return factor.template conditional<TagPath<Path...>>(values);
        }
        else
        {
            using Raw = decltype(factor.template conditional<TagPath<Path...>>(values));
            using Replacement = internal::ConditionalValue<Raw>;
            using Error = internal::ConditionalError<Raw>;
            static_assert(ProductFactor<Replacement>, "this factor's conditional cannot be a factor of a product");
            using Result = internal::ReplacedProduct<Scalar, SubSpace, K, Replacement, std::tuple<Factors_...>>;

            auto const build = [&](Replacement replacement) {
                // The replacement covers factor K's coordinates that were not given, in order.
                auto const & own = std::get<K>(m_indices);
                auto const isRemaining = [&](Index i) {
                    return std::ranges::find(given, i) == std::ranges::end(given);
                };
                auto remaining = internal::IndexVector<
                    Replacement::ValueDimensionAtCompileTime,
                    Replacement::MaxValueDimensionAtCompileTime>{};
                remaining.resize(Index(std::ranges::count_if(own, isRemaining)));
                Index r = 0;
                for (Index const i : own)
                {
                    if (isRemaining(i))
                    {
                        remaining(r++) = i;
                    }
                }
                auto positions = internal::positionsIn<
                    Replacement::ValueDimensionAtCompileTime,
                    Replacement::MaxValueDimensionAtCompileTime>(remaining, target);
                return [&]<std::size_t... J>(std::index_sequence<J...>) {
                    return Result{std::move(space), placeFactor<J, K>(target, positions, replacement)...};
                }(std::index_sequence_for<Factors_...>{});
            };

            auto raw = factor.template conditional<TagPath<Path...>>(values);
            if constexpr (std::is_void_v<Error>)
            {
                return build(std::move(raw));
            }
            else
            {
                using Return = std::expected<Result, Error>;
                if constexpr (internal::isExpected<Raw>)
                {
                    if (!raw)
                    {
                        return Return{std::unexpect, raw.error()};
                    }
                    return Return{build(std::move(*raw))};
                }
                else
                {
                    if (auto checked = raw.toDensity(); !checked)
                    {
                        return Return{std::unexpect, checked.error()};
                    }
                    return Return{build(std::move(raw))};
                }
            }
        }
    }

    /** \returns factor \a J placed on its positions within \a target, or for \a J equal to \a K,
     * \a replacement placed on \a positions
     */
    template <std::size_t J, std::size_t K, typename Target, typename Positions, typename Replacement>
    [[nodiscard]] auto
    placeFactor(Target const & target, Positions const & positions, Replacement & replacement) const
    {
        if constexpr (J == K)
        {
            return internal::placedAt(positions, std::move(replacement));
        }
        else
        {
            return internal::placedAt(positionsOf<J>(target), std::get<J>(m_factors));
        }
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space of the whole vector

    /// the coordinates each factor covers
    std::tuple<
        internal::IndexVector<Factors_::ValueDimensionAtCompileTime, Factors_::MaxValueDimensionAtCompileTime>...>
        m_indices;

    std::tuple<Factors_...> m_factors;  ///< the factors, parameters or densities, in order
};

/// \cond DEDUCTION_GUIDES
template <typename Space, typename First, typename... Rest>
Product(Space, First, Rest...) -> Product<
    Space,
    typename internal::PlacedFactor<First>::template TypeIn<Space>::Scalar,
    typename internal::PlacedFactor<First>::template TypeIn<Space>,
    typename internal::PlacedFactor<Rest>::template TypeIn<Space>...>;

template <typename Space, typename First, typename... Rest>
Product(Space, Factors<First, Rest...>) -> Product<
    Space,
    typename internal::PlacedFactor<First>::template TypeIn<Space>::Scalar,
    typename internal::PlacedFactor<First>::template TypeIn<Space>,
    typename internal::PlacedFactor<Rest>::template TypeIn<Space>...>;

/// \endcond

/** \ingroup Distributions
 * \returns \a n independent draws from \a product, one per column: each factor's draws in
 * its coordinates, factor by factor.
 *
 * \sa drawSamplesWithLogDensity(Product const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, typename... Factors_, std::uniform_random_bit_generator G>
[[nodiscard]] SampleSet<Space, Scalar>
drawSamples(Product<Space, Scalar, Factors_...> const & product, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    auto samples = SampleSet<Space, Scalar>{product.valueDimension(), n};
    product.forEachFactor([&](auto const & density, auto const & indices) {
        samples(indices, Eigen::placeholders::all) = drawSamples(density, n, rng);
    });
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a product, each factor's drawSample() in its coordinates, factor by
 * factor, in a vector sized by the space: the stream of draws that drawSample() of a %Gaussian is.
 */
template <typename Scalar, typename Space, typename... Factors_, std::uniform_random_bit_generator G>
[[nodiscard]] typename Moments<Space, Scalar>::VectorType
drawSample(Product<Space, Scalar, Factors_...> const & product, G & rng)
{
    auto sample = typename Moments<Space, Scalar>::VectorType{};
    sample.resize(product.valueDimension());
    product.forEachFactor([&](auto const & density, auto const & indices) {
        sample(indices) = drawSample(density, rng);
    });
    return sample;
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a product, one per column, with the log-density at
 * each: the sum of the factors' log-densities at their draws.
 *
 * \sa drawSamples(Product const &, Eigen::Index, G &)
 */
template <typename Scalar, typename Space, typename... Factors_, std::uniform_random_bit_generator G>
[[nodiscard]] SamplesWithLogDensity<Space, Scalar>
drawSamplesWithLogDensity(Product<Space, Scalar, Factors_...> const & product, Eigen::Index n, G & rng)
{
    EMRYS_FILTER_ASSERT(n >= 0);
    auto result = SamplesWithLogDensity<Space, Scalar>{
        SampleSet<Space, Scalar>{product.valueDimension(), n},
        Eigen::Array<Scalar, Eigen::Dynamic, 1>::Zero(n)
    };
    product.forEachFactor([&](auto const & density, auto const & indices) {
        auto draws = drawSamplesWithLogDensity(density, n, rng);
        result.samples(indices, Eigen::placeholders::all) = draws.samples;
        result.logDensity += draws.logDensity;
    });
    return result;
}

/** \ingroup Distributions
 * \returns one draw from \a product, each factor's drawSampleWithLogDensity() in its coordinates,
 * factor by factor, with the sum of their log-densities, in a vector sized by the space: the stream
 * of draws that drawSampleWithLogDensity() of a %Gaussian is.
 */
template <typename Scalar, typename Space, typename... Factors_, std::uniform_random_bit_generator G>
[[nodiscard]] SampleWithLogDensity<Space, Scalar>
drawSampleWithLogDensity(Product<Space, Scalar, Factors_...> const & product, G & rng)
{
    auto result = SampleWithLogDensity<Space, Scalar>{};
    result.sample.resize(product.valueDimension());
    product.forEachFactor([&](auto const & density, auto const & indices) {
        auto const draw = drawSampleWithLogDensity(density, rng);
        result.sample(indices) = draw.sample;
        result.logDensity += draw.logDensity;
    });
    return result;
}

}  // namespace Emrys

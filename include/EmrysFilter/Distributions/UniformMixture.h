#pragma once

/** \file
 * uniformComponents(): equal-weight %Gaussian components on a regular grid that approximate a
 * Uniform, alone or as factors of a Product whose other factors are %Gaussian [uniform-mixtures],
 * given one at a time as a UniformGrid, which GaussianMixtureParams::assign() takes. Read Hybrid.h
 * next.
 */

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Distributions/Component.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Distributions/Uniform.h>
#include <EmrysFilter/Spaces/Indexing.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

/** \ingroup Distributions
 * \brief The equal-weight components of a grid over some coordinates of a space, made one at a
 * time when read: what uniformComponents() returns.
 *
 * A coordinate \a i with counts()(i) \f$ = N_i > 0 \f$ is gridded: its interval from
 * \f$ \ell_i \f$ with width \f$ w_i \f$ is cut into \f$ N_i \f$ cells, and the components' means
 * sit at the cells' centers, \f$ \ell_i + (j + \tfrac12) w_i / N_i \f$, with variance
 * \f$ (c w_i / N_i)^2 \f$, where \f$ c \f$ is the relative width. Every other coordinate keeps
 * its mean, and its rows and columns of the covariance, in every component. Component \a k is
 * at the cell whose indices are \a k's digits in the mixed radix of the counts, the first gridded
 * coordinate varying fastest, as a grid's flat cell index does.
 *
 * It is a random-access range of size() elements, each a log-weight of \f$ -\log \prod_i N_i \f$
 * and the component's Moments, so GaussianMixtureParams::assign() fills a mixture from it:
 *
 * \code
 * auto prior = GaussianMixtureParams<Linear<2>>{Capacity{256}};
 * prior.assign(uniformComponents(box, Eigen::Array2i{16, 16}));  // within the capacity, allocates nothing
 * \endcode
 *
 * A filter's starting state is made the same way, with the capacity or allocator the filter
 * needs, and moved into the filter.
 *
 * The mixture must be in space(); the elements carry no space, so the mixture's is the one the
 * components take. The grid holds its values and borrows nothing, and its iterators borrow it.
 */
template <Space Space_, RealScalar Scalar_ = double>
class UniformGrid
{
public:

    using Scalar = Scalar_;                               ///< the scalar type
    using Index = Eigen::Index;                           ///< Eigen's index type
    using SpaceType = Space_;                             ///< the space the components live in
    using MomentsType = Moments<Space_, Scalar_>;         ///< a component's mean and covariance
    using VectorType = typename MomentsType::VectorType;  ///< a value, such as a mean
    using MatrixType = typename MomentsType::MatrixType;  ///< a matrix over differences

    /** A count of cells per coordinate, zero for a coordinate that is not gridded. */
    using CountsType =
        internal::IndexVector<MomentsType::ValueDimensionAtCompileTime, MomentsType::MaxValueDimensionAtCompileTime>;

    /** One component: its log-weight and its mean and covariance. */
    struct Element
    {
        Scalar logWeight;     ///< the log of the component's weight, the same for every one
        MomentsType moments;  ///< the component's mean and covariance
    };

    using value_type = Element;                                     ///< an element, made when read
    using const_iterator = internal::ElementIterator<UniformGrid>;  ///< iterates the elements
    using iterator = const_iterator;                                ///< iterates the elements; they are read-only

    /** The grid in \a space over the coordinates \a i with \a counts(i) > 0, \a counts(i) cells
     * across \a width(i) from \a corner(i). \a corner holds the other coordinates' means, and
     * \a covariance their rows and columns; a gridded coordinate's rows and columns must be zero,
     * and its diagonal entry is set from \a relativeWidth. At least one coordinate is gridded.
     */
    UniformGrid(
        SpaceType space,
        VectorType corner,
        VectorType width,
        CountsType counts,
        MatrixType covariance,
        Scalar relativeWidth
    )
        : m_space{std::move(space)}
        , m_corner{std::move(corner)}
        , m_width{std::move(width)}
        , m_counts{std::move(counts)}
        , m_covariance{std::move(covariance)}
    {
        EMRYS_FILTER_ASSERT(
            isFinite(relativeWidth) && relativeWidth > Scalar(0) && "the relative width must be positive and finite"
        );
        EMRYS_FILTER_ASSERT(
            m_corner.size() == m_width.size() && m_corner.size() == m_counts.size()
            && m_covariance.rows() == m_corner.size() && m_covariance.cols() == m_corner.size()
            && "one corner, width, count, and covariance row per coordinate"
        );
        Index gridded = 0;
        for (Index i = 0; i < m_counts.size(); ++i)
        {
            EMRYS_FILTER_ASSERT(m_counts(i) >= 0 && "a count must not be negative");
            if (m_counts(i) > 0)
            {
                ++gridded;
                m_size *= m_counts(i);
                Scalar const deviation = relativeWidth * m_width(i) / Scalar(m_counts(i));
                m_covariance(i, i) = deviation * deviation;
            }
        }
        EMRYS_FILTER_ASSERT(gridded > 0 && "the grid needs a gridded coordinate");
        m_logWeight = -Eigen::numext::log(Scalar(m_size));
    }

    /** \returns the space the components live in */
    [[nodiscard]] SpaceType const &
    space() const
    {
        return m_space;
    }

    /** \returns the length of a component's mean */
    [[nodiscard]] Index
    valueDimension() const
    {
        return m_corner.size();
    }

    /** \returns the number of components, the product of the gridded coordinates' counts */
    [[nodiscard]] Index
    size() const
    {
        return m_size;
    }

    /** \returns the count of cells along each coordinate, zero for one that is not gridded */
    [[nodiscard]] CountsType const &
    counts() const
    {
        return m_counts;
    }

    /** \returns component \a k */
    [[nodiscard]] Element
    operator[](Index k) const
    {
        EMRYS_FILTER_ASSERT(k >= 0 && k < m_size && "component index out of range");
        VectorType mean = m_corner;
        Index rest = k;
        for (Index i = 0; i < m_counts.size(); ++i)
        {
            Index const count = m_counts(i);
            if (count > 0)
            {
                Index const j = rest % count;
                rest /= count;
                mean(i) = m_corner(i) + (Scalar(j) + Scalar(0.5)) * m_width(i) / Scalar(count);
            }
        }
        return Element{m_logWeight, MomentsType{std::move(mean), m_covariance}};
    }

    /** \returns an iterator at the first component */
    [[nodiscard]] const_iterator
    begin() const
    {
        return const_iterator{*this, 0};
    }

    /** \returns an iterator past the last component */
    [[nodiscard]] const_iterator
    end() const
    {
        return const_iterator{*this, m_size};
    }

private:

    SpaceType m_space;        ///< the space the components live in
    VectorType m_corner;      ///< the gridded coordinates' lower bounds and the others' means
    VectorType m_width;       ///< the gridded coordinates' widths
    CountsType m_counts;      ///< the cells per coordinate, zero where not gridded
    MatrixType m_covariance;  ///< every component's covariance
    Index m_size = 1;         ///< the number of components
    Scalar m_logWeight{};     ///< every component's log-weight
};

namespace internal
{

/** uniformComponents() of a Uniform, with coordinate \a i's count read from \a countOf(i). */
template <Space S, RealScalar Scalar, typename CountOf>
[[nodiscard]] UniformGrid<S, Scalar>
uniformGrid(Uniform<S, Scalar> const & uniform, CountOf const & countOf, Scalar relativeWidth)
{
    using Grid = UniformGrid<S, Scalar>;
    Eigen::Index const n = uniform.valueDimension();
    typename Grid::CountsType counts(n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        counts(i) = countOf(i);
        EMRYS_FILTER_ASSERT(counts(i) >= 1 && "each coordinate needs at least one component");
    }
    return Grid{
        uniform.space(),
        uniform.lower(),
        uniform.width(),
        std::move(counts),
        Grid::MatrixType::Zero(n, n),
        relativeWidth
    };
}

/** Whether \a F is a Uniform, a factor of a Product that uniformComponents() grids. */
template <typename F>
inline constexpr bool isUniform = false;

/// \cond INTERNAL_SPECIALIZATIONS
template <Space S, RealScalar Scalar>
inline constexpr bool isUniform<Uniform<S, Scalar>>{true};
/// \endcond

/** Whether \a F is a %Gaussian in covariance form, on one block or by onEach() on each
 * repetition of one: a factor of a Product that uniformComponents() gives every component.
 */
template <typename F>
inline constexpr bool isGaussianFactor = isGaussian<F>;

/// \cond INTERNAL_SPECIALIZATIONS
template <typename F, typename BlockSpace>
inline constexpr bool isGaussianFactor<EachRepetition<F, BlockSpace>>{isGaussian<F>};

/// \endcond

/** uniformComponents() of a Product, with the count of the \a j th gridded coordinate, in the
 * order of the space's coordinates, read from \a countOf(j). \a givenCounts is the number of
 * counts given, or -1 for one count for every gridded coordinate.
 */
template <Space S, RealScalar Scalar, typename... Factors, typename CountOf>
[[nodiscard]] UniformGrid<S, Scalar>
productGrid(
    Product<S, Scalar, Factors...> const & product,
    CountOf const & countOf,
    Eigen::Index givenCounts,
    Scalar relativeWidth
)
{
    static_assert(
        ((isUniform<Factors> || isGaussianFactor<Factors>) && ...),
        "uniformComponents() of a Product takes Uniform factors placed with on<Path>() and GaussianParams factors"
    );
    static_assert((isUniform<Factors> || ...), "uniformComponents() of a Product needs a Uniform factor to grid");
    using Grid = UniformGrid<S, Scalar>;
    using Index = Eigen::Index;
    Index const n = product.valueDimension();
    auto corner = typename Grid::VectorType(n);
    typename Grid::MatrixType covariance = Grid::MatrixType::Zero(n, n);
    typename Grid::VectorType width = Grid::VectorType::Zero(n);
    typename Grid::CountsType counts = Grid::CountsType::Zero(n);
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        (
            [&] {
                auto const & factor = product.template factor<K>();
                auto const & indices = product.template indices<K>();
                if constexpr (isUniform<std::tuple_element_t<K, std::tuple<Factors...>>>)
                {
                    corner(indices) = factor.lower();
                    width(indices) = factor.width();
                    counts(indices).setOnes();
                }
                else
                {
                    auto const & moments = factor.moments();
                    using FactorMatrix = std::remove_cvref_t<decltype(moments.covarianceLower())>;
                    corner(indices) = moments.mean();
                    covariance(indices, indices) =
                        FactorMatrix{moments.covarianceLower().template selfadjointView<Eigen::Lower>()};
                }
            }(),
            ...
        );
    }(std::index_sequence_for<Factors...>{});

    Index gridded = 0;
    for (Index i = 0; i < n; ++i)
    {
        if (counts(i) > 0)
        {
            counts(i) = countOf(gridded++);
            EMRYS_FILTER_ASSERT(counts(i) >= 1 && "each gridded coordinate needs at least one component");
        }
    }
    EMRYS_FILTER_ASSERT(
        (givenCounts < 0 || givenCounts == gridded) && "give one count per coordinate of the Uniform factors"
    );
    return Grid{
        product.space(),
        std::move(corner),
        std::move(width),
        std::move(counts),
        std::move(covariance),
        relativeWidth
    };
}

}  // namespace internal

/** \ingroup Distributions
 * \returns the components that approximate \a uniform, one at a time: \a counts(i) along
 * coordinate \a i, \f$ \prod_i N_i \f$ in all, with equal weights, which
 * GaussianMixtureParams::assign() puts in a mixture.
 *
 * Coordinate \f$ i \f$'s interval, from \f$ \ell_i \f$ with width \f$ w_i \f$, is cut into
 * \f$ N_i \f$ cells of width \f$ s_i = w_i / N_i \f$, and a component sits at the center of each
 * cell, \f$ \ell_i + (j + \tfrac12) s_i \f$, normalized in the mixture's space. Every component
 * has the covariance \f$ \mathrm{diag}(c^2 s_i^2) \f$, where \f$ c \f$ is \a relativeWidth.
 *
 * The width trades flatness against moments. By
 * [Poisson summation](https://en.wikipedia.org/wiki/Poisson_summation_formula), along one
 * coordinate the components sum to \f$ 1 / w \f$ times
 * \f$ 1 + 2 \sum_{m \ge 1} e^{-2 \pi^2 c^2 m^2} \cos(2 \pi m u) \f$, where \f$ u \f$ is the
 * offset from a center in cells, so the relative ripple inside the box is about
 * \f$ \pm 2 e^{-2 \pi^2 c^2} \f$ per coordinate:
 *
 * | \f$ c \f$ | ripple | |
 * |---|---|---|
 * | \f$ 1 / \sqrt{12} \approx 0.29 \f$ | \f$ \pm 39\% \f$ | matches the moments |
 * | 0.5 | \f$ \pm 1.4\% \f$ | |
 * | 0.6 | \f$ \pm 0.16\% \f$ | the default |
 * | 0.7 | \f$ \pm 0.01\% \f$ | |
 *
 * With \f$ c = 1 / \sqrt{12} \f$ the mixture's mean and covariance are the box's: the centers
 * spread \f$ (N_i^2 - 1) s_i^2 / 12 \f$ and each component adds \f$ s_i^2 / 12 \f$, together
 * \f$ w_i^2 / 12 \f$. A wider component flattens the density but adds \f$ c^2 s_i^2 \f$ to the
 * variance in place of \f$ s_i^2 / 12 \f$. Near a face of the box the density falls to about half
 * its interior value over a distance of a few \f$ c s_i \f$, whatever the width.
 *
 * A Periodic coordinate's interval over a whole period has no faces: the components wrap round
 * and the density only ripples. Each component's density is not wrapped, though (a %Gaussian in
 * the tangent space), so keep at least six components per period, which leaves every component's
 * mass beyond half a period below \f$ 10^{-6} \f$ at the default width.
 *
 * The components are in \a uniform's space.
 *
 * \sa UniformGrid, uniformComponents(Uniform const &, Eigen::Index, Scalar),
 *     uniformComponents(Product const &, Eigen::DenseBase const &, Scalar), Uniform
 */
template <Space S, RealScalar Scalar, typename CountsDerived>
    requires std::integral<typename CountsDerived::Scalar>
[[nodiscard]] UniformGrid<S, Scalar>
uniformComponents(
    Uniform<S, Scalar> const & uniform,
    Eigen::DenseBase<CountsDerived> const & counts,
    std::type_identity_t<Scalar> relativeWidth = Scalar(0.6)
)
{
    EMRYS_FILTER_ASSERT(counts.size() == uniform.valueDimension() && "give one count per coordinate");
    return internal::uniformGrid(
        uniform,
        [&](Eigen::Index i) {
            return Eigen::Index(counts(i));
        },
        relativeWidth
    );
}

/** \ingroup Distributions
 * uniformComponents() with \a count components along every coordinate.
 */
template <Space S, RealScalar Scalar>
[[nodiscard]] UniformGrid<S, Scalar>
uniformComponents(
    Uniform<S, Scalar> const & uniform,
    Eigen::Index count,
    std::type_identity_t<Scalar> relativeWidth = Scalar(0.6)
)
{
    return internal::uniformGrid(
        uniform,
        [&](Eigen::Index) {
            return count;
        },
        relativeWidth
    );
}

/** \ingroup Distributions
 * \returns the components, one at a time, of a mixture in \a product's space that is uniform over
 * the blocks of its Uniform factors and %Gaussian over the rest: the Uniform factors' coordinates
 * gridded as uniformComponents(Uniform const &, Eigen::DenseBase const &, Scalar) grids a
 * Uniform's, and every component given each %Gaussian factor's mean and covariance on its block.
 *
 * A state that starts somewhere in a box with a velocity known to a %Gaussian:
 * \code
 * auto prior = GaussianMixtureParams<State>{};
 * prior.assign(uniformComponents(Product{State{}, on<Position>(box), on<Velocity>(velocity)}, Eigen::Array2i{8, 8}));
 * \endcode
 *
 * The factors are Uniform factors placed with on<Path>(), and GaussianParams factors placed with
 * on<Path>() or onEach<Path>(). \a counts holds one count per coordinate of the Uniform factors,
 * in the order of the space's coordinates, and the first of them varies fastest. The blocks are
 * independent in every component, as they are in \a product.
 *
 * \sa uniformComponents(Product const &, Eigen::Index, Scalar), UniformGrid, Product
 */
template <Space S, RealScalar Scalar, typename... Factors, typename CountsDerived>
    requires std::integral<typename CountsDerived::Scalar>
[[nodiscard]] UniformGrid<S, Scalar>
uniformComponents(
    Product<S, Scalar, Factors...> const & product,
    Eigen::DenseBase<CountsDerived> const & counts,
    std::type_identity_t<Scalar> relativeWidth = Scalar(0.6)
)
{
    return internal::productGrid(
        product,
        [&](Eigen::Index j) {
            EMRYS_FILTER_ASSERT(j < counts.size() && "give one count per coordinate of the Uniform factors");
            return Eigen::Index(counts(j));
        },
        counts.size(),
        relativeWidth
    );
}

/** \ingroup Distributions
 * uniformComponents() of a Product with \a count components along every coordinate of its
 * Uniform factors.
 */
template <Space S, RealScalar Scalar, typename... Factors>
[[nodiscard]] UniformGrid<S, Scalar>
uniformComponents(
    Product<S, Scalar, Factors...> const & product,
    Eigen::Index count,
    std::type_identity_t<Scalar> relativeWidth = Scalar(0.6)
)
{
    return internal::productGrid(
        product,
        [&](Eigen::Index) {
            return count;
        },
        -1,
        relativeWidth
    );
}

}  // namespace Emrys

// uniformComponents() (step 12c, [uniform-mixtures]): the grid of components over a Uniform's
// box, given one at a time to fill a mixture made elsewhere without allocating, their
// equal weights and shared covariance, the moments it matches at the narrow width, and its
// density against the Poisson-summation closed form, inside a linear box, at a face, and round a
// whole period with no face at all, alone and in a composite space; and the mixture of a Product,
// its Uniform factors gridded and its Gaussian factors given to every component. Header under test:
// Distributions/UniformMixture.h.

#include <EmrysFilter/Distributions/UniformMixture.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory_resource>
#include <numbers>
#include <ranges>
#include <type_traits>

using Emrys::Composite;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::on;
using Emrys::RadiansPi;
using Emrys::Uniform;
using Emrys::uniformComponents;

namespace
{

constexpr double pi = std::numbers::pi;

struct Heading : RadiansPi<>
{};

using Vector1 = Eigen::Matrix<double, 1, 1>;

/** The density of a whole lattice of components, \a count over a width \a width starting at
 * \a lower, each with deviation \a relativeWidth times the spacing, at \a x, by Poisson summation.
 */
double
latticeDensity(double x, double lower, double width, int count, double relativeWidth)
{
    double const spacing = width / count;
    double const u = (x - lower) / spacing - 0.5;
    double sum = 1;
    for (int m = 1; m <= 8; ++m)
    {
        sum += 2 * std::exp(-2 * pi * pi * relativeWidth * relativeWidth * m * m) * std::cos(2 * pi * m * u);
    }
    return sum / width;
}

/** A new mixture in \a grid's space that holds its components. */
template <typename Grid>
GaussianMixtureParams<typename Grid::SpaceType>
mixtureOf(Grid const & grid)
{
    auto mixture = GaussianMixtureParams<typename Grid::SpaceType>{grid.space()};
    mixture.assign(grid);
    return mixture;
}

}  // namespace

// The components sit at the cell centers with the first coordinate varying fastest, have equal
// weights summing to one, and share the covariance diag((c s)^2), here c = 0.6 and s = (0.5, 2).
TEST(UniformMixture, ComponentsSitAtTheCellCenters)
{
    auto const uniform = Uniform<Linear<2>>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    auto const mixture = mixtureOf(uniformComponents(uniform, Eigen::Array2i{4, 2}));
    ASSERT_EQ(mixture.size(), 8);
    Eigen::Index k = 0;
    for (auto const & component : mixture)
    {
        auto const center = Eigen::Vector2d{-1.0 + 0.5 * (double(k % 4) + 0.5), 2.0 * (double(k / 4) + 0.5)};
        EXPECT_TRUE(component.component.mean().isApprox(center, 1e-15)) << k;
        EXPECT_NEAR(component.logWeight, -std::log(8.0), 1e-15);
        Eigen::Matrix2d const cov = Eigen::Vector2d{0.3 * 0.3, 1.2 * 1.2}.asDiagonal();
        EXPECT_TRUE(component.component.covarianceLower().isApprox(cov, 1e-15)) << k;
        ++k;
    }
}

// One count for every coordinate gives the same components as a count per coordinate.
TEST(UniformMixture, OneCount)
{
    auto const uniform = Uniform<Linear<2>>::create(Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{3.0, 6.0}).value();
    auto const each = uniformComponents(uniform, Eigen::Array2i{3, 3}, 0.5);
    auto const all = uniformComponents(uniform, 3, 0.5);
    ASSERT_EQ(all.size(), 9);
    ASSERT_EQ(all.size(), each.size());
    for (Eigen::Index k = 0; k < all.size(); ++k)
    {
        EXPECT_EQ(all[k].logWeight, each[k].logWeight);
        EXPECT_EQ(all[k].moments.mean(), each[k].moments.mean());
        EXPECT_EQ(all[k].moments.covarianceLower(), each[k].moments.covarianceLower());
    }
}

namespace
{

/** A memory resource that counts the allocations it passes on to the default resource. */
class CountingResource : public std::pmr::memory_resource
{
public:

    int allocations = 0;  ///< the allocations so far

private:

    void *
    do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        ++allocations;
        return std::pmr::get_default_resource()->allocate(bytes, alignment);
    }

    void
    do_deallocate(void * p, std::size_t bytes, std::size_t alignment) override
    {
        std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
    }

    [[nodiscard]] bool
    do_is_equal(std::pmr::memory_resource const & other) const noexcept override
    {
        return this == &other;
    }
};

}  // namespace

// uniformComponents() gives the components one at a time, as a sized random-access range.
// assign() fills a mixture made elsewhere with a capacity and an allocator from them without
// allocating, and fills it again the same way.
TEST(UniformMixture, ComponentsFillAMixtureMadeElsewhere)
{
    auto const uniform = Uniform<Linear<2>>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    auto const grid = uniformComponents(uniform, Eigen::Array2i{4, 2});
    static_assert(std::ranges::random_access_range<decltype(grid)>);
    static_assert(std::ranges::sized_range<decltype(grid)>);
    static_assert(std::is_same_v<decltype(grid), Emrys::UniformGrid<Linear<2>> const>);
    ASSERT_EQ(grid.size(), 8);
    EXPECT_EQ(grid.counts(), Eigen::Vector2<Eigen::Index>(4, 2));

    auto resource = CountingResource{};
    using Pmr = std::pmr::polymorphic_allocator<>;
    auto filled = GaussianMixtureParams<Linear<2>, double, Pmr>{Emrys::Capacity{16}, Pmr{&resource}};
    int const made = resource.allocations;
    filled.assign(grid);
    filled.assign(uniformComponents(uniform, 4));
    EXPECT_EQ(resource.allocations, made);
    EXPECT_EQ(filled.get_allocator().resource(), &resource);
    ASSERT_EQ(filled.size(), 16);
    EXPECT_EQ(filled.begin()->component.mean(), Eigen::Vector2d(-0.75, 0.5));
}

// At c = 1/sqrt(12) the centers' spread, (N^2 - 1) s^2 / 12, plus each component's s^2 / 12 is
// w^2 / 12, so the mixture's moments are the box's; the default width adds more variance.
TEST(UniformMixture, NarrowWidthMatchesTheMoments)
{
    auto const uniform = Uniform<Linear<2>>::create(Eigen::Vector2d{-1.0, 0.0}, Eigen::Vector2d{1.0, 4.0}).value();
    auto const expected = uniform.moments();

    auto const matched = mixtureOf(uniformComponents(uniform, Eigen::Array2i{5, 3}, 1 / std::sqrt(12.0)));
    auto const moments = matched.toDensity().value().moments();
    EXPECT_TRUE(moments.mean().isApprox(expected.mean(), 1e-14));
    EXPECT_TRUE(moments.covarianceLower().isApprox(expected.covarianceLower(), 1e-14));

    auto const flat = mixtureOf(uniformComponents(uniform, Eigen::Array2i{5, 3}));
    auto const spacing = Eigen::Vector2d{0.4, 4.0 / 3.0};
    Eigen::Vector2d const variance =
        expected.covarianceLower().diagonal() + ((0.36 - 1.0 / 12) * spacing.array().square()).matrix();
    EXPECT_TRUE(flat.toDensity().value().moments().covarianceLower().diagonal().isApprox(variance, 1e-14));
}

// Inside a linear interval of 40 cells, more than eight cells (13 deviations) from either face,
// the density is the whole lattice's closed form to 1e-12, within the documented 0.16% of 1/w at
// the default width and 39% at the narrow one (its second harmonic adds 0.3%). At a face it is half the lattice's
// density there, since the components beyond the face are missing and the lattice is symmetric about it.
TEST(UniformMixture, DensityRipplesAsPoissonSummationSays)
{
    auto const uniform = Uniform<Linear<1>>::create(Vector1{0.0}, Vector1{40.0}).value();
    for (double const c : {0.6, 1 / std::sqrt(12.0)})
    {
        auto const mixture = mixtureOf(uniformComponents(uniform, 40, c));
        auto const density = mixture.toDensity().value();
        double largest = 0;
        for (double x = 10.0; x <= 30.0; x += 0.05)
        {
            double const expected = latticeDensity(x, 0.0, 40.0, 40, c);
            EXPECT_NEAR(density.density(Vector1{x}), expected, 1e-12 * expected) << x;
            largest = std::max(largest, std::abs(40 * density.density(Vector1{x}) - 1));
        }
        EXPECT_LT(largest, c == 0.6 ? 0.0017 : 0.40);
        EXPECT_GT(largest, c == 0.6 ? 0.0015 : 0.38);
        EXPECT_NEAR(density.density(Vector1{0.0}), 0.5 * latticeDensity(0.0, 0.0, 40.0, 40, c), 1e-12);
    }
}

// Round a whole period the components wrap and leave no face: with 12 per period, half a period
// is ten deviations, so the unwrapped components lose nothing and the density is the lattice's
// closed form everywhere, across the seam at +-pi included.
TEST(UniformMixture, WholePeriodHasNoFace)
{
    auto const heading = Uniform<Heading>{};
    auto const mixture = mixtureOf(uniformComponents(heading, 12));
    auto const density = mixture.toDensity().value();
    for (double x = -pi; x < pi; x += 0.01)
    {
        double const expected = latticeDensity(x, -pi, 2 * pi, 12, 0.6);
        EXPECT_NEAR(density.density(Vector1{x}), expected, 1e-12 * expected) << x;
    }
    EXPECT_NEAR(density.density(Vector1{pi}), density.density(Vector1{-pi}), 1e-15);
}

// In a heading-by-distance box the density is the product of the two coordinates' closed forms,
// near the seam and inside the distance interval.
TEST(UniformMixture, CompositeIsTheProductOfItsCoordinates)
{
    using Space = Composite<Heading, Linear<1>>;
    auto const uniform = Uniform<Space>::create(Space{}, Eigen::Vector2d{-pi, 0.0}, Eigen::Vector2d{pi, 20.0}).value();
    auto const mixture = mixtureOf(uniformComponents(uniform, Eigen::Array2i{12, 20}));
    auto const density = mixture.toDensity().value();
    for (Eigen::Vector2d const & x :
         {Eigen::Vector2d{3.1, 9.7}, Eigen::Vector2d{-3.1, 10.0}, Eigen::Vector2d{0.2, 6.25}})
    {
        double const expected = latticeDensity(x(0), -pi, 2 * pi, 12, 0.6) * latticeDensity(x(1), 0.0, 20.0, 20, 0.6);
        EXPECT_NEAR(density.density(x), expected, 1e-12 * expected) << x.transpose();
    }
}

namespace
{

struct Position : Linear<2>
{};

struct Speed : Linear<1>
{};

struct Landmark : Linear<2>
{};

struct Landmarks : Emrys::Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using State = Composite<Position, Heading, Speed>;

GaussianParams<Position>
makePosition()
{
    return GaussianParams<Position>{Eigen::Vector2d{10.0, -5.0}, Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}}};
}

}  // namespace

// A Product's Uniform factors are gridded with their counts in the order of the space's
// coordinates, and every component has the Gaussian factor's mean and covariance on its block, with
// no covariance between blocks: the gridded rows are the mixture of the heading-by-speed box.
TEST(UniformMixture, ProductGridsItsUniformsAndKeepsItsGaussians)
{
    auto const speed = Uniform<Speed>::create(Vector1{0.0}, Vector1{10.0}).value();
    auto const prior =
        Emrys::Product{State{}, on<Position>(makePosition()), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)};
    auto const components = uniformComponents(prior, Eigen::Array2i{12, 5});
    static_assert(std::is_same_v<decltype(components), Emrys::UniformGrid<State> const>);
    EXPECT_EQ(components.counts(), (Eigen::Vector4<Eigen::Index>{0, 0, 12, 5}));
    auto const mixture = mixtureOf(components);
    ASSERT_EQ(mixture.size(), 60);

    using Box = Composite<Heading, Speed>;
    auto const box = Uniform<Box>::create(Box{}, Eigen::Vector2d{-pi, 0.0}, Eigen::Vector2d{pi, 10.0}).value();
    auto const grid = mixtureOf(uniformComponents(box, Eigen::Array2i{12, 5}));
    ASSERT_EQ(grid.size(), mixture.size());
    Eigen::Matrix4d expectedCov = Eigen::Matrix4d::Zero();
    expectedCov.topLeftCorner<2, 2>() = Eigen::Matrix2d{{4.0, 1.0}, {1.0, 2.0}};
    for (Eigen::Index k = 0; k < mixture.size(); ++k)
    {
        auto const & component = mixture[k].component;
        auto const & gridded = grid[k].component;
        EXPECT_NEAR(mixture[k].logWeight, -std::log(60.0), 1e-15);
        EXPECT_EQ(component.mean().head<2>(), Eigen::Vector2d(10.0, -5.0)) << k;
        EXPECT_TRUE(component.mean().tail<2>().isApprox(gridded.mean(), 1e-15)) << k;
        expectedCov.bottomRightCorner<2, 2>() = gridded.covarianceLower();
        Eigen::Matrix4d const cov = component.covarianceLower().selfadjointView<Eigen::Lower>();
        EXPECT_TRUE(cov.isApprox(expectedCov, 1e-15)) << k;
    }

    auto const positionParams = makePosition();
    auto const position = positionParams.toDensity().value();
    auto const gridDensity = grid.toDensity().value();
    auto const density = mixture.toDensity().value();
    for (Eigen::Vector4d const & x : {Eigen::Vector4d{11.0, -4.0, 3.1, 9.7}, Eigen::Vector4d{9.0, -6.5, -0.2, 0.3}})
    {
        double const expected = position.logDensity(x.head<2>()) + gridDensity.logDensity(x.tail<2>());
        EXPECT_NEAR(density.logDensity(x), expected, 1e-12) << x.transpose();
    }
}

// The counts follow the space's coordinates, not the order the factors were given in, and one
// count grids every coordinate of the Uniform factors.
TEST(UniformMixture, ProductCountsFollowTheSpace)
{
    auto const speed = Uniform<Speed>::create(Vector1{0.0}, Vector1{10.0}).value();
    auto const inOrder = mixtureOf(uniformComponents(
        Emrys::Product{State{}, on<Position>(makePosition()), on<Heading>(Uniform<Heading>{}), on<Speed>(speed)},
        Eigen::Array2i{6, 6}
    ));
    auto const reversed = mixtureOf(uniformComponents(
        Emrys::Product{State{}, on<Speed>(speed), on<Heading>(Uniform<Heading>{}), on<Position>(makePosition())},
        6
    ));
    ASSERT_EQ(reversed.size(), 36);
    for (Eigen::Index k = 0; k < reversed.size(); ++k)
    {
        EXPECT_EQ(reversed[k].component.mean(), inOrder[k].component.mean());
        EXPECT_EQ(reversed[k].component.covarianceLower(), inOrder[k].component.covarianceLower());
    }
    EXPECT_NEAR(reversed[1].component.mean()(2) - reversed.begin()->component.mean()(2), pi / 3, 1e-15);
}

// A Gaussian placed with onEach() gives every repetition its mean and covariance, and the position
// is gridded alone.
TEST(UniformMixture, ProductWithOnEach)
{
    using Map = Composite<Position, Landmarks>;
    auto const space = Map{Position{}, Landmarks{Landmark{}, 2}};
    auto const box = Uniform<Position>::create(Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{40.0, 20.0}).value();
    auto const landmark = GaussianParams<Landmark>{Eigen::Vector2d{3.0, 1.0}, Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}}};
    auto const mixture = mixtureOf(uniformComponents(
        Emrys::Product{space, on<Position>(box), Emrys::onEach<Landmarks>(landmark)},
        Eigen::Array2i{4, 2}
    ));
    ASSERT_EQ(mixture.size(), 8);
    Eigen::Matrix<double, 6, 6> expectedCov = Eigen::Matrix<double, 6, 6>::Zero();
    expectedCov.diagonal().head<2>() = Eigen::Vector2d{6.0 * 6.0, 6.0 * 6.0};
    for (int r = 0; r < 2; ++r)
    {
        expectedCov.block<2, 2>(2 + 2 * r, 2 + 2 * r) = Eigen::Matrix2d{{2.0, 0.5}, {0.5, 1.0}};
    }
    Eigen::Index k = 0;
    for (auto const & entry : mixture)
    {
        Eigen::Matrix<double, 6, 1> expectedMean;
        expectedMean << 10.0 * (double(k % 4) + 0.5), 10.0 * (double(k / 4) + 0.5), 3.0, 1.0, 3.0, 1.0;
        EXPECT_TRUE(entry.component.mean().isApprox(expectedMean, 1e-15)) << k;
        Eigen::Matrix<double, 6, 6> const cov = entry.component.covarianceLower().selfadjointView<Eigen::Lower>();
        EXPECT_TRUE(cov.isApprox(expectedCov, 1e-15)) << k;
        ++k;
    }
}

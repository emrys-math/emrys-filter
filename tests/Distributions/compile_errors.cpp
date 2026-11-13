// The compile errors that the distributions claim (Distributions/*.h, and the paths of
// Spaces/Subspace.h and Discrete.h that only a distribution reaches): each case below is a misuse
// that must fail to compile with the static_assert or deleted function it names, checked by
// tests/compile_errors.cmake. Without a case the file compiles, which checks what the cases share.
// The evidence for [coverage]'s negative compile tests, for the deleted marginals and densities of
// temporaries ([element-marginals]), and for sampling only in a standard floating-point scalar.

#include <EmrysFilter/Distributions.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include <array>
#include <random>
#include <utility>

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<>
{};

struct Bearing : Emrys::RadiansPi<>
{};

struct Speed : Emrys::Linear<1>
{};

struct Attitude : Emrys::UnitQuaternion
{};

/// a resizable block
struct Landmarks : Emrys::Linear<Eigen::Dynamic, 8>
{};

struct Fleet : Emrys::Repeated<Position>
{
    using Repeated::Repeated;
};

struct Pose : Emrys::Composite<Position, Heading>
{};

struct Lead : Pose
{};

struct Angles : Emrys::Composite<Heading, Bearing>
{};

struct East : Emrys::Linear<1>
{};

struct North : Emrys::Linear<1>
{};

struct Place : Emrys::Composite<East, North>
{};

struct Wing : Pose
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

using Mode = Emrys::DiscreteProduct<Maneuver, Sensor>;
using State = Emrys::Composite<Position, Heading, Speed>;
using Convoy = Emrys::Composite<Position, Fleet>;
using Dual = Eigen::AutoDiffScalar<Eigen::Vector2d>;

[[maybe_unused]] std::mt19937 rng{1};

[[maybe_unused]] Emrys::GaussianParams<Position>
makePosition()
{
    return Emrys::GaussianParams<Position>{Eigen::Vector2d{1.0, 2.0}, Eigen::Matrix2d::Identity()};
}

[[maybe_unused]] Emrys::GaussianParams<Convoy>
makeConvoy()
{
    return Emrys::GaussianParams<Convoy>{
        Convoy{Position{}, Fleet{Position{}, 1}},
        Eigen::Vector4d::Zero(),
        Eigen::Matrix4d::Identity()
    };
}

[[maybe_unused]] auto
makeProduct()
{
    static auto const params = makePosition();
    static auto const position = params.toDensity().value();
    return Emrys::Product{
        State{},
        Emrys::on<Position>(position),
        Emrys::on<Heading>(Emrys::Uniform<Heading>{}),
        Emrys::on<Speed>(Emrys::Gamma<Speed>::create(2.0, 1.0).value())
    };
}

[[maybe_unused]] auto
makeHybrid()
{
    return Emrys::HybridParams{
        Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.5, 0.5}),
        makeConvoy()
    };
}

[[maybe_unused]] auto
makeCategorical()
{
    return Emrys::CategoricalParams<Mode>::fromWeights(Mode{}, Eigen::Vector4d{1.0, 2.0, 3.0, 4.0});
}

[[maybe_unused]] void
everyCaseSharesThis()
{
    [[maybe_unused]] auto const product = makeProduct();
    [[maybe_unused]] auto const hybrid = makeHybrid();
    [[maybe_unused]] auto const conditional = makeConvoy().conditional<Fleet>(Eigen::Vector2d::Zero());
    [[maybe_unused]] auto const maneuver = makeCategorical().marginal<Maneuver>();
}

// Sampling draws through the standard library's distributions, which take float, double, or long
// double, so a distribution in an automatic-differentiation scalar evaluates but does not sample.

#if defined(GAUSSIAN_SAMPLE_IN_DUAL)  // static_assert: Sampling.h: sampling requires a standard floating-point scalar
[[maybe_unused]] void
drawGaussian()
{
    auto const params =
        Emrys::GaussianParams<Position, Dual>{Eigen::Vector<Dual, 2>::Zero(), Eigen::Matrix<Dual, 2, 2>::Identity()};
    [[maybe_unused]] auto const x = Emrys::drawSample(params.toDensity().value(), rng);
}
#endif

#if defined(GAMMA_SAMPLES_IN_DUAL)  // static_assert: Gamma.h: sampling requires a standard floating-point scalar
[[maybe_unused]] void
drawGamma()
{
    auto const gamma =
        Emrys::Gamma<Speed, Dual>::create(Eigen::Vector<Dual, 1>{Dual{2.0}}, Eigen::Vector<Dual, 1>{Dual{1.0}}).value();
    [[maybe_unused]] auto const x = Emrys::drawSamples(gamma, 3, rng);
}
#endif

#if defined(UNIFORM_SAMPLE_IN_DUAL)  // static_assert: Uniform.h: sampling requires a standard floating-point scalar
[[maybe_unused]] void
drawUniform()
{
    auto const box =
        Emrys::Uniform<Position, Dual>::create(Eigen::Vector<Dual, 2>::Zero(), Eigen::Vector<Dual, 2>::Ones()).value();
    [[maybe_unused]] auto const x = Emrys::drawSample(box, rng);
}
#endif

// fromWeights() with the weights as arguments takes one per value: two values, three weights.

#if defined(CATEGORICAL_WEIGHTS_MISCOUNTED)  // static_assert: Categorical.h: give one weight per value
[[maybe_unused]] void
miscountWeights()
{
    [[maybe_unused]] auto const params = Emrys::CategoricalParams<Maneuver>::fromWeights(0.5, 0.25, 0.25);
}
#endif

#if defined(CATEGORICAL_SAMPLES_IN_DUAL)  // static_assert: Categorical.h: sampling requires a standard floating-point
[[maybe_unused]] void
drawCategoricals()
{
    auto const params =
        Emrys::CategoricalParams<Maneuver, Dual>::fromWeights(Maneuver{}, Eigen::Vector<Dual, 2>{Dual{1.0}, Dual{1.0}});
    [[maybe_unused]] auto const x = Emrys::drawSamples(params.toDensity().value(), 3, rng);
}
#endif

#if defined(CATEGORICAL_SAMPLE_IN_DUAL)  // static_assert: Weights.h: sampling requires a standard floating-point scalar
[[maybe_unused]] void
drawCategorical()
{
    auto const params =
        Emrys::CategoricalParams<Maneuver, Dual>::fromWeights(Maneuver{}, Eigen::Vector<Dual, 2>{Dual{1.0}, Dual{1.0}});
    [[maybe_unused]] auto const x = Emrys::drawSample(params.toDensity().value(), rng);
}
#endif

#if defined(ROUGHEN_IN_DUAL)  // static_assert: Upkeep.h: sampling requires a standard floating-point scalar
[[maybe_unused]] void
roughen()
{
    auto particles = Emrys::ParticleSet<Position, Dual>{Eigen::Matrix<Dual, 2, 3>::Zero()};
    [[maybe_unused]] auto const result = Emrys::upkeep::roughen(rng)(particles);
}
#endif

// A view of a temporary would dangle, so marginal() and toDensity() of an rvalue are deleted.

#if defined(GAUSSIAN_MARGINAL_OF_TEMPORARY)  // deleted: Gaussian.h: marginal() const &&
[[maybe_unused]] auto const view = makeConvoy().marginal<Position>();
#endif

#if defined(GAUSSIAN_ELEMENT_MARGINAL_OF_TEMPORARY)  // deleted: Gaussian.h: marginal(Index element) const &&
[[maybe_unused]] auto const view = makeConvoy().marginal<Fleet>(0);
#endif

#if defined(GAUSSIAN_DENSITY_OF_TEMPORARY)  // deleted: Gaussian.h: toDensity() const &&
[[maybe_unused]] auto const density = makePosition().toDensity();
#endif

#if defined(GAUSSIAN_REPAIRED_DENSITY_OF_TEMPORARY)  // deleted: Gaussian.h: toDensity(R const &) &&
[[maybe_unused]] auto const density = makePosition().toDensity(Emrys::roots::ridge());
#endif

#if defined(GAUSSIAN_DENSITY_GIVEN_AXES)  // static_assert: Gaussian.h: a density takes a repair
[[maybe_unused]] void
gaussianDensityGivenAxes()
{
    auto params = makePosition();
    [[maybe_unused]] auto const density = params.toDensity(Emrys::roots::pivoted());
}
#endif

#if defined(MIXTURE_MARGINAL_OF_TEMPORARY)  // deleted: Mixture.h: marginal() const &&
[[maybe_unused]] auto const view = Emrys::GaussianMixtureParams<Convoy>{}.marginal<Position>();
#endif

#if defined(MIXTURE_ELEMENT_MARGINAL_OF_TEMPORARY)  // deleted: Mixture.h: marginal(Index element) const &&
[[maybe_unused]] auto const view = Emrys::GaussianMixtureParams<Convoy>{}.marginal<Fleet>(0);
#endif

#if defined(MIXTURE_DENSITY_OF_TEMPORARY)  // deleted: Mixture.h: toDensity() const &&
[[maybe_unused]] auto const density = Emrys::GaussianMixtureParams<Position>{}.toDensity();
#endif

#if defined(MIXTURE_REPAIRED_DENSITY_OF_TEMPORARY)  // deleted: Mixture.h: toDensity(R const &) &&
[[maybe_unused]] auto const density = Emrys::GaussianMixtureParams<Position>{}.toDensity(Emrys::roots::ridge());
#endif

#if defined(PARTICLE_MARGINAL_OF_TEMPORARY)  // deleted: ParticleSet.h: marginal() const &&
[[maybe_unused]] auto const view = Emrys::ParticleSet<Convoy>{}.marginal<Position>();
#endif

#if defined(PARTICLE_ELEMENT_MARGINAL_OF_TEMPORARY)  // deleted: ParticleSet.h: marginal(Index element) const &&
[[maybe_unused]] auto const view = Emrys::ParticleSet<Convoy>{}.marginal<Fleet>(0);
#endif

#if defined(PARTICLE_DENSITY_OF_TEMPORARY)  // deleted: ParticleSet.h: toDensity() const &&
[[maybe_unused]] auto const density = Emrys::ParticleSet<Position>{}.toDensity();
#endif

#if defined(KERNEL_DENSITY_OF_TEMPORARY)  // deleted: ParticleSet.h: toDensity(Bandwidth && bandwidth) const &&
[[maybe_unused]] auto const density = Emrys::ParticleSet<Position>{}.toDensity(Eigen::Matrix2d::Identity());
#endif

#if defined(HYBRID_CONDITIONAL_OF_TEMPORARY)  // deleted: Hybrid.h: conditional(Eigen::Index mode) const &&
[[maybe_unused]] auto const & mode = makeHybrid().conditional(0);
#endif

#if defined(HYBRID_NAMED_OF_TEMPORARY)  // deleted: Hybrid.h: conditional(AxisIndex<Paths, N> const &...) const &&
[[maybe_unused]] auto const & mode = makeHybrid().conditional<Turn>();
#endif

#if defined(HYBRID_MARGINAL_OF_TEMPORARY)  // deleted: Hybrid.h: marginal() const &&
[[maybe_unused]] auto const view = makeHybrid().marginal<Position>();
#endif

#if defined(HYBRID_ELEMENT_MARGINAL_OF_TEMPORARY)  // deleted: Hybrid.h: marginal(Eigen::Index element) const &&
[[maybe_unused]] auto const view = makeHybrid().marginal<Fleet>(0);
#endif

#if defined(HYBRID_DENSITY_OF_TEMPORARY)  // deleted: Hybrid.h: toDensity() const &&
[[maybe_unused]] auto const density = makeHybrid().toDensity();
#endif

#if defined(HYBRID_REPAIRED_DENSITY_OF_TEMPORARY)  // deleted: Hybrid.h: toDensity(R const &) &&
[[maybe_unused]] auto const density = makeHybrid().toDensity(Emrys::roots::ridge());
#endif

#if defined(PRODUCT_MARGINAL_OF_TEMPORARY)  // deleted: Product.h: marginal() const &&
[[maybe_unused]] auto const & view = makeProduct().marginal<Position>();
#endif

#if defined(CATEGORICAL_DENSITY_OF_TEMPORARY)  // deleted: Categorical.h: toDensity() const &&
[[maybe_unused]] auto const density = makeCategorical().toDensity();
#endif

#if defined(INFORMATION_DENSITY_OF_TEMPORARY)  // deleted: InfoGaussian.h: toDensity() const &&
[[maybe_unused]] auto const density = Emrys::InfoGaussianParams<Position>{}.toDensity();
#endif

#if defined(SQUARE_ROOT_DENSITY_OF_TEMPORARY)  // deleted: SqrtGaussian.h: toDensity() const &&
[[maybe_unused]] auto const density = Emrys::SqrtGaussianParams<Position>{}.toDensity();
#endif

#if defined(SQUARE_ROOT_INFORMATION_DENSITY_OF_TEMPORARY)  // deleted: SqrtInfoGaussian.h: toDensity() const &&
[[maybe_unused]] auto const density = Emrys::SqrtInfoGaussianParams<Position>{}.toDensity();
#endif

// An entry or particle handed to a forEach() callback lives only inside it.

#if defined(MIXTURE_ENTRY_COPIED)  // deleted: Mixture.h: MutableEntry(MutableEntry const &)
[[maybe_unused]] void
copyEntry()
{
    auto mixture = Emrys::GaussianMixtureParams<Position>{};
    mixture.forEach([](auto & entry) {
        [[maybe_unused]] auto copy = entry;
    });
}
#endif

#if defined(MIXTURE_ENTRY_ASSIGNED)  // deleted: Mixture.h: operator=(MutableEntry const &)
[[maybe_unused]] void
assignEntry()
{
    auto mixture = Emrys::GaussianMixtureParams<Position>{};
    mixture.forEach([](auto & entry) {
        entry = std::as_const(entry);
    });
}
#endif

#if defined(PARTICLE_COPIED)  // deleted: ParticleSet.h: MutableParticle(MutableParticle const &)
[[maybe_unused]] void
copyParticle()
{
    auto particles = Emrys::ParticleSet<Position>{};
    particles.forEach([](auto & particle) {
        [[maybe_unused]] auto copy = particle;
    });
}
#endif

#if defined(PARTICLE_ASSIGNED)  // deleted: ParticleSet.h: operator=(MutableParticle const &)
[[maybe_unused]] void
assignParticle()
{
    auto particles = Emrys::ParticleSet<Position>{};
    particles.forEach([](auto & particle) {
        particle = std::as_const(particle);
    });
}
#endif

// A density evaluates one value at a time.

#if defined(GAUSSIAN_DISTANCE_OF_MATRIX)  // static_assert: Gaussian.h: squaredMahalanobisDistance() expects a vector
[[maybe_unused]] void
distanceOfMatrix()
{
    auto const params = makePosition();
    [[maybe_unused]] auto const d = params.toDensity()->squaredMahalanobisDistance(Eigen::Matrix2d::Identity());
}
#endif

#if defined(SQUARE_ROOT_DISTANCE_OF_MATRIX)  // static_assert: GaussianForm.h: squaredMahalanobisDistance() expects a
[[maybe_unused]] void
squareRootDistanceOfMatrix()
{
    auto const params = Emrys::SqrtGaussianParams<Position>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    [[maybe_unused]] auto const d = params.toDensity()->squaredMahalanobisDistance(Eigen::Matrix2d::Identity());
}
#endif

#if defined(GAMMA_DENSITY_OF_MATRIX)  // static_assert: Gamma.h: logDensity() expects a vector
[[maybe_unused]] auto const p = Emrys::Gamma<Position>::create(Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones())
                                    .value()
                                    .logDensity(Eigen::Matrix2d::Ones());
#endif

#if defined(MIXTURE_DENSITY_OF_MATRIX)  // static_assert: Mixture.h: logDensity() expects a vector
[[maybe_unused]] void
mixtureOfMatrix()
{
    auto const params = Emrys::GaussianMixtureParams<Position>{};
    [[maybe_unused]] auto const p = params.toDensity()->logDensity(Eigen::Matrix2d::Identity());
}
#endif

#if defined(KERNEL_DENSITY_OF_MATRIX)  // static_assert: KernelDensity.h: logDensity() expects a vector
[[maybe_unused]] void
kernelOfMatrix()
{
    auto const particles = Emrys::ParticleSet<Position>{Eigen::Matrix<double, 2, 3>::Zero()};
    [[maybe_unused]] auto const p =
        particles.toDensity(Eigen::Matrix2d::Identity())->logDensity(Eigen::Matrix2d::Identity());
}
#endif

#if defined(PRODUCT_DENSITY_OF_MATRIX)  // static_assert: Product.h: logDensity() expects a vector
[[maybe_unused]] auto const p = makeProduct().logDensity(Eigen::Matrix<double, 4, 2>::Zero());
#endif

#if defined(UNIFORM_CONTAINS_MATRIX)  // static_assert: Uniform.h: contains() expects a vector
[[maybe_unused]] auto const inside = Emrys::Uniform<Heading>{}.contains(Eigen::Matrix2d::Zero());
#endif

#if defined(UNIFORM_MIXTURE_PRODUCT_FACTOR)  // static_assert: UniformMixture.h: uniformComponents() of a Product takes
                                             // Uniform factors
[[maybe_unused]] auto const gridded = Emrys::uniformComponents(makeProduct(), 12);
#endif

#if defined(UNIFORM_MIXTURE_NO_UNIFORM)  // static_assert: UniformMixture.h: uniformComponents() of a Product needs
                                         // a Uniform factor to grid
[[maybe_unused]] auto const gridded = Emrys::uniformComponents(
    Emrys::Product{
        State{},
        Emrys::on<Position>(makePosition()),
        Emrys::on<Heading>(
            Emrys::GaussianParams<Heading>{Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}}
        ),
        Emrys::on<Speed>(
            Emrys::GaussianParams<Speed>{Eigen::Matrix<double, 1, 1>{5.0}, Eigen::Matrix<double, 1, 1>{1.0}}
        )
    },
    12
);
#endif

// Spaces a density cannot be on.

#if defined(UNIFORM_ON_QUATERNION)  // static_assert: Uniform.h: a uniform box needs a space whose values and
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Uniform<Attitude>);
#endif

#if defined(UNIFORM_WITHOUT_INTERVAL)  // static_assert: Uniform.h: a coordinate has no default interval; give the box
[[maybe_unused]] auto const box = Emrys::Uniform<Emrys::Composite<Heading, Speed>>{};
#endif

#if defined(GAMMA_ON_QUATERNION)  // static_assert: Gamma.h: a gamma distribution needs a space whose values and
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Gamma<Attitude>);
#endif

#if defined(GAMMA_ON_ANGLE)  // static_assert: Gamma.h: a gamma distribution needs every coordinate to contain [0, inf)
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Gamma<Heading>);
#endif

// Products.

#if defined(PRODUCT_WITHOUT_FACTORS)  // static_assert: Product.h: a product needs at least one factor
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Product<Speed, double>);
#endif

#if defined(PRODUCT_OF_SCALARS)  // static_assert: Product.h: every factor must have the same scalar
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Product<Heading, double, Emrys::Uniform<Heading, float>>);
#endif

#if defined(PRODUCT_ON_QUATERNION)  // static_assert: Product.h: a product places its factors by coordinate
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Product<Attitude, double, Emrys::Gaussian<Attitude>>);
#endif

#if defined(PRODUCT_FACTOR_ON_ANOTHER_SPACE)  // static_assert: Product.h: a factor's space must be the space of the
[[maybe_unused]] auto const product =
    Emrys::Product{Emrys::Composite<Heading>{}, Emrys::on<Heading>(Emrys::Uniform<Bearing>{})};
#endif

#if defined(PRODUCT_EACH_ON_FIXED_BLOCK)  // static_assert: Product.h: onEach<Path>() places a factor on a repeated
[[maybe_unused]] auto const product = Emrys::Product{Pose{}, Emrys::onEach<Position>(makePosition())};
#endif

#if defined(FACTORS_NOT_PLACED)  // static_assert: Product.h: each factor is placed by tag path
[[maybe_unused]] auto const factors = Emrys::Factors{makePosition()};
#endif

#if defined(PRODUCT_MARGINAL_OUTSIDE)  // static_assert: Product.h: the path neither reaches into a factor nor contains
[[maybe_unused]] void
marginalOutside()
{
    auto const product = makeProduct();
    [[maybe_unused]] auto const view = product.marginal<Bearing>();
}
#endif

#if defined(PRODUCT_CONDITIONAL_ON_EVERYTHING)  // static_assert: Product.h: conditioning on every factor leaves nothing
[[maybe_unused]] void
conditionalOnEverything()
{
    auto const product = Emrys::Product{
        Emrys::Composite<Pose>{},
        Emrys::on<Emrys::TagPath<Pose, Heading>>(Emrys::Uniform<Heading>{}),
        Emrys::on<Emrys::TagPath<Pose, Position>>(makePosition())
    };
    [[maybe_unused]] auto const rest = product.conditional<Pose>(Eigen::Vector3d::Zero());
}
#endif

#if defined(PRODUCT_SHARED_SPACE_TYPE)  // static_assert: Product.h: two factors share this space type, so it names
[[maybe_unused]] void
sharedSpaceType()
{
    auto const pose = Emrys::GaussianParams<Pose>{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    auto const product = Emrys::Product{Emrys::Composite<Lead, Wing>{}, Emrys::on<Lead>(pose), Emrys::on<Wing>(pose)};
    [[maybe_unused]] auto const view = product.marginal<Emrys::TagPath<Pose, Heading>>();
}
#endif

#if defined(PRODUCT_TAG_IN_TWO_FACTORS)  // static_assert: Product.h: this tag lies in more than one factor's space
[[maybe_unused]] void
tagInTwoFactors()
{
    auto const pose = Emrys::GaussianParams<Pose>{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    auto const product = Emrys::Product{Emrys::Composite<Lead, Wing>{}, Emrys::on<Lead>(pose), Emrys::on<Wing>(pose)};
    [[maybe_unused]] auto const view = product.marginal<Heading>();
}
#endif

#if defined(PRODUCT_FACTOR_WITHOUT_MARGINAL)  // static_assert: Product.h: this factor has no marginal<Path>()
[[maybe_unused]] void
factorWithoutMarginal()
{
    auto const particles = Emrys::ParticleSet<Place>{Eigen::Matrix<double, 2, 3>::Zero()};
    auto const kernel = particles.toDensity(Eigen::Matrix2d::Identity()).value();
    auto const product = Emrys::Product{
        Emrys::Composite<Place, Heading>{},
        Emrys::on<Place>(kernel),
        Emrys::on<Heading>(Emrys::Uniform<Heading>{})
    };
    [[maybe_unused]] auto const view = product.marginal<East>();
}
#endif

#if defined(PRODUCT_FACTOR_WITHOUT_CONDITIONAL)  // static_assert: Product.h: this factor has no conditional<Path>()
[[maybe_unused]] void
factorWithoutConditional()
{
    auto const particles = Emrys::ParticleSet<Place>{Eigen::Matrix<double, 2, 3>::Zero()};
    auto const kernel = particles.toDensity(Eigen::Matrix2d::Identity()).value();
    auto const product = Emrys::Product{
        Emrys::Composite<Place, Heading>{},
        Emrys::on<Place>(kernel),
        Emrys::on<Heading>(Emrys::Uniform<Heading>{})
    };
    [[maybe_unused]] auto const rest = product.conditional<East>(Eigen::Matrix<double, 1, 1>::Zero());
}
#endif

#if defined(PRODUCT_CONDITIONAL_NOT_A_FACTOR)  // static_assert: Product.h: this factor's conditional cannot be a
/// a factor whose conditional is a number, not a distribution
struct Unusual
{
    using Scalar = double;
    using SpaceType = Angles;
    static constexpr int ValueDimensionAtCompileTime = 2;
    static constexpr int MaxValueDimensionAtCompileTime = 2;

    [[nodiscard]] Angles
    space() const
    {
        return {};
    }

    [[nodiscard]] double
    logDensity(Eigen::Vector2d const &) const
    {
        return 0.0;
    }

    [[nodiscard]] Emrys::Moments<Angles>
    moments() const
    {
        return Emrys::Moments<Angles>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    }

    template <typename Path, typename Derived>
    [[nodiscard]] int
    conditional(Eigen::MatrixBase<Derived> const &) const
    {
        return 0;
    }
};

[[maybe_unused]] void
conditionalNotAFactor()
{
    auto const product = Emrys::Product{
        Emrys::Composite<Angles, Speed>{},
        Emrys::on<Angles>(Unusual{}),
        Emrys::on<Speed>(Emrys::Gamma<Speed>::create(2.0, 1.0).value())
    };
    [[maybe_unused]] auto const rest = product.conditional<Heading>(Eigen::Matrix<double, 1, 1>::Zero());
}
#endif

// Grids, chains, and divergences.

#if defined(POINT_GRID_COVARIANCE)  // static_assert: GridGaussian.h: the cell covariance is SharedCovariance or
[[maybe_unused]] constexpr auto size = sizeof(Emrys::PointGrid<State, Position, Emrys::Finite<4>, int>);
#endif

#if defined(GRID_GAUSSIAN_COVARIANCE)  // static_assert: GridGaussian.h: the covariance is SharedCovariance or
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::GridGaussianParams<Emrys::PointGrid<State, Position, Emrys::Finite<4>>, int>);
#endif

#if defined(GRID_GAUSSIAN_SHARED_OVER_OWN)  // static_assert: GridGaussian.h: cells with covariances of their own give
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::GridGaussianParams<
           Emrys::PointGrid<State, Position, Emrys::Finite<4>, Emrys::PerPointCovariance>,
           Emrys::SharedCovariance>);
#endif

#if defined(GRID_GAUSSIAN_COMPONENT)  // static_assert: GridGaussian.h: a cell holds a Gaussian or a hybrid of
using Cells = Emrys::PointGrid<State, Position, Emrys::Finite<4>>;
[[maybe_unused]] constexpr auto size =
    sizeof(Emrys::GridGaussianParams<Cells, Emrys::PerPointCovariance, Emrys::SqrtGaussian<typename Cells::RestSpace>>);
#endif

#if defined(GRID_GAUSSIAN_COMPONENT_SPACE)  // static_assert: GridGaussian.h: a cell's distribution is over the grid's
[[maybe_unused]] constexpr auto size = sizeof(Emrys::GridGaussianParams<
                                              Emrys::PointGrid<State, Position, Emrys::Finite<4>>,
                                              Emrys::PerPointCovariance,
                                              Emrys::Gaussian<State>>);
#endif

#if defined(GRID_GAUSSIAN_SHARED_HYBRID)  // static_assert: GridGaussian.h: mixing a cell's modes gives each cell
using Cells = Emrys::PointGrid<State, Position, Emrys::Finite<4>>;
[[maybe_unused]] constexpr auto size = sizeof(
    Emrys::
        GridGaussianParams<Cells, Emrys::SharedCovariance, Emrys::HybridGaussian<typename Cells::RestSpace, Maneuver>>
);
#endif

#if defined(REGULAR_GRID_RESIZABLE)  // static_assert: GridGaussian.h: a regular grid needs a fixed-size block
[[maybe_unused]] auto const grid = Emrys::regularGrid<Landmarks>(
    Emrys::Composite<Landmarks>{},
    Eigen::VectorXd::Zero(1),
    Eigen::VectorXd::Ones(1),
    std::array<Eigen::Index, 0>{}
);
#endif

#if defined(GRID_CELLS_REPEATED_TAG)  // static_assert: GridGaussian.h: two coordinates of the block lie in blocks with
struct Pair : Emrys::Composite<Pose, Wing>
{};

[[maybe_unused]] constexpr auto size = sizeof(Emrys::GridCells<Pair>);
#endif

#if defined(REGULAR_GRID_FOREIGN_CELLS)  // static_assert: GridGaussian.h: a regular grid's cells are the GridCells of
[[maybe_unused]] constexpr auto size = sizeof(Emrys::PointGrid<State, Position, Emrys::GridCells<Heading>>);
#endif

#if defined(GRID_DENSITY_OF_TEMPORARY)  // deleted: GridGaussian.h: toDensity() const &&
using Density = decltype(std::declval<Emrys::GridGaussianParams<Emrys::PointGrid<State, Position, Emrys::Finite<4>>>>()
                             .toDensity());
#endif

#if defined(PARTICLE_GAUSSIAN_COVARIANCE)  // static_assert: ParticleGaussian.h: the covariance is SharedCovariance or
[[maybe_unused]] constexpr auto size = sizeof(Emrys::ParticleGaussianParams<State, Position, int>);
#endif

#if defined(PARTICLE_GAUSSIAN_DENSITY_OF_TEMPORARY)  // deleted: ParticleGaussian.h: toDensity() const &&
using Density = decltype(std::declval<Emrys::ParticleGaussianParams<State, Position>>().toDensity());
#endif

#if defined(PARTICLE_GAUSSIAN_COMPONENT)  // static_assert: ParticleGaussian.h: a particle holds a Gaussian or a
using Rest = typename Emrys::ParticleGaussianParams<State, Position>::RestSpace;
[[maybe_unused]] constexpr auto size = sizeof(Emrys::ParticleGaussianParams<
                                              State,
                                              Position,
                                              Emrys::PerPointCovariance,
                                              double,
                                              Emrys::DefaultAllocator,
                                              Emrys::SqrtGaussian<Rest>>);
#endif

#if defined(PARTICLE_GAUSSIAN_COMPONENT_SPACE)  // static_assert: ParticleGaussian.h: a particle's distribution is over
[[maybe_unused]] constexpr auto size = sizeof(Emrys::ParticleGaussianParams<
                                              State,
                                              Position,
                                              Emrys::PerPointCovariance,
                                              double,
                                              Emrys::DefaultAllocator,
                                              Emrys::Gaussian<State>>);
#endif

#if defined(PARTICLE_GAUSSIAN_SHARED_HYBRID)  // static_assert: ParticleGaussian.h: mixing a particle's modes gives
using Rest = typename Emrys::ParticleGaussianParams<State, Position>::RestSpace;
[[maybe_unused]] constexpr auto size = sizeof(Emrys::ParticleGaussianParams<
                                              State,
                                              Position,
                                              Emrys::SharedCovariance,
                                              double,
                                              Emrys::DefaultAllocator,
                                              Emrys::HybridGaussian<Rest, Maneuver>>);
#endif

#if defined(MARKOV_CHAINS_OUT_OF_ORDER)  // static_assert: Categorical.h: give one chain per axis of the product, in
[[maybe_unused]] auto const chain = Emrys::MarkovChain<Mode>{
    Mode{},
    Emrys::MarkovChain<Sensor>::persistent(0.9),
    Emrys::MarkovChain<Maneuver>::persistent(0.9)
};
#endif

#if defined(CONTINUOUS_CHAINS_OUT_OF_ORDER)  // static_assert: ContinuousTimeMarkovChain.h: give one chain per axis of
[[maybe_unused]] auto const chain = Emrys::ContinuousTimeMarkovChain<Mode>{
    Mode{},
    Emrys::ContinuousTimeMarkovChain<Sensor>::persistent(1.0),
    Emrys::ContinuousTimeMarkovChain<Maneuver>::persistent(1.0)
};
#endif

#if defined(KULLBACK_LEIBLER_WITHOUT_CLOSED_FORM)  // static_assert: Divergence.h: kullbackLeibler(p, q) has a closed
[[maybe_unused]] void
kullbackLeibler()
{
    [[maybe_unused]] double const d = Emrys::kullbackLeibler(Emrys::Uniform<Heading>{}, Emrys::Uniform<Heading>{});
}
#endif

#if defined(BHATTACHARYYA_WITHOUT_CLOSED_FORM)  // static_assert: Divergence.h: bhattacharyya(p, q) has a closed form
[[maybe_unused]] void
bhattacharyya()
{
    [[maybe_unused]] double const d = Emrys::bhattacharyya(Emrys::Uniform<Heading>{}, Emrys::Uniform<Heading>{});
}
#endif

#if defined(HELLINGER_WITHOUT_CLOSED_FORM)  // static_assert: Divergence.h: hellinger(p, q) has a closed form only
[[maybe_unused]] void
hellinger()
{
    [[maybe_unused]] double const d = Emrys::hellinger(Emrys::Uniform<Heading>{}, Emrys::Uniform<Heading>{});
}
#endif

#if defined(JENSEN_SHANNON_WITHOUT_CLOSED_FORM)  // static_assert: Divergence.h: jensenShannon(p, q) has a closed form
[[maybe_unused]] void
jensenShannon()
{
    [[maybe_unused]] double const d = Emrys::jensenShannon(Emrys::Uniform<Heading>{}, Emrys::Uniform<Heading>{});
}
#endif

#if defined(WASSERSTEIN_WITHOUT_CLOSED_FORM)  // static_assert: Divergence.h: wasserstein(p, q) has a closed form only
[[maybe_unused]] void
wasserstein()
{
    [[maybe_unused]] double const d = Emrys::wasserstein(Emrys::Uniform<Heading>{}, Emrys::Uniform<Heading>{});
}
#endif

// Paths into the spaces that only a distribution follows.

#if defined(CONDITIONAL_ON_UNTAGGED_BLOCK)  // static_assert: Subspace.h: a path names blocks by tag
[[maybe_unused]] auto const rest =
    Emrys::GaussianParams<Emrys::Composite<Emrys::Linear<2>, Heading>>{}.conditional<Emrys::Linear<2>>(
        Eigen::Vector2d::Zero()
    );
#endif

#if defined(CONDITIONAL_ON_MISSING_TAG)  // static_assert: Subspace.h: the space does not contain this tag
[[maybe_unused]] void
conditionalOnMissingTag()
{
    auto const params = Emrys::GaussianParams<Heading>{};
    [[maybe_unused]] auto const rest = params.conditional<Speed>(Eigen::Matrix<double, 1, 1>::Zero());
}
#endif

#if defined(ELEMENT_MARGINAL_OF_UNREPEATED)  // static_assert: Subspace.h: this takes a repeated block
[[maybe_unused]] void
elementOfUnrepeated()
{
    auto const params = makeConvoy();
    [[maybe_unused]] auto const view = params.marginal<Position>(0);
}
#endif

#if defined(INSERT_INTO_FIXED_BLOCK)  // static_assert: Subspace.h: insert() and erase() take a Repeated block or a
[[maybe_unused]] void
insertIntoFixed()
{
    auto params = makeConvoy();
    params.erase<Position>(0);
}
#endif

#if defined(CATEGORICAL_CONDITIONAL_ON_EVERYTHING)  // static_assert: Discrete.h: nothing is left of the space
[[maybe_unused]] void
categoricalOnEverything()
{
    auto const params = Emrys::CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{1.0, 1.0});
    [[maybe_unused]] auto const rest = params.conditional<Maneuver>(0);
}
#endif

// The information forms are not components, and the callbacks of forEach() and reweight() return
// what each is for.

#if defined(HYBRID_OF_INFO_FORM)  // static_assert: Hybrid.h: a Gaussian in an information form is not a hybrid's
[[maybe_unused]] constexpr auto size = sizeof(Emrys::Hybrid<Emrys::InfoGaussian<Position>, Maneuver>);
#endif

#if defined(MIXTURE_OF_INFO_FORM)  // static_assert: Mixture.h: a Gaussian in an information form is not a mixture
[[maybe_unused]] constexpr auto size = sizeof(Emrys::MixtureParams<Emrys::InfoGaussian<Position>>);
#endif

#if defined(MIXTURE_FOR_EACH_RETURNS)  // static_assert: Mixture.h: forEach() takes a callback that returns nothing
[[maybe_unused]] void
mixtureForEachReturns()
{
    auto mixture = Emrys::GaussianMixtureParams<Position>{Position{}};
    mixture.forEach([](auto &) {
        return 0.0;
    });
}
#endif

#if defined(MIXTURE_REWEIGHT_RETURNS_NOTHING)  // static_assert: Mixture.h: reweight() takes a callback that returns a
                                               // log
[[maybe_unused]] void
mixtureReweightReturnsNothing()
{
    auto mixture = Emrys::GaussianMixtureParams<Position>{Position{}};
    mixture.reweight([](auto &) {});
}
#endif

#if defined(PARTICLES_FOR_EACH_RETURNS)  // static_assert: ParticleSet.h: forEach() takes a callback that returns
                                         // nothing
[[maybe_unused]] void
particlesForEachReturns()
{
    auto particles = Emrys::ParticleSet<Position>{Position{}, Eigen::Matrix2d::Zero(), Eigen::Array2d::Zero()};
    particles.forEach([](auto &) {
        return 0.0;
    });
}
#endif

#if defined(PARTICLES_REWEIGHT_RETURNS_NOTHING)  // static_assert: ParticleSet.h: reweight() takes a callback that
                                                 // returns a
[[maybe_unused]] void
particlesReweightReturnsNothing()
{
    auto particles = Emrys::ParticleSet<Position>{Position{}, Eigen::Matrix2d::Zero(), Eigen::Array2d::Zero()};
    particles.reweight([](auto &) {});
}
#endif

#if defined(SQRT_FACTORIZE_REPAIRS)  // static_assert: SqrtGaussian.h: a square-root form's root is positive
                                     // semidefinite
[[maybe_unused]] void
sqrtFactorizeRepairs()
{
    auto const params = Emrys::SqrtGaussianParams<Position>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    [[maybe_unused]] auto const factorized = params.factorize(Emrys::roots::ridge());
}
#endif

#if defined(ROOTS_FALLBACK_ON_UNFAILING)  // static_assert: Roots.h: axes that cannot fail take no fallback
[[maybe_unused]] void
rootsFallbackOnUnfailing()
{
    [[maybe_unused]] auto const axes = Emrys::roots::principal(Emrys::roots::ridge());
}
#endif

#if defined(ROOTS_FALLBACK_NOT_AN_ENTRY)  // static_assert: Roots.h: a fallback is axes or a repair
[[maybe_unused]] void
rootsFallbackNotAnEntry()
{
    [[maybe_unused]] auto const axes = Emrys::roots::cholesky(0.5);
}
#endif

#if defined(ROOTS_FALLBACK_LIST_STARTS)  // static_assert: Roots.h: a fallback list starts with axes
[[maybe_unused]] void
rootsFallbackListStarts()
{
    [[maybe_unused]] auto const axes = Emrys::roots::WithFallbacks<Emrys::roots::Ridge<>, Emrys::roots::Pivoted>{};
}
#endif

#if defined(ROOTS_FALLBACK_NESTED)  // static_assert: Roots.h: a fallback list is flat
[[maybe_unused]] void
rootsFallbackNested()
{
    [[maybe_unused]] auto const axes =
        Emrys::roots::cholesky(Emrys::roots::along(Eigen::Vector2d{1.0, 0.0}, Emrys::roots::pivoted()));
}
#endif

#if defined(ROOTS_FALLBACK_UNREACHED)  // static_assert: Roots.h: a fallback after axes that cannot fail is never
                                       // reached
[[maybe_unused]] void
rootsFallbackUnreached()
{
    [[maybe_unused]] auto const axes = Emrys::roots::cholesky(Emrys::roots::pivoted(), Emrys::roots::ridge());
}
#endif

#if defined(ROOTS_RIDGE_NEGATIVE)  // static_assert: Roots.h: a ridge's margins are at least zero
[[maybe_unused]] void
rootsRidgeNegative()
{
    [[maybe_unused]] auto const repair = Emrys::roots::ridge<-1.0>();
}
#endif

#if defined(ROOTS_ALONG_ROW)  // static_assert: Roots.h: along() takes a direction as a column vector
[[maybe_unused]] void
rootsAlongRow()
{
    [[maybe_unused]] auto const axes = Emrys::roots::along(Eigen::RowVector2d{1.0, 0.0});
}
#endif

}  // namespace

// The measurement of a discrete state (step 15, [emissions]): the EmissionModel concept and update() of a
// Categorical by one; EmissionTable, DensityEmission, PoissonEmission, and ExponentialEmission
// against their closed forms; and ArrivalInterval's predict(), the exact step of a
// Markov-modulated Poisson process, against a plain-Eigen evaluation of the same matrix product.
// Header under test: Models/Emission.h.

#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Models/Emission.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <utility>

using Emrys::ArrivalInterval;
using Emrys::CategoricalParams;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::DensityEmission;
using Emrys::EmissionModel;
using Emrys::EmissionTable;
using Emrys::ExponentialEmission;
using Emrys::Finite;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::Named;
using Emrys::PoissonEmission;
using Emrys::WeightError;

namespace
{

struct Rain
{};

struct Dry
{};

struct Weather : Named<Rain, Dry>
{};

struct Umbrella
{};

struct NoUmbrella
{};

struct Sighting : Named<Umbrella, NoUmbrella>
{};

using Table = EmissionTable<Weather, Sighting>;

// observe() borrows its table, so a temporary table does not compile (hidden-markov-filter.md Q4).
template <typename T>
concept Observable = requires(T && table) { std::forward<T>(table).observe(0); };
static_assert(Observable<Table const &> && !Observable<Table>);

static_assert(EmissionModel<Table::Observation, Weather>);
static_assert(EmissionModel<PoissonEmission<Weather>, Weather>);

// A count is an integer, so a step and a count given in each other's place do not compile.
static_assert(std::constructible_from<PoissonEmission<Weather>, Eigen::Vector3d, int, double>);
static_assert(!std::constructible_from<PoissonEmission<Weather>, Eigen::Vector3d, double, double>);
static_assert(!std::constructible_from<PoissonEmission<Weather>, Eigen::Vector3d, bool, double>);
static_assert(EmissionModel<ExponentialEmission<Weather>, Weather>);
static_assert(!EmissionModel<Table, Weather>, "the table is not itself a measurement");

// The buffer is fixed in size where the space's cardinality is, and run-time sized where it is not.
static_assert(Emrys::EmissionBuffer<Weather, double>::RowsAtCompileTime == 2);
static_assert(Emrys::EmissionBuffer<Finite<Eigen::Dynamic, 6>, double>::RowsAtCompileTime == Eigen::Dynamic);

Table const umbrellaTable{Weather{}, Sighting{}, Eigen::Matrix2d{{0.9, 0.1}, {0.2, 0.8}}};

// A user's umbrella sighting written against a fixed-size buffer, as the space's cardinality allows.
struct FixedSighting
{
    using Scalar = double;

    void
    logLikelihoods(Weather const &, Eigen::Ref<Eigen::Array2d> out) const
    {
        out = Eigen::Array2d{0.9, 0.2}.log();
    }
};

// The same sighting written against a run-time sized buffer, which the fixed one binds to.
struct DynamicSighting
{
    using Scalar = double;

    void
    logLikelihoods(Weather const &, Eigen::Ref<Eigen::ArrayXd> out) const
    {
        out = Eigen::Array2d{0.9, 0.2}.log();
    }
};

static_assert(EmissionModel<FixedSighting, Weather>);
static_assert(EmissionModel<DynamicSighting, Weather>);

}  // namespace

// An umbrella seen under P(umbrella | rain) = 0.9 and P(umbrella | dry) = 0.2 from even odds:
// the evidence is 0.5 · 0.9 + 0.5 · 0.2 = 0.55 and the posterior of rain 0.45 / 0.55 = 9/11, by hand.
// The table keeps its spaces and matrix, and the observation its value.
TEST(Emission, TableUpdateIsBayesRule)
{
    EXPECT_EQ(umbrellaTable.space(), Weather{});
    EXPECT_EQ(umbrellaTable.observationSpace(), Sighting{});
    EXPECT_EQ(umbrellaTable.matrix()(1, 0), 0.2);
    auto const prior = CategoricalParams<Weather>{};
    auto const observation = umbrellaTable.observe(Sighting::index<Umbrella>());
    EXPECT_EQ(observation.observed(), Sighting::index<Umbrella>());
    auto const updated = update(prior, observation);
    ASSERT_TRUE(updated);
    EXPECT_NEAR(updated->posterior.toDensity().value().probability<Rain>(), 9.0 / 11.0, 1e-15);
    EXPECT_NEAR(updated->logEvidence, std::log(0.55), 1e-15);
}

// A user's model updates by Bayes' rule whether it takes the fixed-size buffer or a run-time sized
// one: the same hand computation as TableUpdateIsBayesRule.
TEST(Emission, UserModelTakesEitherBuffer)
{
    auto const fixed = update(CategoricalParams<Weather>{}, FixedSighting{});
    auto const dynamic = update(CategoricalParams<Weather>{}, DynamicSighting{});
    ASSERT_TRUE(fixed);
    ASSERT_TRUE(dynamic);
    EXPECT_NEAR(fixed->posterior.toDensity().value().probability<Rain>(), 9.0 / 11.0, 1e-15);
    EXPECT_NEAR(dynamic->posterior.toDensity().value().probability<Rain>(), 9.0 / 11.0, 1e-15);
    EXPECT_NEAR(fixed->logEvidence, std::log(0.55), 1e-15);
}

// A measurement that every value with probability gives zero likelihood is impossible: the update
// is a WeightError.
TEST(Emission, ImpossibleMeasurementIsAnError)
{
    auto const certain = Table{Weather{}, Sighting{}, Eigen::Matrix2d{{1.0, 0.0}, {1.0, 0.0}}};
    auto const updated = update(CategoricalParams<Weather>{}, certain.observe(Sighting::index<NoUmbrella>()));
    ASSERT_FALSE(updated);
    EXPECT_EQ(updated.error(), WeightError::ZeroSum);
}

// DensityEmission evaluates the observation under each value's density: here a unit-variance
// Gaussian at 0 for rain and at 2 for dry, observed at 0.5, whose log-densities are
// -½ log 2π - ½ (0.5)² and -½ log 2π - ½ (1.5)².
TEST(Emission, DensityEmissionUsesEachValuesDensity)
{
    auto const gaussians = std::array{
        GaussianParams<Linear<1>>{Linear<1>{}, Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0}},
        GaussianParams<Linear<1>>{Linear<1>{}, Eigen::Matrix<double, 1, 1>{2.0}, Eigen::Matrix<double, 1, 1>{1.0}},
    };
    auto const densities = std::array{gaussians[0].toDensity().value(), gaussians[1].toDensity().value()};
    auto const emission = DensityEmission{Eigen::Matrix<double, 1, 1>{0.5}, [&](Eigen::Index v) -> auto const & {
                                              return densities[std::size_t(v)];
                                          }};
    static_assert(EmissionModel<decltype(emission), Weather>);
    EXPECT_EQ(emission.observation()(0), 0.5);
    auto logLikelihoods = Eigen::Array2d{};
    emission.logLikelihoods(Weather{}, logLikelihoods);
    double const logNormalizer = -0.5 * std::log(2 * std::numbers::pi);
    EXPECT_NEAR(logLikelihoods(0), logNormalizer - 0.5 * 0.25, 1e-14);
    EXPECT_NEAR(logLikelihoods(1), logNormalizer - 0.5 * 2.25, 1e-14);

    auto const updated = update(CategoricalParams<Weather>{}, emission);
    ASSERT_TRUE(updated);
    EXPECT_NEAR(updated->posterior.toDensity().value().probability<Rain>(), 1.0 / (1.0 + std::exp(-1.0)), 1e-14);
}

// PoissonEmission is n log(λ Δt) - λ Δt - log n!; a zero rate gives zero likelihood to a positive
// count and certainty to none.
TEST(Emission, PoissonEmissionMatchesTheClosedForm)
{
    auto const rates = Eigen::Array<double, 3, 1>{0.5, 4.0, 0.0};
    auto logLikelihoods = Eigen::Array<double, 3, 1>{};
    auto const three = PoissonEmission<Finite<3>>{rates, 3, 2.0};
    EXPECT_EQ(three.count(), 3);
    three.logLikelihoods(Finite<3>{}, logLikelihoods);
    EXPECT_NEAR(logLikelihoods(0), 3 * std::log(1.0) - 1.0 - std::log(6.0), 1e-14);
    EXPECT_NEAR(logLikelihoods(1), 3 * std::log(8.0) - 8.0 - std::log(6.0), 1e-14);
    EXPECT_EQ(logLikelihoods(2), -std::numeric_limits<double>::infinity());
    PoissonEmission<Finite<3>>{rates, 0, 2.0}.logLikelihoods(Finite<3>{}, logLikelihoods);
    EXPECT_EQ(logLikelihoods(2), 0.0);
}

// ExponentialEmission is log λ - λ τ.
TEST(Emission, ExponentialEmissionMatchesTheClosedForm)
{
    auto logLikelihoods = Eigen::Array2d{};
    ExponentialEmission<Weather>{Eigen::Array2d{0.5, 3.0}, 0.4}.logLikelihoods(Weather{}, logLikelihoods);
    EXPECT_NEAR(logLikelihoods(0), std::log(0.5) - 0.2, 1e-15);
    EXPECT_NEAR(logLikelihoods(1), std::log(3.0) - 1.2, 1e-15);
}

// An interval ending in an arrival moves the distribution as p e^{(Q - Λ)τ} Λ, and the log of its
// total is the interval's log-likelihood; without the arrival it is p e^{(Q - Λ)τ}. The reference
// evaluates the product with Eigen directly, without the shift predict() applies.
TEST(Emission, ArrivalIntervalIsTheModulatedStep)
{
    auto const generator = Eigen::Matrix2d{{-0.4, 0.4}, {1.5, -1.5}};
    auto const rates = Eigen::Array2d{0.2, 6.0};
    auto const chain = ContinuousTimeMarkovChain<Weather>{generator};
    auto const prior = CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0.3, 0.7});
    double const tau = 0.8;
    Eigen::Matrix2d const survival =
        (Eigen::Matrix2d{generator - Eigen::Matrix2d{rates.matrix().asDiagonal()}} * tau).exp();

    for (bool const arrival : {true, false})
    {
        Eigen::RowVector2d moved = prior.logWeights().exp().matrix().transpose() * survival;
        if (arrival)
        {
            moved = moved * rates.matrix().asDiagonal();
        }
        auto const stepped = predict(prior, chain, tau, ArrivalInterval<Weather>{rates, arrival});
        ASSERT_TRUE(stepped);
        EXPECT_NEAR(stepped->logEvidence, std::log(moved.sum()), 1e-13);
        EXPECT_NEAR(stepped->posterior.toDensity().value().probability<Rain>(), moved(0) / moved.sum(), 1e-13);
    }
}

// Over a long interval the survival e^{-λ τ} of every value underflows a double, but the shift by
// the slowest rate keeps the step finite: with a chain that never moves, the likelihood of an
// arrival after τ = 1000 is Σ p_v λ_v e^{-λ_v τ}, whose log is log(0.5) - 1000 to far more than
// double precision, since the faster value's term is e^{-4000} smaller.
TEST(Emission, LongArrivalIntervalDoesNotUnderflow)
{
    auto const still = ContinuousTimeMarkovChain<Weather>{};
    auto const stepped =
        predict(CategoricalParams<Weather>{}, still, 1000.0, ArrivalInterval<Weather>{Eigen::Array2d{1.0, 5.0}});
    ASSERT_TRUE(stepped);
    EXPECT_NEAR(stepped->logEvidence, std::log(0.5) - 1000.0, 1e-9);
    EXPECT_DOUBLE_EQ(stepped->posterior.toDensity().value().probability<Rain>(), 1.0);
}

// An arrival from values that never arrive is impossible, as are prior weights that fail to define
// a distribution: both steps return ZeroSum.
TEST(Emission, ImpossibleArrivalIsAnError)
{
    auto const still = ContinuousTimeMarkovChain<Weather>{};
    auto const rainOnly = CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{1.0, 0.0});
    auto const onlyDryArrives = ArrivalInterval<Weather>{Eigen::Array2d{0.0, 5.0}};
    EXPECT_EQ(predict(rainOnly, still, 1.0, onlyDryArrives).error(), Emrys::WeightError::ZeroSum);
    auto const zero = CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0.0, 0.0});
    EXPECT_EQ(predict(zero, still, 1.0, onlyDryArrives).error(), Emrys::WeightError::ZeroSum);
}

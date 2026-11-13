// The hidden Markov model filter (step 15, [filters-and-recursions], [emissions]): HiddenMarkovFilter's forward
// recursion against a hand-computed pass, from weights that need not sum to one, with an identity emission table, with
// a run-time cardinality, and on an impossible measurement, which it rejects [update-outcomes]; and with a
// ContinuousTimeMarkovChain as the
// Markov-modulated Poisson process filter, against the closed-form likelihood of Fischer and Meier-Hellstern (1993) and
// as the limit of the approximate filter that counts arrivals over short intervals. Header under test:
// Filters/HiddenMarkovFilter.h.

#include <EmrysFilter/Filters/HiddenMarkovFilter.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Models/Emission.h>
#include <EmrysFilter/Models/Simulation.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using Emrys::ArrivalInterval;
using Emrys::CategoricalParams;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::EmissionTable;
using Emrys::Finite;
using Emrys::HiddenMarkovFilter;
using Emrys::MarkovChain;
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

EmissionTable const umbrellaTable{Weather{}, Sighting{}, Eigen::Matrix2d{{0.9, 0.1}, {0.2, 0.8}}};

// An emission model whose every log-likelihood is NaN, as a broken model's would be.
struct NanEmission
{
    using Scalar = double;

    template <typename Space>
    void
    logLikelihoods(Space const & /*space*/, Emrys::EmissionBuffer<Space, double> out) const
    {
        out.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
};

}  // namespace

// The filter deduces its space, chain, and scalar from the chain, with or without a starting state.
TEST(HiddenMarkovFilter, DeducesFromTheChain)
{
    auto const chain = MarkovChain<Weather>::persistent(0.7);
    auto const fromState = HiddenMarkovFilter{chain, CategoricalParams<Weather>{}};
    auto const uniform = HiddenMarkovFilter{chain};
    static_assert(std::is_same_v<decltype(fromState), HiddenMarkovFilter<Weather, MarkovChain<Weather>, double> const>);
    static_assert(std::is_same_v<decltype(uniform), decltype(fromState)>);
    auto const rates = ContinuousTimeMarkovChain<Weather>{Eigen::Matrix2d{{-1.0, 1.0}, {1.0, -1.0}}};
    auto const continuous = HiddenMarkovFilter{rates};
    static_assert(std::is_same_v<
                  decltype(continuous),
                  HiddenMarkovFilter<Weather, ContinuousTimeMarkovChain<Weather>, double> const>);
    EXPECT_DOUBLE_EQ(uniform.state().toDensity().value().probability<Rain>(), 0.5);
}

// setState() replaces the state and chain() the chain between steps: from certain rain, a chain
// that keeps the weather with probability 0.9 predicts rain with 0.9. A mutable state() changes the
// state where it is: from certain dry weather, the chain predicts rain with 0.1.
TEST(HiddenMarkovFilter, ReplacesItsStateAndChain)
{
    auto filter = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7)};
    EXPECT_EQ(std::as_const(filter).chain().matrix(), MarkovChain<Weather>::persistent(0.7).matrix());
    filter.setState(CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{1.0, 0.0}));
    filter.chain() = MarkovChain<Weather>::persistent(0.9);
    filter.predict();
    EXPECT_DOUBLE_EQ(filter.state().toDensity().value().probability<Rain>(), 0.9);

    EXPECT_EQ(&filter.state(), &std::as_const(filter).state());
    filter.state().setWeights(Eigen::Vector2d{0.0, 1.0});
    filter.predict();
    EXPECT_DOUBLE_EQ(filter.state().toDensity().value().probability<Rain>(), 0.1);
}

// Two days with an umbrella, Π = [[0.7, 0.3], [0.3, 0.7]], from even odds. By hand: day one's
// evidence is 0.55 and the posterior of rain 9/11; day two predicts rain with
// (0.7 · 9 + 0.3 · 2) / 11 = 6.9/11, its evidence is (6.9 · 0.9 + 4.1 · 0.2) / 11 = 7.03/11, and the
// posterior of rain is 6.21/7.03. The log-evidence is the sum of the two days'.
TEST(HiddenMarkovFilter, MatchesAHandComputedForwardPass)
{
    auto filter = HiddenMarkovFilter{MarkovChain<Weather>::persistent(0.7)};
    auto const umbrella = umbrellaTable.observe(Sighting::index<Umbrella>());

    auto const first = filter.update(umbrella);
    ASSERT_TRUE(first);
    EXPECT_TRUE(first->accepted);
    EXPECT_NEAR(first->logLikelihood(), std::log(0.55), 1e-15);
    EXPECT_NEAR(filter.state().toDensity().value().probability<Rain>(), 9.0 / 11.0, 1e-15);

    filter.predict();
    EXPECT_NEAR(filter.state().toDensity().value().probability<Rain>(), 6.9 / 11.0, 1e-15);
    auto const second = filter.update(umbrella);
    ASSERT_TRUE(second);
    EXPECT_NEAR(second->logLikelihood(), std::log(7.03 / 11.0), 1e-15);
    EXPECT_NEAR(filter.state().toDensity().value().probability<Rain>(), 6.21 / 7.03, 1e-15);
    EXPECT_NEAR(filter.logEvidence(), std::log(0.55) + std::log(7.03 / 11.0), 1e-14);

    filter.resetLogEvidence();
    EXPECT_EQ(filter.logEvidence(), 0.0);
}

// A starting state's weights need not sum to one: weights of 5 and 5 give the hand-computed pass
// above, the update normalizing them. Weights of zero are carried through a prediction unchecked
// and fail the update with ZeroSum, which leaves the state and the log-evidence as they were.
TEST(HiddenMarkovFilter, ChecksTheStateWeightsAtTheUpdate)
{
    auto const umbrella = umbrellaTable.observe(Sighting::index<Umbrella>());
    auto const chain = MarkovChain<Weather>::persistent(0.7);
    auto weighted =
        HiddenMarkovFilter{chain, CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{5, 5})};
    auto const evidence = weighted.update(umbrella);
    ASSERT_TRUE(evidence);
    EXPECT_NEAR(evidence->logLikelihood(), std::log(0.55), 1e-15);
    EXPECT_NEAR(weighted.state().toDensity().value().probability<Rain>(), 9.0 / 11.0, 1e-15);

    auto zero = HiddenMarkovFilter{chain, CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0, 0})};
    zero.predict();
    EXPECT_EQ(zero.update(umbrella).error(), WeightError::ZeroSum);
    EXPECT_EQ(zero.state().logTotalWeight().error(), WeightError::ZeroSum);
    EXPECT_EQ(zero.logEvidence(), 0.0);
}

// With an identity emission table every observation names the value, so the filter's state is
// certain of the true value at every step of a simulated run, and each step's evidence is the
// predicted probability of the value observed.
TEST(HiddenMarkovFilter, IdentityEmissionRecoversTheSequence)
{
    using Space = Finite<4>;
    auto const chain = MarkovChain<Space>::persistent(0.6);
    auto const identity = EmissionTable{Space{}, Space{}, Eigen::Matrix4d::Identity()};
    auto filter = HiddenMarkovFilter{chain};
    auto rng = std::mt19937{7};
    Eigen::Index value = 0;
    for (int k = 0; k < 50; ++k)
    {
        if (k > 0)
        {
            value = Emrys::simulate(value, chain, rng);
            filter.predict();
        }
        double const predicted = filter.state().toDensity().value().probability(value);
        auto const evidence = filter.update(identity.observe(value));
        ASSERT_TRUE(evidence);
        EXPECT_NEAR(evidence->logLikelihood(), std::log(predicted), 1e-14);
        EXPECT_EQ(filter.state().toDensity().value().mostProbable(), value);
        EXPECT_DOUBLE_EQ(filter.state().toDensity().value().probability(value), 1.0);
    }
}

// An impossible measurement is rejected, not an error (decisions.md [update-outcomes]): the state is
// as it was, the update's log-likelihood is -inf, and so is the log-evidence until it is reset,
// since the measurements so far have probability zero under the model. A NaN log-likelihood is an
// error, with the state and the log-evidence unchanged.
TEST(HiddenMarkovFilter, ImpossibleMeasurementIsRejected)
{
    auto const onlyUmbrellas = EmissionTable{Weather{}, Sighting{}, Eigen::Matrix2d{{1.0, 0.0}, {1.0, 0.0}}};
    auto filter = HiddenMarkovFilter{
        MarkovChain<Weather>::persistent(0.7),
        CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0.2, 0.8})
    };
    auto const before = filter.state().logWeights();
    auto const updated = filter.update(onlyUmbrellas.observe(Sighting::index<NoUmbrella>()));
    ASSERT_TRUE(updated);
    EXPECT_FALSE(updated->accepted);
    EXPECT_EQ(updated->logLikelihood(), -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(filter.state().logWeights().matrix() == before.matrix());
    EXPECT_EQ(filter.logEvidence(), -std::numeric_limits<double>::infinity());
    filter.resetLogEvidence();

    auto const failed = filter.update(NanEmission{});
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error(), WeightError::NonFinite);
    EXPECT_TRUE(filter.state().logWeights().matrix() == before.matrix());
    EXPECT_EQ(filter.logEvidence(), 0.0);
}

// An arrival the state makes impossible fails the step that records its joint for a smoother, and
// leaves the state and the log-evidence as they were.
TEST(HiddenMarkovFilter, ImpossibleArrivalLeavesTheState)
{
    auto const still = ContinuousTimeMarkovChain<Weather>{};
    auto filter =
        HiddenMarkovFilter{still, CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{1.0, 0.0})};
    auto const joint = filter.predictJoint(1.0, ArrivalInterval<Weather>{Eigen::Array2d{0.0, 5.0}});
    ASSERT_FALSE(joint);
    EXPECT_EQ(joint.error(), WeightError::ZeroSum);
    EXPECT_EQ(filter.state().toDensity().value().probability<Rain>(), 1.0);
    EXPECT_EQ(filter.logEvidence(), 0.0);
}

// A run-time cardinality takes the same steps: three values, a persistent chain, and a table onto
// two observations, against the forward step written out with plain Eigen.
TEST(HiddenMarkovFilter, RunTimeCardinality)
{
    using Space = Finite<Eigen::Dynamic, 6>;
    auto const space = Space{3};
    auto const emission = Eigen::MatrixXd{{0.5, 0.5}, {0.9, 0.1}, {0.2, 0.8}};
    auto const table = EmissionTable{space, Finite<2>{}, emission};
    auto const chain = MarkovChain<Space>::persistent(0.8, space);
    auto filter = HiddenMarkovFilter{chain};
    Eigen::VectorXd reference = Eigen::VectorXd::Constant(3, 1.0 / 3.0);
    double logEvidence = 0;
    for (Eigen::Index const z : {0, 1, 1, 0})
    {
        filter.predict();
        reference = chain.matrix().transpose() * reference;
        ASSERT_TRUE(filter.update(table.observe(z)));
        reference = reference.cwiseProduct(emission.col(z));
        logEvidence += std::log(reference.sum());
        reference /= reference.sum();
    }
    EXPECT_TRUE(filter.state().toDensity().value().probabilities().matrix().isApprox(reference, 1e-14));
    EXPECT_NEAR(filter.logEvidence(), logEvidence, 1e-13);
}

namespace
{

// A two-state Markov-modulated Poisson process: a quiet state at rate 0.5 and a busy one at 6,
// switching at rates 0.4 and 1.1, and a record of eight arrivals and a closing gap without one,
// its times in hundredths so that a grid of 0.01 or 0.001 holds every arrival on a boundary.
Eigen::Matrix2d const mmppGenerator{{-0.4, 0.4}, {1.1, -1.1}};

Eigen::Array2d const mmppRates{0.5, 6.0};
std::array<int, 8> const mmppGaps{37, 112, 5, 240, 81, 30, 190, 14};
int const mmppClosingGap = 63;

}  // namespace

// Over a record of arrivals, the filter stepping through each inter-arrival interval accumulates
// the log of the likelihood π [Π_k e^{(Q - Λ)τ_k} Λ] e^{(Q - Λ)τ_end} 1 of Fischer and
// Meier-Hellstern (1993), here evaluated by plain-Eigen matrix exponentials, rescaling after each
// arrival; and its state is the normalized row vector of that product so far.
TEST(HiddenMarkovFilter, ModulatedPoissonMatchesTheClosedFormLikelihood)
{
    auto const chain = ContinuousTimeMarkovChain<Weather>{mmppGenerator};
    auto const initial = CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0.7, 0.3});
    auto filter = HiddenMarkovFilter{chain, initial};

    Eigen::Matrix2d const rates = mmppRates.matrix().asDiagonal();
    Eigen::Matrix2d const subGenerator = mmppGenerator - rates;
    auto forward = Eigen::RowVector2d{0.7, 0.3};
    double logLikelihood = 0;
    for (int const gap : mmppGaps)
    {
        double const tau = gap / 100.0;
        ASSERT_TRUE(filter.predict(tau, ArrivalInterval<Weather>{mmppRates}));
        forward = forward * Eigen::Matrix2d{(subGenerator * tau).exp()} * rates;
        logLikelihood += std::log(forward.sum());
        forward /= forward.sum();
        EXPECT_NEAR(filter.state().toDensity().value().probability<Rain>(), forward(0), 1e-12);
    }
    ASSERT_TRUE(filter.predict(mmppClosingGap / 100.0, ArrivalInterval<Weather>{mmppRates, false}));
    forward = forward * Eigen::Matrix2d{(subGenerator * (mmppClosingGap / 100.0)).exp()};
    logLikelihood += std::log(forward.sum());
    EXPECT_NEAR(filter.logEvidence(), logLikelihood, 1e-11);
}

// The approximate filter predicts over short steps of h and counts the arrivals in each with a
// PoissonEmission. A count is a probability where an arrival time is a density, so its
// log-evidence exceeds the exact one by the number of arrivals times log h, and what is left is
// the error of splitting e^{(Q - Λ)h} into e^{Qh} e^{-Λh}, of first order in h. The tolerances
// are the errors measured on Clang, rounded up: 5.8e-4 at h = 0.01 and 8.8e-5 at h = 0.001, a
// ratio of 6.6 for a tenfold refinement, which is that first order.
TEST(HiddenMarkovFilter, CountedArrivalsConvergeToTheExactFilter)
{
    auto const chain = ContinuousTimeMarkovChain<Weather>{mmppGenerator};
    auto const initial = CategoricalParams<Weather>::fromWeights(Weather{}, Eigen::Vector2d{0.7, 0.3});
    auto exact = HiddenMarkovFilter{chain, initial};
    for (int const gap : mmppGaps)
    {
        ASSERT_TRUE(exact.predict(gap / 100.0, ArrivalInterval<Weather>{mmppRates}));
    }
    ASSERT_TRUE(exact.predict(mmppClosingGap / 100.0, ArrivalInterval<Weather>{mmppRates, false}));

    auto errors = std::array<double, 2>{};
    for (std::size_t const refinement : {0, 1})
    {
        int const stepsPerHundredth = refinement == 0 ? 1 : 10;
        double const h = 0.01 / stepsPerHundredth;
        auto counted = HiddenMarkovFilter{chain, initial};
        auto const countStep = [&](Eigen::Index count) {
            counted.predict(h);
            ASSERT_TRUE(counted.update(PoissonEmission<Weather>{mmppRates, count, h}));
        };
        for (int const gap : mmppGaps)
        {
            for (int s = 1; s < gap * stepsPerHundredth; ++s)
            {
                countStep(0);
            }
            countStep(1);
        }
        for (int s = 0; s < mmppClosingGap * stepsPerHundredth; ++s)
        {
            countStep(0);
        }
        double const arrivals = double(mmppGaps.size());
        errors[refinement] = std::abs(counted.logEvidence() - arrivals * std::log(h) - exact.logEvidence());
        EXPECT_NEAR(
            counted.state().toDensity().value().probability<Rain>(),
            exact.state().toDensity().value().probability<Rain>(),
            0.05 / stepsPerHundredth
        );
    }
    EXPECT_LT(errors[0], 1e-3);
    EXPECT_LT(errors[1], 1.5e-4);
    EXPECT_GT(errors[0] / errors[1], 5.0);
}

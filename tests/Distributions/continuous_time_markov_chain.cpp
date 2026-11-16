// The continuous-time chain on a discrete space (step 15, [emissions]): ContinuousTimeMarkovChain built
// from a generator's rates, with the ChainError of rates that fail to define one, as persistent, from mean sojourn
// times and exit weights, and from independent axes of a product by the Kronecker sum; its transition over an interval
// through the matrix exponential; predict() over an interval; and its stationary distribution. Header under test:
// Distributions/ContinuousTimeMarkovChain.h.

#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using Emrys::CategoricalParams;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::DiscreteProduct;
using Emrys::Finite;
using Emrys::MarkovChain;
using Emrys::Named;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

struct Cruise
{};

struct Turn
{};

struct Maneuver : Named<Cruise, Turn>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Named<Inlier, Outlier>
{};

using Mode = DiscreteProduct<Maneuver, Sensor>;

}  // namespace

// A two-state chain leaving the first value at rate a and the second at rate b stays in the first
// over t with probability (b + a e^{-(a+b)t}) / (a+b), the closed form of the two-state chain's
// forward equation, and in the second with (a + b e^{-(a+b)t}) / (a+b).
TEST(ContinuousTimeMarkovChain, TwoStateTransitionMatchesClosedForm)
{
    double const a = 0.3;
    double const b = 1.2;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{-a, a}, {b, -b}}).value();
    for (double const t : {0.0, 0.1, 1.0, 7.5})
    {
        auto const step = chain.transition(t);
        double const decay = std::exp(-(a + b) * t);
        EXPECT_NEAR(step.transition(0, 0), (b + a * decay) / (a + b), 1e-13);
        EXPECT_NEAR(step.transition(0, 1), a * (1 - decay) / (a + b), 1e-13);
        EXPECT_NEAR(step.transition(1, 1), (a + b * decay) / (a + b), 1e-13);
        EXPECT_NEAR(step.transition(1, 0), b * (1 - decay) / (a + b), 1e-13);
    }
}

// persistent(r) on n values leaves at total rate r, to each other value at r / (n - 1), so the
// diagonal of the generator is -r and each row sums to zero. Over t the probability of staying
// is 1/n + (n-1)/n e^{-n r t / (n-1)}, the closed form for a chain whose off-diagonal rates are equal.
TEST(ContinuousTimeMarkovChain, PersistentLeavesAtItsRate)
{
    auto const space = Finite<4>{};
    auto const chain = ContinuousTimeMarkovChain<Finite<4>>::persistent(2.0, space);
    EXPECT_DOUBLE_EQ(chain.rate(1, 1), -2.0);
    EXPECT_DOUBLE_EQ(chain.rate(1, 3), 2.0 / 3.0);
    EXPECT_NEAR(chain.generator().rowwise().sum().cwiseAbs().maxCoeff(), 0.0, 1e-15);
    double const t = 0.4;
    auto const step = chain.transition(t);
    double const stay = 0.25 + 0.75 * std::exp(-4.0 * 2.0 * t / 3.0);
    EXPECT_NEAR(step.transition(2, 2), stay, 1e-13);
    EXPECT_NEAR(step.transition(2, 0), (1 - stay) / 3, 1e-13);
}

// Independent axes never jump together, so the product chain's generator is the Kronecker sum
// Q_m ⊗ I + I ⊗ Q_s, and its transition over any interval is the Kronecker product of the axes'
// transitions, which the discrete MarkovChain::independent() builds.
TEST(ContinuousTimeMarkovChain, ProductIsTheKroneckerSum)
{
    auto const maneuver =
        ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{-0.1, 0.1}, {0.4, -0.4}}).value();
    auto const sensor = ContinuousTimeMarkovChain<Sensor>::create(Eigen::Matrix2d{{-2.0, 2.0}, {5.0, -5.0}}).value();
    auto const mode = ContinuousTimeMarkovChain<Mode>::independent(maneuver, sensor);

    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int const mi = i / 2;
            int const si = i % 2;
            int const mj = j / 2;
            int const sj = j % 2;
            expected(i, j) = (si == sj ? maneuver.rate(mi, mj) : 0.0) + (mi == mj ? sensor.rate(si, sj) : 0.0);
        }
    }
    EXPECT_TRUE(mode.generator().isApprox(expected, 1e-15));

    double const t = 0.7;
    auto const product = MarkovChain<Mode>::independent(maneuver.transition(t), sensor.transition(t));
    EXPECT_TRUE(mode.transition(t).matrix().isApprox(product.matrix(), 1e-12));
}

// predict() over an interval is the prediction through the transition over it, and over a long
// interval it reaches the stationary distribution (b, a) / (a + b) of the two-state chain.
TEST(ContinuousTimeMarkovChain, PredictMovesOverTheInterval)
{
    double const a = 0.3;
    double const b = 1.2;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{-a, a}, {b, -b}}).value();
    auto const prior = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.1, 0.9});
    auto const predicted = predict(prior, chain, 0.5);
    auto const through = predict(prior, chain.transition(0.5));
    EXPECT_TRUE(predicted.logWeights().isApprox(through.logWeights(), 1e-15));
    auto const stationaryWeights = predict(prior, chain, 100.0);
    auto const stationary = stationaryWeights.toDensity().value();
    EXPECT_NEAR(stationary.probability<Cruise>(), b / (a + b), 1e-12);
    EXPECT_NEAR(stationary.probability<Turn>(), a / (a + b), 1e-12);
}

// fromSojournTimes() leaves value i at the rate 1/tau_i, so the exponential time there has mean
// tau_i, and splits the rate by the exit weights: the generator is computed by hand. Without exit
// weights it splits evenly, as persistent() does with each value's own rate.
TEST(ContinuousTimeMarkovChain, FromSojournTimes)
{
    auto const exits = Eigen::Matrix2d{{0, 1}, {1, 0}};
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{20.0, 2.5}, exits).value();
    EXPECT_TRUE(chain.generator().isApprox(Eigen::Matrix2d{{-0.05, 0.05}, {0.4, -0.4}}, 1e-15));

    auto const even = ContinuousTimeMarkovChain<Finite<3>>::fromSojournTimes(Eigen::Vector3d{0.5, 0.5, 0.5}).value();
    EXPECT_TRUE(even.generator().isApprox(ContinuousTimeMarkovChain<Finite<3>>::persistent(2.0).generator(), 1e-15));

    auto const flight = ContinuousTimeMarkovChain<Finite<3>>::fromSojournTimes(
                            Eigen::Vector3d{30.0, 8.0, 4.0},
                            Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}}
    )
                            .value();
    auto const expected =
        Eigen::Matrix3d{{-1.0 / 30, 2.0 / 90, 1.0 / 90}, {1.0 / 8, -1.0 / 8, 0}, {1.0 / 4, 0, -1.0 / 4}};
    EXPECT_TRUE(flight.generator().isApprox(expected, 1e-15));
}

// create() reads only the rates off the diagonal and sets each diagonal entry to minus its row's
// total, so a generator, one with any diagonal, and one written to a few digits give the same chain,
// whose rows sum to zero.
TEST(ContinuousTimeMarkovChain, CreateSetsTheDiagonal)
{
    auto const generator = Eigen::Matrix3d{{-0.3, 0.1, 0.2}, {0.4, -0.4, 0}, {0.5, 0.25, -0.75}};
    auto const chain = ContinuousTimeMarkovChain<Finite<3>>::create(generator).value();
    EXPECT_TRUE(chain.generator().isApprox(generator, 1e-15));

    Eigen::Matrix3d rates = generator;
    rates.diagonal() = Eigen::Vector3d{0, 7, NAN};
    EXPECT_TRUE(ContinuousTimeMarkovChain<Finite<3>>::create(rates).value().generator().isApprox(generator, 1e-15));

    auto const rounded =
        ContinuousTimeMarkovChain<Finite<3>>::create(Eigen::Matrix3d{{-0.67, 0.333, 0.333}, {1, -1, 0}, {0, 0, 0}});
    EXPECT_DOUBLE_EQ(rounded.value().rate(0, 0), -0.666);
    EXPECT_EQ(rounded.value().rate(2, 2), 0.0);
    EXPECT_TRUE(rounded.value().generator().rowwise().sum().isZero(1e-15));
}

// create() and fromSojournTimes() return the ChainError of values that fail to define a chain: a
// negative rate or exit weight, a rate or sojourn time that is NaN, an infinite rate, a sojourn
// time that is not positive, and a value that is left with nowhere to go.
TEST(ContinuousTimeMarkovChain, CreateAndFromSojournTimesCheckTheirValues)
{
    using Emrys::ChainError;
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, -1}, {1, 0}}).error(),
        ChainError::Negative
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, NAN}, {1, 0}}).error(),
        ChainError::NonFinite
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{0, 1}, {inf, 0}}).error(),
        ChainError::NonFinite
    );

    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{0, 1}).error(),
        ChainError::ShortSojourn
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{-2, 1}).error(),
        ChainError::ShortSojourn
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{NAN, 1}).error(),
        ChainError::NonFinite
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{2, 1}, Eigen::Matrix2d{{0, 0}, {1, 0}})
            .error(),
        ChainError::ZeroRow
    );
    EXPECT_EQ(
        ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{2, 1}, Eigen::Matrix2d{{0, inf}, {1, 0}})
            .error(),
        ChainError::NonFinite
    );
    EXPECT_TRUE(ContinuousTimeMarkovChain<Maneuver>::fromSojournTimes(Eigen::Vector2d{0.25, inf}));
}

// stationary() gives the probabilities that pi Q = 0: (b, a) / (a + b) for the two-state chain,
// the same as the stationary distribution of its transition over any interval, and the time
// fractions 15 : 8/3 : 2/3 of the three-value chain of FromSojournTimes, its visits 1/2, 1/3, 1/6
// times its mean times 30, 8, 4. A chain that never moves is Reducible.
TEST(ContinuousTimeMarkovChain, Stationary)
{
    double const a = 0.3;
    double const b = 1.2;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::create(Eigen::Matrix2d{{-a, a}, {b, -b}}).value();
    auto const prior = stationary(chain);
    ASSERT_TRUE(prior);
    Eigen::Array2d const p = prior->toDensity().value().probabilities();
    EXPECT_NEAR(p(0), b / (a + b), 1e-15);
    EXPECT_NEAR(p(1), a / (a + b), 1e-15);
    auto const stepped = stationary(chain.transition(0.7));
    ASSERT_TRUE(stepped);
    EXPECT_TRUE(stepped->toDensity().value().probabilities().isApprox(p, 1e-14));

    auto const flight = ContinuousTimeMarkovChain<Finite<3>>::fromSojournTimes(
                            Eigen::Vector3d{30.0, 8.0, 4.0},
                            Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}}
    )
                            .value();
    auto const fractions = stationary(flight);
    ASSERT_TRUE(fractions);
    EXPECT_TRUE(
        fractions->toDensity().value().probabilities().isApprox(Eigen::Array3d{9.0 / 11, 8.0 / 55, 2.0 / 55}, 1e-14)
    );

    EXPECT_EQ(stationary(ContinuousTimeMarkovChain<Maneuver>{}).error(), Emrys::ChainError::Reducible);
}

// A run-time cardinality takes a dynamic generator and gives a dynamic transition; zero time is
// the identity.
TEST(ContinuousTimeMarkovChain, RunTimeCardinality)
{
    auto const space = Finite<Eigen::Dynamic, 8>{3};
    auto const chain = ContinuousTimeMarkovChain<Finite<Eigen::Dynamic, 8>>::persistent(1.0, space);
    EXPECT_EQ(chain.generator().rows(), 3);
    EXPECT_TRUE(chain.transition(0.0).matrix().isApprox(Eigen::Matrix3d::Identity()));
    EXPECT_NEAR(chain.transition(0.3).matrix().rowwise().sum().maxCoeff(), 1.0, 1e-15);
}

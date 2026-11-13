// The continuous-time chain on a discrete space (step 15, [emissions]): ContinuousTimeMarkovChain built
// from a generator, as persistent, from mean dwell times and exit weights, and from independent axes
// of a product by the Kronecker sum; its transition over an interval through the matrix exponential;
// predict() over an interval; and its stationary distribution.
// Header under test: Distributions/ContinuousTimeMarkovChain.h.

#include <EmrysFilter/Distributions/ContinuousTimeMarkovChain.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Spaces/Discrete.h>

#include <gtest/gtest.h>

#include <cmath>

using Emrys::CategoricalParams;
using Emrys::ContinuousTimeMarkovChain;
using Emrys::DiscreteProduct;
using Emrys::Finite;
using Emrys::MarkovChain;
using Emrys::Named;

namespace
{

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
    auto const chain = ContinuousTimeMarkovChain<Maneuver>{Eigen::Matrix2d{{-a, a}, {b, -b}}};
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
// transitions, which the discrete MarkovChain's product constructor builds.
TEST(ContinuousTimeMarkovChain, ProductIsTheKroneckerSum)
{
    auto const maneuver = ContinuousTimeMarkovChain<Maneuver>{Eigen::Matrix2d{{-0.1, 0.1}, {0.4, -0.4}}};
    auto const sensor = ContinuousTimeMarkovChain<Sensor>{Eigen::Matrix2d{{-2.0, 2.0}, {5.0, -5.0}}};
    auto const mode = ContinuousTimeMarkovChain<Mode>{maneuver, sensor};

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
    auto const product = MarkovChain<Mode>{maneuver.transition(t), sensor.transition(t)};
    EXPECT_TRUE(mode.transition(t).matrix().isApprox(product.matrix(), 1e-12));
}

// predict() over an interval is the prediction through the transition over it, and over a long
// interval it reaches the stationary distribution (b, a) / (a + b) of the two-state chain.
TEST(ContinuousTimeMarkovChain, PredictMovesOverTheInterval)
{
    double const a = 0.3;
    double const b = 1.2;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>{Eigen::Matrix2d{{-a, a}, {b, -b}}};
    auto const prior = CategoricalParams<Maneuver>::fromWeights(Maneuver{}, Eigen::Vector2d{0.1, 0.9});
    auto const predicted = predict(prior, chain, 0.5);
    auto const through = predict(prior, chain.transition(0.5));
    EXPECT_TRUE(predicted.logWeights().isApprox(through.logWeights(), 1e-15));
    auto const stationaryWeights = predict(prior, chain, 100.0);
    auto const stationary = stationaryWeights.toDensity().value();
    EXPECT_NEAR(stationary.probability<Cruise>(), b / (a + b), 1e-12);
    EXPECT_NEAR(stationary.probability<Turn>(), a / (a + b), 1e-12);
}

// fromDwellTimes() leaves value i at the rate 1/tau_i, so the exponential time there has mean
// tau_i, and splits the rate by the exit weights: the generator is computed by hand. Without exit
// weights it splits evenly, as persistent() does with each value's own rate.
TEST(ContinuousTimeMarkovChain, FromDwellTimes)
{
    auto const exits = Eigen::Matrix2d{{0, 1}, {1, 0}};
    auto const chain = ContinuousTimeMarkovChain<Maneuver>::fromDwellTimes(Eigen::Vector2d{20.0, 2.5}, exits);
    EXPECT_TRUE(chain.generator().isApprox(Eigen::Matrix2d{{-0.05, 0.05}, {0.4, -0.4}}, 1e-15));

    auto const even = ContinuousTimeMarkovChain<Finite<3>>::fromDwellTimes(Eigen::Vector3d{0.5, 0.5, 0.5});
    EXPECT_TRUE(even.generator().isApprox(ContinuousTimeMarkovChain<Finite<3>>::persistent(2.0).generator(), 1e-15));

    auto const flight = ContinuousTimeMarkovChain<Finite<3>>::fromDwellTimes(
        Eigen::Vector3d{30.0, 8.0, 4.0},
        Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}}
    );
    auto const expected =
        Eigen::Matrix3d{{-1.0 / 30, 2.0 / 90, 1.0 / 90}, {1.0 / 8, -1.0 / 8, 0}, {1.0 / 4, 0, -1.0 / 4}};
    EXPECT_TRUE(flight.generator().isApprox(expected, 1e-15));
}

// stationary() gives the probabilities that pi Q = 0: (b, a) / (a + b) for the two-state chain,
// the same as the stationary distribution of its transition over any interval, and the time
// fractions 15 : 8/3 : 2/3 of the three-value chain of FromDwellTimes, its visits 1/2, 1/3, 1/6
// times its mean times 30, 8, 4. A chain that never moves is Reducible.
TEST(ContinuousTimeMarkovChain, Stationary)
{
    double const a = 0.3;
    double const b = 1.2;
    auto const chain = ContinuousTimeMarkovChain<Maneuver>{Eigen::Matrix2d{{-a, a}, {b, -b}}};
    auto const prior = stationary(chain);
    ASSERT_TRUE(prior);
    Eigen::Array2d const p = prior->toDensity().value().probabilities();
    EXPECT_NEAR(p(0), b / (a + b), 1e-15);
    EXPECT_NEAR(p(1), a / (a + b), 1e-15);
    auto const stepped = stationary(chain.transition(0.7));
    ASSERT_TRUE(stepped);
    EXPECT_TRUE(stepped->toDensity().value().probabilities().isApprox(p, 1e-14));

    auto const flight = ContinuousTimeMarkovChain<Finite<3>>::fromDwellTimes(
        Eigen::Vector3d{30.0, 8.0, 4.0},
        Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}}
    );
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

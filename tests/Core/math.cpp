// The constants and scalar functions of Core/Math.h, a follow-up to roadmap step 13b's review
// findings: pi(), log2Pi(), scalarFromCount(), logGamma(), logOnePlus(), floorOf(), isFinite(), and isNaN() in
// each scalar kind they serve, a floating-point type, Eigen's AutoDiffScalar, and a scalar whose
// literal type is itself, standing in for a multiprecision type; and compileTimeSqrt(), in double.

#include <EmrysFilter/Core/Math.h>

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include <unsupported/Eigen/AutoDiff>

namespace
{

// A scalar that is not floating point and whose NumTraits literal is itself, as a multiprecision
// type's is. It carries a long double and counts the calls of its own atan and lgamma, so a test
// can tell that the library computed through them, and has a log for log2Pi().
struct Extended
{
    long double value{};

    Extended() = default;

    Extended(long double v)
        : value{v}
    {}

    friend Extended
    operator*(Extended a, Extended b)
    {
        return Extended{a.value * b.value};
    }
};

int atanCalls = 0;
int lgammaCalls = 0;

Extended
atan(Extended x)
{
    ++atanCalls;
    return Extended{std::atan(x.value)};
}

Extended
lgamma(Extended x)
{
    ++lgammaCalls;
    return Extended{std::lgamma(x.value)};
}

Extended
log(Extended x)
{
    return Extended{std::log(x.value)};
}

}  // namespace

template <>
struct Eigen::NumTraits<Extended> : Eigen::GenericNumTraits<Extended>
{
    using Real = Extended;
    using NonInteger = Extended;
    using Nested = Extended;
    using Literal = Extended;
};

namespace
{

using Dual = Eigen::AutoDiffScalar<Eigen::Vector2d>;

static_assert(Emrys::internal::isAutoDiffScalar<Dual>);
static_assert(!Emrys::internal::isAutoDiffScalar<double>);
static_assert(!Emrys::internal::isAutoDiffScalar<Extended>);

}  // namespace

// A floating-point scalar gets std::numbers::pi_v of its own type, exactly.
TEST(Math, PiOfAFloatingPointScalarIsTheStandardConstant)
{
    EXPECT_EQ(Emrys::pi<float>(), std::numbers::pi_v<float>);
    EXPECT_EQ(Emrys::pi<double>(), std::numbers::pi);
    EXPECT_EQ(Emrys::pi<long double>(), std::numbers::pi_v<long double>);
}

// log2Pi() of a floating-point scalar is log 2 pi correctly rounded to its type: the expected
// literals are 40 digits of log 2 pi from a 60-digit decimal computation, which the compiler rounds
// once, and casting the library's long double literal rounds twice to the same value for float and
// double (checked in exact rational arithmetic). std::log(2 * pi) is one ulp low in double, since
// pi is rounded before the log, so it is no reference.
TEST(Math, Log2PiOfAFloatingPointScalarIsCorrectlyRounded)
{
    EXPECT_EQ(Emrys::log2Pi<float>(), 1.837877066409345483560659472811235279723F);
    EXPECT_EQ(Emrys::log2Pi<double>(), 1.837877066409345483560659472811235279723);
    EXPECT_EQ(Emrys::log2Pi<long double>(), 1.837877066409345483560659472811235279723L);
    EXPECT_EQ(Emrys::log2Pi<double>(), std::nextafter(std::log(2 * std::numbers::pi), 2.0));
}

// An AutoDiffScalar, whose literal is double, gets pi_v<double> with zero derivatives: a constant
// does not depend on the variables.
TEST(Math, PiOfAnAutoDiffScalarIsAConstant)
{
    Dual const pi = Emrys::pi<Dual>();
    EXPECT_EQ(pi.value(), std::numbers::pi);
    EXPECT_EQ(pi.derivatives(), Eigen::Vector2d::Zero());
    Dual const log2Pi = Emrys::log2Pi<Dual>();
    EXPECT_EQ(log2Pi.value(), Emrys::log2Pi<double>());
    EXPECT_EQ(log2Pi.derivatives(), Eigen::Vector2d::Zero());
}

// A scalar whose literal is itself takes the log of twice its own pi, computed in its precision
// rather than taken from a double or long double literal.
TEST(Math, Log2PiOfAnotherScalarComesFromItsOwnPi)
{
    atanCalls = 0;
    Extended const log2Pi = Emrys::log2Pi<Extended>();
    EXPECT_EQ(atanCalls, 1);
    EXPECT_EQ(log2Pi.value, std::log(2 * (4 * std::atan(1.0L))));
}

// A scalar whose literal is itself computes pi as 4 atan 1 through its own atan, so it carries the
// type's precision rather than double's.
TEST(Math, PiOfAnotherScalarComesFromItsOwnArctangent)
{
    atanCalls = 0;
    Extended const pi = Emrys::pi<Extended>();
    EXPECT_EQ(atanCalls, 1);
    EXPECT_EQ(pi.value, 4 * std::atan(1.0L));
}

// scalarFromCount() converts an integer exactly in each scalar kind, and an AutoDiffScalar's count
// is a constant, with zero derivatives. Under MSVC's /W4 the Dual case is the one that used to warn
// (C4244) when written Scalar(count).
TEST(Math, ScalarFromCountInEachScalarKind)
{
    Eigen::Index const count = 3;
    EXPECT_EQ(Emrys::internal::scalarFromCount<double>(count), 3.0);
    EXPECT_EQ(Emrys::internal::scalarFromCount<float>(count), 3.0F);
    Dual const dual = Emrys::internal::scalarFromCount<Dual>(count);
    EXPECT_EQ(dual.value(), 3.0);
    EXPECT_EQ(dual.derivatives(), Eigen::Vector2d::Zero());
    EXPECT_EQ(Emrys::internal::scalarFromCount<Extended>(3).value, 3.0L);
}

// compileTimeSqrt() is exact on squares, above and below 1, is 0 at and below 0, and is within an
// ulp of std::sqrt elsewhere, over the arguments the methods give it: a dimension, and n + lambda.
TEST(Math, CompileTimeSqrtIsWithinAnUlpOfTheStandardOne)
{
    static_assert(Emrys::internal::compileTimeSqrt(4.0) == 2.0);
    static_assert(Emrys::internal::compileTimeSqrt(0.25) == 0.5);
    static_assert(Emrys::internal::compileTimeSqrt(0.0) == 0.0);
    static_assert(Emrys::internal::compileTimeSqrt(-1.0) == 0.0);
    constexpr std::array arguments{2.0, 3.0, 0.5, 6.0, 1e-3, 4096.0, 1e12};
    constexpr std::array roots{
        Emrys::internal::compileTimeSqrt(arguments[0]),
        Emrys::internal::compileTimeSqrt(arguments[1]),
        Emrys::internal::compileTimeSqrt(arguments[2]),
        Emrys::internal::compileTimeSqrt(arguments[3]),
        Emrys::internal::compileTimeSqrt(arguments[4]),
        Emrys::internal::compileTimeSqrt(arguments[5]),
        Emrys::internal::compileTimeSqrt(arguments[6]),
    };
    for (std::size_t i = 0; i < arguments.size(); ++i)
    {
        double const expected = std::sqrt(arguments[i]);
        EXPECT_GE(roots[i], std::nextafter(expected, 0.0)) << arguments[i];
        EXPECT_LE(roots[i], std::nextafter(expected, 2 * expected)) << arguments[i];
    }
}

// logGamma() is std::lgamma for floating-point scalars, the scalar's own lgamma otherwise, and for
// an AutoDiffScalar the value's lgamma with derivatives scaled by the digamma function. The
// expected digamma values are psi(1) = -gamma (the Euler-Mascheroni constant) and
// psi(1/2) = -gamma - 2 log 2, which Eigen's digamma reaches to a few ulp.
TEST(Math, LogGammaInEachScalarKind)
{
    EXPECT_EQ(Emrys::logGamma<double>(2.5), std::lgamma(2.5));
    EXPECT_EQ(Emrys::logGamma<long double>(2.5L), std::lgamma(2.5L));

    lgammaCalls = 0;
    EXPECT_EQ(Emrys::logGamma<Extended>(Extended{2.5L}).value, std::lgamma(2.5L));
    EXPECT_EQ(lgammaCalls, 1);

    auto const one = Dual{1.0, Eigen::Vector2d{1.0, 2.0}};
    Dual const logGammaOfOne = Emrys::logGamma<Dual>(one);
    EXPECT_EQ(logGammaOfOne.value(), 0.0);
    EXPECT_NEAR(logGammaOfOne.derivatives()(0), -std::numbers::egamma, 1e-14);
    EXPECT_NEAR(logGammaOfOne.derivatives()(1), -2 * std::numbers::egamma, 1e-14);

    auto const half = Dual{0.5, Eigen::Vector2d{1.0, 0.0}};
    Dual const logGammaOfHalf = Emrys::logGamma<Dual>(half);
    EXPECT_NEAR(logGammaOfHalf.value(), 0.5 * std::log(std::numbers::pi), 1e-15);
    EXPECT_NEAR(logGammaOfHalf.derivatives()(0), -std::numbers::egamma - 2 * std::numbers::ln2, 1e-14);
    EXPECT_EQ(logGammaOfHalf.derivatives()(1), 0.0);
}

// logOnePlus() of an AutoDiffScalar has the value log1p(x) and derivatives scaled by 1 / (1 + x),
// keeping log1p's accuracy for small x, and an expression of Duals converts to a Dual first.
TEST(Math, LogOnePlusCarriesDerivatives)
{
    EXPECT_EQ(Emrys::logOnePlus<double>(1e-20), 1e-20);
    auto const x = Dual{1e-20, Eigen::Vector2d{1.0, 0.0}};
    Dual const small = Emrys::logOnePlus<Dual>(x);
    EXPECT_EQ(small.value(), 1e-20);
    EXPECT_EQ(small.derivatives()(0), 1.0);
    auto const y = Dual{1.0, Eigen::Vector2d{0.0, 1.0}};
    Dual const sum = Emrys::logOnePlus<Dual>(x + y);
    EXPECT_EQ(sum.value(), std::log1p(1.0 + 1e-20));
    EXPECT_EQ(sum.derivatives(), Eigen::Vector2d(0.5, 0.5));
}

// floorOf() is std::floor for floating-point scalars, numext::floor otherwise, and for an
// AutoDiffScalar the value's floor with zero derivatives, an expression of Duals converting to a
// Dual first.
TEST(Math, FloorOfInEachScalarKind)
{
    EXPECT_EQ(Emrys::floorOf<double>(-2.5), -3.0);
    EXPECT_EQ(Emrys::floorOf<float>(2.5f), 2.0f);
    EXPECT_EQ(Emrys::floorOf<long double>(2.5L), 2.0L);

    auto const x = Dual{-2.5, Eigen::Vector2d{1.0, 0.0}};
    auto const y = Dual{5.0, Eigen::Vector2d{0.0, 1.0}};
    Dual const floor = Emrys::floorOf<Dual>(x + y);
    EXPECT_EQ(floor.value(), 2.0);
    EXPECT_EQ(floor.derivatives(), Eigen::Vector2d::Zero());
}

// isFinite() and isNaN() of an AutoDiffScalar ask of its value, which Eigen's numext::isfinite
// and numext::isnan cannot do for it.
TEST(Math, FinitenessOfAnAutoDiffScalarIsItsValues)
{
    auto const finite = Dual{2.0, Eigen::Vector2d{1.0, 0.0}};
    auto const infinite = Dual{std::numeric_limits<double>::infinity(), Eigen::Vector2d{1.0, 0.0}};
    auto const nan = Dual{std::numeric_limits<double>::quiet_NaN(), Eigen::Vector2d{1.0, 0.0}};
    EXPECT_TRUE(Emrys::isFinite(finite));
    EXPECT_FALSE(Emrys::isFinite(infinite));
    EXPECT_FALSE(Emrys::isFinite(nan));
    EXPECT_FALSE(Emrys::isNaN(infinite));
    EXPECT_TRUE(Emrys::isNaN(nan));
    EXPECT_TRUE(Emrys::isFinite(finite * finite));
    EXPECT_FALSE(Emrys::isFinite(infinite * finite));

    auto matrix = Eigen::Matrix<Dual, 2, 2>{{finite, finite}, {finite, finite}};
    EXPECT_TRUE(Emrys::allFinite(matrix));
    matrix(1, 1) = nan;
    EXPECT_FALSE(Emrys::allFinite(matrix));
}

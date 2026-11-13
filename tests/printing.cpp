// Printing (Core/Print.h and the describe() overloads beside each type): what operator<< writes for
// the Gaussian's forms, mixtures, particle sets, categoricals, the error codes, and a filter, which
// names itself, that std::format writes the same with a string's width and alignment, that a
// description summarizes a particle set rather than listing it, that invalid weights and a singular
// precision print their error instead of asserting, and that only the library's types are
// Printable. The Rao-Blackwellized and hybrid states are printed in their filters' tests.

#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Fitting.h>
#include <EmrysFilter/Distributions/Gamma.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/ParticleSet.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Distributions/StudentT.h>
#include <EmrysFilter/Distributions/Uniform.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/ParticleFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/UnitQuaternion.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Emrys::CategoricalParams;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::Finite;
using Emrys::FitError;
using Emrys::Gaussian;
using Emrys::GaussianError;
using Emrys::GaussianMixtureParams;
using Emrys::GaussianParams;
using Emrys::InfoGaussianParams;
using Emrys::KalmanFilter;
using Emrys::Linear;
using Emrys::MomentsError;
using Emrys::ParticleSet;
using Emrys::UnitQuaternion;
using Emrys::WeightError;
using Emrys::internal::Printable;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

// What operator<< writes, as a string.
template <typename T>
std::string
printed(T const & value)
{
    auto stream = std::ostringstream{};
    stream << value;
    return stream.str();
}

GaussianParams<Linear<2>>
makeGaussian()
{
    return {Eigen::Vector2d{1, -2}, Eigen::Matrix2d{{4, 1}, {1, 9}}};
}

}  // namespace

// The library prints its own types and no others: not Eigen's matrices, not a container of its
// types, and not a plain value, so its operator<< and std::formatter claim nothing they shouldn't.
TEST(Printing, OnlyTheLibrarysTypesArePrintable)
{
    static_assert(Printable<GaussianParams<Linear<2>>>);
    static_assert(Printable<GaussianMixtureParams<Linear<2>>>);
    static_assert(Printable<ParticleSet<Linear<2>>>);
    static_assert(Printable<WeightError>);
    static_assert(Printable<KalmanFilter<Gaussian<State>, ConstantVelocity<Position, Velocity>>>);
    static_assert(Printable<Emrys::InfoFilter<State, ConstantVelocity<Position, Velocity>>>);
    static_assert(!Printable<Eigen::Vector2d>);
    static_assert(!Printable<Eigen::Matrix2d>);
    static_assert(!Printable<std::vector<GaussianParams<Linear<2>>>>);
    static_assert(!Printable<double>);
    static_assert(!Printable<State>);
}

// A Gaussian prints its dimension on the first line, then its mean on one line and its covariance
// under a line of its own, each level two spaces deeper. The expected text is written by hand.
TEST(Printing, GaussianPrintsItsMoments)
{
    EXPECT_EQ(
        printed(makeGaussian()),
        "Gaussian of dimension 2\n"
        "  mean: 1 -2\n"
        "  covariance:\n"
        "    4 1\n"
        "    1 9"
    );
}

// std::format writes what operator<< writes, and a width and alignment pad the whole description
// as they would a string.
TEST(Printing, FormatMatchesTheStream)
{
    auto const gaussian = makeGaussian();
    std::string const text = printed(gaussian);
    EXPECT_EQ(std::format("{}", gaussian), text);
    std::string const padded = std::format("{:*>100}", gaussian);
    ASSERT_EQ(padded.size(), 100u);
    EXPECT_EQ(padded.substr(100 - text.size()), text);
    EXPECT_EQ(padded.front(), '*');
}

// The numbers follow the stream's precision, as Eigen's do.
TEST(Printing, NumbersFollowTheStreamsPrecision)
{
    auto const third =
        GaussianParams<Linear<1>>{Eigen::Matrix<double, 1, 1>{1.0 / 3}, Eigen::Matrix<double, 1, 1>{2.0 / 3}};
    auto stream = std::ostringstream{};
    stream << std::setprecision(3) << third;
    EXPECT_EQ(
        stream.str(),
        "Gaussian of dimension 1\n"
        "  mean: 0.333\n"
        "  covariance:\n"
        "    0.667"
    );
}

// A space whose differences have another size than its values says so: a unit quaternion has
// four coordinates and three degrees of freedom.
TEST(Printing, DifferenceDimensionIsNamedWhereItDiffers)
{
    auto const attitude = GaussianParams<UnitQuaternion>{Eigen::Vector4d{0, 0, 0, 1}, Eigen::Matrix3d::Identity()};
    std::string const text = printed(attitude);
    EXPECT_EQ(text.substr(0, text.find('\n')), "Gaussian of dimension 4 (3 in differences)");
}

// Every form of the Gaussian prints its mean and covariance under its own name; an information
// form with no information has none and says why.
TEST(Printing, FormsPrintTheirMomentsOrWhyNot)
{
    auto const gaussian = makeGaussian();
    std::string const moments = printed(gaussian).substr(printed(gaussian).find('\n'));
    auto const sqrt = Emrys::toSqrt(gaussian);
    ASSERT_TRUE(sqrt);
    EXPECT_EQ(printed(*sqrt), "square-root Gaussian of dimension 2" + moments);
    auto const info = Emrys::toInfo(gaussian);
    ASSERT_TRUE(info);
    std::string const infoText = printed(*info);
    EXPECT_EQ(infoText.substr(0, infoText.find('\n')), "information Gaussian of dimension 2");
    auto const sqrtInfo = Emrys::toSqrtInfo(gaussian);
    ASSERT_TRUE(sqrtInfo);
    std::string const rootText = printed(*sqrtInfo);
    EXPECT_EQ(rootText.substr(0, rootText.find('\n')), "square-root information Gaussian of dimension 2");

    auto const nothing =
        InfoGaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero()};
    EXPECT_EQ(
        printed(nothing),
        "information Gaussian of dimension 2\n"
        "  no moments: the precision is singular"
    );
}

// A mixture prints the moments of its collapse and then each component, nested, with its
// normalized weight. The collapse of N(0, 1) and N(4, 1) weighted 1:3 has mean 3 and variance
// 1 + 0.25 * 9 + 0.75 * 1 = 4, by hand.
TEST(Printing, MixtureNestsItsComponents)
{
    auto mixture = GaussianMixtureParams<Linear<1>>{Linear<1>{}};
    mixture.add(std::log(1.0), Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0});
    mixture.add(std::log(3.0), Eigen::Matrix<double, 1, 1>{4.0}, Eigen::Matrix<double, 1, 1>{1.0});
    EXPECT_EQ(
        printed(mixture),
        "mixture of 2 components of dimension 1\n"
        "  mean: 3\n"
        "  covariance:\n"
        "    4\n"
        "  component 0, weight 0.25:\n"
        "    Gaussian of dimension 1\n"
        "      mean: 0\n"
        "      covariance:\n"
        "        1\n"
        "  component 1, weight 0.75:\n"
        "    Gaussian of dimension 1\n"
        "      mean: 4\n"
        "      covariance:\n"
        "        1"
    );
}

// A mixture whose weights are all zero prints the error and each component's log-weight, rather
// than asserting in the collapse.
TEST(Printing, MixtureWithInvalidWeightsPrintsTheError)
{
    double const inf = std::numeric_limits<double>::infinity();
    auto mixture = GaussianMixtureParams<Linear<1>>{Linear<1>{}};
    mixture.add(-inf, Eigen::Matrix<double, 1, 1>{0.0}, Eigen::Matrix<double, 1, 1>{1.0});
    std::string const text = printed(mixture);
    EXPECT_NE(text.find("\n  every weight is zero\n"), std::string::npos) << text;
    EXPECT_NE(text.find("component 0, log-weight -inf:"), std::string::npos) << text;
}

// A particle set prints a summary of a fixed number of lines, however many particles it holds:
// its count, effective sample size, and weighted moments, which for the four corners of a square
// with equal weights are the center and the sample covariance diag(1, 1), by hand.
TEST(Printing, ParticleSetPrintsASummary)
{
    auto const corners = Eigen::Matrix<double, 2, 4>{{-1, 1, -1, 1}, {-1, -1, 1, 1}};
    auto const particles = ParticleSet<Linear<2>>{Linear<2>{}, corners};
    EXPECT_EQ(
        printed(particles),
        "particle set of 4 particles of dimension 2, effective sample size 4\n"
        "  mean: 0 0\n"
        "  covariance:\n"
        "    1 0\n"
        "    0 1"
    );

    Eigen::Matrix<double, 2, Eigen::Dynamic> const many = Eigen::Matrix<double, 2, Eigen::Dynamic>::Random(2, 10000);
    std::string const text = printed(ParticleSet<Linear<2>>{Linear<2>{}, many});
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 4);
}

// A particle set whose weights are all zero has no moments and prints why; its effective sample
// size is left out.
TEST(Printing, ParticleSetWithInvalidWeightsPrintsTheError)
{
    double const inf = std::numeric_limits<double>::infinity();
    auto const particles =
        ParticleSet<Linear<1>>{Linear<1>{}, Eigen::Matrix<double, 1, 2>{0, 1}, Eigen::Array2d{-inf, -inf}};
    EXPECT_EQ(
        printed(particles),
        "particle set of 2 particles of dimension 1\n"
        "  no moments: every weight is zero"
    );
}

// A categorical prints its probabilities on one line.
TEST(Printing, CategoricalPrintsItsProbabilities)
{
    auto const categorical = CategoricalParams<Finite<3>>::fromWeights(Finite<3>{}, Eigen::Array3d{2, 3, 5});
    EXPECT_EQ(
        printed(categorical),
        "categorical over 3 values\n"
        "  probabilities: 0.2 0.3 0.5"
    );
}

// Parameters whose weights fail to define a distribution print their log-weights and why.
TEST(Printing, CategoricalWithInvalidWeightsPrintsLogWeights)
{
    auto const zero = CategoricalParams<Finite<2>>::fromWeights(Finite<2>{}, Eigen::Array2d{0, 0});
    EXPECT_EQ(
        printed(zero),
        "categorical over 2 values, whose weights are not normalizable\n"
        "  log-weights: -inf -inf"
    );
}

// The error codes print in words, alone or through std::format.
TEST(Printing, ErrorsPrintInWords)
{
    EXPECT_EQ(printed(WeightError::Empty), "no weights");
    EXPECT_EQ(printed(WeightError::ZeroSum), "every weight is zero");
    EXPECT_EQ(printed(WeightError::NonFinite), "a value is not finite");
    EXPECT_EQ(printed(GaussianError::NotPositiveDefinite), "the covariance is not positive definite");
    // GaussianError and WeightError are both NumericError, so NonFinite has one wording.
    static_assert(std::is_same_v<GaussianError, WeightError>);
    EXPECT_EQ(printed(GaussianError::Singular), "the precision is singular");
    EXPECT_EQ(printed(MomentsError::Infinite), "the moments are infinite");
    EXPECT_EQ(printed(Emrys::GammaError::NotPositive), "a shape or scale is not positive");
    EXPECT_EQ(printed(Emrys::GammaError::NonFinite), "a shape or scale is not finite");
    EXPECT_EQ(printed(Emrys::UniformError::Empty), "an upper bound is not above its lower bound");
    EXPECT_EQ(printed(Emrys::UniformError::NonFinite), "a bound is not finite");
    EXPECT_EQ(printed(Emrys::UniformError::WiderThanPeriod), "an interval is wider than its coordinate's period");
    EXPECT_EQ(printed(Emrys::StudentTError::NotPositiveDefinite), "the scale matrix is not positive definite");
    EXPECT_EQ(printed(Emrys::StudentTError::NonFiniteScale), "the scale matrix is not finite");
    EXPECT_EQ(printed(Emrys::StudentTError::NotPositiveDegreesOfFreedom), "the degrees of freedom are not positive");
    EXPECT_EQ(printed(Emrys::StudentTError::NonFiniteDegreesOfFreedom), "the degrees of freedom are not finite");
    EXPECT_EQ(printed(FitError::Degenerate), "the source has no spread in some direction");
    EXPECT_EQ(std::format("{}", FitError::InvalidWeights), "the weights cannot be normalized");
}

// A filter prints its name and then its state, nested a level under it, so that a log line says
// which filter wrote it; a filter derived from another, such as the information filter from the
// Kalman filter, names itself, through both operator<< and std::format. The Kalman filter's text
// is hand-written; the others' states are their own descriptions, one level deeper.
TEST(Printing, FiltersNameThemselvesAndNestTheirState)
{
    auto const nested = [](auto const & state) {
        auto stream = std::ostringstream{};
        Emrys::internal::describe(stream, state, Emrys::internal::Indent{1});
        return stream.str();
    };
    auto const model = ConstantVelocity<Position, Velocity>{0.1};
    auto const initial = GaussianParams<State>{State{}, Eigen::Vector4d{0, 0, 1, 0}, Eigen::Matrix4d::Identity()};
    auto const kalman = KalmanFilter{model, initial};
    EXPECT_EQ(
        printed(kalman),
        "Kalman filter\n"
        "  Gaussian of dimension 4\n"
        "    mean: 0 0 1 0\n"
        "    covariance:\n"
        "      1 0 0 0\n"
        "      0 1 0 0\n"
        "      0 0 1 0\n"
        "      0 0 0 1"
    );

    auto const info = Emrys::toInfo(initial);
    ASSERT_TRUE(info);
    auto const infoFilter = Emrys::InfoFilter{model, info.value()};
    EXPECT_EQ(printed(infoFilter), "information filter\n" + nested(infoFilter.state()));
    EXPECT_EQ(std::format("{}", infoFilter), printed(infoFilter));

    auto const sqrt = Emrys::toSqrt(initial);
    ASSERT_TRUE(sqrt);
    auto const sqrtFilter = Emrys::SqrtKalmanFilter{model, sqrt.value()};
    EXPECT_EQ(printed(sqrtFilter), "square-root Kalman filter\n" + nested(sqrtFilter.state()));

    auto const particleFilter = Emrys::ParticleFilter{model, ParticleSet<State>{State{}, Eigen::Matrix4d::Identity()}};
    EXPECT_EQ(printed(particleFilter), "particle filter\n" + nested(particleFilter.state()));
}

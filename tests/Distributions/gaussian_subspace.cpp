// GaussianMarginal and GaussianConditional (steps 1, 2; [view-stamp]): a marginal selects its block from the
// lower triangle, a copy is independent while a view is live, no marginal is made of a temporary, a
// conditional matches the Schur-complement formula and its parameters' density,
// conditionalWithEvidence() adds the given values' log-density ([conditional-evidence]) and refuses a
// block known exactly, which conditional() conditions on exactly ([innovation-cholesky]), and reading a
// marginal after a structural change of its parameters dies.

#include <EmrysFilter/Distributions/Gaussian.h>

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <cmath>
#include <expected>
#include <numbers>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;

namespace
{

struct Front : Linear<2>
{};

struct Back : Linear<2>
{};

struct Rest : Linear<Eigen::Dynamic>
{};

using State = Composite<Front, Back>;  // Front is coordinates 0 and 1, Back 2 and 3
using Open = Composite<Front, Rest>;   // the same, with a resizable second block

using MG4 = GaussianParams<State>;
using OpenGaussian = GaussianParams<Open>;

Eigen::Matrix4d
makeSpd()
{
    Eigen::Matrix4d const a = Eigen::Matrix4d::NullaryExpr([](Eigen::Index i, Eigen::Index j) {
        return std::sin(7.0 * double(i) + 3.0 * double(j) + 1.0);
    });
    return a * a.transpose() + Eigen::Matrix4d::Identity();
}

Eigen::Vector4d const mean4{1.0, -2.0, 3.0, 0.5};

// The covariance with its strict upper triangle overwritten, which only the lower triangle may be read past.
Eigen::Matrix4d
withGarbageUpper(Eigen::Matrix4d cov)
{
    cov.triangularView<Eigen::StrictlyUpper>().setConstant(12345.0);
    return cov;
}

}  // namespace

// A tag marginal has the block's mean and covariance taken from the stored lower triangle alone
// (the upper triangle holds garbage), with the block's compile-time dimension.
TEST(GaussianMarginal, SelectsBlockFromLowerTriangle)
{
    auto const mg = MG4{State{}, mean4, withGarbageUpper(makeSpd())};
    auto const marginal = mg.marginal<Back>();
    static_assert(decltype(marginal)::ValueDimensionAtCompileTime == 2);
    EXPECT_EQ(marginal.mean(), Eigen::Vector2d(3.0, 0.5));
    EXPECT_EQ(Eigen::Matrix2d{marginal.covarianceLower()}, (Eigen::Matrix2d{makeSpd().bottomRightCorner<2, 2>()}));
}

// A marginal of a fixed block is fixed-size and one of a resizable block is dynamic, with the
// parameters' current size.
TEST(GaussianMarginal, KeepsTheBlockSize)
{
    auto const mg = OpenGaussian{Open{}, mean4, makeSpd()};
    auto const fixed = mg.marginal<Front>();
    static_assert(decltype(fixed)::ValueDimensionAtCompileTime == 2);
    EXPECT_EQ(fixed.mean(), mean4.head<2>());

    auto const resizable = mg.marginal<Rest>();
    static_assert(decltype(resizable)::ValueDimensionAtCompileTime == Eigen::Dynamic);
    EXPECT_EQ(resizable.valueDimension(), 2);
    EXPECT_EQ(resizable.mean(), mean4.tail<2>());
}

// copy() is a Gaussian over the block that keeps the old values, while the view follows a later
// change of the parameters' mean, which is allowed under a marginal.
TEST(GaussianMarginal, CopyIsIndependentAndViewIsLive)
{
    auto mg = MG4{State{}, mean4, makeSpd()};
    auto const marginal = mg.marginal<Back>();
    auto copy = marginal.copy();
    static_assert(std::is_same_v<decltype(copy), GaussianParams<Back>>);

    mg.setMean(Eigen::Vector4d::Zero());  // values may change under a marginal
    EXPECT_EQ(marginal.mean(), Eigen::Vector2d::Zero());
    EXPECT_EQ(copy.mean(), Eigen::Vector2d(3.0, 0.5));
    EXPECT_TRUE(copy.toDensity().has_value());
}

// Copies and moves of a marginal read the same block, and marginals that are not read again leave
// the parameters free to change structure.
TEST(GaussianMarginal, MarginalsCopyAndMoveFreely)
{
    auto mg = OpenGaussian{Open{}, mean4, makeSpd()};
    {
        auto const marginal = mg.marginal<Front>();
        auto copy = marginal;
        auto const moved = std::move(copy);
        EXPECT_EQ(moved.mean(), marginal.mean());
    }
    mg.resize(3);
    EXPECT_EQ(mg.marginal<Rest>().valueDimension(), 1);
}

/// Whether \a G, with its value category, can make the marginal of the block Front.
template <typename G>
concept MakesMarginal = requires(G && g) { std::forward<G>(g).template marginal<Front>(); };

// A marginal of a temporary would dangle at the end of the statement, so marginal() on an rvalue
// does not compile, while a named Gaussian's does [view-stamp].
TEST(GaussianMarginal, NoMarginalOfATemporary)
{
    static_assert(MakesMarginal<MG4 &>);
    static_assert(MakesMarginal<MG4 const &>);
    static_assert(!MakesMarginal<MG4>);
    static_assert(!MakesMarginal<MG4 const>);
}

// A marginal taken from the factorized view reads the same values.
TEST(GaussianMarginal, FromView)
{
    auto const mg = MG4{State{}, mean4, makeSpd()};
    auto const marginal = mg.toDensity()->marginal<Back>();
    EXPECT_EQ(marginal.mean()(0), 3.0);
    EXPECT_EQ(marginal.copy().covarianceLower()(0, 0), makeSpd()(2, 2));
}

namespace
{

// Reference conditional through the explicit formula, with indices a (target) and b (given).
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
referenceConditional(std::vector<int> const & a, std::vector<int> const & b, Eigen::VectorXd const & v)
{
    Eigen::Matrix4d const p = makeSpd();
    Eigen::MatrixXd const pbb = p(b, b);
    Eigen::MatrixXd const gain = p(a, b) * pbb.inverse();
    return {mean4(a) + gain * (v - mean4(b)), p(a, a) - gain * p(b, a)};
}

}  // namespace

// conditional<Front>(v) matches the explicit Schur-complement formula computed with plain Eigen,
// reading only the lower triangle, and is a Gaussian over the remaining block.
TEST(GaussianConditional, MatchesFormula)
{
    auto const mg = MG4{State{}, mean4, withGarbageUpper(makeSpd())};
    auto const v = Eigen::Vector2d{0.3, -0.7};
    auto const conditional = mg.conditional<Front>(v);
    static_assert(std::is_same_v<decltype(conditional), GaussianParams<Back> const>);

    auto const [mean, cov] = referenceConditional({2, 3}, {0, 1}, v);
    EXPECT_LT((conditional.mean() - mean).norm(), 1e-12);
    EXPECT_LT((conditional.covarianceLower() - cov).norm(), 1e-12);
}

// Conditioning with a resizable remaining block gives the same result as the fixed one, over the
// resizable space.
TEST(GaussianConditional, ResizableRest)
{
    auto const v = Eigen::Vector2d{0.25, -1.0};
    auto const fixed = MG4{State{}, mean4, makeSpd()}.conditional<Front>(v);
    auto const resizable = OpenGaussian{Open{}, mean4, makeSpd()}.conditional<Front>(v);
    static_assert(std::is_same_v<decltype(resizable), GaussianParams<Rest> const>);
    EXPECT_EQ(resizable.valueDimension(), 2);
    EXPECT_LT((resizable.mean() - fixed.mean()).norm(), 1e-12);
    EXPECT_LT((resizable.covarianceLower() - fixed.covarianceLower()).norm(), 1e-12);
}

// Conditioning through the parameters and through its factorized view agree exactly.
TEST(GaussianConditional, DensityMatchesParams)
{
    auto const mg = MG4{State{}, mean4, makeSpd()};
    auto const v = Eigen::Vector2d{1.0, 2.0};
    auto const fromGaussian = mg.conditional<Back>(v);
    auto const fromDensity = mg.toDensity()->conditional<Back>(v);
    EXPECT_EQ(fromGaussian.mean(), fromDensity.mean());
    EXPECT_EQ(fromGaussian.covarianceLower(), fromDensity.covarianceLower());
    EXPECT_TRUE(fromDensity.toDensity().has_value());
}

// conditionalWithEvidence<Front>(v) returns conditional<Front>(v), to rounding (GCC on arm64 fuses
// their multiply-adds differently), beside the log-density of v under the Front marginal, computed
// with plain Eigen from the closed form -(2 log 2 pi + log|P_bb| + r' P_bb^-1 r) / 2; the
// parameters and their view agree exactly.
TEST(GaussianConditional, WithEvidence)
{
    auto const mg = MG4{State{}, mean4, withGarbageUpper(makeSpd())};
    auto const v = Eigen::Vector2d{0.3, -0.7};
    auto const [conditional, logEvidence] = mg.conditionalWithEvidence<Front>(v).value();
    static_assert(std::is_same_v<
                  decltype(mg.conditionalWithEvidence<Front>(v)),
                  std::expected<Emrys::GaussianConditional<Back>, Emrys::GaussianError>>);
    auto const plain = mg.conditional<Front>(v);
    EXPECT_TRUE(conditional.mean().isApprox(plain.mean(), 1e-14));
    EXPECT_TRUE(conditional.covarianceLower().isApprox(plain.covarianceLower(), 1e-14));

    Eigen::Matrix2d const pbb = makeSpd().topLeftCorner<2, 2>();
    Eigen::Vector2d const r = v - mean4.head<2>();
    double const expected =
        -0.5 * (2.0 * std::log(2.0 * std::numbers::pi) + std::log(pbb.determinant()) + r.dot(pbb.inverse() * r));
    EXPECT_NEAR(logEvidence, expected, 1e-12);

    auto const fromDensity = mg.toDensity()->conditionalWithEvidence<Front>(v).value();
    EXPECT_EQ(fromDensity.logEvidence, logEvidence);
    EXPECT_EQ(fromDensity.conditional.mean(), conditional.mean());
}

// A given block known exactly has no density: conditionalWithEvidence() refuses it, while
// conditional() skips the zero pivot, whose direction the gain has no part in, and gives the exact
// conditional: here the Back block is independent of the known coordinate, so only the other
// Front coordinate moves it [innovation-cholesky].
TEST(GaussianConditional, BlockKnownExactly)
{
    Eigen::Matrix4d cov = makeSpd();
    cov.row(0).setZero();
    cov.col(0).setZero();
    auto const mg = MG4{State{}, mean4, cov};
    auto const v = Eigen::Vector2d{mean4(0), -0.7};

    auto const withEvidence = mg.conditionalWithEvidence<Front>(v);
    ASSERT_FALSE(withEvidence);
    EXPECT_EQ(withEvidence.error(), Emrys::GaussianError::NotPositiveDefinite);

    auto const conditional = mg.conditional<Front>(v);
    double const p11 = cov(1, 1);
    Eigen::Vector2d const cross = cov.block<2, 1>(2, 1);
    Eigen::Vector2d const mean = mean4.tail<2>() + cross / p11 * (v(1) - mean4(1));
    Eigen::Matrix2d const reduced = cov.bottomRightCorner<2, 2>() - cross * cross.transpose() / p11;
    EXPECT_LT((conditional.mean() - mean).norm(), 1e-12);
    EXPECT_LT((conditional.covarianceLower() - reduced).norm(), 1e-12);
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)

// Reading a marginal after its parameters were resized, moved from, or assigned to dies [view-stamp].
TEST(GaussianMarginalDeathTest, ReadAfterStructuralChange)
{
    auto mg = OpenGaussian{Open{}, mean4, makeSpd()};
    auto const marginal = mg.marginal<Front>();
    EXPECT_DEATH(
        {
            mg.resize(3);
            std::ignore = marginal.mean();
        },
        "changed size or structure while a marginal"
    );
    EXPECT_DEATH(
        {
            auto stolen = std::move(mg);
            std::ignore = marginal.mean();
        },
        "changed size or structure while a marginal"
    );
    EXPECT_DEATH(
        {
            mg = OpenGaussian{3};
            std::ignore = marginal.covarianceLower();
        },
        "changed size or structure while a marginal"
    );
}

// A marginal taken from the factorized view holds a copy of the view after the view is gone, so
// reading it after the mean changes dies as the view would.
TEST(GaussianMarginalDeathTest, MarginalOfViewChecks)
{
    auto mg = MG4{State{}, mean4, makeSpd()};
    auto marginal = std::optional<decltype(mg.toDensity()->marginal<Front>())>{};
    marginal.emplace(mg.toDensity()->marginal<Front>());
    mg.setMean(Eigen::Vector4d::Zero());
    EXPECT_DEATH(std::ignore = marginal->mean(), "changed while a view");
}

#endif

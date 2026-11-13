// The build itself: the umbrella header EmrysFilter.h compiles as C++23 against Eigen 5, and Eigen
// does arithmetic. Evidence for the baseline; everything else assumes it.

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

// The compiler reports C++23 or later.
TEST(Smoke, CompilesAsCxx23)
{
    EXPECT_GE(__cplusplus, 202302L);
}

// The Eigen found is release 5 or later.
TEST(Smoke, EigenIsVersion5)
{
    // Eigen 5 keeps EIGEN_WORLD_VERSION at 3; the release number is the major.
    EXPECT_GE(EIGEN_MAJOR_VERSION, 5);
}

// A matrix-vector product evaluates: Eigen is linked and sane.
TEST(Smoke, EigenWorks)
{
    Eigen::Matrix2d const a = Eigen::Matrix2d::Identity();
    auto const x = Eigen::Vector2d{1.0, 2.0};
    EXPECT_TRUE((a * x).isApprox(x));
}

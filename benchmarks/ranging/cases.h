#pragma once

// The cases the ranging tests run: a dimension, an anchor count, a linearization, and a scalar, as
// GoogleTest types, shared by the agreement and allocation tests.

#include "scenario.h"

#include <gtest/gtest.h>

namespace Ranging
{

/// A case of the benchmark: a dimension, an anchor count, a linearization, and a scalar.
template <int Dimension_, int AnchorCount_, Jacobian Kind_, typename Scalar_ = double>
struct Case
{
    static constexpr int Dimension = Dimension_;
    static constexpr int AnchorCount = AnchorCount_;
    static constexpr Jacobian Kind{Kind_};
    using Scalar = Scalar_;
};

/// The planar and spatial cases, under both linearizations, in double and in float, the scalar of
/// an embedded target [embedded-builds].
using Cases = ::testing::Types<
    Case<2, 4, Jacobian::Analytic>,
    Case<2, 4, Jacobian::AutoDiff>,
    Case<3, 6, Jacobian::Analytic>,
    Case<3, 6, Jacobian::AutoDiff>,
    Case<2, 4, Jacobian::Analytic, float>,
    Case<2, 4, Jacobian::AutoDiff, float>,
    Case<3, 6, Jacobian::Analytic, float>,
    Case<3, 6, Jacobian::AutoDiff, float>>;

}  // namespace Ranging

#pragma once

// The cases the attitude tests run: a linearization of the reference vectors, as GoogleTest types,
// shared by the agreement and allocation tests.

#include "scenario.h"

#include <gtest/gtest.h>

namespace Orientation
{

/// A case of the benchmark: a linearization.
template <Jacobian Kind_>
struct Case
{
    static constexpr Jacobian Kind{Kind_};
};

/// Both linearizations.
using Cases = ::testing::Types<Case<Jacobian::Analytic>, Case<Jacobian::AutoDiff>>;

}  // namespace Orientation

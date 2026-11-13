#pragma once

// The cases the particle tests run: a dimension, an anchor count, a particle count, and a
// likelihood, as GoogleTest types, shared by the agreement and allocation tests.

#include "scenario.h"

#include <gtest/gtest.h>

namespace Particles
{

/// A case of the benchmark: a dimension, an anchor count, a particle count, and a likelihood.
template <int Dimension_, int AnchorCount_, int ParticleCount_, Likelihood Kind_>
struct Case
{
    static constexpr int Dimension = Dimension_;
    static constexpr int AnchorCount = AnchorCount_;
    static constexpr int ParticleCount = ParticleCount_;
    static constexpr Likelihood Kind{Kind_};
};

/// The planar case with a thousand particles and the spatial one with four thousand, under each
/// likelihood. A thousand particles are too few for a six-dimensional state: the filter's NEES
/// is then about twice the degrees of freedom, where four thousand bring it within a quarter.
using Cases = ::testing::Types<
    Case<2, 4, 1000, Likelihood::Noise>,
    Case<2, 4, 1000, Likelihood::Own>,
    Case<2, 4, 1000, Likelihood::Batched>,
    Case<3, 6, 4000, Likelihood::Noise>,
    Case<3, 6, 4000, Likelihood::Own>,
    Case<3, 6, 4000, Likelihood::Batched>>;

}  // namespace Particles

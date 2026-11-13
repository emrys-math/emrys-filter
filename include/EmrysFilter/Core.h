#pragma once

/** \file
 * Everything in the Core layer, which depends on Eigen alone: the version, the configuration
 * macros, the mathematical constants and scalar concept, how matrix arguments are forwarded, the
 * allocator conventions, execution policies, log-weight arithmetic, resamplers, the chi-squared
 * distribution, triangularization, products of fixed-size matrices, solving by a small factorization, the modification
 * stamp that views check, and printing.
 */

/** \defgroup Core Core
 *
 * The first layer (decisions.md [layered-headers]): what the rest of the library builds on and what depends
 * on nothing but Eigen. It holds the version (Version.h), the macros (Macros.h), the RealScalar
 * concept and constants and scalar functions in any such scalar (Math.h), the concept that lets
 * a function move a matrix argument it stores (Forwarding.h, [forwarded-matrices]), the allocator conventions and
 * buffers (Allocation.h, [allocation]), the execution policies and the primitives that run loops over a
 * count under them (Execution.h, [execution-policies]), the time steps a prediction takes and their conversion to
 * a scalar (Time.h, [time-steps]), log-weights and their normalization (Weights.h), the resamplers
 * (Resampling.h), the chi-squared distribution's CDF and quantile (ChiSquared.h), the
 * triangularization that the square-root forms of the %Gaussian are built on
 * (Triangularization.h, [gaussian-forms]), the products that keep a fixed size on Eigen's fixed-size path
 * (Multiply.h), the solve that gives a gain from a factorization of a small matrix
 * one column at a time (Solve.h), the modification stamp that parameters keep and the checks by which
 * their views read it (Stamp.h, [view-stamp]), and the \c operator<< and \c std::formatter that print every
 * distribution and filter (Print.h), left out under \c EMRYS_FILTER_NO_IO. Read Version.h first, and \ref Spaces next.
 */

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/ChiSquared.h>
#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Resampling.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Time.h>
#include <EmrysFilter/Core/Triangularization.h>
#include <EmrysFilter/Core/Version.h>
#include <EmrysFilter/Core/Weights.h>

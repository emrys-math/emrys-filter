#pragma once

/** \file
 * The distribution over a discrete space (decisions.md [discrete-spaces], [params-and-density]): CategoricalParams, one
 * weight per value, written freely and updated by Bayes' rule, with marginals and conditionals by axis; and
 * Categorical, the view its toDensity() returns once the weights define a distribution, with
 * probabilities by value and by named event. MarkovChain moves the parameters a step, built
 * directly, from independent chains on the axes of a product, as persistent or memoryless, or
 * from mean dwell times and exit weights; predict() moves them, stationary() gives the chain's
 * stationary distribution as a prior, and drawSamples() draws from the density. Read
 * ContinuousTimeMarkovChain.h next.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <expected>
#include <random>
#include <utility>
#include <vector>

#include <EmrysFilter/Core/Forwarding.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Print.h>
#include <EmrysFilter/Core/Stamp.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Params.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

namespace Emrys
{

template <DiscreteSpace Space_, RealScalar Scalar_>
class CategoricalParams;

template <DiscreteSpace Space_, RealScalar Scalar_>
class Categorical;

/** \ingroup Distributions
 * Reasons a Markov chain has no single stationary distribution.
 *
 * \sa stationary()
 */
enum class ChainError
{
    Reducible,  ///< some value can't be reached from some other, so the chain is not irreducible
};

namespace internal
{

/** \returns the generator of the chain that stays at value \f$ i \f$ for a mean time of
 * \f$ \tau_i \f$, \a dwellTimes, and leaves it for each other value \f$ j \f$ in proportion to
 * \f$ w_{ij} \f$, \a exitWeights: \f$ G_{ii} = -1/\tau_i \f$ and
 * \f$ G_{ij} = w_{ij} / (\tau_i \sum_{k} w_{ik}) \f$. An infinite \f$ \tau_i \f$ never leaves,
 * and its row of weights may be zero.
 */
template <typename Scalar, typename Matrix, typename DwellTimes, typename ExitWeights>
[[nodiscard]] Matrix
dwellGenerator(Eigen::MatrixBase<DwellTimes> const & dwellTimes, Eigen::MatrixBase<ExitWeights> const & exitWeights)
{
    Eigen::Index const n = exitWeights.rows();
    EMRYS_FILTER_ASSERT(
        dwellTimes.size() == n && exitWeights.cols() == n && "give a dwell time and a row of exit weights per value"
    );
    auto generator = Matrix{};
    generator.resize(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        Scalar const dwell = Scalar(dwellTimes(i));
        EMRYS_FILTER_ASSERT(dwell > Scalar(0) && "a dwell time must be positive");
        EMRYS_FILTER_ASSERT(exitWeights(i, i) == 0 && "a value's weight of exiting to itself must be zero");
        EMRYS_FILTER_ASSERT((exitWeights.row(i).array() >= 0).all() && "exit weights must be nonnegative");
        Scalar const leave = Scalar(1) / dwell;
        Scalar const total = Scalar(exitWeights.row(i).sum());
        EMRYS_FILTER_ASSERT(
            (leave == Scalar(0) || total > Scalar(0)) && "a value that is left needs a value to exit to"
        );
        if (leave == Scalar(0))
        {
            generator.row(i).setZero();
        }
        else
        {
            generator.row(i) = exitWeights.row(i).template cast<Scalar>() * Scalar(leave / total);
        }
        generator(i, i) = -leave;
    }
    return generator;
}

/** \returns the stationary probabilities of the irreducible chain whose off-diagonal entries of
 * \a kernel, a transition matrix or a generator, are its probabilities or rates of moving between
 * values, or ChainError::Reducible if the chain is not irreducible.
 *
 * The algorithm is that of \ref Grassmann1985 "Grassmann, Taksar, and Heyman (1985)": it removes
 * the values from the last to the second, each time replacing the chain with the chain watched
 * only at the values left, then builds the probabilities back up from the first. It reads only
 * the off-diagonal entries, so a transition matrix and a generator give the same answer, and it
 * only adds and multiplies nonnegative numbers, so it loses no precision to cancellation when a
 * chain stays a long time at each value, and an entry is zero exactly when it is structurally
 * zero. So a chain is reducible exactly when a value about to be removed can't reach an earlier
 * one, or a value's probability comes out zero.
 */
template <typename Scalar, typename Array, typename Matrix>
[[nodiscard]] std::expected<Array, ChainError>
stationaryProbabilities(Matrix kernel)
{
    Eigen::Index const n = kernel.rows();
    EMRYS_FILTER_ASSERT(n > 0 && "a chain has at least one value");
    for (Eigen::Index k = n - 1; k > 0; --k)
    {
        // Watched only at the values before k, the chain moves from i to j directly or through
        // k, which it leaves for an earlier value with the total of row k before it.
        Scalar const leave = kernel.row(k).head(k).sum();
        if (!(leave > Scalar(0)))
        {
            return std::unexpected{ChainError::Reducible};
        }
        kernel.col(k).head(k) /= leave;
        kernel.topLeftCorner(k, k).noalias() += kernel.col(k).head(k) * kernel.row(k).head(k);
    }
    Array probabilities = Array::Zero(n);
    probabilities(0) = Scalar(1);
    for (Eigen::Index k = 1; k < n; ++k)
    {
        Scalar const flow = probabilities.head(k).matrix().dot(kernel.col(k).head(k));
        if (!(flow > Scalar(0)))
        {
            return std::unexpected{ChainError::Reducible};
        }
        probabilities(k) = flow;
    }
    probabilities /= probabilities.sum();
    return probabilities;
}

/** \returns the distribution of the axis named by the path \a P (a tag or a TagPath) of
 * \a space, from \a logValues, one per value of \a space: for each value of the axis, the log of
 * the sum of the exponentials of the values that have it. The log-weights of a CategoricalParams
 * give log-weights with the same total, and log-probabilities give log-probabilities.
 */
template <typename P, typename Scalar, DiscreteSpace Space, typename Values>
[[nodiscard]] auto
categoricalMarginal(Space const & space, Values const & logValues)
{
    auto const [stride, axis] = locateAxisByPath<P>(space);
    using Axis = std::remove_cvref_t<decltype(*axis)>;
    using AxisArray = typename CategoricalParams<Axis, Scalar>::ArrayType;
    Eigen::Index const count = axis->cardinality();
    // A LogSumExp per value of the axis, kept as its running maximum and sum in the axis's own
    // storage, so a bounded axis allocates nothing.
    AxisArray maxima = AxisArray::Constant(count, -Eigen::NumTraits<Scalar>::infinity());
    AxisArray sums = AxisArray::Zero(count);
    for (Eigen::Index v = 0; v < logValues.size(); ++v)
    {
        Eigen::Index const i = (v / stride) % count;
        Scalar const x = logValues(v);
        if (x == -Eigen::NumTraits<Scalar>::infinity())
        {
            continue;
        }
        if (x <= maxima(i))
        {
            sums(i) += Eigen::numext::exp(x - maxima(i));
        }
        else
        {
            sums(i) = sums(i) * Eigen::numext::exp(maxima(i) - x) + Scalar(1);
            maxima(i) = x;
        }
    }
    for (Eigen::Index i = 0; i < count; ++i)
    {
        maxima(i) =
            sums(i) == Scalar(0) ? -Eigen::NumTraits<Scalar>::infinity() : maxima(i) + Eigen::numext::log(sums(i));
    }
    return CategoricalParams<Axis, Scalar>{*axis, maxima};
}

/** \returns the parameters over \a rest, the axes other than the one at \a location, whose
 * log-weights are the entries of \a logValues, one per value of the whole space, of the values
 * whose axis has \a location's value
 */
template <typename Scalar, DiscreteSpace Rest, typename Values>
[[nodiscard]] CategoricalParams<Rest, Scalar>
categoricalConditional(Rest rest, ValueLocation const & location, Values const & logValues)
{
    using RestArray = typename CategoricalParams<Rest, Scalar>::ArrayType;
    auto selected = RestArray{};
    selected.resize(rest.cardinality());
    Eigen::Index next = 0;
    for (Eigen::Index v = 0; v < logValues.size(); ++v)
    {
        if (location.matches(v))
        {
            selected(next++) = logValues(v);
        }
    }
    EMRYS_FILTER_ASSERT(next == selected.size());
    return CategoricalParams<Rest, Scalar>{std::move(rest), selected};
}

/** \returns the log-weights \a logWeights moved by \a kernel, \f$ \log (K^T e^{\ell}) \f$,
 * computed with the largest log-weight factored out, so that weights far below one neither
 * underflow nor lose their ratios. A NaN or \f$ +\infty \f$ log-weight stays in the result, for
 * the check that reads it.
 */
template <typename Scalar, typename Array, typename Kernel>
[[nodiscard]] Array
movedLogWeights(Array const & logWeights, Kernel const & kernel)
{
    // Every discrete space has a value, so there is a largest log-weight.
    Scalar const largest = logWeights.maxCoeff();
    Scalar const shift = isFinite(largest) ? largest : Scalar(0);
    Array const weights = (logWeights - shift).exp();
    Array moved = multiply(kernel.transpose(), weights.matrix()).array();
    return moved.log() + shift;
}

}  // namespace internal

/** \ingroup Distributions
 * \brief The parameters of a probability distribution over a discrete space: one weight per
 * value.
 *
 * The [categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution), kept as
 * one log-weight per value, so that a value can be improbable without underflowing. The weights
 * need not sum to one, and they are not checked when they are written: a histogram can be filled,
 * a weight set by name, or several sensors' log-likelihoods added with reweight(), one write at a
 * time, since only the whole set has to sum to something. toDensity() checks them once and
 * returns the Categorical that evaluates the probabilities (decisions.md [params-and-density]). A value is its flat
 * index (see \ref Discrete), and a value named by tags is <tt>valueOf<Mode, Turn, Outlier>()</tt>.
 *
 * \code
 * auto p = CategoricalParams<Mode>::fromWeights(6.0, 1.0, 2.5, 0.5);
 * p.setWeight<Turn, Outlier>(0.0);          // rule a value out
 * auto const evidence = p.update(logLikelihoods);   // Bayes' rule, or a WeightError
 * auto const density = p.toDensity();      // the probabilities, or a WeightError
 * density->probability<Turn>();            // an event
 * p = predict(p, chain);                   // one step of a MarkovChain
 * \endcode
 *
 * The array holds one weight per value: inline up to the space's \c MaxCardinality, as a mean
 * stores its coordinates, and on the heap only for an unbounded space.
 *
 * \tparam Space_   the discrete space (see \ref Discrete)
 * \tparam Scalar_  scalar type of the weights, \c double by default
 *
 * \sa Categorical, MarkovChain, predict()
 */
template <DiscreteSpace Space_, RealScalar Scalar_ = double>
class CategoricalParams
{
public:

    using Scalar = Scalar_;                                               ///< the scalar type of the weights
    using SpaceType = Space_;                                             ///< the discrete space
    using DensityType = Categorical<Space_, Scalar_>;                     ///< the view toDensity() returns
    static constexpr int CardinalityAtCompileTime = Space_::Cardinality;  ///< the number of values, or Eigen::Dynamic
    static constexpr int MaxCardinalityAtCompileTime = Space_::MaxCardinality;  ///< a bound on it, or Eigen::Dynamic
    static constexpr bool IsParams = true;                                      ///< parameters, which views check

    /** One entry per value, in flat-index order. */
    using ArrayType =
        Eigen::Array<Scalar_, CardinalityAtCompileTime, 1, Eigen::ColMajor, MaxCardinalityAtCompileTime, 1>;

    /** The uniform distribution over \a space, with log-weights that are its log-probabilities. */
    explicit CategoricalParams(Space_ space = {})
        : m_space{std::move(space)}
        , m_logWeights{ArrayType::Constant(m_space.cardinality(), -Eigen::numext::log(Scalar(m_space.cardinality())))}
    {}

    /** The distribution over \a space whose probabilities are proportional to \f$ e^{\ell_v} \f$,
     * with \a logWeights \f$ \ell \f$, one per value.
     */
    template <typename Derived>
    CategoricalParams(Space_ space, Eigen::DenseBase<Derived> const & logWeights)
        : m_space{std::move(space)}
        , m_logWeights{logWeights.derived().array().template cast<Scalar>()}
    {
        EMRYS_FILTER_ASSERT(m_logWeights.size() == m_space.cardinality() && "give one log-weight per value");
    }

    /** \returns the distribution over \a space proportional to \a weights, one per value, which
     * need not sum to one. A negative weight has no logarithm, so toDensity() reports it as
     * WeightError::NonFinite.
     */
    template <typename Derived>
    [[nodiscard]] static CategoricalParams
    fromWeights(Space_ space, Eigen::DenseBase<Derived> const & weights)
    {
        return CategoricalParams{std::move(space), weights.derived().array().template cast<Scalar>().log()};
    }

    /** \returns the distribution over the fixed-size space proportional to \a weights:
     * fromWeights(space, weights) in <tt>Space_{}</tt>.
     */
    template <typename Derived>
        requires FixedSizeSpace<Space_>
    [[nodiscard]] static CategoricalParams
    fromWeights(Eigen::DenseBase<Derived> const & weights)
    {
        return fromWeights(Space_{}, weights);
    }

    /** \returns the distribution over the fixed-size space proportional to \a weights, one
     * argument per value in flat-index order, written without an Eigen vector:
     * <tt>CategoricalParams<Maneuver>::fromWeights(0.9, 0.1)</tt>. The space must have a fixed
     * number of values, and the count is checked at compile time.
     */
    template <typename... Weights>
        requires FixedSizeSpace<Space_> && (CardinalityAtCompileTime != Eigen::Dynamic) && (sizeof...(Weights) > 0)
              && (std::convertible_to<Weights, Scalar_> && ...)
    [[nodiscard]] static CategoricalParams
    fromWeights(Weights... weights)
    {
        static_assert(sizeof...(Weights) == CardinalityAtCompileTime, "give one weight per value");
        auto array = ArrayType{};
        Eigen::Index i = 0;
        ((array(i++) = Scalar_(weights)), ...);
        return fromWeights(Space_{}, array);
    }

    /** \returns the discrete space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the number of values */
    [[nodiscard]] Eigen::Index
    cardinality() const
    {
        return m_logWeights.size();
    }

    /** \returns the log-weight of each value */
    [[nodiscard]] ArrayType const &
    logWeights() const
    {
        return m_logWeights;
    }

    /** \returns the log-weight of the value with flat index \a value */
    [[nodiscard]] Scalar
    logWeight(Eigen::Index value) const
    {
        EMRYS_FILTER_ASSERT(value >= 0 && value < cardinality() && "the value is out of range");
        return m_logWeights(value);
    }

    /** \returns the log of the sum of the weights, or the WeightError that makes them fail to
     * define a distribution: they are all zero, or one is NaN or \f$ +\infty \f$
     */
    [[nodiscard]] std::expected<Scalar, WeightError>
    logTotalWeight() const
    {
        return internal::logTotalWeight<Scalar>(m_logWeights);
    }

    /** Sets the log-weights to \a logWeights, one per value. */
    template <typename Derived>
    void
    setLogWeights(Eigen::DenseBase<Derived> const & logWeights)
    {
        EMRYS_FILTER_ASSERT(logWeights.size() == cardinality() && "give one log-weight per value");
        m_stamp.touchValues();
        m_logWeights = logWeights.derived().array().template cast<Scalar>();
    }

    /** Sets the weights to \a weights, one per value, which need not sum to one. */
    template <typename Derived>
    void
    setWeights(Eigen::DenseBase<Derived> const & weights)
    {
        setLogWeights(weights.derived().array().template cast<Scalar>().log());
    }

    /** Sets the log-weight of the value with flat index \a value to \a logWeight. */
    void
    setLogWeight(Eigen::Index value, Scalar logWeight)
    {
        EMRYS_FILTER_ASSERT(value >= 0 && value < cardinality() && "the value is out of range");
        m_stamp.touchValues();
        m_logWeights(value) = logWeight;
    }

    /** Sets the weight of the value with flat index \a value to \a weight. */
    void
    setWeight(Eigen::Index value, Scalar weight)
    {
        setLogWeight(value, Scalar(Eigen::numext::log(weight)));
    }

    /** Sets the log-weight of the value with the named values \a Values, one per axis (see
     * valueOf()), to \a logWeight.
     */
    template <typename... Values>
        requires(sizeof...(Values) > 0)
    void
    setLogWeight(Scalar logWeight)
    {
        setLogWeight(valueOf<Values...>(m_space), logWeight);
    }

    /** Sets the weight of the value with the named values \a Values, one per axis (see
     * valueOf()), to \a weight.
     */
    template <typename... Values>
        requires(sizeof...(Values) > 0)
    void
    setWeight(Scalar weight)
    {
        setWeight(valueOf<Values...>(m_space), weight);
    }

    /** Multiplies each value's weight by its likelihood \f$ e^{\ell_v} \f$, with
     * \a logLikelihoods \f$ \ell \f$ one per value, and checks nothing: several measurements can
     * be added before one toDensity() or normalize().
     */
    template <typename Derived>
    void
    reweight(Eigen::DenseBase<Derived> const & logLikelihoods)
    {
        EMRYS_FILTER_ASSERT(logLikelihoods.size() == cardinality() && "give one log-likelihood per value");
        m_stamp.touchValues();
        m_logWeights += logLikelihoods.derived().array().template cast<Scalar>();
    }

    /** [Bayes' rule](https://en.wikipedia.org/wiki/Bayes%27_theorem): multiplies each value's
     * probability by its likelihood \f$ e^{\ell_v} \f$, with \a logLikelihoods \f$ \ell \f$ one
     * per value, and normalizes.
     *
     * \returns the log of the evidence \f$ \sum_v p_v e^{\ell_v} \f$, or a WeightError if the
     * weights before or after fail to define a distribution, in which case they are unchanged
     */
    template <typename Derived>
    [[nodiscard]] std::expected<Scalar, WeightError>
    update(Eigen::DenseBase<Derived> const & logLikelihoods)
    {
        EMRYS_FILTER_ASSERT(logLikelihoods.size() == cardinality() && "give one log-likelihood per value");
        auto const prior = logTotalWeight();
        if (!prior)
        {
            return std::unexpected{prior.error()};
        }
        ArrayType posterior = m_logWeights + logLikelihoods.derived().array().template cast<Scalar>();
        auto const total = internal::logTotalWeight<Scalar>(posterior);
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        m_stamp.touchValues();
        m_logWeights = posterior - *total;
        return *total - *prior;
    }

    /** Shifts the log-weights so that the weights sum to one, or returns the WeightError that
     * stops it, leaving them unchanged.
     */
    [[nodiscard]] std::expected<void, WeightError>
    normalize()
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        m_stamp.touchValues();
        m_logWeights -= *total;
        return {};
    }

    /** \returns the parameters of the axis named by the path \a P (a tag or a TagPath), summed
     * over the others, with the same total weight
     */
    template <typename P>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::categoricalMarginal<P, Scalar>(m_space, m_logWeights);
    }

    /** \returns the parameters of the other axes given that the axis named by the path \a P (a
     * tag or a TagPath) has the value with index \a value in it: the weights of the values that
     * have it. If they are all zero, toDensity() on the result says so.
     */
    template <typename P>
    [[nodiscard]] auto
    conditional(Eigen::Index value) const
    {
        auto const [stride, axis] = internal::locateAxisByPath<P>(m_space);
        EMRYS_FILTER_ASSERT(value >= 0 && value < axis->cardinality() && "the value is out of range");
        return internal::categoricalConditional<Scalar>(
            internal::complementAxes<P>(m_space),
            internal::ValueLocation{stride, axis->cardinality(), value},
            m_logWeights
        );
    }

    /** \returns the parameters of the other axes given the named value \a Value (a value's tag or
     * a TagPath ending in one): <tt>conditional<Turn>()</tt> is
     * conditional<Maneuver>(Maneuver::index<Turn>())
     */
    template <typename Value>
    [[nodiscard]] auto
    conditional() const
    {
        return internal::categoricalConditional<Scalar>(
            internal::complementAxes<internal::ValueAxisPath<Value, Space_>>(m_space),
            internal::locateValuePath<Value>(m_space),
            m_logWeights
        );
    }

    /** \returns a Categorical that checks this object, or the WeightError that makes the weights
     * fail to define a distribution (see logTotalWeight())
     */
    [[nodiscard]] std::expected<DensityType, WeightError>
    toDensity() const &
    {
        auto const total = logTotalWeight();
        if (!total)
        {
            return std::unexpected{total.error()};
        }
        return DensityType{*this, *total};
    }

    /** A density of a temporary would outlive it. */
    void
    toDensity() const && = delete;

private:

    friend DensityType;  ///< the view reads the log-weights

    /** A density checks the stamp. */
    template <typename>
    friend class internal::ValuesCheck;

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the discrete space
    ArrayType m_logWeights;                         ///< the log-weight of each value, not necessarily normalized
    internal::Stamp m_stamp;                        ///< what views check [view-stamp]
};

/// \cond DEDUCTION_GUIDES
template <DiscreteSpace Space, typename Derived>
CategoricalParams(Space, Eigen::DenseBase<Derived> const &) -> CategoricalParams<Space, typename Derived::Scalar>;

/// \endcond

namespace internal
{

/** \returns the log-probabilities of \a params, a lazy Eigen expression that must not outlive
 * them: the log-weights less the log of their total, or NaN if they fail to define a distribution.
 * For the library's own steps, which read parameters they produced normalized, where a
 * toDensity() would only add a check of something that cannot fail.
 */
template <typename Space, typename Scalar>
[[nodiscard]] auto
logProbabilitiesOf(CategoricalParams<Space, Scalar> const & params)
{
    return params.logWeights() - params.logTotalWeight().value_or(Eigen::NumTraits<Scalar>::quiet_NaN());
}

}  // namespace internal

/** \ingroup Distributions
 * \brief The probabilities of a CategoricalParams whose weights define a distribution: a
 * read-only view.
 *
 * Obtained from CategoricalParams::toDensity(), which checks the weights once and gives this view
 * the log of their total, so a value's log-probability is its log-weight minus that total and
 * nothing is copied. It checks its parameters for changes, as its copies do: reading it after
 * they have changed is a programmer error (decisions.md [view-stamp]). A Categorical must not outlive its
 * parameters.
 *
 * A set of values is named by tag: <tt>probability<Turn>()</tt> is the probability of every value
 * whose \c Maneuver is \c Turn, and <tt>probability<Turn, Outlier>()</tt> of the one value with
 * both.
 *
 * \code
 * auto const params = CategoricalParams<Mode>::fromWeights(0.6, 0.1, 0.25, 0.05);
 * auto const p = params.toDensity().value();
 * p.probability<Turn>();                 // 0.3
 * p.marginal<Sensor>();                  // CategoricalParams<Sensor>: 0.85, 0.15
 * p.conditional<Turn>();                 // CategoricalParams<Sensor> given Turn: 0.833, 0.167
 * \endcode
 *
 * \sa class CategoricalParams
 */
template <DiscreteSpace Space_, RealScalar Scalar_ = double>
class Categorical
{
public:

    using ParamsType = CategoricalParams<Space_, Scalar_>;  ///< the parameters it reads
    using Scalar = Scalar_;                                 ///< the scalar type of the probabilities
    using SpaceType = Space_;                               ///< the discrete space
    using ArrayType = typename ParamsType::ArrayType;       ///< one entry per value, in flat-index order
    static constexpr int CardinalityAtCompileTime =
        ParamsType::CardinalityAtCompileTime;  ///< the number of values, or Eigen::Dynamic
    static constexpr int MaxCardinalityAtCompileTime =
        ParamsType::MaxCardinalityAtCompileTime;  ///< a bound on it, or Eigen::Dynamic
    static constexpr bool IsParams = false;       ///< a view: it checks its parameters

    /** \returns the discrete space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return source().space();
    }

    /** \returns the number of values */
    [[nodiscard]] Eigen::Index
    cardinality() const
    {
        return source().cardinality();
    }

    /** \returns the log-probability of each value, a lazy Eigen expression that must not outlive
     * the parameters
     */
    [[nodiscard]] auto
    logProbabilities() const
    {
        return source().m_logWeights - m_logTotal;
    }

    /** \returns the probability of each value, a lazy Eigen expression that must not outlive the
     * parameters
     */
    [[nodiscard]] auto
    probabilities() const
    {
        return logProbabilities().exp();
    }

    /** \returns the log-probability of the value with flat index \a value */
    [[nodiscard]] Scalar
    logProbability(Eigen::Index value) const
    {
        return source().logWeight(value) - m_logTotal;
    }

    /** \returns the probability of the value with flat index \a value */
    [[nodiscard]] Scalar
    probability(Eigen::Index value) const
    {
        return Eigen::numext::exp(logProbability(value));
    }

    /** \returns the log-probability of the values that have every named value in \a Values, each a
     * value's tag or a TagPath ending in one: one value if every axis is named, otherwise a
     * marginal
     */
    template <typename... Values>
        requires(sizeof...(Values) > 0)
    [[nodiscard]] Scalar
    logProbability() const
    {
        ParamsType const & params = source();
        auto const locations = std::array<internal::ValueLocation, sizeof...(Values)>{
            internal::locateValuePath<Values>(params.m_space)...
        };
        auto total = LogSumExp<Scalar>{};
        for (Eigen::Index v = 0; v < params.cardinality(); ++v)
        {
            bool matches = true;
            for (auto const & location : locations)
            {
                matches = matches && location.matches(v);
            }
            if (matches)
            {
                total.add(params.m_logWeights(v));
            }
        }
        return total.value() - m_logTotal;
    }

    /** \returns the probability of the values that have every named value in \a Values, as logProbability<Values...>()
     */
    template <typename... Values>
        requires(sizeof...(Values) > 0)
    [[nodiscard]] Scalar
    probability() const
    {
        return Eigen::numext::exp(logProbability<Values...>());
    }

    /** \returns the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) in nats,
     * \f$ H(p) = -\sum_v p_v \log p_v \f$, with a value of probability zero contributing nothing
     *
     * It is 0 for a single certain value and \f$ \log n \f$ for the uniform distribution on
     * \f$ n \f$ values, the most there is. Its exponential, the perplexity, is the number of
     * equally likely values that would be as uncertain.
     *
     * \sa mostProbable(), kullbackLeibler()
     */
    [[nodiscard]] Scalar
    entropy() const
    {
        ArrayType const & logWeights = source().m_logWeights;
        Scalar result = 0;
        for (Eigen::Index v = 0; v < logWeights.size(); ++v)
        {
            if (logWeights(v) != -Eigen::NumTraits<Scalar>::infinity())
            {
                Scalar const logProbability = logWeights(v) - m_logTotal;
                result -= Eigen::numext::exp(logProbability) * logProbability;
            }
        }
        return Eigen::numext::maxi(result, Scalar(0));
    }

    /** \returns the flat index of the most probable value; ties go to the lowest */
    [[nodiscard]] Eigen::Index
    mostProbable() const
    {
        ArrayType const & logWeights = source().m_logWeights;
        Eigen::Index best = 0;
        for (Eigen::Index v = 1; v < logWeights.size(); ++v)
        {
            if (logWeights(v) > logWeights(best))
            {
                best = v;
            }
        }
        return best;
    }

    /** \returns the parameters of the axis named by the path \a P (a tag or a TagPath), summed
     * over the others, with log-weights that are its log-probabilities
     */
    template <typename P>
    [[nodiscard]] auto
    marginal() const
    {
        return internal::categoricalMarginal<P, Scalar>(space(), logProbabilities());
    }

    /** \returns the parameters of the other axes given that the axis named by the path \a P has
     * the value with index \a value in it; see CategoricalParams::conditional()
     */
    template <typename P>
    [[nodiscard]] auto
    conditional(Eigen::Index value) const
    {
        return source().template conditional<P>(value);
    }

    /** \returns the parameters of the other axes given the named value \a Value; see
     * CategoricalParams::conditional()
     */
    template <typename Value>
    [[nodiscard]] auto
    conditional() const
    {
        return source().template conditional<Value>();
    }

private:

    friend ParamsType;  ///< the parameters construct views

    /** Views \a source, whose weights sum to \f$ e^{\mathrm{logTotal}} \f$, and records its stamp;
     * only CategoricalParams::toDensity() calls this.
     */
    Categorical(ParamsType const & source, Scalar logTotal)
        : m_check{source}
        , m_logTotal{logTotal}
    {}

    /** \returns the source; a programmer error if it has changed since toDensity() */
    [[nodiscard]] ParamsType const &
    source() const
    {
        return m_check.get();
    }

    internal::ValuesCheck<ParamsType> m_check;  ///< the check of the viewed parameters
    Scalar m_logTotal;                          ///< the log of the sum of the source's weights
};

/** \ingroup Distributions
 * \brief A Markov chain on a discrete space: the probability of moving from each value to each
 * value in one step.
 *
 * A time-homogeneous [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) on a finite set.
 * The transition matrix \f$ \Pi \f$, a right
 * [stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix), has a row per value moved
 * from and a column per value moved to, so \f$ \Pi_{ij} = P(j \mid i) \f$ and each row sums to
 * one. predict() moves a Categorical one step: \f$ p'_j = \sum_i p_i \Pi_{ij} \f$.
 *
 * A chain on a DiscreteProduct can be built from independent chains on its axes, whose matrix is
 * their [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product), as a maneuver
 * regime and a sensor's outlier state switch independently:
 *
 * \code
 * MarkovChain const maneuver = MarkovChain<Maneuver>::persistent(0.95);
 * MarkovChain const sensor = MarkovChain<Sensor>::persistent(0.8);
 * auto const mode = MarkovChain<Mode>{maneuver, sensor};
 * \endcode
 *
 * A chain is often known by how long it stays at each value and where it goes when it leaves.
 * fromDwellTimes() builds it from both, and stationary() gives the distribution it settles into,
 * the prior for a mode that has run long before the first measurement:
 *
 * \code
 * // a cruise lasts 30 steps, a turn 8, a dive 4; a cruise ends in a turn twice as often as in a dive
 * auto const exits = Eigen::Matrix3d{{0, 2, 1}, {1, 0, 0}, {1, 0, 0}};
 * auto const chain = MarkovChain<Flight>::fromDwellTimes(Eigen::Vector3d{30, 8, 4}, exits);
 * auto const prior = stationary(chain);  // 9/11, 8/55, 2/55, or a ChainError
 * \endcode
 *
 * \tparam Space_   the discrete space (see \ref Discrete)
 * \tparam Scalar_  scalar type of the probabilities, \c double by default
 */
template <DiscreteSpace Space_, RealScalar Scalar_ = double>
class MarkovChain
{
public:

    using Scalar = Scalar_;                                               ///< the scalar type of the probabilities
    using SpaceType = Space_;                                             ///< the discrete space
    static constexpr int CardinalityAtCompileTime = Space_::Cardinality;  ///< the number of values, or Eigen::Dynamic
    static constexpr int MaxCardinalityAtCompileTime = Space_::MaxCardinality;  ///< a bound on it, or Eigen::Dynamic

    /** The transition matrix's type: a row per value moved from, a column per value moved to. */
    using MatrixType = Eigen::Matrix<
        Scalar_,
        CardinalityAtCompileTime,
        CardinalityAtCompileTime,
        Eigen::RowMajor,
        MaxCardinalityAtCompileTime,
        MaxCardinalityAtCompileTime>;

    /** The chain that never moves. */
    explicit MarkovChain(Space_ space = {})
        : m_space{std::move(space)}
        , m_transitions{MatrixType::Identity(m_space.cardinality(), m_space.cardinality())}
    {}

    /** The chain on \a space with transition matrix \a transitions: nonnegative, with each row
     * summing to one.
     */
    template <MatrixExpression Transitions>
    MarkovChain(Space_ space, Transitions && transitions)
        : m_space{std::move(space)}
        , m_transitions{internal::castScalar<Scalar>(std::forward<Transitions>(transitions))}
    {
        EMRYS_FILTER_ASSERT(
            m_transitions.rows() == m_space.cardinality() && m_transitions.cols() == m_space.cardinality()
            && "the transition matrix has a row and a column per value"
        );
        EMRYS_FILTER_ASSERT(
            (m_transitions.array() >= Scalar(0)).all() && "transition probabilities must be nonnegative"
        );
        EMRYS_FILTER_ASSERT(
            ((m_transitions.rowwise().sum().array() - Scalar(1)).abs() <= Eigen::NumTraits<Scalar>::dummy_precision())
                .all()
            && "each row of the transition matrix must sum to one"
        );
    }

    /** The chain on a FixedSizeSpace, named as the template argument, with transition matrix
     * \a transitions: <tt>MarkovChain<Maneuver>{matrix}</tt> (decisions.md [space-storage]).
     */
    template <MatrixExpression Transitions>
        requires FixedSizeSpace<Space_>
    explicit MarkovChain(Transitions && transitions)
        : MarkovChain{Space_{}, std::forward<Transitions>(transitions)}
    {}

    /** The chain on the DiscreteProduct \a space whose axes move independently, each by its own chain
     * in \a chains, one per axis in order. Its matrix is their Kronecker product.
     */
    template <DiscreteSpace... Axes>
        requires(
            internal::IsDiscreteProduct<Space_>
            && sizeof...(Axes) == std::tuple_size_v<internal::DiscreteProductAxes<Space_>>
        )
    MarkovChain(Space_ space, MarkovChain<Axes, Scalar_> const &... chains)
        : m_space{std::move(space)}
    {
        static_assert(
            std::is_same_v<std::tuple<Axes...>, internal::DiscreteProductAxes<Space_>>,
            "give one chain per axis of the product, in order"
        );
        Eigen::Index const n = m_space.cardinality();
        m_transitions = MatrixType::Ones(n, n);
        Eigen::Index stride = n;
        (
            [&] {
                Eigen::Index const count = chains.space().cardinality();
                stride /= count;
                for (Eigen::Index i = 0; i < n; ++i)
                {
                    for (Eigen::Index j = 0; j < n; ++j)
                    {
                        m_transitions(i, j) *= chains.matrix()((i / stride) % count, (j / stride) % count);
                    }
                }
            }(),
            ...
        );
        EMRYS_FILTER_ASSERT(stride == 1 && "each chain must have its axis's cardinality");
    }

    /** The chain on a DiscreteProduct that is a FixedSizeSpace, named as the template argument,
     * whose axes move independently by \a chains: <tt>MarkovChain<Mode>{maneuver, sensor}</tt>.
     */
    template <DiscreteSpace... Axes>
        requires(
            FixedSizeSpace<Space_> && internal::IsDiscreteProduct<Space_>
            && sizeof...(Axes) == std::tuple_size_v<internal::DiscreteProductAxes<Space_>>
        )
    explicit MarkovChain(MarkovChain<Axes, Scalar_> const &... chains)
        : MarkovChain{Space_{}, chains...}
    {}

    /** \returns the chain on \a space that stays at its value with probability \a stay and
     * otherwise moves to any other value with equal probability
     */
    [[nodiscard]] static MarkovChain
    persistent(Scalar stay, Space_ space = {})
    {
        EMRYS_FILTER_ASSERT(stay >= Scalar(0) && stay <= Scalar(1) && "the probability must be in [0, 1]");
        Eigen::Index const n = space.cardinality();
        auto result = MarkovChain{std::move(space)};
        if (n > 1)
        {
            result.m_transitions.setConstant((Scalar(1) - stay) / Scalar(n - 1));
            result.m_transitions.diagonal().setConstant(stay);
        }
        return result;
    }

    /** \returns the chain that moves to each value with the probabilities of \a next, wherever it
     * starts: every row is the same, so the value at each step is independent of the last
     */
    [[nodiscard]] static MarkovChain
    memoryless(Categorical<Space_, Scalar_> const & next)
    {
        auto result = MarkovChain{next.space()};
        result.m_transitions = next.probabilities().matrix().transpose().replicate(next.cardinality(), 1);
        return result;
    }

    /** \returns the chain on \a space that stays at value \f$ i \f$ for a mean of \f$ \tau_i \f$
     * steps, \a dwellTimes, and on leaving moves to each other value \f$ j \f$ in proportion to
     * \f$ w_{ij} \f$, \a exitWeights: \f$ \Pi_{ii} = 1 - 1/\tau_i \f$ and
     * \f$ \Pi_{ij} = w_{ij} / (\tau_i \sum_k w_{ik}) \f$.
     *
     * The number of steps at a value, counting the step that reached it, is
     * [geometric](https://en.wikipedia.org/wiki/Geometric_distribution) with mean
     * \f$ 1 / (1 - \Pi_{ii}) \f$, so each \f$ \tau_i \f$ is at least one, and an infinite one never
     * leaves. The exit weights are nonnegative, need not sum to one, and are zero on the diagonal;
     * a value that is left needs one that isn't zero. A dwell time in seconds with a fixed step
     * is ContinuousTimeMarkovChain::fromDwellTimes() and its transition() over the step, which is
     * exact where \f$ 1 - \Delta t / \tau_i \f$ is not.
     */
    template <typename DwellTimes, typename ExitWeights>
    [[nodiscard]] static MarkovChain
    fromDwellTimes(
        Eigen::MatrixBase<DwellTimes> const & dwellTimes,
        Eigen::MatrixBase<ExitWeights> const & exitWeights,
        Space_ space = {}
    )
    {
        EMRYS_FILTER_ASSERT((dwellTimes.array() >= 1).all() && "a dwell time is at least one step");
        MatrixType transitions = internal::dwellGenerator<Scalar, MatrixType>(dwellTimes, exitWeights);
        transitions.diagonal().array() += Scalar(1);
        EMRYS_FILTER_ASSERT(transitions.rows() == space.cardinality() && "give a dwell time per value");
        return MarkovChain{std::move(space), std::move(transitions)};
    }

    /** \returns the chain on \a space that stays at value \f$ i \f$ for a mean of \f$ \tau_i \f$
     * steps, \a dwellTimes, and on leaving moves to any other value with equal probability: the
     * other fromDwellTimes() with every exit weight one
     */
    template <typename DwellTimes>
    [[nodiscard]] static MarkovChain
    fromDwellTimes(Eigen::MatrixBase<DwellTimes> const & dwellTimes, Space_ space = {})
    {
        Eigen::Index const n = space.cardinality();
        return fromDwellTimes(dwellTimes, MatrixType::Ones(n, n) - MatrixType::Identity(n, n), std::move(space));
    }

    /** \returns the discrete space */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

    /** \returns the transition matrix \f$ \Pi \f$, with \f$ \Pi_{ij} = P(j \mid i) \f$ */
    [[nodiscard]] MatrixType const &
    matrix() const
    {
        return m_transitions;
    }

    /** \returns the probability of moving from the value \a from to the value \a to in one step */
    [[nodiscard]] Scalar
    transition(Eigen::Index from, Eigen::Index to) const
    {
        return m_transitions(from, to);
    }

    /** \returns the distribution of the value one step after the value \a from: row \a from of the
     * transition matrix, as parameters whose log-weights are its log-probabilities. simulate()
     * draws from it without building it.
     */
    [[nodiscard]] CategoricalParams<Space_, Scalar_>
    next(Eigen::Index from) const
    {
        EMRYS_FILTER_ASSERT(from >= 0 && from < m_space.cardinality() && "the value is not in the chain's space");
        return CategoricalParams<Space_, Scalar_>{m_space, m_transitions.row(from).transpose().array().log()};
    }

private:

    Space_ m_space;            ///< the discrete space
    MatrixType m_transitions;  ///< the transition matrix \f$ \Pi \f$
};

/// \cond DEDUCTION_GUIDES
template <DiscreteSpace Space, MatrixExpression Transitions>
MarkovChain(Space, Transitions &&) -> MarkovChain<Space, typename std::remove_cvref_t<Transitions>::Scalar>;

/// \endcond

/** \ingroup Distributions
 * \returns \a prior moved one step by \a chain: \f$ w'_j = \sum_i w_i \Pi_{ij} \f$, the
 * Chapman-Kolmogorov step of a discrete chain. The rows of \f$ \Pi \f$ sum to one, so the weights
 * keep their total, and weights that fail to define a distribution still fail to.
 */
template <typename Scalar, typename Space>
[[nodiscard]] CategoricalParams<Space, Scalar>
predict(CategoricalParams<Space, Scalar> const & prior, MarkovChain<Space, Scalar> const & chain)
{
    EMRYS_FILTER_ASSERT(prior.space() == chain.space() && "the chain is on a different space");
    return CategoricalParams<Space, Scalar>{
        prior.space(),
        internal::movedLogWeights<Scalar>(prior.logWeights(), chain.matrix())
    };
}

/** \ingroup Distributions
 * \returns the stationary distribution of \a chain, the probabilities \f$ \pi \f$ that one step
 * leaves as they are, \f$ \pi \Pi = \pi \f$: the distribution the chain settles into from any
 * start, and the prior for a mode that has run long before the first measurement. It exists and
 * is unique when every value can be reached from every other, and otherwise is
 * ChainError::Reducible. It is computed by the algorithm of
 * \ref Grassmann1985 "Grassmann, Taksar, and Heyman (1985)", which keeps every digit when the chain
 * stays a long time at each value, as \f$ \pi (\Pi - I) = 0 \f$ solved directly doesn't.
 */
template <typename Scalar, typename Space>
[[nodiscard]] std::expected<CategoricalParams<Space, Scalar>, ChainError>
stationary(MarkovChain<Space, Scalar> const & chain)
{
    using ArrayType = typename CategoricalParams<Space, Scalar>::ArrayType;
    std::expected<ArrayType, ChainError> const probabilities =
        internal::stationaryProbabilities<Scalar, ArrayType>(chain.matrix());
    if (!probabilities)
    {
        return std::unexpected{probabilities.error()};
    }
    return CategoricalParams<Space, Scalar>{chain.space(), probabilities->log()};
}

/** \ingroup Distributions
 * \returns \a n independent draws from \a distribution, each a value's flat index, by
 * <tt>std::discrete_distribution</tt> over the probabilities
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] std::vector<Eigen::Index>
drawSamples(Categorical<Space, Scalar> const & distribution, Eigen::Index n, G & rng)
{
    static_assert(std::floating_point<Scalar>, "sampling requires a standard floating-point scalar");
    EMRYS_FILTER_ASSERT(n >= 0);
    typename Categorical<Space, Scalar>::ArrayType const probabilities = distribution.probabilities();
    auto draw =
        std::discrete_distribution<Eigen::Index>{probabilities.data(), probabilities.data() + probabilities.size()};
    std::size_t const count = std::size_t(n);
    auto samples = std::vector<Eigen::Index>(count);
    for (auto & sample : samples)
    {
        sample = draw(rng);
    }
    return samples;
}

/** \ingroup Distributions
 * \returns one draw from \a distribution, a value's flat index: the stream of draws that
 * drawSample() of a %Gaussian is. It compares one uniform draw with the running sum of the
 * probabilities, since <tt>std::discrete_distribution</tt> copies them onto the heap, so it
 * allocates nothing, and is not the sequence drawSamples() gives from the same generator.
 */
template <typename Scalar, typename Space, std::uniform_random_bit_generator G>
[[nodiscard]] Eigen::Index
drawSample(Categorical<Space, Scalar> const & distribution, G & rng)
{
    auto const probabilities = distribution.probabilities();
    return internal::drawIndex<Scalar>(
        probabilities.size(),
        [&](Eigen::Index i) {
            return probabilities(i);
        },
        rng
    );
}

}  // namespace Emrys

#ifndef EMRYS_FILTER_NO_IO

namespace Emrys
{

namespace internal
{

/** Describes \a categorical: its value count and each value's probability. */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, Categorical<Space_, Scalar_> const & categorical, Indent indent)
{
    typename Categorical<Space_, Scalar_>::ArrayType const probabilities = categorical.probabilities();
    stream << indent << "categorical over ";
    printCount(stream, probabilities.size(), "value");
    stream << '\n';
    printRow(stream, indent.deeper(), "probabilities", probabilities);
}

/** Describes \a params: its value count and each value's probability, or its log-weights if they
 * fail to define a distribution.
 */
template <typename Space_, typename Scalar_>
void
describe(std::ostream & stream, CategoricalParams<Space_, Scalar_> const & params, Indent indent)
{
    if (auto const density = params.toDensity())
    {
        describe(stream, *density, indent);
        return;
    }
    stream << indent << "categorical over ";
    printCount(stream, params.cardinality(), "value");
    stream << ", whose weights are not normalizable\n";
    printRow(stream, indent.deeper(), "log-weights", params.logWeights());
}

}  // namespace internal

}  // namespace Emrys

#endif  // EMRYS_FILTER_NO_IO

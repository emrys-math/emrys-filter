#pragma once

/** \file
 * Consistency checks (decisions.md [composed-features]): NisMonitor, a windowed chi-squared test on the
 * normalized innovation squared of a filter's updates, and nees() with NeesMonitor, the same test
 * on the normalized estimation error squared against a true state, which a simulation knows (see
 * simulate()). Both are fed by the caller. Read \ref Filters next.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <expected>
#include <limits>
#include <type_traits>
#include <utility>

#include <EmrysFilter/Core/Allocation.h>
#include <EmrysFilter/Core/ChiSquared.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/NumericError.h>
#include <EmrysFilter/Spaces/TagPath.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Models/Measurement.h>

namespace Emrys
{

/** \ingroup Filters
 * The window of a NisMonitor or NeesMonitor that drops nothing: its test runs over every
 * statistic added since construction or reset(), and it keeps only their sum and their total
 * degrees of freedom. Where Eigen::Dynamic is a window whose length is given at run time, this is
 * one with no length at all.
 */
inline constexpr int Cumulative = -2;

static_assert(Cumulative != Eigen::Dynamic, "a cumulative window is not a dynamic one");

/** \ingroup Filters
 * The verdict of a NisMonitor's or NeesMonitor's verdict(): where the sum of the window's
 * statistics lies against the two-sided bounds() at a probability \f$ p \f$, which split
 * \f$ 1 - p \f$ equally between the two tails. A consistent filter is found Overconfident with
 * probability \f$ (1 - p) / 2 \f$, and Underconfident with the same probability; a one-sided test
 * at level \f$ \alpha \f$ passes \f$ p = 1 - 2\alpha \f$ and reads one of the two.
 *
 * The two differ in kind. Either way the filter combines its prior and the measurements by
 * covariances that are not the true ones, and so less precisely than it could. An overconfident
 * filter's covariance is smaller than its errors, so it reports its estimates as better than they
 * are: an error, which can leave an estimate far from the truth and still trusted. An
 * underconfident filter's covariance is larger than its errors, so its estimates are no worse
 * than it reports: not an error, only the loss of precision, such as from measurement or process
 * noise set larger than the system's.
 */
enum class Consistency
{
    Consistent,      ///< the sum lies within the bounds
    Overconfident,   ///< the sum lies above the upper bound: the covariance understates the errors
    Underconfident,  ///< the sum lies below the lower bound: the covariance overstates the errors
};

namespace internal
{

/** The two-sided test of a sum of chi-squared statistics against the chi-squared distribution
 * with their total degrees of freedom, over the count(), sum(), and degreesOfFreedom() of
 * \a Derived: the test of NisMonitor and NeesMonitor, whatever their window keeps. Each fails on
 * a window that holds no statistic, and those that read the sum on a NaN in it, both of which
 * valid use can leave: a monitor fed only the runs a test trusts is empty when it trusts none,
 * and NeesMonitor adds NaN for an estimate whose covariance doesn't factorize.
 */
template <typename Derived, typename Scalar>
class ChiSquaredTest
{
public:

    /** \returns sum() over degreesOfFreedom(), which is near 1 for a consistent filter; or Empty
     * of NumericError if the window holds no statistic, or NonFinite if one it holds is NaN
     */
    [[nodiscard]] std::expected<Scalar, NumericError>
    average() const
    {
        if (derived().count() == 0)
        {
            return std::unexpected{NumericError::Empty};
        }
        Scalar const total = derived().sum();
        if (isNaN(total))
        {
            return std::unexpected{NumericError::NonFinite};
        }
        return total / internal::scalarFromCount<Scalar>(derived().degreesOfFreedom());
    }

    /** \returns the two-sided bounds on sum() that a consistent filter stays within with
     * probability \a probability, below the lower with probability (1 - \a probability) / 2 and
     * above the upper with the same; or Empty of NumericError if the window holds no statistic
     */
    [[nodiscard]] std::expected<std::pair<Scalar, Scalar>, NumericError>
    bounds(Scalar probability) const
    {
        if (derived().count() == 0)
        {
            return std::unexpected{NumericError::Empty};
        }
        Scalar const dof = internal::scalarFromCount<Scalar>(derived().degreesOfFreedom());
        return std::pair<Scalar, Scalar>{
            chiSquaredQuantile<Scalar>((Scalar(1) - probability) / Scalar(2), dof),
            chiSquaredQuantile<Scalar>((Scalar(1) + probability) / Scalar(2), dof)
        };
    }

    /** \returns where sum() lies against bounds() at \a probability, as a Consistency; or Empty of
     * NumericError if the window holds no statistic, or NonFinite if one it holds is NaN. The
     * verdict is no \c bool, so that <tt>!monitor.verdict(p)</tt>, which tests whether there is
     * one, can't be read as a test of consistency.
     */
    [[nodiscard]] std::expected<Consistency, NumericError>
    verdict(Scalar probability) const
    {
        auto const range = bounds(probability);
        if (!range)
        {
            return std::unexpected{range.error()};
        }
        Scalar const total = derived().sum();
        if (isNaN(total))
        {
            return std::unexpected{NumericError::NonFinite};
        }
        if (total > range->second)
        {
            return Consistency::Overconfident;
        }
        if (total < range->first)
        {
            return Consistency::Underconfident;
        }
        return Consistency::Consistent;
    }

private:

    /** \returns this as the window it is */
    [[nodiscard]] Derived const &
    derived() const
    {
        return static_cast<Derived const &>(*this);
    }
};

/** A sliding window of chi-squared statistics, each with its degrees of freedom: the body of
 * NisMonitor and NeesMonitor with a window of \a Window statistics, or of a length given at run
 * time if it is Eigen::Dynamic, up to \a MaxWindow. A window with a bound is a ring inline; one
 * without takes its ring from \a Allocator once, at construction.
 */
template <int Window, int MaxWindow, typename Scalar, typename Allocator>
class ChiSquaredWindow : public ChiSquaredTest<ChiSquaredWindow<Window, MaxWindow, Scalar, Allocator>, Scalar>
{
    static_assert(Window == Eigen::Dynamic || Window > 0, "a window holds at least one statistic");
    static_assert(
        Window == Eigen::Dynamic ? MaxWindow == Eigen::Dynamic || MaxWindow > 0 : MaxWindow == Window,
        "a window's bound is positive, and a fixed window is its own bound"
    );

    using Entry = std::pair<Scalar, Eigen::Index>;               ///< a statistic and its degrees of freedom
    static constexpr bool Inline = MaxWindow != Eigen::Dynamic;  ///< whether the ring has a bound
    using Entries = std::conditional_t<
        Inline,
        std::array<Entry, std::size_t(Inline ? MaxWindow : 1)>,
        internal::Buffer<Entry, Allocator>>;  ///< the ring
    using Length = std::conditional_t<
        Window == Eigen::Dynamic,
        Eigen::Index,
        std::integral_constant<Eigen::Index, Window == Eigen::Dynamic ? 1 : Window>>;  ///< the window's length

public:

    using allocator_type = Allocator;  ///< the allocator of an unbounded window's ring

    /** A window over the most recent \a Window statistics. */
    ChiSquaredWindow()
        requires(Window != Eigen::Dynamic)
    = default;

    /** A window over the most recent \a window statistics, whose ring, if it has no bound, comes
     * from \a allocator.
     */
    ChiSquaredWindow(Eigen::Index window, Allocator const & allocator)
        requires(Window == Eigen::Dynamic)
        : m_window{window}
        , m_entries(entries(window, allocator))
    {
        EMRYS_FILTER_ASSERT(window > 0 && "the window must hold at least one statistic");
        EMRYS_FILTER_ASSERT((!Inline || window <= MaxWindow) && "the window must be within its bound");
    }

    /** Copies \a other into a ring from \a allocator. */
    ChiSquaredWindow(ChiSquaredWindow const & other, Allocator const & allocator)
        requires(!Inline)
        : m_window{other.m_window}
        , m_entries(other.m_entries, internal::ReBound<Allocator, Entry>(allocator))
        , m_count{other.m_count}
        , m_next{other.m_next}
    {}

    /** \returns the allocator of the ring */
    [[nodiscard]] Allocator
    get_allocator() const
        requires(!Inline)
    {
        return Allocator(m_entries.get_allocator());
    }

    /** Adds the statistic \a value with \a dimension degrees of freedom, dropping the oldest once
     * the window is full.
     */
    void
    add(Scalar value, Eigen::Index dimension)
    {
        EMRYS_FILTER_ASSERT(dimension > 0 && "the statistic must have at least one degree of freedom");
        m_entries[std::size_t(m_next)] = Entry{value, dimension};
        m_next = m_next + 1 == window() ? 0 : m_next + 1;
        m_count = std::min(m_count + 1, window());
    }

    /** Empties the window. */
    void
    reset()
    {
        m_count = 0;
        m_next = 0;
    }

    /** \returns the window's length */
    [[nodiscard]] Eigen::Index
    window() const
    {
        return m_window;
    }

    /** \returns the number of statistics in the window */
    [[nodiscard]] Eigen::Index
    count() const
    {
        return m_count;
    }

    /** \returns the sum of the statistics in the window */
    [[nodiscard]] Scalar
    sum() const
    {
        Scalar total = 0;
        for (Eigen::Index i = 0; i < m_count; ++i)
        {
            total += m_entries[std::size_t(i)].first;
        }
        return total;
    }

    /** \returns the total degrees of freedom over the window */
    [[nodiscard]] Eigen::Index
    degreesOfFreedom() const
    {
        Eigen::Index total = 0;
        for (Eigen::Index i = 0; i < m_count; ++i)
        {
            total += m_entries[std::size_t(i)].second;
        }
        return total;
    }

private:

    /** \returns a ring for \a window statistics: inline, or from \a allocator */
    [[nodiscard]] static Entries
    entries(Eigen::Index window, Allocator const & allocator)
    {
        if constexpr (Inline)
        {
            return Entries{};
        }
        else
        {
            return Entries(std::size_t(window), Entry{}, internal::ReBound<Allocator, Entry>(allocator));
        }
    }

    EMRYS_FILTER_NO_UNIQUE_ADDRESS Length m_window{};  ///< the window's length
    Entries m_entries{};                               ///< each statistic and its degrees of freedom
    Eigen::Index m_count = 0;                          ///< the number of statistics in the window
    Eigen::Index m_next = 0;                           ///< where the next statistic goes
};

/** The Cumulative window of chi-squared statistics: their sum and total degrees of freedom since
 * construction or reset(). A NaN added stays in the sum until reset().
 */
template <typename Scalar>
class ChiSquaredSum : public ChiSquaredTest<ChiSquaredSum<Scalar>, Scalar>
{
public:

    /** Adds the statistic \a value with \a dimension degrees of freedom. */
    void
    add(Scalar value, Eigen::Index dimension)
    {
        EMRYS_FILTER_ASSERT(dimension > 0 && "the statistic must have at least one degree of freedom");
        m_sum += value;
        m_degreesOfFreedom += dimension;
        ++m_count;
    }

    /** Empties the window. */
    void
    reset()
    {
        m_sum = 0;
        m_degreesOfFreedom = 0;
        m_count = 0;
    }

    /** \returns the number of statistics added */
    [[nodiscard]] Eigen::Index
    count() const
    {
        return m_count;
    }

    /** \returns the sum of the statistics added */
    [[nodiscard]] Scalar
    sum() const
    {
        return m_sum;
    }

    /** \returns the total degrees of freedom of the statistics added */
    [[nodiscard]] Eigen::Index
    degreesOfFreedom() const
    {
        return m_degreesOfFreedom;
    }

private:

    Scalar m_sum = 0;                     ///< the sum of the statistics
    Eigen::Index m_degreesOfFreedom = 0;  ///< their total degrees of freedom
    Eigen::Index m_count = 0;             ///< their number
};

/** The window of a monitor with \a Window and \a MaxWindow: ChiSquaredSum for Cumulative, and
 * ChiSquaredWindow otherwise.
 */
template <int Window, int MaxWindow, typename Scalar, typename Allocator>
using ChiSquaredWindowFor = std::conditional_t<
    Window == Cumulative,  // a sum since the first statistic, or a ring of the latest
    ChiSquaredSum<Scalar>,
    ChiSquaredWindow<Window, MaxWindow, Scalar, Allocator>>;

}  // namespace internal

/** \ingroup Filters
 * \brief A chi-squared consistency check over a window of updates.
 *
 * For a consistent filter, each update's normalized innovation squared (NIS, see UpdateBranch)
 * is chi-squared with the innovation's dimension as its degrees of freedom, and the sum over
 * the window is chi-squared with the total. verdict() tests that sum against two-sided
 * bounds: above them the filter is overconfident, such as with noise set too small, an error;
 * below them it is underconfident, a loss of precision but no error (see Consistency).
 *
 * The monitor is fed by the caller, so a filter that doesn't use one pays nothing:
 *
 * \code
 * auto monitor = NisMonitor<50>{};
 * if (auto const updated = filter.update(fix)) { monitor.add(*updated); }
 * if (monitor.verdict(0.99) == Consistency::Overconfident) { ... }
 * \endcode
 *
 * A std::expected compares equal to a value only when it holds one, so the comparison is false
 * while the window is empty.
 *
 * The window is sized as an Eigen vector is. \a Window is its length, a ring inline; with
 * Eigen::Dynamic the length is given at construction, as in <tt>NisMonitor{50}</tt>, inline up to
 * \a MaxWindow where that is a bound and from \a Allocator where it is Eigen::Dynamic too. With
 * Cumulative there is no window: the test runs over every update since construction or reset(),
 * the time-average test over a whole run, and keeps only a sum. Its bounds narrow as the run
 * grows, so a long run fails on a small mistuning that a moving window, which a monitor on a
 * running system wants, passes; and a NaN, once added, stays until reset().
 *
 * add(update) counts an update whether or not its policy accepted it (decisions.md
 * [update-outcomes]). The test asks whether the filter predicts the measurements it is given,
 * and a gate rejects the ones it predicts worst: leaving them out would hide a filter that has
 * diverged behind its gate, and biases the sum low even for a consistent filter, whose accepted
 * updates under a 0.99 gate on two coordinates average 0.953 rather than 1. A caller who wants the
 * accepted updates alone tests \c accepted and passes those. With mixture noise there are several
 * branches, each with its own NIS, so pass the one to test, such as the inlier component's, with
 * add(nis, dimension). The filters of several components, modes, or points are tested the same
 * way, on the NIS of the dominant branch: the most probable branch of the most probable component
 * or mode. That NIS is chi-squared only while the dominant branch carries nearly all of the
 * probability, since the predictive distribution of the measurement is a mixture, so the test
 * reads a filter that is often split between branches as inconsistent when it may not be.
 *
 * The NIS needs nothing but the filter and the data, so it runs on a real system. It can pass
 * while the state's error is far larger than the covariance says, in a direction the
 * measurements don't see; in a simulation, where the true state is known, NeesMonitor tests
 * the state itself.
 *
 * The time-average normalized innovation squared test of
 * \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", over a moving window or a whole
 * run. The NIS and its chi-squared law are as given in \ref Chen2023 "Chen et al. (2023)", §III-A,
 * equation (8); the test over one run's window is this library's form of their test across runs,
 * §III-B.
 *
 * \sa NeesMonitor
 */
template <
    int Window = Eigen::Dynamic,
    int MaxWindow = Window,
    typename Scalar = double,
    typename Allocator = DefaultAllocator>
class NisMonitor : public internal::ChiSquaredWindowFor<Window, MaxWindow, Scalar, Allocator>
{
    static_assert(Window != Cumulative || MaxWindow == Cumulative, "a cumulative window has no bound");

    using Base = internal::ChiSquaredWindowFor<Window, MaxWindow, Scalar, Allocator>;  ///< the window

public:

    /** A monitor over the most recent \a Window updates, or over every update for Cumulative. */
    NisMonitor()
        requires(Window != Eigen::Dynamic)
    = default;

    /** A monitor over the most recent \a window updates, whose window, if it has no bound, comes
     * from \a allocator.
     */
    explicit NisMonitor(Eigen::Index window, Allocator const & allocator = {})
        requires(Window == Eigen::Dynamic)
        : Base{window, allocator}
    {}

    /** Copies \a other into a window from \a allocator. */
    NisMonitor(NisMonitor const & other, Allocator const & allocator)
        requires(Window == Eigen::Dynamic && MaxWindow == Eigen::Dynamic)
        : Base{other, allocator}
    {}

    using Base::add;

    /** Adds an update with a single branch, as from %Gaussian noise: a MeasurementUpdate, or the
     * FilterUpdate a filter returns, accepted or not (see the class comment).
     */
    template <typename Update>
        requires requires(Update const & update) {
            update.branches.front().nis();
            update.innovationSize();
        }
    void
    add(Update const & update)
    {
        EMRYS_FILTER_ASSERT(
            update.branches.size() == 1 && "with several branches, pass the NIS to test with add(nis, dimension)"
        );
        add(update.branches.front().nis(), update.innovationSize());
    }
};

/** \ingroup Filters
 * \returns the normalized estimation error squared (NEES) of \a estimate at the true state
 * \a truth, \f$ (x \boxminus \hat x)^T P^{-1} (x \boxminus \hat x) \f$, or the GaussianError of
 * a covariance that doesn't factorize.
 *
 * \a estimate is a GaussianForm: a %Gaussian in any of its forms, such as a Kalman filter's
 * state, or the collapse of a mixture or a Rao-Blackwellized state, or a marginal of one; its
 * difference dimension is the degrees of freedom. It is the squared Mahalanobis distance of the
 * truth from the estimate, the density's squaredMahalanobisDistance(), with the error taken by
 * \f$ \boxminus \f$ in the estimate's space, so a quaternion's is a rotation vector. A
 * GaussianMarginal view, which has no density of its own, is copied into parameters first. For a
 * consistent filter it is chi-squared with that dimension; see NeesMonitor for the test, and
 * nees<Path>() for one block of the state.
 *
 * The normalized state estimation error squared of
 * \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", as given in
 * \ref Chen2023 "Chen et al. (2023)", §III-A, equation (7).
 */
template <GaussianForm Estimate, typename Derived>
[[nodiscard]] std::expected<typename Estimate::Scalar, GaussianError>
nees(Eigen::MatrixBase<Derived> const & truth, Estimate const & estimate)
{
    EMRYS_FILTER_ASSERT(truth.size() == estimate.valueDimension() && "the true state must fit the estimate's space");
    if constexpr (requires { estimate.toDensity(); })
    {
        auto const density = estimate.toDensity();
        if (!density)
        {
            return std::unexpected{density.error()};
        }
        return typename Estimate::Scalar(density->squaredMahalanobisDistance(truth));
    }
    else
    {
        auto const params = toGaussian(estimate);
        if (!params)
        {
            return std::unexpected{params.error()};
        }
        return nees(truth, *params);
    }
}

/** \ingroup Filters
 * \returns the nees() of the block of \a estimate named by the path \a Path (a tag or a TagPath),
 * at the same block of the true state \a truth, which is a value of the whole state: the
 * estimate's marginal<Path>() at the truth's block<Path>(), or the GaussianError of either. The
 * degrees of freedom are the block's difference dimension, one for a block of one coordinate.
 *
 * Testing a block apart is how to find which part of a state is inconsistent: a filter
 * underconfident in a velocity its fixes don't measure passes on the position and fails on the
 * whole state. Reading both by one path keeps the truth's entries
 * matched to the estimate's, where an offset written by hand would follow the space's layout.
 *
 * \code
 * auto const positionNees = nees<Position>(truth, filter.state());
 * \endcode
 */
template <typename Path, GaussianForm Estimate, typename Derived>
    requires requires(Estimate const & estimate) { estimate.template marginal<Path>(); }
[[nodiscard]] std::expected<typename Estimate::Scalar, GaussianError>
nees(Eigen::MatrixBase<Derived> const & truth, Estimate const & estimate)
{
    EMRYS_FILTER_ASSERT(truth.size() == estimate.valueDimension() && "the true state must fit the estimate's space");
    // block() is a vector even for one coordinate, where truth(idx<Path>()) would be a scalar.
    auto const block = Emrys::block<Path>(estimate.space(), truth);
    auto const marginal = estimate.template marginal<Path>();
    if constexpr (internal::isExpected<std::remove_cvref_t<decltype(marginal)>>)
    {
        if (!marginal)
        {
            return std::unexpected{marginal.error()};
        }
        return nees(block, *marginal);
    }
    else
    {
        return nees(block, marginal);
    }
}

/** \ingroup Filters
 * \brief A chi-squared consistency check of a filter's estimates against the true states, over a
 * window of them.
 *
 * For a consistent filter, each estimate's normalized estimation error squared (see nees()) is
 * chi-squared with the state's difference dimension as its degrees of freedom, and the sum of
 * independent ones is chi-squared with the total. verdict() tests that sum against two-sided
 * bounds, as NisMonitor does: above them the filter is overconfident, an error, and below them
 * underconfident, a loss of precision but no error (see Consistency). The truth comes from a simulation (see
 * simulate()), and the estimates must be independent, so the monitor takes one per run, from runs with their own truth
 * and noise, at the same step:
 *
 * \code
 * auto monitor = NeesMonitor<Cumulative>{};
 * for (int run = 0; run < 100; ++run)
 * {
 *     auto truth = drawSample(prior.toDensity().value(), rng);
 *     auto filter = KalmanFilter{model, prior};
 *     for (int k = 0; k < 50; ++k)
 *     {
 *         truth = simulate(space, truth, model, rng, dt);
 *         filter.predict(dt);
 *         filter.update(observe(gps, simulate(space, truth, gps, rng)));
 *     }
 *     monitor.add(truth, filter.state());
 * }
 * auto const verdict = monitor.verdict(0.99);
 * if (!verdict) { ... }                                   // no run was added, or one's estimate had no density
 * else if (*verdict == Consistency::Overconfident) { ... }
 * \endcode
 *
 * Here \c gps is a measurement model (see measurementModel()), whose simulate() draws a
 * measurement of the true state from the noise the filter assumes. The window is sized as NisMonitor's is.
 *
 * This is the Monte Carlo average test of
 * \ref BarShalom2001 "Bar-Shalom, Li, and Kirubarajan (2001)", as given in
 * \ref Chen2023 "Chen et al. (2023)", §III-B, equations (12) and (14). Unlike the NIS, the NEES
 * of one run is no test over its steps: a filter's errors are correlated from step to step, where
 * its innovations are white, so their sum spreads far wider than the chi-squared with the total,
 * and a consistent filter fails often. A linear Kalman filter, consistent to a mean NEES of 1.00
 * over 60 seeds, failed the 0.999 test over 400 steps of one run on 11 seeds in 100.
 *
 * \sa NisMonitor
 */
template <
    int Window = Eigen::Dynamic,
    int MaxWindow = Window,
    typename Scalar = double,
    typename Allocator = DefaultAllocator>
class NeesMonitor : public internal::ChiSquaredWindowFor<Window, MaxWindow, Scalar, Allocator>
{
    static_assert(Window != Cumulative || MaxWindow == Cumulative, "a cumulative window has no bound");

    using Base = internal::ChiSquaredWindowFor<Window, MaxWindow, Scalar, Allocator>;  ///< the window

public:

    /** A monitor over the most recent \a Window estimates, or over every estimate for Cumulative. */
    NeesMonitor()
        requires(Window != Eigen::Dynamic)
    = default;

    /** A monitor over the most recent \a window estimates, whose window, if it has no bound, comes
     * from \a allocator.
     */
    explicit NeesMonitor(Eigen::Index window, Allocator const & allocator = {})
        requires(Window == Eigen::Dynamic)
        : Base{window, allocator}
    {}

    /** Copies \a other into a window from \a allocator. */
    NeesMonitor(NeesMonitor const & other, Allocator const & allocator)
        requires(Window == Eigen::Dynamic && MaxWindow == Eigen::Dynamic)
        : Base{other, allocator}
    {}

    using Base::add;

    /** Adds the estimate \a estimate of the true state \a truth: its nees(), with its difference
     * dimension as the degrees of freedom. An estimate whose covariance doesn't factorize adds
     * NaN, so average() and verdict() fail with NonFinite until it leaves the window, or for
     * Cumulative until reset().
     */
    template <GaussianForm Estimate, typename Derived>
        requires requires(Estimate const & estimate, Eigen::MatrixBase<Derived> const & truth) {
            nees(truth, estimate);
        }
    void
    add(Eigen::MatrixBase<Derived> const & truth, Estimate const & estimate)
    {
        auto const value = nees(truth, estimate);
        add(value ? Scalar(*value) : std::numeric_limits<Scalar>::quiet_NaN(), estimate.differenceDimension());
    }

    /** Adds the block of \a estimate named by the path \a Path at the same block of the true state
     * \a truth: its nees<Path>(), with the block's difference dimension as the degrees of freedom,
     * and NaN as add(truth, estimate) adds it. A monitor per block, fed alongside one of the whole
     * state, says which block is inconsistent.
     */
    template <typename Path, typename Derived, typename Estimate>
        requires requires(Estimate const & estimate, Eigen::MatrixBase<Derived> const & truth) {
            nees<Path>(truth, estimate);
        }
    void
    add(Eigen::MatrixBase<Derived> const & truth, Estimate const & estimate)
    {
        auto const value = nees<Path>(truth, estimate);
        add(value ? Scalar(*value) : std::numeric_limits<Scalar>::quiet_NaN(),
            internal::differenceBlockRows<Path>(estimate.space()).size);
    }
};

}  // namespace Emrys

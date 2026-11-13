#pragma once

// Support shared by the benchmark runners, so that a new problem writes only its scenario and its
// two implementations: the command line, the compiler line, a way to keep a measured function out
// of line so that Callgrind can attribute instructions to it by name, the summary statistics of a
// set of timed samples, the replay of a filter over a simulated run, the pair driver that times
// two implementations of one problem against each other, interleaved, and the table's columns.
//
// A runner checks that its two implementations agree before it times them, in its own terms, since
// what agreement means differs by problem: to rounding for a pair of Kalman filters, particle by
// particle for a pair of particle filters driven by the same draws.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/// Keeps a function out of line, so that it appears under its own name in a profile and in
/// Callgrind's --toggle-collect.
#if defined(_MSC_VER) && !defined(__clang__)
#    define EMRYS_BENCHMARK_NOINLINE __declspec(noinline)
#else
#    define EMRYS_BENCHMARK_NOINLINE __attribute__((noinline))
#endif

namespace Benchmark
{

/// What the command line asks for.
struct Options
{
    int samples = 2000;       ///< timed batches of each filter in each case
    bool count = false;       ///< run only the agreement pass, for Callgrind
    long long particles = 0;  ///< for a runner with a particle count axis: run only this count, or every one if 0
};

/// Reads the command line into \a options, or returns false if it can't.
[[nodiscard]] inline bool
parse(int argc, char ** argv, Options & options)
{
    for (int i = 1; i < argc; ++i)
    {
        auto const argument = std::string_view{argv[i]};
        if (argument == "--count")
        {
            options.count = true;
        }
        else if (argument == "--samples" && i + 1 < argc)
        {
            options.samples = std::atoi(argv[++i]);
            if (options.samples < 1)
            {
                return false;
            }
        }
        else if (argument == "--particles" && i + 1 < argc)
        {
            options.particles = std::atoll(argv[++i]);
            if (options.particles < 1)
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

/// Prints how to call a runner, for a command line parse() rejects.
inline void
printUsage(char const * program)
{
    std::fprintf(stderr, "usage: %s [--samples N] [--particles N] [--count]\n", program);
}

/// The compiler, so that a table records what it measured.
[[nodiscard]] inline char const *
compilerName()
{
#define EMRYS_BENCHMARK_STRING(x) #x
#define EMRYS_BENCHMARK_EXPAND(x) EMRYS_BENCHMARK_STRING(x)
#if defined(__clang__)
    return "Clang " EMRYS_BENCHMARK_EXPAND(__clang_major__) "." EMRYS_BENCHMARK_EXPAND(
        __clang_minor__
    ) "." EMRYS_BENCHMARK_EXPAND(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " EMRYS_BENCHMARK_EXPAND(_MSC_FULL_VER);
#else
    return "an unknown compiler";
#endif
#undef EMRYS_BENCHMARK_EXPAND
#undef EMRYS_BENCHMARK_STRING
}

/// Prints the line a table opens with: what \a title times, and the compiler. Warns when the build
/// has assertions, whose cost would be in the times.
inline void
printPreamble(char const * title)
{
    std::printf("%s, EmrysFilter against plain Eigen, built with %s.\n", title, compilerName());
#ifndef NDEBUG
    std::printf("Warning: built without NDEBUG, so the times include assertions; use a release build.\n");
#endif
}

/// Keeps the compiler from discarding a filter's steps: their result goes somewhere it can't see.
inline void
consume(double value)
{
    static double volatile sink{0.0};
    sink = sink + value;
}

/// The median, 99th percentile, and largest of a set of timed samples.
struct Summary
{
    double median;  ///< the middle sample
    double p99;     ///< the sample that 99% of samples do not exceed
    double worst;   ///< the largest sample
};

/// Summarizes \a samples, which must not be empty.
[[nodiscard]] inline Summary
summarize(std::vector<double> samples)
{
    std::ranges::sort(samples);
    std::size_t const n = samples.size();
    double const median = n % 2 == 1 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
    auto const p99Index = static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(n))) - 1;
    return Summary{median, samples[p99Index], samples.back()};
}

/// Divides every statistic of \a summary by \a divisor: a time per step becomes a time per
/// particle, say.
[[nodiscard]] inline Summary
operator/(Summary const & summary, double divisor)
{
    return Summary{summary.median / divisor, summary.p99 / divisor, summary.worst / divisor};
}

/// Runs steps \a begin to \a end of a run through \a filter, each as \a step(filter, k). Out of
/// line, so that Callgrind counts it by name, once per filter type and case:
///
///     valgrind --tool=callgrind --toggle-collect='*runSteps*' <runner> --count
template <typename Filter, typename Step>
EMRYS_BENCHMARK_NOINLINE void
runSteps(Filter & filter, Step const & step, std::size_t begin, std::size_t end)
{
    for (std::size_t k = begin; k < end; ++k)
    {
        step(filter, k);
    }
}

/// A filter replayed over a simulated run of \a length steps, timed a batch of steps at a time.
/// \a Make builds a fresh filter at its initial estimate, and \a Step runs step k of the run on
/// one.
template <typename Make, typename Step>
class Replay
{
public:

    using Filter = std::invoke_result_t<Make const &>;

    Replay(Make make, Step step, std::size_t length)
        : m_make{std::move(make)}
        , m_step{std::move(step)}
        , m_length{length}
    {
        m_filter.emplace(m_make());
    }

    /// Runs the next \a count steps and returns the time per step, in nanoseconds. Starts over from
    /// the initial estimate first, untimed, if the run has fewer than \a count steps left.
    [[nodiscard]] double
    timeSteps(std::size_t count)
    {
        if (m_next + count > m_length)
        {
            m_filter.emplace(m_make());
            m_next = 0;
        }
        auto const start = std::chrono::steady_clock::now();
        runSteps(*m_filter, m_step, m_next, m_next + count);
        auto const stop = std::chrono::steady_clock::now();
        m_next += count;
        return std::chrono::duration<double, std::nano>(stop - start).count() / static_cast<double>(count);
    }

    /// The steps in the run.
    [[nodiscard]] std::size_t
    length() const
    {
        return m_length;
    }

    /// The filter, so that its result can be used and its steps not optimized away.
    [[nodiscard]] Filter const &
    filter() const
    {
        return *m_filter;
    }

private:

    Make m_make;                     ///< builds the filter at its initial estimate
    Step m_step;                     ///< runs one step of the run
    std::size_t m_length;            ///< the steps in the run
    std::optional<Filter> m_filter;  ///< the filter, restarted at the end of the run
    std::size_t m_next = 0;          ///< the next step of the run
};

/// The timings of the two implementations of one case, in nanoseconds per step.
struct PairTiming
{
    Summary emrys;  ///< the EmrysFilter implementation's
    Summary eigen;  ///< the plain-Eigen implementation's
};

/// Times \a emrys against \a eigen, two replays of the same run: batches of steps, alternating
/// between the two filters so that drift in the machine's clock speed falls on both alike, with
/// `options.samples` batches of each. A batch is long enough that the clock's resolution and the
/// cost of reading it are under 1% of it.
template <typename EmrysReplay, typename EigenReplay>
[[nodiscard]] PairTiming
timePair(Options const & options, EmrysReplay & emrys, EigenReplay & eigen)
{
    /// The shortest batch worth timing, in nanoseconds.
    constexpr double TargetBatchNanoseconds = 20'000.0;

    // Double the batch until the faster filter's quickest of a few batches is long enough, so that
    // every batch of either is, even where the clock ticks only every microsecond.
    std::size_t batch = 1;
    while (batch < emrys.length() / 2)
    {
        double quickest = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 5; ++i)
        {
            quickest = std::min({quickest, emrys.timeSteps(batch), eigen.timeSteps(batch)});
        }
        if (quickest * static_cast<double>(batch) >= TargetBatchNanoseconds)
        {
            break;
        }
        batch *= 2;
    }
    for (int i = 0; i < options.samples / 10; ++i)
    {
        consume(emrys.timeSteps(batch) + eigen.timeSteps(batch));
    }

    auto emrysSamples = std::vector<double>{};
    auto eigenSamples = std::vector<double>{};
    emrysSamples.reserve(static_cast<std::size_t>(options.samples));
    eigenSamples.reserve(static_cast<std::size_t>(options.samples));
    for (int s = 0; s < options.samples; ++s)
    {
        // Alternate which goes first, so that neither always follows the other's cache footprint.
        if (s % 2 == 0)
        {
            emrysSamples.push_back(emrys.timeSteps(batch));
            eigenSamples.push_back(eigen.timeSteps(batch));
        }
        else
        {
            eigenSamples.push_back(eigen.timeSteps(batch));
            emrysSamples.push_back(emrys.timeSteps(batch));
        }
    }
    return PairTiming{summarize(emrysSamples), summarize(eigenSamples)};
}

/// Prints the column headings of a table whose first column, \a labelWidth wide, is headed
/// \a labelHeading: the three statistics of each implementation and the ratio of the medians and
/// of the 99th percentiles.
inline void
printColumns(char const * labelHeading, int labelWidth)
{
    std::printf("%*s%-26s    %-26s    %s\n", labelWidth + 2, "", "EmrysFilter", "Eigen", "ratio");
    std::printf(
        "%-*s  %8s %8s %8s    %8s %8s %8s    %6s %6s\n",
        labelWidth,
        labelHeading,
        "median",
        "p99",
        "worst",
        "median",
        "p99",
        "worst",
        "median",
        "p99"
    );
}

/// Prints one row of the table: the case's \a label, then \a timing in the columns of
/// printColumns(), with \a decimals places after the point. Flushes, so that a long run shows its
/// rows as they come.
inline void
printRow(char const * label, int labelWidth, PairTiming const & timing, int decimals = 0)
{
    Summary const & e = timing.emrys;
    Summary const & g = timing.eigen;
    std::printf(
        "%-*s  %8.*f %8.*f %8.*f    %8.*f %8.*f %8.*f    %6.2f %6.2f\n",
        labelWidth,
        label,
        decimals,
        e.median,
        decimals,
        e.p99,
        decimals,
        e.worst,
        decimals,
        g.median,
        decimals,
        g.p99,
        decimals,
        g.worst,
        e.median / g.median,
        e.p99 / g.p99
    );
    std::fflush(stdout);
}

}  // namespace Benchmark

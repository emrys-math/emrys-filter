// A tag at an unknown, fixed place on a 20 m square warehouse floor, such as an asset tag on a
// shelf, located by ultra-wideband ranges to six anchors on the walls. Shelving blocks the direct
// path from each anchor with probability 0.4. A clear path gives the true range plus Gaussian
// noise; a blocked one makes the signal arrive by a longer path, adding a delay drawn from an
// exponential distribution, so the range is never much short of the true one but can be well past
// it. Each range's error is then a mixture: a Gaussian, and an exponentially modified Gaussian, the
// Gaussian convolved with the exponential. The program simulates the tag and the ranges, locates
// the tag with a grid filter, and checks that the posterior is calibrated: the cells holding the
// top 90% of the posterior should contain the tag's cell in about 90% of the runs. It returns
// nonzero if the count is improbable for 90%.
//
// A Kalman filter can't take this measurement. Its update needs the error as a Gaussian, or a
// mixture of Gaussians, added to a function of the state, and the exponentially modified Gaussian
// is neither, so the measurement below supplies only its likelihood, which KalmanFilter refuses at
// compile time. A grid filter needs only the likelihood at each cell's center. Matching the error's
// mean and variance with one Gaussian would give a Kalman filter something to take, but not the
// error's shape: the true likelihood falls off sharply where the range is short of the cell's and
// slowly where it is long, so a clear range pins the tag to a narrow ring and a blocked one only
// bounds it from outside, while the Gaussian widens every ring and moves it toward the anchor. The
// program runs a second grid filter on that Gaussian and prints the cost: its 90% set covers the
// tag about as often, but over a larger area.

#include <EmrysFilter/EmrysFilter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <print>
#include <random>
#include <utility>
#include <vector>

// The state is the tag's position, in meters; the grid covers all of it.
struct Position : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position>;

// A range in meters, the space of the measurement.
using Range = Emrys::Linear<1>;

double const noise = 0.3;    // the standard deviation sigma of a range's Gaussian noise, in m
double const delay = 1.0;    // the mean 1 / lambda of a blocked path's extra length, in m
double const blocked = 0.4;  // the probability that shelving blocks an anchor's path

// The log-density at e of a Gaussian of standard deviation sigma.
double
gaussianLogDensity(double e)
{
    return -0.5 * e * e / (noise * noise) - std::log(noise * std::sqrt(2 * std::numbers::pi));
}

// The log-density at e of the Gaussian convolved with an exponential of mean 1 / lambda:
// log(lambda) + lambda^2 sigma^2 / 2 - lambda e + log(erfc((lambda sigma^2 - e) / (sqrt(2) sigma)) / 2).
// Where e is far below zero the erfc underflows to zero and this to minus infinity, but there the
// Gaussian term of the mixture is larger and exact.
double
exponentiallyModifiedLogDensity(double e)
{
    double const lambda = 1 / delay;
    double const u = (lambda * noise * noise - e) / (std::numbers::sqrt2 * noise);
    return std::log(lambda) + 0.5 * lambda * lambda * noise * noise - lambda * e + std::log(0.5 * std::erfc(u));
}

// The log-density at e of a range's error: the clear path's Gaussian with probability 1 - p, the
// blocked path's exponentially modified Gaussian with probability p, summed from their logs.
double
rangeErrorLogDensity(double e)
{
    double const clear = std::log(1 - blocked) + gaussianLogDensity(e);
    double const obstructed = std::log(blocked) + exponentiallyModifiedLogDensity(e);
    double const larger = std::max(clear, obstructed);
    return larger + std::log(std::exp(clear - larger) + std::exp(obstructed - larger));
}

// The range in m from the anchor at `anchor` to the position in the state x, read through in.
[[nodiscard]] double
distance(auto const & in, auto const & x, Emrys::ValueVector<Position> const & anchor)
{
    return Emrys::minus<Position>(Emrys::block<Position>(in, x), anchor).norm();
}

// The error of the observed range z against the range from the anchor to the position in x.
[[nodiscard]] double
rangeError(auto const & in, auto const & x, auto const & z, Emrys::ValueVector<Position> const & anchor)
{
    return Emrys::minus<Range>(z, Emrys::ValueVector<Range>{distance(in, x, anchor)})(0);
}

// The range from the anchor at `anchor`, known only by its likelihood: the model has the block it
// reads, logLikelihood() of the observation at a state, and sample(), a draw of the observation
// at a state, which simulate() calls. It has no function or noise, so no Kalman filter takes it.
[[nodiscard]] auto
range(Emrys::ValueVector<Position> const & anchor)
{
    auto const logLikelihood = [anchor](auto const & in, auto const & x, auto const & z) {
        return rangeErrorLogDensity(rangeError(in, x, z, anchor));
    };

    // The true range, plus Gaussian noise, plus an exponential delay where shelving blocks the path.
    auto const sample = [anchor](auto const & in, auto const & x, auto & rng) {
        double const gaussian = std::normal_distribution<double>{0.0, noise}(rng);
        double const extra =
            std::bernoulli_distribution{blocked}(rng) ? std::exponential_distribution<double>{1 / delay}(rng) : 0.0;
        return Emrys::plus<Range>(
            Emrys::ValueVector<Range>{distance(in, x, anchor)},
            Emrys::DifferenceVector<Range>{gaussian + extra}
        );
    };
    return Emrys::measurementModel(
        Emrys::reads(Emrys::path<Position>),
        Emrys::logLikelihood(logLikelihood),
        Emrys::sample(sample)
    );
}

// The same range with its error taken to be the Gaussian with the mixture's mean and variance.
[[nodiscard]] auto
matchedRange(Emrys::ValueVector<Position> const & anchor)
{
    auto const logLikelihood = [anchor](auto const & in, auto const & x, auto const & z) {
        // The exponential's mean is 1 / lambda and its second moment 2 / lambda^2.
        double const mean = blocked * delay;
        double const variance = noise * noise + blocked * (2 - blocked) * delay * delay;
        double const d = rangeError(in, x, z, anchor) - mean;
        return -0.5 * (d * d / variance + std::log(2 * std::numbers::pi * variance));
    };
    return Emrys::measurementModel(Emrys::reads(Emrys::path<Position>), Emrys::logLikelihood(logLikelihood));
}

// The cells of the grid that hold the top 90% of the posterior: their number, and whether the
// tag's cell is among them.
template <typename Filter>
std::pair<Eigen::Index, bool>
topNinetyPercent(Filter const & filter, Emrys::ValueVector<State> const & truth, double spacing)
{
    auto const & state = filter.state();
    auto probabilities = std::vector<double>(std::size_t(state.size()));
    double truthProbability = 0.0;  // zero if pruned
    for (Eigen::Index i = 0; i < state.size(); ++i)
    {
        probabilities[std::size_t(i)] = std::exp(state.logWeight(i));
        if ((Emrys::minus<State>(state.center(i), truth).array().abs() < spacing / 2).all())
        {
            truthProbability = probabilities[std::size_t(i)];
        }
    }
    std::ranges::sort(probabilities, std::greater{});
    double total = 0.0;
    Eigen::Index count = 0;
    while (total < 0.9)
    {
        total += probabilities[std::size_t(count++)];
    }
    return {count, truthProbability >= probabilities[std::size_t(count - 1)]};
}

int
main()
{
    int const runs = 200;
    double const spacing = 0.1;

    // At the corners of the floor and the middles of two opposite walls.
    auto const anchors = std::array{
        Emrys::ValueVector<Position>{0, 0},
        Emrys::ValueVector<Position>{20, 0},
        Emrys::ValueVector<Position>{20, 20},
        Emrys::ValueVector<Position>{0, 20},
        Emrys::ValueVector<Position>{10, 0},
        Emrys::ValueVector<Position>{10, 20}
    };

    // 200 by 200 cells of 0.1 m, their centers from (0.05, 0.05) to (19.95, 19.95). The filter
    // starts with every cell equally probable, as the tag is drawn uniformly over the floor.
    auto const grid = Emrys::regularGrid<State, Position>(
        Emrys::ValueVector<Position>{spacing / 2, spacing / 2},
        Emrys::DifferenceVector<Position>{spacing, spacing},
        {200, 200}
    );

    auto rng = std::mt19937_64{1};
    auto uniform = std::uniform_real_distribution<double>{0.0, 20.0};
    auto const prune = Emrys::upkeep::pruneBelow(1e-12);
    int covered = 0;
    int coveredMatched = 0;
    double area = 0.0;
    double areaMatched = 0.0;
    for (int run = 0; run < runs; ++run)
    {
        auto const truth = Emrys::ValueVector<State>{uniform(rng), uniform(rng)};
        auto filter = Emrys::PointMassFilter{grid};
        auto matched = Emrys::PointMassFilter{grid};
        for (auto const & anchor : anchors)
        {
            // Each anchor gives one range, drawn by the model the filter weights by, and whether its
            // path is blocked is drawn for each, so the errors of the ranges are independent.
            auto const ranging = range(anchor);
            auto const observed = Emrys::simulate<State>(truth, ranging, rng);
            auto const exact = filter.update(Emrys::observe(ranging, observed));
            auto const approximate = matched.update(Emrys::observe(matchedRange(anchor), observed));
            if (!exact || !exact->accepted || !approximate || !approximate->accepted)
            {
                // No cell could explain the range, which a tag on the floor rules out.
                std::println("run {}: the update was rejected", run);
                return 1;
            }

            // Cells less probable than 1e-12 are dropped, so that later updates evaluate fewer.
            if (!prune(filter.state()) || !prune(matched.state()))
            {
                // The probabilities could not be normalized, which an accepted update rules out.
                std::println("run {}: pruning failed", run);
                return 1;
            }
        }
        auto const [cells, inside] = topNinetyPercent(filter, truth, spacing);
        auto const [cellsMatched, insideMatched] = topNinetyPercent(matched, truth, spacing);
        covered += inside ? 1 : 0;
        coveredMatched += insideMatched ? 1 : 0;
        area += double(cells) * spacing * spacing;
        areaMatched += double(cellsMatched) * spacing * spacing;
    }

    std::println("the 90% set held the tag in {} of {} runs, over {:.2f} m^2 on average", covered, runs, area / runs);
    std::println(
        "with the moment-matched Gaussian: in {} of {} runs, over {:.2f} m^2 on average",
        coveredMatched,
        runs,
        areaMatched / runs
    );

    // The count is binomial with probability 0.9 if the posterior is calibrated; 3.29 standard
    // deviations either side is its two-sided 99.9% interval.
    double const bound = 3.29 * std::sqrt(runs * 0.9 * 0.1);
    bool const calibrated = std::abs(covered - 0.9 * runs) <= bound;
    std::println("{}", calibrated ? "calibrated" : "NOT calibrated");
    return calibrated ? 0 : 1;
}

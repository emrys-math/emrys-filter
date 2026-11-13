// A target moving in the plane at nearly constant velocity, tracked by a Gaussian mixture filter from
// position fixes, one in five of them an outlier. The fix noise is a mixture of a narrow inlier
// Gaussian and a wide outlier one. Keep updates every component with each of the two, so the
// number of components doubles with every fix, and the program merges them back down by the cost of
// Runnalls (2007). The library has no merging action: which pairs to merge, and how to search for
// them, depends on the problem and the hardware, so the reduction below is this program's own, a
// starting point to copy and adapt. The program checks the combined estimate against its covariance
// and against the same filter with Collapse, which merges each component's branches at once, and
// returns nonzero if either check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <functional>
#include <print>
#include <expected>
#include <limits>
#include <random>

// The state is a position and a velocity in the plane, as in constant_velocity.cpp.
struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;
using Mixture = Emrys::GaussianMixtureParams<State>;

// The cost of merging components first and second into merged, the Gaussian with their weight and
// their combined mean and covariance: w_1 KL(g_1 || g) + w_2 KL(g_2 || g), with each weight the
// component's share of the mixture's, which bounds the divergence the merge adds (Runnalls 2007).
double
mergeCost(
    Mixture::Entry const & first,
    Mixture::Entry const & second,
    Mixture::ComponentType const & merged,
    double logTotal
)
{
    // Every covariance here is positive definite, the components' by the filter's updates and the
    // merge's as a weighted sum of positive definite matrices, so no toDensity() can fail.
    auto const g = merged.toDensity().value();
    return std::exp(first.logWeight - logTotal) * Emrys::kullbackLeibler(first.component.toDensity().value(), g)
         + std::exp(second.logWeight - logTotal) * Emrys::kullbackLeibler(second.component.toDensity().value(), g);
}

// Merges the mixture's components, the cheapest pair first, until at most maxSize remain. Each merge
// scans every pair again, so reducing n components evaluates O(n^3) costs, which is fine for a few
// dozen components. For more, keep the costs between merges and refresh only those of the merged
// component, or keep each component's cheapest partner. Returns whether it merged any, or the
// error that stopped it, as the library's upkeep actions do, so a trigger can run it.
std::expected<bool, Emrys::WeightError>
mergeDownTo(Mixture & mixture, Eigen::Index maxSize)
{
    bool merged = false;
    while (mixture.size() > maxSize)
    {
        auto const logTotal = mixture.logTotalWeight();
        if (!logTotal)
        {
            return std::unexpected{logTotal.error()};
        }
        Eigen::Index keep = 0;
        Eigen::Index absorb = 0;
        double cheapest = std::numeric_limits<double>::infinity();
        for (Eigen::Index first = 0; first < mixture.size(); ++first)
        {
            for (Eigen::Index second = first + 1; second < mixture.size(); ++second)
            {
                double const cost =
                    mergeCost(mixture[first], mixture[second], mixture.merged(first, second), *logTotal);
                if (cost < cheapest)
                {
                    cheapest = cost;
                    keep = first;
                    absorb = second;
                }
            }
        }
        mixture.merge(keep, absorb);
        merged = true;
    }
    return merged;
}

int
main()
{
    double const dt = 1.0;  // seconds between fixes
    int const runs = 100;
    int const steps = 50;

    // The target moves at nearly constant velocity, driven by a white-noise acceleration of spectral
    // density 0.1 m^2/s^3, as in constant_velocity.cpp.
    double const accelerationDensity = 0.1;
    auto const motion = Emrys::fixedStep(Emrys::ConstantVelocity<Position, Velocity>{accelerationDensity}, dt);

    // A fix measures the position with noise of 1 m^2 variance on each axis, but one fix in five is
    // an outlier, with a standard deviation of 10 m. The noise is that mixture, which simulate()
    // draws from and the filter assumes. An outlier that close is hard to tell from a fix at first,
    // so keeping both branches until later fixes settle it pays.
    double const inlierVariance = 1.0;
    double const outlierVariance = 100.0;
    Emrys::DifferenceMatrix<Position> const identity = Emrys::DifferenceMatrix<Position>::Identity();
    auto noise = Emrys::GaussianMixtureParams<Position>{};
    noise.add(std::log(0.8), Emrys::ValueVector<Position>::Zero(), inlierVariance * identity);
    noise.add(std::log(0.2), Emrys::ValueVector<Position>::Zero(), outlierVariance * identity);
    auto const gps = Emrys::measurementModel(
        Emrys::DifferenceMatrix<Position>::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(noise)
    );

    // The filter starts from one component: near the origin, moving slowly.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0, 1.0, 0.5},
        Emrys::DifferenceVector<State>{10.0, 10.0, 1.0, 1.0}.asDiagonal()
    };
    auto initial = Mixture{};
    initial.add(0.0, prior);

    // The reduction runs under a trigger, as the library's actions do: whenever an update leaves
    // more than 8 components, they are merged down to 4.
    auto const reduce = Emrys::upkeep::whenSize(std::greater{}, 8, [](Mixture & mixture) {
        return mergeDownTo(mixture, 4);
    });

    // The estimation error is tested over independent runs, one final estimate from each (the Monte
    // Carlo test of constant_velocity.cpp); the position errors are counted at every step.
    auto nees = Emrys::NeesMonitor{runs};
    auto rng = std::mt19937_64{11};
    double keptSquaredError = 0;
    double collapsedSquaredError = 0;
    for (int run = 0; run < runs; ++run)
    {
        auto kept = Emrys::GaussianMixtureFilter{motion, initial};
        auto collapsed = Emrys::GaussianMixtureFilter{motion, initial};

        // The prior's covariance is positive definite by construction, so toDensity() can't fail.
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng);
            Emrys::ValueVector<Position> const z = Emrys::simulate<State>(truth, gps, rng);

            kept.predict();
            collapsed.predict();
            auto const fix = Emrys::observe(gps, z);
            if (!kept.update(fix, Emrys::reductions::keep()) || !collapsed.update(fix, Emrys::reductions::collapse()))
            {
                // An innovation covariance was not positive definite, and the state is unchanged.
                std::println("run {}, step {}: the update failed", run, k);
                return 1;
            }
            if (!reduce(kept.state()))
            {
                std::println("run {}, step {}: the weights are not a distribution", run, k);
                return 1;
            }

            // The combined estimate is the mixture's collapse, the Gaussian with its mean and
            // covariance. The weights were checked by the reduction, and the update leaves them
            // finite, so toDensity() can't fail here.
            Emrys::DifferenceVector<State> const keptError =
                Emrys::minus<State>(kept.state().toDensity().value().collapsed().mean(), truth);
            Emrys::DifferenceVector<State> const collapsedError =
                Emrys::minus<State>(collapsed.state().toDensity().value().collapsed().mean(), truth);
            keptSquaredError += Emrys::diffBlock<State, Position>(keptError).squaredNorm();
            collapsedSquaredError += Emrys::diffBlock<State, Position>(collapsedError).squaredNorm();
        }
        nees.add(truth, kept.state().toDensity().value().collapsed());
    }

    double const keptRms = std::sqrt(keptSquaredError / (runs * steps));
    double const collapsedRms = std::sqrt(collapsedSquaredError / (runs * steps));
    std::println("position RMS error {:.3f} m, against {:.3f} m with Collapse", keptRms, collapsedRms);

    // The average is the sum divided by its degrees of freedom, so 1 is ideal.
    double const average = nees.average().value();
    std::println("average NEES {:.2f} per degree of freedom", average);

    // The combined estimate is the moments of a mixture, not a Gaussian, so its NEES is not
    // chi-squared, as in imm.cpp: the check asks that its average be within a factor of 1.5 of 1,
    // and that keeping the branches and merging them by cost track the position better than
    // collapsing them at every fix.
    bool const consistent = average > 1 / 1.5 && average < 1.5;
    bool const better = keptRms < collapsedRms;
    std::println("{}, {}", consistent ? "consistent" : "NOT consistent", better ? "better" : "NOT better");
    return consistent && better ? 0 : 1;
}

// An aircraft located by the terrain below it, terrain-referenced navigation, with a particle filter
// whose measurement looks up a table for every particle in one call. The aircraft flies a known
// course, its position drifting from it, and each second its barometric altitude less its radar
// altimeter's reading gives the height of the terrain below it. A map gives the terrain's height
// on a grid of 1 m cells, and the measurement interpolates it. A particle filter evaluates the
// measurement at every particle, and a table lookup costs more per call than per point when the
// table lives behind a lock, in a file, or on a device, so the measurement is batched: it takes every
// particle at once, as the columns of a matrix, and the filter calls it once per update.
//
// The program simulates the aircraft and the measurements from the same models the filter assumes,
// runs the filter many times, and checks that a Gaussian fit to the particles at the end of each run
// is consistent with the truth: its normalized estimation error squared (NEES), over the runs, must
// pass the chi-squared test. It returns nonzero if the check fails, or if the filter called the
// measurement other than once per update.

#include <EmrysFilter/EmrysFilter.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <print>
#include <random>

// The state is the aircraft's position in the plane, in meters east and north.
struct Position : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position>;

// The height of the terrain below the aircraft, the space of the measurement.
struct Elevation : Emrys::Linear<1>
{};

// The terrain's height on a grid of 1 m cells, cell (i, j) at i m east and j m north, and its
// height between the cells by bilinear interpolation, at the nearest edge outside the grid.
struct Terrain
{
    Eigen::MatrixXd heights;

    [[nodiscard]] double
    height(Emrys::ValueVector<Position> const & position) const
    {
        double const x = std::clamp(position(0), 0.0, double(heights.rows() - 2));
        double const y = std::clamp(position(1), 0.0, double(heights.cols() - 2));
        auto const i = Eigen::Index(x);
        auto const j = Eigen::Index(y);
        double const u = x - double(i);
        double const v = y - double(j);
        return (1 - u) * (1 - v) * heights(i, j) + u * (1 - v) * heights(i + 1, j) + (1 - u) * v * heights(i, j + 1)
             + u * v * heights(i + 1, j + 1);
    }
};

int
main()
{
    double const dt = 1.0;  // seconds between measurements
    int const runs = 20;
    int const steps = 40;
    Eigen::Index const count = 5000;  // particles

    // Rolling terrain over 200 m by 200 m, its slopes up to about 3 m per m.
    auto terrain = Terrain{Eigen::MatrixXd(200, 200)};
    for (Eigen::Index i = 0; i < terrain.heights.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < terrain.heights.cols(); ++j)
        {
            double const x = double(i);
            double const y = double(j);
            terrain.heights(i, j) =
                20.0 * std::sin(x / 9.0) * std::cos(y / 13.0) + 10.0 * std::sin((x + 2.0 * y) / 17.0);
        }
    }

    // The aircraft flies at 3 m/s east and 1 m/s north, its position drifting from the course by
    // 0.5 m^2 a second on each axis.
    auto const velocity = Emrys::DifferenceVector<Position>{3.0, 1.0};
    auto const motion = Emrys::processModel(
        [velocity]<typename T>(auto const & in, auto const & out, auto const & x, double step) {
            auto y = Emrys::zeros<T>(out);
            Emrys::DifferenceVector<Position, T> const travel = T(step) * velocity.cast<T>();
            Emrys::block<Position>(out, y) = Emrys::plus<Position>(Emrys::block<Position>(in, x), travel);
            return y;
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::writes(Emrys::path<Position>),
        Emrys::continuousNoise(Emrys::on<Position>(0.5))
    );

    // The terrain's height below each particle, every particle a column of X, with noise of 1 m^2
    // variance. A batched function writes into the output Y the library gives it, so the model names
    // its output space. The counter shows how often the filter calls it.
    int lookups = 0;
    double const elevationVariance = 1.0;
    auto const elevation = Emrys::measurementModel(
        Emrys::batched([&terrain, &lookups](auto const & in, auto const &, auto const & X, auto & Y) {
            ++lookups;
            auto const position = Emrys::block<Position>(in, X);
            for (Eigen::Index j = 0; j < X.cols(); ++j)
            {
                Y.values()(0, j) = terrain.height(position.col(j));
            }
        }),
        Emrys::reads(Emrys::path<Position>),
        Emrys::outputSpace(Elevation{}),
        Emrys::additiveNoise(elevationVariance * Emrys::DifferenceMatrix<Elevation>::Identity())
    );

    // The prior puts the aircraft within about 5 m of (40, 60) on each axis. Its covariance is
    // positive definite by construction, so toDensity() can't fail here.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{40.0, 60.0},
        25.0 * Emrys::DifferenceMatrix<State>::Identity()
    };
    auto const priorDensity = prior.toDensity().value();

    // After each step's update, the particles are resampled when their effective sample size falls
    // below half their number.
    auto rng = std::mt19937_64{1};
    auto const resampling = Emrys::upkeep::whenEss(std::less{}, 0.5, Emrys::upkeep::resample(rng));

    // The test of a filter is a simulation: draw a true path from the models the filter assumes,
    // filter its measurements, and compare the final estimates with the truth over independent runs.
    auto nees = Emrys::NeesMonitor{runs};
    int updates = 0;
    for (int run = 0; run < runs; ++run)
    {
        auto filter =
            Emrys::ParticleFilter{motion, Emrys::ParticleSet<State>{Emrys::drawSamples(priorDensity, count, rng)}};
        Emrys::ValueVector<State> truth = Emrys::drawSamples(priorDensity, 1, rng);
        for (int k = 1; k <= steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng, dt);
            filter.predict(rng, dt);

            // The simulated measurement calls the function at the one true position, a batch of one
            // column, which the counter leaves out.
            int const before = lookups;
            Emrys::ValueVector<Elevation> const z = Emrys::simulate<State>(truth, elevation, rng);
            lookups = before;

            // An update fails if the weights are invalid, and is rejected, leaving the weights alone,
            // if no particle can explain the measurement.
            auto const updated = filter.update(Emrys::observe(elevation, z));
            ++updates;
            if (!updated || !updated->accepted)
            {
                std::println("run {}, step {}: the update failed or was rejected", run, k);
                return 1;
            }
            if (!resampling(filter.state()))
            {
                std::println("run {}, step {}: the particles' weights are invalid", run, k);
                return 1;
            }
        }

        // An estimate is read from the particles: their weighted mean and covariance, which fail
        // only if the weights are invalid.
        auto const moments = filter.state().moments();
        if (!moments)
        {
            std::println("run {}: the particles' weights are invalid", run);
            return 1;
        }
        if (run == 0)
        {
            std::println(
                "aircraft at ({:.1f}, {:.1f}) m, estimate ({:.1f}, {:.1f}) m",
                truth(0),
                truth(1),
                moments->mean()(0),
                moments->mean()(1)
            );
        }
        nees.add(truth, Emrys::GaussianParams<State>{*moments});
    }

    std::println("{} updates of {} particles looked the terrain up in {} calls", updates, count, lookups);

    // The average is the NEES sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES of the final estimates of {} runs {:.2f} per degree of freedom",
        runs,
        nees.average().value()
    );

    bool const passed = lookups == updates && nees.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}

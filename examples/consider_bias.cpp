// The heading of a vehicle, integrated from a gyroscope whose bias is known only from its
// calibration, and corrected by a compass fix once a second. The bias enters through the prediction:
// the heading turns by the reading less the bias. Three filters run on the same readings. One
// estimates the bias, one considers it (the Schmidt-Kalman update: it keeps the bias's mean and
// variance as calibrated and carries the heading's covariance with it), and one leaves it out. The
// program checks each filter's estimates against the true heading by the normalized estimation error
// squared (NEES) over independent runs. It returns nonzero if the filter that considers the bias is
// not consistent.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <numbers>
#include <print>
#include <random>

// The heading, in radians, an angle in [-pi, pi) that plus() and minus() wrap.
struct Heading : Emrys::RadiansPi<1>
{};

// The gyroscope's reading, in radians per second, an input of each prediction: the rate of change
// of the heading.
struct Gyro : Emrys::Linear<1>
{};

// The gyroscope's bias, in radians per second, constant over a run: an error in the reading, so a
// difference of readings, declared in their difference space.
struct Bias : Emrys::DifferenceSpace<Gyro>
{};

using State = Emrys::Composite<Heading, Bias>;

// The heading turned by the reading less the bias over the step, with the reading's white noise
// integrated into the heading's variance. The function computes in T, the scalar of x, which the
// filter supplies: dual numbers when it differentiates the function. less() takes the bias from the
// reading; the rate times the step is a difference of headings, and plus() turns the heading by it
// and wraps the result.
[[nodiscard]] auto
gyroIntegration(double noiseDensity)  // spectral density of the gyroscope's white noise, rad^2/s
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto const rate = Emrys::less<Gyro>(Emrys::block<Gyro>(in, x), Emrys::block<Bias>(in, x));
            Emrys::DifferenceVector<Heading, T> const turn = rate * dt;
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Heading>(out, y) = Emrys::plus<Heading>(Emrys::block<Heading>(in, x), turn);
            return y;
        },
        Emrys::reads(Emrys::path<Heading>, Emrys::path<Bias>, Emrys::path<Gyro>),
        Emrys::writes(Emrys::path<Heading>),
        Emrys::continuousNoise(Emrys::on<Heading>(noiseDensity))
    );
}

// The same integration for a filter that has no bias in its state, and takes the reading as true.
[[nodiscard]] auto
unbiasedIntegration(double noiseDensity)
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            Emrys::DifferenceVector<Heading, T> const turn = Emrys::block<Gyro>(in, x) * dt;
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Heading>(out, y) = Emrys::plus<Heading>(Emrys::block<Heading>(in, x), turn);
            return y;
        },
        Emrys::reads(Emrys::path<Heading>, Emrys::path<Gyro>),
        Emrys::writes(Emrys::path<Heading>),
        Emrys::continuousNoise(Emrys::on<Heading>(noiseDensity))
    );
}

// A compass fix of the heading, with a standard deviation of 0.05 rad (about 3 degrees). The fix
// is an angle, so writes() names its space, the heading's, and the residual wraps as the heading
// does: a fix just below pi of a heading just above -pi is a small correction, not one of 2 pi.
[[nodiscard]] auto
compassModel()
{
    double const fixVariance = 0.0025;
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            return Emrys::ValueVector<Heading, T>{Emrys::block<Heading>(in, x)};
        },
        Emrys::reads(Emrys::path<Heading>),
        Emrys::writes(Emrys::path<Heading>),
        Emrys::additiveNoise(Emrys::DifferenceMatrix<Heading>{{fixVariance}})
    );
}

int
main()
{
    double const dt = 0.01;    // seconds between gyroscope readings
    int const fixEvery = 100;  // a compass fix once a second
    int const steps = 3000;    // 30 seconds a run
    int const runs = 200;

    // The gyroscope's white noise has a spectral density of 1e-6 rad^2/s.
    double const gyroDensity = 1e-6;
    auto const motion = Emrys::compose(gyroIntegration(gyroDensity), Emrys::Constant<Bias>{});
    auto const unbiased = unbiasedIntegration(gyroDensity);
    auto const compass = compassModel();

    // The calibration gives the bias as zero with a standard deviation of 0.005 rad/s, about
    // 0.3 degrees per second. The heading starts known to 0.05 rad.
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0},
        Emrys::DifferenceVector<State>{0.0025, 2.5e-5}.asDiagonal()
    };
    auto const headingPrior =
        Emrys::GaussianParams<Heading>{Emrys::ValueVector<Heading>{0.0}, Emrys::DifferenceMatrix<Heading>{{0.0025}}};

    // The NEES test takes the final heading of independent runs, one per run.
    auto rng = std::mt19937_64{5};
    auto estimated = Emrys::NeesMonitor{runs};
    auto considered = Emrys::NeesMonitor{runs};
    auto dropped = Emrys::NeesMonitor{runs};
    auto const method = Emrys::methods::linearized();
    for (int run = 0; run < runs; ++run)
    {
        // The true heading and bias are drawn from the prior, whose covariance is positive definite
        // by construction, so toDensity() can't fail here.
        Emrys::ValueVector<State> truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        auto estimating = Emrys::KalmanFilter{motion, prior};
        auto considering = Emrys::KalmanFilter{motion, prior};
        auto dropping = Emrys::KalmanFilter{unbiased, headingPrior};
        for (int k = 1; k <= steps; ++k)
        {
            auto const gyro = Emrys::ValueVector<Gyro>{0.2 * std::sin(0.4 * k * dt)};
            truth = Emrys::simulate<State>(truth, motion, rng, dt, Emrys::sure<Gyro>(gyro));
            estimating.predict(dt, Emrys::sure<Gyro>(gyro));
            considering.predict(dt, Emrys::sure<Gyro>(gyro));
            dropping.predict(dt, Emrys::sure<Gyro>(gyro));
            if (k % fixEvery != 0)
            {
                continue;
            }
            Emrys::ValueVector<Heading> const fix = Emrys::simulate<State>(truth, compass, rng);

            // The considering filter's fix names the bias in consider(): the update leaves the bias's
            // mean and variance as they were, and moves the heading as the full update would.
            auto const a = estimating.update(Emrys::observe(method, compass, fix));
            auto const b = considering.update(Emrys::observe(method, compass, fix, Emrys::consider(Emrys::path<Bias>)));
            auto const c = dropping.update(Emrys::observe(method, compass, fix));
            if (!a || !b || !c)
            {
                // An innovation covariance was not positive definite, and that filter's state is unchanged.
                std::println("run {}, step {}: an update failed", run, k);
                return 1;
            }
        }

        // Each filter's heading against the true heading. add<Heading>() reads the heading's block
        // of both the truth and the estimate; the filter that leaves the bias out has only the heading.
        estimated.add<Heading>(truth, estimating.state());
        considered.add<Heading>(truth, considering.state());
        dropped.add(Emrys::block<State, Heading>(truth), dropping.state());
    }

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal; above the bounds a
    // filter is overconfident, and below them underconfident.
    auto const report = [](char const * name, auto const & monitor) {
        auto const verdict = monitor.verdict(0.999).value();
        std::println(
            "{:<20} average NEES {:6.2f}  {}",
            name,
            monitor.average().value(),
            verdict == Emrys::Consistency::Consistent      ? "consistent"
            : verdict == Emrys::Consistency::Overconfident ? "overconfident"
                                                           : "underconfident"
        );
        return verdict == Emrys::Consistency::Consistent;
    };
    report("estimating the bias", estimated);
    bool const consistent = report("considering it", considered);
    report("leaving it out", dropped);
    return consistent ? 0 : 1;
}

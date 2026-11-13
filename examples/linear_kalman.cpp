// The textbook Kalman filter: a target moving in the plane at nearly constant velocity, its state a
// plain vector of four coordinates, tracked from noisy position fixes. The models are given by
// their matrices, F and H, and their zero-mean noises by their covariances, Q and R, written out as
// the textbook writes them; they name no block of the state, so they read and write the whole
// state, and the filter takes the state's space from the prior. A plain vector's matrices and
// vectors are Eigen's own fixed-size types, with no space to size them by name. Every method is
// exact for a linear model, so the update names none. The program simulates the target and the
// fixes from the same models the filter assumes, runs the filter, and checks that its estimates are
// consistent with their covariances: the normalized estimation error squared (NEES) against the
// true state, and the normalized innovation squared (NIS) of the fixes. It returns nonzero if
// either check fails.
//
// It is the filter of constant_velocity.cpp written another way: the same motion, noises, prior,
// and random seed, so it prints the same NEES and NIS. Only the syntax differs. There, the state
// has named blocks and the library builds F and Q; here, the state is a plain vector and the
// matrices are written out.

#include <EmrysFilter/EmrysFilter.h>

#include <print>
#include <random>

// x, y, and their rates. A plain vector has no blocks to name, so the models take the whole of it.
using State = Emrys::Linear<4>;

int
main()
{
    double const dt = 0.5;  // seconds between fixes
    int const runs = 50;
    int const steps = 100;

    // F moves each position by its rate over the step, and Q is the noise that a white
    // acceleration of spectral density q adds over it, exactly discretized.
    double const q = 0.1;
    double const a = q * dt * dt * dt / 3;
    double const b = q * dt * dt / 2;
    double const c = q * dt;
    Eigen::Matrix4d const F{{1, 0, dt, 0}, {0, 1, 0, dt}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Eigen::Matrix4d const Q{{a, 0, b, 0}, {0, a, 0, b}, {b, 0, c, 0}, {0, b, 0, c}};

    // The fixes arrive at a fixed rate, so the motion is taken over that one step. A filter given it
    // computes its transition, F, and its noise once, in the state's space.
    auto const motion = Emrys::fixedStep(Emrys::processModel(F, Emrys::additiveNoise(Q)), dt);

    // A fix measures the position, H x, with noise of 1 m^2 variance on each axis.
    Eigen::Matrix<double, 2, 4> const H{{1, 0, 0, 0}, {0, 1, 0, 0}};
    double const fixVariance = 1.0;
    Eigen::Matrix2d const R = fixVariance * Eigen::Matrix2d::Identity();
    auto const fix = Emrys::measurementModel(H, Emrys::additiveNoise(R));

    // The prior: near the origin, moving slowly, its covariance given by its variances. The filter
    // takes the state's space from it.
    auto const prior = Emrys::GaussianParams<State>{
        Eigen::Vector4d{0.0, 0.0, 1.0, 0.5},
        Eigen::Vector4d{10.0, 10.0, 1.0, 1.0}.asDiagonal()
    };

    // Independent runs, one final estimate from each for the NEES test, and every fix of every run
    // for the NIS test, as in constant_velocity.cpp.
    auto nees = Emrys::NeesMonitor{runs};
    auto nis = Emrys::NisMonitor{runs * steps};
    auto rng = std::mt19937_64{2};
    for (int run = 0; run < runs; ++run)
    {
        auto filter = Emrys::KalmanFilter{motion, prior};

        // The prior's covariance is positive definite by construction, so toDensity() can't fail here.
        Eigen::Vector4d truth = Emrys::drawSamples(prior.toDensity().value(), 1, rng);
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate<State>(truth, motion, rng);
            Eigen::Vector2d const z = Emrys::simulate<State>(truth, fix, rng);
            filter.predict();
            auto const updated = filter.update(Emrys::observe(fix, z));
            if (!updated)
            {
                // The innovation covariance was not positive definite, and the state is unchanged.
                std::println("run {}, step {}: the update failed", run, k);
                return 1;
            }
            nis.add(*updated);
        }
        nees.add(truth, filter.state());
    }

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES {:.2f}, average NIS {:.2f} per degree of freedom",
        nees.average().value(),
        nis.average().value()
    );
    bool const consistent =
        nees.verdict(0.999) == Emrys::Consistency::Consistent && nis.verdict(0.999) == Emrys::Consistency::Consistent;
    std::println("{}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}

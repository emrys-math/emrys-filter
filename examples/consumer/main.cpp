// The program to replace with your estimator. It builds a Kalman filter for a target moving in the
// plane, takes one position fix, and prints the estimate, which shows that the project builds and
// runs. examples/constant_velocity.cpp in EmrysFilter is the same filter checked in simulation.

#include <EmrysFilter/EmrysFilter.h>

#include <print>

struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;

int
main()
{
    double const accelerationDensity = 0.1;  // m^2/s^3
    auto const motion = Emrys::ConstantVelocity<Position, Velocity>{accelerationDensity};
    double const fixVariance = 1.0;  // m^2 on each axis
    auto const gps = Emrys::measurementModel(
        Emrys::DifferenceMatrix<Position>::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixVariance * Emrys::DifferenceMatrix<Position>::Identity())
    );
    auto const prior = Emrys::GaussianParams<State>{
        Emrys::ValueVector<State>{0.0, 0.0, 1.0, 0.5},
        Emrys::DifferenceVector<State>{10.0, 10.0, 1.0, 1.0}.asDiagonal()
    };

    auto filter = Emrys::KalmanFilter{motion, prior};
    filter.predict(0.5);
    if (!filter.update(Emrys::observe(gps, Emrys::ValueVector<Position>{0.6, 0.2})))
    {
        std::println("the update failed");
        return 1;
    }
    Emrys::ValueVector<State> const mean = filter.state().mean();
    std::println("position ({:.2f}, {:.2f}) m, velocity ({:.2f}, {:.2f}) m/s", mean[0], mean[1], mean[2], mean[3]);
    return 0;
}

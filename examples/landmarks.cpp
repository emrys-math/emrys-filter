// A unicycle mapping the landmarks it sights: the state holds the vehicle and a repeated block of
// landmarks whose positions it learns, and each step the vehicle measures the range and bearing to
// one of them. The measurement model reads that one landmark, path<Landmarks>[k], so the unscented
// Kalman filter spreads its sigma points over the vehicle and that landmark alone, not over every
// landmark in the map, and the model is built for each sighting, since it holds which landmark it
// reads. The program simulates the vehicle, the landmarks, and the sightings from the models the
// filter assumes, and checks the filter's estimates against the truth by the normalized estimation
// error squared (NEES) over independent runs, of the whole state and of the landmark last sighted,
// and its innovations by the NIS. It returns nonzero if a check fails.

#include <EmrysFilter/EmrysFilter.h>

#include <cmath>
#include <print>
#include <random>

// The vehicle is a position in meters, a heading in radians, a speed in m/s, and a turn rate in
// rad/s, as in unicycle.cpp.
struct Position : Emrys::Linear<2>
{};

struct Heading : Emrys::RadiansPi<1>
{};

struct Speed : Emrys::Linear<1>
{};

struct TurnRate : Emrys::Linear<1>
{};

// A landmark is a position in meters, and the map is a repeated block of up to eight of them, as
// many as the state's space says.
struct Landmark : Emrys::Linear<2>
{};

struct Landmarks : Emrys::Repeated<Landmark, 8>
{
    using Repeated::Repeated;
};

using State = Emrys::Composite<Position, Heading, Speed, TurnRate, Landmarks>;

// A sighting is a range in meters and a bearing in radians, relative to the heading.
struct Range : Emrys::Linear<1>
{};

struct Bearing : Emrys::RadiansPi<1>
{};

using Sighting = Emrys::Composite<Range, Bearing>;

// The unicycle's motion over a step dt, as in unicycle.cpp: the position moves along the heading by
// the speed, and the heading turns by the turn rate.
[[nodiscard]] auto
unicycle()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            T const heading = Emrys::block<Heading>(in, x)(0);
            T const distance = dt * Emrys::block<Speed>(in, x)(0);
            auto const displacement = Emrys::DifferenceVector<Position, T>{
                distance * Eigen::numext::cos(heading),
                distance * Eigen::numext::sin(heading)
            };
            Emrys::DifferenceVector<Heading, T> const turn = dt * Emrys::block<TurnRate>(in, x);
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::plus<Position>(Emrys::block<Position>(in, x), displacement);
            Emrys::block<Heading>(out, y) = Emrys::plus<Heading>(Emrys::block<Heading>(in, x), turn);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>, Emrys::path<Speed>, Emrys::path<TurnRate>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Heading>)
    );
}

// The range and bearing from the vehicle to landmark k, with noise of 0.5 m standard deviation in
// range and 1 degree in bearing. reads() names the landmark by path<Landmarks>[k], one repetition
// of the repeated block, read under the repetition's tag, Landmark; the model holds k, so a model
// is built for each sighting, which costs nothing it would not cost to read k from a variable.
[[nodiscard]] auto
sightingOf(Eigen::Index k)
{
    Emrys::DifferenceMatrix<Sighting> const noise = Emrys::DifferenceVector<Sighting>{0.25, 3e-4}.asDiagonal();
    return Emrys::measurementModel(
        []<typename T>(auto const & in, auto const &, auto const & x) {
            Emrys::DifferenceVector<Position, T> const offset =
                Emrys::block<Landmark>(in, x) - Emrys::block<Position>(in, x);
            auto const direction = Emrys::ValueVector<Heading, T>{Eigen::numext::atan2(offset(1), offset(0))};
            Emrys::DifferenceVector<Heading, T> const bearing =
                Emrys::minus<Heading>(direction, Emrys::block<Heading>(in, x));
            return Emrys::ValueVector<Sighting, T>{T(offset.norm()), bearing(0)};
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Heading>, Emrys::path<Landmarks>[k]),
        Emrys::writes(Emrys::path<Range>, Emrys::path<Bearing>),
        Emrys::additiveNoise(noise)
    );
}

int
main()
{
    double const dt = 0.5;  // seconds between steps
    int const runs = 50;
    int const steps = 80;
    Eigen::Index const count = 4;  // landmarks in the map

    // The speed is a random walk and the turn rate decays toward zero, as in unicycle.cpp; the
    // landmarks stay where they are.
    auto const motion = Emrys::compose(
        unicycle(),
        Emrys::RandomWalk<Speed>{0.1},
        Emrys::OrnsteinUhlenbeck<TurnRate>{Emrys::TimeConstant{10.0}, Emrys::StationaryVariance{0.01}},
        Emrys::Constant<Landmarks>{}
    );

    // The vehicle starts near the origin heading east at about 3 m/s, and the landmarks are known to
    // about 3 m around where a survey put them.
    auto const space = State{Position{}, Heading{}, Speed{}, TurnRate{}, Landmarks{Landmark{}, count}};
    Eigen::VectorXd mean(5 + 2 * count);
    mean << 0.0, 0.0, 0.0, 3.0, 0.0, 60.0, 0.0, 40.0, 50.0, -30.0, 50.0, -30.0, -50.0;
    Eigen::VectorXd variances(5 + 2 * count);
    variances << 1.0, 1.0, 0.01, 0.25, 0.01, Eigen::VectorXd::Constant(2 * count, 9.0);
    auto const prior = Emrys::GaussianParams<State>{space, mean, Eigen::MatrixXd{variances.asDiagonal()}};

    auto nees = Emrys::NeesMonitor{runs};
    auto landmarkNees = Emrys::NeesMonitor{runs};
    auto nis = Emrys::NisMonitor{runs * steps};
    auto rng = std::mt19937_64{3};
    for (int run = 0; run < runs; ++run)
    {
        auto filter = Emrys::KalmanFilter{motion, prior, Emrys::methods::unscented()};

        // The prior's covariance is positive definite by construction, so toDensity() can't fail here.
        Eigen::VectorXd truth = Emrys::drawSample(prior.toDensity().value(), rng);
        Eigen::Index sighted = 0;
        for (int k = 0; k < steps; ++k)
        {
            truth = Emrys::simulate(space, truth, motion, dt, rng);
            filter.predict(dt);

            // The vehicle sights the landmarks in turn, one a step.
            sighted = Eigen::Index(k) % count;
            auto const sighting = sightingOf(sighted);
            Emrys::ValueVector<Sighting> const z = Emrys::simulate(space, truth, sighting, rng);
            auto const updated = filter.update(Emrys::observe(Emrys::methods::unscented(), sighting, z));
            if (!updated)
            {
                // The innovation covariance was not positive definite, and the state is unchanged.
                std::println("run {}, step {}: an update failed", run, k);
                return 1;
            }
            nis.add(*updated);
        }
        nees.add(truth, filter.state());

        // The NEES of the landmark sighted last, read by the same path value as the sighting.
        landmarkNees.add<Landmarks>(truth, filter.state(), sighted);
    }

    // Each average is the sum divided by its degrees of freedom, so 1 is ideal.
    std::println(
        "average NEES: whole state {:.2f}, landmark last sighted {:.2f}; average NIS {:.2f}",
        nees.average().value(),
        landmarkNees.average().value(),
        nis.average().value()
    );
    constexpr auto Consistent = Emrys::Consistency::Consistent;
    bool const consistent = nees.verdict(0.999) == Consistent && landmarkNees.verdict(0.999) == Consistent
                         && nis.verdict(0.999) == Consistent;
    std::println("{}", consistent ? "consistent" : "NOT consistent");
    return consistent ? 0 : 1;
}

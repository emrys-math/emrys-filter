#include <EmrysFilter/EmrysFilter.h>

#include <cstdio>

struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;

int
main()
{
    auto const model = Emrys::ConstantVelocity<Position, Velocity>{0.1};
    auto const fixNoise = Emrys::GaussianParams<Emrys::Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
    auto const gps = Emrys::measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return Emrys::block<Position>(in, x);
        },
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixNoise)
    );
    auto const prior = Emrys::GaussianParams<State>{
        State{},
        Eigen::Vector4d{0.0, 0.0, 1.0, 0.5},
        Eigen::Vector4d{10.0, 10.0, 1.0, 1.0}.asDiagonal().toDenseMatrix()
    };

    auto filter = Emrys::KalmanFilter{model, prior};
    filter.predict(0.5);
    if (!filter.update(Emrys::observe(Emrys::methods::linearized(), gps, Eigen::Vector2d{0.6, 0.2})))
    {
        return 1;
    }
    Eigen::Vector4d const mean = filter.state().mean();
    std::printf("position (%.2f, %.2f) m, velocity (%.2f, %.2f) m/s\n", mean[0], mean[1], mean[2], mean[3]);
    return 0;
}

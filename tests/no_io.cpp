// EMRYS_FILTER_NO_IO (Core/Macros.h, Core/Print.h): defined before the first EmrysFilter header,
// it leaves out the library's printing, so a distribution and a filter have no operator<< and no
// std::formatter, while the rest of the library builds and runs unchanged. The evidence for the
// opt-out that a build without iostreams relies on.

#define EMRYS_FILTER_NO_IO

#include <EmrysFilter/EmrysFilter.h>

#include <gtest/gtest.h>

#include <format>
#include <ostream>

namespace
{

struct Position : Emrys::Linear<2>
{};

struct Velocity : Emrys::Linear<2>
{};

using State = Emrys::Composite<Position, Velocity>;
using Model = Emrys::ConstantVelocity<Position, Velocity>;

template <typename T>
concept Streamable = requires(std::ostream & stream, T const & value) { stream << value; };

}  // namespace

// Nothing the library defines streams or formats, and a filter still predicts.
TEST(NoIo, LeavesOutPrinting)
{
    static_assert(!Streamable<Emrys::GaussianParams<State>>);
    static_assert(!Streamable<Emrys::GaussianMixtureParams<State>>);
    static_assert(!Streamable<Emrys::WeightError>);
    static_assert(!Streamable<Emrys::KalmanFilter<Emrys::Gaussian<State>, Model>>);
    static_assert(!std::formattable<Emrys::GaussianParams<State>, char>);

    auto filter = Emrys::KalmanFilter{
        Model{Emrys::ConstantVelocity<Position, Velocity>{0.1}},
        Emrys::GaussianParams<State>{State{}, Eigen::Vector4d{0, 0, 1, 0}, Eigen::Matrix4d::Identity()}
    };
    filter.predict(1.0);
    EXPECT_DOUBLE_EQ(filter.state().mean()(0), 1.0);
}

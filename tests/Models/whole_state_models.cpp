// Function forms and whole-state models (roadmap steps 17zb and 17zc; decisions.md [model-policies],
// [function-scalars], [whole-state-models], [sensors]), in Models/Methods.h (internal::callForm()),
// Models/ProcessModel.h, Models/MeasurementModel.h, and Models/Batched.h: a model's function written as h(x),
// h(in, x), or h(in, out, x), and a process model's with dt after, with or without its scalar as a template
// parameter, predicts and updates bitwise alike, per point and batched; a model without reads() and writes() reads and
// writes the whole state, with its noise over it, and a constant-velocity filter on Linear<4> written so equals the
// tagged one on Composite<Position, Velocity>; a whole-state measurement's output space is its noise's; and the matrix
// process model processModel(F, noise) is the textbook Kalman filter's prediction. Expected values come from the
// tagged models, which the other tests check, and from the textbook equations written out in plain Eigen.

#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Models/Batched.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>

#include <gtest/gtest.h>

#include <type_traits>

using Emrys::Composite;
using Emrys::GaussianParams;
using Emrys::Linear;

namespace jacobians = Emrys::jacobians;
namespace methods = Emrys::methods;

namespace
{

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

struct Fix : Linear<2>
{};

using Tagged = Composite<Position, Velocity>;
using PlainState = Linear<4>;

constexpr double timeStep = 0.5;
constexpr double density = 0.3;  // the velocity's white-noise spectral density

Eigen::Matrix4d
transitionMatrix()
{
    Eigen::Matrix4d f = Eigen::Matrix4d::Identity();
    f.topRightCorner<2, 2>() = timeStep * Eigen::Matrix2d::Identity();
    return f;
}

// Q_c over the whole state: the velocity's density, and none on the position.
Eigen::Matrix4d
spectralDensity()
{
    Eigen::Matrix4d qc = Eigen::Matrix4d::Zero();
    qc.bottomRightCorner<2, 2>() = density * Eigen::Matrix2d::Identity();
    return qc;
}

Eigen::Matrix2d
fixCovariance()
{
    return Eigen::Vector2d{0.8, 1.3}.asDiagonal();
}

Eigen::Matrix<double, 2, 4>
fixMatrix()
{
    Eigen::Matrix<double, 2, 4> h = Eigen::Matrix<double, 2, 4>::Zero();
    h.leftCols<2>().setIdentity();
    return h;
}

Eigen::Vector4d
priorMean()
{
    return {3.0, 2.0, 0.5, -0.2};
}

Eigen::Matrix4d
priorCovariance()
{
    Eigen::Matrix4d cov = Eigen::Vector4d{4.0, 3.0, 1.0, 0.5}.asDiagonal();
    cov(0, 1) = cov(1, 0) = 0.8;
    cov(0, 2) = cov(2, 0) = 0.3;
    return cov;
}

template <typename S>
GaussianParams<S>
prior()
{
    return GaussianParams<S>{S{}, priorMean(), priorCovariance()};
}

// Constant velocity on the tagged state, in the full form.
auto
taggedProcess()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & out, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(out);
            Emrys::block<Position>(out, y) = Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(out, y) = Emrys::block<Velocity>(in, x);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(density))
    );
}

// The same model with its one space, f(in, x, dt): reads() and writes() define one space.
auto
taggedProcessInOneSpace()
{
    return Emrys::processModel(
        []<typename T>(auto const & in, auto const & x, double dt) {
            auto y = Emrys::zeros<T>(in);
            Emrys::block<Position>(in, y) = Emrys::block<Position>(in, x) + T(dt) * Emrys::block<Velocity>(in, x);
            Emrys::block<Velocity>(in, y) = Emrys::block<Velocity>(in, x);
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(density))
    );
}

// The same model without spaces, f(x, dt).
auto
taggedProcessWithoutSpaces()
{
    return Emrys::processModel(
        []<typename T>(auto const & x, double dt) {
            auto y = x.eval();
            y.template head<2>() += T(dt) * x.template tail<2>();
            return y;
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
        Emrys::continuousNoise(Emrys::on<Velocity>(density))
    );
}

// A position fix on the tagged state in each form: h(in, out, x), h(in, x), and h(x).
auto
fixNoise()
{
    return Emrys::additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), fixCovariance()});
}

auto
fixInFullForm()
{
    return Emrys::measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return Emrys::block<Position>(in, x).eval();
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        fixNoise()
    );
}

auto
fixInOneSpace()
{
    return Emrys::measurementModel(
        [](auto const & in, auto const & x) {
            return Emrys::block<Position>(in, x).eval();
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        fixNoise()
    );
}

auto
fixWithoutSpaces()
{
    return Emrys::measurementModel(
        [](auto const & x) {
            return x.template head<2>().eval();
        },
        Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
        fixNoise()
    );
}

// Expects two Gaussians to be equal bit for bit.
template <typename A, typename B>
void
expectBitwise(A const & a, B const & b)
{
    EXPECT_EQ(a.mean(), b.mean());
    EXPECT_EQ(a.covarianceLower(), b.covarianceLower());
}

// Expects two Gaussians to agree to 1e-12, as one computation in two orders of operations does.
template <typename A, typename B>
void
expectSame(A const & a, B const & b)
{
    EXPECT_LT((a.mean() - b.mean()).norm(), 1e-12);
    EXPECT_LT((a.covarianceLower() - b.covarianceLower()).norm(), 1e-12);
}

}  // namespace

// The form a function takes is told by the count of its arguments, before any body is instantiated.
static_assert(
    Emrys::internal::functionForm<
        decltype([](auto const &, auto const &, auto const &) {
            return 0;
        }),
        int,
        int,
        Eigen::Vector2d>
    == 3
);
static_assert(
    Emrys::internal::functionForm<
        decltype([](auto const &, auto const &) {
            return 0;
        }),
        int,
        int,
        Eigen::Vector2d>
    == 2
);
static_assert(
    Emrys::internal::functionForm<
        decltype([](auto const &) {
            return 0;
        }),
        int,
        int,
        Eigen::Vector2d>
    == 1
);
static_assert(
    Emrys::internal::functionForm<
        decltype([](auto const &, double) {
            return 0;
        }),
        int,
        int,
        Eigen::Vector2d,
        double>
    == 1
);
static_assert(
    Emrys::internal::functionForm<
        decltype([] {
            return 0;
        }),
        int,
        int,
        Eigen::Vector2d>
    == 0
);

// A function may take the scalar of x as its one template parameter in each form, which the call
// supplies (decisions.md [function-scalars]); a generic function without one takes the call as it is.
static_assert(
    Emrys::internal::functionForm<
        decltype([]<typename T>(auto const &, auto const &, auto const &) {
            return T(0);
        }),
        int,
        int,
        Eigen::Vector2d>
    == 3
);
static_assert(
    Emrys::internal::functionForm<
        decltype([]<typename T>(auto const &, auto const &, double) {
            return T(0);
        }),
        int,
        int,
        Eigen::Vector2d,
        double>
    == 2
);
static_assert(
    Emrys::internal::functionForm<
        decltype([]<typename T>(auto const &) {
            return T(0);
        }),
        int,
        int,
        Eigen::Vector2d>
    == 1
);
static_assert(std::is_same_v<
              decltype(Emrys::internal::callForm(
                  []<typename T>(auto const &) {
                      return T(0);
                  },
                  0,
                  0,
                  Eigen::Vector2f{}
              )),
              float>);
static_assert(std::is_same_v<
              decltype(Emrys::internal::callForm(
                  [](auto const & x) {
                      return x(0);
                  },
                  0,
                  0,
                  Eigen::Vector2f{}
              )),
              float>);

// A process model and a measurement model written in each of the three forms predict and update bit
// for bit alike, under a linear and a sigma-point method.
TEST(FunctionForms, ThreeFormsPredictAndUpdateAlike)
{
    auto const x = prior<Tagged>();
    auto const z = Eigen::Vector2d{3.4, 1.7};
    auto const check = [&](auto const & method) {
        auto const full = Emrys::predict(method, x, taggedProcess(), timeStep);
        expectBitwise(Emrys::predict(method, x, taggedProcessInOneSpace(), timeStep), full);
        expectBitwise(Emrys::predict(method, x, taggedProcessWithoutSpaces(), timeStep), full);
        auto const expected = Emrys::update(full, Emrys::observe(method, fixInFullForm(), z));
        ASSERT_TRUE(expected);
        auto const compare = [&](auto const & model) {
            auto const updated = Emrys::update(full, Emrys::observe(method, model, z));
            ASSERT_TRUE(updated);
            expectBitwise(updated->branches.front().posterior, expected->branches.front().posterior);
            EXPECT_EQ(updated->logLikelihood(), expected->logLikelihood());
        };
        compare(fixInOneSpace());
        compare(fixWithoutSpaces());
    };
    check(methods::linearized());
    check(methods::unscented());
}

// A batched function in each of its forms predicts and updates bit for bit alike, and as the
// per-point form does to rounding.
TEST(FunctionForms, BatchedFormsPredictAndUpdateAlike)
{
    auto const x = prior<Tagged>();
    auto const process = [](auto f) {
        return Emrys::processModel(
            Emrys::batched(f),
            Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
            Emrys::writes(Emrys::path<Position>, Emrys::path<Velocity>),
            Emrys::continuousNoise(Emrys::on<Velocity>(density))
        );
    };
    auto const step = [](auto const & in, auto const & x, double dt, auto & y) {
        using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
        y.values() = x;
        Emrys::block<Position>(in, y.values()) += T(dt) * Emrys::block<Velocity>(in, x);
    };
    auto const full = process([step](auto const & in, auto const &, auto const & x, double dt, auto & y) {
        step(in, x, dt, y);
    });
    auto const oneSpace = process([step](auto const & in, auto const & x, double dt, auto & y) {
        step(in, x, dt, y);
    });
    auto const withoutSpaces = process([](auto const & x, double dt, auto & y) {
        using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
        y.values() = x;
        y.values().template topRows<2>() += T(dt) * x.template bottomRows<2>();
    });
    auto const measurement = [](auto h) {
        return Emrys::measurementModel(
            Emrys::batched(h),
            Emrys::reads(Emrys::path<Position>, Emrys::path<Velocity>),
            Emrys::writes(Emrys::path<Fix>),
            fixNoise()
        );
    };
    auto const fullFix = measurement([](auto const & in, auto const &, auto const & x, auto & y) {
        y.values() = Emrys::block<Position>(in, x);
    });
    auto const oneSpaceFix = measurement([](auto const & in, auto const & x, auto & y) {
        y.values() = Emrys::block<Position>(in, x);
    });
    auto const fixWithoutSpace = measurement([](auto const & x, auto & y) {
        y.values() = x.template topRows<2>();
    });
    auto const z = Eigen::Vector2d{3.4, 1.7};
    auto const check = [&](auto const & method) {
        auto const predicted = Emrys::predict(method, x, full, timeStep);
        expectBitwise(Emrys::predict(method, x, oneSpace, timeStep), predicted);
        expectBitwise(Emrys::predict(method, x, withoutSpaces, timeStep), predicted);
        expectSame(Emrys::predict(method, x, taggedProcess(), timeStep), predicted);
        auto const expected = Emrys::update(predicted, Emrys::observe(method, fullFix, z));
        ASSERT_TRUE(expected);
        auto const compare = [&](auto const & model) {
            auto const updated = Emrys::update(predicted, Emrys::observe(method, model, z));
            ASSERT_TRUE(updated);
            expectBitwise(updated->branches.front().posterior, expected->branches.front().posterior);
        };
        compare(oneSpaceFix);
        compare(fixWithoutSpace);
        auto const perPoint = Emrys::update(predicted, Emrys::observe(method, fixInFullForm(), z));
        ASSERT_TRUE(perPoint);
        expectSame(expected->branches.front().posterior, perPoint->branches.front().posterior);
    };
    check(methods::linearized());
    check(methods::unscented());
}

// A whole-state constant-velocity model on Linear<4>, f(x, dt) with continuousNoise() of a matrix,
// predicts as the tagged model does on Composite<Position, Velocity>, and so does the same model with
// additiveNoise() of the discretized Gaussian.
TEST(WholeStateModels, ConstantVelocityOnLinearMatchesTheTaggedModel)
{
    auto const step = []<typename T>(auto const & x, double dt) {
        auto y = x.eval();
        y.template head<2>() += T(dt) * x.template tail<2>();
        return y;
    };
    auto const continuous = Emrys::processModel(step, Emrys::continuousNoise(spectralDensity()));
    auto const additive = Emrys::processModel(
        step,
        Emrys::additiveNoise(
            GaussianParams<PlainState>{Eigen::Vector4d::Zero(), Eigen::Matrix4d{timeStep * spectralDensity()}}
        )
    );
    static_assert(Emrys::ProcessModel<decltype(continuous)>);
    EXPECT_TRUE(continuous.partitions(PlainState{}));
    static_assert(decltype(continuous)::staticPartition<PlainState>() == Emrys::internal::StaticPartition::holds);
    auto const check = [&](auto const & method) {
        auto const expected = Emrys::predict(method, prior<Tagged>(), taggedProcess(), timeStep);
        auto const predicted = Emrys::predict(method, prior<PlainState>(), continuous, timeStep);
        static_assert(std::is_same_v<decltype(predicted), GaussianParams<PlainState> const>);
        expectSame(predicted, expected);
        expectSame(Emrys::predict(method, prior<PlainState>(), additive, timeStep), expected);
    };
    check(methods::linearized());
    check(methods::unscented());
    check(methods::cubature());
}

// A whole-state position fix on Linear<4>, by function and by matrix, updates as the tagged fix on
// the position block does; its output space is its noise's, Linear<2> here and the tag Fix for a
// noise over it, and the function's, Linear<2>, for zero-mean noise given by its covariance alone.
TEST(WholeStateModels, PositionFixOnLinearMatchesTheTaggedFix)
{
    auto const z = Eigen::Vector2d{3.4, 1.7};
    auto const tagged =
        Emrys::measurementModel(Eigen::Matrix2d::Identity(), Emrys::reads(Emrys::path<Position>), fixNoise());
    auto const byFunction = Emrys::measurementModel(
        [](auto const & x) {
            return x.template head<2>().eval();
        },
        fixNoise()
    );
    auto const byMatrix = Emrys::measurementModel(fixMatrix(), fixNoise());
    auto const zeroMean = Emrys::measurementModel(fixMatrix(), Emrys::additiveNoise(fixCovariance()));
    auto const taggedZeroMean = Emrys::measurementModel(
        Eigen::Matrix2d::Identity(),
        Emrys::reads(Emrys::path<Position>),
        Emrys::additiveNoise(fixCovariance())
    );
    auto const intoFix = Emrys::measurementModel(
        fixMatrix(),
        Emrys::additiveNoise(GaussianParams<Fix>{Eigen::Vector2d::Zero(), fixCovariance()})
    );
    auto const check = [&](auto const & method) {
        auto const expected = Emrys::update(prior<Tagged>(), Emrys::observe(method, tagged, z));
        ASSERT_TRUE(expected);
        auto const compare = [&](auto const & model) {
            auto const updated = Emrys::update(prior<PlainState>(), Emrys::observe(method, model, z));
            ASSERT_TRUE(updated);
            expectSame(updated->branches.front().posterior, expected->branches.front().posterior);
            EXPECT_NEAR(updated->logLikelihood(), expected->logLikelihood(), 1e-12);
            EXPECT_NEAR(updated->branches.front().nis(), expected->branches.front().nis(), 1e-12);
        };
        compare(byFunction);
        compare(byMatrix);
        compare(intoFix);
        compare(zeroMean);
        auto const zeroMeanTagged = Emrys::update(prior<Tagged>(), Emrys::observe(method, taggedZeroMean, z));
        ASSERT_TRUE(zeroMeanTagged);
        expectSame(zeroMeanTagged->branches.front().posterior, expected->branches.front().posterior);
        static_assert(
            std::is_same_v<
                std::remove_cvref_t<decltype(Emrys::update(prior<PlainState>(), Emrys::observe(method, zeroMean, z))
                                                 ->branches.front())>,
                Emrys::UpdateBranch<PlainState, Linear<2>>>
        );
        auto const updated = Emrys::update(prior<PlainState>(), Emrys::observe(method, intoFix, z));
        static_assert(std::is_same_v<
                      std::remove_cvref_t<decltype(updated->branches.front())>,
                      Emrys::UpdateBranch<PlainState, Fix>>);
    };
    check(methods::linearized());
    check(methods::unscented());
}

// The matrix process model processModel(F, noise) and the matrix fix are the textbook Kalman
// filter: x = F x, P = F P F' + Q, then the update with K = P H' S^-1, written out in plain Eigen,
// under a linear and a sigma-point method, and as a KalmanFilter with no lambda written. It predicts
// as the whole-state function does, and its fixed step caches F as its transition. Its noises are
// given by their covariances alone, Q and R, and are zero-mean.
TEST(WholeStateModels, MatrixModelsAreTheTextbookKalmanFilter)
{
    Eigen::Matrix4d const f = transitionMatrix();
    Eigen::Matrix4d const q = timeStep * spectralDensity() + 0.01 * Eigen::Matrix4d::Identity();
    auto const model = Emrys::processModel(f, Emrys::additiveNoise(q));
    auto const fix = Emrys::measurementModel(fixMatrix(), Emrys::additiveNoise(fixCovariance()));
    auto const z = Eigen::Vector2d{3.4, 1.7};

    Eigen::Vector4d const predictedMean = f * priorMean();
    Eigen::Matrix4d const predictedCov = f * priorCovariance() * f.transpose() + q;
    Eigen::Matrix<double, 2, 4> const h = fixMatrix();
    Eigen::Matrix2d const s = h * predictedCov * h.transpose() + fixCovariance();
    Eigen::Matrix<double, 4, 2> const gain = predictedCov * h.transpose() * s.inverse();
    Eigen::Vector4d const mean = predictedMean + gain * (z - h * predictedMean);
    Eigen::Matrix4d const cov = (Eigen::Matrix4d::Identity() - gain * h) * predictedCov;

    auto const check = [&](auto const & method) {
        auto const predicted = Emrys::predict(method, prior<PlainState>(), model, timeStep);
        EXPECT_LT((predicted.mean() - predictedMean).norm(), 1e-12);
        EXPECT_LT(
            (predicted.covarianceLower() - predictedCov).template triangularView<Eigen::Lower>().toDenseMatrix().norm(),
            1e-12
        );
        auto const updated = Emrys::update(predicted, Emrys::observe(method, fix, z));
        ASSERT_TRUE(updated);
        auto const & posterior = updated->branches.front().posterior;
        EXPECT_LT((posterior.mean() - mean).norm(), 1e-12);
        EXPECT_LT(
            (posterior.covarianceLower() - cov).template triangularView<Eigen::Lower>().toDenseMatrix().norm(),
            1e-12
        );
    };
    check(methods::linearized());
    check(methods::unscented());

    auto filter = Emrys::KalmanFilter{Emrys::fixedStep(model, timeStep), prior<PlainState>()};
    EXPECT_LT((filter.model().transition() - f).norm(), 1e-15);
    filter.predict();
    ASSERT_TRUE(filter.update(Emrys::observe(fix, z)));
    EXPECT_LT((filter.state().mean() - mean).norm(), 1e-12);
}

// A whole-state batched process model writes the whole Jacobian, Y.jacobian() under
// jacobians::shared(), and predicts as the matrix model does.
TEST(WholeStateModels, BatchedWholeStateModelWritesTheWholeJacobian)
{
    Eigen::Matrix4d const f = transitionMatrix();
    auto const noise = Emrys::continuousNoise(spectralDensity());
    auto const batched = Emrys::processModel(
        Emrys::batched(
            [f](auto const & x, double, auto & y) {
                using T = typename std::remove_cvref_t<decltype(x)>::Scalar;
                y.values() = f.cast<T>() * x;
                if (y.wantsJacobians())
                {
                    y.jacobian() = f.cast<T>();
                }
            },
            jacobians::shared()
        ),
        noise
    );
    auto const matrix = Emrys::processModel(f, noise);
    auto const check = [&](auto const & method) {
        expectSame(
            Emrys::predict(method, prior<PlainState>(), batched, timeStep),
            Emrys::predict(method, prior<PlainState>(), matrix, timeStep)
        );
    };
    check(methods::linearized());
    check(methods::unscented());
}

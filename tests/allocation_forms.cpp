// Allocation control [allocation] for the square-root and information forms of the Gaussian (roadmap
// steps 16 and 17, decisions.md [gaussian-forms]): on a bounded space the accessors, the conversions between
// forms, the density, multiply(), the marginals and conditionals, and moving the reference, on a
// unit quaternion too, allocate nothing, and neither do the steps of the square-root, information,
// and square-root information filters under Linearized and Unscented and with a fixed-step model.
// Part of the allocation test target, built with EIGEN_RUNTIME_NO_MALLOC; allocation.cpp replaces
// the global operator new.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/InfoGaussian.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Distributions/SqrtInfoGaussian.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>

#include <gtest/gtest.h>

#include "Spaces/unit_quaternion.h"
#include "allocation.h"

using namespace EmrysTest::Allocation;

using Emrys::FixedStepModel;
using Emrys::InfoFilter;
using Emrys::InfoGaussianParams;
using Emrys::SqrtGaussianParams;
using Emrys::SqrtInfoFilter;
using Emrys::SqrtInfoGaussianParams;
using Emrys::SqrtKalmanFilter;

namespace methods = Emrys::methods;

namespace
{

// Reads a result so that nothing is optimized away, without allocating.
template <typename T>
void
use(T const & value)
{
    static double volatile sink = 0;
    sink = double(value) + sink;
}

}  // namespace

// The four accessors of every form, the conversions between them, and the densities on the
// bounded four-dimensional state allocate nothing.
TEST(Allocation, GaussianFormsAccessorsAndConversionsAllocateNothing)
{
    GaussianParams<State> const g = makeInitial();
    auto x = Vector4{0.5, -0.5, 1.5, 0.0};
    expectStepsAllocateNothing([&](int k) {
        x(0) = 0.01 * k;
        use(g.covariance().value()(0, 0) + g.precision().value()(0, 0) + g.covarianceRoot().value()(0, 0));
        use(g.precisionRoot().value()(0, 0));
        auto const root = Emrys::toSqrt(g);
        auto const info = Emrys::toInfo(g);
        auto const rootInfo = Emrys::toSqrtInfo(g);
        use(root->precision().value()(0, 0) + info->covariance().value()(0, 0) + rootInfo->covariance().value()(0, 0));
        use(root->precisionRoot().value()(0, 0) + info->covarianceRoot().value()(0, 0));
        use(rootInfo->covarianceRoot().value()(0, 0) + rootInfo->mean().value()(0));
        use(Emrys::toGaussian(*root).value().mean()(0) + Emrys::toGaussian(*info).value().mean()(0));
        use(Emrys::toSqrt(*rootInfo).value().mean()(0) + Emrys::toInfo(*root).value().reference()(0));
        use(Emrys::toSqrtInfo(*info).value().reference()(0));
        use(root->toDensity()->logDensity(x) + info->toDensity()->logDensity(x));
        use(rootInfo->toDensity()->logDensity(x) + root->toDensity()->squaredMahalanobisDistance(x));
    });
}

// The factor operations on the bounded state allocate nothing: adding information, multiplying
// in both information forms, the marginals and conditionals by tag path, and moving the
// reference.
TEST(Allocation, GaussianFormsFactorOperationsAllocateNothing)
{
    GaussianParams<State> const g = makeInitial();
    auto const info = Emrys::toInfo(g);
    auto const rootInfo = Emrys::toSqrtInfo(g);
    auto const root = Emrys::toSqrt(g);
    ASSERT_TRUE(info && rootInfo && root);
    Matrix4 const added = 0.1 * Matrix4::Identity();
    auto const vector = Vector4{0.1, 0.2, 0.3, 0.4};
    auto const values = Eigen::Vector2d{0.3, 0.9};
    expectStepsAllocateNothing([&](int k) {
        Vector4 reference = Vector4::Constant(0.01 * k);
        InfoGaussianParams<State> moved = *info;
        moved.addInfo(added, vector);
        moved.setReference(reference);
        SqrtInfoGaussianParams<State> rootMoved = *rootInfo;
        rootMoved.setReference(reference);
        use(moved.mean().value()(0) + rootMoved.mean().value()(0));
        use(Emrys::multiply(*info, moved).mean().value()(0));
        use(Emrys::multiply(*rootInfo, rootMoved).mean().value()(0));
        use(info->marginal<Position>().value().mean().value()(0)
            + rootInfo->marginal<Velocity>().value().mean().value()(0));
        use(root->marginal<Position>().mean()(0) + root->marginal<Velocity>().root()(0, 0));
        use(info->conditional<Velocity>(values).mean().value()(0));
        use(rootInfo->conditional<Position>(values).mean().value()(0));
    });
}

// The three filters over the forms predict and update without allocating, under Linearized,
// Iterated over Linearized and over Cubature, Unscented, and the sigma-point sets of step 17e, and
// with a fixed-step model under Linearized and Unscented, as the Kalman filter does.
TEST(Allocation, FormFiltersStepWithoutAllocating)
{
    auto const check = [](auto & filter, auto const & method) {
        using Model = typename std::remove_cvref_t<decltype(filter)>::ModelType;
        expectStepsAllocateNothing([&](int k) {
            auto const predicted = [&] {
                if constexpr (Emrys::isFixedStep<Model>)
                {
                    return filter.predict();
                }
                else
                {
                    return filter.predict(timeStep);
                }
            }();
            use(predicted.has_value());
            auto const updated = filter.update(makeFix(fixAt(k), method));
            if constexpr (requires { updated.has_value(); })
            {
                use(updated.has_value());
            }
            else
            {
                use(updated.accepted);
            }
        });
    };
    auto const checkAll = [&](auto const & model, auto const & method, auto const &... methodArgument) {
        auto root = SqrtKalmanFilter{model, Emrys::toSqrt(makeInitial()).value(), methodArgument...};
        check(root, method);
        auto info = InfoFilter{model, Emrys::toInfo(makeInitial()).value(), methodArgument...};
        check(info, method);
        auto rootInfo = SqrtInfoFilter{model, Emrys::toSqrtInfo(makeInitial()).value(), methodArgument...};
        check(rootInfo, method);
    };
    checkAll(makeModel(), methods::linearized(), methods::linearized());
    checkAll(
        makeModel(),
        Emrys::methods::iterated(methods::linearized(), {.tolerance = 0.0, .maxIterations = 4}),
        methods::linearized()
    );
    checkAll(makeModel(), methods::unscented(), methods::unscented());
    checkAll(makeModel(), Emrys::methods::cubature(), Emrys::methods::cubature());
    checkAll(makeModel(), Emrys::methods::sphericalSimplex<0.2>(), Emrys::methods::sphericalSimplex<0.2>());
    checkAll(makeModel(), Emrys::methods::gaussHermite<3>(), Emrys::methods::gaussHermite<3>());
    checkAll(
        makeModel(),
        Emrys::methods::iterated(Emrys::methods::cubature(), {.tolerance = 0.0, .maxIterations = 4}),
        methods::linearized()
    );
    checkAll(FixedStepModel{makeModel(), State{}, timeStep}, methods::linearized());
    checkAll(FixedStepModel{makeModel(), State{}, timeStep, methods::unscented()}, methods::unscented());
}

namespace
{

struct Attitude : EmrysTest::AutoDiffQuaternion
{};

struct ClosedFormAttitude : EmrysTest::UnitQuaternion
{};

template <typename Q>
GaussianParams<Emrys::Composite<Q, Position>>
makeRigid()
{
    auto mean = Eigen::Matrix<double, 6, 1>{};
    mean << EmrysTest::exponential(Eigen::Vector3d{0.3, -0.2, 0.1}), 1.0, 2.0;
    Eigen::Matrix<double, 5, 5> cov = Eigen::Matrix<double, 5, 5>::Identity();
    cov(0, 3) = cov(3, 0) = 0.3;
    return GaussianParams<Emrys::Composite<Q, Position>>{mean, cov};
}

}  // namespace

// Moving the reference on a unit quaternion, whose reset Jacobian comes from automatic
// differentiation or from the closed form, allocates nothing in either information form.
TEST(Allocation, GaussianFormsMoveTheReferenceOnAQuaternionWithoutAllocating)
{
    auto const check = [](auto const & g) {
        using Space = typename std::remove_cvref_t<decltype(g)>::SpaceType;
        auto const info = Emrys::toInfo(g);
        auto const rootInfo = Emrys::toSqrtInfo(g);
        ASSERT_TRUE(info && rootInfo);
        expectStepsAllocateNothing([&](int k) {
            Eigen::Matrix<double, 5, 1> const d = 0.002 * k * Eigen::Matrix<double, 5, 1>::Ones();
            Eigen::Matrix<double, 6, 1> const reference = plus(Space{}, g.mean(), d);
            InfoGaussianParams<Space> moved = *info;
            moved.setReference(reference);
            SqrtInfoGaussianParams<Space> rootMoved = *rootInfo;
            rootMoved.setReference(reference);
            use(moved.mean().value()(0) + rootMoved.mean().value()(0));
        });
    };
    check(makeRigid<Attitude>());
    check(makeRigid<ClosedFormAttitude>());
}

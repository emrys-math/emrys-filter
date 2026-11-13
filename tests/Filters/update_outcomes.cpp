// The outcomes of an update (decisions.md [update-outcomes], [gaussian-forms]): a Gate on one branch
// decides on the NIS before the state is written, in the covariance, square-root, information, and
// square-root information forms, so a fix it rejects leaves the state as it was and still reports
// the fix's NIS and finite log-likelihood, and a fix it passes gives the state Collapse gives; and a
// mixture filter whose only accepting component has weight zero rejects the measurement and keeps its
// weights. Headers under test: Filters/KalmanFilter.h, Models/Measurement.h, Models/Condition.h,
// Models/FormSteps.h, Models/Reduction.h, Filters/MixtureFilter.h.

#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Filters/InfoFilter.h>
#include <EmrysFilter/Filters/KalmanFilter.h>
#include <EmrysFilter/Filters/MixtureFilter.h>
#include <EmrysFilter/Filters/SqrtInfoFilter.h>
#include <EmrysFilter/Filters/SqrtKalmanFilter.h>
#include <EmrysFilter/Models/Reduction.h>

#include <gtest/gtest.h>

#include "Filters/form_filters.h"

#include <cmath>
#include <limits>
#include <type_traits>

using namespace EmrysTest::FormFilters;

using Emrys::GaussianMixtureFilter;
using Emrys::GaussianMixtureParams;
using Emrys::InfoFilter;
using Emrys::KalmanFilter;
using Emrys::SqrtInfoFilter;
using Emrys::SqrtKalmanFilter;
using Emrys::toInfo;
using Emrys::toSqrt;
using Emrys::toSqrtInfo;

namespace reductions = Emrys::reductions;

namespace
{

// Predicts filter over one step, as every form can from makeInitial(). \returns whether it could.
template <typename Filter>
bool
predictOnce(Filter & filter)
{
    if constexpr (std::is_void_v<decltype(filter.predict(timeStep))>)
    {
        filter.predict(timeStep);
        return true;
    }
    else
    {
        return filter.predict(timeStep).has_value();
    }
}

// The state of a filter over any form, as a Gaussian.
template <typename Form>
GaussianParams<State>
gaussianOf(Form const & form)
{
    if constexpr (std::is_same_v<Form, GaussianParams<State>>)
    {
        return form;
    }
    else
    {
        return Emrys::toGaussian(form).value();
    }
}

// Calls check with a filter over each form of makeInitial().
template <typename Check>
void
forEachForm(Check const & check)
{
    check(KalmanFilter{makeModel(), makeInitial()});
    check(SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value()});
    check(InfoFilter{makeModel(), toInfo(makeInitial()).value()});
    check(SqrtInfoFilter{makeModel(), toSqrtInfo(makeInitial()).value()});
}

}  // namespace

// A fix at (40, 0.5) after one prediction has a NIS of 180.5 under the covariance form, far beyond
// the 0.99 threshold of 9.21 for two coordinates. Every form rejects it before writing its state,
// which stays as it was, and reports the NIS and log-likelihood the covariance form's Collapse
// computes for the same fix. The tolerance is rounding in the forms' own arithmetic; the covariance
// form's state is compared exactly.
TEST(UpdateOutcomes, AGateRejectsInEveryFormAndWritesNothing)
{
    auto const fix = makeFix(Eigen::Vector2d{40.0, 0.5});
    auto reference = KalmanFilter{makeModel(), makeInitial()};
    reference.predict(timeStep);
    auto const collapsed = reference.update(fix);
    ASSERT_TRUE(collapsed);
    EXPECT_GT(collapsed->branches[0].nis(), reductions::gate(0.99).threshold(2));

    forEachForm([&](auto filter) {
        ASSERT_TRUE(predictOnce(filter));
        auto const before = filter.state();
        auto const gated = filter.update(fix, reductions::gate(0.99));
        ASSERT_TRUE(gated);
        EXPECT_FALSE(gated->accepted);
        EXPECT_NEAR(gated->branches[0].nis(), collapsed->branches[0].nis(), 1e-9);
        EXPECT_NEAR(gated->logLikelihood(), collapsed->logLikelihood(), 1e-9);
        EXPECT_TRUE(std::isfinite(gated->logLikelihood()));
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(before)>, GaussianParams<State>>)
        {
            EXPECT_TRUE(filter.state().mean() == before.mean());
            EXPECT_TRUE(filter.state().covarianceLower() == before.covarianceLower());
        }
        expectSameState(gaussianOf(filter.state()), gaussianOf(before), 1e-12);
    });
}

// A fix near the prediction passes the gate, and every form's state is then the state Collapse gives
// the same form, since one branch is its own collapse; the covariance form's is the same exactly.
TEST(UpdateOutcomes, AGatePassesAFixAsCollapseDoes)
{
    auto const fix = makeFix(fixAt(1));
    forEachForm([&](auto filter) {
        ASSERT_TRUE(predictOnce(filter));
        auto collapsed = filter;
        auto const plain = collapsed.update(fix);
        auto const gated = filter.update(fix, reductions::gate(0.99));
        ASSERT_TRUE(plain && gated);
        EXPECT_TRUE(gated->accepted);
        EXPECT_NEAR(gated->branches[0].nis(), plain->branches[0].nis(), 1e-12);
        EXPECT_NEAR(gated->logLikelihood(), plain->logLikelihood(), 1e-12);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(filter.state())>, GaussianParams<State>>)
        {
            EXPECT_TRUE(filter.state().mean() == collapsed.state().mean());
            EXPECT_TRUE(filter.state().covarianceLower() == collapsed.state().covarianceLower());
        }
        expectSameState(gaussianOf(filter.state()), gaussianOf(collapsed.state()), 1e-12);
    });
}

// A gate given through select() decides one branch in place as one over collapse() does.
TEST(UpdateOutcomes, AGateOverSelectRejectsInPlace)
{
    auto filter = SqrtKalmanFilter{makeModel(), toSqrt(makeInitial()).value()};
    ASSERT_TRUE(filter.predict(timeStep));
    auto const before = filter.state();
    auto const gated = filter.update(makeFix(Eigen::Vector2d{40.0, 0.5}), reductions::gate(0.99, reductions::select()));
    ASSERT_TRUE(gated);
    EXPECT_FALSE(gated->accepted);
    EXPECT_TRUE(filter.state().mean() == before.mean());
    EXPECT_TRUE(filter.state().root() == before.root());
}

// Two components: one of weight one, far from the fix, which the gate rejects, and one of weight
// zero at the fix, which it passes. No component of nonzero weight accepts the fix, so the mixture
// rejects it, with both components marked rejected, and keeps its weights and its first component as
// they were; the log-likelihood is the first component's, the second adding nothing.
TEST(UpdateOutcomes, AMixtureRejectsAFixThatOnlyAComponentOfWeightZeroAccepts)
{
    auto prior = GaussianMixtureParams<State>{State{}};
    prior.add(0.0, Vector4{0.0, 0.0, 1.0, 0.5}, makeInitial().covarianceLower());
    prior.add(-std::numeric_limits<double>::infinity(), Vector4{40.0, 0.5, 1.0, 0.5}, makeInitial().covarianceLower());
    auto filter = GaussianMixtureFilter{makeModel(), prior};
    auto const fix = makeFix(Eigen::Vector2d{40.0, 0.5});
    auto alone = KalmanFilter{makeModel(), makeInitial()};
    auto const expected = alone.update(fix);
    ASSERT_TRUE(expected);

    auto const updated = filter.update(fix, reductions::gate(0.99));
    ASSERT_TRUE(updated);
    EXPECT_FALSE(updated->accepted);
    ASSERT_EQ(updated->components.size(), 2u);
    EXPECT_FALSE(updated->components[0].accepted);
    EXPECT_FALSE(updated->components[1].accepted);
    EXPECT_NEAR(updated->logLikelihood(), expected->logLikelihood(), 1e-12);
    ASSERT_EQ(filter.state().size(), 2);
    EXPECT_EQ(filter.state().begin()->logWeight, 0.0);
    EXPECT_EQ(filter.state()[1].logWeight, -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(filter.state().begin()->component.mean() == makeInitial().mean());
    EXPECT_TRUE(filter.state().begin()->component.covarianceLower() == makeInitial().covarianceLower());
}

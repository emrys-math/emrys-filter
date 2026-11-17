#pragma once

// A recursion of the user's own, for tests/Filters/kalman_filter.cpp and tests/Filters/gaussian_mixture_filter.cpp:
// FadingMemory wraps a KalmanRecursion and inflates each predicted covariance by alpha squared, a fading-memory
// filter that forgets old measurements, which the library doesn't ship. It meets the Recursion concept by its
// interface alone, with nothing of the library's specialized, so it runs inside a KalmanFilter and on each component
// of a MixtureFilter [filters-and-recursions], [user-extension-points]. usage.md, "Writing a recursion", shows it.
// FadingModes does the same to each mode of a hybrid over an ImmRecursion, for tests/Filters/imm_filter.cpp, and runs
// inside an ImmFilter.

#include <EmrysFilter/Filters/Recursion.h>
#include <EmrysFilter/Models/Measurement.h>

#include <utility>

namespace EmrysTest
{

// The recursion Inner with every predicted covariance multiplied by alpha squared.
template <Emrys::Recursion Inner>
class FadingMemory
{
public:

    using Scalar = typename Inner::Scalar;
    using SpaceType = typename Inner::SpaceType;
    using StateType = typename Inner::StateType;
    using ModelType = typename Inner::ModelType;
    using MethodType = typename Inner::MethodType;
    static constexpr bool Reentrant = Inner::Reentrant;

    template <Emrys::Measurement M>
    using UpdateType = typename Inner::template UpdateType<M>;

    template <Emrys::Measurement M>
    using ResultType = typename Inner::template ResultType<M>;

    FadingMemory(Inner inner, Scalar alpha)
        : m_inner{std::move(inner)}
        , m_alpha{alpha}
    {}

    [[nodiscard]] ModelType const &
    model() const
    {
        return m_inner.model();
    }

    [[nodiscard]] ModelType &
    model()
    {
        return m_inner.model();
    }

    [[nodiscard]] MethodType const &
    method() const
    {
        return m_inner.method();
    }

    [[nodiscard]] Scalar
    alpha() const
    {
        return m_alpha;
    }

    // Predicts as the inner recursion does, then inflates the covariance. The step and inputs, if
    // any, are passed through.
    template <typename... Args>
    void
    predict(StateType & state, Args const &... args) const
    {
        m_inner.predict(state, args...);
        state.setCovariance(m_alpha * m_alpha * state.covarianceLower());
    }

    // Updates as the inner recursion does: the policy, wrapped or not, is passed through.
    template <Emrys::Measurement M, typename Policy>
    [[nodiscard]] auto
    update(StateType & state, M const & measurement, Policy const & policy) const
    {
        return m_inner.update(state, measurement, policy);
    }

    // The update into a result, which the inner recursion refills.
    template <Emrys::Measurement M, typename Policy, typename Result>
    [[nodiscard]] auto
    update(StateType & state, M const & measurement, Policy const & policy, Result & result) const
    {
        return m_inner.update(state, measurement, policy, result);
    }

private:

    Inner m_inner;
    Scalar m_alpha;
};

// The recursion Inner over a hybrid, an ImmRecursion, with every mode's predicted covariance
// multiplied by alpha squared: a fading-memory IMM. Its steps write the inner recursion's scratch,
// so they are not const, and it has no resetScratch(), so a state from another allocator must not
// replace its filter's.
template <Emrys::Recursion Inner>
class FadingModes
{
public:

    using Scalar = typename Inner::Scalar;
    using SpaceType = typename Inner::SpaceType;
    using StateType = typename Inner::StateType;
    using ModelType = typename Inner::ModelType;
    using MethodType = typename Inner::MethodType;
    static constexpr bool Reentrant = Inner::Reentrant;

    FadingModes(Inner inner, Scalar alpha)
        : m_inner{std::move(inner)}
        , m_alpha{alpha}
    {}

    [[nodiscard]] ModelType const &
    model() const
    {
        return m_inner.model();
    }

    [[nodiscard]] ModelType &
    model()
    {
        return m_inner.model();
    }

    [[nodiscard]] MethodType const &
    method() const
    {
        return m_inner.method();
    }

    [[nodiscard]] Scalar
    alpha() const
    {
        return m_alpha;
    }

    // Mixes and predicts as the inner recursion does, then inflates each mode's covariance.
    template <typename... Args>
    void
    predict(StateType & state, Args const &... args)
    {
        m_inner.predict(state, args...);
        for (Eigen::Index m = 0; m < state.size(); ++m)
        {
            state.updateConditional(m, [&](auto & component) {
                component.setCovariance(m_alpha * m_alpha * component.covarianceLower());
            });
        }
    }

    // Updates as the inner recursion does, for a measurement or a switching measurement.
    template <typename M, typename Policy>
    [[nodiscard]] auto
    update(StateType & state, M const & measurement, Policy const & policy)
    {
        return m_inner.update(state, measurement, policy);
    }

    // The update into a result, which the inner recursion refills.
    template <typename M, typename Policy, typename Result>
    [[nodiscard]] auto
    update(StateType & state, M const & measurement, Policy const & policy, Result & result)
    {
        return m_inner.update(state, measurement, policy, result);
    }

private:

    Inner m_inner;
    Scalar m_alpha;
};

}  // namespace EmrysTest

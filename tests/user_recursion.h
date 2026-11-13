#pragma once

// A recursion of the user's own, for tests/Filters/kalman_filter.cpp and tests/Filters/gaussian_mixture_filter.cpp:
// FadingMemory wraps a KalmanRecursion and inflates each predicted covariance by alpha squared, a fading-memory
// filter that forgets old measurements, which the library doesn't ship. It meets the Recursion concept by its
// interface alone, with nothing of the library's specialized, so it runs inside a KalmanFilter and on each component
// of a MixtureFilter [filters-and-recursions], [user-extension-points]. usage.md, "Writing a recursion", shows it.

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

}  // namespace EmrysTest

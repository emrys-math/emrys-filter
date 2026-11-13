#pragma once

// The heap counter and the fixtures shared by the allocation tests (tests/allocation*.cpp, [allocation]):
// a constant-velocity state with a maneuver and sensor mode, a memory resource that counts its
// allocations, a guard that forbids Eigen's heap and counts the global one's, and the check that a
// step allocates nothing after its warm-up. allocation.cpp replaces the global operator new to
// feed the counter.

#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Models/Measurement.h>
#include <EmrysFilter/Models/ProcessModel.h>
#include <EmrysFilter/Spaces/Composite.h>
#include <EmrysFilter/Spaces/Coordinates.h>
#include <EmrysFilter/Spaces/Discrete.h>
#include <EmrysFilter/Spaces/Space.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef EIGEN_RUNTIME_NO_MALLOC
#    error "the allocation tests need EIGEN_RUNTIME_NO_MALLOC for the whole target"
#endif

namespace EmrysTest::Allocation
{

// Whether the replaced operator new counts, and what it has counted.
inline bool countingHeap = false;

inline int heapAllocations = 0;

using Emrys::additiveNoise;
using Emrys::compose;
using Emrys::Composite;
using Emrys::ConstantVelocity;
using Emrys::DiscreteProduct;
using Emrys::GaussianParams;
using Emrys::Linear;
using Emrys::Named;
using Emrys::ScalarOf;

namespace methods = Emrys::methods;

struct Position : Linear<2>
{};

struct Velocity : Linear<2>
{};

using State = Composite<Position, Velocity>;

struct Cruise
{};

struct Maneuvering
{};

struct Maneuver : Named<Cruise, Maneuvering>
{};

struct Inlier
{};

struct Outlier
{};

struct Sensor : Named<Inlier, Outlier>
{};

using Mode = DiscreteProduct<Maneuver, Sensor>;

using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Pmr = std::pmr::polymorphic_allocator<>;

constexpr double timeStep = 1.0;
constexpr Eigen::Index particleCount = 200;

inline auto
makeModel(double density = 0.1)
{
    return ConstantVelocity<Position, Velocity>{density};
}

inline GaussianParams<State>
makeInitial()
{
    Matrix4 cov = Matrix4::Zero();
    cov.diagonal() << 4.0, 4.0, 1.0, 1.0;
    cov(0, 2) = cov(2, 0) = 1.2;
    return {State{}, Vector4{0.0, 0.0, 1.0, 0.5}, cov};
}

// The model of a noisy fix of the block named by P, z = x + v, with noise of one component.
template <typename P = Position>
auto
makeFixModel()
{
    return Emrys::measurementModel(
        [](auto const & in, auto const &, auto const & x) {
            return Emrys::block<P>(in, x);
        },
        Emrys::reads(Emrys::path<P>),
        additiveNoise(GaussianParams<Linear<2>>{Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}}})
    );
}

// A noisy fix of the block named by P, z = x + v, by method, with any further arguments of
// observe(), such as a covariance update form.
template <typename P = Position, typename Method = methods::Linearized, typename... Args>
auto
makeFix(Eigen::Vector2d const & z, Method const & method = {}, Args const &... args)
{
    return observe(method, makeFixModel<P>(), z, args...);
}

// A noisy fix of the block named by P under noise of two components, an inlier and a wide outlier,
// so that an update has a run-time number of branches. The fix reads its observation from  z
// at each update, so one fix built outside a step serves every step: building the mixture
// allocates, which would hide what the filter allocates.
template <typename P = Position, typename Method = methods::Linearized>
auto
makeMixtureFix(Eigen::Vector2d const & z, Method const & method = {})
{
    auto noise = Emrys::GaussianMixtureParams<Linear<2>>{Linear<2>{}, Emrys::Capacity{2}};
    noise.add(std::log(0.9), Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1.0, 0.2}, {0.2, 0.8}});
    noise.add(std::log(0.1), Eigen::Vector2d::Zero(), Eigen::Matrix2d{{25.0, 0.0}, {0.0, 25.0}});
    return observe(
        method,
        Emrys::measurementModel(
            [](auto const & in, auto const &, auto const & x) {
                return Emrys::block<P>(in, x);
            },
            Emrys::reads(Emrys::path<P>),
            additiveNoise(std::move(noise))
        ),
        z
    );
}

// A target moving at the initial velocity, seen at step k.
inline Eigen::Vector2d
fixAt(int k)
{
    return Eigen::Vector2d{1.0, 0.5} * k + 0.8 * Eigen::Vector2d{std::cos(2.4 * k), std::sin(2.4 * k)};
}

// How many times a Propagating allocator of each identity, 0 to 3, has allocated, so that a test
// can tell which identity a container's storage came from.
inline std::array<std::size_t, 4> propagatingAllocations{};

// An allocator of the heap with an identity, 0 to 3, which compares equal only to one of the same
// identity and propagates on every assignment and swap: the kind whose state an assignment
// carries in, as an arena's allocator does.
template <typename T>
struct Propagating
{
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    Propagating() = default;

    explicit Propagating(int id)
        : id{id}
    {}

    template <typename U>
    Propagating(Propagating<U> const & other)
        : id{other.id}
    {}

    [[nodiscard]] T *
    allocate(std::size_t n)
    {
        ++propagatingAllocations.at(std::size_t(id));
        return std::allocator<T>{}.allocate(n);
    }

    void
    deallocate(T * p, std::size_t n)
    {
        std::allocator<T>{}.deallocate(p, n);
    }

    friend bool
    operator==(Propagating const & a, Propagating const & b)
    {
        return a.id == b.id;
    }

    int id = 0;
};

// A memory resource that counts its allocations, over a monotonic resource on a fixed buffer
// that throws rather than reaching past it.
class CountingResource : public std::pmr::memory_resource
{
public:

    explicit CountingResource(std::size_t bytes)
        : m_buffer(bytes)
        , m_arena{m_buffer.data(), m_buffer.size(), std::pmr::null_memory_resource()}
    {}

    [[nodiscard]] int
    allocations() const
    {
        return m_allocations;
    }

private:

    void *
    do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        ++m_allocations;
        return m_arena.allocate(bytes, alignment);
    }

    void
    do_deallocate(void * p, std::size_t bytes, std::size_t alignment) override
    {
        m_arena.deallocate(p, bytes, alignment);
    }

    [[nodiscard]] bool
    do_is_equal(std::pmr::memory_resource const & other) const noexcept override
    {
        return this == &other;
    }

    std::vector<std::byte> m_buffer;
    std::pmr::monotonic_buffer_resource m_arena;
    int m_allocations = 0;
};

// While it lives, Eigen may not allocate and the global heap's allocations are counted.
class HeapWatch
{
public:

    HeapWatch()
        : m_before{heapAllocations}
    {
        Eigen::internal::set_is_malloc_allowed(false);
        countingHeap = true;
    }

    HeapWatch(HeapWatch const &) = delete;

    HeapWatch &
    operator=(HeapWatch const &) = delete;

    ~HeapWatch()
    {
        countingHeap = false;
        Eigen::internal::set_is_malloc_allowed(true);
    }

    [[nodiscard]] int
    allocations() const
    {
        return heapAllocations - m_before;
    }

private:

    int m_before;
};

// Runs step(k) once to warm up, then a hundred times with Eigen's heap forbidden, and checks that
// neither the global heap nor resource, if given, was asked for memory after the warm-up.
template <typename Step>
void
expectStepsAllocateNothing(Step step, CountingResource const * resource = nullptr)
{
    step(0);
    int const reserved = resource ? resource->allocations() : 0;
    int heap = 0;
    {
        auto const watch = HeapWatch{};
        for (int k = 1; k <= 100; ++k)
        {
            step(k);
        }
        heap = watch.allocations();
    }
    EXPECT_EQ(heap, 0) << "allocations from the global heap";
    if (resource)
    {
        EXPECT_EQ(resource->allocations(), reserved) << "allocations from the resource after warm-up";
    }
}

constexpr std::size_t arenaBytes = 1 << 20;

// Checks that a non-empty pmr smoothing history copies and moves as an allocator-aware container
// does [allocation]: a copy keeps the source's resource, the allocator-extended copy and move take
// the one given, copying the steps when it differs, and each keeps the steps and the lag; one made
// from the allocator alone is empty and keeps every step. Then clear() empties \a history and keeps its lag.
template <typename History>
void
expectHistoryIsAllocatorAware(History & history)
{
    ASSERT_FALSE(history.empty());
    auto other = std::pmr::monotonic_buffer_resource{};
    Pmr const allocator = history.get_allocator();
    auto const expectSteps = [&](History const & copy, Pmr const & expected) {
        EXPECT_EQ(copy.get_allocator(), expected);
        EXPECT_EQ(copy.size(), history.size());
        EXPECT_EQ(copy.lag(), history.lag());
        EXPECT_NE(&copy.front(), &history.front());
        EXPECT_NE(&copy.back(), &history.back());
    };

    History const copy = history;
    expectSteps(copy, allocator);
    auto const elsewhere = History{history, Pmr{&other}};
    expectSteps(elsewhere, Pmr{&other});
    auto here = History{History{history}, allocator};
    expectSteps(here, allocator);
    auto const movedElsewhere = History{std::move(here), Pmr{&other}};
    expectSteps(movedElsewhere, Pmr{&other});
    auto const fresh = History{Pmr{&other}};
    EXPECT_TRUE(fresh.empty());
    EXPECT_FALSE(fresh.lag());

    auto const lag = history.lag();
    history.clear();
    EXPECT_TRUE(history.empty());
    EXPECT_EQ(history.lag(), lag);
}

// Predicts with \a filter over timeStep, or over its model's fixed step.
template <typename Filter, typename... Leading>
void
predictOverTheStep(Filter & filter, Leading &... leading)
{
    if constexpr (Emrys::isFixedStep<typename Filter::ModelType>)
    {
        filter.predict(leading...);
    }
    else
    {
        filter.predict(leading..., timeStep);
    }
}

}  // namespace EmrysTest::Allocation

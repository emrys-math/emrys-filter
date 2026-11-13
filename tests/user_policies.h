#pragma once

// Execution policies of the user's own, for tests/Core/execution.cpp and tests/parallel.cpp: each
// has the const forEachIndex(count, f) that the ExecutionPolicy concept admits [execution-policies].

#include <Eigen/Core>

#include <cstddef>
#include <thread>
#include <vector>

namespace EmrysTest
{

// Visits the indices in reverse, so that a result depending on the order of the calls would show.
struct ReversedPolicy
{
    template <typename F>
    void
    forEachIndex(Eigen::Index count, F const & f) const
    {
        for (Eigen::Index i = count - 1; i >= 0; --i)
        {
            f(i);
        }
    }
};

// Runs the indices on threads of its own, interleaved, as a pool might.
struct ThreadsPolicy
{
    int threads = 4;

    template <typename F>
    void
    forEachIndex(Eigen::Index count, F const & f) const
    {
        auto workers = std::vector<std::thread>{};
        workers.reserve(std::size_t(threads));
        for (int t = 0; t < threads; ++t)
        {
            workers.emplace_back([&, t] {
                for (Eigen::Index i = t; i < count; i += threads)
                {
                    f(i);
                }
            });
        }
        for (std::thread & worker : workers)
        {
            worker.join();
        }
    }
};

}  // namespace EmrysTest

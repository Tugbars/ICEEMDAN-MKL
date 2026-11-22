/**
 * EMD and EEMD benchmark (include/emd.hpp): the time of one decomposition
 * at a few signal lengths, and of a batch of channels.
 *
 *   emd_bench [threads]      (default: every logical processor)
 */

#include "emd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::vector<double> channels(int count, int n)
{
    // a few rhythms with drifting amplitude, and noise, different per channel
    std::vector<double> x(static_cast<size_t>(count) * n);
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    const double pi = 3.14159265358979323846;
    for (int c = 0; c < count; ++c)
        for (int t = 0; t < n; ++t)
        {
            const double s = static_cast<double>(t) / n;
            x[static_cast<size_t>(c) * n + t] = (1.0 + 0.3 * std::sin(2 * pi * (1 + c % 3) * s)) *
                                                    std::sin(2 * pi * (40 + c % 7) * s) +
                                                0.5 * std::sin(2 * pi * (8 + c % 5) * s + c) + 0.3 * g(rng);
        }
    return x;
}

template <class F>
static double best_ms(F &&f, int reps)
{
    double best = 1e300;
    for (int r = 0; r < reps; ++r)
    {
        const auto t0 = Clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return best;
}

int main(int argc, char **argv)
{
    emd::Config plain, ensemble;
    ensemble.ensemble_size = 100;
    if (argc > 1)
        plain.nthreads = ensemble.nthreads = std::atoi(argv[1]);
    emd::Decomposer emd_d(plain), eemd_d(ensemble);
    std::printf("threads: %d\n", emd_d.threads());
    std::printf("%-8s %14s %18s %22s\n", "N", "EMD, ms", "EEMD (100), ms", "EMD, 64 channels, ms");
    for (int n : {1024, 4096, 16384})
    {
        const std::vector<double> x = channels(64, n);
        emd_d.decompose(x.data(), n); // threads started, memory touched
        const double one = best_ms([&] { emd_d.decompose(x.data(), n); }, 7);
        const double many = best_ms([&] { eemd_d.decompose(x.data(), n); }, 3);
        const double batch = best_ms([&] { emd_d.decompose(x.data(), 64, n); }, 3);
        std::printf("%-8d %14.3f %18.2f %22.2f\n", n, one, many, batch);
        std::fflush(stdout);
    }
    return 0;
}

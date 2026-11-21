/**
 * Batch benchmark: many channels of one length, decomposed one after another
 * with decompose() (each on all threads) and in one decompose_batch() call
 * (the channels' trials interleaved on the threads). The noise bank is built
 * before timing, as it is kept between calls; both give the same results bit
 * for bit.
 *
 *   batch_bench [threads]      (default: the library's default thread count)
 *
 * Prints, for each length and channel count, the time per channel of both and
 * the ratio.
 */

#include "iceemdan.hpp"

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
    // EEG-like: a few rhythms with drifting amplitude, and noise, different per channel
    std::vector<double> x(static_cast<size_t>(count) * n);
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int c = 0; c < count; ++c)
        for (int t = 0; t < n; ++t)
        {
            const double s = static_cast<double>(t) / n;
            x[static_cast<size_t>(c) * n + t] = (1.0 + 0.3 * std::sin(2 * M_PI * (1 + c % 3) * s)) *
                                                    std::sin(2 * M_PI * (40 + c % 7) * s) +
                                                0.5 * std::sin(2 * M_PI * (8 + c % 5) * s + c) + 0.3 * g(rng);
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
    if (argc > 1)
        eemd::set_num_threads(std::atoi(argv[1]));
    std::printf("threads: %d\n", eemd::get_num_threads());
    std::printf("%-7s %-9s %16s %16s %8s\n", "N", "channels", "one by one, ms", "batch, ms", "ratio");
    std::printf("%-7s %-9s %16s %16s\n", "", "", "(per channel)", "(per channel)");
    for (int n : {1000, 2048, 4096})
    {
        for (int count : {8, 32, 64, 128})
        {
            if (static_cast<int64_t>(n) * count > 4096 * 64)
                continue;
            const std::vector<double> x = channels(count, n);
            eemd::ICEEMDAN d;
            std::vector<std::vector<double>> imfs;
            std::vector<double> res;
            d.decompose(x.data(), n, imfs, res); // the noise bank, kept from here on
            const int reps = n <= 1000 ? 3 : 2;
            const double single = best_ms([&] {
                for (int c = 0; c < count; ++c)
                    d.decompose(&x[static_cast<size_t>(c) * n], n, imfs, res);
            }, reps);
            std::vector<std::vector<std::vector<double>>> bimfs;
            std::vector<std::vector<double>> bres;
            const double batch = best_ms([&] { d.decompose_batch(x.data(), count, n, bimfs, bres); }, reps);
            std::printf("%-7d %-9d %16.2f %16.2f %7.2fx\n", n, count, single / count, batch / count, single / batch);
            std::fflush(stdout);
        }
    }
    return 0;
}

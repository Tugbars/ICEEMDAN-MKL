/**
 * Random number benchmark: the library's generator (eemd_rng.hpp) against
 * Intel MKL's VSL_BRNG_PHILOX4X32X10 with VSL_RNG_METHOD_GAUSSIAN_ICDF, which
 * the library used before. Through MKL's public interface only.
 *
 * Prints whether the raw streams are the same, how far the Gaussians are from
 * MKL's and from the exact inverse normal distribution function, and the time
 * per number, alone and as the library draws a trial's noise.
 */

#include "eemd_rng.hpp"

#include <mkl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static VSLStreamStatePtr mkl_stream(uint32_t seed, uint32_t stream)
{
    VSLStreamStatePtr s = nullptr;
    const unsigned int params[2] = {seed, stream};
    vslNewStreamEx(&s, VSL_BRNG_PHILOX4X32X10, 2, params);
    return s;
}

// exact Phi^-1(u) in long double, Halley's method on erf (middle) or erfc (tails)
static long double exact_inverse(long double u)
{
    const long double q = u - 0.5L;
    if (q == 0)
        return 0;
    const long double r2 = std::sqrt(2.0L), r2pi = std::sqrt(2.0L * 3.14159265358979323846264338327950288L);
    const bool lower = q < 0;
    const long double p = lower ? u : 1.0L - u;
    long double z = std::fabs(q) < 0.4L ? q * r2pi : (lower ? -1 : 1) * std::sqrt(-2.0L * std::log(p));
    for (int i = 0; i < 40; ++i)
    {
        const long double e = std::fabs(q) < 0.4L ? 0.5L * std::erf(z / r2) - q
                              : lower           ? 0.5L * std::erfc(-z / r2) - p
                                                : p - 0.5L * std::erfc(z / r2);
        const long double step = e / (std::exp(-0.5L * z * z) / r2pi);
        z -= step / (1.0L + 0.5L * z * step);
    }
    return z;
}

template <class F>
static double best_ns(F &&f, double per_call, int reps)
{
    f();
    double best = 1e300;
    for (int pass = 0; pass < 5; ++pass)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            f();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * per_call));
    }
    return best;
}

int main()
{
#ifdef EEMD_MKL_SINGLE_DYNAMIC_LIBRARY
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); // mkl_rt: no OpenMP runtime needed
#endif
    const uint32_t seed = 42, stream = 7;
    const int64_t n = 1 << 21;

    // ---- the raw streams
    std::vector<uint32_t> mine(n), theirs(n);
    eemd::rng::philox_words(seed, stream, 0, n, mine.data());
    VSLStreamStatePtr s = mkl_stream(seed, stream);
    viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, s, static_cast<MKL_INT>(n), theirs.data());
    vslDeleteStream(&s);
    int64_t same = 0;
    while (same < n && mine[same] == theirs[same])
        ++same;
    std::printf("Raw 32-bit words, stream (%u, %u): %lld of %lld the same as MKL's\n", seed, stream,
                static_cast<long long>(same), static_cast<long long>(n));

    // ---- the Gaussians
    std::vector<double> g(n), m(n);
    eemd::rng::gaussian(seed, stream, n, g.data());
    s = mkl_stream(seed, stream);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s, static_cast<MKL_INT>(n), m.data(), 0.0, 1.0);
    vslDeleteStream(&s);
    double vs_mkl = 0, mine_err = 0, mkl_err = 0;
    for (int64_t i = 0; i < n; ++i)
    {
        vs_mkl = std::max(vs_mkl, std::fabs(g[i] - m[i]));
        if (i % 8 == 0) // the exact values are slow: every eighth
        {
            const long double z = exact_inverse(eemd::rng::uniform(mine[i]));
            const double scale = std::max(1.0, std::fabs(static_cast<double>(z)));
            mine_err = std::max(mine_err, static_cast<double>(std::fabs(g[i] - z)) / scale);
            mkl_err = std::max(mkl_err, static_cast<double>(std::fabs(m[i] - z)) / scale);
        }
    }
    std::printf("Gaussians: largest difference from MKL's: %.3g\n", vs_mkl);
    std::printf("Largest error against the exact inverse (relative, or absolute below 1):\n");
    std::printf("  this library %.3g\n  MKL          %.3g\n\n", mine_err, mkl_err);

    // ---- time, one thread
    std::printf("Time per number, one thread (best of 5), ns:\n");
    const int64_t chunk = 1 << 16;
    std::vector<double> out(chunk);
    s = mkl_stream(seed, stream);
    std::printf("  %-44s %9s %9s\n", "", "library", "MKL");
    const double lib_one = best_ns([&] { eemd::rng::gaussian(seed, stream, chunk, out.data()); }, chunk, 100);
    const double mkl_one = best_ns([&] { vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s, chunk, out.data(), 0.0, 1.0); },
                                   chunk, 100);
    vslDeleteStream(&s);
    std::printf("  %-44s %9.3f %9.3f\n", "65,536 Gaussians in one call", lib_one, mkl_one);
    for (int len : {1024, 8192})
    {
        std::vector<double> o(len);
        uint32_t t = 0, u = 0;
        const double lib = best_ns([&] { eemd::rng::gaussian(seed, t++, len, o.data()); }, len, 2000);
        const double mkl = best_ns([&] {
            VSLStreamStatePtr st = mkl_stream(seed, u++);
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, st, len, o.data(), 0.0, 1.0);
            vslDeleteStream(&st);
        }, len, 2000);
        char what[64];
        std::snprintf(what, sizeof(what), "a trial's noise, N = %d (stream per trial)", len);
        std::printf("  %-44s %9.3f %9.3f\n", what, lib, mkl);
    }
    return 0;
}

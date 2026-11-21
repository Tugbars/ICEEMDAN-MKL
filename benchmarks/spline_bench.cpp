/**
 * Spline benchmark: the library's cubic spline (cubic_spline.hpp) against MKL
 * Data Fitting set up the same way (not-a-knot ends, evaluated at the samples
 * 0 .. n-1), on knot counts typical of the envelopes.
 *
 * Prints the largest difference between the two and the time of one envelope
 * (one construction and one evaluation) for each.
 */

#include "cubic_spline.hpp"

#include <mkl.h>
#include <mkl_df.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// One envelope with MKL Data Fitting: a task, the spline, its values, the task deleted
static bool mkl_envelope(const double *x, const double *y, int k, double *coeff, double *out, int n)
{
    DFTaskPtr task = nullptr;
    if (dfdNewTask1D(&task, k, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT) != DF_STATUS_OK)
        return false;
    bool ok = dfdEditPPSpline1D(task, DF_PP_CUBIC, DF_PP_NATURAL, k >= 4 ? DF_BC_NOT_A_KNOT : DF_BC_FREE_END,
                                nullptr, DF_NO_IC, nullptr, coeff, DF_NO_HINT) == DF_STATUS_OK &&
              dfdConstruct1D(task, DF_PP_SPLINE, DF_METHOD_STD) == DF_STATUS_OK;
    if (ok)
    {
        const double interval[2] = {0.0, static_cast<double>(n - 1)};
        const MKL_INT dorder[] = {1};
        ok = dfdInterpolate1D(task, DF_INTERP, DF_METHOD_PP, n, interval, DF_UNIFORM_PARTITION, 1, dorder,
                              nullptr, out, DF_NO_HINT, nullptr) == DF_STATUS_OK;
    }
    dfDeleteTask(&task);
    return ok;
}

int main()
{
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    eemd::CubicSpline spline;

    // ---- agreement on random knot sets
    double worst = 0;
    int sets = 0;
    for (int trial = 0; trial < 4000; ++trial)
    {
        const int k = 2 + static_cast<int>(rng() % (trial % 7 == 0 ? 400 : 60));
        const int n = 1 + static_cast<int>(rng() % 1500);
        std::vector<double> x(k), y(k), a(n), b(n), coeff(4 * k);
        const int max_step = 1 + static_cast<int>(rng() % 12);
        double pos = static_cast<double>(static_cast<int>(rng() % 41) - 30);
        double scale = 0;
        for (int i = 0; i < k; ++i)
        {
            x[i] = pos;
            pos += 1 + static_cast<double>(rng() % max_step);
            y[i] = u(rng);
            scale = std::max(scale, std::abs(y[i]));
        }
        if (!spline.construct(x.data(), y.data(), k) || !spline.evaluate_uniform(a.data(), n) ||
            !mkl_envelope(x.data(), y.data(), k, coeff.data(), b.data(), n))
            continue;
        ++sets;
        for (int j = 0; j < n; ++j)
            if (j >= x[0] && j <= x[k - 1])
                worst = std::max(worst, std::abs(a[j] - b[j]) / scale);
    }
    std::printf("Agreement with MKL Data Fitting on %d random knot sets (2 to 401 knots):\n", sets);
    std::printf("  largest difference between the knots: %.2e of the data\n\n", worst);

    // ---- time of one envelope at 1024 and 4096 samples
    std::printf("Time of one envelope (construct + evaluate), microseconds\n");
    std::printf("  %-9s %-7s %10s %10s %8s\n", "samples", "knots", "MKL", "library", "ratio");
    for (int n : {1024, 4096})
    {
        for (int k : {n / 3, n / 6, n / 12, n / 24, n / 48, n / 96})
        {
            std::vector<double> x(k), y(k), out(n), coeff(4 * k);
            for (int i = 0; i < k; ++i)
            {
                x[i] = std::floor(-3.0 + i * ((n + 6.0) / (k - 1)));
                y[i] = u(rng);
            }
            for (int i = 1; i < k; ++i)
                if (x[i] <= x[i - 1])
                    x[i] = x[i - 1] + 1;
            const int reps = 400000 / (n / 256);
            double sink = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r)
            {
                y[r % k] += 1e-9;
                mkl_envelope(x.data(), y.data(), k, coeff.data(), out.data(), n);
                sink += out[r % n];
            }
            const auto t1 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r)
            {
                y[r % k] += 1e-9;
                spline.construct(x.data(), y.data(), k);
                spline.evaluate_uniform(out.data(), n);
                sink += out[r % n];
            }
            const auto t2 = std::chrono::steady_clock::now();
            const double mkl_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
            const double lib_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
            std::printf("  %-9d %-7d %10.3f %10.3f %7.1fx%s\n", n, k, mkl_us, lib_us, mkl_us / lib_us,
                        sink == 12345.678 ? "!" : "");
        }
    }
    return 0;
}

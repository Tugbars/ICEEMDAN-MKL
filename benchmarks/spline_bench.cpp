/**
 * Spline benchmark: the library's envelope splines (cubic_spline.hpp) against
 * MKL Data Fitting set up the same way, evaluated at the samples 0 .. n-1:
 * the cubic spline with not-a-knot ends, Akima's spline, straight lines.
 * Built only when MKL is found; the library itself does not use MKL.
 *
 * Prints the largest difference between the two for each kind and the time
 * of one envelope (one construction and one evaluation).
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

enum class Kind
{
    Cubic,
    Akima,
    Linear
};

// One envelope with MKL Data Fitting: a task, the spline, its values, the task deleted
static bool mkl_envelope(Kind kind, const double *x, const double *y, int k, double *coeff, double *out, int n)
{
    DFTaskPtr task = nullptr;
    if (dfdNewTask1D(&task, k, x, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT) != DF_STATUS_OK)
        return false;
    MKL_INT order = DF_PP_CUBIC, type = DF_PP_NATURAL, bc = k >= 4 ? DF_BC_NOT_A_KNOT : DF_BC_FREE_END;
    if (kind == Kind::Akima)
    {
        type = DF_PP_AKIMA; // order cubic, type Akima
        bc = DF_NO_BC;
    }
    else if (kind == Kind::Linear)
    {
        order = DF_PP_LINEAR;
        type = DF_PP_DEFAULT;
        bc = DF_NO_BC;
    }
    bool ok = dfdEditPPSpline1D(task, order, type, bc, nullptr, DF_NO_IC, nullptr, coeff, DF_NO_HINT) == DF_STATUS_OK &&
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

static bool library_envelope(eemd::CubicSpline &s, Kind kind, const double *x, const double *y, int k, double *out,
                             int n)
{
    const bool built = kind == Kind::Cubic   ? s.construct(x, y, k)
                       : kind == Kind::Akima ? s.construct_akima(x, y, k)
                                             : s.construct_linear(x, y, k);
    return built && s.evaluate_uniform(out, n);
}

int main()
{
#ifdef EEMD_MKL_SINGLE_DYNAMIC_LIBRARY
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); // mkl_rt: no OpenMP runtime needed
#endif
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    eemd::CubicSpline spline;
    const char *names[] = {"cubic (not-a-knot)", "Akima", "linear"};

    // ---- agreement on random knot sets
    std::printf("Agreement with MKL Data Fitting on random knot sets (5 to 404 knots),\n");
    std::printf("largest difference between the knots, relative to the data:\n");
    for (Kind kind : {Kind::Cubic, Kind::Akima, Kind::Linear})
    {
        double inner = 0, ends = 0;
        int sets = 0;
        for (int trial = 0; trial < 4000; ++trial)
        {
            const int k = 5 + static_cast<int>(rng() % (trial % 7 == 0 ? 400 : 60));
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
            if (!library_envelope(spline, kind, x.data(), y.data(), k, a.data(), n) ||
                !mkl_envelope(kind, x.data(), y.data(), k, coeff.data(), b.data(), n))
                continue;
            ++sets;
            for (int j = 0; j < n; ++j)
            {
                if (j < x[0] || j > x[k - 1])
                    continue;
                // the end pieces, from the first knot to the third and from the third-last on
                const bool end = j < x[2] || j > x[k - 3];
                double &worst = end ? ends : inner;
                worst = std::max(worst, std::abs(a[j] - b[j]) / scale);
            }
        }
        std::printf("  %-20s %4d sets: inner pieces %.2e, end pieces %.2e\n", names[static_cast<int>(kind)], sets,
                    inner, ends);
    }

    // ---- time of one envelope
    std::printf("\nTime of one envelope (construct + evaluate), microseconds\n");
    std::printf("  %-20s %-9s %-7s %10s %10s %8s\n", "", "samples", "knots", "MKL", "library", "ratio");
    for (Kind kind : {Kind::Cubic, Kind::Akima, Kind::Linear})
    {
        for (int n : {1024, 4096})
        {
            for (int k : {n / 6, n / 24, n / 96})
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
                    mkl_envelope(kind, x.data(), y.data(), k, coeff.data(), out.data(), n);
                    sink += out[r % n];
                }
                const auto t1 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r)
                {
                    y[r % k] += 1e-9;
                    library_envelope(spline, kind, x.data(), y.data(), k, out.data(), n);
                    sink += out[r % n];
                }
                const auto t2 = std::chrono::steady_clock::now();
                const double mkl_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
                const double lib_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
                std::printf("  %-20s %-9d %-7d %10.3f %10.3f %7.1fx%s\n", names[static_cast<int>(kind)], n, k, mkl_us,
                            lib_us, mkl_us / lib_us, sink == 12345.678 ? "!" : "");
            }
        }
    }
    return 0;
}

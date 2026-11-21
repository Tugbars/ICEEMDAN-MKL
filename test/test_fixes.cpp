/**
 * Regression tests for the reproducibility, diagnostics, extrema and input
 * sanitizing fixes. Built with the project's own flags (-ffast-math on
 * GCC/Clang), which is what the NaN test needs to exercise.
 *
 * Exits non-zero on any failure.
 */

#include "iceemdan_mkl.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %-66s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

static std::vector<double> test_signal(int n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> x(n);
    double p = 100.0;
    for (int i = 0; i < n; ++i)
    {
        const double t = double(i) / n;
        p += 0.05 * g(rng);
        x[i] = p + 3.0 * std::sin(2 * M_PI * 9 * t) + 1.0 * std::sin(2 * M_PI * 61 * t);
    }
    return x;
}

struct Run
{
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    eemd::DecompositionDiagnostics diag;
    bool ok = false;
};

static Run run(const std::vector<double> &x, eemd::ProcessingMode mode,
               eemd::LocalMeanMethod lm, bool adaptive, int threads)
{
    eemd::set_num_threads(threads);
    eemd::ICEEMDAN d(mode);
    d.config().ensemble_size = 50;
    d.config().local_mean_method = lm;
    d.config().adaptive_ensemble = adaptive;
    Run r;
    r.ok = d.decompose_with_diagnostics(x.data(), static_cast<int>(x.size()), r.imfs, r.residue, r.diag);
    return r;
}

static bool bitwise_equal(const Run &a, const Run &b)
{
    if (a.imfs.size() != b.imfs.size())
        return false;
    for (size_t k = 0; k < a.imfs.size(); ++k)
        if (std::memcmp(a.imfs[k].data(), b.imfs[k].data(), a.imfs[k].size() * sizeof(double)) != 0)
            return false;
    return std::memcmp(a.residue.data(), b.residue.data(), a.residue.size() * sizeof(double)) == 0;
}

int main()
{
    const std::vector<double> x = test_signal(1500, 7);
    const int n = static_cast<int>(x.size());

    // ------------------------------------------------------------------
    std::printf("ICEEMDAN: identical results at 1, 2, 3 and 4 threads\n");
    const struct
    {
        eemd::ProcessingMode mode;
        const char *name;
    } modes[] = {{eemd::ProcessingMode::Standard, "Standard"},
                 {eemd::ProcessingMode::Finance, "Finance"},
                 {eemd::ProcessingMode::Scientific, "Scientific"}};
    for (const auto &m : modes)
    {
        for (eemd::LocalMeanMethod lm : {eemd::LocalMeanMethod::Sifted, eemd::LocalMeanMethod::SingleEnvelope})
        {
            for (bool adaptive : {false, true})
            {
                const Run ref = run(x, m.mode, lm, adaptive, 1);
                bool same = ref.ok;
                for (int t : {2, 3, 4})
                    same = same && bitwise_equal(ref, run(x, m.mode, lm, adaptive, t));
                char label[128];
                std::snprintf(label, sizeof(label), "%-10s %-14s adaptive=%d (%zu IMFs)", m.name,
                              lm == eemd::LocalMeanMethod::Sifted ? "Sifted" : "SingleEnvelope",
                              adaptive ? 1 : 0, ref.imfs.size());
                check(same, label);
            }
        }
    }

    // ------------------------------------------------------------------
    std::printf("ICEEMDAN: completeness and diagnostics\n");
    {
        const Run r = run(x, eemd::ProcessingMode::Standard, eemd::LocalMeanMethod::Sifted, false, 2);
        double err = 0, amax = 0;
        for (int i = 0; i < n; ++i)
        {
            double s = r.residue[i];
            for (const auto &imf : r.imfs)
                s += imf[i];
            err = std::max(err, std::abs(s - x[i]));
            amax = std::max(amax, std::abs(x[i]));
        }
        check(r.ok && err / amax < 1e-12, "sum(IMFs) + residue == signal (relative 1e-12)");

        const size_t K = r.imfs.size();
        check(r.diag.sift_iterations.size() == K && r.diag.mean_sift_iterations.size() == K &&
                  r.diag.nonconverged_trials.size() == K && r.diag.convergence_flags.size() == K &&
                  r.diag.trials_per_imf.size() == K,
              "one diagnostics entry per IMF");
        bool iters_ok = true, trials_ok = true;
        for (size_t k = 0; k < K; ++k)
        {
            iters_ok = iters_ok && r.diag.sift_iterations[k] >= 1 &&
                       r.diag.mean_sift_iterations[k] >= 1.0 &&
                       r.diag.mean_sift_iterations[k] <= r.diag.sift_iterations[k];
            trials_ok = trials_ok && r.diag.trials_per_imf[k] >= 1 && r.diag.trials_per_imf[k] <= 50;
        }
        check(iters_ok, "Sifted: 1 <= mean iterations <= max iterations, every IMF");
        check(trials_ok, "trials per IMF within 1..ensemble_size");
        check(r.diag.noise_bank_multiplier == 0, "default: independent noise realizations (multiplier 0)");

        const Run s = run(x, eemd::ProcessingMode::Standard, eemd::LocalMeanMethod::SingleEnvelope, false, 2);
        bool one = s.ok;
        for (size_t k = 0; k < s.imfs.size(); ++k)
            one = one && s.diag.sift_iterations[k] == 1 && s.diag.convergence_flags[k];
        check(one, "SingleEnvelope: exactly one iteration per trial");

        // max_sift_iters = 1 cannot converge the SD criterion on the first IMF
        eemd::set_num_threads(2);
        eemd::ICEEMDAN d;
        d.config().ensemble_size = 20;
        d.config().max_sift_iters = 1;
        d.config().s_number = 0;
        std::vector<std::vector<double>> imfs;
        std::vector<double> res;
        eemd::DecompositionDiagnostics diag;
        d.decompose_with_diagnostics(x.data(), n, imfs, res, diag);
        check(!diag.nonconverged_trials.empty() && diag.nonconverged_trials[0] > 0 && !diag.convergence_flags[0],
              "max_sift_iters = 1: non-converged trials are reported");
    }

    // ------------------------------------------------------------------
    std::printf("ICEEMDAN: noise bank memory cap\n");
    {
        eemd::set_num_threads(2);
        eemd::ICEEMDAN d;
        d.config().ensemble_size = 20;
        std::vector<std::vector<double>> imfs;
        std::vector<double> res;
        const size_t per_unit = size_t(d.config().max_imfs) * n * sizeof(double);

        d.config().use_circular_bank = true;
        check(d.decompose(x.data(), n, imfs, res) && d.last_diagnostics().noise_bank_multiplier == 20,
              "use_circular_bank: circular bank with multiplier 20");
        d.config().use_circular_bank = false;

        // independent bank needs 10 trial units (20 trials, antithetic); allow 6
        d.config().noise_bank_max_bytes = per_unit * 6;
        const bool ok6 = d.decompose(x.data(), n, imfs, res);
        check(ok6 && d.last_diagnostics().noise_bank_multiplier == 6,
              "over the cap: falls back to the circular bank, multiplier lowered to 6");

        d.config().noise_bank_max_bytes = per_unit * 3; // below the floor of 4
        check(!d.decompose(x.data(), n, imfs, res), "cap below the floor: decompose() fails cleanly");
    }

    // ------------------------------------------------------------------
    std::printf("EEMD: repeatable at a fixed thread count, stable across counts\n");
    {
        eemd::EEMDConfig cfg;
        cfg.ensemble_size = 24;
        eemd::EEMD e(cfg);
        std::vector<std::vector<double>> a, b, c;
        int32_t na = 0, nb = 0, nc = 0;
        eemd::set_num_threads(2);
        e.decompose(x.data(), n, a, na);
        e.decompose(x.data(), n, b, nb);
        bool same = na == nb;
        for (int k = 0; same && k < na; ++k)
            same = std::memcmp(a[k].data(), b[k].data(), n * sizeof(double)) == 0;
        check(same, "two runs at 2 threads are bit-identical");

        eemd::set_num_threads(1);
        e.decompose(x.data(), n, c, nc);
        double diff = 0, amax = 0;
        bool count_ok = nc == na;
        for (int k = 0; count_ok && k < na; ++k)
            for (int i = 0; i < n; ++i)
            {
                diff = std::max(diff, std::abs(a[k][i] - c[k][i]));
                amax = std::max(amax, std::abs(a[k][i]));
            }
        check(count_ok && diff / amax < 1e-10, "1 vs 2 threads agree to summation rounding (1e-10)");
    }

    // ------------------------------------------------------------------
    std::printf("Extrema: flat peaks and troughs\n");
    {
        // two flat maxima, one flat minimum, one strict maximum; plateaus at the
        // ends are not extrema
        const double s[] = {5, 5, 1, 3, 3, 3, 2, 0, 0, 2, 4, 1, 1};
        const int m = sizeof(s) / sizeof(s[0]);
        int32_t idx[16], cnt = 0;
        eemd::find_maxima_raw(s, m, idx, cnt);
        check(cnt == 2 && idx[0] == 4 && idx[1] == 10, "maxima at plateau centre (4) and strict peak (10)");
        eemd::find_minima_raw(s, m, idx, cnt);
        check(cnt == 2 && idx[0] == 2 && idx[1] == 7, "minima at strict trough (2) and plateau [7, 8] (7)");
        check(eemd::count_extrema(s, m) == 4, "count_extrema agrees");

        // prices on a tick grid: strict comparisons would find almost nothing
        std::vector<double> ticks(400);
        for (int i = 0; i < 400; ++i)
            ticks[i] = std::round(20.0 * std::sin(2 * M_PI * i / 50.0)) * 0.25;
        std::vector<int32_t> big(400);
        eemd::find_maxima_raw(ticks.data(), 400, big.data(), cnt);
        check(cnt == 8, "tick-quantized sine (8 periods): 8 maxima");
    }

    // ------------------------------------------------------------------
    std::printf("Input sanitizing under the project's fast-math flags\n");
    {
        std::vector<double> y(x);
        y[10] = std::numeric_limits<double>::quiet_NaN();
        y[11] = std::numeric_limits<double>::infinity();
        y[12] = -std::numeric_limits<double>::infinity();
        y[0] = std::numeric_limits<double>::quiet_NaN();
        y[n - 1] = std::numeric_limits<double>::quiet_NaN();
        const int bad = eemd::sanitize_signal(y.data(), n);
        bool finite = true;
        for (double v : y)
            finite = finite && eemd::is_finite_bits(v);
        check(bad == 5 && finite, "5 NaN/Inf found and replaced");
        const double expect = x[9] + (x[13] - x[9]) * 0.25;
        check(std::abs(y[10] - expect) < 1e-12, "interior run linearly interpolated");
        check(y[0] == x[1] && y[n - 1] == x[n - 2], "edge samples extrapolated from the nearest finite one");

        eemd::set_num_threads(2);
        eemd::ICEEMDAN d(eemd::ProcessingMode::Finance);
        d.config().ensemble_size = 20;
        std::vector<std::vector<double>> imfs;
        std::vector<double> res;
        eemd::DecompositionDiagnostics diag;
        std::vector<double> z(x);
        z[500] = std::numeric_limits<double>::quiet_NaN();
        const bool ok = d.decompose_with_diagnostics(z.data(), n, imfs, res, diag);
        bool clean = ok;
        for (const auto &imf : imfs)
            for (double v : imf)
                clean = clean && eemd::is_finite_bits(v);
        check(ok && clean && diag.nan_count == 1, "Finance mode decomposes a series with a NaN");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

/**
 * Regression tests for the reproducibility, diagnostics, extrema and input
 * sanitizing fixes. Built with the project's own flags (-ffast-math on
 * GCC/Clang), which is what the NaN test needs to exercise.
 *
 * Exits non-zero on any failure.
 */

#include "iceemdan.hpp"

#include <algorithm>
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
        // the bank holds about log2(n) modes per realization, as the library sizes it
        int32_t bank_modes = 2;
        for (int32_t len = n; len > 1; len >>= 1)
            ++bank_modes;
        bank_modes = std::min(bank_modes, d.config().max_imfs);
        const size_t per_unit = size_t(bank_modes) * n * sizeof(double);

        check(d.noise_bank_bytes(n) == per_unit * 20, "noise_bank_bytes(): 20 realizations of that many modes");

        // the independent bank needs 20 units (20 realizations): allow 20, then 19
        d.config().noise_bank_max_bytes = per_unit * 20;
        check(d.decompose(x.data(), n, imfs, res) && d.last_diagnostics().noise_bank_multiplier == 0,
              "at the cap: independent realizations");
        d.config().noise_bank_max_bytes = per_unit * 19;
        check(!d.decompose(x.data(), n, imfs, res),
              "over the cap: decompose() fails; the library does not switch to the circular bank itself");

        d.config().noise_bank_max_bytes = size_t(2) << 30;
        d.config().use_circular_bank = true;
        check(d.decompose(x.data(), n, imfs, res) && d.last_diagnostics().noise_bank_multiplier == 20,
              "use_circular_bank: circular bank with multiplier 20");
        d.config().noise_bank_max_bytes = per_unit * 6;
        check(d.decompose(x.data(), n, imfs, res) && d.last_diagnostics().noise_bank_multiplier == 6,
              "circular bank over the cap: multiplier lowered to 6");
        d.config().noise_bank_max_bytes = per_unit * 3; // below the floor of 4
        check(!d.decompose(x.data(), n, imfs, res), "circular bank, cap below the floor: decompose() fails cleanly");
    }

    // ------------------------------------------------------------------
    std::printf("ICEEMDAN: the noise bank kept between calls\n");
    {
        eemd::set_num_threads(2);
        const std::vector<double> other = test_signal(n, 99); // same length, another signal
        auto decompose = [&](eemd::ICEEMDAN &d, const std::vector<double> &s)
        {
            Run r;
            r.ok = d.decompose_with_diagnostics(s.data(), static_cast<int>(s.size()), r.imfs, r.residue, r.diag);
            return r;
        };
        auto fresh = [&](const std::vector<double> &s, uint32_t seed)
        {
            eemd::ICEEMDAN d;
            d.config().ensemble_size = 20;
            d.config().rng_seed = seed;
            d.config().keep_noise_bank = false;
            return decompose(d, s);
        };

        eemd::ICEEMDAN d;
        d.config().ensemble_size = 20;
        const Run first = decompose(d, x);
        const Run second = decompose(d, x);
        check(first.ok && !first.diag.noise_bank_reused && second.diag.noise_bank_reused &&
                  bitwise_equal(first, second),
              "second call: the bank is used again, the result is the same");

        const Run third = decompose(d, other);
        check(third.diag.noise_bank_reused && bitwise_equal(third, fresh(other, d.config().rng_seed)),
              "another signal of that length: bank used again, same as a new decomposer");

        d.config().rng_seed = 1234;
        const Run reseeded = decompose(d, x);
        check(!reseeded.diag.noise_bank_reused && bitwise_equal(reseeded, fresh(x, 1234)),
              "another seed: the bank is built again");

        const std::vector<double> shorter(x.begin(), x.begin() + 900);
        const Run cut = decompose(d, shorter);
        check(!cut.diag.noise_bank_reused && bitwise_equal(cut, fresh(shorter, 1234)),
              "another length: the bank is built again");

        d.release_noise_bank();
        const Run released = decompose(d, shorter);
        check(!released.diag.noise_bank_reused && bitwise_equal(released, cut),
              "release_noise_bank(): built again, same result");

        d.config().keep_noise_bank = false;
        const Run a = decompose(d, shorter); // uses the bank kept so far, then frees it
        const Run b = decompose(d, shorter);
        check(a.diag.noise_bank_reused && !b.diag.noise_bank_reused && bitwise_equal(a, cut) && bitwise_equal(b, cut),
              "keep_noise_bank = false: freed after the call, same result");
    }

    // ------------------------------------------------------------------
    std::printf("ICEEMDAN: decompose_batch, signal by signal as decompose()\n");
    {
        // 13 channels of 900 samples: different signals, some noisier (other IMF counts)
        const int count = 13, len = 900;
        std::vector<double> channels(static_cast<size_t>(count) * len);
        for (int s = 0; s < count; ++s)
        {
            const std::vector<double> c = test_signal(len, 200 + s);
            std::mt19937 rng(s);
            std::normal_distribution<double> g(0.0, 0.3 * (s % 4));
            for (int j = 0; j < len; ++j)
                channels[static_cast<size_t>(s) * len + j] = c[j] + g(rng);
        }
        auto one_by_one = [&](int threads)
        {
            eemd::set_num_threads(threads);
            eemd::ICEEMDAN d;
            d.config().ensemble_size = 24;
            std::vector<Run> out(count);
            for (int s = 0; s < count; ++s)
                out[s].ok = d.decompose_with_diagnostics(&channels[static_cast<size_t>(s) * len], len, out[s].imfs,
                                                         out[s].residue, out[s].diag);
            return out;
        };
        auto batch = [&](int threads, int n_signals, std::vector<eemd::DecompositionDiagnostics> *diags = nullptr)
        {
            eemd::set_num_threads(threads);
            eemd::ICEEMDAN d;
            d.config().ensemble_size = 24;
            std::vector<std::vector<std::vector<double>>> imfs;
            std::vector<std::vector<double>> res;
            const bool ok = d.decompose_batch(channels.data(), n_signals, len, imfs, res, diags);
            std::vector<Run> out(n_signals);
            for (int s = 0; s < n_signals; ++s)
            {
                out[s].ok = ok;
                out[s].imfs = imfs[s];
                out[s].residue = res[s];
            }
            return out;
        };
        const std::vector<Run> alone = one_by_one(4);
        bool counts_differ = false;
        for (int s = 1; s < count; ++s)
            counts_differ = counts_differ || alone[s].imfs.size() != alone[0].imfs.size();

        for (int threads : {1, 3, 8})
        {
            std::vector<eemd::DecompositionDiagnostics> diags;
            const std::vector<Run> b = batch(threads, count, &diags);
            bool same = true, diag_ok = diags.size() == static_cast<size_t>(count);
            for (int s = 0; s < count; ++s)
            {
                same = same && b[s].ok && bitwise_equal(b[s], alone[s]);
                diag_ok = diag_ok && diags[s].valid && diags[s].sift_iterations.size() == b[s].imfs.size() &&
                          diags[s].orthogonality_index == alone[s].diag.orthogonality_index;
            }
            char label[96];
            std::snprintf(label, sizeof(label), "13 channels on %d thread%s: each as decompose() alone", threads,
                          threads == 1 ? "" : "s");
            check(same, label);
            if (threads == 3)
                check(diag_ok, "diagnostics per channel, as decompose_with_diagnostics() gives them");
        }
        check(counts_differ, "the channels do not all have the same number of IMFs (the test covers that)");

        // fewer signals than threads: one after another, each on all threads
        const std::vector<Run> few = batch(8, 3);
        check(bitwise_equal(few[0], alone[0]) && bitwise_equal(few[2], alone[2]),
              "3 channels on 8 threads: each as decompose() alone");

        // the bank: built once for the batch, then kept for a single decomposition of that length
        eemd::set_num_threads(4);
        eemd::ICEEMDAN d;
        d.config().ensemble_size = 24;
        std::vector<std::vector<std::vector<double>>> imfs;
        std::vector<std::vector<double>> res;
        std::vector<eemd::DecompositionDiagnostics> diags;
        d.decompose_batch(channels.data(), count, len, imfs, res, &diags);
        Run single;
        single.ok = d.decompose_with_diagnostics(channels.data(), len, single.imfs, single.residue, single.diag);
        check(!diags[0].noise_bank_reused && single.diag.noise_bank_reused,
              "the batch builds the noise bank once; a later call of that length keeps it");

        check(!d.decompose_batch(channels.data(), 0, len, imfs, res) && !d.decompose_batch(nullptr, 2, len, imfs, res) &&
                  !d.decompose_batch(channels.data(), 2, 3, imfs, res),
              "no signals, no data, or signals of 3 samples: refused");
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
        // a plateau of even length: the later of its two middle samples, as emd.m takes it
        check(cnt == 2 && idx[0] == 2 && idx[1] == 8, "minima at strict trough (2) and plateau [7, 8] (8)");
        check(eemd::count_extrema(s, m) == 4, "count_extrema agrees");

        // prices on a tick grid: strict comparisons would find almost nothing
        std::vector<double> ticks(400);
        for (int i = 0; i < 400; ++i)
            ticks[i] = std::round(20.0 * std::sin(2 * M_PI * i / 50.0)) * 0.25;
        std::vector<int32_t> big(400);
        eemd::find_maxima_raw(ticks.data(), 400, big.data(), cnt);
        check(cnt == 8, "tick-quantized sine (8 periods): 8 maxima");

        // find_extrema_raw (both kinds in one pass) against the two separate scans,
        // on noise, on noise rounded to a grid (flat runs) and at every short length
        std::mt19937 rng(5);
        std::normal_distribution<double> g(0.0, 1.0);
        bool same = true;
        for (int trial = 0; trial < 400 && same; ++trial)
        {
            const int len = trial < 80 ? trial : 80 + static_cast<int>(rng() % 3000);
            std::vector<double> v(len);
            for (double &e : v)
                e = (trial % 3 == 1) ? std::round(2.0 * g(rng)) : g(rng);
            std::vector<int32_t> mx(len / 2 + 2), mn(len / 2 + 2), mx2(len / 2 + 2), mn2(len / 2 + 2);
            int32_t n_mx = 0, n_mn = 0, n_mx2 = -1, n_mn2 = -1;
            eemd::find_maxima_raw(v.data(), len, mx.data(), n_mx);
            eemd::find_minima_raw(v.data(), len, mn.data(), n_mn);
            eemd::find_extrema_raw(v.data(), len, mx2.data(), n_mx2, mn2.data(), n_mn2);
            same = n_mx == n_mx2 && n_mn == n_mn2 &&
                   std::equal(mx.begin(), mx.begin() + n_mx, mx2.begin()) &&
                   std::equal(mn.begin(), mn.begin() + n_mn, mn2.begin());
        }
        check(same, "find_extrema_raw gives the indices of the two separate scans");
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

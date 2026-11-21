/**
 * Hilbert-Huang analysis tests (eemd_hht.hpp): instantaneous amplitude, phase
 * and frequency of signals whose values are known exactly (a tone and an
 * amplitude-modulated tone, both periodic over the record) and of a chirp;
 * the Hilbert spectrum and the marginal spectrum of two tones, with time
 * columns, frequency limits and amplitude instead of energy; the spectrum's
 * sum against the attributes it is made of; the same result at any thread
 * count; refused settings.
 *
 * Exits non-zero on any failure.
 */

#include "eemd_hht.hpp"
#include "eemd_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_failures = 0;
using eemd::hht::two_pi;

static void check(bool ok, const char *what, double value, double limit)
{
    std::printf("  %-64s %s  %.2e (limit %.0e)\n", what, ok ? "PASS" : "FAIL", value, limit);
    if (!ok)
        ++g_failures;
}

static void check(bool ok, const char *what)
{
    std::printf("  %-64s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

int main()
{
    using namespace eemd::hht;

    std::printf("Instantaneous attributes\n");
    {
        // a tone of 25 cycles in 1000 samples at 250 Hz: 6.25 Hz, amplitude 1.5
        const int n = 1000;
        const double fs = 250.0, f0 = 6.25, A = 1.5, phi0 = 0.7;
        std::vector<double> x(n), a(n), f(n), p(n);
        for (int t = 0; t < n; ++t)
            x[t] = A * std::cos(two_pi * f0 * t / fs + phi0);
        const bool ok = instantaneous(x.data(), n, fs, a.data(), f.data(), p.data());
        double ea = 0, ef = 0, ep = 0;
        for (int t = 0; t < n; ++t)
        {
            ea = std::max(ea, std::abs(a[t] - A));
            ef = std::max(ef, std::abs(f[t] - f0));
            ep = std::max(ep, std::abs(p[t] - (phi0 + two_pi * f0 * t / fs)));
        }
        check(ok && ea < 1e-10, "tone: amplitude 1.5 at every sample", ea, 1e-10);
        check(ok && ef < 1e-9, "tone: frequency 6.25 Hz at every sample", ef, 1e-9);
        check(ok && ep < 1e-9, "tone: phase 0.7 + 2 pi f t, unwrapped", ep, 1e-9);
        check(std::abs(mean_frequency(x.data(), n, fs) - f0) < 1e-9, "tone: mean frequency 6.25 Hz",
              std::abs(mean_frequency(x.data(), n, fs) - f0), 1e-9);

        // a carrier of 100 cycles modulated by 5 cycles in 2000 samples (fs = 1):
        // the envelope and the carrier frequency come back exactly
        const int m = 2000;
        std::vector<double> y(m), env(m), b(m), g(m);
        for (int t = 0; t < m; ++t)
        {
            env[t] = 1.0 + 0.5 * std::cos(two_pi * 5.0 * t / m);
            y[t] = env[t] * std::cos(two_pi * 100.0 * t / m);
        }
        const bool ok2 = instantaneous(y.data(), m, 1.0, b.data(), g.data(), nullptr);
        double eb = 0, eg = 0;
        for (int t = 0; t < m; ++t)
        {
            eb = std::max(eb, std::abs(b[t] - env[t]));
            eg = std::max(eg, std::abs(g[t] - 0.05));
        }
        check(ok2 && eb < 1e-9, "modulated tone: amplitude = the envelope", eb, 1e-9);
        check(ok2 && eg < 1e-9, "modulated tone: frequency = the carrier, 0.05 cycles/sample", eg, 1e-9);

        // a chirp from 0.02 to 0.12 cycles per sample: tracked away from the ends
        const int c = 4096;
        std::vector<double> z(c), h(c);
        for (int t = 0; t < c; ++t)
            z[t] = std::cos(two_pi * (0.02 * t + 0.5 * (0.10 / c) * t * t));
        instantaneous(z.data(), c, 1.0, nullptr, h.data(), nullptr);
        double eh = 0;
        for (int t = c / 10; t < c - c / 10; ++t)
            eh = std::max(eh, std::abs(h[t] - (0.02 + 0.10 * t / c)));
        check(eh < 2e-3, "chirp 0.02 -> 0.12: frequency followed (middle 80 %)", eh, 2e-3);

        // the sample rate only scales the frequency
        std::vector<double> f1(n), f2(n);
        instantaneous(x.data(), n, 1.0, nullptr, f1.data(), nullptr);
        instantaneous(x.data(), n, 250.0, nullptr, f2.data(), nullptr);
        double es = 0;
        for (int t = 0; t < n; ++t)
            es = std::max(es, std::abs(f2[t] - 250.0 * f1[t]));
        check(es < 1e-9, "frequency in Hz = cycles per sample x sample rate", es, 1e-9);
    }

    std::printf("Hilbert spectrum\n");
    {
        // two IMFs: 6.25 Hz with amplitude 2, 30.5 Hz with amplitude 0.5 (both inside a
        // bin, not on its edge); 1 Hz bins to 125 Hz
        const int n = 1000;
        const double fs = 250.0;
        std::vector<double> imfs(2 * n);
        for (int t = 0; t < n; ++t)
        {
            imfs[t] = 2.0 * std::cos(two_pi * 6.25 * t / fs);
            imfs[n + t] = 0.5 * std::sin(two_pi * 30.5 * t / fs + 0.3);
        }
        SpectrumOptions opt;
        opt.frequency_bins = 125;
        std::vector<double> S(125 * n), marg(125);
        const bool ok = hilbert_spectrum(imfs.data(), 2, n, fs, opt, S.data(), marg.data());
        double other = 0;
        for (int b = 0; b < 125; ++b)
            if (b != 6 && b != 30)
                other += marg[b];
        check(ok && std::abs(marg[6] - 4.0 * n) < 1e-6 && std::abs(marg[30] - 0.25 * n) < 1e-6 && other == 0.0,
              "marginal: energy 4 x 1000 in the 6 Hz bin, 0.25 x 1000 in the 30 Hz bin, nothing else");
        double row = 0;
        for (int t = 0; t < n; ++t)
            row = std::max(row, std::abs(S[static_cast<size_t>(6) * n + t] - 4.0));
        check(row < 1e-9, "map: energy 4 in the 6 Hz bin at every sample", row, 1e-9);
        check(std::abs(bin_centre(6, fs, opt) - 6.5) < 1e-12, "bin 6 is centred on 6.5 Hz");

        SpectrumOptions coarse = opt;
        coarse.time_step = 10;
        std::vector<double> C(125 * spectrum_columns(n, coarse));
        hilbert_spectrum(imfs.data(), 2, n, fs, coarse, C.data(), nullptr);
        double ec = 0;
        for (int col = 0; col < spectrum_columns(n, coarse); ++col)
            ec = std::max(ec, std::abs(C[static_cast<size_t>(6) * 100 + col] - 40.0));
        check(spectrum_columns(n, coarse) == 100 && ec < 1e-9, "time step 10: 100 columns, 10 samples summed in each",
              ec, 1e-9);

        SpectrumOptions low = opt;
        low.fmax = 20.0;
        low.frequency_bins = 20;
        std::vector<double> ml(20);
        hilbert_spectrum(imfs.data(), 2, n, fs, low, nullptr, ml.data());
        double total = 0;
        for (double v : ml)
            total += v;
        check(std::abs(total - 4.0 * n) < 1e-6, "fmax 20 Hz: the 30.5 Hz IMF is left out", std::abs(total - 4.0 * n), 1e-6);

        SpectrumOptions amp = opt;
        amp.amplitude = true;
        std::vector<double> ma(125);
        hilbert_spectrum(imfs.data(), 2, n, fs, amp, nullptr, ma.data());
        check(std::abs(ma[6] - 2.0 * n) < 1e-6, "amplitude instead of energy", std::abs(ma[6] - 2.0 * n), 1e-6);
    }

    std::printf("Sums and threads\n");
    {
        // ten noisy IMF-like rows: the spectrum's total is the energy of the samples whose
        // frequency falls in range, and the map's rows sum to the marginal
        const int k = 10, n = 5000;
        std::mt19937 rng(3);
        std::normal_distribution<double> g(0.0, 1.0);
        std::vector<double> imfs(static_cast<size_t>(k) * n);
        for (int r = 0; r < k; ++r)
        {
            double s = 0;
            for (int t = 0; t < n; ++t)
            {
                s = 0.9 * s + g(rng);
                imfs[static_cast<size_t>(r) * n + t] = std::sin(two_pi * (0.01 + 0.03 * r) * t) + 0.2 * s;
            }
        }
        SpectrumOptions opt;
        opt.frequency_bins = 64;
        opt.time_step = 7;
        const int cols = spectrum_columns(n, opt);
        std::vector<double> S(static_cast<size_t>(64) * cols), marg(64);
        eemd::set_num_threads(1);
        hilbert_spectrum(imfs.data(), k, n, 1.0, opt, S.data(), marg.data());

        double expected = 0;
        std::vector<double> a(n), f(n);
        for (int r = 0; r < k; ++r)
        {
            instantaneous(&imfs[static_cast<size_t>(r) * n], n, 1.0, a.data(), f.data(), nullptr);
            for (int t = 0; t < n; ++t)
                if (f[t] >= 0.0 && f[t] <= 0.5)
                    expected += a[t] * a[t];
        }
        double total = 0, rows = 0;
        for (double v : marg)
            total += v;
        for (double v : S)
            rows += v;
        check(std::abs(total - expected) < 1e-9 * expected, "marginal total = energy of the in-range samples",
              std::abs(total - expected) / expected, 1e-9);
        check(std::abs(rows - total) < 1e-9 * total, "the map sums to the marginal", std::abs(rows - total) / total, 1e-9);

        std::vector<double> S4(S.size()), m4(64);
        eemd::set_num_threads(4);
        hilbert_spectrum(imfs.data(), k, n, 1.0, opt, S4.data(), m4.data());
        check(S4 == S && m4 == marg, "4 threads: the same spectrum, bit for bit");
        eemd::set_num_threads(0);
    }

    std::printf("Refused\n");
    {
        std::vector<double> x(16, 1.0), out(16);
        SpectrumOptions opt;
        SpectrumOptions inverted = opt;
        inverted.fmin = 0.4;
        inverted.fmax = 0.1;
        check(!instantaneous(x.data(), 1, 1.0, out.data(), nullptr, nullptr) &&
                  !instantaneous(x.data(), 16, 0.0, out.data(), nullptr, nullptr) &&
                  !hilbert_spectrum(x.data(), 1, 16, 1.0, inverted, nullptr, out.data()),
              "one sample, a sample rate of 0, fmax below fmin");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

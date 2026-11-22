/**
 * FFT backend tests: Hilbert transform and instantaneous frequency against
 * direct O(n^2) DFT references, at even, odd and prime n.
 *
 * Exits non-zero on any failure.
 */

#include "eemd.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_MXCSR 1
#endif

static int g_failures = 0;

static void check(bool ok, const char *what, double value, double limit)
{
    std::printf("  %-52s %s  %.3e (limit %.1e)\n", what, ok ? "PASS" : "FAIL", value, limit);
    if (!ok)
        ++g_failures;
}

// Direct Hilbert transform: imaginary part of IDFT(H * DFT(x)), long double.
static std::vector<double> hilbert_reference(const std::vector<double> &x)
{
    const int n = static_cast<int>(x.size());
    const long double pi = 3.141592653589793238462643383279502884L;
    std::vector<long double> re(n), im(n);
    for (int k = 0; k < n; ++k)
    {
        long double sr = 0, si = 0;
        for (int t = 0; t < n; ++t)
        {
            const long double a = -2.0L * pi * k * t / n;
            sr += x[t] * std::cos(a);
            si += x[t] * std::sin(a);
        }
        // analytic-signal weights: 1 at DC (and Nyquist for even n), 2 for the
        // positive frequencies, 0 for the negative ones
        long double w = 0;
        if (k == 0 || (n % 2 == 0 && k == n / 2))
            w = 1;
        else if (k < (n + 1) / 2)
            w = 2;
        re[k] = w * sr;
        im[k] = w * si;
    }
    std::vector<double> y(n);
    for (int t = 0; t < n; ++t)
    {
        long double si = 0;
        for (int k = 0; k < n; ++k)
        {
            const long double a = 2.0L * pi * k * t / n;
            si += re[k] * std::sin(a) + im[k] * std::cos(a);
        }
        y[t] = static_cast<double>(si / n);
    }
    return y;
}

int main()
{
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> gauss(0.0, 1.0);

#ifdef HAVE_MXCSR
    // run the checks with the default mode (FTZ/DAZ off) so a backend that
    // leaked FTZ/DAZ into this thread would show up below
    _mm_setcsr(_mm_getcsr() & ~0x8040u);
    const unsigned mxcsr_before = _mm_getcsr();
#endif

    std::printf("Hilbert transform vs direct DFT\n");
    const int sizes[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 100, 101, 127, 128, 243, 1000, 1001, 1009, 2048};
    for (int n : sizes)
    {
        std::vector<double> x(n), y(n);
        double amax = 0;
        for (auto &v : x)
        {
            v = gauss(rng);
            amax = std::max(amax, std::abs(v));
        }
        const bool ok = eemd::compute_hilbert_transform(x.data(), n, y.data());
        std::vector<double> ref = hilbert_reference(x);
        double err = 0;
        for (int i = 0; i < n; ++i)
            err = std::max(err, std::abs(y[i] - ref[i]));
        err /= std::max(amax, 1e-300);
        char label[96];
        std::snprintf(label, sizeof(label), "n=%-5d %s", n,
                      eemd::fft::backend_name(eemd::fft::backend_for(n)));
        check(ok && err < 1e-12, label, err, 1e-12);
    }

    std::printf("Instantaneous frequency of a pure tone\n");
    for (int n : {1000, 1001, 2048})
    {
        const double f = 0.0371; // cycles per sample
        std::vector<double> x(n), inst(n);
        for (int i = 0; i < n; ++i)
            x[i] = std::cos(2 * M_PI * f * i + 0.3);
        const bool ok = eemd::compute_instantaneous_frequency(x.data(), n, inst.data(), 1.0);
        // edges of a non-periodic tone are distorted; check the middle half
        double err = 0;
        for (int i = n / 4; i < 3 * n / 4; ++i)
            err = std::max(err, std::abs(inst[i] - f));
        char label[96];
        std::snprintf(label, sizeof(label), "n=%d, f=%.4f (middle half)", n, f);
        check(ok && err < 2e-3, label, err, 2e-3);
    }

#ifdef HAVE_MXCSR
    std::printf("Floating-point environment\n");
    // compare the control bits only (6..15: DAZ, masks, rounding, FTZ);
    // bits 0..5 are sticky exception flags that ordinary arithmetic sets
    const unsigned control = 0xFFC0u;
    const unsigned mxcsr_after = _mm_getcsr();
    check((mxcsr_before & control) == (mxcsr_after & control),
          "caller's MXCSR control bits unchanged by the FFT calls",
          static_cast<double>((mxcsr_before ^ mxcsr_after) & control), 0.0);
#endif

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

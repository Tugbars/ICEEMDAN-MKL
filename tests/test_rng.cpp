/**
 * Random number tests (eemd_rng.hpp): Philox4x32-10 against the published
 * known answers and against the stream MKL gave the library before; the
 * uniform; the logarithm and the inverse normal distribution function against
 * extended-precision references; the coefficients against Wichura's check
 * sums; the statistics of ten million Gaussians; and a checksum of the first
 * million Gaussians of one stream, which every compiler must reproduce.
 *
 * Exits non-zero on any failure.
 */

#include "eemd_rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %-66s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

static void check(bool ok, const char *what, double value, double limit)
{
    std::printf("  %-66s %s  %.3g (limit %.3g)\n", what, ok ? "PASS" : "FAIL", value, limit);
    if (!ok)
        ++g_failures;
}

// ---- references in extended precision (80-bit long double with GCC on x86)
static const long double sqrt2 = std::sqrt(2.0L);
static const long double sqrt2pi = std::sqrt(2.0L * 3.14159265358979323846264338327950288L);

// z with Phi(z) = u, by Halley's method on the equation that keeps its precision: near the
// middle Phi(z) - 1/2 = erf(z / sqrt 2) / 2 (no cancellation), in the tails Phi(z) through erfc
static long double inv_norm(long double u)
{
    const long double q = u - 0.5L;
    if (q == 0)
        return 0;
    const bool lower = q < 0;
    const long double p = lower ? u : 1.0L - u; // the tail's probability, exact
    long double z;
    if (std::fabs(q) < 0.4L)
        z = q * sqrt2pi; // start
    else
    {
        const long double t = std::sqrt(-2.0L * std::log(p));
        z = t - (2.515517L + 0.802853L * t + 0.010328L * t * t) /
                    (1.0L + 1.432788L * t + 0.189269L * t * t + 0.001308L * t * t * t);
        z = lower ? -z : z;
    }
    for (int i = 0; i < 12; ++i)
    {
        long double e; // Phi(z) - u, accurately
        if (std::fabs(q) < 0.4L)
            e = 0.5L * std::erf(z / sqrt2) - q;
        else if (lower)
            e = 0.5L * std::erfc(-z / sqrt2) - p;
        else
            e = p - 0.5L * std::erfc(z / sqrt2);
        const long double d = std::exp(-0.5L * z * z) / sqrt2pi;
        const long double step = e / d;
        z -= step / (1.0L + 0.5L * z * step);
    }
    return z;
}

// error of a double result in units in the last place of the exact value
static double ulps(double got, long double exact)
{
    const double e = static_cast<double>(exact);
    if (e == 0)
        return got == 0 ? 0 : 1e300;
    const double ulp = std::ldexp(1.0, std::ilogb(e) - 52);
    return static_cast<double>(std::fabs(static_cast<long double>(got) - exact)) / ulp;
}

static double significand_sum(const double *c, int count)
{
    double s = 0;
    for (int i = 0; i < count; ++i)
        s += c[i] / std::pow(10.0, std::floor(std::log10(c[i])));
    return s;
}

int main()
{
    using namespace eemd::rng;
    const bool extended = sizeof(long double) > sizeof(double);

    std::printf("Philox4x32-10\n");
    {
        struct Kat
        {
            uint32_t ctr[4], k0, k1, out[4];
        } kats[] = {
            {{0, 0, 0, 0}, 0, 0, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
            {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, 0xffffffffu, 0xffffffffu,
             {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
            {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, 0xa4093822u, 0x299f31d0u,
             {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
        };
        bool ok = true;
        for (const Kat &k : kats)
        {
            uint32_t out[4];
            philox4x32_10(k.ctr, k.k0, k.k1, out);
            ok = ok && std::memcmp(out, k.out, sizeof(out)) == 0;
        }
        check(ok, "the published known answers (Random123 kat_vectors)");

        // the first words MKL's VSL_BRNG_PHILOX4X32X10 gave for the parameters {42, 7}
        uint32_t w[6];
        philox_words(42, 7, 0, 6, w);
        check(w[0] == 0x64d43a77u && w[1] == 0xff08a6bfu && w[2] == 0xff050829u && w[3] == 0x1e30fa6bu,
              "the stream keyed (42, 7) starts as MKL's did");

        std::vector<uint32_t> all(1000), part(400);
        philox_words(3, 9, 0, 1000, all.data());
        bool same = true;
        for (uint64_t first : {0u, 1u, 2u, 3u, 4u, 5u, 517u})
        {
            philox_words(3, 9, first, 400, part.data());
            same = same && std::equal(part.begin(), part.end(), all.begin() + static_cast<long>(first));
        }
        check(same, "words from any starting point equal the same words of the whole");
    }

    std::printf("Uniform\n");
    {
        check(uniform(0x80000000u) == 0.0 && uniform(0x80000001u) == 0x1p-32 && uniform(0u) == 0.5 &&
                  uniform(0x7fffffffu) == 1.0 - 0x1p-32,
              "u = (w xor 2^31) 2^-32: 0, 2^-32, 0.5 and 1 - 2^-32 where expected");
    }

    std::printf("Logarithm\n");
    if (extended)
    {
        std::mt19937_64 gen(11);
        double worst = 0;
        auto test = [&](double x) { worst = std::max(worst, ulps(ln(x), std::log(static_cast<long double>(x)))); };
        for (int k = -1022; k <= 1023; ++k)
            for (double m : {1.0, 1.0000000000000002, 1.4142135623730951, 1.4142135623730954, 1.9999999999999998})
                if (k != 0 || m != 1.0)
                    test(std::ldexp(m, k));
        for (int i = 0; i < 2000000; ++i)
        {
            const double x = std::ldexp(1.0 + (gen() >> 11) * 0x1p-53, static_cast<int>(gen() % 120) - 100);
            if (x != 1.0)
                test(x);
        }
        check(worst <= 1.0, "ln(x) against long double, 2 million x and the binade edges (ulps)", worst, 1.0);
        check(ln(1.0) == 0.0 && ln(2.0) == 0.69314718055994530942, "ln(1) = 0 and ln(2) = the double nearest ln 2");
    }
    else
    {
        std::printf("  skipped: long double is not wider than double with this compiler\n");
    }

    std::printf("Inverse normal distribution function (AS241)\n");
    {
        const double sums[3] = {significand_sum(detail::as241_a, 8) + significand_sum(detail::as241_b, 7),
                                significand_sum(detail::as241_c, 8) + significand_sum(detail::as241_d, 7),
                                significand_sum(detail::as241_e, 8) + significand_sum(detail::as241_f, 7)};
        const double paper[3] = {55.8831928806149014439, 49.33206503301610289036, 47.52583317549289671629};
        double worst = 0;
        for (int r = 0; r < 3; ++r)
            worst = std::max(worst, std::fabs(sums[r] - paper[r]));
        check(worst < 1e-12, "the coefficients against Wichura's check sums", worst, 1e-12);

        if (extended)
        {
            // every kind of input the generator makes: random words, the tails word by word
            std::mt19937 gen(5);
            double central = 0, tail = 0;
            auto test = [&](uint32_t w)
            {
                const double u = uniform(w);
                if (u == 0.0)
                    return;
                const double e = ulps(inverse_normal_cdf(u), inv_norm(static_cast<long double>(u)));
                double &worst = std::fabs(u - 0.5) <= 0.425 ? central : tail;
                worst = std::max(worst, e);
            };
            for (int i = 0; i < 4000000; ++i)
                test(gen());
            for (uint32_t w = 0; w < 200000; ++w)
            {
                test(0x80000001u + w); // u = 2^-32 upwards
                test(0x7fffffffu - w); // u = 1 - 2^-32 downwards
            }
            for (int k = 1; k <= 31; ++k)
            {
                test(0x80000000u + (1u << k));
                test(0x80000000u - (1u << k));
            }
            // AS241 evaluated in double: about 5 ulps (5e-16) at worst; MKL's ICDF was 7e-9
            check(central <= 6.0, "central region, |u - 1/2| <= 0.425 (ulps)", central, 6.0);
            check(tail <= 6.0, "tails, down to u = 2^-32 (ulps)", tail, 6.0);
            const double z0 = inverse_normal_cdf(0.0);
            check(ulps(z0, inv_norm(0x1p-54L)) <= 4.0, "u = 0 gives Phi^-1(2^-54) = -8.29 (ulps)",
                  ulps(z0, inv_norm(0x1p-54L)), 4.0);
        }
        bool symmetric = true;
        std::mt19937 gen(8);
        for (int i = 0; i < 1000000 && symmetric; ++i)
        {
            const uint32_t w = gen() | 1u; // u and 1 - u both exact
            const double u = uniform(w);
            symmetric = inverse_normal_cdf(u) == -inverse_normal_cdf(1.0 - u);
        }
        check(symmetric, "Phi^-1(1 - u) = -Phi^-1(u) exactly");
        std::vector<double> zs;
        for (uint64_t w = 0; w <= 0xffffffffull; w += 65537)
            zs.push_back(inverse_normal_cdf(uniform(static_cast<uint32_t>(w) ^ 0x80000000u)));
        check(std::is_sorted(zs.begin(), zs.end()), "increasing in u (65,537 points across [0, 1))");
    }

    std::printf("Gaussians\n");
    {
        const int64_t n = 10000000;
        std::vector<double> z(n);
        gaussian(2024, 3, n, z.data());
        long double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (double v : z)
        {
            const long double x = v;
            s1 += x;
            s2 += x * x;
            s3 += x * x * x;
            s4 += x * x * x * x;
        }
        const double mean = static_cast<double>(s1 / n);
        const double var = static_cast<double>(s2 / n) - mean * mean;
        const double skew = static_cast<double>(s3 / n) / std::pow(var, 1.5);
        const double kurt = static_cast<double>(s4 / n) / (var * var);
        const double se = 1.0 / std::sqrt(static_cast<double>(n));
        check(std::fabs(mean) < 5 * se, "mean of 10 million (5 standard errors)", std::fabs(mean), 5 * se);
        check(std::fabs(var - 1) < 5 * std::sqrt(2.0) * se, "variance - 1", std::fabs(var - 1), 5 * std::sqrt(2.0) * se);
        check(std::fabs(skew) < 5 * std::sqrt(6.0) * se, "skewness", std::fabs(skew), 5 * std::sqrt(6.0) * se);
        check(std::fabs(kurt - 3) < 5 * std::sqrt(24.0) * se, "kurtosis - 3", std::fabs(kurt - 3), 5 * std::sqrt(24.0) * se);

        // 100 equally likely bins: chi-square with 99 degrees of freedom (mean 99, sd 14)
        std::vector<int64_t> bins(100, 0);
        for (double v : z)
        {
            const double p = 0.5 * std::erfc(-v / std::sqrt(2.0));
            ++bins[std::min<int64_t>(99, static_cast<int64_t>(p * 100))];
        }
        double chi2 = 0;
        for (int64_t c : bins)
            chi2 += (c - n / 100.0) * (c - n / 100.0) / (n / 100.0);
        check(chi2 < 160, "chi-square over 100 equally likely bins (99 degrees of freedom)", chi2, 160);

        std::vector<double> a(1000), b(1000), shifted(1000);
        gaussian(2024, 3, 777, a.data());
        gaussian(2024, 3, 1000, b.data());
        gaussian(2024, 3, 1000, shifted.data(), 5.0, 2.0);
        bool prefix = std::equal(a.begin(), a.begin() + 777, b.begin());
        bool scaled = true;
        for (int i = 0; i < 1000; ++i)
            scaled = scaled && shifted[i] == std::fma(2.0, b[i], 5.0);
        check(prefix, "the first m Gaussians do not depend on how many are drawn");
        check(scaled, "mean and sigma: mean + sigma z");
        gaussian(2024, 4, 1000, a.data());
        check(!std::equal(a.begin(), a.end(), b.begin()), "another stream, other numbers");
    }

    std::printf("The same numbers from every compiler\n");
    {
        const int64_t n = 1000000;
        std::vector<double> z(n);
        gaussian(42, 7, n, z.data());
        uint64_t h = 1469598103934665603ull; // FNV-1a over the bytes
        const unsigned char *p = reinterpret_cast<const unsigned char *>(z.data());
        for (size_t i = 0; i < z.size() * sizeof(double); ++i)
            h = (h ^ p[i]) * 1099511628211ull;
        std::printf("  checksum of the first million Gaussians of stream (42, 7): %016llx\n",
                    static_cast<unsigned long long>(h));
        // GCC 16.2 with -O3 -march=native -ffast-math, with -O2 and with -O0 (no FMA unit)
        check(h == 0x1f34cd8ec74f2571ull, "the checksum this file expects");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

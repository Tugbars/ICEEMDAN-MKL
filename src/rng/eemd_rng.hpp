/**
 * The library's random numbers: Gaussian noise, with no external library.
 *
 * Generator: Philox4x32-10 (Salmon, Moraes, Dror & Shaw, "Parallel random
 * numbers: as easy as 1, 2, 3", SC 2011). It is counter-based: word j of the
 * stream with key (k0, k1) is a function of (k0, k1, j) alone. The noise of a
 * trial is the stream keyed (seed, trial), so it does not depend on the thread
 * that draws it, the number of threads or the order trials run in.
 *
 * Uniform: u = (w xor 2^31) 2^-32, in [0, 1), from one 32-bit word.
 *
 * Gaussian: z = Phi^-1(u), the inverse of the normal distribution function,
 * by Wichura's algorithm AS241 (Applied Statistics 37(3), 1988, 477-484),
 * accurate to about 1e-16. The one word that gives u = 0 is taken as
 * u = 2^-54 (z = -8.29).
 *
 * The streams, the word-to-uniform rule and the value at u = 0 are those of
 * Intel MKL's VSL_BRNG_PHILOX4X32X10 with VSL_RNG_METHOD_GAUSSIAN_ICDF, which
 * the library used before: the uniforms are the same bit for bit, and the
 * Gaussians differ from MKL's by MKL's own error, below 1e-8 (measured
 * through MKL's interface; benchmarks/rng_bench.cpp).
 *
 * The same numbers from every compiler: every multiply-add is an explicit
 * fma, the logarithm is this file's own (std::log differs between C
 * libraries in the last bit), and the functions below are compiled without
 * the fast-math reordering the library's Release flags allow elsewhere.
 *
 * License: MIT
 */

#ifndef EEMD_RNG_HPP
#define EEMD_RNG_HPP

#include <cmath>
#include <cstdint>
#include <cstring>

// Strict floating point for this file only: no reassociation, no reciprocal
// approximations, no contraction beyond the explicit std::fma calls.
#if defined(__clang__) || defined(_MSC_VER)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "fp-contract=off")
#endif

namespace eemd
{
    namespace rng
    {
        /** The stream of the circular noise bank; trials use the streams below it. */
        constexpr uint32_t circular_stream = 0xFFFFFFFFu;

        /** Philox4x32-10: the four words of counter ctr under key (k0, k1). */
        inline void philox4x32_10(const uint32_t ctr[4], uint32_t k0, uint32_t k1, uint32_t out[4])
        {
            uint32_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
            for (int round = 0; round < 10; ++round)
            {
                if (round)
                {
                    k0 += 0x9E3779B9u; // the Weyl key schedule
                    k1 += 0xBB67AE85u;
                }
                const uint64_t p0 = static_cast<uint64_t>(0xD2511F53u) * c0;
                const uint64_t p1 = static_cast<uint64_t>(0xCD9E8D57u) * c2;
                const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c1 ^ k0;
                const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c3 ^ k1;
                c1 = static_cast<uint32_t>(p1);
                c3 = static_cast<uint32_t>(p0);
                c0 = n0;
                c2 = n2;
            }
            out[0] = c0;
            out[1] = c1;
            out[2] = c2;
            out[3] = c3;
        }

        /**
         * Words first .. first + n - 1 of the stream keyed (k0, k1). Counter j
         * (a 128-bit number, low word first) holds words 4j .. 4j + 3.
         */
        inline void philox_words(uint32_t k0, uint32_t k1, uint64_t first, int64_t n, uint32_t *out)
        {
            uint64_t counter = first / 4;
            int skip = static_cast<int>(first % 4);
            int64_t j = 0;
            while (j < n)
            {
                const uint32_t ctr[4] = {static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), 0, 0};
                uint32_t w[4];
                philox4x32_10(ctr, k0, k1, w);
                for (int i = skip; i < 4 && j < n; ++i)
                    out[j++] = w[i];
                skip = 0;
                ++counter;
            }
        }

        /** The uniform of a word: (w xor 2^31) 2^-32, in [0, 1). */
        inline double uniform(uint32_t w)
        {
            return static_cast<double>(w ^ 0x80000000u) * 0x1p-32;
        }

        /**
         * Natural logarithm of a positive normal number, below one unit in the
         * last place. x = 2^k (1 + f) with 1 + f in [sqrt(1/2), sqrt(2)], and
         * with s = f / (2 + f), ln(1 + f) = 2 atanh(s) = 2s + s R, where
         * R = 2 (s^2/3 + s^4/5 + ...), |s| < 0.172; the terms after s^20/21
         * are below 1e-18 of the sum. Arranged as in fdlibm's log,
         * ln x = k ln2_hi - ((f^2/2 - (s (f^2/2 + R) + k ln2_lo)) - f), so
         * that the large terms k ln2_hi and f are exact.
         */
        inline double ln(double x)
        {
            uint64_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            int k = static_cast<int>((bits >> 52) & 0x7FFu) - 1023;
            bits = (bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull; // 1 + f in [1, 2)
            double m;
            std::memcpy(&m, &bits, sizeof(m));
            if (m > 1.4142135623730951)
            {
                m *= 0.5; // exact
                ++k;
            }
            const double f = m - 1.0; // exact (Sterbenz)
            const double s = f / (2.0 + f);
            const double z = s * s;
            double p = 2.0 / 21.0;
            p = std::fma(p, z, 2.0 / 19.0);
            p = std::fma(p, z, 2.0 / 17.0);
            p = std::fma(p, z, 2.0 / 15.0);
            p = std::fma(p, z, 2.0 / 13.0);
            p = std::fma(p, z, 2.0 / 11.0);
            p = std::fma(p, z, 2.0 / 9.0);
            p = std::fma(p, z, 2.0 / 7.0);
            p = std::fma(p, z, 2.0 / 5.0);
            p = std::fma(p, z, 2.0 / 3.0);
            const double R = z * p;
            const double hfsq = 0.5 * f * f;
            // ln 2 = ln2_hi + ln2_lo, ln2_hi with 21 trailing zero bits: k ln2_hi is exact
            constexpr double ln2_hi = 6.93147180369123816490e-01;
            constexpr double ln2_lo = 1.90821492927058770002e-10;
            const double dk = static_cast<double>(k);
            return dk * ln2_hi - ((hfsq - std::fma(s, hfsq + R, dk * ln2_lo)) - f);
        }

        namespace detail
        {
            // AS241 (PPND16): numerator and denominator of the three regions,
            // highest power first, as in Wichura (1988)
            constexpr double as241_a[8] = {2.5090809287301226727e+3, 3.3430575583588128105e+4,
                                           6.7265770927008700853e+4, 4.5921953931549871457e+4,
                                           1.3731693765509461125e+4, 1.9715909503065514427e+3,
                                           1.3314166789178437745e+2, 3.3871328727963666080e+0};
            constexpr double as241_b[8] = {5.2264952788528545610e+3, 2.8729085735721942674e+4,
                                           3.9307895800092710610e+4, 2.1213794301586595867e+4,
                                           5.3941960214247511077e+3, 6.8718700749205790830e+2,
                                           4.2313330701600911252e+1, 1.0};
            constexpr double as241_c[8] = {7.74545014278341407640e-4, 2.27238449892691845833e-2,
                                           2.41780725177450611770e-1, 1.27045825245236838258e+0,
                                           3.64784832476320460504e+0, 5.76949722146069140550e+0,
                                           4.63033784615654529590e+0, 1.42343711074968357734e+0};
            constexpr double as241_d[8] = {1.05075007164441684324e-9, 5.47593808499534494600e-4,
                                           1.51986665636164571966e-2, 1.48103976427480074590e-1,
                                           6.89767334985100004550e-1, 1.67638483018380384940e+0,
                                           2.05319162663775882187e+0, 1.0};
            constexpr double as241_e[8] = {2.01033439929228813265e-7, 2.71155556874348757815e-5,
                                           1.24266094738807843860e-3, 2.65321895265761230930e-2,
                                           2.96560571828504891230e-1, 1.78482653991729133580e+0,
                                           5.46378491116411436990e+0, 6.65790464350110377720e+0};
            constexpr double as241_f[8] = {2.04426310338993978564e-15, 1.42151175831644588870e-7,
                                           1.84631831751005468180e-5, 7.86869131145613259100e-4,
                                           1.48753612908506148525e-2, 1.36929880922735805310e-1,
                                           5.99832206555887937690e-1, 1.0};

            inline double horner8(const double *c, double r)
            {
                double p = c[0];
                for (int i = 1; i < 8; ++i)
                    p = std::fma(p, r, c[i]);
                return p;
            }

            /** AS241's middle region, |q| <= 0.425, q = u - 1/2. */
            inline double inverse_central(double q)
            {
                const double r = std::fma(-q, q, 0.180625);
                return (horner8(as241_a, r) * q) / horner8(as241_b, r);
            }

            /** AS241's tails, |u - 1/2| > 0.425; u = 0 counts as 2^-54. */
            inline double inverse_tail(double u)
            {
                if (u <= 0.0)
                    u = 0x1p-54;
                const double q = u - 0.5;
                double r = std::sqrt(-ln(q < 0.0 ? u : 1.0 - u));
                double x;
                if (r <= 5.0)
                {
                    r -= 1.6;
                    x = horner8(as241_c, r) / horner8(as241_d, r);
                }
                else
                {
                    r -= 5.0;
                    x = horner8(as241_e, r) / horner8(as241_f, r);
                }
                return q < 0.0 ? -x : x;
            }

            /** Philox4x32-10 of counters c .. c + 63: their 256 words, in order. */
            inline void philox_block64(uint64_t c, uint32_t k0, uint32_t k1, uint32_t *words)
            {
                // lane-wise, so that the compiler can run the lanes side by side
                uint32_t x0[64], x1[64], x2[64], x3[64];
                for (int j = 0; j < 64; ++j)
                {
                    x0[j] = static_cast<uint32_t>(c + j);
                    x1[j] = static_cast<uint32_t>((c + j) >> 32);
                    x2[j] = 0;
                    x3[j] = 0;
                }
                for (int round = 0; round < 10; ++round)
                {
                    if (round)
                    {
                        k0 += 0x9E3779B9u;
                        k1 += 0xBB67AE85u;
                    }
                    for (int j = 0; j < 64; ++j)
                    {
                        const uint64_t p0 = static_cast<uint64_t>(0xD2511F53u) * x0[j];
                        const uint64_t p1 = static_cast<uint64_t>(0xCD9E8D57u) * x2[j];
                        const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ x1[j] ^ k0;
                        const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ x3[j] ^ k1;
                        x1[j] = static_cast<uint32_t>(p1);
                        x3[j] = static_cast<uint32_t>(p0);
                        x0[j] = n0;
                        x2[j] = n2;
                    }
                }
                for (int j = 0; j < 64; ++j)
                {
                    words[4 * j] = x0[j];
                    words[4 * j + 1] = x1[j];
                    words[4 * j + 2] = x2[j];
                    words[4 * j + 3] = x3[j];
                }
            }
        } // namespace detail

        /** Phi^-1(u) for u in [0, 1); u = 0 counts as 2^-54. */
        inline double inverse_normal_cdf(double u)
        {
            const double q = u - 0.5; // exact: u has 32 bits after the point
            return std::fabs(q) <= 0.425 ? detail::inverse_central(q) : detail::inverse_tail(u);
        }

        /**
         * out[0 .. n) = mean + sigma z, the first n Gaussians of the stream
         * keyed (seed, stream): z = inverse_normal_cdf(uniform(word j)).
         *
         * In blocks of 256: the words of 64 counters at once, then the middle
         * region's formula for every number, then the tails (about 15 %)
         * again. Each number gets the same operations as inverse_normal_cdf
         * gives it, so the result is the same bit for bit, only faster.
         */
        inline void gaussian(uint32_t seed, uint32_t stream, int64_t n, double *out,
                             double mean = 0.0, double sigma = 1.0)
        {
            uint32_t words[256];
            double u[256];
            for (int64_t base = 0; base < n; base += 256)
            {
                const int m = static_cast<int>(n - base < 256 ? n - base : 256);
                detail::philox_block64(static_cast<uint64_t>(base) / 4, seed, stream, words);
                double *o = out + base;
                for (int i = 0; i < m; ++i)
                {
                    u[i] = uniform(words[i]);
                    o[i] = detail::inverse_central(u[i] - 0.5);
                }
                for (int i = 0; i < m; ++i)
                    if (!(std::fabs(u[i] - 0.5) <= 0.425))
                        o[i] = detail::inverse_tail(u[i]);
                for (int i = 0; i < m; ++i)
                    o[i] = std::fma(sigma, o[i], mean);
            }
        }
    } // namespace rng
} // namespace eemd

#if defined(__clang__) || defined(_MSC_VER)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // EEMD_RNG_HPP

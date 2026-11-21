/**
 * The envelope splines: piecewise cubics through the knots, evaluated at the
 * samples 0, 1, ..., n - 1. Three kinds, which differ only in the slopes at
 * the knots:
 *
 * - construct(): the cubic spline (C2) with not-a-knot ends, as MATLAB's
 *   spline and SciPy's CubicSpline make it (so the reference ICEEMDAN code's
 *   envelopes): at 3 knots that is the parabola through them, at 2 the
 *   straight line.
 * - construct_akima(): Akima's spline (H. Akima, J. ACM 17(4), 1970, 589-602),
 *   C1 and local: the slope at a knot is a weighted mean of the chords on
 *   either side, from the two knots before it to the two after, so a single
 *   outlying knot moves the curve only near itself and the curve does not
 *   overshoot a step. Past the ends, two chords on each side are extended
 *   linearly, as in Akima's paper.
 * - construct_linear(): straight lines between the knots.
 *
 * Outside the knots the first and the last piece are continued.
 *
 * An envelope needs one construct() and one evaluate_uniform(), tens of
 * thousands of times per decomposition, so both are written for that case:
 * the slopes at the knots come from one tridiagonal solve, each piece keeps
 * its four coefficients, and the evaluation walks the pieces once, four
 * samples at a time where AVX2 and FMA are there. No memory is allocated
 * after the buffers have grown to the largest knot count seen.
 *
 * License: MIT
 */

#ifndef EEMD_CUBIC_SPLINE_HPP
#define EEMD_CUBIC_SPLINE_HPP

#include <cmath>
#include <cstdint>
#include <vector>

// Visual C++ has no __FMA__: its /arch:AVX2 stands for both
#if defined(__AVX2__) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__)))
#include <immintrin.h>
#define EEMD_SPLINE_AVX2 1
#endif

namespace eemd
{
    class CubicSpline
    {
    public:
        /**
         * Fits the spline through (x[i], y[i]), i = 0 .. n_knots - 1, x
         * strictly increasing. x and y are read again by evaluate_uniform():
         * they must stay as they are until then. False when there are fewer
         * than 2 knots or x does not increase.
         */
        bool construct(const double *x, const double *y, int32_t n_knots)
        {
            const int32_t k = n_knots;
            if (!prepare(x, y, k))
                return false;
            const double *h = h_.data();
            const double *delta = delta_.data();

            double *s = slope_.data();
            if (k == 2)
            {
                s[0] = s[1] = delta[0];
            }
            else if (k == 3)
            {
                // the parabola through the three knots: its slopes there
                const double c = (delta[1] - delta[0]) / (h[0] + h[1]);
                s[0] = delta[0] - c * h[0];
                s[1] = delta[0] + c * h[0];
                s[2] = delta[1] + c * h[1];
            }
            else
            {
                // Slopes s at the knots: a tridiagonal system, one row per knot.
                //   interior row i:  h[i] s[i-1] + 2 (h[i-1] + h[i]) s[i] + h[i-1] s[i+1]
                //                    = 3 (h[i] delta[i-1] + h[i-1] delta[i])
                // First and last row by the end condition. Solved by elimination
                // from the first row down (cp, dp), then substitution back up.
                double *cp = cp_.data();
                double *dp = dp_.data();

                // not-a-knot: the third derivative is continuous at the second knot
                const double span0 = h[0] + h[1];
                const double b0 = h[1];
                const double c0 = span0;
                const double d0 = ((3.0 * h[0] + 2.0 * h[1]) * h[1] * delta[0] + h[0] * h[0] * delta[1]) / span0;
                cp[0] = c0 / b0;
                dp[0] = d0 / b0;

                for (int32_t i = 1; i < k - 1; ++i)
                {
                    const double a = h[i];
                    const double c = h[i - 1];
                    const double d = 3.0 * (h[i] * delta[i - 1] + h[i - 1] * delta[i]);
                    const double denom = 2.0 * (h[i - 1] + h[i]) - a * cp[i - 1];
                    cp[i] = c / denom;
                    dp[i] = (d - a * dp[i - 1]) / denom;
                }

                // and at the last but one
                const double span1 = h[k - 3] + h[k - 2];
                const double a_last = span1;
                const double b_last = h[k - 3];
                const double d_last = (h[k - 2] * h[k - 2] * delta[k - 3] +
                                       (2.0 * h[k - 3] + 3.0 * h[k - 2]) * h[k - 3] * delta[k - 2]) /
                                      span1;
                s[k - 1] = (d_last - a_last * dp[k - 2]) / (b_last - a_last * cp[k - 2]);
                for (int32_t i = k - 2; i >= 0; --i)
                    s[i] = dp[i] - cp[i] * s[i + 1];
            }
            finish_hermite(x, y, k);
            return true;
        }

        /**
         * Akima's spline through the knots (same contract as construct()).
         * At 2 knots, the straight line.
         */
        bool construct_akima(const double *x, const double *y, int32_t n_knots)
        {
            const int32_t k = n_knots;
            if (!prepare(x, y, k))
                return false;
            const double *delta = delta_.data();
            double *s = slope_.data();
            if (k == 2)
            {
                s[0] = s[1] = delta[0];
            }
            else
            {
                // chords m[-2 .. k]: m[j] = delta[j] for 0 <= j <= k - 2, and two
                // more on each side extended linearly (Akima 1970, eq. 7)
                double *m = chord_.data() + 2;
                for (int32_t j = 0; j < k - 1; ++j)
                    m[j] = delta[j];
                m[-1] = 2.0 * m[0] - m[1];
                m[-2] = 2.0 * m[-1] - m[0];
                m[k - 1] = 2.0 * m[k - 2] - m[k - 3];
                m[k] = 2.0 * m[k - 1] - m[k - 2];
                // slope at knot i: the chords m[i-1] and m[i], each weighted by how
                // much the chords on the far side change (eq. 1); their mean where
                // neither side changes
                for (int32_t i = 0; i < k; ++i)
                {
                    const double w_left = std::fabs(m[i + 1] - m[i]);  // weight of m[i-1]
                    const double w_right = std::fabs(m[i - 1] - m[i - 2]); // weight of m[i]
                    const double w = w_left + w_right;
                    s[i] = w > 0.0 ? (w_left * m[i - 1] + w_right * m[i]) / w : 0.5 * (m[i - 1] + m[i]);
                }
            }
            finish_hermite(x, y, k);
            return true;
        }

        /** Straight lines between the knots (same contract as construct()). */
        bool construct_linear(const double *x, const double *y, int32_t n_knots)
        {
            const int32_t k = n_knots;
            if (!prepare(x, y, k))
                return false;
            // piece i: y[i] + delta[i] t
            const double *delta = delta_.data();
            double *s = slope_.data();
            double *c2 = c2_.data();
            double *c3 = c3_.data();
            for (int32_t i = 0; i < k - 1; ++i)
            {
                s[i] = delta[i];
                c2[i] = 0.0;
                c3[i] = 0.0;
            }
            commit(x, y, k);
            return true;
        }

        /** out[j] = spline at j, for j = 0 .. n_sites - 1. False before a successful construct(). */
        bool evaluate_uniform(double *out, int32_t n_sites) const
        {
            if (!valid_)
                return false;

            const double *x = x_;
            const double *y = y_;
            const double *s = slope_.data();
            const double *c2 = c2_.data();
            const double *c3 = c3_.data();
            const int32_t pieces = n_knots_ - 1;

            int32_t j = 0; // first sample not written yet
            for (int32_t i = 0; i < pieces && j < n_sites; ++i)
            {
                // Piece i takes the samples below x[i+1]; the last piece takes all that are left
                int32_t j_end = n_sites;
                if (i < pieces - 1)
                {
                    const double next = x[i + 1];
                    if (next <= static_cast<double>(j))
                        continue; // no sample left in this piece
                    if (next < static_cast<double>(n_sites))
                    {
                        j_end = static_cast<int32_t>(next);
                        if (static_cast<double>(j_end) < next)
                            ++j_end; // ceil
                    }
                }

                const double t0 = static_cast<double>(j) - x[i];
                const double a0 = y[i], a1 = s[i], a2 = c2[i], a3 = c3[i];
#ifdef EEMD_SPLINE_AVX2
                // Four samples at a time. A group may run past this piece's
                // last sample: those entries are written again by the pieces
                // that own them. Only the end of the output is done singly.
                if (j + 4 <= n_sites)
                {
                    const __m256d v0 = _mm256_set1_pd(a0);
                    const __m256d v1 = _mm256_set1_pd(a1);
                    const __m256d v2 = _mm256_set1_pd(a2);
                    const __m256d v3 = _mm256_set1_pd(a3);
                    const __m256d four = _mm256_set1_pd(4.0);
                    __m256d t = _mm256_add_pd(_mm256_set1_pd(t0), _mm256_set_pd(3.0, 2.0, 1.0, 0.0));
                    int32_t jj = j;
                    do
                    {
                        __m256d p = _mm256_fmadd_pd(t, v3, v2);
                        p = _mm256_fmadd_pd(t, p, v1);
                        p = _mm256_fmadd_pd(t, p, v0);
                        _mm256_storeu_pd(out + jj, p);
                        t = _mm256_add_pd(t, four);
                        jj += 4;
                    } while (jj < j_end && jj + 4 <= n_sites);
                    if (jj >= j_end)
                    {
                        j = j_end;
                        continue;
                    }
                    // fewer than four samples left in the output: finish below
                    for (; jj < j_end; ++jj)
                    {
                        const double tt = static_cast<double>(jj) - x[i];
                        out[jj] = std::fma(tt, std::fma(tt, std::fma(tt, a3, a2), a1), a0);
                    }
                    j = j_end;
                    continue;
                }
                for (int32_t jj = j; jj < j_end; ++jj)
                {
                    const double tt = t0 + static_cast<double>(jj - j);
                    out[jj] = std::fma(tt, std::fma(tt, std::fma(tt, a3, a2), a1), a0);
                }
#else
                for (int32_t jj = j; jj < j_end; ++jj)
                {
                    const double tt = t0 + static_cast<double>(jj - j);
                    out[jj] = a0 + tt * (a1 + tt * (a2 + tt * a3));
                }
#endif
                j = j_end;
            }
            return true;
        }

    private:
        /** Spacing and chord slopes; false for fewer than 2 knots or x not increasing. */
        bool prepare(const double *x, const double *y, int32_t k)
        {
            valid_ = false;
            if (k < 2)
                return false;
            reserve(k);
            double *h = h_.data();
            double *delta = delta_.data();
            double *inv_h = inv_h_.data();
            double min_h = x[1] - x[0];
            for (int32_t i = 0; i < k - 1; ++i)
            {
                h[i] = x[i + 1] - x[i];
                min_h = h[i] < min_h ? h[i] : min_h;
            }
            if (!(min_h > 0.0))
                return false;
            for (int32_t i = 0; i < k - 1; ++i)
            {
                inv_h[i] = 1.0 / h[i];
                delta[i] = (y[i + 1] - y[i]) * inv_h[i];
            }
            return true;
        }

        /** The cubic of each piece from the slopes at its two knots (Hermite form). */
        void finish_hermite(const double *x, const double *y, int32_t k)
        {
            // Piece i on [x[i], x[i+1]], t = x - x[i]:  y[i] + s[i] t + c2[i] t^2 + c3[i] t^3
            const double *s = slope_.data();
            const double *delta = delta_.data();
            const double *inv_h = inv_h_.data();
            double *c2 = c2_.data();
            double *c3 = c3_.data();
            for (int32_t i = 0; i < k - 1; ++i)
            {
                c2[i] = (3.0 * delta[i] - 2.0 * s[i] - s[i + 1]) * inv_h[i];
                c3[i] = (s[i] + s[i + 1] - 2.0 * delta[i]) * inv_h[i] * inv_h[i];
            }
            commit(x, y, k);
        }

        void commit(const double *x, const double *y, int32_t k)
        {
            x_ = x;
            y_ = y;
            n_knots_ = k;
            valid_ = true;
        }

        void reserve(int32_t k)
        {
            if (static_cast<size_t>(k) <= h_.size())
                return;
            const size_t cap = static_cast<size_t>(k) + static_cast<size_t>(k) / 2 + 16;
            h_.resize(cap);
            inv_h_.resize(cap);
            delta_.resize(cap);
            slope_.resize(cap);
            c2_.resize(cap);
            c3_.resize(cap);
            cp_.resize(cap);
            dp_.resize(cap);
            chord_.resize(cap + 4);
        }

        const double *x_ = nullptr;
        const double *y_ = nullptr;
        int32_t n_knots_ = 0;
        bool valid_ = false;

        std::vector<double> h_, inv_h_, delta_; // knot spacing, its reciprocal, chord slopes
        std::vector<double> slope_, c2_, c3_;   // per knot / per piece
        std::vector<double> cp_, dp_;           // elimination workspace
        std::vector<double> chord_;             // Akima: the chords, two extra on each side
    };

} // namespace eemd

#endif // EEMD_CUBIC_SPLINE_HPP

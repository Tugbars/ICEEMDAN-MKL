/**
 * What the engines measure on a signal before and while they decompose it:
 * its standard deviation, whether it is monotonic, how many extrema it has,
 * and the repair of NaN and Inf samples.
 *
 * License: MIT
 */

#ifndef EEMD_SIGNAL_HPP
#define EEMD_SIGNAL_HPP

#include "eemd_extrema.hpp"

namespace eemd
{

    inline bool is_monotonic(const double *signal, int32_t n, double threshold = 1e-6)
    {
        if (n < 3)
            return true;

        int32_t n_increasing = 0;
        int32_t n_decreasing = 0;

        for (int32_t i = 1; i < n; ++i)
        {
            double diff = signal[i] - signal[i - 1];
            if (diff > threshold)
                ++n_increasing;
            else if (diff < -threshold)
                ++n_decreasing;
        }

        double ratio = static_cast<double>(std::max(n_increasing, n_decreasing)) / (n - 1);
        return ratio > 0.95;
    }

    /** Local maxima plus minima, plateau-aware (same rule as the sifting). */
    inline int32_t count_extrema(const double *signal, int32_t n)
    {
        return count_extrema_raw(signal, n);
    }

    inline double compute_std(const double *signal, int32_t n)
    {
        double mean = 0.0;
        EEMD_OMP_SIMD_REDUCTION(+, mean)
        for (int32_t i = 0; i < n; ++i)
        {
            mean += signal[i];
        }
        mean /= n;

        double var = 0.0;
        EEMD_OMP_SIMD_REDUCTION(+, var)
        for (int32_t i = 0; i < n; ++i)
        {
            double d = signal[i] - mean;
            var += d * d;
        }

        return std::sqrt(var / n);
    }

    /** Standard deviation with n - 1 in the denominator (MATLAB's std). */
    inline double compute_sample_std(const double *signal, int32_t n)
    {
        return n > 1 ? compute_std(signal, n) * std::sqrt(static_cast<double>(n) / (n - 1)) : 0.0;
    }

    /**
     * Sanitize input signal: replace NaN/Inf with interpolated values
     * Returns count of sanitized points (for diagnostics)
     *
     * The envelopes and the sifting assume finite input
     *
     * Uses is_finite_bits(), which -ffast-math cannot optimize away (with
     * std::isfinite the whole check compiled to nothing under the project's
     * own GCC flags). One pass over each run of bad samples: O(n).
     */
    inline int32_t sanitize_signal(double *signal, int32_t n)
    {
        int32_t nan_count = 0;
        int32_t prev_good = -1; // index of the last finite sample

        int32_t i = 0;
        while (i < n)
        {
            if (is_finite_bits(signal[i]))
            {
                prev_good = i++;
                continue;
            }

            // Bad run [i, next)
            int32_t next = i;
            while (next < n && !is_finite_bits(signal[next]))
                ++next;
            nan_count += next - i;

            if (prev_good < 0 && next >= n)
            {
                std::memset(signal, 0, n * sizeof(double)); // all bad
                return nan_count;
            }

            for (int32_t j = i; j < next; ++j)
            {
                if (prev_good < 0)
                    signal[j] = signal[next]; // extrapolate from the right
                else if (next >= n)
                    signal[j] = signal[prev_good]; // extrapolate from the left
                else
                {
                    const double t = static_cast<double>(j - prev_good) / (next - prev_good);
                    signal[j] = signal[prev_good] + t * (signal[next] - signal[prev_good]);
                }
            }
            i = next;
        }

        return nan_count;
    }

} // namespace eemd

#endif // EEMD_SIGNAL_HPP

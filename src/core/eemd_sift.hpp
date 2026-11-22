/**
 * Sifting: when it stops (the SD rule, or the rule of Rilling, Flandrin and
 * Goncalves as in their emd.m), and the Sifter, which extracts one EMD mode
 * of a signal. The noise banks and the EEMD ensemble are built on it.
 *
 * License: MIT
 */

#ifndef EEMD_SIFT_HPP
#define EEMD_SIFT_HPP

#include "cubic_spline.hpp" // the envelopes
#include "eemd_extrema.hpp"
#include "eemd_threads.hpp" // a long series takes its two envelopes on two threads

namespace eemd
{

    /**
     * When sifting a mode stops
     * - SD:      sum(m^2) / sum(h^2) < sift_threshold, plus the S-number and
     *            extrema/zero-crossing early exits. Cheap; typically stops
     *            after 2-3 iterations at the default threshold of 0.05.
     * - Rilling: the criterion of Rilling, Flandrin & Goncalves (2003), as in
     *            their emd.m that the ICEEMDAN authors' reference code uses:
     *            with mode amplitude a = (U - L) / 2 and envelope mean m,
     *            sift while |m| > theta1 * a on more than a fraction alpha of
     *            the samples or |m| > theta2 * a anywhere, and stop only when
     *            the extrema and zero-crossing counts differ by at most one.
     */
    enum class SiftStopRule : uint8_t
    {
        SD = 0,
        Rilling = 1
    };

    /** Thresholds of SiftStopRule::Rilling (emd.m defaults [0.05, 0.5, 0.05]). */
    struct RillingThresholds
    {
        double theta1 = 0.05;
        double theta2 = 0.5;
        double alpha = 0.05;
    };

    // ============================================================================
    // Configuration
    // ============================================================================

    struct EEMDConfig
    {
        int32_t max_imfs = 10;
        int32_t max_sift_iters = 100;
        double sift_threshold = 0.05;
        int32_t ensemble_size = 100;
        double noise_std = 0.2;
        int32_t boundary_extend = 2;
        uint32_t rng_seed = 42;

        // S-number stopping criterion: stop after S consecutive iterations
        // where extrema count is stable. Benchmarks show S=6 gives ~12% speedup.
        // Set to 0 to disable (use only SD criterion).
        int32_t s_number = 6;

        // Sifting stop rule (see SiftStopRule); sift_threshold and s_number
        // apply to SD, rilling to Rilling.
        SiftStopRule stop_rule = SiftStopRule::SD;
        RillingThresholds rilling;

        // Envelope ends: false = mirror boundary_extend extrema plus a linear
        // end point (extend_extrema_raw, the original scheme); true = the
        // symmetric extension of emd.m (extend_extrema_symmetric), which stays
        // stable when sifting runs for many iterations.
        bool symmetric_boundary = false;

        // Threads to use: 0 = the process's thread setting (set_num_threads).
        // The result does not depend on it.
        int32_t num_threads = 0;
    };

    /**
     * SiftStopRule::Rilling: true when the mode h (with upper/lower
     * envelopes U, L and n_extrema extrema) should NOT be sifted further.
     */
    inline bool rilling_should_stop(const double *__restrict h,
                                    const double *__restrict upper,
                                    const double *__restrict lower,
                                    int32_t n, int32_t n_extrema,
                                    const RillingThresholds &th)
    {
        int64_t over = 0;
        int32_t big = 0;
        for (int32_t i = 0; i < n; ++i)
        {
            const double m = std::abs(0.5 * (upper[i] + lower[i]));
            const double a = 0.5 * std::abs(upper[i] - lower[i]);
            over += (m > th.theta1 * a);
            big |= (m > th.theta2 * a);
        }
        const bool keep_sifting = (static_cast<double>(over) > th.alpha * n || big) && n_extrema > 2;
        if (keep_sifting)
            return false;
        return std::abs(count_zero_crossings(h, n) - n_extrema) <= 1;
    }

    /**
     * A sifting step with SiftStopRule::Rilling after its envelopes, in one
     * pass over the samples: the decision of rilling_should_stop() on the
     * mode h (envelopes U, L, n_extrema extrema; its zero crossings counted on
     * the way) and the next mode, next = h - (U + L) / 2. True when h should
     * not be sifted further; next is then not to be used. next must not
     * overlap h, as emd.m keeps h itself when it stops.
     */
    inline bool rilling_step(const double *__restrict h,
                             const double *__restrict upper,
                             const double *__restrict lower,
                             int32_t n, int32_t n_extrema,
                             const RillingThresholds &th,
                             double *__restrict next)
    {
        int64_t over = 0;  // samples with |mean| > theta1 * amplitude
        int64_t big = 0;   // ... > theta2 * amplitude
        int64_t zeros = 0; // zero crossings of h, counted as count_zero_crossings() does
        const auto sample = [&](int32_t i)
        {
            const double u = upper[i], l = lower[i];
            const double mean = 0.5 * (u + l);
            const double m = std::abs(mean);
            const double a = 0.5 * std::abs(u - l);
            over += (m > th.theta1 * a);
            big |= (m > th.theta2 * a);
            next[i] = h[i] - mean;
            if (i > 0)
            {
                const double p = h[i - 1], c = h[i];
                zeros += ((p < 0.0) & (c > 0.0)) | ((p > 0.0) & (c < 0.0));
                zeros += (c == 0.0) & (p != 0.0);
            }
            else
                zeros += (h[0] == 0.0);
        };

        int32_t i = 0;
#if defined(__AVX2__)
        // Four samples at a time: the same arithmetic per sample as above.
        // A lane's comparison is all ones (-1) where true, so subtracting it
        // counts. The first four samples are done singly (h[i - 1] is read).
        for (; i < 4 && i < n; ++i)
            sample(i);
        if (n >= 8)
        {
            const __m256d half = _mm256_set1_pd(0.5);
            const __m256d sign = _mm256_set1_pd(-0.0);
            const __m256d zero = _mm256_setzero_pd();
            const __m256d t1 = _mm256_set1_pd(th.theta1);
            const __m256d t2 = _mm256_set1_pd(th.theta2);
            __m256i over_v = _mm256_setzero_si256();
            __m256i zeros_v = _mm256_setzero_si256();
            __m256d big_v = _mm256_setzero_pd();
            for (; i + 4 <= n; i += 4)
            {
                const __m256d u = _mm256_loadu_pd(upper + i);
                const __m256d l = _mm256_loadu_pd(lower + i);
                const __m256d c = _mm256_loadu_pd(h + i);
                const __m256d p = _mm256_loadu_pd(h + i - 1);
                const __m256d mean = _mm256_mul_pd(half, _mm256_add_pd(u, l));
                const __m256d m = _mm256_andnot_pd(sign, mean);
                const __m256d a = _mm256_mul_pd(half, _mm256_andnot_pd(sign, _mm256_sub_pd(u, l)));
                over_v = _mm256_sub_epi64(over_v, _mm256_castpd_si256(
                                                      _mm256_cmp_pd(m, _mm256_mul_pd(t1, a), _CMP_GT_OQ)));
                big_v = _mm256_or_pd(big_v, _mm256_cmp_pd(m, _mm256_mul_pd(t2, a), _CMP_GT_OQ));
                _mm256_storeu_pd(next + i, _mm256_sub_pd(c, mean));
                const __m256d cross = _mm256_or_pd(
                    _mm256_and_pd(_mm256_cmp_pd(p, zero, _CMP_LT_OQ), _mm256_cmp_pd(c, zero, _CMP_GT_OQ)),
                    _mm256_and_pd(_mm256_cmp_pd(p, zero, _CMP_GT_OQ), _mm256_cmp_pd(c, zero, _CMP_LT_OQ)));
                const __m256d run = _mm256_and_pd(_mm256_cmp_pd(c, zero, _CMP_EQ_OQ),
                                                  _mm256_cmp_pd(p, zero, _CMP_NEQ_UQ));
                zeros_v = _mm256_sub_epi64(zeros_v, _mm256_castpd_si256(cross));
                zeros_v = _mm256_sub_epi64(zeros_v, _mm256_castpd_si256(run));
            }
            alignas(32) int64_t lanes[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), over_v);
            over += lanes[0] + lanes[1] + lanes[2] + lanes[3];
            _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), zeros_v);
            zeros += lanes[0] + lanes[1] + lanes[2] + lanes[3];
            big |= (_mm256_movemask_pd(big_v) != 0);
        }
#endif
        for (; i < n; ++i)
            sample(i);

        const bool keep_sifting = (static_cast<double>(over) > th.alpha * n || big) && n_extrema > 2;
        if (keep_sifting)
            return false;
        return std::abs(zeros - n_extrema) <= 1;
    }

    // ============================================================================
    // Thread Scratch Pad - Fixed-Size Arrays with Counters
    // ============================================================================

    struct ThreadScratchPad
    {
        // Fixed-size arrays (resized once at construction)
        std::vector<int32_t> max_idx;
        std::vector<int32_t> min_idx;
        std::vector<double> ext_x;
        std::vector<double> ext_y;

        // Counters (avoids vector size() calls in hot path)
        int32_t n_max = 0;
        int32_t n_min = 0;
        int32_t n_ext = 0;
        int32_t ext_start = 0;

        // Second knot buffer pair (symmetric extension builds both envelopes at once)
        std::vector<double> ext_x2;
        std::vector<double> ext_y2;
        int32_t n_ext2 = 0;

        explicit ThreadScratchPad(int32_t n)
        {
            const int32_t max_extrema = n / 2 + 2;
            const int32_t ext_size = n + 128; // extrema + up to 2 x 31 mirrored knots + end points

            max_idx.resize(max_extrema);
            min_idx.resize(max_extrema);
            ext_x.resize(ext_size);
            ext_y.resize(ext_size);
            ext_x2.resize(ext_size);
            ext_y2.resize(ext_size);
        }
    };

    // ============================================================================
    // Sifter - Fused Loops, Raw Pointers
    // ============================================================================

    class Sifter
    {
    public:
        explicit Sifter(int32_t max_len, const EEMDConfig &cfg)
            : config_(cfg), max_len_(max_len), scratch_(max_len), work_(max_len), next_(max_len),
              upper_env_(max_len), lower_env_(max_len)
        {
        }

        bool sift_imf(double *signal, double *imf, int32_t n)
        {
            // The mode being sifted. With the Rilling rule a step writes the
            // next mode to the other buffer, and the two change places.
            double *w = work_.data;
            double *w_next = next_.data;
            std::memcpy(w, signal, n * sizeof(double));

            // Raw pointers for hot path
            int32_t *p_max = scratch_.max_idx.data();
            int32_t *p_min = scratch_.min_idx.data();
            double *p_ext_x = scratch_.ext_x.data();
            double *p_ext_y = scratch_.ext_y.data();

            // S-number tracking: count consecutive iterations with stable extrema
            int32_t prev_n_extrema = -1;
            int32_t s_count = 0;

            for (int32_t iter = 0; iter < config_.max_sift_iters; ++iter)
            {
                // Find extrema (raw pointer output)
                find_extrema_raw(w, n, p_max, scratch_.n_max, p_min, scratch_.n_min);

                const bool rilling = (config_.stop_rule == SiftStopRule::Rilling);

                // Enough extrema for envelopes: with the symmetric ends, three
                // in all (emd.m); otherwise two of each kind
                const bool enough = config_.symmetric_boundary
                                        ? (scratch_.n_max >= 1 && scratch_.n_min >= 1 &&
                                           scratch_.n_max + scratch_.n_min >= 3)
                                        : (scratch_.n_max >= 2 && scratch_.n_min >= 2);
                if (!enough)
                {
                    // Rilling (as emd.m): a mode that runs out of extrema
                    // mid-sift is kept as it stands
                    if (rilling && iter > 0)
                        break;
                    return false;
                }

                // S-number criterion: track consecutive stable extrema counts
                const int32_t n_extrema = scratch_.n_max + scratch_.n_min;
                if (!rilling && config_.s_number > 0)
                {
                    if (n_extrema == prev_n_extrema)
                    {
                        ++s_count;
                        if (s_count >= config_.s_number)
                        {
                            // Converged by S-number criterion
                            break;
                        }
                    }
                    else
                    {
                        s_count = 0;
                        prev_n_extrema = n_extrema;
                    }
                }

                // Knots of the upper envelope in (ext_x, ext_y), of the lower
                // one in (ext_x2, ext_y2)
                double *p_ext_x2 = scratch_.ext_x2.data();
                double *p_ext_y2 = scratch_.ext_y2.data();
                bool knots_ok = true;
                if (config_.symmetric_boundary)
                {
                    knots_ok = extend_extrema_symmetric(p_max, scratch_.n_max, p_min, scratch_.n_min,
                                                        w, n, config_.boundary_extend,
                                                        p_ext_x, p_ext_y, scratch_.n_ext,
                                                        p_ext_x2, p_ext_y2, scratch_.n_ext2);
                }
                else
                {
                    extend_extrema_raw(p_max, scratch_.n_max, w, n,
                                       config_.boundary_extend, p_ext_x, p_ext_y,
                                       scratch_.n_ext, scratch_.ext_start);
                    extend_extrema_raw(p_min, scratch_.n_min, w, n,
                                       config_.boundary_extend, p_ext_x2, p_ext_y2,
                                       scratch_.n_ext2, scratch_.ext_start);
                }

                // The two envelopes share nothing (a spline, a knot set and an
                // output each). A long series sifted outside a fork, as the
                // circular noise bank's is, takes them on two threads; inside
                // a fork team_size() is 1, and on one thread their two splines
                // are built together.
                bool env_ok[2] = {false, false};
                if (knots_ok)
                {
                    const int env_threads =
                        (n >= envelope_fork_min) ? std::min(2, threads::team_size(config_.num_threads)) : 1;
                    if (env_threads == 1)
                    {
                        env_ok[0] = env_ok[1] =
                            CubicSpline::construct_pair(upper_spline_, p_ext_x, p_ext_y, scratch_.n_ext,
                                                        lower_spline_, p_ext_x2, p_ext_y2, scratch_.n_ext2) &&
                            upper_spline_.evaluate_uniform(upper_env_.data, n) &&
                            lower_spline_.evaluate_uniform(lower_env_.data, n);
                    }
                    else
                    {
                        threads::for_static(env_threads, 2, [&](int32_t first, int32_t last, int)
                        {
                            for (int32_t e = first; e < last; ++e)
                            {
                                env_ok[e] = (e == 0)
                                                ? upper_spline_.construct(p_ext_x, p_ext_y, scratch_.n_ext) &&
                                                      upper_spline_.evaluate_uniform(upper_env_.data, n)
                                                : lower_spline_.construct(p_ext_x2, p_ext_y2, scratch_.n_ext2) &&
                                                      lower_spline_.evaluate_uniform(lower_env_.data, n);
                            }
                        });
                    }
                }
                if (!(env_ok[0] && env_ok[1]))
                {
                    if (config_.symmetric_boundary && rilling && iter > 0)
                        break; // keep the mode as it stands (emd.m)
                    return false;
                }

                const double *__restrict upper = upper_env_.data;
                const double *__restrict lower = lower_env_.data;

                if (rilling)
                {
                    // stop test, zero crossings and the next mode in one pass
                    if (rilling_step(w, upper, lower, n, n_extrema, config_.rilling, w_next))
                        break;
                    std::swap(w, w_next);
                    continue;
                }

                // Fused: compute mean, update work, calculate SD in single pass
                double sd_num = 0.0;
                double sd_den = 0.0;
                double *__restrict ws = w; // work_ itself: the SD rule does not swap

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    const double u = upper[i];
                    const double l = lower[i];
                    const double val = ws[i];
                    const double mean = 0.5 * (u + l);
                    const double diff = val - mean;

                    sd_num += mean * mean; // sum of mean^2
                    sd_den += val * val;   // sum of original^2

                    ws[i] = diff; // update in-place
                }

                const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;

                if (sd < config_.sift_threshold)
                    break;

                // Early termination: extrema ≈ zero crossings
                const int32_t n_zero = count_zero_crossings(w, n);

                if (std::abs(n_extrema - n_zero) <= 1 && sd < config_.sift_threshold * 10)
                {
                    break;
                }
            }

            std::memcpy(imf, w, n * sizeof(double));

            // Subtract IMF from signal
            double *__restrict sig = signal;
            const double *__restrict mode = w;

            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                sig[i] -= mode[i];
            }

            return true;
        }

    private:
        // Shortest series whose two envelopes are worth a fork
        static constexpr int32_t envelope_fork_min = 16384;

        const EEMDConfig &config_;
        int32_t max_len_;

        ThreadScratchPad scratch_;
        AlignedBuffer<double> work_;
        AlignedBuffer<double> next_; // the next mode of a Rilling step
        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;

        CubicSpline upper_spline_; // the envelopes: cubic_spline.hpp
        CubicSpline lower_spline_;
    };

} // namespace eemd

#endif // EEMD_SIFT_HPP

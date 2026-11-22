/**
 * ICEEMDAN (Improved Complete Ensemble EMD with Adaptive Noise)
 *
 * Algorithm: Colominas, Schlotthauer & Torres (2014). The defaults follow the
 * authors' reference code: local means from sifting to convergence with the
 * Rilling stop rule, the symmetric boundary extension of emd.m, cubic
 * (not-a-knot) envelopes, independent noise realizations, beta_k as in the
 * paper.
 *
 * Speed options (ICEEMDANConfig), none of them the published algorithm: the
 * SD stop rule, one envelope mean as the local mean, and the circular noise
 * bank (one long noise decomposition instead of one per trial).
 *
 * License: MIT
 */

#ifndef ICEEMDAN_HPP
#define ICEEMDAN_HPP

#include "eemd.hpp"

// One copy of a function's machine code for every caller: decompose() and
// decompose_batch() must compute the same bits, and under -ffast-math two
// inlined copies of the same code may sum in different orders
#if defined(_MSC_VER) && !defined(__clang__)
#define EEMD_NOINLINE __declspec(noinline)
#else
#define EEMD_NOINLINE __attribute__((noinline))
#endif
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>

namespace eemd
{

    /**
     * Local-mean operator M(.) of the ICEEMDAN stages
     *
     * Sifted:         M(x) = x - E1(x), with E1 the first EMD mode sifted to
     *                 convergence. This is the operator of Colominas et al.
     *                 (2014) and of their reference implementation. Default.
     * SingleEnvelope: M(x) = mean of the upper and lower envelopes of x, i.e.
     *                 one sifting iteration. Several times faster, but each
     *                 mode is under-sifted, so the IMFs differ from published
     *                 ICEEMDAN.
     */
    enum class LocalMeanMethod : uint8_t
    {
        Sifted = 0,
        SingleEnvelope = 1
    };

    // ============================================================================
    // ICEEMDAN Configuration
    // ============================================================================

    struct ICEEMDANConfig
    {
        // At most this many IMFs. The authors' code has no limit: it stops when
        // the residue needs no sifting, after about log2(N) modes; 32 is a
        // safety net that a signal does not reach in practice.
        int32_t max_imfs = 32;
        int32_t max_sift_iters = 100;
        int32_t ensemble_size = 100;
        // The noise amplitude, the paper's epsilon: beta_k = noise_std * std(r_k)
        double noise_std = 0.2;
        uint32_t rng_seed = 42;

        // Threads this decomposer uses: 0 = the process's thread setting
        // (set_num_threads). The result does not depend on it.
        int32_t num_threads = 0;

        // Extrema mirrored at each end by the symmetric boundary extension
        // (emd.m: NBSYM = 2)
        int32_t boundary_extend = 2;

        // Local-mean operator (see LocalMeanMethod). Sifted = the published
        // algorithm; SingleEnvelope = the faster one-iteration approximation.
        LocalMeanMethod local_mean_method = LocalMeanMethod::Sifted;

        // When sifting stops, for the local means and for the noise modes
        // (see SiftStopRule). Rilling = the criterion of the authors'
        // reference code (default); SD = sift_threshold + s_number, cheaper.
        SiftStopRule sift_stop_rule = SiftStopRule::Rilling;
        RillingThresholds rilling;
        // SD rule: its threshold, and the S-number (stop after S consecutive
        // iterations with an unchanged extrema count; 0 = off)
        double sift_threshold = 0.05;
        int32_t s_number = 6;
        // SD rule and SingleEnvelope: the decomposition ends when the residue
        // is monotonic to this tolerance or has fewer extrema than this. (The
        // published path ends as the authors' code does: when the residue
        // needs no sifting.)
        double monotonic_threshold = 1e-6;
        int32_t min_extrema = 3;

        // Noise modes from windows of ONE long decomposed realization instead
        // of independent realizations. Off by default, and never chosen by the
        // library itself: the windows of a long realization overlap and their
        // low-frequency modes are not those of a length-N realization, which
        // visibly changes the low-frequency IMFs (see README, "Accuracy").
        bool use_circular_bank = false;
        int32_t circular_multiplier = 20; // Long buffer = N x multiplier
        // Memory cap for the noise bank. The independent bank holds
        // noise_bank_bytes(N) (ensemble_size x modes x N doubles): above the
        // cap decompose() fails instead of exhausting memory or changing the
        // algorithm on its own. The circular bank (use_circular_bank) lowers
        // its multiplier to fit, not below 4; below that it fails as well.
        size_t noise_bank_max_bytes = size_t(2) << 30; // 2 GiB
        // The noise bank (the EMD modes of the noise realizations) depends on
        // the signal length, the seed and the sifting settings, not on the
        // signal. true: the decomposer keeps it, and the next decomposition of
        // the same length and settings skips that work, close to half of a
        // decomposition, with the same result bit for bit. false: it is freed
        // when decompose() returns. ICEEMDAN::release_noise_bank() frees it at
        // any time.
        bool keep_noise_bank = true;

        // An envelope through fewer knots than this is drawn with straight
        // lines instead of the spline
        int32_t spline_fast_threshold = 6;
    };

    /**
     * What a decomposition reports besides its result. Filled by every
     * decompose(); returned by decompose_with_diagnostics() and
     * last_diagnostics().
     */
    struct DecompositionDiagnostics
    {
        double orthogonality_index = 0.0;         // IO metric (lower = better)
        std::vector<int32_t> sift_iterations;     // Per-IMF: most sifting iterations any trial needed
        std::vector<double> mean_sift_iterations; // Per-IMF: mean iterations per trial
        std::vector<int32_t> nonconverged_trials; // Per-IMF: trials that hit max_sift_iters
        std::vector<bool> convergence_flags;      // Per-IMF: true if no trial hit max_sift_iters
        std::vector<int32_t> trials_per_imf;      // Per-IMF: local means averaged (valid trials)
        int32_t nan_count = 0;                    // NaN/Inf samples of the input, filled in
        uint32_t rng_seed_used = 0;               // The seed of the noise
        int32_t noise_bank_multiplier = 0;        // Circular bank length / N (0 = independent realizations)
        bool noise_bank_reused = false;           // The bank of an earlier call was used (config.keep_noise_bank)
        bool valid = true;                        // False if decomposition failed
    };

    // ============================================================================
    // Helper Functions
    // ============================================================================

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

    // ============================================================================
    // Fast Linear Interpolation (for sparse extrema)
    // ============================================================================

    inline void fast_linear_interp(
        const double *__restrict x_knots,
        const double *__restrict y_knots,
        int32_t n_knots,
        double *__restrict output,
        int32_t n_output)
    {
        if (n_knots < 2)
        {
            double val = (n_knots == 1) ? y_knots[0] : 0.0;
            for (int32_t i = 0; i < n_output; ++i)
            {
                output[i] = val;
            }
            return;
        }

        int32_t knot_idx = 0;

        for (int32_t i = 0; i < n_output; ++i)
        {
            double x = static_cast<double>(i);

            // Advance knot index if needed
            while (knot_idx < n_knots - 2 && x > x_knots[knot_idx + 1])
            {
                ++knot_idx;
            }

            // Handle boundaries
            if (x <= x_knots[0])
            {
                // Extrapolate left (with division-by-zero protection)
                double dx = x_knots[1] - x_knots[0];
                if (std::abs(dx) < 1e-10)
                    dx = 1.0; // Fallback to flat
                double slope = (y_knots[1] - y_knots[0]) / dx;
                output[i] = y_knots[0] + slope * (x - x_knots[0]);
            }
            else if (x >= x_knots[n_knots - 1])
            {
                // Extrapolate right (with division-by-zero protection)
                double dx = x_knots[n_knots - 1] - x_knots[n_knots - 2];
                if (std::abs(dx) < 1e-10)
                    dx = 1.0;
                double slope = (y_knots[n_knots - 1] - y_knots[n_knots - 2]) / dx;
                output[i] = y_knots[n_knots - 1] + slope * (x - x_knots[n_knots - 1]);
            }
            else
            {
                // Linear interpolation (with division-by-zero protection)
                double x0 = x_knots[knot_idx];
                double x1 = x_knots[knot_idx + 1];
                double y0 = y_knots[knot_idx];
                double y1 = y_knots[knot_idx + 1];
                double dx = x1 - x0;
                if (std::abs(dx) < 1e-10)
                {
                    output[i] = 0.5 * (y0 + y1); // Degenerate: average
                }
                else
                {
                    double t = (x - x0) / dx;
                    output[i] = y0 + t * (y1 - y0);
                }
            }
        }
    }

    // ============================================================================
    // Local Mean Computer
    // - Envelopes: cubic splines through the extrema, continued past the ends
    //   by the symmetric extension of emd.m; straight lines for sparse extrema
    // - Sifted mode: M(x) = x - E1(x), E1 sifted to convergence (the operator
    //   of Colominas et al. 2014); SingleEnvelope mode: one envelope mean
    // ============================================================================

    /** What one local-mean computation cost. */
    struct SiftStats
    {
        int32_t iterations = 0; // envelope means subtracted
        bool converged = true;  // false: stopped by max_sift_iters
        // Rilling only: the input was not sifted at all, because it has too
        // few extrema for envelopes or already passes the stop rule as it
        // stands. emd.m then returns one row, the input itself.
        bool unsifted = false;
    };

    /** Sifting parameters of the Sifted local-mean operator. */
    struct SiftParams
    {
        int32_t max_iters = 100;
        double threshold = 0.05; // SD rule: sum(m^2) / sum(h^2)
        int32_t s_number = 6;    // SD rule: stop after S iterations with a stable extrema count (0 = off)
        SiftStopRule rule = SiftStopRule::Rilling;
        RillingThresholds rilling;
    };

    class LocalMeanComputer
    {
    public:
        explicit LocalMeanComputer(int32_t max_len, int32_t boundary_extend,
                                   int32_t fast_threshold = 6,
                                   LocalMeanMethod method = LocalMeanMethod::Sifted,
                                   SiftParams sift = SiftParams())
            : max_len_(max_len),
              boundary_extend_(boundary_extend),
              fast_threshold_(fast_threshold),
              method_(method),
              sift_(sift),
              max_idx_(max_len / 2 + 2),
              min_idx_(max_len / 2 + 2),
              ext_x_(max_len + 128),
              ext_y_(max_len + 128),
              ext_x2_(max_len + 128),
              ext_y2_(max_len + 128),
              upper_env_(max_len),
              lower_env_(max_len),
              h_(method == LocalMeanMethod::Sifted ? max_len : 0)
        {
        }

        /**
         * local_mean = M(signal). Returns false when the signal has too few
         * extrema to build envelopes (local_mean is then a copy of signal and
         * the trial should not be averaged).
         */
        bool compute(const double *signal, int32_t n, double *local_mean, SiftStats *stats = nullptr)
        {
            if (method_ == LocalMeanMethod::Sifted)
                return compute_sifted(signal, n, local_mean, stats);

            if (stats)
            {
                stats->iterations = 1;
                stats->converged = true;
            }
            return envelope_mean(signal, n, local_mean);
        }

        int32_t get_n_extrema() const { return n_max_ + n_min_; }

        /**
         * Whether sifting x would change it: false when x has too few extrema
         * for envelopes or already passes the Rilling stop rule as it stands.
         * emd(x, 'MAXMODES', 1) returns exactly one row then, and the authors'
         * ceemdan.m stops the decomposition. Sifted method only.
         */
        bool sifting_needed(const double *x, int32_t n)
        {
            find_extrema_raw(x, n, max_idx_.data(), n_max_, min_idx_.data(), n_min_);
            const bool enough = n_max_ >= 1 && n_min_ >= 1 && n_max_ + n_min_ >= 3; // as emd.m
            if (!enough || !build_envelopes(x, n))
                return false;
            if (sift_.rule != SiftStopRule::Rilling)
                return true;
            return !rilling_should_stop(x, upper_env_.data, lower_env_.data, n, n_max_ + n_min_, sift_.rilling);
        }

    private:
        /**
         * Mean of the upper and lower envelopes of x (one sifting step).
         * Leaves the extrema counts of x in n_max_ / n_min_.
         */
        bool envelope_mean(const double *x, int32_t n, double *out_mean)
        {
            // Find extrema
            find_extrema_raw(x, n, max_idx_.data(), n_max_, min_idx_.data(), n_min_);

            if (n_max_ < 2 || n_min_ < 2)
            {
                std::memcpy(out_mean, x, n * sizeof(double));
                return false;
            }
            return envelope_mean_from_extrema(x, n, out_mean);
        }

        bool envelope_mean_from_extrema(const double *x, int32_t n, double *out_mean)
        {
            if (!build_envelopes(x, n))
            {
                std::memcpy(out_mean, x, n * sizeof(double));
                return false;
            }
            // Local mean = (upper + lower) / 2
            const double *__restrict upper = upper_env_.data;
            const double *__restrict lower = lower_env_.data;
            double *__restrict out = out_mean;
            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                out[i] = 0.5 * (upper[i] + lower[i]);
            }
            return true;
        }

        /**
         * The upper and lower envelopes of x, from its extrema in max_idx_ /
         * min_idx_, into upper_env_ / lower_env_. False when they cannot be
         * built.
         */
        bool build_envelopes(const double *x, int32_t n)
        {
            // the knots: the extrema, and those mirrored past each end (emd.m)
            int32_t n_up = 0, n_lo = 0;
            if (!extend_extrema_symmetric(max_idx_.data(), n_max_, min_idx_.data(), n_min_, x, n,
                                          boundary_extend_, ext_x_.data(), ext_y_.data(), n_up,
                                          ext_x2_.data(), ext_y2_.data(), n_lo))
                return false;
            return compute_envelopes(ext_x_.data(), ext_y_.data(), n_up, ext_x2_.data(), ext_y2_.data(), n_lo, n);
        }

        /**
         * M(x) = x - E1(x): sift the first mode to convergence and return what
         * is left. Same stopping rules as Sifter::sift_imf (SiftStopRule: SD
         * with S-number and extrema ~ zero crossings, or Rilling; max_iters).
         * A step that cannot build envelopes after the first iteration ends
         * the sifting and keeps the mode as it stands (as Rilling & Flandrin's
         * emd.m does); only a first step that cannot is a failure.
         */
        bool compute_sifted(const double *signal, int32_t n, double *local_mean, SiftStats *stats)
        {
            // The mode being sifted. With the Rilling rule a step writes the
            // next mode to the other buffer (local_mean, free until the end),
            // and the two change places.
            double *h = h_.data;
            double *h_next = local_mean;
            std::memcpy(h, signal, n * sizeof(double));

            int32_t prev_n_extrema = -1;
            int32_t s_count = 0;
            int32_t iterations = 0;
            bool converged = false;

            if (stats)
                stats->unsifted = false;
            for (int32_t iter = 0; iter < sift_.max_iters; ++iter)
            {
                find_extrema_raw(h, n, max_idx_.data(), n_max_, min_idx_.data(), n_min_);

                // Enough extrema for envelopes: three in all (emd.m)
                const bool enough = n_max_ >= 1 && n_min_ >= 1 && n_max_ + n_min_ >= 3;
                if (!enough)
                {
                    if (iter == 0)
                    {
                        if (stats)
                            stats->unsifted = true;
                        std::memcpy(local_mean, signal, n * sizeof(double));
                        return false;
                    }
                    converged = true; // nothing left to sift
                    break;
                }

                const int32_t n_extrema = n_max_ + n_min_;
                const bool rilling = (sift_.rule == SiftStopRule::Rilling);
                if (!rilling && sift_.s_number > 0)
                {
                    if (n_extrema == prev_n_extrema)
                    {
                        if (++s_count >= sift_.s_number)
                        {
                            converged = true;
                            break;
                        }
                    }
                    else
                    {
                        s_count = 0;
                        prev_n_extrema = n_extrema;
                    }
                }

                // The envelopes; for the SD rule also their mean, into
                // local_mean (scratch here)
                if (!(rilling ? build_envelopes(h, n) : envelope_mean_from_extrema(h, n, local_mean)))
                {
                    if (iter == 0)
                    {
                        if (stats)
                            stats->unsifted = true;
                        std::memcpy(local_mean, signal, n * sizeof(double));
                        return false;
                    }
                    break; // keep the mode as it stands
                }

                if (rilling)
                {
                    // stop test, zero crossings and the next mode in one pass
                    if (rilling_step(h, upper_env_.data, lower_env_.data, n, n_extrema, sift_.rilling, h_next))
                    {
                        if (stats && iter == 0)
                            stats->unsifted = true;
                        converged = true;
                        break;
                    }
                    std::swap(h, h_next);
                    ++iterations;
                    continue;
                }

                // Fused: SD criterion and h -= m in one pass
                double sd_num = 0.0;
                double sd_den = 0.0;
                const double *__restrict m = local_mean;
                double *__restrict hs = h; // h_ itself: the SD rule does not swap

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    const double val = hs[i];
                    const double mean = m[i];
                    sd_num += mean * mean;
                    sd_den += val * val;
                    hs[i] = val - mean;
                }
                ++iterations;

                const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;
                if (sd < sift_.threshold)
                {
                    converged = true;
                    break;
                }

                // Early termination: extrema ~ zero crossings
                const int32_t n_zero = count_zero_crossings(h, n);
                if (std::abs(n_extrema - n_zero) <= 1 && sd < sift_.threshold * 10)
                {
                    converged = true;
                    break;
                }
            }

            // M(x) = x - E1(x). The mode may be in local_mean itself: each
            // sample is read before it is written.
            const double *__restrict x = signal;
            double *out = local_mean;
            const double *mode = h;
            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                out[i] = x[i] - mode[i];
            }

            if (stats)
            {
                stats->iterations = iterations;
                stats->converged = converged;
            }
            return true;
        }

        /** One envelope through its knots: the cubic spline, or straight lines for sparse knots. */
        bool compute_envelope(const double *x_knots, const double *y_knots,
                              int32_t n_knots, double *envelope, int32_t n)
        {
            if (n_knots < fast_threshold_)
            {
                fast_linear_interp(x_knots, y_knots, n_knots, envelope, n);
                return true;
            }
            return spline_.construct(x_knots, y_knots, n_knots) && spline_.evaluate_uniform(envelope, n);
        }

        /**
         * Both envelopes, the same as compute_envelope() for the upper and
         * then the lower one. Two cubic splines are built together
         * (CubicSpline::construct_pair).
         */
        bool compute_envelopes(const double *xu, const double *yu, int32_t ku,
                               const double *xl, const double *yl, int32_t kl, int32_t n)
        {
            if (ku >= fast_threshold_ && kl >= fast_threshold_ &&
                CubicSpline::construct_pair(spline_, xu, yu, ku, spline2_, xl, yl, kl) &&
                spline_.evaluate_uniform(upper_env_.data, n) && spline2_.evaluate_uniform(lower_env_.data, n))
                return true;
            // otherwise, and should that fail, one after the other as before
            return compute_envelope(xu, yu, ku, upper_env_.data, n) && compute_envelope(xl, yl, kl, lower_env_.data, n);
        }

        int32_t max_len_;
        int32_t boundary_extend_;
        int32_t fast_threshold_;
        LocalMeanMethod method_;
        SiftParams sift_;

        std::vector<int32_t> max_idx_;
        std::vector<int32_t> min_idx_;
        std::vector<double> ext_x_;
        std::vector<double> ext_y_;
        std::vector<double> ext_x2_; // the lower envelope's knots
        std::vector<double> ext_y2_;

        int32_t n_max_ = 0;
        int32_t n_min_ = 0;

        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;
        AlignedBuffer<double> h_; // mode being sifted (Sifted mode only)

        CubicSpline spline_;  // the upper envelope (and both, one after the other)
        CubicSpline spline2_; // the lower envelope when the two are built together
    };

    // ============================================================================
    // Circular Noise Bank
    // Decomposes ONE long signal, provides sliding window views
    // ~5x faster initialization than M separate decompositions
    //
    // IMPORTANT STATISTICAL NOTE:
    // EMD is a local adaptive method - the stopping criterion depends on the
    // entire signal. When decomposing one giant noise signal (long_n = N * 20):
    //   - A high-energy event at index 15,000 might force extra sifting iterations
    //   - But the slice at index 0-1000 might have converged earlier
    //   - Result: Some slices are "over-sifted", others "under-sifted"
    //   - This creates weak temporal coupling in noise IMFs that standard EEMD lacks
    //
    // This is an inherent trade-off of the circular optimization. For most
    // applications the statistical impact is negligible, but for rigorous
    // hypothesis testing, consider using StandardNoiseBank instead.
    // ============================================================================

    class CircularNoiseBank
    {
    public:
        CircularNoiseBank() = default;

        /**
         * Returns false (and stays uninitialized) when even the smallest
         * buffer (multiplier 4) would exceed max_bytes.
         */
        bool initialize(int32_t n, int32_t ensemble_size, int32_t max_imfs,
                        const EEMDConfig &emd_config, uint32_t base_seed,
                        int32_t multiplier = 20,
                        size_t max_bytes = size_t(2) << 30)
        {
            n_ = n;
            max_imfs_ = max_imfs;
            initialized_ = false;

            // Long buffer provides statistical variance without O(M) decomposition cost
            multiplier = std::max(multiplier, ensemble_size / 5);

            // Memory cap: max_imfs buffers of N * multiplier doubles
            const size_t bytes_per_unit = static_cast<size_t>(max_imfs) * static_cast<size_t>(n) * sizeof(double);
            const size_t affordable = bytes_per_unit > 0 ? max_bytes / bytes_per_unit : 0;
            if (static_cast<size_t>(multiplier) > affordable)
            {
                if (affordable < 4)
                    return false;
                multiplier = static_cast<int32_t>(std::min<size_t>(affordable, 1u << 30));
            }
            // The long buffer is sifted with int32 indices
            const int64_t int_limit = (static_cast<int64_t>(INT32_MAX) - 64) / std::max(n, 1);
            if (multiplier > int_limit)
            {
                if (int_limit < 4)
                    return false;
                multiplier = static_cast<int32_t>(int_limit);
            }
            multiplier_ = multiplier;
            long_n_ = static_cast<int64_t>(n) * multiplier;
            max_offset_ = static_cast<size_t>(long_n_ - n_);

            // One window per trial, shared by every stage k: trial t uses the
            // modes E_1..E_K of the SAME stretch of noise, as ICEEMDAN uses
            // E_k(w^(t)) of one realization w^(t) at every stage.
            offset_table_.resize(ensemble_size);
            for (int32_t t = 0; t < ensemble_size; ++t)
            {
                // Deterministic hash (splitmix64 finalizer) of the trial index
                uint64_t z = static_cast<uint64_t>(t) + 0x9E3779B97F4A7C15ULL;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                z ^= z >> 31;
                offset_table_[t] = static_cast<size_t>(z % max_offset_);
            }

            // Allocate master buffers [max_imfs][long_n]
            master_imfs_.resize(max_imfs);
            for (auto &buf : master_imfs_)
            {
                buf.resize(static_cast<size_t>(long_n_));
            }

            // Generate ONE long white noise signal
            AlignedBuffer<double> long_noise(static_cast<size_t>(long_n_));
            AlignedBuffer<double> work(static_cast<size_t>(long_n_));

            // its own stream of the seed, apart from the trials' streams
            rng::gaussian(base_seed, rng::circular_stream, long_n_, long_noise.data);

            // Decompose the long signal ONCE
            const int32_t long_n32 = static_cast<int32_t>(long_n_);
            Sifter sifter(long_n32, emd_config);
            std::memcpy(work.data, long_noise.data, static_cast<size_t>(long_n_) * sizeof(double));

            actual_imfs_ = 0;
            for (int32_t k = 0; k < max_imfs; ++k)
            {
                if (!sifter.sift_imf(work.data, master_imfs_[k].data(), long_n32))
                {
                    // Fill remaining with zeros
                    for (int32_t z = k; z < max_imfs; ++z)
                    {
                        std::memset(master_imfs_[z].data(), 0, static_cast<size_t>(long_n_) * sizeof(double));
                    }
                    break;
                }
                ++actual_imfs_;
            }

            initialized_ = true;
            return true;
        }

        /**
         * Get a slice of pre-computed noise IMF
         * O(1) lookup via precomputed offset table (no modulo in hot path)
         */
        const double *get_noise_slice(int32_t trial_idx, int32_t imf_idx) const
        {
            if (!initialized_ || imf_idx >= max_imfs_ ||
                trial_idx >= static_cast<int32_t>(offset_table_.size()))
            {
                return nullptr;
            }

            return &master_imfs_[imf_idx][offset_table_[trial_idx]];
        }

        int32_t get_actual_imfs() const { return actual_imfs_; }
        int32_t multiplier() const { return multiplier_; }
        bool is_initialized() const { return initialized_; }

        /** Frees the buffers. */
        void release()
        {
            std::vector<std::vector<double>>().swap(master_imfs_);
            std::vector<size_t>().swap(offset_table_);
            initialized_ = false;
        }

    private:
        int32_t n_ = 0;
        int64_t long_n_ = 0;
        size_t max_offset_ = 0;
        int32_t max_imfs_ = 0;
        int32_t actual_imfs_ = 0;
        int32_t multiplier_ = 0;
        bool initialized_ = false;
        std::vector<size_t> offset_table_; // [trial] -> offset, same for every stage
        std::vector<std::vector<double>> master_imfs_;
    };

    // ============================================================================
    // Legacy Noise Bank (for comparison / fallback)
    // ============================================================================

    class StandardNoiseBank
    {
    public:
        StandardNoiseBank() = default;

        /**
         * The modes of each noise realization as the authors' code takes them
         * (ceemdan.m uses the rows of emd.m's output): the IMFs, then the
         * residue as one more mode unless it is all zero. At most max_modes
         * in all; each realization keeps only the modes it has.
         */
        void initialize(int32_t n, int32_t ensemble_size, int32_t max_modes,
                        const EEMDConfig &emd_config, uint32_t base_seed)
        {
            n_ = n;
            ensemble_size_ = ensemble_size;
            max_imfs_ = max_modes;

            noise_imfs_.assign(ensemble_size, {});
            imf_counts_.assign(ensemble_size, 0);

            threads::for_static(threads::team_size(emd_config.num_threads), ensemble_size,
                                [&](int32_t i_begin, int32_t i_end, int)
            {
                if (i_begin == i_end)
                    return;

                AlignedBuffer<double> work(n);
                Sifter sifter(n, emd_config);

                for (int32_t i = i_begin; i < i_end; ++i)
                {
                    // Realization i depends on (seed, i) only: the same at any thread count
                    if (!trial_gaussian(base_seed, static_cast<uint32_t>(i), n, work.data))
                    {
                        work.zero();
                    }

                    std::vector<std::vector<double>> &modes = noise_imfs_[i];
                    while (static_cast<int32_t>(modes.size()) < max_modes)
                    {
                        std::vector<double> imf(n);
                        if (!sifter.sift_imf(work.data, imf.data(), n))
                            break;
                        modes.push_back(std::move(imf));
                    }
                    // the residue, as emd.m returns it: when not all zero
                    bool any = false;
                    for (int32_t j = 0; j < n && !any; ++j)
                        any = work.data[j] != 0.0;
                    if (any && static_cast<int32_t>(modes.size()) < max_modes)
                        modes.emplace_back(work.data, work.data + n);
                    imf_counts_[i] = static_cast<int32_t>(modes.size());
                }
            });

            initialized_ = true;
        }

        const double *get_noise_imf(int32_t trial_idx, int32_t imf_idx) const
        {
            if (trial_idx >= ensemble_size_ || imf_idx >= imf_counts_[trial_idx])
            {
                return nullptr;
            }
            return noise_imfs_[trial_idx][imf_idx].data();
        }

        bool is_initialized() const { return initialized_; }

        /** Frees the buffers. */
        void release()
        {
            std::vector<std::vector<std::vector<double>>>().swap(noise_imfs_);
            std::vector<int32_t>().swap(imf_counts_);
            initialized_ = false;
        }

    private:
        int32_t n_ = 0;
        int32_t ensemble_size_ = 0;
        int32_t max_imfs_ = 0;
        bool initialized_ = false;
        std::vector<std::vector<std::vector<double>>> noise_imfs_;
        std::vector<int32_t> imf_counts_;
    };

    // ============================================================================
    // ICEEMDAN Optimized
    // ============================================================================

    class ICEEMDAN
    {
    public:
        explicit ICEEMDAN(const ICEEMDANConfig &config = ICEEMDANConfig())
            : config_(config)
        {
        }

        /**
         * Decompose signal into IMFs using ICEEMDAN (Colominas et al. 2014)
         *
         *   r_0 = x
         *   r_k = < M( r_{k-1} + beta_{k-1} E_k(w^(i)) ) >_i     (average over trials)
         *   IMF_k = r_{k-1} - r_k
         *   beta_0 = eps * std(x) / std(E_1(w^(i))),  beta_k = eps * std(r_k)  (k >= 1)
         *
         * with eps = noise_std, E_k the k-th EMD mode of white noise and M the
         * local-mean operator (local_mean_method). NaN and Inf samples of the
         * signal are filled in first, by straight lines from their neighbours.
         *
         * Reproducibility: for a given seed and configuration the result does
         * not depend on the number of threads (bit for bit). Trials are
         * summed in fixed blocks, in trial order, and the blocks are reduced in
         * block order; each realization's noise depends on (seed, trial) only.
         */
        bool decompose(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue)
        {
            last_diag_ = DecompositionDiagnostics();
            last_diag_.rng_seed_used = config_.rng_seed;
            last_diag_.valid = false;
            if (n < 4 || config_.ensemble_size < 1 || config_.max_imfs < 1)
                return false;

            BankGuard bank_guard{this}; // frees the bank afterwards unless it is kept
            BankUse bank;
            if (!prepare_bank(n, bank))
                return false;
            return run(signal, n, bank, imfs, residue, last_diag_);
        }

        /**
         * Decomposes count signals of n samples each, stored one after another
         * (signal s at signals + s * n): the channels of a recording, or its
         * epochs. Every result is the same, bit for bit, as decompose() gives
         * for that signal alone with these settings.
         *
         * The signals share one noise bank, built once or kept from an earlier
         * call, and their trials are interleaved on the library's threads: a
         * thread that would wait for the end of one signal's stage takes trials
         * of the next signal instead (run_interleaved()).
         *
         * imfs[s] and residues[s] receive signal s's IMFs and residue; diags,
         * when given, its diagnostics.
         * last_diagnostics() afterwards holds the last signal's. False when the
         * arguments are bad or the noise bank cannot be built.
         */
        bool decompose_batch(
            const double *signals,
            int32_t count,
            int32_t n,
            std::vector<std::vector<std::vector<double>>> &imfs,
            std::vector<std::vector<double>> &residues,
            std::vector<DecompositionDiagnostics> *diags = nullptr)
        {
            last_diag_ = DecompositionDiagnostics();
            imfs.assign(static_cast<size_t>(std::max(count, 0)), {});
            residues.assign(static_cast<size_t>(std::max(count, 0)), {});
            if (diags)
                diags->assign(static_cast<size_t>(std::max(count, 0)), DecompositionDiagnostics());
            if (!signals || count < 1 || n < 4 || config_.ensemble_size < 1 || config_.max_imfs < 1)
                return false;

            BankGuard bank_guard{this};
            BankUse bank;
            if (!prepare_bank(n, bank))
                return false;

            std::vector<DecompositionDiagnostics> d(static_cast<size_t>(count));
            const int32_t team = threads::team_size(config_.num_threads);
            if (count == 1 || team <= 1)
            {
                for (int32_t s = 0; s < count; ++s)
                    run(signals + static_cast<size_t>(s) * n, n, bank, imfs[s], residues[s], d[s]);
            }
            else
            {
                run_interleaved(signals, count, n, bank, team, imfs, residues, d);
            }

            last_diag_ = d.back();
            if (diags)
                *diags = std::move(d);
            return true;
        }

    private:
        /** What prepare_bank() left for run(): which bank, and whether it was kept from before. */
        struct BankUse
        {
            bool circular = false;
            bool reused = false;
        };

        /** Frees the noise bank when the decomposer is not to keep it (config.keep_noise_bank). */
        struct BankGuard
        {
            ICEEMDAN *self;
            ~BankGuard()
            {
                if (!self->config_.keep_noise_bank)
                    self->release_noise_bank();
            }
        };

        /** The sifting settings of the noise bank's EMD. */
        EEMDConfig bank_sifting() const
        {
            EEMDConfig emd_config;
            emd_config.max_imfs = config_.max_imfs;
            emd_config.max_sift_iters = config_.max_sift_iters;
            emd_config.sift_threshold = config_.sift_threshold;
            emd_config.boundary_extend = config_.boundary_extend;
            emd_config.s_number = config_.s_number;
            emd_config.stop_rule = config_.sift_stop_rule;
            emd_config.rilling = config_.rilling;
            emd_config.num_threads = config_.num_threads;
            emd_config.symmetric_boundary = true; // the boundary extension of emd.m, as for the signal
            return emd_config;
        }

        /** Realizations in the bank: one per trial. */
        int32_t base_trials() const
        {
            return config_.ensemble_size;
        }

        /**
         * The modes the bank holds for each noise realization of length n: a
         * realization has about log2(n) of them.
         */
        int32_t bank_modes(int32_t n) const
        {
            int32_t modes = 2;
            for (int32_t len = n; len > 1; len >>= 1)
                ++modes;
            return std::min(modes, config_.max_imfs);
        }

        /**
         * The noise bank for signals of length n (independent realizations,
         * or circular with use_circular_bank): kept from the last call when
         * everything it depends on is the same, built otherwise. False when
         * it would not fit in noise_bank_max_bytes.
         */
        bool prepare_bank(int32_t n, BankUse &use)
        {
            const EEMDConfig emd_config = bank_sifting();
            const int32_t trials = base_trials();
            const int32_t bank_modes = this->bank_modes(n);
            const bool use_circular = config_.use_circular_bank;
            if (!use_circular && noise_bank_bytes(n) > config_.noise_bank_max_bytes)
                return false; // the library does not change the algorithm on its own

            NoiseBankKey key;
            key.n = n;
            key.ensemble_size = config_.ensemble_size;
            key.base_trials = trials;
            key.seed = config_.rng_seed;
            key.circular = use_circular;
            key.circular_multiplier = config_.circular_multiplier;
            key.max_bytes = config_.noise_bank_max_bytes;
            key.sifting = emd_config;

            use.circular = use_circular;
            use.reused = bank_valid_ && key == bank_key_;
            if (!use.reused)
            {
                release_noise_bank();
                if (use_circular)
                {
                    if (!circular_bank_.initialize(n, config_.ensemble_size, bank_modes,
                                                   emd_config, config_.rng_seed,
                                                   config_.circular_multiplier,
                                                   config_.noise_bank_max_bytes))
                    {
                        return false; // noise bank would not fit in noise_bank_max_bytes
                    }
                }
                else
                {
                    standard_bank_.initialize(n, trials, bank_modes, emd_config, config_.rng_seed);
                }
                bank_key_ = key;
                bank_valid_ = true;
            }
            return true;
        }

        /**
         * IO = sum over pairs |<IMF_i, IMF_j>| / sum <IMF_i, IMF_i> (lower =
         * better). Given the 64-byte aligned buffers of run(): a vectorized
         * sum can depend on where its data starts, so it is taken where that
         * is always the same.
         */
        EEMD_NOINLINE static double orthogonality_index(const AlignedBuffer<double> *imfs, int32_t count, int32_t n)
        {
            double cross_sum = 0.0;
            double auto_sum = 0.0;
            for (int32_t i = 0; i < count; ++i)
            {
                const double *a = imfs[i].data;
                double norm_i = 0.0;
                for (int32_t k = 0; k < n; ++k)
                    norm_i += a[k] * a[k];
                auto_sum += norm_i;
                for (int32_t j = i + 1; j < count; ++j)
                {
                    const double *b = imfs[j].data;
                    double inner = 0.0;
                    for (int32_t k = 0; k < n; ++k)
                        inner += a[k] * b[k];
                    cross_sum += std::abs(inner);
                }
            }
            return (auto_sum > 1e-10) ? cross_sum / auto_sum : 0.0;
        }

        /** The sifting settings of the trials (and of the stop test). */
        static SiftParams sift_params(const ICEEMDANConfig &c)
        {
            return SiftParams{c.max_sift_iters, c.sift_threshold, c.s_number, c.sift_stop_rule, c.rilling};
        }

        /**
         * A thread's working memory for the trials of signals of one length.
         * Kept in the decomposer from call to call (and freed with the noise
         * bank), so a decomposition does not allocate it again.
         */
        struct TrialWorkspace
        {
            int32_t n;
            ICEEMDANConfig made_for; // the settings lm_computer was made with
            LocalMeanComputer lm_computer;
            AlignedBuffer<double> perturbed;
            AlignedBuffer<double> local_mean;

            TrialWorkspace(int32_t len, const ICEEMDANConfig &c)
                : n(len), made_for(c),
                  lm_computer(len, c.boundary_extend, c.spline_fast_threshold, c.local_mean_method, sift_params(c)),
                  perturbed(len), local_mean(len)
            {
            }

            /** Whether it serves signals of length len with the settings c. */
            bool fits(int32_t len, const ICEEMDANConfig &c) const
            {
                const ICEEMDANConfig &m = made_for;
                return n == len && m.boundary_extend == c.boundary_extend &&
                       m.spline_fast_threshold == c.spline_fast_threshold &&
                       m.local_mean_method == c.local_mean_method && m.max_sift_iters == c.max_sift_iters &&
                       m.sift_threshold == c.sift_threshold && m.s_number == c.s_number &&
                       m.sift_stop_rule == c.sift_stop_rule && m.rilling.theta1 == c.rilling.theta1 &&
                       m.rilling.theta2 == c.rilling.theta2 && m.rilling.alpha == c.rilling.alpha;
            }
        };

        /** The threads' workspaces, by slot. A copy of the decomposer starts without any. */
        struct Workspaces
        {
            std::vector<std::unique_ptr<TrialWorkspace>> slots;

            Workspaces() = default;
            Workspaces(const Workspaces &) {}
            Workspaces &operator=(const Workspaces &)
            {
                slots.clear();
                return *this;
            }
            Workspaces(Workspaces &&) = default;
            Workspaces &operator=(Workspaces &&) = default;
        };

        /** Room for team workspaces; called before a fork, inside which each slot touches its own only. */
        void reserve_workspaces(int team)
        {
            if (static_cast<int>(workspaces_.slots.size()) < team)
                workspaces_.slots.resize(static_cast<size_t>(team));
        }

        /** The workspace of a slot, for signals of length n with the current settings. */
        TrialWorkspace &workspace(int slot, int32_t n)
        {
            std::unique_ptr<TrialWorkspace> &w = workspaces_.slots[static_cast<size_t>(slot)];
            if (!w || !w->fits(n, config_))
                w.reset(new TrialWorkspace(n, config_));
            return *w;
        }

        /**
         * One signal's decomposition on the bank prepare_bank() made, step by
         * step: start(); then, while next_stage() gives a stage, the blocks of
         * its trials (block(), in any order, on any thread), their sum
         * (reduce(), over any split of the samples) and end_stage(); finally
         * finish(). It reads the decomposer and its bank and writes only its
         * own buffers, so several signals can be decomposed at once. A block's
         * sum depends only on its trials and the blocks are added in block
         * order, so the result does not depend on which thread does what.
         */
        class SignalRun
        {
        public:
            SignalRun(const ICEEMDAN &d, const BankUse &bank, int32_t n)
                : d_(d), c_(d.config_), bank_(bank), n_(n), trials_(d.config_.ensemble_size),
                  reference_path_(d.config_.sift_stop_rule == SiftStopRule::Rilling &&
                                  d.config_.local_mean_method == LocalMeanMethod::Sifted),
                  signal_clean_(n), r_current_(n), mean_acc_(n),
                  imf_storage_(static_cast<size_t>(d.config_.max_imfs))
            {
                for (auto &buf : imf_storage_)
                    buf.resize(n);

                // Deterministic reduction layout: trials are summed in blocks
                // of block_trials (in trial order), blocks are reduced in
                // block order. Nothing here depends on the number of threads.
                block_trials_ = std::max(4, (trials_ + 63) / 64); // at most 64 blocks
                n_blocks_ = (trials_ + block_trials_ - 1) / block_trials_;
                block_accs_ = std::vector<AlignedBuffer<double>>(static_cast<size_t>(n_blocks_));
                for (auto &acc : block_accs_)
                    acc.resize(n);
                block_stats_.resize(static_cast<size_t>(n_blocks_));
            }

            /** Begins the decomposition of signal (n samples); diag receives its diagnostics. */
            EEMD_NOINLINE void start(const double *signal, DecompositionDiagnostics &diag)
            {
                diag_ = &diag;
                diag = DecompositionDiagnostics();
                diag.rng_seed_used = c_.rng_seed;
                diag.valid = false;

                // NaN and Inf samples filled in; finite ones are left as they are
                std::memcpy(signal_clean_.data, signal, n_ * sizeof(double));
                diag.nan_count = sanitize_signal(signal_clean_.data, n_);
                const double *clean_signal = signal_clean_.data;

                // The noise amplitudes beta_k use the standard deviation with n - 1,
                // as ceemdan.m (MATLAB's std) does
                signal_sample_std_ = std::max(amplitude_floor, compute_sample_std(clean_signal, n_));

                diag.noise_bank_reused = bank_.reused;
                if (bank_.circular)
                    diag.noise_bank_multiplier = d_.circular_bank_.multiplier();

                std::memcpy(r_current_.data, clean_signal, n_ * sizeof(double));
                actual_imf_count_ = 0;
                stop_ = false;
                stage_scale_ = 0.0;
                k_ = 0;
                begin_stage();
            }

            /** Whether there is another stage (another IMF to extract); false when the decomposition is complete. */
            bool next_stage() const { return !finished_; }

            /** Blocks of trials in a stage. */
            int32_t blocks() const { return n_blocks_; }

            /** The trials of block b: their local means summed in trial order. */
            EEMD_NOINLINE void block(int32_t b, TrialWorkspace &ws)
            {
                const int32_t n = n_;
                const int32_t k = k_;
                LocalMeanComputer &lm_computer = ws.lm_computer;
                AlignedBuffer<double> &tl_perturbed = ws.perturbed;
                AlignedBuffer<double> &tl_local_mean = ws.local_mean;

                double *__restrict acc = block_accs_[b].data;
                std::memset(acc, 0, n * sizeof(double));
                BlockStats st;

                const int32_t t_begin = b * block_trials_;
                const int32_t t_end = std::min(trials_, t_begin + block_trials_);

                for (int32_t i = t_begin; i < t_end; ++i)
                {
                    const double *noise_imf = noise(i, k);

                    // Amplitude of this trial's noise mode
                    double amp = stage_scale_;
                    if (noise_imf && k == 0)
                    {
                        // beta_0 = eps * std(x) / std(E_1(w^(i)))
                        const double sw = compute_sample_std(noise_imf, n);
                        amp = (sw > 1e-300) ? stage_scale_ / sw : 0.0;
                    }

                    // the residue plus this trial's noise mode (a realization
                    // that has no mode k adds nothing)
                    const double *__restrict r = r_current_.data;
                    double *__restrict p = tl_perturbed.data;
                    if (!noise_imf)
                    {
                        std::memcpy(p, r, n * sizeof(double));
                    }
                    else
                    {
                        const double a = amp;
                        const double *__restrict nz = noise_imf;
                        EEMD_OMP_SIMD
                        for (int32_t j = 0; j < n; ++j)
                        {
                            p[j] = r[j] + a * nz[j];
                        }
                    }

                    SiftStats ss;
                    ++st.attempted;
                    const bool lm_ok = lm_computer.compute(tl_perturbed.data, n, tl_local_mean.data, &ss);
                    if (reference_path_ && ss.unsifted)
                    {
                        // emd.m returned one row, the input itself:
                        // ceemdan.m's first stage takes the input
                        // minus that row (a local mean of 0), the
                        // later stages that row (the input)
                        ++st.valid;
                        if (k > 0)
                        {
                            const double *__restrict pin = tl_perturbed.data;
                            EEMD_OMP_SIMD
                            for (int32_t j = 0; j < n; ++j)
                            {
                                acc[j] += pin[j];
                            }
                        }
                        continue;
                    }
                    if (lm_ok)
                    {
                        ++st.valid;
                        st.iterations += ss.iterations;
                        st.max_iterations = std::max(st.max_iterations, ss.iterations);
                        st.nonconverged += ss.converged ? 0 : 1;

                        const double *__restrict lm = tl_local_mean.data;
                        EEMD_OMP_SIMD
                        for (int32_t j = 0; j < n; ++j)
                        {
                            acc[j] += lm[j];
                        }
                    }
                }
                block_stats_[b] = st;
            }

            /** Adds the stage's blocks, in block order, to the stage sum: samples [j_begin, j_end). */
            EEMD_NOINLINE void reduce(int32_t j_begin, int32_t j_end)
            {
                for (int32_t j = j_begin; j < j_end; ++j)
                {
                    double sum = 0.0;
                    for (int32_t b = 0; b < n_blocks_; ++b)
                    {
                        sum += block_accs_[b].data[j];
                    }
                    mean_acc_.data[j] += sum;
                }
            }

            /**
             * After reduce(): the stage's IMF and residue, whether the
             * decomposition goes on, and the next stage (stop_test: a
             * LocalMeanComputer this thread may use).
             */
            EEMD_NOINLINE void end_stage(LocalMeanComputer &stop_test)
            {
                const int32_t n = n_;
                BlockStats total;
                for (int32_t b = 0; b < n_blocks_; ++b)
                {
                    const BlockStats &st = block_stats_[b];
                    total.valid += st.valid;
                    total.attempted += st.attempted;
                    total.iterations += st.iterations;
                    total.max_iterations = std::max(total.max_iterations, st.max_iterations);
                    total.nonconverged += st.nonconverged;
                }

                if (total.valid > 0)
                {
                    const double scale = 1.0 / total.valid;
                    double *__restrict acc = mean_acc_.data;

                    EEMD_OMP_SIMD
                    for (int32_t j = 0; j < n; ++j)
                    {
                        acc[j] *= scale;
                    }

                    // Extract IMF
                    const double *__restrict r = r_current_.data;
                    const double *__restrict m = mean_acc_.data;
                    double *__restrict out = imf_storage_[actual_imf_count_].data;

                    EEMD_OMP_SIMD
                    for (int32_t j = 0; j < n; ++j)
                    {
                        out[j] = r[j] - m[j];
                    }

                    ++actual_imf_count_;

                    // Diagnostics for this IMF
                    DecompositionDiagnostics &diag = *diag_;
                    diag.sift_iterations.push_back(total.max_iterations);
                    diag.mean_sift_iterations.push_back(static_cast<double>(total.iterations) / total.valid);
                    diag.nonconverged_trials.push_back(total.nonconverged);
                    diag.convergence_flags.push_back(total.nonconverged == 0);
                    diag.trials_per_imf.push_back(total.valid);

                    // Update residue
                    std::memcpy(r_current_.data, mean_acc_.data, n * sizeof(double));

                    // Check stopping criteria
                    if (reference_path_)
                    {
                        // ceemdan.m goes on while emd(r, 'MAXMODES', 1) gives a
                        // mode and a residue: r has enough extrema and is not
                        // already a mode as it stands
                        if (!stop_test.sifting_needed(r_current_.data, n))
                            stop_ = true;
                    }
                    else if (is_monotonic(r_current_.data, n, c_.monotonic_threshold) ||
                             count_extrema(r_current_.data, n) < c_.min_extrema)
                    {
                        stop_ = true;
                    }
                }
                else
                {
                    stop_ = true;
                }
                ++k_;
                begin_stage();
            }

            /** The IMFs and the residue; the diagnostics complete. */
            EEMD_NOINLINE void finish(std::vector<std::vector<double>> &imfs, std::vector<double> &residue)
            {
                imfs.clear();
                imfs.resize(static_cast<size_t>(actual_imf_count_));
                for (int32_t k = 0; k < actual_imf_count_; ++k)
                {
                    imfs[k].resize(n_);
                    std::memcpy(imfs[k].data(), imf_storage_[k].data, n_ * sizeof(double));
                }

                residue.resize(n_);
                std::memcpy(residue.data(), r_current_.data, n_ * sizeof(double));

                diag_->orthogonality_index = orthogonality_index(imf_storage_.data(), actual_imf_count_, n_);
                diag_->valid = true;
            }

        private:
            // The standard deviation that scales the noise is taken as at least this
            static constexpr double amplitude_floor = 1e-8;

            struct BlockStats
            {
                int32_t valid = 0;
                int32_t attempted = 0;
                int64_t iterations = 0;
                int32_t max_iterations = 0;
                int32_t nonconverged = 0;
            };

            /** Noise mode imf of trial's realization (the bank's). */
            const double *noise(int32_t trial, int32_t imf) const
            {
                return bank_.circular ? d_.circular_bank_.get_noise_slice(trial, imf)
                                      : d_.standard_bank_.get_noise_imf(trial, imf);
            }

            /** Stage k_: its sum reset and its noise amplitude; or the end. */
            void begin_stage()
            {
                finished_ = stop_ || k_ >= c_.max_imfs;
                if (finished_)
                    return;
                mean_acc_.zero();

                // beta_k = eps * std(r_k) for k >= 1; for k = 0 the
                // per-trial factor std(x) / std(E_1(w)) is applied in block()
                stage_scale_ = (k_ == 0) ? c_.noise_std * signal_sample_std_
                                         : c_.noise_std * compute_sample_std(r_current_.data, n_);
            }

            const ICEEMDAN &d_;
            const ICEEMDANConfig &c_;
            const BankUse bank_;
            const int32_t n_;
            const int32_t trials_;
            // The published algorithm (Rilling stop rule, sifted local means)
            // follows the authors' ceemdan.m also where the paper leaves
            // details open: a trial whose input is not sifted at all, and when
            // the decomposition stops (end_stage())
            const bool reference_path_;
            int32_t block_trials_ = 4;
            int32_t n_blocks_ = 0;

            AlignedBuffer<double> signal_clean_;
            AlignedBuffer<double> r_current_;
            AlignedBuffer<double> mean_acc_;
            std::vector<AlignedBuffer<double>> imf_storage_;
            std::vector<AlignedBuffer<double>> block_accs_;
            std::vector<BlockStats> block_stats_;

            DecompositionDiagnostics *diag_ = nullptr;
            double signal_sample_std_ = 0.0;
            double stage_scale_ = 0.0;
            int32_t k_ = 0;
            int32_t actual_imf_count_ = 0;
            bool stop_ = false;
            bool finished_ = true;
        };

        /** One signal's decomposition, its trials forked to the library's threads. */
        EEMD_NOINLINE bool run(
            const double *signal,
            int32_t n,
            const BankUse &bank,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue,
            DecompositionDiagnostics &diag)
        {
            SignalRun job(*this, bank, n);
            job.start(signal, diag);
            const int32_t n_threads = threads::team_size(config_.num_threads);
            reserve_workspaces(n_threads);
            while (job.next_stage())
            {
                // A block goes to whichever thread is free: its sum does
                // not depend on the thread that computes it
                threads::for_dynamic(n_threads, job.blocks(), [&](int32_t b, int slot)
                                     { job.block(b, workspace(slot, n)); });

                // Each sample sums its blocks in block order; a short
                // reduction is not worth a fork
                const int32_t reduce_threads =
                    (static_cast<int64_t>(n) * job.blocks() >= (int64_t(1) << 18)) ? n_threads : 1;
                threads::for_static(reduce_threads, n, [&](int32_t j_begin, int32_t j_end, int)
                                    { job.reduce(j_begin, j_end); });

                job.end_stage(workspace(0, n).lm_computer); // the stop test, on this thread
            }
            job.finish(imfs, residue);
            return true;
        }

        /**
         * decompose_batch() on team threads, the signals' trials interleaved.
         * A thread takes the next block of trials of the earliest signal that
         * has one; the thread that completes a stage's last block adds the
         * blocks up and sets up that signal's next stage; a thread that finds
         * no block starts the next signal. So no thread waits at the end of a
         * stage while another signal has trials left, and the last signals'
         * trials are spread over all the threads.
         */
        void run_interleaved(const double *signals, int32_t count, int32_t n, const BankUse &bank, int team,
                             std::vector<std::vector<std::vector<double>>> &imfs,
                             std::vector<std::vector<double>> &residues,
                             std::vector<DecompositionDiagnostics> &diags)
        {
            struct Flight // a signal being decomposed
            {
                std::unique_ptr<SignalRun> job; // made on first use, then reused
                int32_t signal = -1;            // -1: free
                bool open = false;              // the current stage's blocks are handed out
                int32_t claimed = 0;            // blocks of the stage handed out
                int32_t done = 0;               // ... and completed
            };
            std::vector<Flight> flights(static_cast<size_t>(std::min<int32_t>(count, team)));
            reserve_workspaces(team);

            std::mutex m; // guards the flights' signal, open, claimed, done; next_signal; finished; error
            std::condition_variable wake;
            int32_t next_signal = 0;
            int32_t finished = 0;
            std::exception_ptr error;

            // The flight's next stage (true), or its result handed in (false)
            const auto advance = [&](Flight &f) -> bool
            {
                if (f.job->next_stage())
                    return true;
                f.job->finish(imfs[f.signal], residues[f.signal]);
                return false;
            };

            threads::run(team, [&](int slot)
            {
                std::unique_lock<std::mutex> lock(m);
                while (!error && finished < count)
                {
                    Flight *f = nullptr; // the earliest signal with a block to hand out
                    for (Flight &c : flights)
                        if (c.open && c.claimed < c.job->blocks() && (!f || c.signal < f->signal))
                            f = &c;
                    Flight *fresh = nullptr; // else a free flight for the next signal
                    if (!f && next_signal < count)
                        for (Flight &c : flights)
                            if (c.signal < 0)
                            {
                                fresh = &c;
                                break;
                            }
                    if (!f && !fresh)
                    {
                        wake.wait(lock); // every block handed out: wait for a stage or a free flight
                        continue;
                    }
                    try
                    {
                        bool open;
                        if (f)
                        {
                            const int32_t b = f->claimed++;
                            lock.unlock();
                            f->job->block(b, workspace(slot, n));
                            lock.lock();
                            if (++f->done < f->job->blocks())
                                continue;
                            // the stage's last block: its sum and the signal's next step, here
                            f->open = false;
                            lock.unlock();
                            f->job->reduce(0, n);
                            f->job->end_stage(workspace(slot, n).lm_computer);
                            open = advance(*f);
                        }
                        else
                        {
                            f = fresh;
                            f->signal = next_signal++;
                            lock.unlock();
                            if (!f->job)
                                f->job.reset(new SignalRun(*this, bank, n));
                            f->job->start(signals + static_cast<size_t>(f->signal) * n, diags[f->signal]);
                            open = advance(*f);
                        }
                        lock.lock();
                        if (open)
                        {
                            f->claimed = 0;
                            f->done = 0;
                            f->open = true;
                        }
                        else
                        {
                            f->signal = -1;
                            ++finished;
                        }
                        wake.notify_all();
                    }
                    catch (...)
                    {
                        if (!lock.owns_lock())
                            lock.lock();
                        if (!error)
                            error = std::current_exception();
                        wake.notify_all();
                    }
                }
            });
            if (error)
                std::rethrow_exception(error);
        }

    public:
        /**
         * Convenience wrapper
         */
        bool decompose_with_residue(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs_and_residue)
        {
            std::vector<std::vector<double>> imfs;
            std::vector<double> residue;

            if (!decompose(signal, n, imfs, residue))
            {
                return false;
            }

            imfs_and_residue = std::move(imfs);
            imfs_and_residue.push_back(std::move(residue));

            return true;
        }

        /**
         * Decompose and return the run's diagnostics: per-IMF sifting
         * iterations and convergence, trials used, NaN count, seed, and the
         * orthogonality index of the IMFs.
         */
        bool decompose_with_diagnostics(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue,
            DecompositionDiagnostics &diag)
        {
            const bool ok = decompose(signal, n, imfs, residue);
            diag = last_diag_;
            return ok;
        }

        /** Diagnostics of the most recent decompose() or decompose_batch() (its last signal). */
        const DecompositionDiagnostics &last_diagnostics() const { return last_diag_; }

        ICEEMDANConfig &config() { return config_; }
        const ICEEMDANConfig &config() const { return config_; }

        /**
         * The memory the noise bank of signals of n samples takes with the
         * current settings (independent realizations: one of every trial,
         * about log2(n) modes each). decompose() fails when it is more than
         * config.noise_bank_max_bytes.
         */
        size_t noise_bank_bytes(int32_t n) const
        {
            return static_cast<size_t>(base_trials()) * static_cast<size_t>(bank_modes(n)) *
                   static_cast<size_t>(std::max(n, 0)) * sizeof(double);
        }

        /**
         * Frees the noise bank kept from the last decomposition
         * (config.keep_noise_bank), and the threads' working memory kept with
         * it. The next decomposition builds them again.
         */
        void release_noise_bank()
        {
            circular_bank_.release();
            standard_bank_.release();
            bank_valid_ = false;
            workspaces_.slots.clear();
        }

    private:
        /** Everything the noise bank depends on. */
        struct NoiseBankKey
        {
            int32_t n = 0;
            int32_t ensemble_size = 0;
            int32_t base_trials = 0;
            uint32_t seed = 0;
            bool circular = false;
            int32_t circular_multiplier = 0;
            size_t max_bytes = 0;
            EEMDConfig sifting; // the fields decompose() sets, compared below

            bool operator==(const NoiseBankKey &o) const
            {
                const EEMDConfig &a = sifting;
                const EEMDConfig &b = o.sifting;
                return n == o.n && ensemble_size == o.ensemble_size && base_trials == o.base_trials &&
                       seed == o.seed && circular == o.circular &&
                       circular_multiplier == o.circular_multiplier && max_bytes == o.max_bytes &&
                       a.max_imfs == b.max_imfs && a.max_sift_iters == b.max_sift_iters &&
                       a.sift_threshold == b.sift_threshold && a.boundary_extend == b.boundary_extend &&
                       a.s_number == b.s_number &&
                       a.stop_rule == b.stop_rule && a.rilling.theta1 == b.rilling.theta1 &&
                       a.rilling.theta2 == b.rilling.theta2 && a.rilling.alpha == b.rilling.alpha &&
                       a.symmetric_boundary == b.symmetric_boundary;
            }
        };

        ICEEMDANConfig config_;
        DecompositionDiagnostics last_diag_;

        // The noise bank of the last decomposition and what it was built for
        CircularNoiseBank circular_bank_;
        StandardNoiseBank standard_bank_;
        NoiseBankKey bank_key_;
        bool bank_valid_ = false;

        Workspaces workspaces_; // the threads' working memory, kept with the noise bank
    };

} // namespace eemd

#endif // ICEEMDAN_HPP
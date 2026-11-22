/**
 * The local mean M(x) of the ICEEMDAN stages: x minus its first EMD mode
 * sifted to convergence (Colominas et al. 2014), or one envelope mean (the
 * faster approximation).
 *
 * License: MIT
 */

#ifndef EEMD_LOCAL_MEAN_HPP
#define EEMD_LOCAL_MEAN_HPP

#include "eemd_sift.hpp"

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
    // Local Mean Computer
    // - Envelopes: the cubic spline through the extrema, continued past the
    //   ends by the symmetric extension of emd.m. Always the spline, as
    //   emd.m's interp1(..., 'spline'), however few the knots: the extension
    //   leaves at least 3, where the spline is the parabola through them
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
        int32_t max_iters = 2000; // as emd.m (MAXITERATIONS)
        double threshold = 0.05;  // SD rule: sum(m^2) / sum(h^2)
        int32_t s_number = 6;     // SD rule: stop after S iterations with a stable extrema count (0 = off)
        SiftStopRule rule = SiftStopRule::Rilling;
        RillingThresholds rilling;
    };

    class LocalMeanComputer
    {
    public:
        explicit LocalMeanComputer(int32_t max_len, int32_t boundary_extend,
                                   LocalMeanMethod method = LocalMeanMethod::Sifted,
                                   SiftParams sift = SiftParams())
            : max_len_(max_len),
              boundary_extend_(boundary_extend),
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

        /**
         * Both envelopes: the cubic spline through the upper knots and the
         * one through the lower knots (CubicSpline: not-a-knot ends; the
         * parabola at 3 knots, the straight line at 2), each over the samples
         * 0 .. n - 1. The two are built together (construct_pair). False when
         * a spline cannot be built: fewer than 2 knots, or knots that do not
         * increase.
         */
        bool compute_envelopes(const double *xu, const double *yu, int32_t ku,
                               const double *xl, const double *yl, int32_t kl, int32_t n)
        {
            return CubicSpline::construct_pair(spline_, xu, yu, ku, spline2_, xl, yl, kl) &&
                   spline_.evaluate_uniform(upper_env_.data, n) && spline2_.evaluate_uniform(lower_env_.data, n);
        }

        int32_t max_len_;
        int32_t boundary_extend_;
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

} // namespace eemd

#endif // EEMD_LOCAL_MEAN_HPP

/**
 * ICEEMDAN-MKL Optimized
 *
 * Algorithm: Colominas, Schlotthauer & Torres (2014). The defaults follow the
 * authors' reference code: local means from sifting to convergence with the
 * Rilling stop rule, independent noise realizations, beta_k as in the paper.
 *
 * Optional speed-ups (see ICEEMDANConfig):
 * 1. Circular Noise Bank - 1 long decomposition vs M short ones (off by default)
 * 2. Antithetic Variables - Process ±noise pairs for 2x variance reduction
 * 3. Sorted Spline Evaluation - Faster than uniform for dense extrema (~2.8x)
 * 4. Spline Fast Path - Linear interp for n_extrema < 6
 *
 * License: MIT
 */

#ifndef ICEEMDAN_MKL_OPT_HPP
#define ICEEMDAN_MKL_OPT_HPP

#include "eemd_mkl.hpp"
#include <memory>
#include <numeric>

namespace eemd
{

    // ============================================================================
    // Processing Mode Enum
    // ============================================================================

    /**
     * Processing modes with automatic defaults for different domains
     *
     * Standard:   Smooth boundaries (symmetric), global volatility, max quality
     *             Best for: Audio, seismic, general signal processing
     *
     * Finance:    Causal boundaries (AR), EMA volatility, input sanitization
     *             Best for: Trading, HFT, regime-switching data
     *
     * Scientific: Reproducible, no shortcuts, strict convergence tracking
     *             (symmetric boundaries, independent noise, no antithetic pairs)
     *             Best for: Research, publication, validation
     */
    enum class ProcessingMode : uint8_t
    {
        Standard = 0,  // Audio/Seismic: smooth, global vol
        Finance = 1,   // Trading: causal, adaptive vol
        Scientific = 2 // Research: reproducible, no shortcuts
    };

    /**
     * Volatility estimation method
     */
    enum class VolatilityMethod : uint8_t
    {
        Global = 0, // Single std-dev over entire signal
        SMA = 1,    // Simple Moving Average (rolling window)
        EMA = 2     // Exponential Moving Average (fast adaptation)
    };

    /**
     * Boundary handling method
     */
    enum class BoundaryMethod : uint8_t
    {
        Mirror = 0,   // Reflect extrema + linear end points (original scheme)
        AR = 1,       // AR(1) extrapolation (causal, no look-ahead)
        Linear = 2,   // Linear extrapolation (simplest)
        Symmetric = 3 // Symmetric extension of emd.m (Rilling et al. 2003), as in
                      // the authors' reference code; stable under long sifting
    };

    /**
     * Local-mean operator M(.) of the ICEEMDAN stages
     *
     * Sifted:         M(x) = x - E1(x), with E1 the first EMD mode sifted to
     *                 convergence (SD threshold, S-number, max_sift_iters).
     *                 This is the operator of Colominas et al. (2014) and of
     *                 their reference implementation. Default.
     * SingleEnvelope: M(x) = mean of the upper and lower envelopes of x, i.e.
     *                 one sifting iteration. Several times faster, but each
     *                 mode is under-sifted, so the IMFs differ from published
     *                 ICEEMDAN. For latency-bound uses that accept that.
     */
    enum class LocalMeanMethod : uint8_t
    {
        Sifted = 0,
        SingleEnvelope = 1
    };

    // ============================================================================
    // Cache-Line Padded Counter (Prevents False Sharing)
    // ============================================================================
    
    /**
     * Padded counter to prevent false sharing between threads.
     * Each counter occupies its own 64-byte cache line.
     */
    struct alignas(64) PaddedCounter
    {
        int32_t value;
        char padding[60];  // Pad to 64 bytes
        
        PaddedCounter() : value(0) {}
        void reset() { value = 0; }
    };

    // ============================================================================
    // ICEEMDAN Configuration (Mode-Based)
    // ============================================================================

    struct ICEEMDANConfig
    {
        // === Core EMD Parameters ===
        int32_t max_imfs = 10;
        int32_t max_sift_iters = 100;
        double sift_threshold = 0.05;
        int32_t ensemble_size = 100;
        double noise_std = 0.2;
        double noise_decay = 1.0;
        int32_t boundary_extend = 2;
        uint32_t rng_seed = 42;
        double monotonic_threshold = 1e-6;
        int32_t min_extrema = 3;
        
        // S-number stopping criterion for sifting: stop after S consecutive
        // iterations where extrema count is stable. Reduces iterations ~10-14%.
        // Benchmarks show S=6-8 gives best speedup with identical quality.
        // Set to 0 to disable (use only SD criterion).
        int32_t s_number = 6;

        // Local-mean operator (see LocalMeanMethod). Sifted = the published
        // algorithm; SingleEnvelope = the faster one-iteration approximation.
        LocalMeanMethod local_mean_method = LocalMeanMethod::Sifted;

        // When sifting stops, for the local means and for the noise modes
        // (see SiftStopRule). Rilling = the criterion of the authors'
        // reference code (default); SD = sift_threshold + s_number, cheaper.
        SiftStopRule sift_stop_rule = SiftStopRule::Rilling;
        RillingThresholds rilling;

        // === Optimization Flags ===
        bool use_antithetic = true;        // ±noise pairs (2x variance reduction)
        // Noise modes from windows of ONE long decomposed realization instead
        // of independent realizations. Off by default: the windows of a long
        // realization overlap and their low-frequency modes are not those of a
        // length-N realization, which visibly changes the low-frequency IMFs
        // (see README, "Accuracy"); with the Rilling stop rule it is also not
        // faster, since the long sift runs on one thread.
        bool use_circular_bank = false;
        int32_t circular_multiplier = 20;  // Long buffer = N × multiplier
        // Memory cap for the noise bank. The independent bank holds
        // base_trials x max_imfs x N doubles; above the cap the circular bank
        // is used instead (multiplier lowered, not below 4); if even that does
        // not fit, decompose() fails instead of exhausting memory.
        size_t noise_bank_max_bytes = size_t(2) << 30; // 2 GiB
        int32_t spline_fast_threshold = 6; // Linear interp if extrema < this
        
        // Spline method for envelope interpolation
        // Akima is recommended for finance (resistant to outliers/gaps)
        SplineMethod spline_method = SplineMethod::Cubic;

        // === Adaptive Ensemble Options ===
        // NOTE: Disabled by default. Benchmarks showed marginal benefit (~4-13%)
        // only with loose tolerance (0.07), and adds code complexity.
        // Keep code for experimentation with very large signals.
        bool adaptive_ensemble = false;      // Stop early when mean stabilizes
        int32_t adaptive_min_trials = 30;    // Minimum trials before checking
        int32_t adaptive_check_interval = 10; // Check every N trials
        double adaptive_rel_tol = 0.07;      // 7% relative change threshold
        int32_t adaptive_min_signal_len = 0; // Enable for all signal lengths

        // === Mode-Controlled Settings ===
        VolatilityMethod volatility_method = VolatilityMethod::Global;
        BoundaryMethod boundary_method = BoundaryMethod::Symmetric;
        bool sanitize_input = false;    // NaN/Inf protection
        // Kept for API compatibility: convergence statistics are now always
        // collected (the cost is a few integer adds per trial) and returned by
        // decompose_with_diagnostics() / last_diagnostics().
        bool track_convergence = false;

        // === Volatility Parameters ===
        int32_t vol_window = 50;   // SMA window size
        int32_t vol_ema_span = 20; // EMA span (α = 2/(span+1))
        double vol_floor = 1e-8;   // Floor for quiet markets

        // === AR Boundary Parameters ===
        int32_t ar_lookback = 5;       // Points for AR(1) estimation
        double ar_damping = 0.5;       // 0=mean revert, 1=full AR
        double ar_max_slope_atr = 2.0; // Clamp to ATR multiple

        /**
         * Apply mode-specific defaults
         * Call this after setting mode to override with domain presets
         */
        void apply_mode(ProcessingMode mode)
        {
            switch (mode)
            {
            case ProcessingMode::Standard:
                // Audio/Seismic: prioritize smoothness
                volatility_method = VolatilityMethod::Global;
                boundary_method = BoundaryMethod::Symmetric;
                sanitize_input = false;
                track_convergence = false;
                use_antithetic = true;
                use_circular_bank = false;
                adaptive_ensemble = false;  // Disabled - marginal benefit
                break;

            case ProcessingMode::Finance:
                // Trading: prioritize causality and robustness
                volatility_method = VolatilityMethod::EMA;
                boundary_method = BoundaryMethod::AR;
                sanitize_input = true;
                track_convergence = true; // For risk engine
                use_antithetic = true;
                use_circular_bank = false;
                adaptive_ensemble = false;  // Disabled - marginal benefit
                spline_method = SplineMethod::Akima;  // Outlier-resistant (falls back to Cubic if needed)
                vol_ema_span = 20;
                ar_damping = 0.5;
                ar_max_slope_atr = 2.0;
                break;

            case ProcessingMode::Scientific:
                // Research: prioritize reproducibility
                volatility_method = VolatilityMethod::Global;
                boundary_method = BoundaryMethod::Symmetric;
                sanitize_input = true;
                track_convergence = true;
                use_antithetic = false;    // No variance tricks
                use_circular_bank = false; // Independent noise
                adaptive_ensemble = false; // Fixed ensemble for reproducibility
                break;
            }
        }
    };

    /**
     * Diagnostic metadata for audit trails (e.g. model-risk review).
     * Filled by every decompose(); returned by decompose_with_diagnostics()
     * and last_diagnostics().
     */
    struct DecompositionDiagnostics
    {
        double orthogonality_index = 0.0;     // IO metric (lower = better)
        std::vector<int32_t> sift_iterations; // Per-IMF: most sifting iterations any trial needed
        std::vector<double> mean_sift_iterations; // Per-IMF: mean iterations per trial
        std::vector<int32_t> nonconverged_trials; // Per-IMF: trials that hit max_sift_iters
        std::vector<bool> convergence_flags;  // Per-IMF: true if no trial hit max_sift_iters
        std::vector<int32_t> trials_per_imf;  // Per-IMF: local means averaged (valid trials)
        int32_t nan_count = 0;                // Input NaN/Inf count (sanitized)
        uint32_t rng_seed_used = 0;           // For audit trail reproducibility
        ProcessingMode mode_used = ProcessingMode::Standard;
        LocalMeanMethod local_mean_method_used = LocalMeanMethod::Sifted;
        int32_t noise_bank_multiplier = 0;    // Circular bank length / N (0 = independent realizations)
        bool valid = true; // False if decomposition failed
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

    // ============================================================================
    // Unified Volatility Engine
    // Supports Global (scalar), SMA (rolling), and EMA (exponential) methods
    // ============================================================================

    /**
     * Volatility computation result
     * Global mode returns scalar (no array allocation needed)
     * SMA/EMA modes fill the provided array
     */
    struct VolatilityResult
    {
        double scalar = 1.0;   // Used when is_scalar=true
        bool is_scalar = true; // True for Global mode
    };

    /**
     * Compute volatility using specified method
     *
     * Global: Returns scalar in result.scalar (NO array write - saves O(N) bandwidth)
     * SMA:    Fills local_vol array with rolling std-dev
     * EMA:    Fills local_vol array with exponential std-dev
     *
     * @param method    Volatility estimation method
     * @param signal    Input signal
     * @param n         Signal length
     * @param window    SMA window size (ignored for Global/EMA)
     * @param ema_span  EMA span, α = 2/(span+1) (ignored for Global/SMA)
     * @param floor     Minimum volatility (prevents div-by-zero in quiet markets)
     * @param local_vol Output array (only written for SMA/EMA modes)
     * @return          VolatilityResult with scalar value and is_scalar flag
     */
    inline VolatilityResult compute_volatility(
        VolatilityMethod method,
        const double *signal,
        int32_t n,
        int32_t window,
        int32_t ema_span,
        double floor,
        double *local_vol) // Only written if SMA/EMA
    {
        VolatilityResult result;

        if (n <= 0)
        {
            result.scalar = floor;
            result.is_scalar = true;
            return result;
        }

        switch (method)
        {
        case VolatilityMethod::Global:
        {
            // Return scalar - NO array write (saves 80MB for N=10^7)
            const double global_std = compute_std(signal, n);
            result.scalar = std::max(floor, global_std);
            result.is_scalar = true;
            break;
        }

        case VolatilityMethod::SMA:
        {
            // Simple Moving Average: O(N) via running sums
            double sum_x = 0.0;
            double sum_x2 = 0.0;

            for (int32_t i = 0; i < n; ++i)
            {
                const double x = signal[i];
                sum_x += x;
                sum_x2 += x * x;

                if (i >= window)
                {
                    const double old_x = signal[i - window];
                    sum_x -= old_x;
                    sum_x2 -= old_x * old_x;
                }

                const int32_t w = std::min(i + 1, window);
                const double inv_w = 1.0 / w;
                const double mean = sum_x * inv_w;
                const double var = sum_x2 * inv_w - mean * mean;
                local_vol[i] = std::max(floor, std::sqrt(std::max(0.0, var)));
            }
            result.is_scalar = false;
            break;
        }

        case VolatilityMethod::EMA:
        {
            // Exponential Moving Average: fast adaptation to regime changes
            const double alpha = 2.0 / (ema_span + 1);
            const double one_minus_alpha = 1.0 - alpha;

            double ema_mean = signal[0];
            double ema_var = 0.0;
            local_vol[0] = floor;

            for (int32_t i = 1; i < n; ++i)
            {
                const double x = signal[i];
                ema_mean = alpha * x + one_minus_alpha * ema_mean;
                const double dev = x - ema_mean;
                ema_var = alpha * (dev * dev) + one_minus_alpha * ema_var;
                local_vol[i] = std::max(floor, std::sqrt(ema_var));
            }
            result.is_scalar = false;
            break;
        }
        }

        return result;
    }

    // Legacy wrapper for backward compatibility
    inline void compute_local_volatility(
        const double *signal,
        int32_t n,
        int32_t lookback,
        double floor,
        double *local_vol)
    {
        compute_volatility(VolatilityMethod::SMA, signal, n, lookback, 0, floor, local_vol);
    }

    /**
     * Sanitize input signal: replace NaN/Inf with interpolated values
     * Returns count of sanitized points (for diagnostics)
     *
     * CRITICAL: MKL spline functions crash on NaN/Inf input
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

    /**
     * Compute Average True Range (ATR) from recent extrema
     * Used to constrain AR extrapolation slope
     */
    inline double compute_local_atr(
        const int32_t *indices,
        int32_t n_indices,
        const double *signal,
        int32_t lookback)
    {
        if (n_indices < 2)
            return 1.0;

        const int32_t start = std::max(0, n_indices - lookback);
        double sum_range = 0.0;
        int32_t count = 0;

        for (int32_t i = start + 1; i < n_indices; ++i)
        {
            double range = std::abs(signal[indices[i]] - signal[indices[i - 1]]);
            sum_range += range;
            ++count;
        }

        return (count > 0) ? sum_range / count : 1.0;
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

    /**
     * Causal boundary extension for finance applications
     *
     * Left edge: Mirror (past data is known)
     * Right edge: AR(1) extrapolation (future is unknown)
     *
     * This prevents false reversal signals at the trading edge caused by
     * mirroring a downtrend into a V-shape.
     *
     * Damping prevents overshoot on fat-tailed moves (flash crashes)
     * ATR constraint limits slope to reasonable range
     */
    inline void extend_extrema_causal(
        const int32_t *__restrict indices,
        int32_t n_indices,
        const double *__restrict signal,
        int32_t signal_len,
        int32_t left_extend,
        int32_t ar_lookback,
        double ar_damping,       // 0=mean revert, 1=full AR
        double ar_max_slope_atr, // Max slope as multiple of ATR
        double *__restrict out_x,
        double *__restrict out_y,
        int32_t &out_count,
        int32_t &original_start)
    {
        if (n_indices < 2)
        {
            out_count = n_indices;
            original_start = 0;
            for (int32_t i = 0; i < n_indices; ++i)
            {
                out_x[i] = static_cast<double>(indices[i]);
                out_y[i] = signal[indices[i]];
            }
            return;
        }

        int32_t pos = 0;

        // === LEFT EDGE: Mirror (past is known) ===
        const int32_t left_ext = std::min(left_extend, n_indices - 1);

        double leftmost_x = static_cast<double>(indices[0]);
        if (left_ext > 0)
        {
            leftmost_x = 2.0 * indices[0] - indices[left_ext];
        }

        if (leftmost_x > 0.0)
        {
            const double x0 = static_cast<double>(indices[0]);
            const double x1 = static_cast<double>(indices[1]);
            const double y0 = signal[indices[0]];
            const double y1 = signal[indices[1]];
            double dx = x1 - x0;
            if (std::abs(dx) < 1e-10)
                dx = 1.0;
            const double slope = (y1 - y0) / dx;
            out_x[pos] = -1.0;
            out_y[pos] = y0 + slope * (-1.0 - x0);
            ++pos;
        }

        for (int32_t i = 0; i < left_ext; ++i)
        {
            const int32_t src = left_ext - i;
            out_x[pos] = 2.0 * indices[0] - indices[src];
            out_y[pos] = signal[indices[src]];
            ++pos;
        }

        original_start = pos;

        // === ORIGINAL EXTREMA ===
        for (int32_t i = 0; i < n_indices; ++i)
        {
            out_x[pos] = static_cast<double>(indices[i]);
            out_y[pos] = signal[indices[i]];
            ++pos;
        }

        // === RIGHT EDGE: AR(1) Extrapolation with Dampening ===
        const int32_t ar_n = std::min(ar_lookback, n_indices - 1);

        // Compute ATR for slope constraint
        const double atr = compute_local_atr(indices, n_indices, signal, ar_lookback);

        if (ar_n >= 2)
        {
            double mean_spacing = 0.0;
            double mean_value = 0.0;

            for (int32_t i = n_indices - ar_n; i < n_indices; ++i)
            {
                mean_value += signal[indices[i]];
                if (i > n_indices - ar_n)
                {
                    mean_spacing += indices[i] - indices[i - 1];
                }
            }
            mean_spacing /= (ar_n - 1);
            mean_value /= ar_n;

            // AR(1) coefficient
            double num = 0.0, den = 0.0;
            for (int32_t i = n_indices - ar_n + 1; i < n_indices; ++i)
            {
                double y_prev = signal[indices[i - 1]] - mean_value;
                double y_curr = signal[indices[i]] - mean_value;
                num += y_prev * y_curr;
                den += y_prev * y_prev;
            }

            double phi = (std::abs(den) > 1e-10) ? num / den : 0.0;
            phi = std::max(-0.99, std::min(0.99, phi));

            // Apply damping: phi_damped = damping * phi
            // damping=0 → mean revert, damping=1 → full AR
            phi *= ar_damping;

            double last_x = static_cast<double>(indices[n_indices - 1]);
            double last_y = signal[indices[n_indices - 1]];

            double next_x = last_x + mean_spacing;
            double next_y = mean_value + phi * (last_y - mean_value);

            // Constrain slope to ATR multiple
            double raw_slope = (next_y - last_y) / std::max(1.0, mean_spacing);
            double max_slope = ar_max_slope_atr * atr / mean_spacing;
            if (std::abs(raw_slope) > max_slope)
            {
                raw_slope = std::copysign(max_slope, raw_slope);
                next_y = last_y + raw_slope * mean_spacing;
            }

            if (next_x < signal_len + mean_spacing)
            {
                out_x[pos] = next_x;
                out_y[pos] = next_y;
                ++pos;
            }

            if (next_x < signal_len - 1)
            {
                double dx = next_x - last_x;
                if (std::abs(dx) < 1e-10)
                    dx = 1.0;
                double slope = (next_y - last_y) / dx;
                // Also constrain boundary slope
                double max_boundary_slope = ar_max_slope_atr * atr / dx;
                if (std::abs(slope) > max_boundary_slope)
                {
                    slope = std::copysign(max_boundary_slope, slope);
                }
                out_x[pos] = static_cast<double>(signal_len);
                out_y[pos] = last_y + slope * (signal_len - last_x);
                ++pos;
            }
        }
        else
        {
            // Fallback: linear extrapolation
            const double x0 = static_cast<double>(indices[n_indices - 2]);
            const double x1 = static_cast<double>(indices[n_indices - 1]);
            const double y0 = signal[indices[n_indices - 2]];
            const double y1 = signal[indices[n_indices - 1]];
            double dx = x1 - x0;
            if (std::abs(dx) < 1e-10)
                dx = 1.0;
            const double slope = (y1 - y0) / dx;
            out_x[pos] = static_cast<double>(signal_len);
            out_y[pos] = y1 + slope * (signal_len - x1);
            ++pos;
        }

        out_count = pos;
    }

    // ============================================================================
    // Optimized Local Mean Computer
    // - Uses sorted evaluation (faster for dense extrema)
    // - Falls back to linear interp for sparse extrema
    // - Optional causal right-edge for finance (AR extrapolation)
    // - Sifted mode: M(x) = x - E1(x), E1 sifted to convergence (the operator
    //   of Colominas et al. 2014); SingleEnvelope mode: one envelope mean
    // ============================================================================

    /** What one local-mean computation cost. */
    struct SiftStats
    {
        int32_t iterations = 0; // envelope means subtracted
        bool converged = true;  // false: stopped by max_sift_iters
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
                                   BoundaryMethod boundary_method = BoundaryMethod::Mirror,
                                   int32_t ar_lookback = 5,
                                   double ar_damping = 0.5, double ar_max_slope_atr = 2.0,
                                   SplineMethod spline_method = SplineMethod::Cubic,
                                   LocalMeanMethod method = LocalMeanMethod::Sifted,
                                   SiftParams sift = SiftParams())
            : max_len_(max_len),
              boundary_extend_(boundary_extend),
              fast_threshold_(fast_threshold),
              boundary_method_(boundary_method),
              ar_lookback_(ar_lookback),
              ar_damping_(ar_damping),
              ar_max_slope_atr_(ar_max_slope_atr),
              spline_method_(spline_method),
              method_(method),
              sift_(sift),
              max_idx_(max_len / 2 + 2),
              min_idx_(max_len / 2 + 2),
              ext_x_(max_len + 128),
              ext_y_(max_len + 128),
              ext_x2_(max_len + 128),
              ext_y2_(max_len + 128),
              eval_sites_(max_len),
              upper_env_(max_len),
              lower_env_(max_len),
              h_(method == LocalMeanMethod::Sifted ? max_len : 0)
        {
            // Pre-compute evaluation sites [0, 1, 2, ..., max_len-1]
            for (int32_t i = 0; i < max_len; ++i)
            {
                eval_sites_[i] = static_cast<double>(i);
            }
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

    private:
        /**
         * Mean of the upper and lower envelopes of x (one sifting step).
         * Leaves the extrema counts of x in n_max_ / n_min_.
         */
        bool envelope_mean(const double *x, int32_t n, double *out_mean)
        {
            // Find extrema
            find_maxima_raw(x, n, max_idx_.data(), n_max_);
            find_minima_raw(x, n, min_idx_.data(), n_min_);

            if (n_max_ < 2 || n_min_ < 2)
            {
                std::memcpy(out_mean, x, n * sizeof(double));
                return false;
            }
            return envelope_mean_from_extrema(x, n, out_mean);
        }

        bool envelope_mean_from_extrema(const double *x, int32_t n, double *out_mean)
        {
            if (boundary_method_ == BoundaryMethod::Symmetric)
            {
                int32_t n_up = 0, n_lo = 0;
                const bool ok =
                    extend_extrema_symmetric(max_idx_.data(), n_max_, min_idx_.data(), n_min_, x, n,
                                             boundary_extend_, ext_x_.data(), ext_y_.data(), n_up,
                                             ext_x2_.data(), ext_y2_.data(), n_lo) &&
                    compute_envelope(ext_x_.data(), ext_y_.data(), n_up, upper_env_.data, n) &&
                    compute_envelope(ext_x2_.data(), ext_y2_.data(), n_lo, lower_env_.data, n);
                if (!ok)
                {
                    std::memcpy(out_mean, x, n * sizeof(double));
                    return false;
                }
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

            // Upper envelope - dispatch based on boundary method
            int32_t n_ext, ext_start;
            extend_extrema(max_idx_.data(), n_max_, x, n, ext_x_.data(), ext_y_.data(), n_ext, ext_start);

            if (!compute_envelope(ext_x_.data(), ext_y_.data(), n_ext,
                                  upper_env_.data, n))
            {
                std::memcpy(out_mean, x, n * sizeof(double));
                return false;
            }

            // Lower envelope
            extend_extrema(min_idx_.data(), n_min_, x, n, ext_x_.data(), ext_y_.data(), n_ext, ext_start);

            if (!compute_envelope(ext_x_.data(), ext_y_.data(), n_ext,
                                  lower_env_.data, n))
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
         * M(x) = x - E1(x): sift the first mode to convergence and return what
         * is left. Same stopping rules as Sifter::sift_imf (SiftStopRule: SD
         * with S-number and extrema ~ zero crossings, or Rilling; max_iters).
         * A step that cannot build envelopes after the first iteration ends
         * the sifting and keeps the mode as it stands (as Rilling & Flandrin's
         * emd.m does); only a first step that cannot is a failure.
         */
        bool compute_sifted(const double *signal, int32_t n, double *local_mean, SiftStats *stats)
        {
            double *__restrict h = h_.data;
            std::memcpy(h, signal, n * sizeof(double));

            int32_t prev_n_extrema = -1;
            int32_t s_count = 0;
            int32_t iterations = 0;
            bool converged = false;

            for (int32_t iter = 0; iter < sift_.max_iters; ++iter)
            {
                find_maxima_raw(h, n, max_idx_.data(), n_max_);
                find_minima_raw(h, n, min_idx_.data(), n_min_);

                if (n_max_ < 2 || n_min_ < 2)
                {
                    if (iter == 0)
                    {
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

                // local_mean doubles as the envelope-mean scratch here
                if (!envelope_mean_from_extrema(h, n, local_mean))
                {
                    if (iter == 0)
                    {
                        std::memcpy(local_mean, signal, n * sizeof(double));
                        return false;
                    }
                    break; // keep the mode as it stands
                }

                if (rilling)
                {
                    if (rilling_should_stop(h, upper_env_.data, lower_env_.data, n, n_extrema, sift_.rilling))
                    {
                        converged = true;
                        break;
                    }
                    const double *__restrict m = local_mean;
                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        h[i] -= m[i];
                    }
                    ++iterations;
                    continue;
                }

                // Fused: SD criterion and h -= m in one pass
                double sd_num = 0.0;
                double sd_den = 0.0;
                const double *__restrict m = local_mean;

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    const double val = h[i];
                    const double mean = m[i];
                    sd_num += mean * mean;
                    sd_den += val * val;
                    h[i] = val - mean;
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

            // M(x) = x - E1(x)
            const double *__restrict x = signal;
            double *__restrict out = local_mean;
            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                out[i] = x[i] - h[i];
            }

            if (stats)
            {
                stats->iterations = iterations;
                stats->converged = converged;
            }
            return true;
        }

        /**
         * Dispatch extrema extension based on boundary method
         */
        void extend_extrema(const int32_t *indices, int32_t n_indices,
                            const double *signal, int32_t n,
                            double *out_x, double *out_y,
                            int32_t &out_count, int32_t &original_start)
        {
            switch (boundary_method_)
            {
            case BoundaryMethod::AR:
                extend_extrema_causal(indices, n_indices, signal, n,
                                      boundary_extend_, ar_lookback_,
                                      ar_damping_, ar_max_slope_atr_,
                                      out_x, out_y, out_count, original_start);
                break;

            case BoundaryMethod::Linear:
                extend_extrema_linear(indices, n_indices, signal, n,
                                      out_x, out_y, out_count, original_start);
                break;

            case BoundaryMethod::Mirror:
            default:
                extend_extrema_raw(indices, n_indices, signal, n,
                                   boundary_extend_, out_x, out_y,
                                   out_count, original_start);
                break;
            }
        }

        /**
         * Simple linear extrapolation at both edges
         */
        void extend_extrema_linear(const int32_t *indices, int32_t n_indices,
                                   const double *signal, int32_t n,
                                   double *out_x, double *out_y,
                                   int32_t &out_count, int32_t &original_start)
        {
            if (n_indices < 2)
            {
                out_count = n_indices;
                original_start = 0;
                for (int32_t i = 0; i < n_indices; ++i)
                {
                    out_x[i] = static_cast<double>(indices[i]);
                    out_y[i] = signal[indices[i]];
                }
                return;
            }

            int32_t pos = 0;

            // Left edge: linear extrapolation
            {
                double x0 = static_cast<double>(indices[0]);
                double x1 = static_cast<double>(indices[1]);
                double y0 = signal[indices[0]];
                double y1 = signal[indices[1]];
                double dx = x1 - x0;
                if (std::abs(dx) < 1e-10)
                    dx = 1.0;
                double slope = (y1 - y0) / dx;
                out_x[pos] = -1.0;
                out_y[pos] = y0 + slope * (-1.0 - x0);
                ++pos;
            }

            original_start = pos;

            // Original extrema
            for (int32_t i = 0; i < n_indices; ++i)
            {
                out_x[pos] = static_cast<double>(indices[i]);
                out_y[pos] = signal[indices[i]];
                ++pos;
            }

            // Right edge: linear extrapolation
            {
                double x0 = static_cast<double>(indices[n_indices - 2]);
                double x1 = static_cast<double>(indices[n_indices - 1]);
                double y0 = signal[indices[n_indices - 2]];
                double y1 = signal[indices[n_indices - 1]];
                double dx = x1 - x0;
                if (std::abs(dx) < 1e-10)
                    dx = 1.0;
                double slope = (y1 - y0) / dx;
                out_x[pos] = static_cast<double>(n);
                out_y[pos] = y1 + slope * (n - x1);
                ++pos;
            }

            out_count = pos;
        }

        /**
         * Robust envelope construction: Tries requested method, falls back to Cubic
         */
        bool compute_envelope(const double *x_knots, const double *y_knots,
                              int32_t n_knots, double *envelope, int32_t n)
        {
            // Fast path: linear interpolation for sparse extrema
            if (n_knots < fast_threshold_)
            {
                fast_linear_interp(x_knots, y_knots, n_knots, envelope, n);
                return true;
            }

            // Try primary method (e.g., Akima)
            if (spline_.construct(x_knots, y_knots, n_knots, spline_method_))
            {
                if (spline_.evaluate(eval_sites_.data(), envelope, n))
                    return true;
            }

            // Fallback: If primary method failed (e.g., Akima geometric constraints),
            // try Cubic as a robust fallback
            if (spline_method_ != SplineMethod::Cubic)
            {
                if (spline_.construct(x_knots, y_knots, n_knots, SplineMethod::Cubic))
                {
                    return spline_.evaluate(eval_sites_.data(), envelope, n);
                }
            }

            return false;
        }

        int32_t max_len_;
        int32_t boundary_extend_;
        int32_t fast_threshold_;
        BoundaryMethod boundary_method_;
        int32_t ar_lookback_;
        double ar_damping_;
        double ar_max_slope_atr_;
        SplineMethod spline_method_;
        LocalMeanMethod method_;
        SiftParams sift_;

        std::vector<int32_t> max_idx_;
        std::vector<int32_t> min_idx_;
        std::vector<double> ext_x_;
        std::vector<double> ext_y_;
        std::vector<double> ext_x2_; // second knot set (Symmetric builds both at once)
        std::vector<double> ext_y2_;
        std::vector<double> eval_sites_;

        int32_t n_max_ = 0;
        int32_t n_min_ = 0;

        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;
        AlignedBuffer<double> h_; // mode being sifted (Sifted mode only)

        MKLSpline spline_;
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

            VSLStreamStatePtr stream = nullptr;
            vslNewStream(&stream, VSL_BRNG_MT19937, base_seed);
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream,
                          static_cast<MKL_INT>(long_n_), long_noise.data, 0.0, 1.0);
            vslDeleteStream(&stream);

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

        void initialize(int32_t n, int32_t ensemble_size, int32_t max_imfs,
                        const EEMDConfig &emd_config, uint32_t base_seed)
        {
            n_ = n;
            ensemble_size_ = ensemble_size;
            max_imfs_ = max_imfs;

            noise_imfs_.resize(ensemble_size);
            imf_counts_.resize(ensemble_size);

            for (int32_t i = 0; i < ensemble_size; ++i)
            {
                noise_imfs_[i].resize(max_imfs);
                for (int32_t k = 0; k < max_imfs; ++k)
                {
                    noise_imfs_[i][k].resize(n);
                }
            }

            threads::for_static(threads::team_size(), ensemble_size,
                                [&](int32_t i_begin, int32_t i_end, int)
            {
                if (i_begin == i_end)
                    return;

                AlignedBuffer<double> noise(n);
                AlignedBuffer<double> work(n);

                Sifter sifter(n, emd_config);

                for (int32_t i = i_begin; i < i_end; ++i)
                {
                    // Realization i depends on (seed, i) only: the same at any thread count
                    if (!trial_gaussian(base_seed, static_cast<uint32_t>(i), n, noise.data))
                    {
                        noise.zero();
                    }

                    std::memcpy(work.data, noise.data, n * sizeof(double));

                    int32_t imf_count = 0;
                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        if (!sifter.sift_imf(work.data, noise_imfs_[i][k].data(), n))
                        {
                            break;
                        }
                        ++imf_count;
                    }
                    imf_counts_[i] = imf_count;
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
        /**
         * Construct with explicit configuration
         */
        explicit ICEEMDAN(const ICEEMDANConfig &config = ICEEMDANConfig())
            : config_(config), mode_(ProcessingMode::Standard)
        {
        }

        /**
         * Construct with processing mode (applies mode defaults automatically)
         */
        explicit ICEEMDAN(ProcessingMode mode)
            : mode_(mode)
        {
            config_.apply_mode(mode);
        }

        /**
         * Decompose signal into IMFs using ICEEMDAN (Colominas et al. 2014)
         *
         *   r_0 = x
         *   r_k = < M( r_{k-1} + beta_{k-1} E_k(w^(i)) ) >_i     (average over trials)
         *   IMF_k = r_{k-1} - r_k
         *   beta_0 = eps * std(x) / std(E_1(w^(i))),  beta_k = eps * std(r_k)  (k >= 1)
         *
         * with eps = noise_std (times noise_decay^k), E_k the k-th EMD mode of
         * white noise and M the local-mean operator (local_mean_method).
         *
         * Optimizations:
         * - Circular noise bank (optional, default off; automatic fallback
         *   when the independent bank exceeds noise_bank_max_bytes)
         * - Antithetic variables (optional, default on)
         * - Sorted spline evaluation
         * - Linear interp fast path for sparse extrema
         *
         * Mode-specific behavior:
         * - Standard: Global volatility, mirror boundaries
         * - Finance:  EMA volatility, AR boundaries, input sanitization
         * - Scientific: Global volatility, mirror boundaries, no shortcuts
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
            last_diag_.mode_used = mode_;
            last_diag_.local_mean_method_used = config_.local_mean_method;
            last_diag_.valid = false;

            if (n < 4 || config_.ensemble_size < 1 || config_.max_imfs < 1)
                return false;

            // Input sanitization (NaN/Inf protection for MKL)
            AlignedBuffer<double> signal_clean(n);
            std::memcpy(signal_clean.data, signal, n * sizeof(double));

            int32_t nan_count = 0;
            if (config_.sanitize_input)
            {
                nan_count = sanitize_signal(signal_clean.data, n);
            }
            last_diag_.nan_count = nan_count;

            const double *clean_signal = signal_clean.data;

            // Compute volatility using unified engine
            // Global mode returns scalar (no array allocation needed)
            AlignedBuffer<double> local_vol_array;
            if (config_.volatility_method != VolatilityMethod::Global)
            {
                local_vol_array.resize(n);
            }

            VolatilityResult vol_result = compute_volatility(
                config_.volatility_method, clean_signal, n,
                config_.vol_window, config_.vol_ema_span,
                config_.vol_floor,
                local_vol_array.data); // Only written if SMA/EMA

            const double signal_std = std::max(config_.vol_floor, compute_std(clean_signal, n));

            // Build EMD config
            EEMDConfig emd_config;
            emd_config.max_imfs = config_.max_imfs;
            emd_config.max_sift_iters = config_.max_sift_iters;
            emd_config.sift_threshold = config_.sift_threshold;
            emd_config.boundary_extend = config_.boundary_extend;
            emd_config.s_number = config_.s_number;
            emd_config.spline_method = config_.spline_method;
            emd_config.stop_rule = config_.sift_stop_rule;
            emd_config.rilling = config_.rilling;
            // Noise modes are synthetic (causality does not apply to them):
            // use the stable symmetric ends whenever sifting runs long
            emd_config.symmetric_boundary = (config_.sift_stop_rule == SiftStopRule::Rilling) ||
                                            (config_.boundary_method == BoundaryMethod::Symmetric);

            // Effective ensemble size (halved if using antithetic)
            // Round UP to avoid silently losing trials with odd ensemble sizes
            const int32_t base_trials = config_.use_antithetic ? (config_.ensemble_size + 1) / 2 : config_.ensemble_size;

            // Initialize noise bank (independent realizations or circular)
            // Decision made ONCE here, not in hot loop. One realization per
            // base trial (antithetic pairs share theirs).
            CircularNoiseBank circular_bank;
            StandardNoiseBank standard_bank;
            const size_t independent_bytes = static_cast<size_t>(base_trials) *
                                             static_cast<size_t>(config_.max_imfs) *
                                             static_cast<size_t>(n) * sizeof(double);
            const bool use_circular = config_.use_circular_bank ||
                                      independent_bytes > config_.noise_bank_max_bytes;

            if (use_circular)
            {
                if (!circular_bank.initialize(n, config_.ensemble_size, config_.max_imfs,
                                              emd_config, config_.rng_seed,
                                              config_.circular_multiplier,
                                              config_.noise_bank_max_bytes))
                {
                    return false; // noise bank would not fit in noise_bank_max_bytes
                }
                last_diag_.noise_bank_multiplier = circular_bank.multiplier();
            }
            else
            {
                standard_bank.initialize(n, base_trials, config_.max_imfs,
                                         emd_config, config_.rng_seed);
            }

            // Capture noise bank accessor as lambda (hoisted out of hot loop)
            auto get_noise = [&](int32_t trial, int32_t imf) -> const double *
            {
                return use_circular ? circular_bank.get_noise_slice(trial, imf) : standard_bank.get_noise_imf(trial, imf);
            };

            // Prepare output
            imfs.clear();
            imfs.reserve(config_.max_imfs);

            AlignedBuffer<double> r_current(n);
            std::memcpy(r_current.data, clean_signal, n * sizeof(double));

            AlignedBuffer<double> mean_accumulator(n);

            std::vector<AlignedBuffer<double>> imf_storage(config_.max_imfs);
            for (auto &buf : imf_storage)
            {
                buf.resize(n);
            }

            int32_t actual_imf_count = 0;
            bool stop_decomposition = false;

            const bool use_scalar_vol = vol_result.is_scalar;

            // Noise decay multiplier (decreases each IMF)
            double decay_factor = 1.0;

            // Stage amplitude of the noise mode (see the formula above)
            double stage_scale = 0.0;

            // Adaptive ensemble tracking
            AlignedBuffer<double> prev_mean_estimate(n);

            // Heuristic: disable adaptive for short signals (overhead > benefit)
            const bool effective_adaptive = config_.adaptive_ensemble &&
                                            (n >= config_.adaptive_min_signal_len);

            // =================================================================
            // Deterministic reduction layout: trials are summed in blocks of
            // block_trials (in trial order), blocks are reduced in block order.
            // Nothing here depends on the number of threads.
            // =================================================================
            const int32_t batch_max = effective_adaptive
                                          ? std::max(1, std::min(config_.adaptive_check_interval, base_trials))
                                          : base_trials;
            const int32_t block_trials = std::max(4, (batch_max + 63) / 64); // at most 64 blocks
            const int32_t n_blocks_max = (batch_max + block_trials - 1) / block_trials;

            std::vector<AlignedBuffer<double>> block_accs(n_blocks_max);
            for (auto &acc : block_accs)
            {
                acc.resize(n);
            }
            struct BlockStats
            {
                int32_t valid = 0;
                int32_t attempted = 0;
                int64_t iterations = 0;
                int32_t max_iterations = 0;
                int32_t nonconverged = 0;
            };
            std::vector<BlockStats> block_stats(n_blocks_max);

            // Batch processing state (shared across threads)
            int32_t batch_start = 0;
            int32_t batch_end = 0;
            int32_t n_blocks_batch = 0;
            bool ensemble_converged = false;
            BlockStats stage_total;

            const SiftParams sift_params{config_.max_sift_iters, config_.sift_threshold, config_.s_number,
                                         config_.sift_stop_rule, config_.rilling};

            {
                // Per-thread resources with mode-specific boundary handling.
                // Each thread makes its own the first time it takes a block
                // and keeps them for every stage of this decomposition.
                struct ThreadState
                {
                    LocalMeanComputer lm_computer;
                    AlignedBuffer<double> perturbed;
                    AlignedBuffer<double> local_mean;

                    ThreadState(int32_t len, const ICEEMDANConfig &c, const SiftParams &sp)
                        : lm_computer(len, c.boundary_extend,
                                      c.spline_fast_threshold,
                                      c.boundary_method,
                                      c.ar_lookback,
                                      c.ar_damping,
                                      c.ar_max_slope_atr,
                                      c.spline_method,
                                      c.local_mean_method,
                                      sp),
                          perturbed(len),
                          local_mean(len)
                    {
                    }
                };
                const int32_t n_threads = threads::team_size();
                std::vector<std::unique_ptr<ThreadState>> thread_states(n_threads);

                // =============================================================
                // IMF EXTRACTION LOOP (on the calling thread; the trials of a
                // stage and their reduction are forked to the pool)
                // =============================================================
                for (int32_t k = 0; k < config_.max_imfs; ++k)
                {
                    if (stop_decomposition)
                        break;

                    // Reset for new IMF
                    batch_start = 0;
                    batch_end = 0;
                    ensemble_converged = false;
                    stage_total = BlockStats();
                    mean_accumulator.zero();
                    prev_mean_estimate.zero();

                    // beta_k = eps * std(r_k) for k >= 1; for k = 0 the
                    // per-trial factor std(x) / std(E_1(w)) is applied below
                    stage_scale = (k == 0)
                                      ? decay_factor * config_.noise_std * signal_std
                                      : decay_factor * config_.noise_std * compute_std(r_current.data, n);

                    // =============================================================
                    // BATCH PROCESSING LOOP (one batch unless adaptive)
                    // =============================================================
                    while (true)
                    {
                        batch_start = batch_end;
                        batch_end = effective_adaptive
                                        ? std::min(batch_start + config_.adaptive_check_interval, base_trials)
                                        : base_trials;
                        n_blocks_batch = (batch_end - batch_start + block_trials - 1) / block_trials;

                        if (batch_start >= base_trials)
                            break;

                        // A block goes to whichever thread is free: its sum does
                        // not depend on the thread that computes it
                        threads::for_dynamic(n_threads, n_blocks_batch, [&](int32_t b, int slot)
                        {
                            if (!thread_states[slot])
                                thread_states[slot].reset(new ThreadState(n, config_, sift_params));
                            LocalMeanComputer &lm_computer = thread_states[slot]->lm_computer;
                            AlignedBuffer<double> &tl_perturbed = thread_states[slot]->perturbed;
                            AlignedBuffer<double> &tl_local_mean = thread_states[slot]->local_mean;

                            double *__restrict acc = block_accs[b].data;
                            std::memset(acc, 0, n * sizeof(double));
                            BlockStats st;

                            const int32_t t_begin = batch_start + b * block_trials;
                            const int32_t t_end = std::min(batch_end, t_begin + block_trials);

                            for (int32_t i = t_begin; i < t_end; ++i)
                            {
                                const double *noise_imf = get_noise(i, k);

                                // Amplitude of this trial's noise mode
                                double amp = stage_scale;
                                if (noise_imf && k == 0)
                                {
                                    // beta_0 = eps * std(x) / std(E_1(w^(i)))
                                    const double sw = compute_std(noise_imf, n);
                                    amp = (sw > 1e-300) ? stage_scale / sw : 0.0;
                                }
                                // Local-volatility modes keep their time profile,
                                // scaled to the stage level: lv[j] / std(x) * amp
                                const double lv_scale = use_scalar_vol ? 0.0 : amp / signal_std;

                                const int32_t n_branches = (config_.use_antithetic && noise_imf) ? 2 : 1;
                                for (int32_t branch = 0; branch < n_branches; ++branch)
                                {
                                    const double sign = (branch == 0) ? 1.0 : -1.0;
                                    const double *__restrict r = r_current.data;
                                    double *__restrict p = tl_perturbed.data;

                                    if (!noise_imf)
                                    {
                                        std::memcpy(p, r, n * sizeof(double));
                                    }
                                    else if (use_scalar_vol)
                                    {
                                        const double a = sign * amp;
                                        const double *__restrict nz = noise_imf;
                                        EEMD_OMP_SIMD
                                        for (int32_t j = 0; j < n; ++j)
                                        {
                                            p[j] = r[j] + a * nz[j];
                                        }
                                    }
                                    else
                                    {
                                        const double a = sign * lv_scale;
                                        const double *__restrict nz = noise_imf;
                                        const double *__restrict lv = local_vol_array.data;
                                        EEMD_OMP_SIMD
                                        for (int32_t j = 0; j < n; ++j)
                                        {
                                            p[j] = r[j] + a * lv[j] * nz[j];
                                        }
                                    }

                                    SiftStats ss;
                                    ++st.attempted;
                                    if (lm_computer.compute(tl_perturbed.data, n, tl_local_mean.data, &ss))
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
                            }
                            block_stats[b] = st;
                        });

                        // =========================================================
                        // BATCH REDUCTION (block order) AND CONVERGENCE CHECK
                        // Each sample sums its blocks in block order; a short
                        // reduction is not worth a fork.
                        // =========================================================
                        const int32_t reduce_threads =
                            (static_cast<int64_t>(n) * n_blocks_batch >= (int64_t(1) << 18)) ? n_threads : 1;
                        threads::for_static(reduce_threads, n, [&](int32_t j_begin, int32_t j_end, int)
                        {
                            for (int32_t j = j_begin; j < j_end; ++j)
                            {
                                double sum = 0.0;
                                for (int32_t b = 0; b < n_blocks_batch; ++b)
                                {
                                    sum += block_accs[b].data[j];
                                }
                                mean_accumulator.data[j] += sum;
                            }
                        });

                        for (int32_t b = 0; b < n_blocks_batch; ++b)
                        {
                            const BlockStats &st = block_stats[b];
                            stage_total.valid += st.valid;
                            stage_total.attempted += st.attempted;
                            stage_total.iterations += st.iterations;
                            stage_total.max_iterations = std::max(stage_total.max_iterations, st.max_iterations);
                            stage_total.nonconverged += st.nonconverged;
                        }

                        if (batch_end >= base_trials)
                        {
                            ensemble_converged = true; // all trials done
                        }
                        else if (effective_adaptive &&
                                 batch_end >= config_.adaptive_min_trials &&
                                 stage_total.valid > 0)
                        {
                            // Relative L2 change of the running mean estimate
                            const double inv_valid = 1.0 / stage_total.valid;
                            double diff_sq = 0.0;
                            double curr_sq = 0.0;

                            for (int32_t j = 0; j < n; ++j)
                            {
                                double curr = mean_accumulator.data[j] * inv_valid;
                                double prev = prev_mean_estimate.data[j];
                                double d = curr - prev;
                                diff_sq += d * d;
                                curr_sq += curr * curr;
                            }

                            double rel_change = (curr_sq > 1e-20) ? std::sqrt(diff_sq / curr_sq) : 0.0;

                            if (rel_change < config_.adaptive_rel_tol)
                            {
                                ensemble_converged = true;
                            }
                            else
                            {
                                for (int32_t j = 0; j < n; ++j)
                                {
                                    prev_mean_estimate.data[j] = mean_accumulator.data[j] * inv_valid;
                                }
                            }
                        }

                        if (ensemble_converged)
                            break;
                    } // end batch loop

                    // =========================================================
                    // FINALIZE IMF (after all batches complete)
                    // =========================================================
                    if (stage_total.valid > 0)
                    {
                        const double scale = 1.0 / stage_total.valid;
                        double *__restrict acc = mean_accumulator.data;

                        EEMD_OMP_SIMD
                        for (int32_t j = 0; j < n; ++j)
                        {
                            acc[j] *= scale;
                        }

                        // Extract IMF
                        const double *__restrict r = r_current.data;
                        const double *__restrict m = mean_accumulator.data;
                        double *__restrict out = imf_storage[actual_imf_count].data;

                        EEMD_OMP_SIMD
                        for (int32_t j = 0; j < n; ++j)
                        {
                            out[j] = r[j] - m[j];
                        }

                        ++actual_imf_count;

                        // Diagnostics for this IMF
                        last_diag_.sift_iterations.push_back(stage_total.max_iterations);
                        last_diag_.mean_sift_iterations.push_back(
                            static_cast<double>(stage_total.iterations) / stage_total.valid);
                        last_diag_.nonconverged_trials.push_back(stage_total.nonconverged);
                        last_diag_.convergence_flags.push_back(stage_total.nonconverged == 0);
                        last_diag_.trials_per_imf.push_back(stage_total.valid);

                        // Update residue
                        std::memcpy(r_current.data, mean_accumulator.data, n * sizeof(double));

                        decay_factor *= config_.noise_decay;

                        // Check stopping criteria
                        if (is_monotonic(r_current.data, n, config_.monotonic_threshold) ||
                            count_extrema(r_current.data, n) < config_.min_extrema)
                        {
                            stop_decomposition = true;
                        }
                    }
                    else
                    {
                        stop_decomposition = true;
                    }
                } // end IMF loop
            } // the per-thread resources end here

            // Copy to output
            imfs.resize(actual_imf_count);
            for (int32_t k = 0; k < actual_imf_count; ++k)
            {
                imfs[k].resize(n);
                std::memcpy(imfs[k].data(), imf_storage[k].data, n * sizeof(double));
            }

            residue.resize(n);
            std::memcpy(residue.data(), r_current.data, n * sizeof(double));

            last_diag_.valid = true;
            return true;
        }

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
            if (ok)
            {
                // Orthogonality index (lower = better)
                // IO = sum(|<IMF_i, IMF_j>|) / sum(<IMF_i, IMF_i>)
                double cross_sum = 0.0;
                double auto_sum = 0.0;

                for (size_t i = 0; i < imfs.size(); ++i)
                {
                    double norm_i = 0.0;
                    for (int32_t k = 0; k < n; ++k)
                    {
                        norm_i += imfs[i][k] * imfs[i][k];
                    }
                    auto_sum += norm_i;

                    for (size_t j = i + 1; j < imfs.size(); ++j)
                    {
                        double inner = 0.0;
                        for (int32_t k = 0; k < n; ++k)
                        {
                            inner += imfs[i][k] * imfs[j][k];
                        }
                        cross_sum += std::abs(inner);
                    }
                }

                last_diag_.orthogonality_index = (auto_sum > 1e-10) ? cross_sum / auto_sum : 0.0;
            }
            diag = last_diag_;
            return ok;
        }

        /** Diagnostics of the most recent decompose() (orthogonality index only
         *  from decompose_with_diagnostics()). */
        const DecompositionDiagnostics &last_diagnostics() const { return last_diag_; }

        ICEEMDANConfig &config() { return config_; }
        const ICEEMDANConfig &config() const { return config_; }
        ProcessingMode mode() const { return mode_; }

    private:
        ICEEMDANConfig config_;
        ProcessingMode mode_ = ProcessingMode::Standard;
        DecompositionDiagnostics last_diag_;
    };

    // ============================================================================
    // Analysis Utilities (same as before)
    // ============================================================================

    inline double estimate_hurst_rs(const double *signal, int32_t n)
    {
        if (n < 20)
            return 0.5;

        std::vector<double> log_n;
        std::vector<double> log_rs;

        for (int32_t win = 10; win <= n / 2; win = static_cast<int32_t>(win * 1.5))
        {
            int32_t n_windows = n / win;
            if (n_windows < 2)
                break;

            double rs_sum = 0.0;
            int32_t rs_count = 0;

            for (int32_t w = 0; w < n_windows; ++w)
            {
                const double *chunk = signal + w * win;

                double mean = 0.0;
                for (int32_t i = 0; i < win; ++i)
                    mean += chunk[i];
                mean /= win;

                double cum = 0.0, max_cum = -1e30, min_cum = 1e30, var = 0.0;

                for (int32_t i = 0; i < win; ++i)
                {
                    double dev = chunk[i] - mean;
                    cum += dev;
                    var += dev * dev;
                    max_cum = std::max(max_cum, cum);
                    min_cum = std::min(min_cum, cum);
                }

                double range = max_cum - min_cum;
                double std_dev = std::sqrt(var / win);

                if (std_dev > 1e-10)
                {
                    rs_sum += range / std_dev;
                    ++rs_count;
                }
            }

            if (rs_count > 0)
            {
                log_n.push_back(std::log(static_cast<double>(win)));
                log_rs.push_back(std::log(rs_sum / rs_count));
            }
        }

        if (log_n.size() < 2)
            return 0.5;

        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
        int32_t m = static_cast<int32_t>(log_n.size());

        for (int32_t i = 0; i < m; ++i)
        {
            sum_x += log_n[i];
            sum_y += log_rs[i];
            sum_xy += log_n[i] * log_rs[i];
            sum_xx += log_n[i] * log_n[i];
        }

        double H = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x);
        return std::max(0.0, std::min(1.0, H));
    }

    /**
     * Normalized spectral entropy of the one-sided power spectrum (bins
     * 0..n/2) of the mean-removed signal: 0 = a single tone, 1 = flat.
     */
    inline double compute_spectral_entropy(const double *signal, int32_t n)
    {
        if (n < 4)
            return 1.0;

        double mean = 0.0;
        for (int32_t i = 0; i < n; ++i)
            mean += signal[i];
        mean /= n;

        std::vector<double> centered(n);
        for (int32_t i = 0; i < n; ++i)
            centered[i] = signal[i] - mean;

        const int32_t n_freq = n / 2 + 1;
        std::vector<double> psd(n_freq);
        if (!fft::power_spectrum(centered.data(), n, psd.data()))
            return 1.0;

        double total_power = 0.0;
        for (int32_t i = 0; i < n_freq; ++i)
            total_power += psd[i];

        if (total_power < 1e-15)
            return 1.0;

        double entropy = 0.0;
        for (int32_t i = 0; i < n_freq; ++i)
        {
            double p = psd[i] / total_power;
            if (p > 1e-15)
                entropy -= p * std::log(p);
        }

        double max_entropy = std::log(static_cast<double>(n_freq));
        return entropy / max_entropy;
    }

    struct IMFAnalysis
    {
        double hurst;
        double spectral_entropy;
        double energy;
        double mean_frequency;
        bool likely_noise;
        bool likely_structure;
    };

    inline IMFAnalysis analyze_imf(const double *imf, int32_t n, double sample_rate = 1.0)
    {
        IMFAnalysis result;

        result.hurst = estimate_hurst_rs(imf, n);
        result.spectral_entropy = compute_spectral_entropy(imf, n);

        result.energy = 0.0;
        for (int32_t i = 0; i < n; ++i)
        {
            result.energy += imf[i] * imf[i];
        }

        // Simplified mean frequency (skip Hilbert for speed)
        result.mean_frequency = 0.0;

        bool hurst_noise = (result.hurst > 0.4 && result.hurst < 0.6);
        bool high_entropy = (result.spectral_entropy > 0.8);

        result.likely_noise = hurst_noise && high_entropy;
        result.likely_structure = !hurst_noise || (result.spectral_entropy < 0.6);

        return result;
    }

} // namespace eemd

#endif // ICEEMDAN_MKL_OPT_HPP
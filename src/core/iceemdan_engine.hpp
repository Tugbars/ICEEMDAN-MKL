/**
 * ICEEMDAN (Improved Complete Ensemble EMD with Adaptive Noise): the engine.
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
 * This is the implementation, with every setting of the algorithm. Programs
 * use iceemdan.hpp (C++) or iceemdan.h (C), which stand on it.
 *
 * License: MIT
 */

#ifndef ICEEMDAN_ENGINE_HPP
#define ICEEMDAN_ENGINE_HPP

#include "eemd_local_mean.hpp"
#include "eemd_noise.hpp"
#include "eemd_threads.hpp"

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>

// One copy of a function's machine code for every caller: decompose() and
// decompose_batch() must compute the same bits, and under -ffast-math two
// inlined copies of the same code may sum in different orders
#if defined(_MSC_VER) && !defined(__clang__)
#define EEMD_NOINLINE __declspec(noinline)
#else
#define EEMD_NOINLINE __attribute__((noinline))
#endif

namespace eemd
{

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
    // ICEEMDAN
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

#endif // ICEEMDAN_ENGINE_HPP

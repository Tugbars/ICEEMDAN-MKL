/**
 * EEMD (Ensemble Empirical Mode Decomposition) and plain EMD.
 *
 * The sifting is eemd_sift.hpp's, the noise eemd_noise.hpp's, the threads
 * eemd_threads.hpp's; the Hilbert transform runs on VectorFFT (eemd_fft.hpp).
 *
 * License: MIT
 */

#ifndef EEMD_HPP
#define EEMD_HPP

#include "eemd_fft.hpp" // real FFTs (VectorFFT)
#include "eemd_hht.hpp" // Hilbert-Huang analysis of the IMFs
#include "eemd_noise.hpp"
#include "eemd_sift.hpp"
#include "eemd_threads.hpp"

// ============================================================================
// Hardware Constants (tune for your CPU)
// ============================================================================

#define EEMD_P_CORES 8

/**
 * Low-latency mode: n_cores threads that never sleep between decompositions
 * (each worker keeps one core busy).
 */
inline void eemd_init_low_latency(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    eemd::set_num_threads(n_cores);
    eemd::set_thread_block_ms(-1.0);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD: Low-latency mode (%d threads, never sleeping, DAZ/FTZ)\n", n_cores);
    }
}

/**
 * Throughput mode: two threads per core (hyperthreading); a worker sleeps as
 * soon as its share is done.
 */
inline void eemd_init_throughput(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    eemd::set_num_threads(n_cores * 2);
    eemd::set_thread_block_ms(0.0);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD: Throughput mode (%d threads with HT)\n", n_cores * 2);
    }
}

namespace eemd
{

    // ============================================================================
    // EEMD Main Class
    // ============================================================================

    class EEMD
    {
    public:
        explicit EEMD(const EEMDConfig &config = EEMDConfig())
            : config_(config)
        {
        }

        bool decompose(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            int32_t &n_imfs)
        {
            if (n < 4)
                return false;

            // Compute signal stats
            double signal_mean = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_mean)
            for (int32_t i = 0; i < n; ++i)
            {
                signal_mean += signal[i];
            }
            signal_mean /= n;

            double signal_var = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_var)
            for (int32_t i = 0; i < n; ++i)
            {
                const double d = signal[i] - signal_mean;
                signal_var += d * d;
            }
            const double signal_std = std::sqrt(signal_var / n);
            const double noise_amplitude = config_.noise_std * signal_std;

            const int32_t max_imfs = config_.max_imfs;

            // Global accumulator
            std::vector<AlignedBuffer<double>> global_sum(max_imfs);
            for (auto &buf : global_sum)
            {
                buf.resize(n);
                buf.zero();
            }

            int32_t global_max_imfs = 0;

            // Per-thread sums, reduced afterwards in thread order (not in the
            // order the threads finish), so a given thread count always
            // produces the same result bit for bit. Each trial's noise comes
            // from its own stream (trial_gaussian), so the noise of every
            // trial is the same at any thread count.
            const int32_t n_threads_used = threads::team_size(config_.num_threads);
            std::vector<std::vector<AlignedBuffer<double>>> thread_sums(n_threads_used);
            std::vector<int32_t> thread_max_imfs(n_threads_used, 0);

            // Thread tid takes the trials [trial_begin, trial_end): consecutive
            // ranges in thread order
            threads::for_static(n_threads_used, config_.ensemble_size,
                                [&](int32_t trial_begin, int32_t trial_end, int tid)
            {
                // Thread-local accumulator
                std::vector<AlignedBuffer<double>> &thread_sum = thread_sums[tid];
                thread_sum.resize(max_imfs);
                for (auto &buf : thread_sum)
                {
                    buf.resize(n);
                    buf.zero();
                }

                int32_t local_max_imfs = 0;

                // Thread-local buffers
                AlignedBuffer<double> noisy_signal(n);
                AlignedBuffer<double> noise(n);
                std::vector<AlignedBuffer<double>> local_imfs(max_imfs);
                for (auto &buf : local_imfs)
                {
                    buf.resize(n);
                }

                Sifter sifter(n, config_);

                for (int32_t trial = trial_begin; trial < trial_end; ++trial)
                {
                    // Generate this trial's noise (depends on seed and trial only)
                    if (!trial_gaussian(config_.rng_seed, static_cast<uint32_t>(trial),
                                        n, noise.data, 0.0, noise_amplitude))
                    {
                        noise.zero();
                    }

                    // Add noise
                    const double *__restrict sig = signal;
                    const double *__restrict nz = noise.data;
                    double *__restrict ns = noisy_signal.data;

                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        ns[i] = sig[i] + nz[i];
                    }

                    // EMD
                    int32_t imf_count = 0;
                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        if (!sifter.sift_imf(noisy_signal.data, local_imfs[k].data, n))
                        {
                            break;
                        }
                        ++imf_count;
                    }

                    local_max_imfs = std::max(local_max_imfs, imf_count);

                    // Thread-local accumulation
                    for (int32_t k = 0; k < imf_count; ++k)
                    {
                        double *__restrict ts = thread_sum[k].data;
                        const double *__restrict li = local_imfs[k].data;

                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            ts[i] += li[i];
                        }
                    }
                }

                thread_max_imfs[tid] = local_max_imfs;
            });

            // Ordered reduction: element i sums threads 0, 1, 2, ... in turn
            threads::for_static(n_threads_used, n, [&](int32_t i_begin, int32_t i_end, int)
            {
                for (int32_t k = 0; k < max_imfs; ++k)
                {
                    double *__restrict gs = global_sum[k].data;
                    for (int32_t i = i_begin; i < i_end; ++i)
                    {
                        double acc = 0.0;
                        for (int32_t t = 0; t < n_threads_used; ++t)
                        {
                            acc += thread_sums[t][k].data[i];
                        }
                        gs[i] += acc;
                    }
                }
            });

            for (int32_t t = 0; t < n_threads_used; ++t)
            {
                global_max_imfs = std::max(global_max_imfs, thread_max_imfs[t]);
            }

            n_imfs = global_max_imfs;

            // Ensemble average
            const double scale = 1.0 / config_.ensemble_size;
            imfs.resize(n_imfs);

            for (int32_t k = 0; k < n_imfs; ++k)
            {
                imfs[k].resize(n);
                const double *__restrict gs = global_sum[k].data;
                double *__restrict out = imfs[k].data();

                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    out[i] = gs[i] * scale;
                }
            }

            return true;
        }

        bool decompose_emd(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue)
        {
            if (n < 4)
                return false;

            AlignedBuffer<double> work(n);
            std::memcpy(work.data, signal, n * sizeof(double));

            Sifter sifter(n, config_);

            imfs.clear();
            imfs.reserve(config_.max_imfs);

            for (int32_t k = 0; k < config_.max_imfs; ++k)
            {
                std::vector<double> imf(n);
                if (!sifter.sift_imf(work.data, imf.data(), n))
                {
                    break;
                }
                imfs.push_back(std::move(imf));
            }

            residue.resize(n);
            std::memcpy(residue.data(), work.data, n * sizeof(double));

            return true;
        }

        EEMDConfig &config() { return config_; }
        const EEMDConfig &config() const { return config_; }

    private:
        EEMDConfig config_;
    };

    // ============================================================================
    // Hilbert Transform - Instantaneous Frequency
    // ============================================================================

    /**
     * Hilbert transform y = H{x} (imaginary part of the analytic signal).
     * Any n >= 1; FFTs through eemd_fft.hpp (VectorFFT).
     */
    inline bool compute_hilbert_transform(const double *signal, int32_t n, double *hilbert_out)
    {
        return fft::hilbert(signal, n, hilbert_out);
    }

    /**
     * Instantaneous frequency in Hz (cycles per sample for a sample_rate of
     * 1) from the phase of the analytic signal x + i H{x}; see
     * hht::instantaneous() (eemd_hht.hpp), which also gives the amplitude and
     * the phase. inst_freq must hold n values; n >= 3.
     */
    inline bool compute_instantaneous_frequency(
        const double *imf,
        int32_t n,
        double *inst_freq,
        double sample_rate = 1.0)
    {
        if (n < 3)
            return false;
        return hht::instantaneous(imf, n, sample_rate, nullptr, inst_freq, nullptr);
    }

} // namespace eemd

#endif // EEMD_HPP

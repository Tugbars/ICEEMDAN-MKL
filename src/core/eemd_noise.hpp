/**
 * The noise of the ensembles: a realization for every trial, each from its
 * own random stream, and the noise banks of ICEEMDAN, which hold the EMD
 * modes of the realizations (independent ones, or windows of one long
 * decomposed realization).
 *
 * License: MIT
 */

#ifndef EEMD_NOISE_HPP
#define EEMD_NOISE_HPP

#include "eemd_rng.hpp" // Philox4x32-10 and the inverse normal distribution function
#include "eemd_sift.hpp"
#include "eemd_threads.hpp"

namespace eemd
{

    // ============================================================================
    // Per-trial random streams
    //
    // Trial t of seed s draws its noise from its own Philox4x32-10 stream, keyed
    // (s, t) (eemd_rng.hpp). The noise of a trial therefore depends only on
    // (s, t): not on the thread that runs it, the number of threads, or the
    // order trials are scheduled in.
    // ============================================================================

    /** Fill out[0..n) with N(mean, sigma) noise of trial `trial` under `seed`. */
    inline bool trial_gaussian(uint32_t seed, uint32_t trial, int32_t n, double *out,
                               double mean = 0.0, double sigma = 1.0)
    {
        if (n < 0 || (n > 0 && !out))
            return false;
        rng::gaussian(seed, trial, n, out, mean, sigma);
        return true;
    }

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
    // Standard Noise Bank: an independent realization for every trial (the
    // published algorithm)
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

} // namespace eemd

#endif // EEMD_NOISE_HPP

/**
 * EMD and EEMD: the engine.
 *
 * EMD (Huang et al. 1998) by the rules of emd.m of Rilling, Flandrin and
 * Goncalves (2003), the EMD the ICEEMDAN authors' code is built on: modes are
 * sifted out of the signal one after another, each until emd.m's stop rule is
 * met, with cubic spline envelopes continued past the ends by its symmetric
 * extension, until what is left has fewer than three extrema or is 1e-10
 * times smaller than the signal. The sifting is eemd_sift.hpp's, the same
 * that decomposes the noise of ICEEMDAN.
 *
 * EEMD (Wu & Huang 2009): the mean of the EMDs of the signal with white
 * noise added, a different realization for every copy. Mode k of the result
 * is the mean of the copies' modes k (a copy with fewer modes counts as zero
 * there), the residue the mean of their residues. For a given seed the
 * result is the same, bit for bit, at any number of threads: the copies are
 * summed in fixed blocks, in copy order, and the blocks in block order.
 *
 * This is the implementation. Programs use emd.hpp (C++) or emd.h (C), which
 * stand on it.
 *
 * License: MIT
 */

#ifndef EMD_ENGINE_HPP
#define EMD_ENGINE_HPP

#include "eemd_noise.hpp" // the noise of the copies
#include "eemd_sift.hpp"
#include "eemd_signal.hpp"
#include "eemd_threads.hpp"

#include <memory>

namespace eemd
{

    class EEMD
    {
    public:
        explicit EEMD(const EEMDConfig &config = EEMDConfig())
            : config_(config)
        {
        }

        EEMDConfig &config() { return config_; }
        const EEMDConfig &config() const { return config_; }

        /**
         * The EMD of signal (n >= 4 samples): its IMFs, from the fastest
         * oscillation to the slowest, and the residue; they add up to the
         * signal. NaN and Inf samples are filled in first, by straight lines
         * from their neighbours; *repaired, when given, counts them. False
         * for bad arguments.
         */
        bool decompose_emd(const double *signal, int32_t n, std::vector<std::vector<double>> &imfs,
                           std::vector<double> &residue, int32_t *repaired = nullptr)
        {
            imfs.clear();
            residue.clear();
            if (!signal || n < 4 || config_.max_imfs < 1)
                return false;
            Workspace ws(n, config_);
            emd_of(signal, n, ws, imfs, residue, repaired);
            return true;
        }

        /**
         * The EMDs of count signals of n samples each, stored one after
         * another (signal s at signals + s * n), a signal per thread. Each is
         * what decompose_emd() gives for that signal alone.
         */
        bool decompose_emd_batch(const double *signals, int32_t count, int32_t n,
                                 std::vector<std::vector<std::vector<double>>> &imfs,
                                 std::vector<std::vector<double>> &residues, std::vector<int32_t> *repaired = nullptr)
        {
            imfs.assign(static_cast<size_t>(std::max(count, 0)), {});
            residues.assign(static_cast<size_t>(std::max(count, 0)), {});
            if (repaired)
                repaired->assign(static_cast<size_t>(std::max(count, 0)), 0);
            if (!signals || count < 1 || n < 4 || config_.max_imfs < 1)
                return false;
            const int32_t team = std::min(threads::team_size(config_.num_threads), count);
            std::vector<std::unique_ptr<Workspace>> ws(static_cast<size_t>(team));
            threads::for_dynamic(team, count, [&](int32_t s, int slot)
            {
                if (!ws[slot])
                    ws[slot].reset(new Workspace(n, config_));
                emd_of(signals + static_cast<size_t>(s) * n, n, *ws[slot], imfs[s], residues[s],
                       repaired ? &(*repaired)[s] : nullptr);
            });
            return true;
        }

        /**
         * The EEMD of signal (n >= 4 samples): the mean over ensemble_size
         * copies of the signal, each with its own white noise of standard
         * deviation noise_std times the signal's, of their EMDs. The IMFs
         * and the residue add up to the signal plus the mean of the noise
         * that was added (a property of EEMD: it shrinks with the ensemble
         * size, and it is what ICEEMDAN removes). NaN and Inf samples are
         * filled in first; *repaired, when given, counts them. False for bad
         * arguments.
         */
        bool decompose(const double *signal, int32_t n, std::vector<std::vector<double>> &imfs,
                       std::vector<double> &residue, int32_t *repaired = nullptr)
        {
            imfs.clear();
            residue.clear();
            if (!signal || n < 4 || config_.ensemble_size < 1 || config_.max_imfs < 1)
                return false;

            AlignedBuffer<double> clean(n);
            std::memcpy(clean.data, signal, n * sizeof(double));
            const int32_t bad = sanitize_signal(clean.data, n);
            if (repaired)
                *repaired = bad;
            // the standard deviation with n - 1, as MATLAB's std
            const double amplitude = config_.noise_std * compute_sample_std(clean.data, n);

            // Copies are summed in blocks of block_copies (in copy order), and
            // the blocks in block order: nothing depends on the thread count
            const int32_t copies = config_.ensemble_size;
            const int32_t block_copies = std::max(1, (copies + 63) / 64); // at most 64 blocks
            const int32_t n_blocks = (copies + block_copies - 1) / block_copies;
            std::vector<Block> blocks(static_cast<size_t>(n_blocks));
            for (Block &b : blocks)
                b.modes = std::vector<AlignedBuffer<double>>(static_cast<size_t>(config_.max_imfs));

            const int32_t team = threads::team_size(config_.num_threads);
            std::vector<std::unique_ptr<Workspace>> ws(static_cast<size_t>(team));
            threads::for_dynamic(team, n_blocks, [&](int32_t b, int slot)
            {
                if (!ws[slot])
                    ws[slot].reset(new Workspace(n, config_));
                sum_block(blocks[b], b * block_copies, std::min(copies, (b + 1) * block_copies), clean.data, n,
                          amplitude, *ws[slot]);
            });

            int32_t n_imfs = 0;
            for (const Block &b : blocks)
                n_imfs = std::max(n_imfs, b.count);
            imfs.assign(static_cast<size_t>(n_imfs), std::vector<double>(static_cast<size_t>(n)));
            residue.assign(static_cast<size_t>(n), 0.0);
            const double scale = 1.0 / copies;
            const int32_t reduce_team = (static_cast<int64_t>(n) * n_blocks >= (int64_t(1) << 18)) ? team : 1;
            threads::for_static(reduce_team, n, [&](int32_t j_begin, int32_t j_end, int)
            {
                for (int32_t k = 0; k < n_imfs; ++k)
                    mean_of_blocks(blocks, k, scale, imfs[k].data(), j_begin, j_end);
                mean_of_blocks(blocks, -1, scale, residue.data(), j_begin, j_end);
            });
            return true;
        }

    private:
        /** What one thread needs to decompose signals of n samples. */
        struct Workspace
        {
            Sifter sifter;
            AlignedBuffer<double> work;  // the signal being decomposed; in the end its residue
            AlignedBuffer<double> mode;  // the mode just sifted out
            AlignedBuffer<double> noise; // EEMD: a copy's noise

            Workspace(int32_t n, const EEMDConfig &config) : sifter(n, config), work(n), mode(n), noise(n) {}
        };

        /** The sums of a block of copies: of their modes k, and of their residues. */
        struct Block
        {
            std::vector<AlignedBuffer<double>> modes; // modes[k]: allocated when a copy has a mode k
            AlignedBuffer<double> residue;
            int32_t count = 0; // the most modes of any of its copies
        };

        static double largest(const double *x, int32_t n)
        {
            double m = 0.0;
            for (int32_t i = 0; i < n; ++i)
                m = std::max(m, std::abs(x[i]));
            return m;
        }

        /**
         * The EMD of ws.work as emd.m runs it: modes are sifted out of it one
         * after another and handed to emit(k, mode), until what is left has
         * fewer than three extrema, or is 1e-10 times smaller than the input
         * (emd.m: "too small amplitude"), or max_imfs modes are out. Leaves
         * the residue in ws.work; returns the number of modes.
         */
        template <class Emit>
        static int32_t extract_modes(Workspace &ws, int32_t n, int32_t max_imfs, Emit &&emit)
        {
            const double too_small = 1e-10 * largest(ws.work.data, n);
            int32_t k = 0;
            for (; k < max_imfs; ++k)
            {
                if (largest(ws.work.data, n) < too_small)
                    break;
                if (!ws.sifter.sift_imf(ws.work.data, ws.mode.data, n))
                    break;
                emit(k, ws.mode.data);
            }
            return k;
        }

        /** One signal's EMD (decompose_emd and its batch). */
        EEMD_NOINLINE void emd_of(const double *signal, int32_t n, Workspace &ws,
                                  std::vector<std::vector<double>> &imfs, std::vector<double> &residue,
                                  int32_t *repaired) const
        {
            std::memcpy(ws.work.data, signal, n * sizeof(double));
            const int32_t bad = sanitize_signal(ws.work.data, n);
            if (repaired)
                *repaired = bad;
            imfs.clear();
            extract_modes(ws, n, config_.max_imfs, [&](int32_t, const double *mode)
                          { imfs.emplace_back(mode, mode + n); });
            residue.assign(ws.work.data, ws.work.data + n);
        }

        /** Copies [first, last) of the ensemble: their EMDs summed into block, in copy order. */
        EEMD_NOINLINE void sum_block(Block &block, int32_t first, int32_t last, const double *clean, int32_t n,
                                     double amplitude, Workspace &ws) const
        {
            block.residue.resize(n);
            block.residue.zero();
            block.count = 0;
            for (int32_t copy = first; copy < last; ++copy)
            {
                // this copy's noise depends on (seed, copy) only
                if (!trial_gaussian(config_.rng_seed, static_cast<uint32_t>(copy), n, ws.noise.data, 0.0, amplitude))
                    ws.noise.zero();
                const double *__restrict nz = ws.noise.data;
                double *__restrict w = ws.work.data;
                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    w[i] = clean[i] + nz[i];
                }

                const int32_t modes = extract_modes(ws, n, config_.max_imfs, [&](int32_t k, const double *mode)
                {
                    AlignedBuffer<double> &sum = block.modes[k];
                    if (!sum.data)
                    {
                        sum.resize(n);
                        sum.zero();
                    }
                    double *__restrict acc = sum.data;
                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        acc[i] += mode[i];
                    }
                });
                block.count = std::max(block.count, modes);

                double *__restrict res = block.residue.data;
                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    res[i] += w[i];
                }
            }
        }

        /**
         * out[j] = scale * (the blocks' sums of mode k at sample j, added in
         * block order), for j in [j_begin, j_end); k = -1: the residues.
         */
        EEMD_NOINLINE static void mean_of_blocks(const std::vector<Block> &blocks, int32_t k, double scale,
                                                 double *out, int32_t j_begin, int32_t j_end)
        {
            for (int32_t j = j_begin; j < j_end; ++j)
            {
                double sum = 0.0;
                for (const Block &b : blocks)
                {
                    const double *part = k < 0 ? b.residue.data : b.modes[static_cast<size_t>(k)].data;
                    if (part)
                        sum += part[j];
                }
                out[j] = sum * scale;
            }
        }

        EEMDConfig config_;
    };

} // namespace eemd

#endif // EMD_ENGINE_HPP

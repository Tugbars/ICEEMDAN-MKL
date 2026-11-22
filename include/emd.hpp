/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/**
 * @file emd.hpp
 * @brief VectorEMD for C++: EMD and EEMD.
 *
 *   emd::Config         what a decomposer computes; zeros are plain EMD
 *   emd::Decomposer     decomposes one signal, or many signals of one length
 *   emd::Decomposition  the IMFs and the residue of a signal
 *
 * EMD (Huang et al., 1998) by the rules of emd.m of Rilling, Flandrin and
 * Goncalves (2003): its stop rule for sifting, cubic spline envelopes, its
 * boundary extension. With an ensemble size it is EEMD (Wu & Huang, 2009):
 * the mean of the EMDs of the signal with white noise added.
 *
 * Header-only. The settings, the defaults and the results are those of the C
 * interface (emd.h), which is built on this header. ICEEMDAN is in
 * iceemdan.hpp, with the Hilbert-Huang analysis of the IMFs of any of them
 * (iceemdan::instantaneous, iceemdan::spectrum).
 *
 * @code
 * #include "emd.hpp"
 *
 * emd::Decomposer d;                               // plain EMD
 * emd::Decomposition r = d.decompose(x.data(), n); // r.imfs[j][i], r.residue[i]
 *
 * emd::Config c;
 * c.ensemble_size = 100;                           // EEMD with 100 noisy copies
 * emd::Decomposition e = emd::Decomposer(c).decompose(x.data(), n);
 * @endcode
 *
 * A call is served or refused. A refused call computes nothing and throws
 * std::invalid_argument for a value out of range; the text says why.
 */
#ifndef EMD_HPP
#define EMD_HPP

#include "eemd_system.hpp"
#include "emd_engine.hpp"

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace emd
{
    // ── the configuration ───────────────────────────────────────────────

    /**
     * @brief What a decomposer computes.
     *
     * As it stands it is plain EMD: the signal itself, no noise, as many IMFs
     * as it has. With ensemble_size it is EEMD.
     */
    struct Config
    {
        int ensemble_size = 0; ///< 0 = EMD. 1 or more = EEMD: that many copies of the signal, each
                               ///< with its own white noise, decomposed and averaged
        double noise = 0.0;    ///< EEMD: the standard deviation of the noise, relative to the
                               ///< signal's; 0 = 0.2. Not used by EMD
        int max_imfs = 0;      ///< the most IMFs to extract; 0 = as many as the signal has
        unsigned seed = 0;     ///< EEMD: the seed of the noise; 0 = 42. The same seed gives the same
                               ///< result. Not used by EMD
        int nthreads = 0;      ///< the threads a decomposition runs on, the caller's included;
                               ///< 0 = every logical processor. The result does not depend on it
    };

    // ── the result ──────────────────────────────────────────────────────

    /** @brief The decomposition of one signal. */
    struct Decomposition
    {
        std::vector<std::vector<double>> imfs; ///< imfs[j][i]: IMF j, from the fastest oscillation to the
                                               ///< slowest
        std::vector<double> residue;           ///< what is left: the trend. EMD: the IMFs and the residue
                                               ///< add up to the signal. EEMD: to the signal plus the mean
                                               ///< of the noise that was added, which shrinks with the
                                               ///< ensemble size (ICEEMDAN adds up exactly)
        int repaired = 0;                      ///< the NaN and Inf samples filled in before the decomposition
    };

    namespace detail
    {
        /** A message made as printf makes it. */
        inline std::string text(const char *format, ...)
        {
            char buffer[320];
            va_list args;
            va_start(args, format);
            std::vsnprintf(buffer, sizeof(buffer), format, args);
            va_end(args);
            return std::string(buffer);
        }
    } // namespace detail

    // ── the decomposer ──────────────────────────────────────────────────

    /**
     * @brief Decomposes signals with the settings it was made with.
     *
     * One decomposition at a time on a decomposer; several decomposers may
     * run at once on several threads.
     */
    class Decomposer
    {
    public:
        /**
         * @param config What it computes.
         * @throws std::invalid_argument for a value out of range.
         */
        explicit Decomposer(const Config &config = Config())
            : ensemble_(config.ensemble_size > 0)
        {
            const Config &c = config;
            if (c.ensemble_size < 0)
                throw std::invalid_argument(
                    detail::text("ensemble_size %d: 0 (EMD) or more (EEMD with that many copies)", c.ensemble_size));
            if (!eemd::is_finite_bits(c.noise) || !(c.noise >= 0.0))
                throw std::invalid_argument(detail::text("noise %g: 0 (0.2) or more", c.noise));
            if (c.max_imfs < 0 || c.max_imfs > 64)
                throw std::invalid_argument(
                    detail::text("max_imfs %d: 0 (as many as the signal has) to 64", c.max_imfs));
            if (c.nthreads < 0 || c.nthreads > 1024)
                throw std::invalid_argument(detail::text("nthreads %d: 0 (every processor) to 1024", c.nthreads));

            eemd::EEMDConfig &k = engine_.config(); // starts as emd.m's rules
            k.max_imfs = c.max_imfs > 0 ? c.max_imfs : 32;
            k.ensemble_size = c.ensemble_size;
            k.noise_std = c.noise > 0.0 ? c.noise : 0.2;
            k.rng_seed = c.seed != 0 ? c.seed : 42u;
            k.num_threads = c.nthreads > 0 ? c.nthreads : processors();
        }

        /**
         * @brief Decomposes one signal.
         * @param signal n values. NaN and Inf samples are filled in from their
         *               neighbours by a straight line.
         * @param n      4 or more.
         * @throws std::invalid_argument for a value out of range.
         */
        Decomposition decompose(const double *signal, int n)
        {
            return std::move(decompose(signal, 1, n).front());
        }

        /** @brief Decomposes one signal. */
        Decomposition decompose(const std::vector<double> &signal)
        {
            return decompose(signal.data(), static_cast<int>(signal.size()));
        }

        /**
         * @brief Decomposes many signals of one length: the channels of a
         *        recording, or its epochs.
         *
         * Every signal gets the result it would get alone, bit for bit.
         *
         * @param signals count x n values: signal s at signals[s * n].
         * @param count   1 or more.
         * @param n       4 or more.
         * @return One decomposition for each signal, in their order.
         * @throws std::invalid_argument for a value out of range.
         */
        std::vector<Decomposition> decompose(const double *signals, int count, int n)
        {
            if (!signals)
                throw std::invalid_argument("signals is NULL");
            if (count < 1)
                throw std::invalid_argument(detail::text("count %d: 1 signal or more", count));
            if (n < 4)
                throw std::invalid_argument(detail::text("n %d: 4 samples or more", n));

            std::vector<Decomposition> out(static_cast<size_t>(count));
            if (!ensemble_)
            {
                // EMD: a signal per thread
                std::vector<std::vector<std::vector<double>>> imfs;
                std::vector<std::vector<double>> residues;
                std::vector<int32_t> repaired;
                engine_.decompose_emd_batch(signals, count, n, imfs, residues, &repaired);
                for (size_t s = 0; s < out.size(); ++s)
                {
                    out[s].imfs = std::move(imfs[s]);
                    out[s].residue = std::move(residues[s]);
                    out[s].repaired = repaired[s];
                }
            }
            else
            {
                // EEMD: one signal after another, the copies of each on all the threads
                for (size_t s = 0; s < out.size(); ++s)
                {
                    int32_t repaired = 0;
                    engine_.decompose(signals + s * static_cast<size_t>(n), n, out[s].imfs, out[s].residue,
                                      &repaired);
                    out[s].repaired = repaired;
                }
            }
            return out;
        }

        /** @brief Decomposes many signals of one length, one per row. */
        std::vector<Decomposition> decompose(const std::vector<std::vector<double>> &signals)
        {
            if (signals.empty())
                throw std::invalid_argument("count 0: 1 signal or more");
            const size_t n = signals.front().size();
            std::vector<double> rows(signals.size() * n);
            for (size_t s = 0; s < signals.size(); ++s)
            {
                if (signals[s].size() != n)
                    throw std::invalid_argument(detail::text("signal %d has %d samples, the first has %d",
                                                             static_cast<int>(s), static_cast<int>(signals[s].size()),
                                                             static_cast<int>(n)));
                std::copy(signals[s].begin(), signals[s].end(), rows.begin() + static_cast<std::ptrdiff_t>(s * n));
            }
            return decompose(rows.data(), static_cast<int>(signals.size()), static_cast<int>(n));
        }

        /** @brief The ensemble it computes: 0 = plain EMD, 1 or more = EEMD with that many copies. */
        int ensemble_size() const { return ensemble_ ? engine_.config().ensemble_size : 0; }

        /** @brief The threads a decomposition runs on. */
        int threads() const { return engine_.config().num_threads; }

        /** @brief This machine's logical processors: the threads a decomposer runs on when its configuration names none. */
        static int processors() { return eemd::machine::logical_processors(); }

    private:
        eemd::EEMD engine_;
        bool ensemble_;
    };

} // namespace emd

#endif // EMD_HPP

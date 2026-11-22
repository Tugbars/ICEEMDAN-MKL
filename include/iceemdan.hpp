/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/**
 * @file iceemdan.hpp
 * @brief VectorEMD for C++: ICEEMDAN decomposition and Hilbert-Huang analysis.
 *
 *   iceemdan::Config         what a decomposer computes; zeros are the published algorithm
 *   iceemdan::Decomposer     decomposes one signal, or many signals of one length
 *   iceemdan::Decomposition  the IMFs and the residue of a signal, and how they came about
 *   iceemdan::instantaneous  the amplitude, frequency and phase of an IMF over time
 *   iceemdan::spectrum       the Hilbert spectrum of IMFs
 *
 * Header-only. The settings, the defaults and the results are those of the C
 * interface (iceemdan.h), which is built on this header.
 *
 * @code
 * #include "iceemdan.hpp"
 *
 * iceemdan::Decomposer d;                               // the published algorithm
 * iceemdan::Decomposition r = d.decompose(x.data(), n); // r.imfs[j][i], r.residue[i]
 * @endcode
 *
 * A call is served or refused. A refused call computes nothing and throws:
 * std::invalid_argument for a value out of range, std::runtime_error when the
 * noise of the ensemble would not fit in memory. The text says why.
 *
 * The library takes care of the rest: the channels of a recording go through
 * one call and are spread over the threads together; the noise a decomposer
 * decomposed for one signal length is kept for the next signal of that
 * length, which then takes half the time or less with the same result; NaN
 * and Inf samples are filled in.
 */
#ifndef ICEEMDAN_HPP
#define ICEEMDAN_HPP

#include "eemd_hht.hpp"
#include "eemd_system.hpp"
#include "iceemdan_engine.hpp"

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/** "MAJOR.MINOR.PATCH"; the same as in iceemdan.h (CMake checks that). */
#ifndef ICEEMDAN_VERSION_STRING
#define ICEEMDAN_VERSION_STRING "1.1.0"
#endif

namespace iceemdan
{
    // ── the configuration ───────────────────────────────────────────────

    /**
     * @brief What a decomposer computes.
     *
     * As it stands it is ICEEMDAN as published (Colominas et al., 2014),
     * computed as the authors' code computes it: 100 noisy copies, a noise
     * amplitude of 0.2, as many IMFs as the signal has, on every processor
     * of this machine.
     */
    struct Config
    {
        int ensemble_size = 0; ///< the noisy copies averaged for each mode; 0 = 100
        double noise = 0.0;    ///< the noise amplitude, the paper's epsilon, relative to the
                               ///< standard deviation of the signal; 0 = 0.2
        int max_imfs = 0;      ///< the most IMFs to extract; 0 = as many as the signal has
        unsigned seed = 0;     ///< the seed of the noise; 0 = 42. The same seed gives the same result
        int nthreads = 0;      ///< the threads a decomposition runs on, the caller's included;
                               ///< 0 = every logical processor. The result does not depend on it
        bool fast = false;     ///< 8 to 12 times faster, no longer the published algorithm: a simpler
                               ///< rule for when sifting stops, one envelope mean for each noisy
                               ///< copy, and the noise of all copies cut from one long decomposed
                               ///< noise signal
    };

    // ── the result ──────────────────────────────────────────────────────

    /** @brief How the sifting went for one IMF. */
    struct ImfStats
    {
        int copies = 0;               ///< the noisy copies averaged for this IMF
        int iterations_max = 0;       ///< the most sifting iterations any copy needed
        double iterations_mean = 0.0; ///< their mean over the copies
        int unconverged = 0;          ///< the copies that reached the iteration limit (2000) before
                                      ///< the stop rule was met; 0 is the normal case
    };

    /** @brief The decomposition of one signal. */
    struct Decomposition
    {
        std::vector<std::vector<double>> imfs; ///< imfs[j][i]: IMF j, from the fastest oscillation to the
                                               ///< slowest. The IMFs and the residue add up to the signal
        std::vector<double> residue;           ///< what is left: the trend
        double orthogonality = 0.0;            ///< the orthogonality index of the IMFs: the sum over pairs
                                               ///< of |<IMF_i, IMF_j>| divided by the sum of
                                               ///< <IMF_i, IMF_i>. Lower is better
        int repaired = 0;                      ///< the NaN and Inf samples filled in before the decomposition
        std::vector<ImfStats> stats;           ///< stats[j]: how the sifting went for IMF j
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

        inline double gib(size_t bytes)
        {
            return static_cast<double>(bytes) / static_cast<double>(static_cast<size_t>(1) << 30);
        }
    } // namespace detail

    // ── the decomposer ──────────────────────────────────────────────────

    /**
     * @brief Decomposes signals with the settings it was made with.
     *
     * It keeps the noise it decomposed for one signal length: further signals
     * of that length skip that step and get the result they would get alone.
     * Another length replaces the kept noise. The kept noise takes about
     * 14 kB for every sample of a signal at the default ensemble; a length
     * whose noise would take more than half of this machine's memory is
     * refused (the library does not change the algorithm on its own to make
     * it fit).
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
        {
            const Config &c = config;
            if (c.ensemble_size < 0)
                throw std::invalid_argument(detail::text("ensemble_size %d: 0 (100 copies) or more", c.ensemble_size));
            if (!eemd::is_finite_bits(c.noise) || !(c.noise >= 0.0))
                throw std::invalid_argument(detail::text("noise %g: 0 (0.2) or more", c.noise));
            if (c.max_imfs < 0 || c.max_imfs > 64)
                throw std::invalid_argument(
                    detail::text("max_imfs %d: 0 (as many as the signal has) to 64", c.max_imfs));
            if (c.nthreads < 0 || c.nthreads > 1024)
                throw std::invalid_argument(detail::text("nthreads %d: 0 (every processor) to 1024", c.nthreads));

            eemd::ICEEMDANConfig &k = engine_.config(); // starts as the published algorithm
            k.ensemble_size = c.ensemble_size > 0 ? c.ensemble_size : 100;
            k.noise_std = c.noise > 0.0 ? c.noise : 0.2;
            k.max_imfs = c.max_imfs > 0 ? c.max_imfs : 32;
            k.rng_seed = c.seed != 0 ? c.seed : 42u;
            k.num_threads = c.nthreads > 0 ? c.nthreads : processors();
            k.keep_noise_bank = true;
            k.noise_bank_max_bytes = eemd::machine::installed_memory() / 2;
            if (c.fast)
            {
                k.sift_stop_rule = eemd::SiftStopRule::SD;
                k.local_mean_method = eemd::LocalMeanMethod::SingleEnvelope;
                k.use_circular_bank = true;
            }
        }

        /**
         * @brief Decomposes one signal.
         * @param signal n values. NaN and Inf samples are filled in from their
         *               neighbours by a straight line.
         * @param n      4 or more.
         * @throws std::invalid_argument, std::runtime_error (see the file comment).
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
         * Every signal gets the result it would get alone, bit for bit; the
         * batch is faster, because the signals share one decomposed noise
         * and are spread over the threads together.
         *
         * @param signals count x n values: signal s at signals[s * n].
         * @param count   1 or more.
         * @param n       4 or more.
         * @return One decomposition for each signal, in their order.
         * @throws std::invalid_argument, std::runtime_error (see the file comment).
         */
        std::vector<Decomposition> decompose(const double *signals, int count, int n)
        {
            if (!signals)
                throw std::invalid_argument("signals is NULL");
            if (count < 1)
                throw std::invalid_argument(detail::text("count %d: 1 signal or more", count));
            if (n < 4)
                throw std::invalid_argument(detail::text("n %d: 4 samples or more", n));

            // The decomposed noise is kept in memory. The library does not change
            // the algorithm on its own to make it fit: the call is refused.
            const eemd::ICEEMDANConfig &k = engine_.config();
            if (!k.use_circular_bank && engine_.noise_bank_bytes(n) > k.noise_bank_max_bytes)
                throw std::runtime_error(detail::text(
                    "the noise of %d copies of %d samples takes %.1f GiB decomposed, and this machine allows %.1f "
                    "(half its memory): a smaller ensemble_size, shorter signals, or the fast option",
                    k.ensemble_size, n, detail::gib(engine_.noise_bank_bytes(n)),
                    detail::gib(k.noise_bank_max_bytes)));

            std::vector<std::vector<std::vector<double>>> imfs;
            std::vector<std::vector<double>> residues;
            std::vector<eemd::DecompositionDiagnostics> facts;
            if (!engine_.decompose_batch(signals, count, n, imfs, residues, &facts))
                throw std::runtime_error(detail::text(
                    "the decomposed noise for signals of %d samples does not fit in the %.1f GiB this machine "
                    "allows (half its memory): shorter signals",
                    n, detail::gib(k.noise_bank_max_bytes)));

            noise_kept_ = facts.front().noise_bank_reused;
            std::vector<Decomposition> out(static_cast<size_t>(count));
            for (size_t s = 0; s < out.size(); ++s)
            {
                Decomposition &r = out[s];
                const eemd::DecompositionDiagnostics &g = facts[s];
                r.imfs = std::move(imfs[s]);
                r.residue = std::move(residues[s]);
                r.orthogonality = g.orthogonality_index;
                r.repaired = g.nan_count;
                r.stats.resize(r.imfs.size());
                for (size_t j = 0; j < r.stats.size(); ++j)
                {
                    r.stats[j].copies = g.trials_per_imf[j];
                    r.stats[j].iterations_max = g.sift_iterations[j];
                    r.stats[j].iterations_mean = g.mean_sift_iterations[j];
                    r.stats[j].unconverged = g.nonconverged_trials[j];
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

        /** @brief The threads a decomposition runs on. */
        int threads() const { return engine_.config().num_threads; }

        /** @brief Whether the last decomposition used the noise kept from an earlier call. */
        bool noise_kept() const { return noise_kept_; }

        /** @brief This machine's logical processors: the threads a decomposer runs on when its configuration names none. */
        static int processors() { return eemd::machine::logical_processors(); }

    private:
        eemd::ICEEMDAN engine_;
        bool noise_kept_ = false;
    };

    /** @brief "MAJOR.MINOR.PATCH" of this library. */
    inline const char *version() { return ICEEMDAN_VERSION_STRING; }

    // ── Hilbert-Huang analysis ──────────────────────────────────────────

    /**
     * @brief The amplitude, frequency and phase of one IMF at every sample,
     *        from its analytic signal.
     *
     * Within a period or two of each end of an IMF that does not fit the
     * window a whole number of times, the values are distorted.
     *
     * @param imf         n values.
     * @param n           2 or more.
     * @param sample_rate Samples per second, above 0.
     * @param amplitude   n values out. May be null.
     * @param frequency   n values out, in Hz. May be null.
     * @param phase       n values out, in radians, unwrapped. May be null.
     * @throws std::invalid_argument for a value out of range.
     */
    inline void instantaneous(const double *imf, int n, double sample_rate, double *amplitude, double *frequency,
                              double *phase)
    {
        if (!imf)
            throw std::invalid_argument("imf is NULL");
        if (n < 2)
            throw std::invalid_argument(detail::text("n %d: 2 samples or more", n));
        if (!eemd::is_finite_bits(sample_rate) || !(sample_rate > 0.0))
            throw std::invalid_argument(detail::text("sample_rate %g: above 0", sample_rate));
        if (!eemd::hht::instantaneous(imf, n, sample_rate, amplitude, frequency, phase))
            throw std::runtime_error(detail::text("the analytic signal of %d samples could not be computed", n));
    }

    /** @brief What instantaneous() gives for one IMF, n values each. */
    struct Instantaneous
    {
        std::vector<double> amplitude;
        std::vector<double> frequency; ///< Hz
        std::vector<double> phase;     ///< radians, unwrapped
    };

    /** @brief The amplitude, frequency and phase of one IMF at every sample. */
    inline Instantaneous instantaneous(const std::vector<double> &imf, double sample_rate)
    {
        Instantaneous out;
        out.amplitude.resize(imf.size());
        out.frequency.resize(imf.size());
        out.phase.resize(imf.size());
        instantaneous(imf.data(), static_cast<int>(imf.size()), sample_rate, out.amplitude.data(),
                      out.frequency.data(), out.phase.data());
        return out;
    }

    /**
     * @brief The Hilbert spectrum to compute. As it stands: 128 frequency
     *        bands from 0 to half the sample rate, a column for every sample,
     *        in energy.
     */
    struct SpectrumOptions
    {
        int frequency_bins = 0; ///< the frequency bands, of equal width; 0 = 128
        double fmin = 0.0;      ///< the lower edge of the first band, Hz
        double fmax = 0.0;      ///< the upper edge of the last band, Hz; 0 = sample_rate / 2
        int time_step = 0;      ///< the samples summed into each time column; 0 or 1 = a column for
                                ///< every sample
        bool amplitude = false; ///< false: each IMF adds its energy (amplitude squared); true: its amplitude
    };

    /** @brief The frequency bands of a spectrum with these options. */
    inline int spectrum_bins(const SpectrumOptions &options)
    {
        return options.frequency_bins > 0 ? options.frequency_bins : 128;
    }

    /** @brief The time columns of a spectrum of n samples with these options. */
    inline int spectrum_columns(int n, const SpectrumOptions &options)
    {
        const int step = options.time_step > 0 ? options.time_step : 1;
        return n > 0 ? (n + step - 1) / step : 0;
    }

    /**
     * @brief The Hilbert spectrum of IMFs: their energy over time and
     *        frequency, and the marginal spectrum.
     *
     * At every sample each IMF adds its energy to the band of its
     * instantaneous frequency; frequencies outside [fmin, fmax] are left out.
     *
     * @param imfs        n_imfs x n values: IMF j at imfs[j * n]. The residue
     *                    is usually left out.
     * @param n_imfs      1 or more.
     * @param n           2 or more.
     * @param sample_rate Samples per second, above 0.
     * @param options     The spectrum.
     * @param values      spectrum_bins() x spectrum_columns() values out: band b,
     *                    from fmin + b w to fmin + (b + 1) w with
     *                    w = (fmax - fmin) / bins, at values[b * columns]. May be null.
     * @param marginal    spectrum_bins() values out: each band summed over time.
     *                    May be null.
     * @throws std::invalid_argument for a value out of range.
     */
    inline void spectrum(const double *imfs, int n_imfs, int n, double sample_rate, const SpectrumOptions &options,
                         double *values, double *marginal)
    {
        const SpectrumOptions &o = options;
        if (!imfs)
            throw std::invalid_argument("imfs is NULL");
        if (n_imfs < 1)
            throw std::invalid_argument(detail::text("n_imfs %d: 1 IMF or more", n_imfs));
        if (n < 2)
            throw std::invalid_argument(detail::text("n %d: 2 samples or more", n));
        if (!eemd::is_finite_bits(sample_rate) || !(sample_rate > 0.0))
            throw std::invalid_argument(detail::text("sample_rate %g: above 0", sample_rate));
        if (o.frequency_bins < 0)
            throw std::invalid_argument(detail::text("frequency_bins %d: 0 (128) or more", o.frequency_bins));
        if (o.time_step < 0)
            throw std::invalid_argument(detail::text("time_step %d: 0 (every sample) or more", o.time_step));
        if (!eemd::is_finite_bits(o.fmin) || !eemd::is_finite_bits(o.fmax) || !(o.fmin >= 0.0) || !(o.fmax >= 0.0))
            throw std::invalid_argument(detail::text("fmin %g, fmax %g: 0 or more", o.fmin, o.fmax));
        const double top = o.fmax > 0.0 ? o.fmax : 0.5 * sample_rate;
        if (!(o.fmin < top))
            throw std::invalid_argument(detail::text("fmin %g is not below fmax %g", o.fmin, top));

        eemd::hht::SpectrumOptions opt;
        opt.frequency_bins = spectrum_bins(o);
        opt.fmin = o.fmin;
        opt.fmax = o.fmax;
        opt.time_step = o.time_step > 0 ? o.time_step : 1;
        opt.amplitude = o.amplitude;
        if (!eemd::hht::hilbert_spectrum(imfs, n_imfs, n, sample_rate, opt, values, marginal))
            throw std::runtime_error(
                detail::text("the spectrum of %d IMFs of %d samples could not be computed", n_imfs, n));
    }

    /** @brief A Hilbert spectrum: bins x columns values, and its marginal. */
    struct Spectrum
    {
        int bins = 0;                 ///< frequency bands (rows)
        int columns = 0;              ///< time columns
        std::vector<double> values;   ///< band b at time column c: values[b * columns + c]
        std::vector<double> marginal; ///< each band summed over time: bins values
    };

    /** @brief The Hilbert spectrum of IMFs of one length (the imfs of a Decomposition). */
    inline Spectrum spectrum(const std::vector<std::vector<double>> &imfs, double sample_rate,
                             const SpectrumOptions &options = SpectrumOptions())
    {
        if (imfs.empty())
            throw std::invalid_argument("n_imfs 0: 1 IMF or more");
        const size_t n = imfs.front().size();
        std::vector<double> rows(imfs.size() * n);
        for (size_t j = 0; j < imfs.size(); ++j)
        {
            if (imfs[j].size() != n)
                throw std::invalid_argument(detail::text("IMF %d has %d samples, the first has %d",
                                                         static_cast<int>(j), static_cast<int>(imfs[j].size()),
                                                         static_cast<int>(n)));
            std::copy(imfs[j].begin(), imfs[j].end(), rows.begin() + static_cast<std::ptrdiff_t>(j * n));
        }
        Spectrum out;
        out.bins = spectrum_bins(options);
        out.columns = spectrum_columns(static_cast<int>(n), options);
        out.values.assign(static_cast<size_t>(out.bins) * static_cast<size_t>(out.columns), 0.0);
        out.marginal.assign(static_cast<size_t>(out.bins), 0.0);
        spectrum(rows.data(), static_cast<int>(imfs.size()), static_cast<int>(n), sample_rate, options,
                 out.values.data(), out.marginal.data());
        return out;
    }

} // namespace iceemdan

#endif // ICEEMDAN_HPP

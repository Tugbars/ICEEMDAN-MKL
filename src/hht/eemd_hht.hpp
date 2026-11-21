/**
 * Hilbert-Huang analysis of the IMFs (Huang et al., Proc. R. Soc. A 454,
 * 1998): what a decomposition is usually followed by.
 *
 * - instantaneous(): the amplitude, phase and frequency of one IMF at every
 *   sample, from its analytic signal z = x + i H{x}.
 * - hilbert_spectrum(): the time-frequency map. At every sample, each IMF
 *   puts its instantaneous energy a^2 (or its amplitude) into the frequency
 *   bin of its instantaneous frequency. Its sum over time is the marginal
 *   spectrum: how much energy the signal has at each frequency.
 * - mean_frequency(): an IMF's instantaneous frequency averaged with its
 *   energy as the weight.
 *
 * Frequencies are in Hz for the sample rate given (cycles per sample for a
 * rate of 1). The Hilbert transform is computed with FFTs (eemd_fft.hpp),
 * which treat the IMF as periodic: within a period or two of each end the
 * amplitude and frequency of a non-periodic IMF are distorted, and are best
 * left out when they are read.
 *
 * License: MIT
 */

#ifndef EEMD_HHT_HPP
#define EEMD_HHT_HPP

#include "eemd_fft.hpp"
#include "eemd_threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace eemd
{
    namespace hht
    {
        constexpr double two_pi = 6.283185307179586476925286766559;

        /**
         * Instantaneous attributes of one IMF of n samples (n >= 2) at
         * sample_rate samples per second:
         *   amplitude[t]  |z(t)|
         *   phase[t]      the angle of z(t) in radians, unwrapped (continuous)
         *   frequency[t]  (1 / 2 pi) d phase / dt in Hz: the central difference
         *                 of the phase, one-sided at the first and last sample
         * Any output may be null. False when n < 2, sample_rate <= 0 or the
         * Hilbert transform fails.
         */
        inline bool instantaneous(const double *imf, int32_t n, double sample_rate,
                                  double *amplitude, double *frequency, double *phase)
        {
            if (n < 2 || !(sample_rate > 0.0) || !imf)
                return false;
            thread_local std::vector<double> h, ph;
            h.resize(static_cast<size_t>(n));
            ph.resize(static_cast<size_t>(n));
            if (!fft::hilbert(imf, n, h.data()))
                return false;

            // The phase, unwrapped as it goes: each step is the angle of
            // z[t] conj(z[t-1]), in (-pi, pi], so no jump of 2 pi can appear
            ph[0] = std::atan2(h[0], imf[0]);
            for (int32_t t = 1; t < n; ++t)
            {
                const double re = imf[t] * imf[t - 1] + h[t] * h[t - 1];
                const double im = h[t] * imf[t - 1] - imf[t] * h[t - 1];
                ph[t] = ph[t - 1] + std::atan2(im, re);
            }

            if (amplitude)
                for (int32_t t = 0; t < n; ++t)
                    amplitude[t] = std::sqrt(imf[t] * imf[t] + h[t] * h[t]);
            if (frequency)
            {
                const double scale = sample_rate / two_pi;
                frequency[0] = (ph[1] - ph[0]) * scale;
                for (int32_t t = 1; t < n - 1; ++t)
                    frequency[t] = 0.5 * (ph[t + 1] - ph[t - 1]) * scale;
                frequency[n - 1] = (ph[n - 1] - ph[n - 2]) * scale;
            }
            if (phase)
                for (int32_t t = 0; t < n; ++t)
                    phase[t] = ph[t];
            return true;
        }

        /**
         * The IMF's instantaneous frequency averaged over its samples with the
         * instantaneous energy a^2 as the weight, in Hz. 0 for an IMF that is
         * zero throughout or that cannot be transformed.
         */
        inline double mean_frequency(const double *imf, int32_t n, double sample_rate)
        {
            std::vector<double> a(static_cast<size_t>(std::max(n, 0))), f(a.size());
            if (!instantaneous(imf, n, sample_rate, a.data(), f.data(), nullptr))
                return 0.0;
            double num = 0.0, den = 0.0;
            for (int32_t t = 0; t < n; ++t)
            {
                const double e = a[t] * a[t];
                num += e * f[t];
                den += e;
            }
            return den > 0.0 ? num / den : 0.0;
        }

        /** Settings of hilbert_spectrum(). */
        struct SpectrumOptions
        {
            int32_t frequency_bins = 128; // bins of equal width from fmin to fmax
            double fmin = 0.0;            // Hz
            double fmax = 0.0;            // Hz; 0 or less: sample_rate / 2
            int32_t time_step = 1;        // samples per time column (columns = ceil(n / time_step))
            bool amplitude = false;       // false: the energy a^2; true: the amplitude a
        };

        /** Time columns of a spectrum of n samples. */
        inline int32_t spectrum_columns(int32_t n, const SpectrumOptions &opt)
        {
            const int32_t step = opt.time_step > 0 ? opt.time_step : 1;
            return n > 0 ? (n + step - 1) / step : 0;
        }

        /** The centre of frequency bin b, Hz. */
        inline double bin_centre(int32_t b, double sample_rate, const SpectrumOptions &opt)
        {
            const double fmax = opt.fmax > 0.0 ? opt.fmax : 0.5 * sample_rate;
            return opt.fmin + (b + 0.5) * (fmax - opt.fmin) / opt.frequency_bins;
        }

        /**
         * The Hilbert spectrum of n_imfs IMFs of n samples each (imf[k] points
         * at IMF k). spectrum receives frequency_bins rows of
         * spectrum_columns(n) values, row-major: row b is frequency bin b,
         * column c sums the samples c * time_step .. (c + 1) * time_step - 1.
         * marginal (frequency_bins values) receives each row summed over
         * time. Either output may be null. At each sample each IMF adds its
         * energy (or amplitude) to the bin of its instantaneous frequency;
         * frequencies below fmin or above fmax are left out.
         *
         * The IMFs' attributes are computed side by side on the library's
         * threads; the sums are made in IMF order, so the result does not
         * depend on the thread count. False on bad settings or a failed
         * transform.
         */
        inline bool hilbert_spectrum(const double *const *imf, int32_t n_imfs, int32_t n, double sample_rate,
                                     const SpectrumOptions &opt, double *spectrum, double *marginal)
        {
            const double fmax = opt.fmax > 0.0 ? opt.fmax : 0.5 * sample_rate;
            if (n_imfs < 0 || n < 2 || !(sample_rate > 0.0) || opt.frequency_bins < 1 || !(fmax > opt.fmin) ||
                opt.fmin < 0.0)
                return false;
            const int32_t bins = opt.frequency_bins;
            const int32_t step = opt.time_step > 0 ? opt.time_step : 1;
            const int32_t cols = spectrum_columns(n, opt);
            if (spectrum)
                std::fill(spectrum, spectrum + static_cast<size_t>(bins) * cols, 0.0);
            if (marginal)
                std::fill(marginal, marginal + bins, 0.0);

            // the attributes of every IMF
            std::vector<double> amp(static_cast<size_t>(n_imfs) * n), freq(amp.size());
            std::vector<char> ok(static_cast<size_t>(n_imfs), 0);
            const int32_t workers = (static_cast<int64_t>(n_imfs) * n >= (int64_t(1) << 15)) ? threads::team_size() : 1;
            threads::for_static(workers, n_imfs, [&](int32_t first, int32_t last, int)
            {
                for (int32_t k = first; k < last; ++k)
                    ok[k] = instantaneous(imf[k], n, sample_rate, &amp[static_cast<size_t>(k) * n],
                                          &freq[static_cast<size_t>(k) * n], nullptr);
            });
            for (char v : ok)
                if (!v)
                    return false;

            // into the bins, IMF by IMF
            const double per_hz = bins / (fmax - opt.fmin);
            for (int32_t k = 0; k < n_imfs; ++k)
            {
                const double *a = &amp[static_cast<size_t>(k) * n];
                const double *f = &freq[static_cast<size_t>(k) * n];
                for (int32_t t = 0; t < n; ++t)
                {
                    if (!(f[t] >= opt.fmin && f[t] <= fmax))
                        continue;
                    int32_t b = static_cast<int32_t>((f[t] - opt.fmin) * per_hz);
                    if (b >= bins)
                        b = bins - 1; // f = fmax
                    const double v = opt.amplitude ? a[t] : a[t] * a[t];
                    if (spectrum)
                        spectrum[static_cast<size_t>(b) * cols + t / step] += v;
                    if (marginal)
                        marginal[b] += v;
                }
            }
            return true;
        }

        /** hilbert_spectrum() of IMFs stored one after another (n_imfs x n, row-major). */
        inline bool hilbert_spectrum(const double *imfs, int32_t n_imfs, int32_t n, double sample_rate,
                                     const SpectrumOptions &opt, double *spectrum, double *marginal)
        {
            std::vector<const double *> rows(static_cast<size_t>(std::max(n_imfs, 0)));
            for (int32_t k = 0; k < n_imfs; ++k)
                rows[k] = imfs + static_cast<size_t>(k) * n;
            return hilbert_spectrum(rows.data(), n_imfs, n, sample_rate, opt, spectrum, marginal);
        }

        /** hilbert_spectrum() of IMFs as decompose() returns them; resizes the outputs. */
        inline bool hilbert_spectrum(const std::vector<std::vector<double>> &imfs, double sample_rate,
                                     const SpectrumOptions &opt, std::vector<double> &spectrum,
                                     std::vector<double> &marginal)
        {
            if (imfs.empty())
                return false;
            const int32_t n = static_cast<int32_t>(imfs[0].size());
            std::vector<const double *> rows;
            for (const auto &m : imfs)
            {
                if (static_cast<int32_t>(m.size()) != n)
                    return false;
                rows.push_back(m.data());
            }
            spectrum.assign(static_cast<size_t>(std::max(opt.frequency_bins, 0)) * spectrum_columns(n, opt), 0.0);
            marginal.assign(static_cast<size_t>(std::max(opt.frequency_bins, 0)), 0.0);
            return hilbert_spectrum(rows.data(), static_cast<int32_t>(rows.size()), n, sample_rate, opt,
                                    spectrum.data(), marginal.data());
        }
    } // namespace hht
} // namespace eemd

#endif // EEMD_HHT_HPP

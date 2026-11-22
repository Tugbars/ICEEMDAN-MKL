/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/**
 * @file iceemdan.h
 * @brief The VectorEMD public API: eight functions.
 *
 *   iceemdan_create         a decomposer from a configuration
 *   iceemdan_decompose      decompose one signal, or many signals of one length
 *   iceemdan_result         copy the IMFs and the residues out
 *   iceemdan_destroy        free a decomposer
 *   iceemdan_info           a decomposer's facts, or the library's
 *   iceemdan_error          why the last call on this thread was refused
 *   iceemdan_instantaneous  the amplitude, frequency and phase of an IMF over time
 *   iceemdan_spectrum       the Hilbert spectrum of IMFs
 *
 * A zeroed configuration is ICEEMDAN as published (Colominas et al., 2014),
 * computed as the authors' code computes it. A call is served or refused: a
 * refused call computes nothing, returns NULL or -1, and iceemdan_error()
 * says why.
 *
 * @code
 * #include "iceemdan.h"
 *
 * iceemdan_t *d = iceemdan_create(NULL);              // the published algorithm
 * int k = iceemdan_decompose(d, x, 1, n);             // one signal of n samples: k IMFs
 * double *imfs = malloc(sizeof(double) * k * n);      // IMF j: imfs[j * n + i]
 * double *residue = malloc(sizeof(double) * n);
 * iceemdan_result(d, imfs, residue, NULL);            // the IMFs and the residue add up to x
 * iceemdan_destroy(d);
 * @endcode
 *
 * The library takes care of the rest: the channels of a recording go through
 * the same call (count signals) and are spread over the threads together; the
 * noise a decomposer decomposed for one signal length is kept for the next
 * signal of that length, which then takes half the time or less with the same
 * result; NaN and Inf samples are filled in; the threads stop when the last
 * decomposer is destroyed.
 *
 * The library runs on x86-64 processors with AVX2 and FMA (Intel since 2013,
 * AMD since 2015). A decomposition comes out the same, bit for bit, on every
 * one of them and at any number of threads. The per-mode audit trail is in
 * iceemdan_diagnostics.h.
 */
#ifndef ICEEMDAN_H
#define ICEEMDAN_H

#include <stddef.h>

/** @brief "MAJOR.MINOR.PATCH" of this header; iceemdan_info() gives the loaded library's. */
#define ICEEMDAN_VERSION_STRING "1.1.0"

/* ICEEMDAN_API marks the functions of the public headers: the shared library exports them and nothing else. */
#ifndef ICEEMDAN_API
#if defined(ICEEMDAN_BUILD) && defined(_WIN32)
#define ICEEMDAN_API __declspec(dllexport)
#elif defined(ICEEMDAN_BUILD)
#define ICEEMDAN_API __attribute__((visibility("default")))
#elif defined(_WIN32)
#define ICEEMDAN_API __declspec(dllimport)
#else
#define ICEEMDAN_API
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /* ── the configuration ──────────────────────────────────────────────── */

  /** @brief iceemdan_config_t.flags. */
  enum
  {
    ICEEMDAN_FLAG_FAST = 1u << 0 /**< 8 to 12 times faster, no longer the published algorithm: a
                                      simpler rule for when sifting stops, one envelope mean for
                                      each noisy copy, and the noise of all copies cut from one
                                      long decomposed noise signal */
  };

  /**
   * @brief What a decomposer computes.
   *
   * A zeroed struct is ICEEMDAN as published: 100 noisy copies, a noise
   * amplitude of 0.2, as many IMFs as the signal has, on every processor of
   * this machine.
   */
  typedef struct
  {
    int ensemble_size; /**< the noisy copies averaged for each mode; 0 = 100 */
    double noise;      /**< the noise amplitude, the paper's epsilon, relative to the
                            standard deviation of the signal; 0 = 0.2 */
    int max_imfs;      /**< the most IMFs to extract; 0 = as many as the signal has */
    unsigned seed;     /**< the seed of the noise; 0 = 42. The same seed gives the same
                            result */
    int nthreads;      /**< the threads a decomposition runs on, the caller's included;
                            0 = every logical processor of this machine
                            (iceemdan_info().processors). The result does not depend on it */
    unsigned flags;    /**< ICEEMDAN_FLAG_* */
  } iceemdan_config_t;

  /* ── create, decompose, result, destroy ─────────────────────────────── */

  typedef struct iceemdan_s iceemdan_t; /**< a decomposer: opaque */

  /**
   * @brief Make a decomposer.
   *
   * @param config What it computes; NULL is a zeroed configuration.
   * @return The decomposer, or NULL: iceemdan_error() says why (a value out
   *         of range, no memory, or a processor without AVX2 and FMA).
   */
  ICEEMDAN_API iceemdan_t *iceemdan_create(const iceemdan_config_t *config);

  /**
   * @brief Decompose one signal, or many signals of one length (the
   *        channels of a recording, its epochs).
   *
   * Each signal is decomposed into IMFs, from the fastest oscillation to the
   * slowest, and a residue; they add up to the signal. Every signal of a
   * batch gets the result it would get alone, bit for bit. The result stays
   * in the decomposer until its next decomposition: iceemdan_result() copies
   * it out.
   *
   * The first decomposition of a signal length also decomposes the noise of
   * the ensemble, which the decomposer keeps: further calls with that length
   * skip the step. A call with another length replaces the kept noise. The
   * kept noise takes about 14 kB for every sample of a signal at the default
   * ensemble; a length whose noise would take more than half of this
   * machine's memory is refused (the library does not change the algorithm
   * on its own to make it fit).
   *
   * NaN and Inf samples are filled in from their neighbours by a straight
   * line before the decomposition; iceemdan_info() counts them.
   *
   * @param d       The decomposer. One decomposition at a time; several
   *                decomposers may run at once on several threads.
   * @param signals count x n values: signal s at signals[s * n].
   * @param count   The number of signals, 1 or more.
   * @param n       The samples of each signal, 4 or more.
   * @return The IMF rows of the result: the most IMFs of any signal (0 or
   *         more). -1 when the call was refused: iceemdan_error() says why.
   */
  ICEEMDAN_API int iceemdan_decompose(iceemdan_t *d, const double *signals, int count, int n);

  /**
   * @brief Copy the last decomposition out.
   *
   * @param d          The decomposer.
   * @param imfs       count x rows x n values (rows as iceemdan_decompose()
   *                   returned it): IMF j of signal s at imfs[(s * rows + j) * n].
   *                   A signal with fewer IMFs than rows gets zeros after its
   *                   own, so the rows and the residue of every signal add
   *                   up to it. May be NULL.
   * @param residues   count x n values: the residue of signal s at residues[s * n].
   *                   May be NULL.
   * @param imf_counts count values: the IMFs of each signal. May be NULL.
   * @return 0, or -1 when the decomposer holds no decomposition.
   */
  ICEEMDAN_API int iceemdan_result(const iceemdan_t *d, double *imfs, double *residues, int *imf_counts);

  /** @brief Free a decomposer and everything it holds. NULL is accepted. */
  ICEEMDAN_API void iceemdan_destroy(iceemdan_t *d);

  /* ── facts ──────────────────────────────────────────────────────────── */

  /** @brief What iceemdan_info() reports. */
  typedef struct
  {
    const char *version; /**< "MAJOR.MINOR.PATCH" of the loaded library */
    int cpu_supported;   /**< 1 when this processor runs the library (AVX2 and FMA) */
    int processors;      /**< this machine's logical processors: the threads a decomposer
                              runs on when its configuration names none */
    int nthreads;        /**< the threads the decomposer runs on */
    int signals;         /**< the last decomposition: its signals; 0 before the first */
    int n;               /**< ... the samples of each */
    int imfs;            /**< ... its IMF rows: the most IMFs of any signal */
    int noise_kept;      /**< ... 1 when it used the noise kept from an earlier call */
    int repaired;        /**< ... the NaN and Inf samples filled in, over all signals */
  } iceemdan_info_t;

  /**
   * @brief Fill a decomposer's facts; with d == NULL, the library's facts
   *        only (version, cpu_supported, processors).
   * @return 0, or -1 when out is NULL.
   */
  ICEEMDAN_API int iceemdan_info(const iceemdan_t *d, iceemdan_info_t *out);

  /**
   * @brief Why the last call on this thread was refused: read it when a call
   *        returns NULL or -1. Static storage; "" when the last call was
   *        served.
   */
  ICEEMDAN_API const char *iceemdan_error(void);

  /* ── Hilbert-Huang analysis ─────────────────────────────────────────── */

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
   * @param amplitude   n values out. May be NULL.
   * @param frequency   n values out, in Hz. May be NULL.
   * @param phase       n values out, in radians, unwrapped. May be NULL.
   * @return 0, or -1 when the call was refused.
   */
  ICEEMDAN_API int iceemdan_instantaneous(const double *imf, int n, double sample_rate,
                                          double *amplitude, double *frequency, double *phase);

  /**
   * @brief The Hilbert spectrum to compute. A zeroed struct is 128 frequency
   *        bands from 0 to half the sample rate, a column for every sample,
   *        in energy.
   */
  typedef struct
  {
    int frequency_bins; /**< the frequency bands, of equal width; 0 = 128 */
    double fmin;        /**< the lower edge of the first band, Hz */
    double fmax;        /**< the upper edge of the last band, Hz; 0 = sample_rate / 2 */
    int time_step;      /**< the samples summed into each time column; 0 or 1 = a column
                             for every sample */
    int amplitude;      /**< 0: each IMF adds its energy (amplitude squared); 1: its amplitude */
  } iceemdan_spectrum_t;

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
   * @param options     The spectrum; NULL is a zeroed struct.
   * @param spectrum    frequency_bins x columns values out, with
   *                    columns = (n + time_step - 1) / time_step: band b, from
   *                    fmin + b w to fmin + (b + 1) w with
   *                    w = (fmax - fmin) / frequency_bins, at spectrum[b * columns].
   *                    May be NULL.
   * @param marginal    frequency_bins values out: each band summed over time.
   *                    May be NULL.
   * @return 0, or -1 when the call was refused.
   */
  ICEEMDAN_API int iceemdan_spectrum(const double *imfs, int n_imfs, int n, double sample_rate,
                                     const iceemdan_spectrum_t *options, double *spectrum, double *marginal);

#ifdef __cplusplus
}
#endif
#endif /* ICEEMDAN_H */

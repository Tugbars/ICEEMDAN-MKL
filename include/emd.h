/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/**
 * @file emd.h
 * @brief EMD and EEMD in the VectorEMD library: six functions.
 *
 *   emd_create     a decomposer from a configuration
 *   emd_decompose  decompose one signal, or many signals of one length
 *   emd_result     copy the IMFs and the residues out
 *   emd_destroy    free a decomposer
 *   emd_info       a decomposer's facts, or the library's
 *   emd_error      why the last call on this thread was refused
 *
 * A zeroed configuration is plain EMD (Huang et al., 1998) by the rules of
 * emd.m of Rilling, Flandrin and Goncalves (2003): its stop rule for sifting,
 * cubic spline envelopes, its boundary extension. With an ensemble size it is
 * EEMD (Wu & Huang, 2009): the mean of the EMDs of the signal with white
 * noise added. A call is served or refused: a refused call computes nothing,
 * returns NULL or -1, and emd_error() says why.
 *
 * @code
 * #include "emd.h"
 *
 * emd_t *d = emd_create(NULL);                    // plain EMD
 * int k = emd_decompose(d, x, 1, n);              // one signal of n samples: k IMFs
 * double *imfs = malloc(sizeof(double) * k * n);  // IMF j: imfs[j * n + i]
 * double *residue = malloc(sizeof(double) * n);
 * emd_result(d, imfs, residue, NULL);             // the IMFs and the residue add up to x
 * emd_destroy(d);
 * @endcode
 *
 * The functions are in the same library as ICEEMDAN's (iceemdan.h), whose
 * Hilbert-Huang analysis (iceemdan_instantaneous(), iceemdan_spectrum())
 * takes the IMFs of any of the three decompositions. The library runs on
 * x86-64 processors with AVX2 and FMA (Intel since 2013, AMD since 2015). A
 * decomposition comes out the same, bit for bit, on every one of them and at
 * any number of threads.
 */
#ifndef EMD_H
#define EMD_H

#include <stddef.h>

/* EMD_API marks the functions of this header: the shared library exports them. */
#ifndef EMD_API
#if defined(ICEEMDAN_BUILD) && defined(_WIN32)
#define EMD_API __declspec(dllexport)
#elif defined(ICEEMDAN_BUILD)
#define EMD_API __attribute__((visibility("default")))
#elif defined(_WIN32)
#define EMD_API __declspec(dllimport)
#else
#define EMD_API
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /* ── the configuration ──────────────────────────────────────────────── */

  /**
   * @brief What a decomposer computes.
   *
   * A zeroed struct is plain EMD: the signal itself, no noise, as many IMFs
   * as it has, on every processor of this machine. With ensemble_size it is
   * EEMD.
   */
  typedef struct
  {
    int ensemble_size; /**< 0 = EMD. 1 or more = EEMD: that many copies of the signal,
                            each with its own white noise, decomposed and averaged */
    double noise;      /**< EEMD: the standard deviation of the noise, relative to the
                            signal's; 0 = 0.2. Not used by EMD */
    int max_imfs;      /**< the most IMFs to extract; 0 = as many as the signal has */
    unsigned seed;     /**< EEMD: the seed of the noise; 0 = 42. The same seed gives the
                            same result. Not used by EMD */
    int nthreads;      /**< the threads a decomposition runs on, the caller's included;
                            0 = every logical processor of this machine
                            (emd_info().processors). The result does not depend on it */
  } emd_config_t;

  /* ── create, decompose, result, destroy ─────────────────────────────── */

  typedef struct emd_s emd_t; /**< a decomposer: opaque */

  /**
   * @brief Make a decomposer.
   *
   * @param config What it computes; NULL is a zeroed configuration (EMD).
   * @return The decomposer, or NULL: emd_error() says why (a value out of
   *         range, no memory, or a processor without AVX2 and FMA).
   */
  EMD_API emd_t *emd_create(const emd_config_t *config);

  /**
   * @brief Decompose one signal, or many signals of one length (the
   *        channels of a recording, its epochs).
   *
   * Each signal is decomposed into IMFs, from the fastest oscillation to the
   * slowest, and a residue. EMD: they add up to the signal. EEMD: to the
   * signal plus the mean of the noise that was added, which shrinks with the
   * ensemble size (ICEEMDAN adds up exactly). Every signal of a batch gets
   * the result it would get alone, bit for bit. The result stays in the
   * decomposer until its next decomposition: emd_result() copies it out.
   *
   * NaN and Inf samples are filled in from their neighbours by a straight
   * line before the decomposition; emd_info() counts them.
   *
   * @param d       The decomposer. One decomposition at a time; several
   *                decomposers may run at once on several threads.
   * @param signals count x n values: signal s at signals[s * n].
   * @param count   The number of signals, 1 or more.
   * @param n       The samples of each signal, 4 or more.
   * @return The IMF rows of the result: the most IMFs of any signal (0 or
   *         more). -1 when the call was refused: emd_error() says why.
   */
  EMD_API int emd_decompose(emd_t *d, const double *signals, int count, int n);

  /**
   * @brief Copy the last decomposition out.
   *
   * @param d          The decomposer.
   * @param imfs       count x rows x n values (rows as emd_decompose()
   *                   returned it): IMF j of signal s at imfs[(s * rows + j) * n].
   *                   A signal with fewer IMFs than rows gets zeros after its
   *                   own. May be NULL.
   * @param residues   count x n values: the residue of signal s at residues[s * n].
   *                   May be NULL.
   * @param imf_counts count values: the IMFs of each signal. May be NULL.
   * @return 0, or -1 when the decomposer holds no decomposition.
   */
  EMD_API int emd_result(const emd_t *d, double *imfs, double *residues, int *imf_counts);

  /** @brief Free a decomposer and everything it holds. NULL is accepted. */
  EMD_API void emd_destroy(emd_t *d);

  /* ── facts ──────────────────────────────────────────────────────────── */

  /** @brief What emd_info() reports. */
  typedef struct
  {
    const char *version; /**< "MAJOR.MINOR.PATCH" of the loaded library */
    int cpu_supported;   /**< 1 when this processor runs the library (AVX2 and FMA) */
    int processors;      /**< this machine's logical processors: the threads a decomposer
                              runs on when its configuration names none */
    int nthreads;        /**< the threads the decomposer runs on */
    int ensemble_size;   /**< the decomposer's ensemble: 0 = EMD, 1 or more = EEMD */
    int signals;         /**< the last decomposition: its signals; 0 before the first */
    int n;               /**< ... the samples of each */
    int imfs;            /**< ... its IMF rows: the most IMFs of any signal */
    int repaired;        /**< ... the NaN and Inf samples filled in, over all signals */
  } emd_info_t;

  /**
   * @brief Fill a decomposer's facts; with d == NULL, the library's facts
   *        only (version, cpu_supported, processors).
   * @return 0, or -1 when out is NULL.
   */
  EMD_API int emd_info(const emd_t *d, emd_info_t *out);

  /**
   * @brief Why the last call on this thread was refused: read it when a call
   *        returns NULL or -1. Static storage; "" when the last call was
   *        served.
   */
  EMD_API const char *emd_error(void);

#ifdef __cplusplus
}
#endif
#endif /* EMD_H */

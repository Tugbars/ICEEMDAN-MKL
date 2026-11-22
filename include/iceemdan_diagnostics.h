/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/**
 * @file iceemdan_diagnostics.h
 * @brief The audit trail of a decomposition: two functions.
 *
 *   iceemdan_signal_stats   one signal: its IMFs, their orthogonality, the samples repaired
 *   iceemdan_imf_stats      one IMF of a signal: how the sifting of its noisy copies went
 *
 * Extra facts for logs, reports and tests; not needed to decompose a signal.
 * Both read the last iceemdan_decompose() of a decomposer.
 */
#ifndef ICEEMDAN_DIAGNOSTICS_H
#define ICEEMDAN_DIAGNOSTICS_H

#include "iceemdan.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /** @brief What iceemdan_signal_stats() reports. */
  typedef struct
  {
    int imfs;             /**< the IMFs of this signal */
    double orthogonality; /**< the orthogonality index of its IMFs: the sum over pairs of
                               |<IMF_i, IMF_j>| divided by the sum of <IMF_i, IMF_i>.
                               Lower is better */
    int repaired;         /**< the NaN and Inf samples filled in before the decomposition */
  } iceemdan_signal_stats_t;

  /**
   * @brief The facts of one signal of the last decomposition.
   * @param d      The decomposer.
   * @param signal 0 to count - 1.
   * @param out    Filled.
   * @return 0, or -1 when there is no such signal or out is NULL.
   */
  ICEEMDAN_API int iceemdan_signal_stats(const iceemdan_t *d, int signal, iceemdan_signal_stats_t *out);

  /** @brief What iceemdan_imf_stats() reports. */
  typedef struct
  {
    int copies;             /**< the noisy copies averaged for this IMF */
    int iterations_max;     /**< the most sifting iterations any copy needed */
    double iterations_mean; /**< their mean over the copies */
    int unconverged;        /**< the copies that reached the iteration limit (2000) before
                                 the stop rule was met; 0 is the normal case */
  } iceemdan_imf_stats_t;

  /**
   * @brief How the sifting went for one IMF of one signal of the last
   *        decomposition.
   * @param d      The decomposer.
   * @param signal 0 to count - 1.
   * @param imf    0 to that signal's IMFs - 1.
   * @param out    Filled.
   * @return 0, or -1 when there is no such IMF or out is NULL.
   */
  ICEEMDAN_API int iceemdan_imf_stats(const iceemdan_t *d, int signal, int imf, iceemdan_imf_stats_t *out);

#ifdef __cplusplus
}
#endif
#endif /* ICEEMDAN_DIAGNOSTICS_H */

/**
 * @file iceemdan.h
 * @brief The C interface of VectorEMD: ICEEMDAN decomposition and
 *        Hilbert-Huang analysis.
 *
 * ICEEMDAN (Colominas et al., 2014) as the authors' code computes it. One
 * header and one library (iceemdan.dll with iceemdan.lib on Windows,
 * libiceemdan.so on Linux), callable from C, C++ and any language that can
 * call C: Python, MATLAB (loadlibrary), R, C#, LabVIEW.
 *
 * @par What you can call
 * | Group                      | Functions |
 * |----------------------------|-----------|
 * | @ref iceemdan_decomposer   | iceemdan_create(), iceemdan_destroy() |
 * | @ref iceemdan_settings     | iceemdan_set_...(), all optional |
 * | @ref iceemdan_single       | iceemdan_decompose() |
 * | @ref iceemdan_result       | iceemdan_get_num_imfs(), iceemdan_get_all_imfs(), iceemdan_get_imf(), iceemdan_get_residue() |
 * | @ref iceemdan_batch        | iceemdan_decompose_batch(), iceemdan_batch_get(), ... |
 * | @ref iceemdan_diagnostics  | iceemdan_get_error(), iceemdan_get_orthogonality_index(), iceemdan_get_imf_...() |
 * | @ref iceemdan_hht          | iceemdan_instantaneous(), iceemdan_mean_frequency(), iceemdan_hilbert_spectrum() |
 * | @ref iceemdan_library      | iceemdan_set_num_threads(), iceemdan_version(), ... |
 *
 * @par Example
 * @code
 * #include "iceemdan.h"
 *
 * ICEEMDANHandle *d = iceemdan_create(ICEEMDAN_STANDARD);
 * int k = iceemdan_decompose(d, x, n);            // the number of IMFs
 * if (k == 0)
 *     printf("failed: %s\n", iceemdan_get_error(d));
 * double *imfs = malloc(sizeof(double) * k * n);  // IMF j: imfs[j * n + i]
 * double *residue = malloc(sizeof(double) * n);
 * iceemdan_get_all_imfs(d, imfs);
 * iceemdan_get_residue(d, residue);                // the IMFs and the residue add up to x
 * iceemdan_destroy(d);
 * @endcode
 *
 * @par Conventions
 * - Signals and results are arrays of double. Several of them are stored one
 *   after another (row-major): row j of an array of rows of n values starts
 *   at j * n.
 * - The caller allocates every output; the library never returns memory the
 *   caller has to free.
 * - A decomposer keeps its last result until the next call; the getters copy
 *   it out.
 * - Functions that do something return 1 when done and 0 when not;
 *   iceemdan_get_error() then says why.
 * - IMF 0 is the one of highest frequency.
 * - Results do not depend on the number of threads: for a given seed, the
 *   same numbers, bit for bit.
 * - Use a decomposer from one thread at a time. Several decomposers may run on
 *   several threads at once; they share the library's worker threads.
 *
 * @copyright MIT License
 */

#ifndef ICEEMDAN_H
#define ICEEMDAN_H

#include <stddef.h> /* size_t */

/** @brief Major version of this header. */
#define ICEEMDAN_VERSION_MAJOR 1
/** @brief Minor version of this header. */
#define ICEEMDAN_VERSION_MINOR 1
/** @brief Patch version of this header. */
#define ICEEMDAN_VERSION_PATCH 0
/** @brief "major.minor.patch"; the loaded library reports its own with iceemdan_version(). */
#define ICEEMDAN_VERSION_STRING "1.1.0"

/**
 * @brief Marks the library's functions: exported when the library is built
 *        (ICEEMDAN_BUILD defined), imported by its users.
 */
#if defined(_WIN32)
#if defined(ICEEMDAN_BUILD)
#define ICEEMDAN_API __declspec(dllexport)
#else
#define ICEEMDAN_API __declspec(dllimport)
#endif
#else
#define ICEEMDAN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A decomposer: its settings, its decomposed noise and its last result.
 *
 * Opaque: made by iceemdan_create(), freed by iceemdan_destroy().
 */
typedef struct ICEEMDANHandle ICEEMDANHandle;

/**
 * @defgroup iceemdan_values Values of the settings
 * @brief The values the settings functions take.
 * @{
 */

/** @brief Presets, for iceemdan_create(). */
typedef enum iceemdan_preset
{
    ICEEMDAN_STANDARD = 0,  /**< The published algorithm, computed as the authors' code does (default). */
    ICEEMDAN_FINANCE = 1,   /**< Akima envelopes, noise scaled to the local volatility, causal (AR) ends,
                                 +/- noise pairs, NaN and Inf removed from the input. */
    ICEEMDAN_SCIENTIFIC = 2 /**< As Standard, and NaN and Inf removed from the input. */
} iceemdan_preset;

/** @brief Envelope splines, for iceemdan_set_spline_method(). */
typedef enum iceemdan_spline
{
    ICEEMDAN_SPLINE_CUBIC = 0, /**< Not-a-knot cubic spline, as MATLAB's spline (default). */
    ICEEMDAN_SPLINE_AKIMA = 1, /**< Akima's spline: local, does not overshoot a step (Finance). */
    ICEEMDAN_SPLINE_LINEAR = 2 /**< Straight lines between the extrema. */
} iceemdan_spline;

/** @brief How the envelopes are continued past the ends, for iceemdan_set_boundary_method(). */
typedef enum iceemdan_boundary
{
    ICEEMDAN_BOUNDARY_MIRROR = 0,   /**< Extrema mirrored at the ends. */
    ICEEMDAN_BOUNDARY_AR = 1,       /**< AR(1) extrapolation: causal, no look-ahead (Finance). */
    ICEEMDAN_BOUNDARY_LINEAR = 2,   /**< Linear extrapolation. */
    ICEEMDAN_BOUNDARY_SYMMETRIC = 3 /**< Symmetric extension of the authors' code (default). */
} iceemdan_boundary;

/** @brief Noise amplitude over time, for iceemdan_set_volatility_method(). */
typedef enum iceemdan_volatility
{
    ICEEMDAN_VOLATILITY_GLOBAL = 0, /**< One amplitude for the whole signal (default). */
    ICEEMDAN_VOLATILITY_SMA = 1,    /**< Follows a moving standard deviation. */
    ICEEMDAN_VOLATILITY_EMA = 2     /**< Follows an exponentially weighted one (Finance). */
} iceemdan_volatility;

/** @brief The local mean of each noisy copy, for iceemdan_set_local_mean_method(). */
typedef enum iceemdan_local_mean
{
    ICEEMDAN_LOCAL_MEAN_SIFTED = 0,         /**< Its first mode sifted out: the published algorithm (default). */
    ICEEMDAN_LOCAL_MEAN_SINGLE_ENVELOPE = 1 /**< One envelope mean: faster, not the published algorithm. */
} iceemdan_local_mean;

/** @brief When sifting a mode stops, for iceemdan_set_stop_rule(). */
typedef enum iceemdan_stop_rule
{
    ICEEMDAN_STOP_SD = 0,     /**< The SD criterion (iceemdan_set_sift_threshold(), iceemdan_set_s_number()): faster. */
    ICEEMDAN_STOP_RILLING = 1 /**< Rilling, Flandrin and Goncalves (2003), as the authors' code (default). */
} iceemdan_stop_rule;

/** @} */

/**
 * @defgroup iceemdan_decomposer A decomposer
 * @brief Making and freeing one.
 * @{
 */

/**
 * @brief Makes a decomposer with the settings of a preset.
 * @param preset One of #iceemdan_preset (#ICEEMDAN_STANDARD for the published algorithm).
 * @return The decomposer, or NULL when there is no memory for it.
 */
ICEEMDAN_API ICEEMDANHandle *iceemdan_create(int preset);

/**
 * @brief Frees a decomposer and everything it holds.
 * @param h The decomposer; NULL is allowed.
 */
ICEEMDAN_API void iceemdan_destroy(ICEEMDANHandle *h);

/** @} */

/**
 * @defgroup iceemdan_settings Settings
 * @brief All optional; each applies from the next decomposition on.
 * @{
 */

/**
 * @brief Sets how many noisy copies are averaged per mode.
 * @param h    The decomposer.
 * @param size Ensemble size (default 100).
 */
ICEEMDAN_API void iceemdan_set_ensemble_size(ICEEMDANHandle *h, int size);

/**
 * @brief Sets the noise amplitude (the paper's epsilon).
 * @param h   The decomposer.
 * @param std Relative to the standard deviation of the signal and of each residue (default 0.2).
 */
ICEEMDAN_API void iceemdan_set_noise_std(ICEEMDANHandle *h, double std);

/**
 * @brief Sets the most IMFs to extract.
 * @param h        The decomposer.
 * @param max_imfs Limit (default 32; in practice the decomposition stops before).
 */
ICEEMDAN_API void iceemdan_set_max_imfs(ICEEMDANHandle *h, int max_imfs);

/**
 * @brief Sets the seed of the noise. The same seed gives the same result.
 * @param h    The decomposer.
 * @param seed Seed (default 42).
 */
ICEEMDAN_API void iceemdan_set_rng_seed(ICEEMDANHandle *h, unsigned int seed);

/**
 * @brief Sets the envelope spline.
 * @param h      The decomposer.
 * @param method One of #iceemdan_spline (default #ICEEMDAN_SPLINE_CUBIC; Finance: Akima).
 */
ICEEMDAN_API void iceemdan_set_spline_method(ICEEMDANHandle *h, int method);

/**
 * @brief Sets how the envelopes are continued past the ends.
 * @param h      The decomposer.
 * @param method One of #iceemdan_boundary (default #ICEEMDAN_BOUNDARY_SYMMETRIC; Finance: AR).
 */
ICEEMDAN_API void iceemdan_set_boundary_method(ICEEMDANHandle *h, int method);

/**
 * @brief Sets how the noise amplitude follows the signal over time.
 * @param h      The decomposer.
 * @param method One of #iceemdan_volatility (default #ICEEMDAN_VOLATILITY_GLOBAL; Finance: EMA).
 */
ICEEMDAN_API void iceemdan_set_volatility_method(ICEEMDANHandle *h, int method);

/**
 * @brief Sets the local mean of each noisy copy.
 * @param h      The decomposer.
 * @param method One of #iceemdan_local_mean (default #ICEEMDAN_LOCAL_MEAN_SIFTED).
 */
ICEEMDAN_API void iceemdan_set_local_mean_method(ICEEMDANHandle *h, int method);

/**
 * @brief Sets when sifting a mode stops.
 * @param h    The decomposer.
 * @param rule One of #iceemdan_stop_rule (default #ICEEMDAN_STOP_RILLING).
 */
ICEEMDAN_API void iceemdan_set_stop_rule(ICEEMDANHandle *h, int rule);

/**
 * @brief Sets the most sifting iterations per mode.
 * @param h     The decomposer.
 * @param iters Limit (default 100).
 */
ICEEMDAN_API void iceemdan_set_max_sift_iters(ICEEMDANHandle *h, int iters);

/**
 * @brief Sets the threshold of the SD stop rule (#ICEEMDAN_STOP_SD only).
 * @param h         The decomposer.
 * @param threshold SD threshold (default 0.05).
 */
ICEEMDAN_API void iceemdan_set_sift_threshold(ICEEMDANHandle *h, double threshold);

/**
 * @brief Sets the S-number of the SD stop rule (#ICEEMDAN_STOP_SD only).
 * @param h The decomposer.
 * @param s Iterations with an unchanged extrema count that end sifting (default 6; 0 = off).
 */
ICEEMDAN_API void iceemdan_set_s_number(ICEEMDANHandle *h, int s);

/**
 * @brief Uses +/- noise pairs: half as many noise realizations to decompose.
 *        Not the published algorithm.
 * @param h  The decomposer.
 * @param on 1 to use them, 0 not to (default 0; Finance: 1).
 */
ICEEMDAN_API void iceemdan_set_antithetic(ICEEMDANHandle *h, int on);

/**
 * @brief Cuts the noise of all copies from one long decomposed noise signal,
 *        which is faster to build. Not the published algorithm.
 * @param h  The decomposer.
 * @param on 1 to do so, 0 not to (default 0).
 */
ICEEMDAN_API void iceemdan_set_circular_bank(ICEEMDANHandle *h, int on);

/**
 * @brief Sets the memory cap of the decomposed noise; above it the circular
 *        bank (iceemdan_set_circular_bank()) is used.
 * @param h     The decomposer.
 * @param bytes Cap in bytes (default 2 GiB).
 */
ICEEMDAN_API void iceemdan_set_noise_bank_max_bytes(ICEEMDANHandle *h, size_t bytes);

/**
 * @brief Keeps the decomposed noise between calls, or not.
 *
 * The decomposed noise depends on the signal length and the settings, not on
 * the signal. Kept, it lets every further signal of the same length skip
 * building it, with the same result as alone.
 *
 * @param h    The decomposer.
 * @param keep 1 to keep it (default), 0 to free it at the end of each call.
 */
ICEEMDAN_API void iceemdan_set_keep_noise_bank(ICEEMDANHandle *h, int keep);

/**
 * @brief Frees the decomposed noise now; the next call builds it again.
 * @param h The decomposer.
 */
ICEEMDAN_API void iceemdan_release_noise_bank(ICEEMDANHandle *h);

/** @} */

/**
 * @defgroup iceemdan_single One signal
 * @brief Decomposing a signal.
 * @{
 */

/**
 * @brief Decomposes a signal into IMFs and a residue.
 *
 * The result and its diagnostics stay in the decomposer until its next
 * decomposition; read them with the functions of @ref iceemdan_result and
 * @ref iceemdan_diagnostics.
 *
 * @param h      The decomposer.
 * @param signal The signal, n values.
 * @param n      Its length, at least 4.
 * @return The number of IMFs, or 0 on failure (see iceemdan_get_error()).
 */
ICEEMDAN_API int iceemdan_decompose(ICEEMDANHandle *h, const double *signal, int n);

/**
 * @brief The same as iceemdan_decompose(), which also records the diagnostics.
 * @param h      The decomposer.
 * @param signal The signal, n values.
 * @param n      Its length, at least 4.
 * @return The number of IMFs, or 0 on failure.
 */
ICEEMDAN_API int iceemdan_decompose_with_diagnostics(ICEEMDANHandle *h, const double *signal, int n);

/** @} */

/**
 * @defgroup iceemdan_result The result
 * @brief The IMFs and the residue of the last iceemdan_decompose().
 * @{
 */

/**
 * @brief The number of IMFs.
 * @param h The decomposer.
 * @return The number of IMFs, 0 when there is no result.
 */
ICEEMDAN_API int iceemdan_get_num_imfs(const ICEEMDANHandle *h);

/**
 * @brief The length of each IMF: that of the signal.
 * @param h The decomposer.
 * @return n, or 0 when there is no result.
 */
ICEEMDAN_API int iceemdan_get_imf_length(const ICEEMDANHandle *h);

/**
 * @brief Copies all IMFs out.
 * @param h   The decomposer.
 * @param out num_imfs x n values; IMF j goes to out[j * n].
 * @return 1, or 0 when there is no result.
 */
ICEEMDAN_API int iceemdan_get_all_imfs(const ICEEMDANHandle *h, double *out);

/**
 * @brief Copies one IMF out.
 * @param h         The decomposer.
 * @param imf_index 0 for the IMF of highest frequency, up to num_imfs - 1.
 * @param out       n values.
 * @return 1, or 0 when there is no such IMF.
 */
ICEEMDAN_API int iceemdan_get_imf(const ICEEMDANHandle *h, int imf_index, double *out);

/**
 * @brief Copies the residue out.
 * @param h   The decomposer.
 * @param out n values.
 * @return 1, or 0 when there is no result.
 */
ICEEMDAN_API int iceemdan_get_residue(const ICEEMDANHandle *h, double *out);

/** @} */

/**
 * @defgroup iceemdan_batch Many signals at once
 * @brief The channels of a recording, or its epochs, in one call.
 *
 * count signals of n samples, stored one after another (count x n). Each
 * result is the same, bit for bit, as iceemdan_decompose() gives for that
 * signal alone, and the batch is faster: the signals share one decomposed
 * noise, and their noisy copies are spread over the threads together.
 * @{
 */

/**
 * @brief Decomposes a batch of signals of one length.
 *
 * The results stay in the decomposer until its next batch; the functions of
 * @ref iceemdan_result keep referring to the last single signal.
 *
 * @param h       The decomposer.
 * @param signals count x n values: signal s at signals[s * n].
 * @param count   The number of signals, at least 1.
 * @param n       Their length, at least 4.
 * @return The largest number of IMFs of any signal, or 0 on failure (see iceemdan_get_error()).
 */
ICEEMDAN_API int iceemdan_decompose_batch(ICEEMDANHandle *h, const double *signals, int count, int n);

/**
 * @brief The number of signals in the last batch.
 * @param h The decomposer.
 * @return count, or 0 when there is no batch result.
 */
ICEEMDAN_API int iceemdan_batch_count(const ICEEMDANHandle *h);

/**
 * @brief The number of IMFs of one signal of the last batch.
 * @param h The decomposer.
 * @param s The signal, 0 to count - 1.
 * @return Its number of IMFs, or -1 when there is no signal s.
 */
ICEEMDAN_API int iceemdan_batch_num_imfs(const ICEEMDANHandle *h, int s);

/**
 * @brief Copies the last batch out.
 *
 * A signal with fewer IMFs than the most gets rows of zeros after its own, so
 * each signal's rows and residue still add up to it.
 *
 * @param h        The decomposer.
 * @param imfs     count x max_imfs x n values (max_imfs as iceemdan_decompose_batch()
 *                 returned it): IMF j of signal s at imfs[(s * max_imfs + j) * n]. May be NULL.
 * @param residues count x n values: the residue of signal s at residues[s * n]. May be NULL.
 * @return 1, or 0 when there is no batch result.
 */
ICEEMDAN_API int iceemdan_batch_get(const ICEEMDANHandle *h, double *imfs, double *residues);

/**
 * @brief The orthogonality index of one signal of the last batch
 *        (see iceemdan_get_orthogonality_index()).
 * @param h The decomposer.
 * @param s The signal, 0 to count - 1.
 * @return The index, or -1 when there is no signal s.
 */
ICEEMDAN_API double iceemdan_batch_orthogonality_index(const ICEEMDANHandle *h, int s);

/** @} */

/**
 * @defgroup iceemdan_diagnostics Diagnostics
 * @brief How the last iceemdan_decompose() went.
 * @{
 */

/**
 * @brief Whether the last decomposition succeeded.
 * @param h The decomposer.
 * @return 1 when it did, 0 otherwise.
 */
ICEEMDAN_API int iceemdan_get_valid(const ICEEMDANHandle *h);

/**
 * @brief Why the last call failed.
 * @param h The decomposer.
 * @return A message, empty when nothing failed. It belongs to the decomposer.
 */
ICEEMDAN_API const char *iceemdan_get_error(const ICEEMDANHandle *h);

/**
 * @brief The orthogonality index of the IMFs: the sum over pairs of
 *        |<IMF_i, IMF_j>| divided by the sum of <IMF_i, IMF_i>.
 * @param h The decomposer.
 * @return The index; lower is better.
 */
ICEEMDAN_API double iceemdan_get_orthogonality_index(const ICEEMDANHandle *h);

/**
 * @brief The NaN and Inf samples replaced in the input (Finance and Scientific presets).
 * @param h The decomposer.
 * @return Their number.
 */
ICEEMDAN_API int iceemdan_get_nan_count(const ICEEMDANHandle *h);

/**
 * @brief The seed the noise came from.
 * @param h The decomposer.
 * @return The seed.
 */
ICEEMDAN_API unsigned int iceemdan_get_rng_seed_used(const ICEEMDANHandle *h);

/**
 * @brief Whether the decomposed noise was kept from an earlier call.
 * @param h The decomposer.
 * @return 1 when it was, 0 when it was built for this call.
 */
ICEEMDAN_API int iceemdan_get_noise_bank_reused(const ICEEMDANHandle *h);

/**
 * @brief The most sifting iterations any noisy copy needed for one IMF.
 * @param h         The decomposer.
 * @param imf_index The IMF, 0 to num_imfs - 1.
 * @return The iterations, or -1 when there is no such IMF.
 */
ICEEMDAN_API int iceemdan_get_imf_sift_iterations(const ICEEMDANHandle *h, int imf_index);

/**
 * @brief The mean sifting iterations of the noisy copies for one IMF.
 * @param h         The decomposer.
 * @param imf_index The IMF, 0 to num_imfs - 1.
 * @return The mean, or -1 when there is no such IMF.
 */
ICEEMDAN_API double iceemdan_get_imf_mean_sift_iterations(const ICEEMDANHandle *h, int imf_index);

/**
 * @brief The noisy copies that hit the iteration limit for one IMF.
 * @param h         The decomposer.
 * @param imf_index The IMF, 0 to num_imfs - 1.
 * @return Their number, or -1 when there is no such IMF.
 */
ICEEMDAN_API int iceemdan_get_imf_nonconverged_trials(const ICEEMDANHandle *h, int imf_index);

/**
 * @brief The noisy copies averaged for one IMF.
 * @param h         The decomposer.
 * @param imf_index The IMF, 0 to num_imfs - 1.
 * @return Their number, or -1 when there is no such IMF.
 */
ICEEMDAN_API int iceemdan_get_imf_trials(const ICEEMDANHandle *h, int imf_index);

/**
 * @brief Whether no noisy copy hit the iteration limit for one IMF.
 * @param h         The decomposer.
 * @param imf_index The IMF, 0 to num_imfs - 1.
 * @return 1 when none did, 0 when some did, -1 when there is no such IMF.
 */
ICEEMDAN_API int iceemdan_get_imf_converged(const ICEEMDANHandle *h, int imf_index);

/** @} */

/**
 * @defgroup iceemdan_hht Hilbert-Huang analysis
 * @brief Instantaneous attributes and spectra of any IMFs; no decomposer needed.
 * @{
 */

/**
 * @brief Instantaneous amplitude, frequency and phase of one IMF, from its
 *        analytic signal.
 *
 * Within a period or two of each end of a non-periodic IMF the values are
 * distorted.
 *
 * @param imf         The IMF, n values.
 * @param n           Its length, at least 2.
 * @param sample_rate Samples per second.
 * @param amplitude   n values out. May be NULL.
 * @param frequency   n values out, in Hz. May be NULL.
 * @param phase       n values out, unwrapped, in radians. May be NULL.
 * @return 1, or 0 on bad arguments.
 */
ICEEMDAN_API int iceemdan_instantaneous(const double *imf, int n, double sample_rate,
                                        double *amplitude, double *frequency, double *phase);

/**
 * @brief The instantaneous frequency of one IMF averaged with its energy as
 *        the weight.
 * @param imf         The IMF, n values.
 * @param n           Its length.
 * @param sample_rate Samples per second.
 * @return The mean frequency in Hz.
 */
ICEEMDAN_API double iceemdan_mean_frequency(const double *imf, int n, double sample_rate);

/**
 * @brief The number of time columns of a Hilbert spectrum.
 * @param n         The length of the IMFs.
 * @param time_step Samples per column.
 * @return The number of columns.
 */
ICEEMDAN_API int iceemdan_spectrum_columns(int n, int time_step);

/**
 * @brief The Hilbert spectrum of IMFs: energy over time and frequency, and
 *        its marginal spectrum.
 *
 * At every sample each IMF adds its energy (amplitude squared, or its
 * amplitude) to the frequency band of its instantaneous frequency;
 * frequencies outside [fmin, fmax] are left out. Band b runs from
 * fmin + b w to fmin + (b + 1) w, with w = (fmax - fmin) / frequency_bins.
 *
 * @param imfs           n_imfs x n values: IMF j at imfs[j * n].
 * @param n_imfs         The number of IMFs, at least 1.
 * @param n              Their length.
 * @param sample_rate    Samples per second.
 * @param frequency_bins Frequency bands (rows).
 * @param fmin           Lower edge of the first band, Hz.
 * @param fmax           Upper edge of the last band, Hz; <= 0 means sample_rate / 2.
 * @param time_step      Samples per time column.
 * @param amplitude      0: add energy (amplitude squared); 1: add amplitude.
 * @param spectrum       frequency_bins x iceemdan_spectrum_columns(n, time_step) values out,
 *                       band b in row b. May be NULL.
 * @param marginal       frequency_bins values out: each band summed over time. May be NULL.
 * @return 1, or 0 on bad arguments.
 */
ICEEMDAN_API int iceemdan_hilbert_spectrum(const double *imfs, int n_imfs, int n, double sample_rate,
                                           int frequency_bins, double fmin, double fmax, int time_step,
                                           int amplitude, double *spectrum, double *marginal);

/** @} */

/**
 * @defgroup iceemdan_library The library
 * @brief Version, threads and helpers.
 * @{
 */

/**
 * @brief The version of the loaded library.
 * @return "major.minor.patch" (#ICEEMDAN_VERSION_STRING of the header it was built with).
 */
ICEEMDAN_API const char *iceemdan_version(void);

/**
 * @brief Sets the threads a decomposition uses, for the whole process.
 *        Results do not depend on it.
 * @param n Threads; n < 1 means every logical processor, the default unless
 *          the environment variable EEMD_NUM_THREADS sets it.
 */
ICEEMDAN_API void iceemdan_set_num_threads(int n);

/**
 * @brief The threads a decomposition uses.
 * @return Their number, the calling thread included.
 */
ICEEMDAN_API int iceemdan_get_num_threads(void);

/**
 * @brief Stops the worker threads; the next decomposition starts them again.
 *
 * Call it before unloading the library from a running process; it is not
 * needed at exit.
 */
ICEEMDAN_API void iceemdan_shutdown_threads(void);

/**
 * @brief The FFT that transforms length n.
 * @param n Transform length.
 * @return "VectorFFT", or "none" for n < 2 and when VectorFFT could not be loaded.
 */
ICEEMDAN_API const char *iceemdan_fft_backend(int n);

/**
 * @brief The energy of a signal: the sum of squares of its values.
 * @param signal n values.
 * @param n      Their number.
 * @return The sum of squares.
 */
ICEEMDAN_API double iceemdan_compute_energy(const double *signal, int n);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* ICEEMDAN_H */

/**
 * The C interface, include/iceemdan.h: what each function does is described
 * there. CMake builds it as the shared library "iceemdan" (target
 * iceemdan_shared), linked with VectorFFT, its only dependency. The Python
 * module (python/iceemdan.py) calls the same functions.
 */

#define ICEEMDAN_BUILD // export, not import
#include "iceemdan.h"

#include "iceemdan.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

// Every definition is extern "C" once more: one that does not match its
// declaration in iceemdan.h is then a compile error, not a second function.
#define ICEEMDAN_EXPORT extern "C" ICEEMDAN_API

// The setting values of iceemdan.h are those of the C++ enums
static_assert(ICEEMDAN_STANDARD == static_cast<int>(eemd::ProcessingMode::Standard) &&
                  ICEEMDAN_FINANCE == static_cast<int>(eemd::ProcessingMode::Finance) &&
                  ICEEMDAN_SCIENTIFIC == static_cast<int>(eemd::ProcessingMode::Scientific),
              "presets");
static_assert(ICEEMDAN_SPLINE_CUBIC == static_cast<int>(eemd::SplineMethod::Cubic) &&
                  ICEEMDAN_SPLINE_AKIMA == static_cast<int>(eemd::SplineMethod::Akima) &&
                  ICEEMDAN_SPLINE_LINEAR == static_cast<int>(eemd::SplineMethod::Linear),
              "splines");
static_assert(ICEEMDAN_BOUNDARY_MIRROR == static_cast<int>(eemd::BoundaryMethod::Mirror) &&
                  ICEEMDAN_BOUNDARY_AR == static_cast<int>(eemd::BoundaryMethod::AR) &&
                  ICEEMDAN_BOUNDARY_LINEAR == static_cast<int>(eemd::BoundaryMethod::Linear) &&
                  ICEEMDAN_BOUNDARY_SYMMETRIC == static_cast<int>(eemd::BoundaryMethod::Symmetric),
              "boundaries");
static_assert(ICEEMDAN_VOLATILITY_GLOBAL == static_cast<int>(eemd::VolatilityMethod::Global) &&
                  ICEEMDAN_VOLATILITY_SMA == static_cast<int>(eemd::VolatilityMethod::SMA) &&
                  ICEEMDAN_VOLATILITY_EMA == static_cast<int>(eemd::VolatilityMethod::EMA),
              "volatility");
static_assert(ICEEMDAN_LOCAL_MEAN_SIFTED == static_cast<int>(eemd::LocalMeanMethod::Sifted) &&
                  ICEEMDAN_LOCAL_MEAN_SINGLE_ENVELOPE == static_cast<int>(eemd::LocalMeanMethod::SingleEnvelope),
              "local mean");
static_assert(ICEEMDAN_STOP_SD == static_cast<int>(eemd::SiftStopRule::SD) &&
                  ICEEMDAN_STOP_RILLING == static_cast<int>(eemd::SiftStopRule::Rilling),
              "stop rules");

// ============================================================================
// Opaque handle for ICEEMDAN instance
// ============================================================================

struct ICEEMDANHandle
{
    eemd::ICEEMDAN *decomposer;
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    eemd::DecompositionDiagnostics diagnostics;
    // the last iceemdan_decompose_batch()
    std::vector<std::vector<std::vector<double>>> batch_imfs;
    std::vector<std::vector<double>> batch_residues;
    std::vector<eemd::DecompositionDiagnostics> batch_diagnostics;
    int batch_max_imfs = 0;
    char error_msg[256];
};

// ============================================================================
// Lifecycle
// ============================================================================

ICEEMDAN_EXPORT ICEEMDANHandle *iceemdan_create(int mode)
{
    try
    {
        auto *handle = new ICEEMDANHandle();
        eemd::ProcessingMode pm;
        switch (mode)
        {
        case ICEEMDAN_FINANCE:
            pm = eemd::ProcessingMode::Finance;
            break;
        case ICEEMDAN_SCIENTIFIC:
            pm = eemd::ProcessingMode::Scientific;
            break;
        default:
            pm = eemd::ProcessingMode::Standard;
            break;
        }
        handle->decomposer = new eemd::ICEEMDAN(pm);
        handle->error_msg[0] = '\0';
        return handle;
    }
    catch (...)
    {
        return nullptr;
    }
}

/**
 * Destroy ICEEMDAN instance
 */
ICEEMDAN_EXPORT void iceemdan_destroy(ICEEMDANHandle *handle)
{
    if (handle)
    {
        delete handle->decomposer;
        delete handle;
    }
}

// ============================================================================
// Configuration
// ============================================================================

ICEEMDAN_EXPORT void iceemdan_set_ensemble_size(ICEEMDANHandle *h, int size)
{
    if (h && h->decomposer)
        h->decomposer->config().ensemble_size = size;
}

ICEEMDAN_EXPORT void iceemdan_set_noise_std(ICEEMDANHandle *h, double std)
{
    if (h && h->decomposer)
        h->decomposer->config().noise_std = std;
}

ICEEMDAN_EXPORT void iceemdan_set_max_imfs(ICEEMDANHandle *h, int max_imfs)
{
    if (h && h->decomposer)
        h->decomposer->config().max_imfs = max_imfs;
}

ICEEMDAN_EXPORT void iceemdan_set_max_sift_iters(ICEEMDANHandle *h, int iters)
{
    if (h && h->decomposer)
        h->decomposer->config().max_sift_iters = iters;
}

ICEEMDAN_EXPORT void iceemdan_set_sift_threshold(ICEEMDANHandle *h, double thresh)
{
    if (h && h->decomposer)
        h->decomposer->config().sift_threshold = thresh;
}

ICEEMDAN_EXPORT void iceemdan_set_rng_seed(ICEEMDANHandle *h, unsigned int seed)
{
    if (h && h->decomposer)
        h->decomposer->config().rng_seed = seed;
}

ICEEMDAN_EXPORT void iceemdan_set_s_number(ICEEMDANHandle *h, int s)
{
    if (h && h->decomposer)
        h->decomposer->config().s_number = s;
}

ICEEMDAN_EXPORT void iceemdan_set_spline_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case ICEEMDAN_SPLINE_AKIMA:
            h->decomposer->config().spline_method = eemd::SplineMethod::Akima;
            break;
        case ICEEMDAN_SPLINE_LINEAR:
            h->decomposer->config().spline_method = eemd::SplineMethod::Linear;
            break;
        default:
            h->decomposer->config().spline_method = eemd::SplineMethod::Cubic;
            break;
        }
    }
}

ICEEMDAN_EXPORT void iceemdan_set_volatility_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case ICEEMDAN_VOLATILITY_SMA:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::SMA;
            break;
        case ICEEMDAN_VOLATILITY_EMA:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::EMA;
            break;
        default:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::Global;
            break;
        }
    }
}

ICEEMDAN_EXPORT void iceemdan_set_boundary_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case ICEEMDAN_BOUNDARY_AR:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::AR;
            break;
        case ICEEMDAN_BOUNDARY_LINEAR:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::Linear;
            break;
        case ICEEMDAN_BOUNDARY_SYMMETRIC:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::Symmetric;
            break;
        default:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::Mirror;
            break;
        }
    }
}

ICEEMDAN_EXPORT void iceemdan_set_local_mean_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
        h->decomposer->config().local_mean_method = (method == ICEEMDAN_LOCAL_MEAN_SINGLE_ENVELOPE)
                                                        ? eemd::LocalMeanMethod::SingleEnvelope
                                                        : eemd::LocalMeanMethod::Sifted;
}

ICEEMDAN_EXPORT void iceemdan_set_stop_rule(ICEEMDANHandle *h, int rule)
{
    if (h && h->decomposer)
        h->decomposer->config().sift_stop_rule =
            (rule == ICEEMDAN_STOP_SD) ? eemd::SiftStopRule::SD : eemd::SiftStopRule::Rilling;
}

ICEEMDAN_EXPORT void iceemdan_set_antithetic(ICEEMDANHandle *h, int on)
{
    if (h && h->decomposer)
        h->decomposer->config().use_antithetic = on != 0;
}

ICEEMDAN_EXPORT void iceemdan_set_circular_bank(ICEEMDANHandle *h, int on)
{
    if (h && h->decomposer)
        h->decomposer->config().use_circular_bank = on != 0;
}

ICEEMDAN_EXPORT void iceemdan_set_noise_bank_max_bytes(ICEEMDANHandle *h, size_t bytes)
{
    if (h && h->decomposer)
        h->decomposer->config().noise_bank_max_bytes = bytes;
}

/**
 * Keep the noise bank between calls (default: 1). The bank is the decomposed
 * noise of the ensemble. It depends on the signal length, the seed, the
 * ensemble size and the sifting settings, not on the signal, so with it kept
 * every further signal of the same length skips building it and gets the same
 * result as before. 0 frees it at the end of each call.
 */
ICEEMDAN_EXPORT void iceemdan_set_keep_noise_bank(ICEEMDANHandle *h, int keep)
{
    if (h && h->decomposer)
        h->decomposer->config().keep_noise_bank = keep != 0;
}

/** Free the noise bank kept from the last call. The next call builds it again. */
ICEEMDAN_EXPORT void iceemdan_release_noise_bank(ICEEMDANHandle *h)
{
    if (h && h->decomposer)
        h->decomposer->release_noise_bank();
}

// ============================================================================
// Decomposition
// ============================================================================

/**
 * Run ICEEMDAN decomposition
 * @param h Handle
 * @param signal Input signal array
 * @param n Signal length
 * @return Number of IMFs extracted (0 on failure)
 */
ICEEMDAN_EXPORT int iceemdan_decompose(ICEEMDANHandle *h, const double *signal, int n)
{
    if (!h || !h->decomposer || !signal || n < 4)
    {
        if (h)
            snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }
    h->error_msg[0] = '\0';

    try
    {
        const bool ok = h->decomposer->decompose(signal, n, h->imfs, h->residue);
        h->diagnostics = h->decomposer->last_diagnostics();
        if (!ok)
        {
            h->imfs.clear();
            h->residue.clear();
            snprintf(h->error_msg, sizeof(h->error_msg),
                     "Decomposition failed (noise bank larger than noise_bank_max_bytes?)");
            return 0;
        }
        return static_cast<int>(h->imfs.size());
    }
    catch (const std::exception &e)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "%s", e.what());
        return 0;
    }
    catch (...)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "Unknown error");
        return 0;
    }
}

/**
 * Run ICEEMDAN decomposition with diagnostics
 * @return Number of IMFs extracted (0 on failure)
 */
ICEEMDAN_EXPORT int iceemdan_decompose_with_diagnostics(ICEEMDANHandle *h, const double *signal, int n)
{
    if (!h || !h->decomposer || !signal || n < 4)
    {
        if (h)
            snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }
    h->error_msg[0] = '\0';

    try
    {
        const bool ok = h->decomposer->decompose_with_diagnostics(signal, n, h->imfs, h->residue, h->diagnostics);
        if (!ok)
        {
            h->imfs.clear();
            h->residue.clear();
            snprintf(h->error_msg, sizeof(h->error_msg),
                     "Decomposition failed (noise bank larger than noise_bank_max_bytes?)");
            return 0;
        }
        return static_cast<int>(h->imfs.size());
    }
    catch (const std::exception &e)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "%s", e.what());
        return 0;
    }
    catch (...)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "Unknown error");
        return 0;
    }
}

// ============================================================================
// Result Access
// ============================================================================

/**
 * Get number of IMFs from last decomposition
 */
ICEEMDAN_EXPORT int iceemdan_get_num_imfs(const ICEEMDANHandle *h)
{
    return h ? static_cast<int>(h->imfs.size()) : 0;
}

/**
 * Get IMF length (same as input signal length)
 */
ICEEMDAN_EXPORT int iceemdan_get_imf_length(const ICEEMDANHandle *h)
{
    return (h && !h->imfs.empty()) ? static_cast<int>(h->imfs[0].size()) : 0;
}

/**
 * Copy IMF data to output buffer
 * @param h Handle
 * @param imf_index Which IMF (0 = highest frequency)
 * @param out Output buffer (must be pre-allocated)
 * @return 1 on success, 0 on failure
 */
ICEEMDAN_EXPORT int iceemdan_get_imf(const ICEEMDANHandle *h, int imf_index, double *out)
{
    if (!h || imf_index < 0 || imf_index >= static_cast<int>(h->imfs.size()) || !out)
    {
        return 0;
    }
    std::memcpy(out, h->imfs[imf_index].data(), h->imfs[imf_index].size() * sizeof(double));
    return 1;
}

/**
 * Copy residue to output buffer
 * @param h Handle
 * @param out Output buffer (must be pre-allocated)
 * @return 1 on success, 0 on failure
 */
ICEEMDAN_EXPORT int iceemdan_get_residue(const ICEEMDANHandle *h, double *out)
{
    if (!h || h->residue.empty() || !out)
    {
        return 0;
    }
    std::memcpy(out, h->residue.data(), h->residue.size() * sizeof(double));
    return 1;
}

/**
 * Copy all IMFs to a contiguous 2D buffer (row-major: imfs[k][i] -> out[k * n + i])
 * @param h Handle
 * @param out Output buffer (must be pre-allocated: num_imfs * signal_length)
 * @return 1 on success, 0 on failure
 */
ICEEMDAN_EXPORT int iceemdan_get_all_imfs(const ICEEMDANHandle *h, double *out)
{
    if (!h || h->imfs.empty() || !out)
    {
        return 0;
    }
    const int n = static_cast<int>(h->imfs[0].size());
    for (size_t k = 0; k < h->imfs.size(); ++k)
    {
        std::memcpy(out + k * n, h->imfs[k].data(), n * sizeof(double));
    }
    return 1;
}

// ============================================================================
// Batches: many signals of one length (channels, epochs) in one call
// ============================================================================

/**
 * Decompose count signals of n samples each, stored one after another
 * (count x n, row-major): the channels of a recording, or its epochs. Each
 * result is the same, bit for bit, as iceemdan_decompose() gives for that
 * signal alone. The signals share one noise bank, and their trials are
 * interleaved on the library's threads: a thread that would wait for the end
 * of one signal's stage takes trials of the next signal instead.
 * The results stay in the handle until the next batch; read them with
 * iceemdan_batch_get() (the single-signal getters keep referring to the last
 * iceemdan_decompose()).
 * @return the largest number of IMFs of any signal, or 0 on failure (see iceemdan_get_error())
 */
ICEEMDAN_EXPORT int iceemdan_decompose_batch(ICEEMDANHandle *h, const double *signals, int count, int n)
{
    if (!h || !h->decomposer || !signals || count < 1 || n < 4)
    {
        if (h)
            snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }
    h->error_msg[0] = '\0';
    try
    {
        h->batch_max_imfs = 0;
        const bool ok = h->decomposer->decompose_batch(signals, count, n, h->batch_imfs, h->batch_residues,
                                                       &h->batch_diagnostics);
        if (!ok)
        {
            h->batch_imfs.clear();
            h->batch_residues.clear();
            snprintf(h->error_msg, sizeof(h->error_msg),
                     "Batch decomposition failed (noise bank larger than noise_bank_max_bytes?)");
            return 0;
        }
        int most = 0;
        for (const auto &m : h->batch_imfs)
            most = std::max(most, static_cast<int>(m.size()));
        h->batch_max_imfs = most;
        return most > 0 ? most : 0;
    }
    catch (const std::exception &e)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "%s", e.what());
        return 0;
    }
    catch (...)
    {
        snprintf(h->error_msg, sizeof(h->error_msg), "Unknown error");
        return 0;
    }
}

/** Signals in the last batch (0 when there is none). */
ICEEMDAN_EXPORT int iceemdan_batch_count(const ICEEMDANHandle *h)
{
    return h ? static_cast<int>(h->batch_residues.size()) : 0;
}

/** IMFs of signal s of the last batch; -1 when s is out of range. */
ICEEMDAN_EXPORT int iceemdan_batch_num_imfs(const ICEEMDANHandle *h, int s)
{
    if (!h || s < 0 || s >= static_cast<int>(h->batch_imfs.size()))
        return -1;
    return static_cast<int>(h->batch_imfs[s].size());
}

/** Orthogonality index of signal s of the last batch; -1 when s is out of range. */
ICEEMDAN_EXPORT double iceemdan_batch_orthogonality_index(const ICEEMDANHandle *h, int s)
{
    if (!h || s < 0 || s >= static_cast<int>(h->batch_diagnostics.size()))
        return -1.0;
    return h->batch_diagnostics[s].orthogonality_index;
}

/**
 * Copy the last batch out. imfs: count x max_imfs x n (max_imfs as
 * iceemdan_decompose_batch() returned it; a signal with fewer IMFs gets rows
 * of zeros after its own, so its IMFs and residue still add up to it).
 * residues: count x n. Either may be NULL.
 * @return 1, or 0 when there is no batch result
 */
ICEEMDAN_EXPORT int iceemdan_batch_get(const ICEEMDANHandle *h, double *imfs, double *residues)
{
    if (!h || h->batch_residues.empty())
        return 0;
    const size_t count = h->batch_residues.size();
    const size_t n = h->batch_residues[0].size();
    const size_t k_max = static_cast<size_t>(h->batch_max_imfs);
    for (size_t s = 0; s < count; ++s)
    {
        if (imfs)
        {
            double *out = imfs + s * k_max * n;
            for (size_t k = 0; k < k_max; ++k)
            {
                if (k < h->batch_imfs[s].size())
                    std::memcpy(out + k * n, h->batch_imfs[s][k].data(), n * sizeof(double));
                else
                    std::memset(out + k * n, 0, n * sizeof(double));
            }
        }
        if (residues)
            std::memcpy(residues + s * n, h->batch_residues[s].data(), n * sizeof(double));
    }
    return 1;
}

// ============================================================================
// Diagnostics Access
// ============================================================================

ICEEMDAN_EXPORT double iceemdan_get_orthogonality_index(const ICEEMDANHandle *h)
{
    return h ? h->diagnostics.orthogonality_index : -1.0;
}

ICEEMDAN_EXPORT int iceemdan_get_nan_count(const ICEEMDANHandle *h)
{
    return h ? h->diagnostics.nan_count : -1;
}

ICEEMDAN_EXPORT unsigned int iceemdan_get_rng_seed_used(const ICEEMDANHandle *h)
{
    return h ? h->diagnostics.rng_seed_used : 0;
}

ICEEMDAN_EXPORT int iceemdan_get_valid(const ICEEMDANHandle *h)
{
    return h ? (h->diagnostics.valid ? 1 : 0) : 0;
}

/** 1 when the last decomposition used the noise bank kept from an earlier call. */
ICEEMDAN_EXPORT int iceemdan_get_noise_bank_reused(const ICEEMDANHandle *h)
{
    return h ? (h->diagnostics.noise_bank_reused ? 1 : 0) : 0;
}

// Per-IMF convergence diagnostics (index 0 = first IMF). -1 / -1.0 when out of range.

static bool iceemdan_valid_imf(const ICEEMDANHandle *h, int k)
{
    return h && k >= 0 && k < static_cast<int>(h->diagnostics.sift_iterations.size());
}

ICEEMDAN_EXPORT int iceemdan_get_imf_sift_iterations(const ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.sift_iterations[imf_index] : -1;
}

ICEEMDAN_EXPORT double iceemdan_get_imf_mean_sift_iterations(const ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.mean_sift_iterations[imf_index] : -1.0;
}

ICEEMDAN_EXPORT int iceemdan_get_imf_nonconverged_trials(const ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.nonconverged_trials[imf_index] : -1;
}

ICEEMDAN_EXPORT int iceemdan_get_imf_trials(const ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.trials_per_imf[imf_index] : -1;
}

ICEEMDAN_EXPORT int iceemdan_get_imf_converged(const ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? (h->diagnostics.convergence_flags[imf_index] ? 1 : 0) : -1;
}

ICEEMDAN_EXPORT const char *iceemdan_get_error(const ICEEMDANHandle *h)
{
    return h ? h->error_msg : "Null handle";
}

// ============================================================================
// Utility
// ============================================================================

/**
 * Get library version
 */
ICEEMDAN_EXPORT const char *iceemdan_version(void)
{
    return ICEEMDAN_VERSION_STRING;
}

/**
 * Name of the FFT backend that transforms length n on the calling thread:
 * "VectorFFT", or "none" for n < 2 (no transform needed) and when VectorFFT
 * could not be loaded. Used by the analysis utilities.
 */
ICEEMDAN_EXPORT const char *iceemdan_fft_backend(int n)
{
    return eemd::fft::backend_name(eemd::fft::backend_for(n));
}

/**
 * Number of threads a decomposition uses, for the whole process (the calling
 * thread included). n < 1: every logical processor, which is also the
 * default unless the environment variable EEMD_NUM_THREADS is set. The
 * results do not depend on it.
 */
ICEEMDAN_EXPORT void iceemdan_set_num_threads(int n)
{
    eemd::set_num_threads(n);
}

ICEEMDAN_EXPORT int iceemdan_get_num_threads(void)
{
    return eemd::get_num_threads();
}

/**
 * Stop the library's worker threads; the next decomposition starts them
 * again. Call it before unloading this library from a running process. It is
 * not needed when the process exits.
 */
ICEEMDAN_EXPORT void iceemdan_shutdown_threads(void)
{
    eemd::shutdown_threads();
}

// ============================================================================
// Hilbert-Huang analysis of the IMFs (eemd_hht.hpp)
// ============================================================================

/**
 * Instantaneous attributes of one IMF of n samples (n >= 2) at sample_rate
 * samples per second, from its analytic signal: amplitude, frequency in Hz
 * and phase in radians (unwrapped), n values each. Any output may be NULL.
 * Within a period or two of each end of a non-periodic IMF the values are
 * distorted. Returns 1, or 0 on bad arguments.
 */
ICEEMDAN_EXPORT int iceemdan_instantaneous(const double *imf, int n, double sample_rate,
                                        double *amplitude, double *frequency, double *phase)
{
    return eemd::hht::instantaneous(imf, n, sample_rate, amplitude, frequency, phase) ? 1 : 0;
}

/** Instantaneous frequency of one IMF averaged with its energy as the weight, Hz. */
ICEEMDAN_EXPORT double iceemdan_mean_frequency(const double *imf, int n, double sample_rate)
{
    return eemd::hht::mean_frequency(imf, n, sample_rate);
}

/** Time columns of a Hilbert spectrum of n samples with time_step samples per column. */
ICEEMDAN_EXPORT int iceemdan_spectrum_columns(int n, int time_step)
{
    eemd::hht::SpectrumOptions opt;
    opt.time_step = time_step;
    return eemd::hht::spectrum_columns(n, opt);
}

/**
 * Hilbert spectrum of n_imfs IMFs of n samples each, stored one after
 * another. spectrum receives frequency_bins rows of
 * iceemdan_spectrum_columns(n, time_step) values (row-major; row b is the
 * bin from fmin + b w to fmin + (b + 1) w, w = (fmax - fmin) / frequency_bins);
 * marginal receives frequency_bins values, each row summed over time. Either
 * may be NULL. At each sample each IMF adds its energy (amplitude squared;
 * its amplitude when amplitude != 0) to the bin of its instantaneous
 * frequency; frequencies outside [fmin, fmax] are left out. fmax <= 0 means
 * sample_rate / 2. Returns 1, or 0 on bad arguments.
 */
ICEEMDAN_EXPORT int iceemdan_hilbert_spectrum(const double *imfs, int n_imfs, int n, double sample_rate,
                                           int frequency_bins, double fmin, double fmax, int time_step,
                                           int amplitude, double *spectrum, double *marginal)
{
    if (!imfs || n_imfs < 1)
        return 0;
    eemd::hht::SpectrumOptions opt;
    opt.frequency_bins = frequency_bins;
    opt.fmin = fmin;
    opt.fmax = fmax;
    opt.time_step = time_step;
    opt.amplitude = amplitude != 0;
    try
    {
        return eemd::hht::hilbert_spectrum(imfs, n_imfs, n, sample_rate, opt, spectrum, marginal) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

/**
 * Compute energy of a signal (sum of squares)
 */
ICEEMDAN_EXPORT double iceemdan_compute_energy(const double *signal, int n)
{
    double energy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        energy += signal[i] * signal[i];
    }
    return energy;
}
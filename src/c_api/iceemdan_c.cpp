/**
 * C API wrapper for ICEEMDAN - Python ctypes bindings
 *
 * CMake builds it as the shared library "iceemdan" (target iceemdan_shared),
 * linked with VectorFFT, its only dependency.
 */

#include "iceemdan.hpp"
#include <cstring>
#include <vector>

#ifdef _WIN32
#define ICEEMDAN_API extern "C" __declspec(dllexport)
#else
#define ICEEMDAN_API extern "C" __attribute__((visibility("default")))
#endif

// ============================================================================
// Opaque handle for ICEEMDAN instance
// ============================================================================

struct ICEEMDANHandle
{
    eemd::ICEEMDAN *decomposer;
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    eemd::DecompositionDiagnostics diagnostics;
    char error_msg[256];
};

// ============================================================================
// Lifecycle
// ============================================================================

/**
 * Create ICEEMDAN instance
 * @param mode 0=Standard, 1=Finance, 2=Scientific
 * @return Handle pointer (NULL on failure)
 */
ICEEMDAN_API ICEEMDANHandle *iceemdan_create(int mode)
{
    try
    {
        auto *handle = new ICEEMDANHandle();
        eemd::ProcessingMode pm;
        switch (mode)
        {
        case 1:
            pm = eemd::ProcessingMode::Finance;
            break;
        case 2:
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
ICEEMDAN_API void iceemdan_destroy(ICEEMDANHandle *handle)
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

ICEEMDAN_API void iceemdan_set_ensemble_size(ICEEMDANHandle *h, int size)
{
    if (h && h->decomposer)
        h->decomposer->config().ensemble_size = size;
}

ICEEMDAN_API void iceemdan_set_noise_std(ICEEMDANHandle *h, double std)
{
    if (h && h->decomposer)
        h->decomposer->config().noise_std = std;
}

ICEEMDAN_API void iceemdan_set_max_imfs(ICEEMDANHandle *h, int max_imfs)
{
    if (h && h->decomposer)
        h->decomposer->config().max_imfs = max_imfs;
}

ICEEMDAN_API void iceemdan_set_max_sift_iters(ICEEMDANHandle *h, int iters)
{
    if (h && h->decomposer)
        h->decomposer->config().max_sift_iters = iters;
}

ICEEMDAN_API void iceemdan_set_sift_threshold(ICEEMDANHandle *h, double thresh)
{
    if (h && h->decomposer)
        h->decomposer->config().sift_threshold = thresh;
}

ICEEMDAN_API void iceemdan_set_rng_seed(ICEEMDANHandle *h, unsigned int seed)
{
    if (h && h->decomposer)
        h->decomposer->config().rng_seed = seed;
}

ICEEMDAN_API void iceemdan_set_s_number(ICEEMDANHandle *h, int s)
{
    if (h && h->decomposer)
        h->decomposer->config().s_number = s;
}

/**
 * Set spline method
 * @param method 0=Cubic, 1=Akima, 2=Linear
 */
ICEEMDAN_API void iceemdan_set_spline_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case 1:
            h->decomposer->config().spline_method = eemd::SplineMethod::Akima;
            break;
        case 2:
            h->decomposer->config().spline_method = eemd::SplineMethod::Linear;
            break;
        default:
            h->decomposer->config().spline_method = eemd::SplineMethod::Cubic;
            break;
        }
    }
}

/**
 * Set volatility method
 * @param method 0=Global, 1=SMA, 2=EMA
 */
ICEEMDAN_API void iceemdan_set_volatility_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case 1:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::SMA;
            break;
        case 2:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::EMA;
            break;
        default:
            h->decomposer->config().volatility_method = eemd::VolatilityMethod::Global;
            break;
        }
    }
}

/**
 * Set boundary method
 * @param method 0=Mirror, 1=AR, 2=Linear
 */
ICEEMDAN_API void iceemdan_set_boundary_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
    {
        switch (method)
        {
        case 1:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::AR;
            break;
        case 2:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::Linear;
            break;
        default:
            h->decomposer->config().boundary_method = eemd::BoundaryMethod::Mirror;
            break;
        }
    }
}

/**
 * Local-mean operator: 0 = Sifted (Colominas et al. 2014, default),
 * 1 = SingleEnvelope (one envelope mean per trial: faster, not the published algorithm)
 */
ICEEMDAN_API void iceemdan_set_local_mean_method(ICEEMDANHandle *h, int method)
{
    if (h && h->decomposer)
        h->decomposer->config().local_mean_method =
            (method == 1) ? eemd::LocalMeanMethod::SingleEnvelope : eemd::LocalMeanMethod::Sifted;
}

/**
 * Keep the noise bank between calls (default: 1). The bank is the decomposed
 * noise of the ensemble. It depends on the signal length, the seed, the
 * ensemble size and the sifting settings, not on the signal, so with it kept
 * every further signal of the same length skips building it and gets the same
 * result as before. 0 frees it at the end of each call.
 */
ICEEMDAN_API void iceemdan_set_keep_noise_bank(ICEEMDANHandle *h, int keep)
{
    if (h && h->decomposer)
        h->decomposer->config().keep_noise_bank = keep != 0;
}

/** Free the noise bank kept from the last call. The next call builds it again. */
ICEEMDAN_API void iceemdan_release_noise_bank(ICEEMDANHandle *h)
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
ICEEMDAN_API int iceemdan_decompose(ICEEMDANHandle *h, const double *signal, int n)
{
    if (!h || !h->decomposer || !signal || n < 4)
    {
        if (h)
            snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }

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
ICEEMDAN_API int iceemdan_decompose_with_diagnostics(ICEEMDANHandle *h, const double *signal, int n)
{
    if (!h || !h->decomposer || !signal || n < 4)
    {
        if (h)
            snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }

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
ICEEMDAN_API int iceemdan_get_num_imfs(ICEEMDANHandle *h)
{
    return h ? static_cast<int>(h->imfs.size()) : 0;
}

/**
 * Get IMF length (same as input signal length)
 */
ICEEMDAN_API int iceemdan_get_imf_length(ICEEMDANHandle *h)
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
ICEEMDAN_API int iceemdan_get_imf(ICEEMDANHandle *h, int imf_index, double *out)
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
ICEEMDAN_API int iceemdan_get_residue(ICEEMDANHandle *h, double *out)
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
ICEEMDAN_API int iceemdan_get_all_imfs(ICEEMDANHandle *h, double *out)
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
// Diagnostics Access
// ============================================================================

ICEEMDAN_API double iceemdan_get_orthogonality_index(ICEEMDANHandle *h)
{
    return h ? h->diagnostics.orthogonality_index : -1.0;
}

ICEEMDAN_API int iceemdan_get_nan_count(ICEEMDANHandle *h)
{
    return h ? h->diagnostics.nan_count : -1;
}

ICEEMDAN_API unsigned int iceemdan_get_rng_seed_used(ICEEMDANHandle *h)
{
    return h ? h->diagnostics.rng_seed_used : 0;
}

ICEEMDAN_API int iceemdan_get_valid(ICEEMDANHandle *h)
{
    return h ? (h->diagnostics.valid ? 1 : 0) : 0;
}

/** 1 when the last decomposition used the noise bank kept from an earlier call. */
ICEEMDAN_API int iceemdan_get_noise_bank_reused(ICEEMDANHandle *h)
{
    return h ? (h->diagnostics.noise_bank_reused ? 1 : 0) : 0;
}

// Per-IMF convergence diagnostics (index 0 = first IMF). -1 / -1.0 when out of range.

static bool iceemdan_valid_imf(ICEEMDANHandle *h, int k)
{
    return h && k >= 0 && k < static_cast<int>(h->diagnostics.sift_iterations.size());
}

ICEEMDAN_API int iceemdan_get_imf_sift_iterations(ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.sift_iterations[imf_index] : -1;
}

ICEEMDAN_API double iceemdan_get_imf_mean_sift_iterations(ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.mean_sift_iterations[imf_index] : -1.0;
}

ICEEMDAN_API int iceemdan_get_imf_nonconverged_trials(ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.nonconverged_trials[imf_index] : -1;
}

ICEEMDAN_API int iceemdan_get_imf_trials(ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? h->diagnostics.trials_per_imf[imf_index] : -1;
}

ICEEMDAN_API int iceemdan_get_imf_converged(ICEEMDANHandle *h, int imf_index)
{
    return iceemdan_valid_imf(h, imf_index) ? (h->diagnostics.convergence_flags[imf_index] ? 1 : 0) : -1;
}

ICEEMDAN_API const char *iceemdan_get_error(ICEEMDANHandle *h)
{
    return h ? h->error_msg : "Null handle";
}

// ============================================================================
// Utility
// ============================================================================

/**
 * Get library version
 */
ICEEMDAN_API const char *iceemdan_version()
{
    return "1.1.0";
}

/**
 * Name of the FFT backend that transforms length n on the calling thread:
 * "VectorFFT", or "none" for n < 2 (no transform needed) and when VectorFFT
 * could not be loaded. Used by the analysis utilities.
 */
ICEEMDAN_API const char *iceemdan_fft_backend(int n)
{
    return eemd::fft::backend_name(eemd::fft::backend_for(n));
}

/**
 * Number of threads a decomposition uses, for the whole process (the calling
 * thread included). n < 1: every logical processor, which is also the
 * default unless the environment variable EEMD_NUM_THREADS is set. The
 * results do not depend on it.
 */
ICEEMDAN_API void iceemdan_set_num_threads(int n)
{
    eemd::set_num_threads(n);
}

ICEEMDAN_API int iceemdan_get_num_threads()
{
    return eemd::get_num_threads();
}

/**
 * Stop the library's worker threads; the next decomposition starts them
 * again. Call it before unloading this library from a running process. It is
 * not needed when the process exits.
 */
ICEEMDAN_API void iceemdan_shutdown_threads()
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
ICEEMDAN_API int iceemdan_instantaneous(const double *imf, int n, double sample_rate,
                                        double *amplitude, double *frequency, double *phase)
{
    return eemd::hht::instantaneous(imf, n, sample_rate, amplitude, frequency, phase) ? 1 : 0;
}

/** Instantaneous frequency of one IMF averaged with its energy as the weight, Hz. */
ICEEMDAN_API double iceemdan_mean_frequency(const double *imf, int n, double sample_rate)
{
    return eemd::hht::mean_frequency(imf, n, sample_rate);
}

/** Time columns of a Hilbert spectrum of n samples with time_step samples per column. */
ICEEMDAN_API int iceemdan_spectrum_columns(int n, int time_step)
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
ICEEMDAN_API int iceemdan_hilbert_spectrum(const double *imfs, int n_imfs, int n, double sample_rate,
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
ICEEMDAN_API double iceemdan_compute_energy(const double *signal, int n)
{
    double energy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        energy += signal[i] * signal[i];
    }
    return energy;
}
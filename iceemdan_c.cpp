/**
 * C API wrapper for ICEEMDAN - Python ctypes bindings
 * 
 * Compile as shared library:
 *   Windows: cl /LD /O2 iceemdan_c.cpp /link mkl_rt.lib
 *   Linux:   g++ -shared -fPIC -O3 -o libiceemdan.so iceemdan_c.cpp -lmkl_rt -liomp5
 */

#include "iceemdan_mkl.hpp"
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

struct ICEEMDANHandle {
    eemd::ICEEMDAN* decomposer;
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
ICEEMDAN_API ICEEMDANHandle* iceemdan_create(int mode) {
    try {
        auto* handle = new ICEEMDANHandle();
        eemd::ProcessingMode pm;
        switch (mode) {
            case 1:  pm = eemd::ProcessingMode::Finance; break;
            case 2:  pm = eemd::ProcessingMode::Scientific; break;
            default: pm = eemd::ProcessingMode::Standard; break;
        }
        handle->decomposer = new eemd::ICEEMDAN(pm);
        handle->error_msg[0] = '\0';
        return handle;
    } catch (...) {
        return nullptr;
    }
}

/**
 * Destroy ICEEMDAN instance
 */
ICEEMDAN_API void iceemdan_destroy(ICEEMDANHandle* handle) {
    if (handle) {
        delete handle->decomposer;
        delete handle;
    }
}

// ============================================================================
// Configuration
// ============================================================================

ICEEMDAN_API void iceemdan_set_ensemble_size(ICEEMDANHandle* h, int size) {
    if (h && h->decomposer) h->decomposer->config().ensemble_size = size;
}

ICEEMDAN_API void iceemdan_set_noise_std(ICEEMDANHandle* h, double std) {
    if (h && h->decomposer) h->decomposer->config().noise_std = std;
}

ICEEMDAN_API void iceemdan_set_max_imfs(ICEEMDANHandle* h, int max_imfs) {
    if (h && h->decomposer) h->decomposer->config().max_imfs = max_imfs;
}

ICEEMDAN_API void iceemdan_set_max_sift_iters(ICEEMDANHandle* h, int iters) {
    if (h && h->decomposer) h->decomposer->config().max_sift_iters = iters;
}

ICEEMDAN_API void iceemdan_set_sift_threshold(ICEEMDANHandle* h, double thresh) {
    if (h && h->decomposer) h->decomposer->config().sift_threshold = thresh;
}

ICEEMDAN_API void iceemdan_set_rng_seed(ICEEMDANHandle* h, unsigned int seed) {
    if (h && h->decomposer) h->decomposer->config().rng_seed = seed;
}

ICEEMDAN_API void iceemdan_set_s_number(ICEEMDANHandle* h, int s) {
    if (h && h->decomposer) h->decomposer->config().s_number = s;
}

/**
 * Set spline method
 * @param method 0=Cubic, 1=Akima, 2=Linear
 */
ICEEMDAN_API void iceemdan_set_spline_method(ICEEMDANHandle* h, int method) {
    if (h && h->decomposer) {
        switch (method) {
            case 1: h->decomposer->config().spline_method = eemd::SplineMethod::Akima; break;
            case 2: h->decomposer->config().spline_method = eemd::SplineMethod::Linear; break;
            default: h->decomposer->config().spline_method = eemd::SplineMethod::Cubic; break;
        }
    }
}

/**
 * Set volatility method
 * @param method 0=Global, 1=SMA, 2=EMA
 */
ICEEMDAN_API void iceemdan_set_volatility_method(ICEEMDANHandle* h, int method) {
    if (h && h->decomposer) {
        switch (method) {
            case 1: h->decomposer->config().volatility_method = eemd::VolatilityMethod::SMA; break;
            case 2: h->decomposer->config().volatility_method = eemd::VolatilityMethod::EMA; break;
            default: h->decomposer->config().volatility_method = eemd::VolatilityMethod::Global; break;
        }
    }
}

/**
 * Set boundary method
 * @param method 0=Mirror, 1=AR, 2=Linear
 */
ICEEMDAN_API void iceemdan_set_boundary_method(ICEEMDANHandle* h, int method) {
    if (h && h->decomposer) {
        switch (method) {
            case 1: h->decomposer->config().boundary_method = eemd::BoundaryMethod::AR; break;
            case 2: h->decomposer->config().boundary_method = eemd::BoundaryMethod::Linear; break;
            default: h->decomposer->config().boundary_method = eemd::BoundaryMethod::Mirror; break;
        }
    }
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
ICEEMDAN_API int iceemdan_decompose(ICEEMDANHandle* h, const double* signal, int n) {
    if (!h || !h->decomposer || !signal || n < 4) {
        if (h) snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }
    
    try {
        h->decomposer->decompose(signal, n, h->imfs, h->residue);
        return static_cast<int>(h->imfs.size());
    } catch (const std::exception& e) {
        snprintf(h->error_msg, sizeof(h->error_msg), "%s", e.what());
        return 0;
    } catch (...) {
        snprintf(h->error_msg, sizeof(h->error_msg), "Unknown error");
        return 0;
    }
}

/**
 * Run ICEEMDAN decomposition with diagnostics
 * @return Number of IMFs extracted (0 on failure)
 */
ICEEMDAN_API int iceemdan_decompose_with_diagnostics(ICEEMDANHandle* h, const double* signal, int n) {
    if (!h || !h->decomposer || !signal || n < 4) {
        if (h) snprintf(h->error_msg, sizeof(h->error_msg), "Invalid input");
        return 0;
    }
    
    try {
        h->decomposer->decompose_with_diagnostics(signal, n, h->imfs, h->residue, h->diagnostics);
        return static_cast<int>(h->imfs.size());
    } catch (const std::exception& e) {
        snprintf(h->error_msg, sizeof(h->error_msg), "%s", e.what());
        return 0;
    } catch (...) {
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
ICEEMDAN_API int iceemdan_get_num_imfs(ICEEMDANHandle* h) {
    return h ? static_cast<int>(h->imfs.size()) : 0;
}

/**
 * Get IMF length (same as input signal length)
 */
ICEEMDAN_API int iceemdan_get_imf_length(ICEEMDANHandle* h) {
    return (h && !h->imfs.empty()) ? static_cast<int>(h->imfs[0].size()) : 0;
}

/**
 * Copy IMF data to output buffer
 * @param h Handle
 * @param imf_index Which IMF (0 = highest frequency)
 * @param out Output buffer (must be pre-allocated)
 * @return 1 on success, 0 on failure
 */
ICEEMDAN_API int iceemdan_get_imf(ICEEMDANHandle* h, int imf_index, double* out) {
    if (!h || imf_index < 0 || imf_index >= static_cast<int>(h->imfs.size()) || !out) {
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
ICEEMDAN_API int iceemdan_get_residue(ICEEMDANHandle* h, double* out) {
    if (!h || h->residue.empty() || !out) {
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
ICEEMDAN_API int iceemdan_get_all_imfs(ICEEMDANHandle* h, double* out) {
    if (!h || h->imfs.empty() || !out) {
        return 0;
    }
    const int n = static_cast<int>(h->imfs[0].size());
    for (size_t k = 0; k < h->imfs.size(); ++k) {
        std::memcpy(out + k * n, h->imfs[k].data(), n * sizeof(double));
    }
    return 1;
}

// ============================================================================
// Diagnostics Access
// ============================================================================

ICEEMDAN_API double iceemdan_get_orthogonality_index(ICEEMDANHandle* h) {
    return h ? h->diagnostics.orthogonality_index : -1.0;
}

ICEEMDAN_API double iceemdan_get_reconstruction_error(ICEEMDANHandle* h) {
    return h ? h->diagnostics.reconstruction_error : -1.0;
}

ICEEMDAN_API double iceemdan_get_energy_conservation(ICEEMDANHandle* h) {
    return h ? h->diagnostics.energy_conservation : -1.0;
}

ICEEMDAN_API unsigned int iceemdan_get_rng_seed_used(ICEEMDANHandle* h) {
    return h ? h->diagnostics.rng_seed_used : 0;
}

ICEEMDAN_API const char* iceemdan_get_error(ICEEMDANHandle* h) {
    return h ? h->error_msg : "Null handle";
}

// ============================================================================
// Utility
// ============================================================================

/**
 * Get library version
 */
ICEEMDAN_API const char* iceemdan_version() {
    return "1.0.0";
}

/**
 * Compute energy of a signal (sum of squares)
 */
ICEEMDAN_API double iceemdan_compute_energy(const double* signal, int n) {
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        energy += signal[i] * signal[i];
    }
    return energy;
}

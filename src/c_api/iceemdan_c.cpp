/**
 * The C interface: include/iceemdan.h (eight functions) and
 * include/iceemdan_diagnostics.h (two). What each function does is described
 * there. CMake builds this file as the shared library "iceemdan" (target
 * iceemdan_shared), linked with VectorFFT, its only dependency. The Python
 * module (python/iceemdan.py) calls the same functions.
 */

#define ICEEMDAN_BUILD // export, not import
#include "iceemdan.h"
#include "iceemdan_diagnostics.h"

#include "iceemdan.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// the machine's memory (after the library's own headers: windows.h defines many macros)
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// Every definition is extern "C" once more: one that does not match its
// declaration in the header is then a compile error, not a second function.
#define ICEEMDAN_EXPORT extern "C" ICEEMDAN_API

#if defined(_MSC_VER) && !defined(__clang__)
#define ICEEMDAN_NOINLINE __declspec(noinline)
#else
#define ICEEMDAN_NOINLINE __attribute__((noinline))
#endif

// ============================================================================
// The processor
//
// The library is compiled for AVX2 and FMA (ICEEMDAN_PORTABLE; for the build
// machine otherwise). On an older processor its first such instruction would
// end the program. What is marked ICEEMDAN_BASELINE is compiled for any
// x86-64 processor instead: every exported function that can be called
// without a decomposer starts there, and refuses before anything else runs.
// ============================================================================

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define ICEEMDAN_X86_GNU 1
#define ICEEMDAN_BASELINE __attribute__((target("arch=x86-64"), noinline))
#else
#define ICEEMDAN_BASELINE
#endif
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

static ICEEMDAN_BASELINE int cpu_has_avx2_fma()
{
#if defined(ICEEMDAN_X86_GNU)
    // also checks that the operating system saves the AVX registers
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
        return 0;
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osxsave && avx) || (_xgetbv(0) & 6) != 6) // the OS saves the YMM registers
        return 0;
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1; // AVX2
#else
    return 1; // not x86: built for the machine it runs on
#endif
}

// ============================================================================
// Why the last call on this thread was refused
// ============================================================================

static thread_local char t_error[256] = "";

/** Sets the reason; for the baseline functions, so nothing but a byte copy. */
static ICEEMDAN_BASELINE void set_error(const char *text)
{
    size_t i = 0;
    for (; text[i] != '\0' && i + 1 < sizeof(t_error); ++i)
        t_error[i] = text[i];
    t_error[i] = '\0';
}

static ICEEMDAN_BASELINE int refuse_processor()
{
    set_error("this processor has no AVX2 and FMA, which the library is built for (Intel since 2013, AMD since 2015)");
    return -1;
}

/** Records the reason and returns -1. */
static int refuse(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    std::vsnprintf(t_error, sizeof(t_error), format, args);
    va_end(args);
    return -1;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE const char *iceemdan_error(void)
{
    return t_error;
}

// ============================================================================
// A decomposer
// ============================================================================

struct iceemdan_s
{
    eemd::ICEEMDAN decomposer;
    int nthreads = 1;
    // the last decomposition; count == 0: none
    std::vector<std::vector<std::vector<double>>> imfs; // [signal][imf][sample]
    std::vector<std::vector<double>> residues;
    std::vector<eemd::DecompositionDiagnostics> diagnostics;
    int count = 0;
    int n = 0;
    int rows = 0; // the most IMFs of any signal
};

// Decomposers alive: the worker threads stop with the last one, so that the
// library can be unloaded from a running process
static std::atomic<int> g_decomposers{0};

/**
 * This machine's logical processors: the default thread count. Measured on a
 * hybrid CPU (8 performance and 16 efficiency cores): a batch on all 32 takes
 * half the time it takes on the 8 performance cores, a single signal the same.
 */
static int logical_processors()
{
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    return n >= 1 ? n : 1;
}

/** The memory installed in this machine, in bytes (4 GiB when the system does not say). */
static size_t installed_memory()
{
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<size_t>(status.ullTotalPhys);
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0)
        return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
    return static_cast<size_t>(4) << 30;
}

/** The most memory the decomposed noise of a decomposer may take: half of what is installed. */
static size_t noise_limit()
{
    return installed_memory() / 2;
}

static double gib(size_t bytes)
{
    return static_cast<double>(bytes) / static_cast<double>(static_cast<size_t>(1) << 30);
}

static ICEEMDAN_NOINLINE iceemdan_t *create(const iceemdan_config_t *config)
{
    iceemdan_config_t c;
    std::memset(&c, 0, sizeof(c));
    if (config)
        c = *config;

    if (c.ensemble_size < 0)
        return refuse("ensemble_size %d: 0 (100 copies) or more", c.ensemble_size), nullptr;
    if (!eemd::is_finite_bits(c.noise) || !(c.noise >= 0.0))
        return refuse("noise %g: 0 (0.2) or more", c.noise), nullptr;
    if (c.max_imfs < 0 || c.max_imfs > 64)
        return refuse("max_imfs %d: 0 (as many as the signal has) to 64", c.max_imfs), nullptr;
    if (c.nthreads < 0 || c.nthreads > 1024)
        return refuse("nthreads %d: 0 (every processor) to 1024", c.nthreads), nullptr;
    if (c.flags & ~static_cast<unsigned>(ICEEMDAN_FLAG_FAST))
        return refuse("flags 0x%x: unknown flag", c.flags), nullptr;

    try
    {
        iceemdan_t *d = new iceemdan_s();
        eemd::ICEEMDANConfig &k = d->decomposer.config(); // starts as the published algorithm
        k.ensemble_size = c.ensemble_size > 0 ? c.ensemble_size : 100;
        k.noise_std = c.noise > 0.0 ? c.noise : 0.2;
        k.max_imfs = c.max_imfs > 0 ? c.max_imfs : 32;
        k.rng_seed = c.seed != 0 ? c.seed : 42u;
        d->nthreads = c.nthreads > 0 ? c.nthreads : logical_processors();
        k.num_threads = d->nthreads;
        k.keep_noise_bank = true;
        k.noise_bank_max_bytes = noise_limit();
        if (c.flags & ICEEMDAN_FLAG_FAST)
        {
            k.sift_stop_rule = eemd::SiftStopRule::SD;
            k.local_mean_method = eemd::LocalMeanMethod::SingleEnvelope;
            k.use_circular_bank = true;
        }
        g_decomposers.fetch_add(1, std::memory_order_relaxed);
        return d;
    }
    catch (const std::exception &e)
    {
        return refuse("no decomposer: %s", e.what()), nullptr;
    }
    catch (...)
    {
        return refuse("no decomposer: out of memory"), nullptr;
    }
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE iceemdan_t *iceemdan_create(const iceemdan_config_t *config)
{
    set_error("");
    if (!cpu_has_avx2_fma())
    {
        refuse_processor();
        return nullptr;
    }
    return create(config);
}

static ICEEMDAN_BASELINE int refuse_null()
{
    set_error("no decomposer (NULL)");
    return -1;
}

static ICEEMDAN_NOINLINE void destroy(iceemdan_t *d)
{
    delete d;
    if (g_decomposers.fetch_sub(1, std::memory_order_acq_rel) == 1)
        eemd::shutdown_threads();
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE void iceemdan_destroy(iceemdan_t *d)
{
    if (d)
        destroy(d);
}

static ICEEMDAN_NOINLINE int decompose(iceemdan_t *d, const double *signals, int count, int n)
{
    if (!signals)
        return refuse("signals is NULL");
    if (count < 1)
        return refuse("count %d: 1 signal or more", count);
    if (n < 4)
        return refuse("n %d: 4 samples or more", n);

    // The decomposed noise is kept in memory. The library does not change the
    // algorithm on its own to make it fit: the call is refused with the numbers.
    const eemd::ICEEMDANConfig &k = d->decomposer.config();
    if (!k.use_circular_bank && d->decomposer.noise_bank_bytes(n) > k.noise_bank_max_bytes)
        return refuse("the noise of %d copies of %d samples takes %.1f GiB decomposed, and this machine allows "
                      "%.1f (half its memory): a smaller ensemble_size, shorter signals, or ICEEMDAN_FLAG_FAST",
                      k.ensemble_size, n, gib(d->decomposer.noise_bank_bytes(n)), gib(k.noise_bank_max_bytes));

    d->count = 0; // from here on the earlier result is gone
    try
    {
        if (!d->decomposer.decompose_batch(signals, count, n, d->imfs, d->residues, &d->diagnostics))
            return refuse("the decomposed noise for signals of %d samples does not fit in the %.1f GiB this "
                          "machine allows (half its memory): shorter signals",
                          n, gib(k.noise_bank_max_bytes));
    }
    catch (const std::exception &e)
    {
        return refuse("decomposition stopped: %s", e.what());
    }
    catch (...)
    {
        return refuse("decomposition stopped: out of memory");
    }
    size_t rows = 0;
    for (const auto &signal_imfs : d->imfs)
        rows = std::max(rows, signal_imfs.size());
    d->rows = static_cast<int>(rows);
    d->n = n;
    d->count = count;
    return d->rows;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_decompose(iceemdan_t *d, const double *signals, int count, int n)
{
    set_error("");
    return d ? decompose(d, signals, count, n) : refuse_null();
}

static ICEEMDAN_NOINLINE int result(const iceemdan_t *d, double *imfs, double *residues, int *imf_counts)
{
    if (d->count == 0)
        return refuse("the decomposer holds no decomposition: iceemdan_decompose() first");
    const size_t n = static_cast<size_t>(d->n);
    const size_t rows = static_cast<size_t>(d->rows);
    for (size_t s = 0; s < static_cast<size_t>(d->count); ++s)
    {
        const auto &own = d->imfs[s];
        if (imfs)
        {
            double *out = imfs + s * rows * n;
            for (size_t j = 0; j < rows; ++j)
            {
                if (j < own.size())
                    std::memcpy(out + j * n, own[j].data(), n * sizeof(double));
                else
                    std::memset(out + j * n, 0, n * sizeof(double));
            }
        }
        if (residues)
            std::memcpy(residues + s * n, d->residues[s].data(), n * sizeof(double));
        if (imf_counts)
            imf_counts[s] = static_cast<int>(own.size());
    }
    return 0;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_result(const iceemdan_t *d, double *imfs, double *residues,
                                                       int *imf_counts)
{
    set_error("");
    return d ? result(d, imfs, residues, imf_counts) : refuse_null();
}

// ============================================================================
// Facts
// ============================================================================

static ICEEMDAN_NOINLINE void fill_info(const iceemdan_t *d, iceemdan_info_t *out)
{
    out->processors = logical_processors();
    if (!d)
        return;
    out->nthreads = d->nthreads;
    if (d->count == 0)
        return;
    out->signals = d->count;
    out->n = d->n;
    out->imfs = d->rows;
    out->noise_kept = d->diagnostics.front().noise_bank_reused ? 1 : 0;
    for (const eemd::DecompositionDiagnostics &g : d->diagnostics)
        out->repaired += g.nan_count;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_info(const iceemdan_t *d, iceemdan_info_t *out)
{
    set_error("");
    if (!out)
    {
        set_error("out is NULL");
        return -1;
    }
    out->version = ICEEMDAN_VERSION_STRING;
    out->cpu_supported = cpu_has_avx2_fma();
    out->processors = 0;
    out->nthreads = 0;
    out->signals = 0;
    out->n = 0;
    out->imfs = 0;
    out->noise_kept = 0;
    out->repaired = 0;
    if (out->cpu_supported)
        fill_info(d, out);
    return 0;
}

// ============================================================================
// The audit trail (iceemdan_diagnostics.h)
// ============================================================================

static ICEEMDAN_NOINLINE int signal_stats(const iceemdan_t *d, int signal, iceemdan_signal_stats_t *out)
{
    if (!out)
        return refuse("out is NULL");
    if (signal < 0 || signal >= d->count)
        return refuse("signal %d: the last decomposition has %d", signal, d->count);
    const eemd::DecompositionDiagnostics &g = d->diagnostics[static_cast<size_t>(signal)];
    out->imfs = static_cast<int>(d->imfs[static_cast<size_t>(signal)].size());
    out->orthogonality = g.orthogonality_index;
    out->repaired = g.nan_count;
    return 0;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_signal_stats(const iceemdan_t *d, int signal,
                                                             iceemdan_signal_stats_t *out)
{
    set_error("");
    return d ? signal_stats(d, signal, out) : refuse_null();
}

static ICEEMDAN_NOINLINE int imf_stats(const iceemdan_t *d, int signal, int imf, iceemdan_imf_stats_t *out)
{
    if (!out)
        return refuse("out is NULL");
    if (signal < 0 || signal >= d->count)
        return refuse("signal %d: the last decomposition has %d", signal, d->count);
    const eemd::DecompositionDiagnostics &g = d->diagnostics[static_cast<size_t>(signal)];
    if (imf < 0 || imf >= static_cast<int>(g.sift_iterations.size()))
        return refuse("imf %d: signal %d has %d", imf, signal, static_cast<int>(g.sift_iterations.size()));
    const size_t k = static_cast<size_t>(imf);
    out->copies = g.trials_per_imf[k];
    out->iterations_max = g.sift_iterations[k];
    out->iterations_mean = g.mean_sift_iterations[k];
    out->unconverged = g.nonconverged_trials[k];
    return 0;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_imf_stats(const iceemdan_t *d, int signal, int imf,
                                                          iceemdan_imf_stats_t *out)
{
    set_error("");
    return d ? imf_stats(d, signal, imf, out) : refuse_null();
}

// ============================================================================
// Hilbert-Huang analysis (eemd_hht.hpp)
// ============================================================================

static ICEEMDAN_NOINLINE int instantaneous(const double *imf, int n, double sample_rate,
                                           double *amplitude, double *frequency, double *phase)
{
    if (!imf)
        return refuse("imf is NULL");
    if (n < 2)
        return refuse("n %d: 2 samples or more", n);
    if (!eemd::is_finite_bits(sample_rate) || !(sample_rate > 0.0))
        return refuse("sample_rate %g: above 0", sample_rate);
    try
    {
        if (!eemd::hht::instantaneous(imf, n, sample_rate, amplitude, frequency, phase))
            return refuse("the analytic signal of %d samples could not be computed", n);
    }
    catch (const std::exception &e)
    {
        return refuse("instantaneous attributes stopped: %s", e.what());
    }
    return 0;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_instantaneous(const double *imf, int n, double sample_rate,
                                                              double *amplitude, double *frequency, double *phase)
{
    set_error("");
    if (!cpu_has_avx2_fma())
        return refuse_processor();
    return instantaneous(imf, n, sample_rate, amplitude, frequency, phase);
}

static ICEEMDAN_NOINLINE int spectrum_of(const double *imfs, int n_imfs, int n, double sample_rate,
                                         const iceemdan_spectrum_t *options, double *spectrum, double *marginal)
{
    iceemdan_spectrum_t o;
    std::memset(&o, 0, sizeof(o));
    if (options)
        o = *options;
    if (!imfs)
        return refuse("imfs is NULL");
    if (n_imfs < 1)
        return refuse("n_imfs %d: 1 IMF or more", n_imfs);
    if (n < 2)
        return refuse("n %d: 2 samples or more", n);
    if (!eemd::is_finite_bits(sample_rate) || !(sample_rate > 0.0))
        return refuse("sample_rate %g: above 0", sample_rate);
    if (o.frequency_bins < 0)
        return refuse("frequency_bins %d: 0 (128) or more", o.frequency_bins);
    if (o.time_step < 0)
        return refuse("time_step %d: 0 (every sample) or more", o.time_step);
    if (!eemd::is_finite_bits(o.fmin) || !eemd::is_finite_bits(o.fmax) || !(o.fmin >= 0.0) || !(o.fmax >= 0.0))
        return refuse("fmin %g, fmax %g: 0 or more", o.fmin, o.fmax);
    const double top = o.fmax > 0.0 ? o.fmax : 0.5 * sample_rate;
    if (!(o.fmin < top))
        return refuse("fmin %g is not below fmax %g", o.fmin, top);

    eemd::hht::SpectrumOptions opt;
    opt.frequency_bins = o.frequency_bins > 0 ? o.frequency_bins : 128;
    opt.fmin = o.fmin;
    opt.fmax = o.fmax;
    opt.time_step = o.time_step > 0 ? o.time_step : 1;
    opt.amplitude = o.amplitude != 0;
    int done = 0;
    try
    {
        done = eemd::hht::hilbert_spectrum(imfs, n_imfs, n, sample_rate, opt, spectrum, marginal) ? 1 : 0;
    }
    catch (const std::exception &e)
    {
        return refuse("the spectrum stopped: %s", e.what());
    }
    // a large spectrum uses the worker threads: with no decomposer alive, stop them again
    if (g_decomposers.load(std::memory_order_acquire) == 0)
        eemd::shutdown_threads();
    return done ? 0 : refuse("the spectrum of %d IMFs of %d samples could not be computed", n_imfs, n);
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_spectrum(const double *imfs, int n_imfs, int n, double sample_rate,
                                                         const iceemdan_spectrum_t *options, double *spectrum,
                                                         double *marginal)
{
    set_error("");
    if (!cpu_has_avx2_fma())
        return refuse_processor();
    return spectrum_of(imfs, n_imfs, n, sample_rate, options, spectrum, marginal);
}

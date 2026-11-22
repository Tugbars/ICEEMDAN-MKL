/**
 * The C interface: include/iceemdan.h (eight functions) and
 * include/iceemdan_diagnostics.h (two), as a shared library.
 *
 * Everything is computed by the C++ interface, include/iceemdan.hpp; a call
 * it refuses with an exception is refused here with -1 or NULL and its text
 * in iceemdan_error(). This file adds what only a ready-built library needs:
 * the check that the processor can run it, and stopping the worker threads
 * when the last decomposer is gone, so that the library can be unloaded.
 * CMake builds it as the shared library "iceemdan" (target iceemdan_shared),
 * linked with VectorFFT, its only dependency. The Python module
 * (python/iceemdan.py) calls the same functions.
 */

#define ICEEMDAN_BUILD // export, not import
#include "iceemdan.h"
#include "iceemdan_diagnostics.h"

#include "iceemdan.hpp"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

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
// x86-64 processor instead: every exported function starts there, and refuses
// before anything else runs.
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

static thread_local char t_error[320] = "";

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

static ICEEMDAN_BASELINE int refuse_null()
{
    set_error("no decomposer (NULL)");
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

/** Records the exception being handled as the reason and returns -1. */
static int refuse_exception()
{
    try
    {
        throw;
    }
    catch (const std::bad_alloc &)
    {
        return refuse("out of memory");
    }
    catch (const std::exception &e)
    {
        return refuse("%s", e.what());
    }
    catch (...)
    {
        return refuse("the call stopped on an unknown error");
    }
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
    iceemdan::Decomposer decomposer;
    std::vector<iceemdan::Decomposition> last; // the last decomposition, a signal each; empty: none
    int n = 0;
    int rows = 0; // the most IMFs of any signal

    explicit iceemdan_s(const iceemdan::Config &config) : decomposer(config) {}
};

// Decomposers alive: the worker threads stop with the last one, so that the
// library can be unloaded from a running process
static std::atomic<int> g_decomposers{0};

static ICEEMDAN_NOINLINE iceemdan_t *create(const iceemdan_config_t *config)
{
    iceemdan::Config c; // zeros: the published algorithm
    if (config)
    {
        if (config->flags & ~static_cast<unsigned>(ICEEMDAN_FLAG_FAST))
            return refuse("flags 0x%x: unknown flag", config->flags), nullptr;
        c.ensemble_size = config->ensemble_size;
        c.noise = config->noise;
        c.max_imfs = config->max_imfs;
        c.seed = config->seed;
        c.nthreads = config->nthreads;
        c.fast = (config->flags & ICEEMDAN_FLAG_FAST) != 0;
    }
    try
    {
        iceemdan_t *d = new iceemdan_s(c);
        g_decomposers.fetch_add(1, std::memory_order_relaxed);
        return d;
    }
    catch (...)
    {
        return refuse_exception(), nullptr;
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
    try
    {
        std::vector<iceemdan::Decomposition> result = d->decomposer.decompose(signals, count, n);
        size_t rows = 0;
        for (const iceemdan::Decomposition &r : result)
            rows = std::max(rows, r.imfs.size());
        d->last = std::move(result);
        d->n = n;
        d->rows = static_cast<int>(rows);
        return d->rows;
    }
    catch (...)
    {
        return refuse_exception();
    }
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_decompose(iceemdan_t *d, const double *signals, int count, int n)
{
    set_error("");
    return d ? decompose(d, signals, count, n) : refuse_null();
}

static ICEEMDAN_NOINLINE int result(const iceemdan_t *d, double *imfs, double *residues, int *imf_counts)
{
    if (d->last.empty())
        return refuse("the decomposer holds no decomposition: iceemdan_decompose() first");
    const size_t n = static_cast<size_t>(d->n);
    const size_t rows = static_cast<size_t>(d->rows);
    for (size_t s = 0; s < d->last.size(); ++s)
    {
        const iceemdan::Decomposition &r = d->last[s];
        if (imfs)
        {
            double *out = imfs + s * rows * n;
            for (size_t j = 0; j < rows; ++j)
            {
                if (j < r.imfs.size())
                    std::memcpy(out + j * n, r.imfs[j].data(), n * sizeof(double));
                else
                    std::memset(out + j * n, 0, n * sizeof(double));
            }
        }
        if (residues)
            std::memcpy(residues + s * n, r.residue.data(), n * sizeof(double));
        if (imf_counts)
            imf_counts[s] = static_cast<int>(r.imfs.size());
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
    out->processors = iceemdan::Decomposer::processors();
    if (!d)
        return;
    out->nthreads = d->decomposer.threads();
    if (d->last.empty())
        return;
    out->signals = static_cast<int>(d->last.size());
    out->n = d->n;
    out->imfs = d->rows;
    out->noise_kept = d->decomposer.noise_kept() ? 1 : 0;
    for (const iceemdan::Decomposition &r : d->last)
        out->repaired += r.repaired;
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
    if (signal < 0 || signal >= static_cast<int>(d->last.size()))
        return refuse("signal %d: the last decomposition has %d", signal, static_cast<int>(d->last.size()));
    const iceemdan::Decomposition &r = d->last[static_cast<size_t>(signal)];
    out->imfs = static_cast<int>(r.imfs.size());
    out->orthogonality = r.orthogonality;
    out->repaired = r.repaired;
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
    if (signal < 0 || signal >= static_cast<int>(d->last.size()))
        return refuse("signal %d: the last decomposition has %d", signal, static_cast<int>(d->last.size()));
    const iceemdan::Decomposition &r = d->last[static_cast<size_t>(signal)];
    if (imf < 0 || imf >= static_cast<int>(r.stats.size()))
        return refuse("imf %d: signal %d has %d", imf, signal, static_cast<int>(r.stats.size()));
    const iceemdan::ImfStats &s = r.stats[static_cast<size_t>(imf)];
    out->copies = s.copies;
    out->iterations_max = s.iterations_max;
    out->iterations_mean = s.iterations_mean;
    out->unconverged = s.unconverged;
    return 0;
}

ICEEMDAN_EXPORT ICEEMDAN_BASELINE int iceemdan_imf_stats(const iceemdan_t *d, int signal, int imf,
                                                          iceemdan_imf_stats_t *out)
{
    set_error("");
    return d ? imf_stats(d, signal, imf, out) : refuse_null();
}

// ============================================================================
// Hilbert-Huang analysis
// ============================================================================

static ICEEMDAN_NOINLINE int instantaneous(const double *imf, int n, double sample_rate,
                                           double *amplitude, double *frequency, double *phase)
{
    try
    {
        iceemdan::instantaneous(imf, n, sample_rate, amplitude, frequency, phase);
        return 0;
    }
    catch (...)
    {
        return refuse_exception();
    }
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
    iceemdan::SpectrumOptions o; // zeros: 128 bands to half the sample rate, a column per sample, energy
    if (options)
    {
        o.frequency_bins = options->frequency_bins;
        o.fmin = options->fmin;
        o.fmax = options->fmax;
        o.time_step = options->time_step;
        o.amplitude = options->amplitude != 0;
    }
    int status = 0;
    try
    {
        iceemdan::spectrum(imfs, n_imfs, n, sample_rate, o, spectrum, marginal);
    }
    catch (...)
    {
        status = refuse_exception();
    }
    // a large spectrum uses the worker threads: with no decomposer alive, stop them again
    if (g_decomposers.load(std::memory_order_acquire) == 0)
        eemd::shutdown_threads();
    return status;
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

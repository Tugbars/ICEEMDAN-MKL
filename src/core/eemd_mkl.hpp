/**
 * EEMD (Ensemble Empirical Mode Decomposition): the sifting, the envelopes and
 * the EEMD ensemble, on which ICEEMDAN (iceemdan_mkl.hpp) builds.
 *
 * - Envelopes evaluated at every sample in one pass (no knot search)
 * - Raw pointer hot paths (no std::vector overhead in inner loops)
 * - Fused mean/SD/update loop (single memory pass)
 * - Zero-malloc scratch pads (pre-allocated, reused buffers, grow-only)
 *
 * Everything is this library's own except the FFTs:
 * - the envelope splines: cubic_spline.hpp (cubic, Akima, linear)
 * - the noise: eemd_rng.hpp (Philox4x32-10 and the inverse normal
 *   distribution function)
 * - the threads: eemd_threads.hpp, a fork-join pool on std::thread
 * - the FFTs: VectorFFT (eemd_fft.hpp)
 *
 * License: MIT
 */

#ifndef EEMD_MKL_HPP
#define EEMD_MKL_HPP

#include "eemd_threads.hpp" // set_num_threads(), the fork-join pool
#include "eemd_rng.hpp"     // the noise

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <new>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h> // _MM_SET_FLUSH_ZERO_MODE, _MM_SET_DENORMALS_ZERO_MODE
#endif

#include "eemd_fft.hpp" // real FFTs (VectorFFT)
#include "cubic_spline.hpp" // the cubic envelopes

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Compiler-aware SIMD pragma
#if defined(__INTEL_LLVM_COMPILER) || defined(__GNUC__) || defined(__clang__)
// _Pragma takes ONE string literal (no concatenation), so the clause is
// stringized as a whole through EEMD_PRAGMA.
#define EEMD_PRAGMA(x) _Pragma(#x)
#define EEMD_OMP_SIMD EEMD_PRAGMA(omp simd)
#define EEMD_OMP_SIMD_REDUCTION(op, var) EEMD_PRAGMA(omp simd reduction(op : var))
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2) EEMD_PRAGMA(omp simd reduction(op : v1, v2))
#else
#define EEMD_OMP_SIMD
#define EEMD_OMP_SIMD_REDUCTION(op, var)
#define EEMD_OMP_SIMD_REDUCTION2(op, v1, v2)
#endif

// ============================================================================
// Hardware Constants (tune for your CPU)
// ============================================================================

#define EEMD_P_CORES 8
#define EEMD_CACHE_LINE 64

/**
 * Low-latency mode: n_cores threads that never sleep between decompositions
 * (each worker keeps one core busy).
 */
inline void eemd_init_low_latency(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    eemd::set_num_threads(n_cores);
    eemd::set_thread_block_ms(-1.0);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD: Low-latency mode (%d threads, never sleeping, DAZ/FTZ)\n", n_cores);
    }
}

/**
 * Throughput mode: two threads per core (hyperthreading); a worker sleeps as
 * soon as its share is done.
 */
inline void eemd_init_throughput(int n_cores = EEMD_P_CORES, bool verbose = false)
{
    eemd::set_num_threads(n_cores * 2);
    eemd::set_thread_block_ms(0.0);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (verbose)
    {
        printf("EEMD: Throughput mode (%d threads with HT)\n", n_cores * 2);
    }
}

namespace eemd
{

    // ============================================================================
    // Spline Method Selection
    // ============================================================================

    /**
     * Spline interpolation method for envelope construction
     * - Cubic: Smooth C2 spline with not-a-knot ends, can overshoot on sharp changes (default)
     * - Akima: Local C1 spline, resistant to outliers (recommended for finance)
     * - Linear: Simple linear interpolation (fastest, least accurate)
     */
    enum class SplineMethod : uint8_t
    {
        Cubic = 0, // smooth (C2), can overshoot on sharp changes
        Akima = 1, // local (C1), outlier-resistant
        Linear = 2 // fastest, for very few extrema
    };

    /**
     * When sifting a mode stops
     * - SD:      sum(m^2) / sum(h^2) < sift_threshold, plus the S-number and
     *            extrema/zero-crossing early exits. Cheap; typically stops
     *            after 2-3 iterations at the default threshold of 0.05.
     * - Rilling: the criterion of Rilling, Flandrin & Goncalves (2003), as in
     *            their emd.m that the ICEEMDAN authors' reference code uses:
     *            with mode amplitude a = (U - L) / 2 and envelope mean m,
     *            sift while |m| > theta1 * a on more than a fraction alpha of
     *            the samples or |m| > theta2 * a anywhere, and stop only when
     *            the extrema and zero-crossing counts differ by at most one.
     */
    enum class SiftStopRule : uint8_t
    {
        SD = 0,
        Rilling = 1
    };

    /** Thresholds of SiftStopRule::Rilling (emd.m defaults [0.05, 0.5, 0.05]). */
    struct RillingThresholds
    {
        double theta1 = 0.05;
        double theta2 = 0.5;
        double alpha = 0.05;
    };

    // ============================================================================
    // Configuration
    // ============================================================================

    struct EEMDConfig
    {
        int32_t max_imfs = 10;
        int32_t max_sift_iters = 100;
        double sift_threshold = 0.05;
        int32_t ensemble_size = 100;
        double noise_std = 0.2;
        int32_t boundary_extend = 2;
        uint32_t rng_seed = 42;

        // S-number stopping criterion: stop after S consecutive iterations
        // where extrema count is stable. Benchmarks show S=6 gives ~12% speedup.
        // Set to 0 to disable (use only SD criterion).
        int32_t s_number = 6;

        // Spline method for envelope interpolation
        SplineMethod spline_method = SplineMethod::Cubic;

        // Sifting stop rule (see SiftStopRule); sift_threshold and s_number
        // apply to SD, rilling to Rilling.
        SiftStopRule stop_rule = SiftStopRule::SD;
        RillingThresholds rilling;

        // Envelope ends: false = mirror boundary_extend extrema plus a linear
        // end point (extend_extrema_raw, the original scheme); true = the
        // symmetric extension of emd.m (extend_extrema_symmetric), which stays
        // stable when sifting runs for many iterations.
        bool symmetric_boundary = false;
    };

    // ============================================================================
    // Finite check that survives -ffast-math
    //
    // The project builds with -ffast-math / -fp-model fast / /fp:fast, under
    // which the compiler may assume no NaN or Inf exists and fold
    // std::isfinite() to true. Testing the exponent bits cannot be folded.
    // ============================================================================

    inline bool is_finite_bits(double x)
    {
        uint64_t u;
        std::memcpy(&u, &x, sizeof(u));
        return ((u >> 52) & 0x7FFu) != 0x7FFu; // exponent all ones = Inf or NaN
    }

    // ============================================================================
    // Per-trial random streams
    //
    // Trial t of seed s draws its noise from its own Philox4x32-10 stream, keyed
    // (s, t) (eemd_rng.hpp). The noise of a trial therefore depends only on
    // (s, t): not on the thread that runs it, the number of threads, or the
    // order trials are scheduled in.
    // ============================================================================

    /** Fill out[0..n) with N(mean, sigma) noise of trial `trial` under `seed`. */
    inline bool trial_gaussian(uint32_t seed, uint32_t trial, int32_t n, double *out,
                               double mean = 0.0, double sigma = 1.0)
    {
        if (n < 0 || (n > 0 && !out))
            return false;
        rng::gaussian(seed, trial, n, out, mean, sigma);
        return true;
    }

    // ============================================================================
    // Aligned Buffer - Grow-Only, 64-byte Aligned
    // ============================================================================

    template <typename T>
    struct AlignedBuffer
    {
        T *data = nullptr;
        size_t size = 0;
        size_t capacity = 0;

        // 64-byte aligned memory; throws std::bad_alloc when there is none
        static T *allocate(size_t n)
        {
            return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(EEMD_CACHE_LINE)));
        }
        static void deallocate(T *p)
        {
            ::operator delete(p, std::align_val_t(EEMD_CACHE_LINE));
        }

        AlignedBuffer() = default;

        explicit AlignedBuffer(size_t n)
        {
            if (n > 0)
            {
                data = allocate(n);
                size = n;
                capacity = n;
            }
        }

        ~AlignedBuffer()
        {
            if (data)
                deallocate(data);
        }

        AlignedBuffer(const AlignedBuffer &) = delete;
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;

        AlignedBuffer(AlignedBuffer &&other) noexcept
            : data(other.data), size(other.size), capacity(other.capacity)
        {
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }

        AlignedBuffer &operator=(AlignedBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (data)
                    deallocate(data);
                data = other.data;
                size = other.size;
                capacity = other.capacity;
                other.data = nullptr;
                other.size = 0;
                other.capacity = 0;
            }
            return *this;
        }

        void resize(size_t n)
        {
            if (n > capacity)
            {
                if (data)
                    deallocate(data);
                data = nullptr;
                capacity = 0;
                data = (n > 0) ? allocate(n) : nullptr;
                capacity = n;
            }
            size = n;
        }

        void zero()
        {
            if (data && size > 0)
            {
                std::memset(data, 0, size * sizeof(T));
            }
        }

        T &operator[](size_t i) { return data[i]; }
        const T &operator[](size_t i) const { return data[i]; }
    };

    // ============================================================================
    // Thread Scratch Pad - Fixed-Size Arrays with Counters
    // ============================================================================

    struct ThreadScratchPad
    {
        // Fixed-size arrays (resized once at construction)
        std::vector<int32_t> max_idx;
        std::vector<int32_t> min_idx;
        std::vector<double> ext_x;
        std::vector<double> ext_y;

        // Counters (avoids vector size() calls in hot path)
        int32_t n_max = 0;
        int32_t n_min = 0;
        int32_t n_ext = 0;
        int32_t ext_start = 0;

        // Second knot buffer pair (symmetric extension builds both envelopes at once)
        std::vector<double> ext_x2;
        std::vector<double> ext_y2;
        int32_t n_ext2 = 0;

        explicit ThreadScratchPad(int32_t n)
        {
            const int32_t max_extrema = n / 2 + 2;
            const int32_t ext_size = n + 128; // extrema + up to 2 x 31 mirrored knots + end points

            max_idx.resize(max_extrema);
            min_idx.resize(max_extrema);
            ext_x.resize(ext_size);
            ext_y.resize(ext_size);
            ext_x2.resize(ext_size);
            ext_y2.resize(ext_size);
        }
    };

    // ============================================================================
    // Peak Finding - Raw Pointer Output (No vector push_back)
    //
    // A strict peak (x[i-1] < x[i] > x[i+1]) is the common case and is found
    // in one pass. Flat extrema - runs of equal samples such as
    // prices quantized to a tick - have no strict peak, so when the pass sees
    // two equal neighbours it rescans with plateau handling: a run bounded by
    // lower (higher) samples on both sides is one maximum (minimum), placed at
    // the centre of the run. Runs touching either end are not extrema.
    // ============================================================================

    namespace detail
    {
        // kMax = true: maxima; false: minima. out_idx may be null (count only).
        template <bool kMax>
        inline int32_t scan_extrema(const double *__restrict s, int32_t n, int32_t *__restrict out_idx)
        {
            if (n < 3)
                return 0;

            auto above = [](double a, double b) { return kMax ? (a > b) : (a < b); };

            int32_t count = 0;
            bool has_tie = false;
            for (int32_t i = 1; i < n - 1; ++i)
            {
                const double c = s[i];
                has_tie |= (c == s[i + 1]);
                if (above(c, s[i - 1]) & above(c, s[i + 1]))
                {
                    if (out_idx)
                        out_idx[count] = i;
                    ++count;
                }
            }
            if (!has_tie)
                return count;

            // Plateau-aware rescan (same result as above wherever there is no tie)
            count = 0;
            int32_t i = 1;
            while (i < n - 1)
            {
                if (above(s[i], s[i - 1]))
                {
                    int32_t j = i;
                    while (j < n - 1 && s[j + 1] == s[i])
                        ++j; // run [i, j] of equal values
                    if (j < n - 1 && above(s[i], s[j + 1]))
                    {
                        // the middle of the run, the later of two middles (emd.m:
                        // round((first + last) / 2) in 1-based indices)
                        if (out_idx)
                            out_idx[count] = i + (j - i + 1) / 2;
                        ++count;
                    }
                    i = j + 1;
                }
                else
                {
                    ++i;
                }
            }
            return count;
        }
    } // namespace detail

    inline void find_maxima_raw(
        const double *__restrict signal,
        int32_t n,
        int32_t *__restrict out_idx,
        int32_t &out_count)
    {
        out_count = detail::scan_extrema<true>(signal, n, out_idx);
    }

    inline void find_minima_raw(
        const double *__restrict signal,
        int32_t n,
        int32_t *__restrict out_idx,
        int32_t &out_count)
    {
        out_count = detail::scan_extrema<false>(signal, n, out_idx);
    }

    /**
     * Maxima and minima together, in one pass over the signal (four samples
     * at a time with AVX2). Gives the same indices as find_maxima_raw and
     * find_minima_raw, flat extrema included. Each index array must have room
     * for one entry more than it receives.
     */
    inline void find_extrema_raw(
        const double *__restrict signal,
        int32_t n,
        int32_t *__restrict max_idx,
        int32_t &n_max,
        int32_t *__restrict min_idx,
        int32_t &n_min)
    {
        n_max = n_min = 0;
        if (n < 3)
            return;

        const double *__restrict s = signal;
        int32_t cmax = 0;
        int32_t cmin = 0;
        bool has_tie = false;
        int32_t i = 1;

#if defined(__AVX2__)
        // Samples i..i+3 against their left and right neighbours. An index is
        // stored for every lane and the count moves on only where the lane
        // holds an extremum, so the loop has no data-dependent branch inside.
        int tie_mask = 0;
        for (; i + 4 <= n - 1; i += 4)
        {
            const __m256d c = _mm256_loadu_pd(s + i);
            const __m256d l = _mm256_loadu_pd(s + i - 1);
            const __m256d r = _mm256_loadu_pd(s + i + 1);
            const int is_max = _mm256_movemask_pd(
                _mm256_and_pd(_mm256_cmp_pd(c, l, _CMP_GT_OQ), _mm256_cmp_pd(c, r, _CMP_GT_OQ)));
            const int is_min = _mm256_movemask_pd(
                _mm256_and_pd(_mm256_cmp_pd(c, l, _CMP_LT_OQ), _mm256_cmp_pd(c, r, _CMP_LT_OQ)));
            tie_mask |= _mm256_movemask_pd(_mm256_cmp_pd(c, r, _CMP_EQ_OQ));
            if (is_max)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    max_idx[cmax] = i + lane;
                    cmax += (is_max >> lane) & 1;
                }
            }
            if (is_min)
            {
                for (int lane = 0; lane < 4; ++lane)
                {
                    min_idx[cmin] = i + lane;
                    cmin += (is_min >> lane) & 1;
                }
            }
        }
        has_tie = tie_mask != 0;
#endif
        for (; i < n - 1; ++i)
        {
            const double c = s[i];
            has_tie |= (c == s[i + 1]);
            if ((c > s[i - 1]) & (c > s[i + 1]))
                max_idx[cmax++] = i;
            if ((c < s[i - 1]) & (c < s[i + 1]))
                min_idx[cmin++] = i;
        }

        if (!has_tie)
        {
            n_max = cmax;
            n_min = cmin;
            return;
        }
        // Equal neighbours somewhere: the plateau-aware scans decide
        n_max = detail::scan_extrema<true>(s, n, max_idx);
        n_min = detail::scan_extrema<false>(s, n, min_idx);
    }

    /** Number of local maxima plus minima (plateau-aware, same rule as above). */
    inline int32_t count_extrema_raw(const double *signal, int32_t n)
    {
        return detail::scan_extrema<true>(signal, n, nullptr) +
               detail::scan_extrema<false>(signal, n, nullptr);
    }

    /**
     * Zero crossings as emd.m counts them: neighbours of opposite sign, plus
     * one for each run of samples that are exactly zero.
     */
    inline int32_t count_zero_crossings(const double *signal, int32_t n)
    {
        int32_t count = 0;
        for (int32_t i = 1; i < n; ++i)
        {
            const double a = signal[i - 1], b = signal[i];
            count += ((a < 0.0) & (b > 0.0)) | ((a > 0.0) & (b < 0.0));
            count += (b == 0.0) & (a != 0.0); // a run of zeros starts at i
        }
        if (n > 0 && signal[0] == 0.0)
            ++count; // a run of zeros starts at 0
        return count;
    }

    /**
     * SiftStopRule::Rilling: true when the mode h (with upper/lower
     * envelopes U, L and n_extrema extrema) should NOT be sifted further.
     */
    inline bool rilling_should_stop(const double *__restrict h,
                                    const double *__restrict upper,
                                    const double *__restrict lower,
                                    int32_t n, int32_t n_extrema,
                                    const RillingThresholds &th)
    {
        int64_t over = 0;
        int32_t big = 0;
        for (int32_t i = 0; i < n; ++i)
        {
            const double m = std::abs(0.5 * (upper[i] + lower[i]));
            const double a = 0.5 * std::abs(upper[i] - lower[i]);
            over += (m > th.theta1 * a);
            big |= (m > th.theta2 * a);
        }
        const bool keep_sifting = (static_cast<double>(over) > th.alpha * n || big) && n_extrema > 2;
        if (keep_sifting)
            return false;
        return std::abs(count_zero_crossings(h, n) - n_extrema) <= 1;
    }

    // ============================================================================
    // Boundary Extension - Raw Pointer Output
    // ============================================================================

    inline void extend_extrema_raw(
        const int32_t *__restrict indices,
        int32_t n_indices,
        const double *__restrict signal,
        int32_t signal_len,
        int32_t extend_count,
        double *__restrict out_x,
        double *__restrict out_y,
        int32_t &out_count,
        int32_t &original_start)
    {
        if (n_indices < 2)
        {
            out_count = n_indices;
            original_start = 0;
            for (int32_t i = 0; i < n_indices; ++i)
            {
                out_x[i] = static_cast<double>(indices[i]);
                out_y[i] = signal[indices[i]];
            }
            return;
        }

        const int32_t left_ext = std::min(extend_count, n_indices - 1);
        const int32_t right_ext = std::min(extend_count, n_indices - 1);

        // Compute coverage after mirroring
        double leftmost_x = static_cast<double>(indices[0]);
        double rightmost_x = static_cast<double>(indices[n_indices - 1]);

        if (left_ext > 0)
        {
            leftmost_x = 2.0 * indices[0] - indices[left_ext];
        }
        if (right_ext > 0)
        {
            rightmost_x = 2.0 * indices[n_indices - 1] - indices[n_indices - 1 - right_ext];
        }

        const bool need_left = (leftmost_x > 0.0);
        const bool need_right = (rightmost_x < static_cast<double>(signal_len - 1));

        int32_t pos = 0;

        // Left boundary point (linear extrapolation)
        if (need_left)
        {
            const double x0 = static_cast<double>(indices[0]);
            const double x1 = static_cast<double>(indices[1]);
            const double y0 = signal[indices[0]];
            const double y1 = signal[indices[1]];
            const double slope = (y1 - y0) / (x1 - x0);
            out_x[pos] = -1.0;
            out_y[pos] = y0 + slope * (-1.0 - x0);
            ++pos;
        }

        // Mirror left
        for (int32_t i = 0; i < left_ext; ++i)
        {
            const int32_t src = left_ext - i;
            out_x[pos] = 2.0 * indices[0] - indices[src];
            out_y[pos] = signal[indices[src]];
            ++pos;
        }

        original_start = pos;

        // Original extrema
        for (int32_t i = 0; i < n_indices; ++i)
        {
            out_x[pos] = static_cast<double>(indices[i]);
            out_y[pos] = signal[indices[i]];
            ++pos;
        }

        // Mirror right
        for (int32_t i = 0; i < right_ext; ++i)
        {
            const int32_t src = n_indices - 2 - i;
            out_x[pos] = 2.0 * indices[n_indices - 1] - indices[src];
            out_y[pos] = signal[indices[src]];
            ++pos;
        }

        // Right boundary point
        if (need_right)
        {
            const double x0 = static_cast<double>(indices[n_indices - 2]);
            const double x1 = static_cast<double>(indices[n_indices - 1]);
            const double y0 = signal[indices[n_indices - 2]];
            const double y1 = signal[indices[n_indices - 1]];
            const double slope = (y1 - y0) / (x1 - x0);
            out_x[pos] = static_cast<double>(signal_len);
            out_y[pos] = y1 + slope * (signal_len - x1);
            ++pos;
        }

        out_count = pos;
    }

    // ============================================================================
    // Symmetric boundary extension (Rilling, Flandrin & Goncalves 2003; the
    // boundary_conditions() of emd.m, which the ICEEMDAN authors' reference
    // code uses). Mirrors nbsym extrema of each kind about the first/last
    // extremum, or about the end sample when that is the better symmetry
    // point, so both envelopes are interpolated, never extrapolated, over
    // [0, n-1]. Writes both envelopes' knots at once (the choice of symmetry
    // point depends on both kinds of extrema). Knot buffers need
    // n_max + 2 * nbsym + 2 (resp. n_min + ...) entries.
    // Returns false when there are not enough extrema (fewer than one of
    // each kind or fewer than three in total).
    // ============================================================================

    inline bool extend_extrema_symmetric(
        const int32_t *__restrict imax, int32_t n_max,
        const int32_t *__restrict imin, int32_t n_min,
        const double *__restrict x, int32_t n, int32_t nbsym,
        double *__restrict tmax, double *__restrict zmax, int32_t &n_tmax,
        double *__restrict tmin, double *__restrict zmin, int32_t &n_tmin)
    {
        n_tmax = n_tmin = 0;
        if (n_max < 1 || n_min < 1 || n_max + n_min < 3 || n < 3 || nbsym < 1)
            return false;

        const int32_t last = n - 1;
        // Index lists of the mirrored knots, in the order emd.m builds them
        // (fliplr: nearest the boundary last on the left, first on the right)
        int32_t lmax[64], lmin[64], rmax[64], rmin[64];
        int32_t n_lmax = 0, n_lmin = 0, n_rmax = 0, n_rmin = 0;
        nbsym = std::min(nbsym, 30);

        // fliplr(v(a:b)) in 0-based form: v[b], v[b-1], ..., v[a]
        auto take_rev = [](const int32_t *v, int32_t a, int32_t b, int32_t *out, int32_t &cnt)
        {
            for (int32_t i = b; i >= a; --i)
                out[cnt++] = v[i];
        };

        int32_t lsym, rsym;
        // ---- left edge ----
        if (imax[0] < imin[0])
        {
            if (x[0] > x[imin[0]])
            {
                take_rev(imax, 1, std::min(n_max - 1, nbsym), lmax, n_lmax);
                take_rev(imin, 0, std::min(n_min, nbsym) - 1, lmin, n_lmin);
                lsym = imax[0];
            }
            else
            {
                take_rev(imax, 0, std::min(n_max, nbsym) - 1, lmax, n_lmax);
                take_rev(imin, 0, std::min(n_min, nbsym - 1) - 1, lmin, n_lmin);
                lmin[n_lmin++] = 0;
                lsym = 0;
            }
        }
        else
        {
            if (x[0] < x[imax[0]])
            {
                take_rev(imax, 0, std::min(n_max, nbsym) - 1, lmax, n_lmax);
                take_rev(imin, 1, std::min(n_min - 1, nbsym), lmin, n_lmin);
                lsym = imin[0];
            }
            else
            {
                take_rev(imax, 0, std::min(n_max, nbsym - 1) - 1, lmax, n_lmax);
                lmax[n_lmax++] = 0;
                take_rev(imin, 0, std::min(n_min, nbsym) - 1, lmin, n_lmin);
                lsym = 0;
            }
        }

        // ---- right edge ----
        if (imax[n_max - 1] < imin[n_min - 1])
        {
            if (x[last] < x[imax[n_max - 1]])
            {
                take_rev(imax, std::max(n_max - nbsym, 0), n_max - 1, rmax, n_rmax);
                take_rev(imin, std::max(n_min - nbsym - 1, 0), n_min - 2, rmin, n_rmin);
                rsym = imin[n_min - 1];
            }
            else
            {
                rmax[n_rmax++] = last;
                take_rev(imax, std::max(n_max - nbsym + 1, 0), n_max - 1, rmax, n_rmax);
                take_rev(imin, std::max(n_min - nbsym, 0), n_min - 1, rmin, n_rmin);
                rsym = last;
            }
        }
        else
        {
            if (x[last] > x[imin[n_min - 1]])
            {
                take_rev(imax, std::max(n_max - nbsym - 1, 0), n_max - 2, rmax, n_rmax);
                take_rev(imin, std::max(n_min - nbsym, 0), n_min - 1, rmin, n_rmin);
                rsym = imax[n_max - 1];
            }
            else
            {
                take_rev(imax, std::max(n_max - nbsym, 0), n_max - 1, rmax, n_rmax);
                rmin[n_rmin++] = last;
                take_rev(imin, std::max(n_min - nbsym + 1, 0), n_min - 1, rmin, n_rmin);
                rsym = last;
            }
        }

        // ---- in case the symmetrized parts do not reach the ends ----
        auto first_mirror = [](int32_t sym, const int32_t *l, int32_t cnt)
        { return cnt > 0 ? 2.0 * sym - l[0] : -1.0; };
        auto last_mirror = [](int32_t sym, const int32_t *r, int32_t cnt)
        { return cnt > 0 ? 2.0 * sym - r[cnt - 1] : 1e300; };

        if (first_mirror(lsym, lmin, n_lmin) > 0.0 || first_mirror(lsym, lmax, n_lmax) > 0.0)
        {
            if (lsym == 0)
                return false; // emd.m: error('bug')
            if (lsym == imax[0])
            {
                n_lmax = 0;
                take_rev(imax, 0, std::min(n_max, nbsym) - 1, lmax, n_lmax);
            }
            else
            {
                n_lmin = 0;
                take_rev(imin, 0, std::min(n_min, nbsym) - 1, lmin, n_lmin);
            }
            lsym = 0;
        }
        if (last_mirror(rsym, rmin, n_rmin) < last || last_mirror(rsym, rmax, n_rmax) < last)
        {
            if (rsym == last)
                return false; // emd.m: error('bug')
            if (rsym == imax[n_max - 1])
            {
                n_rmax = 0;
                take_rev(imax, std::max(n_max - nbsym, 0), n_max - 1, rmax, n_rmax);
            }
            else
            {
                n_rmin = 0;
                take_rev(imin, std::max(n_min - nbsym, 0), n_min - 1, rmin, n_rmin);
            }
            rsym = last;
        }

        // ---- assemble [mirrored left, extrema, mirrored right] ----
        auto assemble = [&](const int32_t *l, int32_t nl, const int32_t *mid, int32_t nmid,
                            const int32_t *r, int32_t nr, double *t, double *z, int32_t &cnt)
        {
            cnt = 0;
            for (int32_t i = 0; i < nl; ++i)
            {
                t[cnt] = 2.0 * lsym - l[i];
                z[cnt++] = x[l[i]];
            }
            for (int32_t i = 0; i < nmid; ++i)
            {
                t[cnt] = static_cast<double>(mid[i]);
                z[cnt++] = x[mid[i]];
            }
            for (int32_t i = 0; i < nr; ++i)
            {
                t[cnt] = 2.0 * rsym - r[i];
                z[cnt++] = x[r[i]];
            }
            // knots must increase strictly (a mirrored end sample can coincide
            // with the first/last extremum's mirror only in degenerate cases)
            int32_t w = 1;
            for (int32_t i = 1; i < cnt; ++i)
            {
                if (t[i] > t[w - 1])
                {
                    t[w] = t[i];
                    z[w++] = z[i];
                }
            }
            cnt = (cnt > 0) ? w : 0;
        };
        assemble(lmax, n_lmax, imax, n_max, rmax, n_rmax, tmax, zmax, n_tmax);
        assemble(lmin, n_lmin, imin, n_min, rmin, n_rmin, tmin, zmin, n_tmin);
        return n_tmax >= 2 && n_tmin >= 2;
    }

    // ============================================================================
    // Envelope spline: the library's own (cubic_spline.hpp)
    //
    // Cubic (the default): not-a-knot ends, natural below 4 knots.
    // Akima: Akima's local spline; below 5 knots the cubic spline.
    // Linear: straight lines between the knots.
    // ============================================================================

    class EnvelopeSpline
    {
    public:
        EnvelopeSpline() = default;

        EnvelopeSpline(const EnvelopeSpline &) = delete;
        EnvelopeSpline &operator=(const EnvelopeSpline &) = delete;

        /**
         * Fits the spline through the knots (x strictly increasing). x and y
         * must stay as they are until the spline has been evaluated.
         */
        bool construct(const double *x, const double *y, int32_t n,
                       SplineMethod method = SplineMethod::Cubic)
        {
            if (method == SplineMethod::Linear)
                return spline_.construct_linear(x, y, n);
            if (method == SplineMethod::Akima && n >= 5)
                return spline_.construct_akima(x, y, n);
            return spline_.construct(x, y, n);
        }

        /** results[j] = spline at j, for j = 0 .. n_sites - 1. */
        bool evaluate_uniform(double *results, int32_t n_sites) const
        {
            return spline_.evaluate_uniform(results, n_sites);
        }

    private:
        CubicSpline spline_;
    };

    // ============================================================================
    // Sifter - Fused Loops, Raw Pointers
    // ============================================================================

    class Sifter
    {
    public:
        explicit Sifter(int32_t max_len, const EEMDConfig &cfg)
            : config_(cfg), max_len_(max_len), scratch_(max_len), work_(max_len), upper_env_(max_len), lower_env_(max_len)
        {
        }

        bool sift_imf(double *signal, double *imf, int32_t n)
        {
            std::memcpy(work_.data, signal, n * sizeof(double));

            // Raw pointers for hot path
            int32_t *p_max = scratch_.max_idx.data();
            int32_t *p_min = scratch_.min_idx.data();
            double *p_ext_x = scratch_.ext_x.data();
            double *p_ext_y = scratch_.ext_y.data();

            // S-number tracking: count consecutive iterations with stable extrema
            int32_t prev_n_extrema = -1;
            int32_t s_count = 0;

            for (int32_t iter = 0; iter < config_.max_sift_iters; ++iter)
            {
                // Find extrema (raw pointer output)
                find_extrema_raw(work_.data, n, p_max, scratch_.n_max, p_min, scratch_.n_min);

                const bool rilling = (config_.stop_rule == SiftStopRule::Rilling);

                // Enough extrema for envelopes: with the symmetric ends, three
                // in all (emd.m); otherwise two of each kind
                const bool enough = config_.symmetric_boundary
                                        ? (scratch_.n_max >= 1 && scratch_.n_min >= 1 &&
                                           scratch_.n_max + scratch_.n_min >= 3)
                                        : (scratch_.n_max >= 2 && scratch_.n_min >= 2);
                if (!enough)
                {
                    // Rilling (as emd.m): a mode that runs out of extrema
                    // mid-sift is kept as it stands
                    if (rilling && iter > 0)
                        break;
                    return false;
                }

                // S-number criterion: track consecutive stable extrema counts
                const int32_t n_extrema = scratch_.n_max + scratch_.n_min;
                if (!rilling && config_.s_number > 0)
                {
                    if (n_extrema == prev_n_extrema)
                    {
                        ++s_count;
                        if (s_count >= config_.s_number)
                        {
                            // Converged by S-number criterion
                            break;
                        }
                    }
                    else
                    {
                        s_count = 0;
                        prev_n_extrema = n_extrema;
                    }
                }

                // Knots of the upper envelope in (ext_x, ext_y), of the lower
                // one in (ext_x2, ext_y2)
                double *p_ext_x2 = scratch_.ext_x2.data();
                double *p_ext_y2 = scratch_.ext_y2.data();
                bool knots_ok = true;
                if (config_.symmetric_boundary)
                {
                    knots_ok = extend_extrema_symmetric(p_max, scratch_.n_max, p_min, scratch_.n_min,
                                                        work_.data, n, config_.boundary_extend,
                                                        p_ext_x, p_ext_y, scratch_.n_ext,
                                                        p_ext_x2, p_ext_y2, scratch_.n_ext2);
                }
                else
                {
                    extend_extrema_raw(p_max, scratch_.n_max, work_.data, n,
                                       config_.boundary_extend, p_ext_x, p_ext_y,
                                       scratch_.n_ext, scratch_.ext_start);
                    extend_extrema_raw(p_min, scratch_.n_min, work_.data, n,
                                       config_.boundary_extend, p_ext_x2, p_ext_y2,
                                       scratch_.n_ext2, scratch_.ext_start);
                }

                // The two envelopes share nothing (a spline, a knot set and an
                // output each). A long series sifted outside a fork, as the
                // circular noise bank's is, takes them on two threads; inside
                // a fork team_size() is 1 and they follow one another.
                bool env_ok[2] = {false, false};
                if (knots_ok)
                {
                    const int env_threads = (n >= envelope_fork_min) ? std::min(2, threads::team_size()) : 1;
                    threads::for_static(env_threads, 2, [&](int32_t first, int32_t last, int)
                    {
                        for (int32_t e = first; e < last; ++e)
                        {
                            env_ok[e] = (e == 0)
                                            ? upper_spline_.construct(p_ext_x, p_ext_y, scratch_.n_ext, config_.spline_method) &&
                                                  upper_spline_.evaluate_uniform(upper_env_.data, n)
                                            : lower_spline_.construct(p_ext_x2, p_ext_y2, scratch_.n_ext2, config_.spline_method) &&
                                                  lower_spline_.evaluate_uniform(lower_env_.data, n);
                        }
                    });
                }
                if (!(env_ok[0] && env_ok[1]))
                {
                    if (config_.symmetric_boundary && rilling && iter > 0)
                        break; // keep the mode as it stands (emd.m)
                    return false;
                }

                const double *__restrict upper = upper_env_.data;
                const double *__restrict lower = lower_env_.data;
                double *__restrict w = work_.data;

                if (rilling)
                {
                    if (rilling_should_stop(w, upper, lower, n, n_extrema, config_.rilling))
                        break;
                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        w[i] -= 0.5 * (upper[i] + lower[i]);
                    }
                    continue;
                }

                // Fused: compute mean, update work, calculate SD in single pass
                double sd_num = 0.0;
                double sd_den = 0.0;

                EEMD_OMP_SIMD_REDUCTION2(+, sd_num, sd_den)
                for (int32_t i = 0; i < n; ++i)
                {
                    const double u = upper[i];
                    const double l = lower[i];
                    const double val = w[i];
                    const double mean = 0.5 * (u + l);
                    const double diff = val - mean;

                    sd_num += mean * mean; // sum of mean^2
                    sd_den += val * val;   // sum of original^2

                    w[i] = diff; // update in-place
                }

                const double sd = (sd_den > 1e-15) ? sd_num / sd_den : 0.0;

                if (sd < config_.sift_threshold)
                    break;

                // Early termination: extrema ≈ zero crossings
                const int32_t n_zero = count_zero_crossings(work_.data, n);

                if (std::abs(n_extrema - n_zero) <= 1 && sd < config_.sift_threshold * 10)
                {
                    break;
                }
            }

            std::memcpy(imf, work_.data, n * sizeof(double));

            // Subtract IMF from signal
            double *__restrict sig = signal;
            const double *__restrict w = work_.data;

            EEMD_OMP_SIMD
            for (int32_t i = 0; i < n; ++i)
            {
                sig[i] -= w[i];
            }

            return true;
        }

    private:
        // Shortest series whose two envelopes are worth a fork
        static constexpr int32_t envelope_fork_min = 16384;

        const EEMDConfig &config_;
        int32_t max_len_;

        ThreadScratchPad scratch_;
        AlignedBuffer<double> work_;
        AlignedBuffer<double> upper_env_;
        AlignedBuffer<double> lower_env_;

        EnvelopeSpline upper_spline_;
        EnvelopeSpline lower_spline_;
    };

    // ============================================================================
    // EEMD Main Class
    // ============================================================================

    class EEMD
    {
    public:
        explicit EEMD(const EEMDConfig &config = EEMDConfig())
            : config_(config)
        {
        }

        bool decompose(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            int32_t &n_imfs)
        {
            if (n < 4)
                return false;

            // Compute signal stats
            double signal_mean = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_mean)
            for (int32_t i = 0; i < n; ++i)
            {
                signal_mean += signal[i];
            }
            signal_mean /= n;

            double signal_var = 0.0;
            EEMD_OMP_SIMD_REDUCTION(+, signal_var)
            for (int32_t i = 0; i < n; ++i)
            {
                const double d = signal[i] - signal_mean;
                signal_var += d * d;
            }
            const double signal_std = std::sqrt(signal_var / n);
            const double noise_amplitude = config_.noise_std * signal_std;

            const int32_t max_imfs = config_.max_imfs;

            // Global accumulator
            std::vector<AlignedBuffer<double>> global_sum(max_imfs);
            for (auto &buf : global_sum)
            {
                buf.resize(n);
                buf.zero();
            }

            int32_t global_max_imfs = 0;

            // Per-thread sums, reduced afterwards in thread order (not in the
            // order the threads finish), so a given thread count always
            // produces the same result bit for bit. Each trial's noise comes
            // from its own stream (trial_gaussian), so the noise of every
            // trial is the same at any thread count.
            const int32_t n_threads_used = threads::team_size();
            std::vector<std::vector<AlignedBuffer<double>>> thread_sums(n_threads_used);
            std::vector<int32_t> thread_max_imfs(n_threads_used, 0);

            // Thread tid takes the trials [trial_begin, trial_end): consecutive
            // ranges in thread order
            threads::for_static(n_threads_used, config_.ensemble_size,
                                [&](int32_t trial_begin, int32_t trial_end, int tid)
            {
                // Thread-local accumulator
                std::vector<AlignedBuffer<double>> &thread_sum = thread_sums[tid];
                thread_sum.resize(max_imfs);
                for (auto &buf : thread_sum)
                {
                    buf.resize(n);
                    buf.zero();
                }

                int32_t local_max_imfs = 0;

                // Thread-local buffers
                AlignedBuffer<double> noisy_signal(n);
                AlignedBuffer<double> noise(n);
                std::vector<AlignedBuffer<double>> local_imfs(max_imfs);
                for (auto &buf : local_imfs)
                {
                    buf.resize(n);
                }

                Sifter sifter(n, config_);

                for (int32_t trial = trial_begin; trial < trial_end; ++trial)
                {
                    // Generate this trial's noise (depends on seed and trial only)
                    if (!trial_gaussian(config_.rng_seed, static_cast<uint32_t>(trial),
                                        n, noise.data, 0.0, noise_amplitude))
                    {
                        noise.zero();
                    }

                    // Add noise
                    const double *__restrict sig = signal;
                    const double *__restrict nz = noise.data;
                    double *__restrict ns = noisy_signal.data;

                    EEMD_OMP_SIMD
                    for (int32_t i = 0; i < n; ++i)
                    {
                        ns[i] = sig[i] + nz[i];
                    }

                    // EMD
                    int32_t imf_count = 0;
                    for (int32_t k = 0; k < max_imfs; ++k)
                    {
                        if (!sifter.sift_imf(noisy_signal.data, local_imfs[k].data, n))
                        {
                            break;
                        }
                        ++imf_count;
                    }

                    local_max_imfs = std::max(local_max_imfs, imf_count);

                    // Thread-local accumulation
                    for (int32_t k = 0; k < imf_count; ++k)
                    {
                        double *__restrict ts = thread_sum[k].data;
                        const double *__restrict li = local_imfs[k].data;

                        EEMD_OMP_SIMD
                        for (int32_t i = 0; i < n; ++i)
                        {
                            ts[i] += li[i];
                        }
                    }
                }

                thread_max_imfs[tid] = local_max_imfs;
            });

            // Ordered reduction: element i sums threads 0, 1, 2, ... in turn
            threads::for_static(n_threads_used, n, [&](int32_t i_begin, int32_t i_end, int)
            {
                for (int32_t k = 0; k < max_imfs; ++k)
                {
                    double *__restrict gs = global_sum[k].data;
                    for (int32_t i = i_begin; i < i_end; ++i)
                    {
                        double acc = 0.0;
                        for (int32_t t = 0; t < n_threads_used; ++t)
                        {
                            acc += thread_sums[t][k].data[i];
                        }
                        gs[i] += acc;
                    }
                }
            });

            for (int32_t t = 0; t < n_threads_used; ++t)
            {
                global_max_imfs = std::max(global_max_imfs, thread_max_imfs[t]);
            }

            n_imfs = global_max_imfs;

            // Ensemble average
            const double scale = 1.0 / config_.ensemble_size;
            imfs.resize(n_imfs);

            for (int32_t k = 0; k < n_imfs; ++k)
            {
                imfs[k].resize(n);
                const double *__restrict gs = global_sum[k].data;
                double *__restrict out = imfs[k].data();

                EEMD_OMP_SIMD
                for (int32_t i = 0; i < n; ++i)
                {
                    out[i] = gs[i] * scale;
                }
            }

            return true;
        }

        bool decompose_emd(
            const double *signal,
            int32_t n,
            std::vector<std::vector<double>> &imfs,
            std::vector<double> &residue)
        {
            if (n < 4)
                return false;

            AlignedBuffer<double> work(n);
            std::memcpy(work.data, signal, n * sizeof(double));

            Sifter sifter(n, config_);

            imfs.clear();
            imfs.reserve(config_.max_imfs);

            for (int32_t k = 0; k < config_.max_imfs; ++k)
            {
                std::vector<double> imf(n);
                if (!sifter.sift_imf(work.data, imf.data(), n))
                {
                    break;
                }
                imfs.push_back(std::move(imf));
            }

            residue.resize(n);
            std::memcpy(residue.data(), work.data, n * sizeof(double));

            return true;
        }

        EEMDConfig &config() { return config_; }
        const EEMDConfig &config() const { return config_; }

    private:
        EEMDConfig config_;
    };

    // ============================================================================
    // Hilbert Transform - Instantaneous Frequency
    // ============================================================================

    /**
     * Hilbert transform y = H{x} (imaginary part of the analytic signal).
     * Any n >= 1; FFTs through eemd_fft.hpp (VectorFFT).
     */
    inline bool compute_hilbert_transform(const double *signal, int32_t n, double *hilbert_out)
    {
        return fft::hilbert(signal, n, hilbert_out);
    }

    /**
     * Instantaneous frequency (cycles per unit of 1/sample_rate) from the
     * phase increments of the analytic signal x + i H{x}.
     * inst_freq must hold n values; n >= 3.
     */
    inline bool compute_instantaneous_frequency(
        const double *imf,
        int32_t n,
        double *inst_freq,
        double sample_rate = 1.0)
    {
        if (n < 3)
            return false;

        thread_local std::vector<double> h;
        h.resize(static_cast<size_t>(n));
        if (!fft::hilbert(imf, n, h.data()))
            return false;

        const double freq_scale = sample_rate / (2.0 * M_PI);

        for (int32_t i = 1; i < n - 1; ++i)
        {
            double dphase = std::atan2(h[i + 1], imf[i + 1]) - std::atan2(h[i], imf[i]);

            while (dphase > M_PI)
                dphase -= 2.0 * M_PI;
            while (dphase < -M_PI)
                dphase += 2.0 * M_PI;

            inst_freq[i] = dphase * freq_scale;
        }

        inst_freq[0] = inst_freq[1];
        inst_freq[n - 1] = inst_freq[n - 2];

        return true;
    }

} // namespace eemd

#endif // EEMD_MKL_HPP
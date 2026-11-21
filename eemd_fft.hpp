/**
 * Real FFT backend for the EEMD/ICEEMDAN analysis utilities
 * (Hilbert transform, instantaneous frequency, spectral entropy).
 *
 * Backends:
 *   - VectorFFT (https://github.com/Tugbars/VectorFFT), when the build defines
 *     EEMD_USE_VECTORFFT and links VectorFFT::vfft (CMake does both by default
 *     when third_party/vectorfft is present).
 *   - Intel MKL DFTI otherwise, and for any length VectorFFT refuses.
 *
 * Plans are created once per (thread, length) and cached, so repeated calls on
 * the same length cost one transform each. VectorFFT plans are made with
 * VFFT_FLAG_ESTIMATE: nothing is measured and no wisdom file is read or
 * written from inside this library.
 *
 * Floating-point environment: vfft_create() leaves flush-to-zero and
 * denormals-are-zero set on the calling thread. Every VectorFFT call here runs
 * inside FpEnvGuard, which sets FTZ/DAZ for the kernels and restores the
 * caller's MXCSR afterwards, so the rest of the decomposition keeps the
 * caller's floating-point mode.
 *
 * License: MIT
 */

#ifndef EEMD_FFT_HPP
#define EEMD_FFT_HPP

#include <mkl_dfti.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#ifdef EEMD_USE_VECTORFFT
#include "vfft.h"
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define EEMD_FFT_X86 1
#endif

namespace eemd
{
    namespace fft
    {
        /**
         * Runs the enclosed calls with FTZ/DAZ set (the mode VectorFFT's
         * kernels are built for) and restores the caller's MXCSR on exit.
         */
        class FpEnvGuard
        {
        public:
            FpEnvGuard()
            {
#ifdef EEMD_FFT_X86
                saved_ = _mm_getcsr();
                _mm_setcsr(saved_ | 0x8040u); // FTZ (bit 15) | DAZ (bit 6)
#endif
            }
            ~FpEnvGuard()
            {
#ifdef EEMD_FFT_X86
                _mm_setcsr(saved_);
#endif
            }
            FpEnvGuard(const FpEnvGuard &) = delete;
            FpEnvGuard &operator=(const FpEnvGuard &) = delete;

        private:
#ifdef EEMD_FFT_X86
            unsigned int saved_ = 0;
#endif
        };

        enum class Backend : uint8_t
        {
            None = 0,
            VectorFFT = 1,
            MKL = 2
        };

        inline const char *backend_name(Backend b)
        {
            switch (b)
            {
            case Backend::VectorFFT:
                return "VectorFFT";
            case Backend::MKL:
                return "MKL DFTI";
            default:
                return "none";
            }
        }

        /**
         * One real transform length, forward (R2C) and backward (C2R).
         *
         * Spectrum format (both backends): bins 0..n/2, n/2 + 1 complex values
         * stored as interleaved (re, im) doubles. backward() is the
         * unnormalized inverse: backward(forward(x)) == n * x.
         */
        class RealFFT
        {
        public:
            RealFFT() = default;
            ~RealFFT() { release(); }
            RealFFT(const RealFFT &) = delete;
            RealFFT &operator=(const RealFFT &) = delete;

            /** Plan length n. Tries VectorFFT first (when built in), then MKL. */
            bool init(int32_t n)
            {
                release();
                if (n < 1)
                    return false;
                n_ = n;
#ifdef EEMD_USE_VECTORFFT
                if (init_vectorfft())
                {
                    backend_ = Backend::VectorFFT;
                    return true;
                }
#endif
                if (init_mkl())
                {
                    backend_ = Backend::MKL;
                    return true;
                }
                n_ = 0;
                return false;
            }

            int32_t size() const { return n_; }
            int32_t bins() const { return n_ / 2 + 1; }
            Backend backend() const { return backend_; }

            /** x: n reals -> X: (n/2 + 1) complex bins, interleaved. */
            bool forward(const double *x, double *X)
            {
                switch (backend_)
                {
#ifdef EEMD_USE_VECTORFFT
                case Backend::VectorFFT:
                {
                    FpEnvGuard env;
                    return vfft_execute(r2c_, VFFT_FORWARD, x, X) == 0;
                }
#endif
                case Backend::MKL:
                    return DftiComputeForward(dfti_, const_cast<double *>(x), X) == DFTI_NO_ERROR;
                default:
                    return false;
                }
            }

            /**
             * X: (n/2 + 1) complex bins -> x: n reals, times n. The imaginary
             * parts of bin 0 and (even n) bin n/2 must be zero.
             */
            bool backward(const double *X, double *x)
            {
                switch (backend_)
                {
#ifdef EEMD_USE_VECTORFFT
                case Backend::VectorFFT:
                {
                    FpEnvGuard env;
                    return vfft_execute(c2r_, VFFT_BACKWARD, X, x) == 0;
                }
#endif
                case Backend::MKL:
                    return DftiComputeBackward(dfti_, const_cast<double *>(X), x) == DFTI_NO_ERROR;
                default:
                    return false;
                }
            }

        private:
#ifdef EEMD_USE_VECTORFFT
            bool init_vectorfft()
            {
                FpEnvGuard env; // vfft_create sets FTZ/DAZ; keep it out of the caller's thread state
                vfft_config_t c;
                std::memset(&c, 0, sizeof(c));
                c.rank = 1;
                c.n[0] = n_;
                c.howmany = 1;
                c.layout = VFFT_LAYOUT_INTERLEAVED;
                c.placement = VFFT_PLACEMENT_OUT;
                c.order = VFFT_ORDER_NATURAL;
                c.batch = VFFT_BATCH_CONTIGUOUS;
                c.nthreads = 1; // called from inside OpenMP regions: no pool of its own
                c.flags = VFFT_FLAG_ESTIMATE;

                c.transform = VFFT_R2C;
                r2c_ = vfft_create(&c);
                if (!r2c_)
                    return false;
                c.transform = VFFT_C2R;
                c2r_ = vfft_create(&c);
                if (!c2r_)
                {
                    vfft_destroy(r2c_);
                    r2c_ = nullptr;
                    return false;
                }
                return true;
            }
#endif

            bool init_mkl()
            {
                if (DftiCreateDescriptor(&dfti_, DFTI_DOUBLE, DFTI_REAL, 1, static_cast<MKL_LONG>(n_)) != DFTI_NO_ERROR)
                {
                    dfti_ = nullptr;
                    return false;
                }
                bool ok = DftiSetValue(dfti_, DFTI_PLACEMENT, DFTI_NOT_INPLACE) == DFTI_NO_ERROR &&
                          DftiSetValue(dfti_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) == DFTI_NO_ERROR &&
                          DftiCommitDescriptor(dfti_) == DFTI_NO_ERROR;
                if (!ok)
                {
                    DftiFreeDescriptor(&dfti_);
                    dfti_ = nullptr;
                }
                return ok;
            }

            void release()
            {
#ifdef EEMD_USE_VECTORFFT
                if (r2c_)
                    vfft_destroy(r2c_);
                if (c2r_)
                    vfft_destroy(c2r_);
                r2c_ = nullptr;
                c2r_ = nullptr;
#endif
                if (dfti_)
                    DftiFreeDescriptor(&dfti_);
                dfti_ = nullptr;
                backend_ = Backend::None;
                n_ = 0;
            }

            int32_t n_ = 0;
            Backend backend_ = Backend::None;
            DFTI_DESCRIPTOR_HANDLE dfti_ = nullptr;
#ifdef EEMD_USE_VECTORFFT
            vfft_plan r2c_ = nullptr;
            vfft_plan c2r_ = nullptr;
#endif
        };

        /**
         * The calling thread's plan for length n (created on first use, kept
         * for the thread's lifetime; the 8 most recent lengths are cached).
         * Returns nullptr when no backend serves n.
         */
        inline RealFFT *thread_plan(int32_t n)
        {
            struct Cache
            {
                std::vector<std::unique_ptr<RealFFT>> plans; // most recent last
            };
            thread_local Cache cache;

            for (size_t i = cache.plans.size(); i-- > 0;)
            {
                if (cache.plans[i]->size() == n)
                    return cache.plans[i].get();
            }
            std::unique_ptr<RealFFT> plan(new RealFFT());
            if (!plan->init(n))
                return nullptr;
            if (cache.plans.size() >= 8)
                cache.plans.erase(cache.plans.begin());
            cache.plans.push_back(std::move(plan));
            return cache.plans.back().get();
        }

        /** Which backend serves length n on this thread (for logs and tests). */
        inline Backend backend_for(int32_t n)
        {
            RealFFT *p = thread_plan(n);
            return p ? p->backend() : Backend::None;
        }

        /**
         * Hilbert transform of a real signal: y = H{x}, the imaginary part of
         * the analytic signal x + i y. Computed with one R2C and one C2R:
         * Y[k] = -i X[k] for 0 < k < n/2, Y[0] = 0 and (even n) Y[n/2] = 0.
         * Works for any n >= 1, even or odd.
         */
        inline bool hilbert(const double *x, int32_t n, double *y)
        {
            if (n < 1)
                return false;
            RealFFT *plan = thread_plan(n);
            if (!plan)
                return false;

            const int32_t nb = plan->bins();
            thread_local std::vector<double> spec;
            spec.resize(2 * static_cast<size_t>(nb));
            double *X = spec.data();

            if (!plan->forward(x, X))
                return false;

            // Positive frequencies 1 .. (n-1)/2 have a negative partner; the
            // Hilbert multiplier there is -i (multiply by -i: (a, b) -> (b, -a)).
            const int32_t last_paired = (n - 1) / 2;
            X[0] = 0.0;
            X[1] = 0.0;
            for (int32_t k = 1; k <= last_paired; ++k)
            {
                const double re = X[2 * k];
                const double im = X[2 * k + 1];
                X[2 * k] = im;
                X[2 * k + 1] = -re;
            }
            if (n % 2 == 0 && nb - 1 > last_paired)
            {
                X[2 * (nb - 1)] = 0.0; // Nyquist bin: no quadrature partner
                X[2 * (nb - 1) + 1] = 0.0;
            }

            if (!plan->backward(X, y))
                return false;
            const double scale = 1.0 / n;
            for (int32_t i = 0; i < n; ++i)
                y[i] *= scale;
            return true;
        }

        /**
         * One-sided power spectrum |X[k]|^2 for k = 0 .. n/2 of a real signal.
         * psd must hold n/2 + 1 doubles.
         */
        inline bool power_spectrum(const double *x, int32_t n, double *psd)
        {
            RealFFT *plan = thread_plan(n);
            if (!plan)
                return false;
            const int32_t nb = plan->bins();
            thread_local std::vector<double> spec;
            spec.resize(2 * static_cast<size_t>(nb));
            if (!plan->forward(x, spec.data()))
                return false;
            for (int32_t k = 0; k < nb; ++k)
            {
                const double re = spec[2 * k];
                const double im = spec[2 * k + 1];
                psd[k] = re * re + im * im;
            }
            return true;
        }

    } // namespace fft
} // namespace eemd

#endif // EEMD_FFT_HPP

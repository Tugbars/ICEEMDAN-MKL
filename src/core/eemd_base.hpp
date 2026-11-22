/**
 * What every part of the library stands on: vectorization hints, a test for
 * finite numbers that survives -ffast-math, and 64-byte aligned buffers.
 *
 * License: MIT
 */

#ifndef EEMD_BASE_HPP
#define EEMD_BASE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

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

// One copy of a function's machine code for every caller: decompose() and
// decompose_batch() must compute the same bits, and under -ffast-math two
// inlined copies of the same code may sum in different orders
#if defined(_MSC_VER) && !defined(__clang__)
#define EEMD_NOINLINE __declspec(noinline)
#else
#define EEMD_NOINLINE __attribute__((noinline))
#endif

// Buffers are aligned to a cache line
#define EEMD_CACHE_LINE 64

namespace eemd
{

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

} // namespace eemd

#endif // EEMD_BASE_HPP

/**
 * The extrema of a signal and the knots of its envelopes: local maxima and
 * minima (flat ones included), zero crossings, and the extension of the
 * extrema past both ends, as emd.m does it (symmetric) or by mirroring.
 *
 * License: MIT
 */

#ifndef EEMD_EXTREMA_HPP
#define EEMD_EXTREMA_HPP

#include "eemd_base.hpp"

namespace eemd
{

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

} // namespace eemd

#endif // EEMD_EXTREMA_HPP

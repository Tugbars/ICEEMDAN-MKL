/**
 * Envelope spline tests (cubic_spline.hpp). The cubic spline against a dense
 * solve of the same equations in extended precision, on random knot sets;
 * the special cases of 2 and 3 knots; samples outside the knots; and that
 * the value at a sample does not depend on how many samples are asked for.
 * Akima's spline against the paper's formula evaluated the slow way, a line
 * reproduced exactly, no overshoot at a step, and locality. The straight
 * lines against exact ones.
 *
 * Exits non-zero on any failure.
 */

#include "cubic_spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *what, double value, double limit)
{
    std::printf("  %-62s %s  %.2e (limit %.0e)\n", what, ok ? "PASS" : "FAIL", value, limit);
    if (!ok)
        ++g_failures;
}

static void check(bool ok, const char *what)
{
    std::printf("  %-62s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

// The spline's own definition, solved the slow way: slopes from the full
// system by Gaussian elimination with pivoting, in long double.
static void reference(const double *x, const double *y, int k, long double *out, int n)
{
    std::vector<long double> h(k - 1), d(k - 1), s(k);
    for (int i = 0; i < k - 1; ++i)
    {
        h[i] = static_cast<long double>(x[i + 1]) - x[i];
        d[i] = (static_cast<long double>(y[i + 1]) - y[i]) / h[i];
    }
    if (k == 2)
    {
        s[0] = s[1] = d[0];
    }
    else if (k == 3) // not-a-knot at 3 knots: the parabola through them
    {
        const long double c = (d[1] - d[0]) / (h[0] + h[1]);
        s[0] = d[0] - c * h[0];
        s[1] = d[0] + c * h[0];
        s[2] = d[1] + c * h[1];
    }
    else
    {
        std::vector<std::vector<long double>> a(k, std::vector<long double>(k + 1, 0.0L));
        // not-a-knot: the third derivative is continuous at the second and the last-but-one knot
        {
            const long double left = h[0] + h[1];
            a[0][0] = h[1];
            a[0][1] = left;
            a[0][k] = ((3 * h[0] + 2 * h[1]) * h[1] * d[0] + h[0] * h[0] * d[1]) / left;
            const long double right = h[k - 3] + h[k - 2];
            a[k - 1][k - 2] = right;
            a[k - 1][k - 1] = h[k - 3];
            a[k - 1][k] = (h[k - 2] * h[k - 2] * d[k - 3] + (2 * h[k - 3] + 3 * h[k - 2]) * h[k - 3] * d[k - 2]) / right;
        }
        for (int i = 1; i < k - 1; ++i)
        {
            a[i][i - 1] = h[i];
            a[i][i] = 2 * (h[i - 1] + h[i]);
            a[i][i + 1] = h[i - 1];
            a[i][k] = 3 * (h[i] * d[i - 1] + h[i - 1] * d[i]);
        }
        for (int c = 0; c < k; ++c)
        {
            int pivot = c;
            for (int r = c + 1; r < k; ++r)
                if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
                    pivot = r;
            std::swap(a[c], a[pivot]);
            for (int r = c + 1; r < k; ++r)
            {
                const long double f = a[r][c] / a[c][c];
                if (f != 0)
                    for (int q = c; q <= k; ++q)
                        a[r][q] -= f * a[c][q];
            }
        }
        for (int r = k - 1; r >= 0; --r)
        {
            long double v = a[r][k];
            for (int q = r + 1; q < k; ++q)
                v -= a[r][q] * s[q];
            s[r] = v / a[r][r];
        }
    }
    for (int j = 0; j < n; ++j)
    {
        int i = 0;
        while (i < k - 2 && static_cast<long double>(j) >= x[i + 1])
            ++i;
        const long double t = static_cast<long double>(j) - x[i];
        const long double c2 = (3 * d[i] - 2 * s[i] - s[i + 1]) / h[i];
        const long double c3 = (s[i] + s[i + 1] - 2 * d[i]) / (h[i] * h[i]);
        out[j] = y[i] + t * (s[i] + t * (c2 + t * c3));
    }
}

// Akima's spline the slow way: the slopes by the formula of Akima (1970) in long double, the
// values from the Hermite basis functions
static void akima_reference(const double *x, const double *y, int k, long double *out, int n)
{
    std::vector<long double> chords(k + 3);
    auto m = [&](int j) -> long double & { return chords[j + 2]; }; // j = -2 .. k
    for (int j = 0; j < k - 1; ++j)
        m(j) = (static_cast<long double>(y[j + 1]) - y[j]) / (static_cast<long double>(x[j + 1]) - x[j]);
    m(-1) = 2 * m(0) - m(1);
    m(-2) = 2 * m(-1) - m(0);
    m(k - 1) = 2 * m(k - 2) - m(k - 3);
    m(k) = 2 * m(k - 1) - m(k - 2);
    std::vector<long double> t(k);
    for (int i = 0; i < k; ++i)
    {
        const long double a = std::fabs(m(i + 1) - m(i)), b = std::fabs(m(i - 1) - m(i - 2));
        t[i] = (a + b == 0) ? (m(i - 1) + m(i)) / 2 : (a * m(i - 1) + b * m(i)) / (a + b);
    }
    for (int j = 0; j < n; ++j)
    {
        int i = 0;
        while (i < k - 2 && static_cast<long double>(j) >= x[i + 1])
            ++i;
        const long double h = static_cast<long double>(x[i + 1]) - x[i];
        const long double u = (j - static_cast<long double>(x[i])) / h;
        const long double h00 = 2 * u * u * u - 3 * u * u + 1, h10 = u * u * u - 2 * u * u + u;
        const long double h01 = -2 * u * u * u + 3 * u * u, h11 = u * u * u - u * u;
        out[j] = h00 * y[i] + h10 * h * t[i] + h01 * y[i + 1] + h11 * h * t[i + 1];
    }
}

int main()
{
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    eemd::CubicSpline spline;

    std::printf("Cubic spline against a dense solve in extended precision\n");
    {
        double worst_inside = 0, worst_outside = 0, worst_knot = 0;
        bool all_built = true;
        for (int trial = 0; trial < 3000; ++trial)
        {
            const int k = 2 + static_cast<int>(rng() % (trial % 10 == 0 ? 120 : 40));
            const int n = 1 + static_cast<int>(rng() % 900);
            // integer knots as the envelopes have them, starting before, at or after sample 0
            std::vector<double> x(k), y(k);
            const int max_step = 1 + static_cast<int>(rng() % 12);
            double pos = static_cast<double>(static_cast<int>(rng() % 41) - 30);
            for (int i = 0; i < k; ++i)
            {
                x[i] = pos;
                pos += 1 + static_cast<double>(rng() % max_step);
                y[i] = u(rng) * (1.0 + (trial % 5)) + (trial % 3 == 0 ? 100.0 : 0.0);
            }
            std::vector<double> out(n);
            std::vector<long double> ref(n);
            if (!spline.construct(x.data(), y.data(), k) || !spline.evaluate_uniform(out.data(), n))
            {
                all_built = false;
                continue;
            }
            reference(x.data(), y.data(), k, ref.data(), n);
            double scale = 0;
            for (double v : y)
                scale = std::max(scale, std::abs(v));
            for (int j = 0; j < n; ++j)
            {
                const double ref_j = static_cast<double>(ref[j]);
                if (j >= x[0] && j <= x[k - 1])
                    worst_inside = std::max(worst_inside, std::abs(out[j] - ref_j) / scale);
                else // a continued end piece grows: measure against its own size
                    worst_outside = std::max(worst_outside, std::abs(out[j] - ref_j) / std::max(scale, std::abs(ref_j)));
            }
            for (int i = 0; i < k; ++i)
                if (x[i] >= 0 && x[i] < n)
                    worst_knot = std::max(worst_knot, std::abs(out[static_cast<int>(x[i])] - y[i]) / scale);
        }
        check(all_built, "3000 random knot sets (2 to 121 knots) build and evaluate");
        check(worst_inside < 1e-11, "samples between the knots", worst_inside, 1e-11);
        check(worst_outside < 1e-8, "samples outside the knots (end pieces continued)", worst_outside, 1e-8);
        check(worst_knot < 1e-12, "the spline passes through its knots", worst_knot, 1e-12);
    }

    std::printf("Few knots\n");
    {
        // 2 knots: the straight line through them
        const double x2[] = {3, 11}, y2[] = {1.5, -2.5};
        std::vector<double> out(16);
        spline.construct(x2, y2, 2);
        spline.evaluate_uniform(out.data(), 16);
        double err = 0;
        for (int j = 0; j < 16; ++j)
            err = std::max(err, std::abs(out[j] - (1.5 + (j - 3) * (-4.0 / 8.0))));
        check(err < 1e-13, "2 knots: a straight line, continued on both sides", err, 1e-13);

        // 3 knots: not-a-knot is the parabola through them (MATLAB's spline, SciPy's CubicSpline)
        const double x3[] = {0, 4, 9}, y3[] = {0.0, 2.0, -1.0};
        std::vector<double> o3(14);
        spline.construct(x3, y3, 3);
        spline.evaluate_uniform(o3.data(), 14);
        // y = a x + b x^2 through (4, 2) and (9, -1): 4a + 16b = 2, 9a + 81b = -1
        const double b = (-1.0 - 9.0 * 0.5) / (81.0 - 36.0), a = 0.5 - 4.0 * b;
        err = 0;
        for (int j = 0; j < 14; ++j)
            err = std::max(err, std::abs(o3[j] - (a * j + b * j * j)));
        check(err < 1e-13, "3 knots: the parabola through them, continued on both sides", err, 1e-13);

        // 4 knots, not-a-knot: one cubic through all four
        const double x4[] = {0, 3, 5, 9};
        double y4[4];
        for (int i = 0; i < 4; ++i)
            y4[i] = 0.5 + 0.25 * x4[i] - 0.125 * x4[i] * x4[i] + 0.01 * x4[i] * x4[i] * x4[i];
        std::vector<double> o4(12);
        spline.construct(x4, y4, 4);
        spline.evaluate_uniform(o4.data(), 12);
        err = 0;
        for (int j = 0; j < 12; ++j)
            err = std::max(err, std::abs(o4[j] - (0.5 + 0.25 * j - 0.125 * j * j + 0.01 * j * j * j)));
        check(err < 1e-12, "4 knots on a cubic: not-a-knot ends return that cubic", err, 1e-12);
    }

    std::printf("Evaluation\n");
    {
        const int k = 57;
        std::vector<double> x(k), y(k);
        double pos = -7;
        for (int i = 0; i < k; ++i)
        {
            x[i] = pos;
            pos += 1 + static_cast<double>(rng() % 6);
            y[i] = u(rng);
        }
        const int n = static_cast<int>(pos) + 9; // runs past the last knot
        std::vector<double> full(n), part(n);
        spline.construct(x.data(), y.data(), k);
        spline.evaluate_uniform(full.data(), n);
        bool same = true;
        for (int m = 1; m <= n; ++m)
        {
            spline.evaluate_uniform(part.data(), m);
            for (int j = 0; j < m; ++j)
                same = same && part[j] == full[j];
        }
        check(same, "a sample's value does not depend on how many are asked for");

        eemd::CubicSpline fresh;
        std::vector<double> o(4);
        const double xb[] = {0, 2, 2, 5}, yb[] = {0, 1, 2, 3};
        check(!fresh.evaluate_uniform(o.data(), 4), "evaluating before construct() is refused");
        check(!fresh.construct(xb, yb, 1), "one knot is refused");
        check(!fresh.construct(xb, yb, 4), "knots that do not increase are refused");
    }

    std::printf("Akima's spline\n");
    {
        double worst = 0, worst_knot = 0;
        bool all_built = true;
        for (int trial = 0; trial < 3000; ++trial)
        {
            const int k = 3 + static_cast<int>(rng() % (trial % 10 == 0 ? 120 : 40));
            const int n = 1 + static_cast<int>(rng() % 900);
            std::vector<double> x(k), y(k);
            const int max_step = 1 + static_cast<int>(rng() % 12);
            double pos = static_cast<double>(static_cast<int>(rng() % 41) - 30);
            for (int i = 0; i < k; ++i)
            {
                x[i] = pos;
                pos += 1 + static_cast<double>(rng() % max_step);
                y[i] = u(rng) * (1.0 + (trial % 5)) + (trial % 3 == 0 ? 100.0 : 0.0);
            }
            std::vector<double> out(n);
            std::vector<long double> ref(n);
            if (!spline.construct_akima(x.data(), y.data(), k) || !spline.evaluate_uniform(out.data(), n))
            {
                all_built = false;
                continue;
            }
            akima_reference(x.data(), y.data(), k, ref.data(), n);
            double scale = 0;
            for (double v : y)
                scale = std::max(scale, std::abs(v));
            for (int j = 0; j < n; ++j)
                if (j >= x[0] && j <= x[k - 1])
                    worst = std::max(worst, std::abs(out[j] - static_cast<double>(ref[j])) / scale);
            for (int i = 0; i < k; ++i)
                if (x[i] >= 0 && x[i] < n)
                    worst_knot = std::max(worst_knot, std::abs(out[static_cast<int>(x[i])] - y[i]) / scale);
        }
        check(all_built, "3000 random knot sets (3 to 122 knots) build and evaluate");
        check(worst < 1e-11, "against the formula of the paper in long double", worst, 1e-11);
        check(worst_knot < 1e-12, "the spline passes through its knots", worst_knot, 1e-12);

        // a straight line comes back exactly as a straight line
        const double xl[] = {0, 3, 4, 9, 15, 16, 22};
        double yl[7];
        for (int i = 0; i < 7; ++i)
            yl[i] = 0.75 * xl[i] - 2.0;
        std::vector<double> ol(23);
        spline.construct_akima(xl, yl, 7);
        spline.evaluate_uniform(ol.data(), 23);
        double err = 0;
        for (int j = 0; j < 23; ++j)
            err = std::max(err, std::abs(ol[j] - (0.75 * j - 2.0)));
        check(err < 1e-13, "knots on a line: the line", err, 1e-13);

        // a step: no overshoot, and the flat parts stay flat (Akima's point)
        const double xs[] = {0, 10, 20, 30, 40, 50, 60, 70};
        const double ys[] = {0, 0, 0, 0, 1, 1, 1, 1};
        std::vector<double> os(71), oc(71);
        spline.construct_akima(xs, ys, 8);
        spline.evaluate_uniform(os.data(), 71);
        bool flat = true;
        double lo = 0, hi = 1;
        for (int j = 0; j <= 70; ++j)
        {
            lo = std::min(lo, os[j]);
            hi = std::max(hi, os[j]);
            flat = flat && (j > 30 || os[j] == 0.0) && (j < 40 || os[j] == 1.0);
        }
        check(lo == 0.0 && hi == 1.0 && flat, "a step: within [0, 1], flat on both sides");
        spline.construct(xs, ys, 8);
        spline.evaluate_uniform(oc.data(), 71);
        const double over = std::max(-*std::min_element(oc.begin(), oc.end()), *std::max_element(oc.begin(), oc.end()) - 1);
        std::printf("  (the cubic spline overshoots the same step by %.3f)\n", over);

        // local: moving one knot changes the curve only from three knots before it to three after
        const int k = 40, target = 20;
        std::vector<double> x(k), y(k);
        double pos = 0;
        for (int i = 0; i < k; ++i)
        {
            x[i] = pos;
            pos += 1 + static_cast<double>(rng() % 9);
            y[i] = u(rng);
        }
        const int n = static_cast<int>(x[k - 1]) + 1;
        std::vector<double> before(n), after(n);
        spline.construct_akima(x.data(), y.data(), k);
        spline.evaluate_uniform(before.data(), n);
        y[target] += 0.5;
        spline.construct_akima(x.data(), y.data(), k);
        spline.evaluate_uniform(after.data(), n);
        bool same_far = true, changed_near = false;
        for (int j = 0; j < n; ++j)
        {
            if (j < x[target - 3] || j >= x[target + 3])
                same_far = same_far && before[j] == after[j];
            else
                changed_near = changed_near || before[j] != after[j];
        }
        check(same_far && changed_near, "one knot moved: the curve beyond three knots unchanged, bit for bit");

        eemd::CubicSpline fresh;
        const double xb[] = {0, 2, 2, 5}, yb[] = {0, 1, 2, 3};
        check(!fresh.construct_akima(xb, yb, 1) && !fresh.construct_akima(xb, yb, 4),
              "one knot, and knots that do not increase, are refused");
    }

    std::printf("Straight lines\n");
    {
        double worst = 0;
        for (int trial = 0; trial < 2000; ++trial)
        {
            const int k = 2 + static_cast<int>(rng() % 60);
            const int n = 1 + static_cast<int>(rng() % 700);
            std::vector<double> x(k), y(k), out(n);
            double pos = static_cast<double>(static_cast<int>(rng() % 21) - 10);
            for (int i = 0; i < k; ++i)
            {
                x[i] = pos;
                pos += 1 + static_cast<double>(rng() % 8);
                y[i] = u(rng);
            }
            spline.construct_linear(x.data(), y.data(), k);
            spline.evaluate_uniform(out.data(), n);
            for (int j = 0; j < n; ++j)
            {
                int i = 0; // the piece; the first and the last continued outside the knots
                while (i < k - 2 && j >= x[i + 1])
                    ++i;
                const long double exact = y[i] + (static_cast<long double>(y[i + 1]) - y[i]) /
                                                      (static_cast<long double>(x[i + 1]) - x[i]) * (j - x[i]);
                worst = std::max(worst, std::abs(out[j] - static_cast<double>(exact)) /
                                            std::max(1.0, std::abs(static_cast<double>(exact))));
            }
        }
        check(worst < 1e-13, "2000 random knot sets against exact straight lines", worst, 1e-13);
    }

    std::printf("Two splines built together (construct_pair)\n");
    {
        // the upper and lower envelope of a sifting step: knot counts equal or a few apart
        eemd::CubicSpline a, b, a1, b1;
        const auto knots = [&](int k, std::vector<double> &x, std::vector<double> &y)
        {
            x.resize(k);
            y.resize(k);
            double pos = static_cast<double>(static_cast<int>(rng() % 9) - 6);
            for (int i = 0; i < k; ++i)
            {
                x[i] = pos;
                pos += 1 + static_cast<double>(rng() % 5);
                y[i] = u(rng);
            }
        };
        bool same = true, built = true;
        for (int trial = 0; trial < 2000; ++trial)
        {
            const int ka = 2 + static_cast<int>(rng() % 60);
            const int kb = std::max(2, ka + static_cast<int>(rng() % 7) - 3);
            std::vector<double> xa, ya, xb, yb;
            knots(ka, xa, ya);
            knots(kb, xb, yb);
            const int n = 1 + static_cast<int>(std::max(xa[ka - 1], xb[kb - 1])) + static_cast<int>(rng() % 5);
            std::vector<double> pa(n), pb(n), sa(n), sb(n);
            const bool ok = eemd::CubicSpline::construct_pair(a, xa.data(), ya.data(), ka, b, xb.data(), yb.data(), kb) &&
                            a.evaluate_uniform(pa.data(), n) && b.evaluate_uniform(pb.data(), n);
            const bool ok1 = a1.construct(xa.data(), ya.data(), ka) && b1.construct(xb.data(), yb.data(), kb) &&
                             a1.evaluate_uniform(sa.data(), n) && b1.evaluate_uniform(sb.data(), n);
            built = built && ok && ok1;
            same = same && std::memcmp(pa.data(), sa.data(), n * sizeof(double)) == 0 &&
                   std::memcmp(pb.data(), sb.data(), n * sizeof(double)) == 0;
        }
        check(built, "2000 random pairs (2 to 64 knots, counts up to 3 apart) build and evaluate");
        check(same, "each spline of a pair equals construct() alone, bit for bit");

        const double xg[] = {0, 2, 4, 7, 9}, yg[] = {1, -1, 2, 0, 1};
        const double xbad[] = {0, 2, 2, 5, 8}, ybad[] = {0, 1, 2, 3, 4};
        check(!eemd::CubicSpline::construct_pair(a, xg, yg, 5, b, xbad, ybad, 5) &&
                  !eemd::CubicSpline::construct_pair(a, xbad, ybad, 5, b, xg, yg, 5),
              "a pair with knots that do not increase in either is refused");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

/**
 * EMD and EEMD: the engine (emd_engine.hpp) and the C++ interface (emd.hpp).
 * Completeness, the simple cases, the same bits at any thread count, what
 * EEMD's IMFs and residue add up to, batches and refusals.
 *
 * Exits non-zero on any failure.
 */

#include "emd.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %-72s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

static void check(bool ok, const char *what, double value, double limit)
{
    std::printf("  %-52s %s  %.2e (limit %.0e)\n", what, ok ? "PASS" : "FAIL", value, limit);
    if (!ok)
        ++g_failures;
}

static bool same(const emd::Decomposition &a, const emd::Decomposition &b)
{
    if (a.imfs.size() != b.imfs.size() || a.residue.size() != b.residue.size())
        return false;
    for (size_t j = 0; j < a.imfs.size(); ++j)
        if (std::memcmp(a.imfs[j].data(), b.imfs[j].data(), a.imfs[j].size() * sizeof(double)) != 0)
            return false;
    return std::memcmp(a.residue.data(), b.residue.data(), a.residue.size() * sizeof(double)) == 0;
}

/** The largest |IMFs + residue - x - offset|, relative to the largest |x|. */
static double sum_error(const std::vector<double> &x, const emd::Decomposition &r, const std::vector<double> *offset = nullptr)
{
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double sum = r.residue[i];
        for (const auto &imf : r.imfs)
            sum += imf[i];
        worst = std::max(worst, std::abs(sum - x[i] - (offset ? (*offset)[i] : 0.0)));
        scale = std::max(scale, std::abs(x[i]));
    }
    return worst / scale;
}

template <class Call>
static bool refused(Call &&call, const char *text_has = "")
{
    try
    {
        call();
    }
    catch (const std::invalid_argument &e)
    {
        return std::strstr(e.what(), text_has) != nullptr;
    }
    catch (...)
    {
    }
    return false;
}

static emd::Config eemd_of(int copies, unsigned seed = 0, int nthreads = 0)
{
    emd::Config c;
    c.ensemble_size = copies;
    c.seed = seed;
    c.nthreads = nthreads;
    return c;
}

int main()
{
    const int n = 900, count = 6;
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<std::vector<double>> channels(count, std::vector<double>(n));
    for (int c = 0; c < count; ++c)
        for (int i = 0; i < n; ++i)
            channels[c][i] = std::sin(0.31 * i + c) + 0.6 * std::sin(0.047 * (1 + c % 3) * i) + 0.002 * i +
                             0.1 * g(rng);
    const std::vector<double> &x = channels[0];

    // ------------------------------------------------------------------
    std::printf("EMD\n");
    emd::Decomposer plain;
    const emd::Decomposition r = plain.decompose(x);
    check(plain.ensemble_size() == 0 && plain.threads() == emd::Decomposer::processors(),
          "a decomposer with no settings is plain EMD, on every processor");
    check(r.imfs.size() >= 2 && r.residue.size() == static_cast<size_t>(n) && r.repaired == 0,
          "a signal of two rhythms, a trend and noise: several IMFs");
    check(sum_error(x, r) < 1e-12, "the IMFs and the residue add up to the signal", sum_error(x, r), 1e-12);
    check(same(plain.decompose(x.data(), n), r), "again: the same, bit for bit");
    {
        // every IMF oscillates slower than the one before it
        std::vector<int> crossings;
        for (const auto &imf : r.imfs)
            crossings.push_back(eemd::count_zero_crossings(imf.data(), n));
        bool slower = true;
        for (size_t j = 1; j < crossings.size(); ++j)
            slower = slower && crossings[j] < crossings[j - 1];
        check(slower, "each IMF has fewer zero crossings than the one before");
    }
    {
        std::vector<double> ramp(n), tone(n);
        for (int i = 0; i < n; ++i)
        {
            ramp[i] = 3.0 + 0.01 * i;
            tone[i] = std::sin(0.2 * i);
        }
        const emd::Decomposition flat = plain.decompose(ramp);
        check(flat.imfs.empty() && flat.residue == ramp, "a ramp: no IMF, the residue is the ramp itself");
        const emd::Decomposition one = plain.decompose(tone);
        double leftover = 0.0;
        for (double v : one.residue)
            leftover = std::max(leftover, std::abs(v));
        check(one.imfs.size() == 1 && leftover < 1e-6, "a tone: one IMF, nothing left");
    }
    {
        emd::Config c;
        c.max_imfs = 2;
        const emd::Decomposition two = emd::Decomposer(c).decompose(x);
        check(two.imfs.size() == 2 && same(emd::Decomposition{{r.imfs[0], r.imfs[1]}, two.residue, 0}, two) &&
                  sum_error(x, two) < 1e-12,
              "max_imfs 2: the first two IMFs, the rest in the residue");
    }

    // ------------------------------------------------------------------
    std::printf("EEMD\n");
    const emd::Decomposition e1 = emd::Decomposer(eemd_of(30, 0, 1)).decompose(x);
    check(e1.imfs.size() >= 2, "30 noisy copies averaged: several IMFs");
    bool any_thread = true;
    for (int threads : {2, 3, 5, 8})
        any_thread = any_thread && same(emd::Decomposer(eemd_of(30, 0, threads)).decompose(x), e1);
    check(any_thread, "the same bits at 1, 2, 3, 5 and 8 threads");
    {
        // IMFs + residue = the signal + the mean of the noise that was added
        const double amplitude = 0.2 * eemd::compute_sample_std(x.data(), n);
        std::vector<double> mean_noise(n, 0.0), noise(n);
        for (int copy = 0; copy < 30; ++copy)
        {
            eemd::trial_gaussian(42, static_cast<uint32_t>(copy), n, noise.data(), 0.0, amplitude);
            for (int i = 0; i < n; ++i)
                mean_noise[i] += noise[i] / 30.0;
        }
        check(sum_error(x, e1, &mean_noise) < 1e-12, "IMFs + residue = signal + mean of the added noise",
              sum_error(x, e1, &mean_noise), 1e-12);
        check(sum_error(x, e1) > 1e-4, "(and not the signal alone: the known gap of EEMD)");
    }
    check(same(emd::Decomposer(eemd_of(30, 42)).decompose(x), e1) &&
              !same(emd::Decomposer(eemd_of(30, 7)).decompose(x), e1),
          "seed 42 is the default; another seed gives another result");
    {
        emd::Config quiet = eemd_of(30);
        quiet.noise = 0.05;
        const emd::Decomposition q = emd::Decomposer(quiet).decompose(x);
        check(!same(q, e1) && sum_error(x, q) < sum_error(x, e1), "noise 0.05: another result, a smaller gap");
    }

    // ------------------------------------------------------------------
    std::printf("Many signals, NaN and Inf\n");
    {
        std::vector<double> rows(static_cast<size_t>(count) * n);
        for (int s = 0; s < count; ++s)
            std::copy(channels[s].begin(), channels[s].end(), rows.begin() + static_cast<size_t>(s) * n);
        for (int copies : {0, 20})
        {
            emd::Decomposer batch(eemd_of(copies)), alone(eemd_of(copies, 0, 3));
            const std::vector<emd::Decomposition> all = batch.decompose(rows.data(), count, n);
            bool each = all.size() == static_cast<size_t>(count);
            for (int s = 0; each && s < count; ++s)
                each = same(all[s], alone.decompose(channels[s]));
            check(each && same(batch.decompose(channels)[4], all[4]),
                  copies ? "EEMD: a batch of 6 channels, every one bit for bit as alone"
                         : "EMD: a batch of 6 channels, every one bit for bit as alone");
        }
        std::vector<double> holed = x, filled = x;
        holed[100] = std::numeric_limits<double>::quiet_NaN();
        filled[100] = x[99] + 0.5 * (x[101] - x[99]);
        holed[300] = std::numeric_limits<double>::infinity();
        holed[301] = -std::numeric_limits<double>::infinity();
        filled[300] = x[299] + (1.0 / 3) * (x[302] - x[299]);
        filled[301] = x[299] + (2.0 / 3) * (x[302] - x[299]);
        const emd::Decomposition repaired = plain.decompose(holed);
        check(repaired.repaired == 3 && sum_error(filled, repaired) < 1e-12,
              "3 bad samples: filled in by straight lines, counted");
    }

    // ------------------------------------------------------------------
    std::printf("The interface on the engine; refused calls\n");
    {
        eemd::EEMD engine; // emd.m's rules
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        engine.decompose_emd(x.data(), n, imfs, residue);
        check(same(emd::Decomposition{imfs, residue, 0}, r), "EMD: the engine with its defaults");
        engine.config().ensemble_size = 30;
        engine.decompose(x.data(), n, imfs, residue);
        check(same(emd::Decomposition{imfs, residue, 0}, e1), "EEMD: the engine with 30 copies, noise 0.2, seed 42");
    }
    emd::Config bad;
    bad.ensemble_size = -1;
    bool config_refused = refused([&] { emd::Decomposer d(bad); }, "ensemble_size");
    bad = emd::Config();
    bad.noise = std::numeric_limits<double>::quiet_NaN();
    config_refused = config_refused && refused([&] { emd::Decomposer d(bad); }, "noise");
    bad = emd::Config();
    bad.max_imfs = 65;
    config_refused = config_refused && refused([&] { emd::Decomposer d(bad); }, "max_imfs");
    bad = emd::Config();
    bad.nthreads = -2;
    config_refused = config_refused && refused([&] { emd::Decomposer d(bad); }, "nthreads");
    check(config_refused, "a setting out of range, NaN: std::invalid_argument, naming it");
    check(refused([&] { plain.decompose(x.data(), 3); }, "samples") &&
              refused([&] { plain.decompose(x.data(), 0, n); }, "count") &&
              refused([&] { plain.decompose(nullptr, 1, n); }) &&
              refused([&] { plain.decompose(std::vector<std::vector<double>>{x, {1.0}}); }, "samples"),
          "3 samples, no signals, a null pointer, signals of unequal length");

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

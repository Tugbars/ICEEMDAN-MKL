/**
 * The C++ interface (include/iceemdan.hpp): its defaults against the engine
 * it stands on, every setting, batches, refusals and Hilbert-Huang analysis.
 *
 * Exits non-zero on any failure.
 */

#include "iceemdan.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %-72s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_failures;
}

static bool same(const iceemdan::Decomposition &a, const iceemdan::Decomposition &b)
{
    if (a.imfs.size() != b.imfs.size() || a.residue.size() != b.residue.size())
        return false;
    for (size_t j = 0; j < a.imfs.size(); ++j)
        if (std::memcmp(a.imfs[j].data(), b.imfs[j].data(), a.imfs[j].size() * sizeof(double)) != 0)
            return false;
    return std::memcmp(a.residue.data(), b.residue.data(), a.residue.size() * sizeof(double)) == 0;
}

static double sum_error(const std::vector<double> &x, const iceemdan::Decomposition &r)
{
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double sum = r.residue[i];
        for (const auto &imf : r.imfs)
            sum += imf[i];
        worst = std::max(worst, std::abs(sum - x[i]));
    }
    return worst;
}

template <class Exception, class Call>
static bool throws(Call &&call, const char *text_has = "")
{
    try
    {
        call();
    }
    catch (const Exception &e)
    {
        return std::strstr(e.what(), text_has) != nullptr;
    }
    catch (...)
    {
    }
    return false;
}

static iceemdan::Decomposition with(const iceemdan::Config &c, const std::vector<double> &x)
{
    return iceemdan::Decomposer(c).decompose(x);
}

int main()
{
    const int n = 600, count = 5;
    std::mt19937 rng(3);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<std::vector<double>> channels(count, std::vector<double>(n));
    for (int c = 0; c < count; ++c)
        for (int i = 0; i < n; ++i)
            channels[c][i] = std::sin(0.05 * (1 + c % 3) * i) + 0.4 * std::sin(0.31 * i + c) + 0.2 * g(rng);
    const std::vector<double> &x = channels[0];

    std::printf("Defaults\n");
    check(std::string(iceemdan::version()) == ICEEMDAN_VERSION_STRING && iceemdan::Decomposer::processors() >= 1,
          "version(), processors()");
    iceemdan::Decomposer d;
    check(d.threads() == iceemdan::Decomposer::processors() && !d.noise_kept(), "a new decomposer: on every processor");
    const iceemdan::Decomposition r = d.decompose(x.data(), n);
    check(!r.imfs.empty() && r.residue.size() == static_cast<size_t>(n) && sum_error(x, r) < 1e-10,
          "one signal: the IMFs and the residue add up to it");
    bool stats_ok = r.stats.size() == r.imfs.size() && r.orthogonality >= 0.0 && r.repaired == 0;
    for (const iceemdan::ImfStats &s : r.stats)
        stats_ok = stats_ok && s.copies == 100 && s.iterations_mean <= s.iterations_max && s.unconverged >= 0;
    check(stats_ok, "its audit trail: one entry per IMF, 100 copies each");
    {
        // the engine with its own defaults: the published algorithm
        eemd::ICEEMDAN engine;
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        const bool ok = engine.decompose(x.data(), n, imfs, residue);
        bool equal = ok && imfs.size() == r.imfs.size();
        for (size_t j = 0; equal && j < imfs.size(); ++j)
            equal = std::memcmp(imfs[j].data(), r.imfs[j].data(), n * sizeof(double)) == 0;
        check(equal && std::memcmp(residue.data(), r.residue.data(), n * sizeof(double)) == 0,
              "the same, bit for bit, as the engine with its defaults");
    }
    check(same(d.decompose(x), r) && d.noise_kept(), "again: the noise kept, the same result");

    std::printf("The configuration\n");
    iceemdan::Config c;
    c.ensemble_size = 100;
    c.noise = 0.2;
    c.seed = 42;
    check(same(with(c, x), r), "ensemble_size 100, noise 0.2 and seed 42 are the defaults");
    c = iceemdan::Config();
    c.seed = 7;
    const iceemdan::Decomposition other = with(c, x);
    check(!same(other, r) && sum_error(x, other) < 1e-10, "another seed: another result, which adds up too");
    c = iceemdan::Config();
    c.nthreads = 1;
    check(iceemdan::Decomposer(c).threads() == 1 && same(with(c, x), r), "nthreads 1: the same result, bit for bit");
    c = iceemdan::Config();
    c.max_imfs = 2;
    const iceemdan::Decomposition two = with(c, x);
    check(two.imfs.size() == 2 && sum_error(x, two) < 1e-10, "max_imfs 2: two IMFs, the rest in the residue");
    c = iceemdan::Config();
    c.fast = true;
    const iceemdan::Decomposition fast = with(c, x);
    {
        eemd::ICEEMDAN engine;
        engine.config().sift_stop_rule = eemd::SiftStopRule::SD;
        engine.config().local_mean_method = eemd::LocalMeanMethod::SingleEnvelope;
        engine.config().use_circular_bank = true;
        std::vector<std::vector<double>> imfs;
        std::vector<double> residue;
        engine.decompose(x.data(), n, imfs, residue);
        bool equal = imfs.size() == fast.imfs.size();
        for (size_t j = 0; equal && j < imfs.size(); ++j)
            equal = std::memcmp(imfs[j].data(), fast.imfs[j].data(), n * sizeof(double)) == 0;
        check(equal && !same(fast, r) && sum_error(x, fast) < 1e-10,
              "fast: the engine's SD rule, single envelope and circular bank");
    }

    std::printf("Many signals\n");
    c = iceemdan::Config();
    c.ensemble_size = 40;
    iceemdan::Decomposer batch(c), alone(c);
    std::vector<double> rows(static_cast<size_t>(count) * n);
    for (int s = 0; s < count; ++s)
        std::copy(channels[s].begin(), channels[s].end(), rows.begin() + static_cast<size_t>(s) * n);
    const std::vector<iceemdan::Decomposition> all = batch.decompose(rows.data(), count, n);
    bool each = all.size() == static_cast<size_t>(count);
    for (int s = 0; each && s < count; ++s)
        each = same(all[s], alone.decompose(channels[s]));
    check(each, "a batch of 5 channels: every one bit for bit as alone");
    const std::vector<iceemdan::Decomposition> listed = batch.decompose(channels);
    check(listed.size() == all.size() && same(listed[2], all[2]), "the same from a vector of signals");

    std::printf("NaN and Inf\n");
    std::vector<double> holed = x, filled = x;
    holed[100] = std::numeric_limits<double>::quiet_NaN();
    filled[100] = x[99] + 0.5 * (x[101] - x[99]);
    holed[300] = std::numeric_limits<double>::infinity();
    holed[301] = -std::numeric_limits<double>::infinity();
    filled[300] = x[299] + (1.0 / 3) * (x[302] - x[299]);
    filled[301] = x[299] + (2.0 / 3) * (x[302] - x[299]);
    const iceemdan::Decomposition repaired = alone.decompose(holed);
    check(repaired.repaired == 3 && sum_error(filled, repaired) < 1e-10,
          "3 bad samples: filled in by straight lines, counted");

    std::printf("Refused calls\n");
    bool refused = true;
    c = iceemdan::Config();
    c.ensemble_size = -1;
    refused = refused && throws<std::invalid_argument>([&] { iceemdan::Decomposer bad(c); }, "ensemble_size");
    c = iceemdan::Config();
    c.noise = std::numeric_limits<double>::quiet_NaN();
    refused = refused && throws<std::invalid_argument>([&] { iceemdan::Decomposer bad(c); }, "noise");
    c = iceemdan::Config();
    c.max_imfs = 65;
    refused = refused && throws<std::invalid_argument>([&] { iceemdan::Decomposer bad(c); }, "max_imfs");
    c = iceemdan::Config();
    c.nthreads = -2;
    refused = refused && throws<std::invalid_argument>([&] { iceemdan::Decomposer bad(c); }, "nthreads");
    check(refused, "a setting out of range, NaN: std::invalid_argument, naming it");
    check(throws<std::invalid_argument>([&] { d.decompose(x.data(), 3); }, "samples") &&
              throws<std::invalid_argument>([&] { d.decompose(x.data(), 0, n); }, "count") &&
              throws<std::invalid_argument>([&] { d.decompose(nullptr, 1, n); }) &&
              throws<std::invalid_argument>([&] { d.decompose(std::vector<std::vector<double>>{x, two.residue, {1.0}}); },
                                            "samples"),
          "3 samples, no signals, a null pointer, signals of unequal length");
    c = iceemdan::Config();
    c.ensemble_size = 1 << 30; // a billion copies: far more noise than any memory
    iceemdan::Decomposer huge(c);
    check(throws<std::runtime_error>([&] { huge.decompose(x); }, "GiB"),
          "noise that does not fit in half the memory: std::runtime_error, with the numbers");

    std::printf("Hilbert-Huang analysis\n");
    {
        // a tone of 32 whole cycles in the window, inside a frequency band
        const double fs = 1000.0, f = 32.0 * fs / n, pi = 3.14159265358979323846;
        std::vector<double> tone(n);
        for (int i = 0; i < n; ++i)
            tone[i] = 2.0 * std::cos(2 * pi * f * i / fs);
        const iceemdan::Instantaneous a = iceemdan::instantaneous(tone, fs);
        check(a.amplitude.size() == tone.size() && std::abs(a.amplitude[n / 2] - 2.0) < 1e-6 &&
                  std::abs(a.frequency[n / 2] - f) < 1e-6 && a.phase[n - 1] > a.phase[0],
              "instantaneous(): amplitude, frequency and phase of a tone");
        std::vector<double> amp(n);
        iceemdan::instantaneous(tone.data(), n, fs, amp.data(), nullptr, nullptr);
        check(amp == a.amplitude, "... through pointers, outputs left out");

        const iceemdan::Spectrum s = iceemdan::spectrum({tone}, fs);
        int peak = 0;
        for (int b = 1; b < s.bins; ++b)
            if (s.marginal[b] > s.marginal[peak])
                peak = b;
        check(s.bins == 128 && s.columns == n && s.values.size() == static_cast<size_t>(128) * n &&
                  peak == static_cast<int>(f / (fs / 2 / 128)) && std::abs(s.marginal[peak] - 4.0 * n) < 1e-6,
              "spectrum(): 128 bands by a column per sample; the tone's energy in its band");
        iceemdan::SpectrumOptions o;
        o.frequency_bins = 50;
        o.fmax = 250.0;
        o.time_step = 10;
        o.amplitude = true;
        const iceemdan::Spectrum coarse = iceemdan::spectrum({tone}, fs, o);
        check(coarse.bins == 50 && coarse.columns == n / 10 && iceemdan::spectrum_columns(n, o) == n / 10 &&
                  std::abs(coarse.marginal[static_cast<int>(f / 5.0)] - 2.0 * n) < 1e-6,
              "50 bands to 250 Hz, 10 samples per column, amplitude");
        o.fmin = 300.0;
        check(throws<std::invalid_argument>([&] { iceemdan::spectrum({tone}, fs, o); }, "fmin") &&
                  throws<std::invalid_argument>([&] { iceemdan::spectrum(std::vector<std::vector<double>>(), fs); }) &&
                  throws<std::invalid_argument>([&] { iceemdan::instantaneous(tone, 0.0); }, "sample_rate") &&
                  throws<std::invalid_argument>([&] { iceemdan::instantaneous(std::vector<double>(1), fs); }),
              "refused: fmin above fmax, no IMFs, no sample rate, one sample");
    }

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

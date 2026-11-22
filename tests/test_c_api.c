/*
 * The C interfaces from C: include/iceemdan.h, include/iceemdan_diagnostics.h
 * and include/emd.h compiled as C, against the shared library. Everything a
 * user can call, and every way a call is refused.
 */
#include "emd.h"
#include "iceemdan.h"
#include "iceemdan_diagnostics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char *what)
{
    printf("  %-70s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

enum { N = 600, COUNT = 5, MAX_ROWS = 64 };

/* a signal per channel: two rhythms and deterministic noise */
static void make_channels(double *x, int count, int n)
{
    unsigned int state = 12345u;
    for (int c = 0; c < count; ++c)
        for (int i = 0; i < n; ++i)
        {
            state = state * 1664525u + 1013904223u;
            const double noise = (double)(state >> 8) / 16777216.0 - 0.5;
            x[c * n + i] = sin(0.05 * (1 + c % 3) * i) + 0.4 * sin(0.31 * i + c) + 0.2 * noise;
        }
}

/* decompose one signal with d and copy the result out; the number of IMFs, or -1 */
static int one(iceemdan_t *d, const double *x, double *imfs, double *residue)
{
    const int k = iceemdan_decompose(d, x, 1, N);
    if (k >= 0 && iceemdan_result(d, imfs, residue, NULL) != 0)
        return -1;
    return k;
}

/* the largest difference of IMFs + residue from x */
static double sum_error(const double *x, const double *imfs, const double *residue, int k)
{
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double sum = residue[i];
        for (int j = 0; j < k; ++j)
            sum += imfs[j * N + i];
        worst = fmax(worst, fabs(sum - x[i]));
    }
    return worst;
}

static iceemdan_t *with(int ensemble_size, unsigned seed, int nthreads, int max_imfs, unsigned flags)
{
    iceemdan_config_t c;
    memset(&c, 0, sizeof(c));
    c.ensemble_size = ensemble_size;
    c.seed = seed;
    c.nthreads = nthreads;
    c.max_imfs = max_imfs;
    c.flags = flags;
    return iceemdan_create(&c);
}

int main(void)
{
    double *x = malloc(sizeof(double) * COUNT * N);
    double *imfs = malloc(sizeof(double) * MAX_ROWS * N);
    double *residue = malloc(sizeof(double) * N);
    double *other = malloc(sizeof(double) * MAX_ROWS * N);
    double *other_res = malloc(sizeof(double) * N);
    iceemdan_info_t info;
    make_channels(x, COUNT, N);

    printf("The library\n");
    check(iceemdan_info(NULL, &info) == 0 && strcmp(info.version, ICEEMDAN_VERSION_STRING) == 0,
          "iceemdan_info(NULL): the library is the version of this header");
    check(info.cpu_supported == 1 && info.processors >= 1 && info.signals == 0,
          "this processor runs it; processors counted");
    const int processors = info.processors;

    printf("Defaults\n");
    iceemdan_t *d = iceemdan_create(NULL);
    check(d != NULL && iceemdan_error()[0] == '\0', "iceemdan_create(NULL)");
    check(iceemdan_info(d, &info) == 0 && info.nthreads == processors && info.signals == 0,
          "a new decomposer: on every processor, no decomposition yet");
    check(iceemdan_result(d, imfs, residue, NULL) == -1 && iceemdan_error()[0] != '\0',
          "iceemdan_result() before a decomposition: refused, with a reason");
    const int k = one(d, x, imfs, residue);
    check(k > 0 && k <= MAX_ROWS && iceemdan_error()[0] == '\0', "one signal decomposed");
    check(sum_error(x, imfs, residue, k) < 1e-10, "the IMFs and the residue add up to the signal");
    check(iceemdan_info(d, &info) == 0 && info.signals == 1 && info.n == N && info.imfs == k && info.noise_kept == 0 &&
              info.repaired == 0,
          "iceemdan_info(): 1 signal, its length and IMFs, the noise built");
    const int k_again = one(d, x, other, other_res);
    check(k_again == k && memcmp(other, imfs, sizeof(double) * k * N) == 0 && iceemdan_info(d, &info) == 0 &&
              info.noise_kept == 1,
          "again: the noise kept, the same result");
    iceemdan_destroy(d);

    printf("The configuration\n");
    d = with(0, 0, 0, 0, 0);
    check(d && one(d, x, other, other_res) == k && memcmp(other, imfs, sizeof(double) * k * N) == 0,
          "a zeroed configuration is the default");
    iceemdan_destroy(d);
    d = with(100, 42, 0, 0, 0);
    check(d && one(d, x, other, other_res) == k && memcmp(other, imfs, sizeof(double) * k * N) == 0,
          "ensemble_size 100 and seed 42 are the default");
    iceemdan_destroy(d);
    d = with(0, 7, 0, 0, 0);
    const int k_seed = one(d, x, other, other_res);
    check(k_seed > 0 && (k_seed != k || memcmp(other, imfs, sizeof(double) * k * N) != 0) &&
              sum_error(x, other, other_res, k_seed) < 1e-10,
          "another seed: another result, which adds up too");
    iceemdan_destroy(d);
    d = with(0, 0, 1, 0, 0);
    check(d && iceemdan_info(d, &info) == 0 && info.nthreads == 1 && one(d, x, other, other_res) == k &&
              memcmp(other, imfs, sizeof(double) * k * N) == 0,
          "nthreads 1: the same result, bit for bit");
    iceemdan_destroy(d);
    d = with(0, 0, 0, 2, 0);
    const int k_two = one(d, x, other, other_res);
    check(k_two == 2 && sum_error(x, other, other_res, 2) < 1e-10, "max_imfs 2: two IMFs, the rest in the residue");
    iceemdan_destroy(d);
    d = with(0, 0, 0, 0, ICEEMDAN_FLAG_FAST);
    const int k_fast = one(d, x, other, other_res);
    check(k_fast > 0 && sum_error(x, other, other_res, k_fast) < 1e-10 &&
              (k_fast != k || memcmp(other, imfs, sizeof(double) * k * N) != 0),
          "ICEEMDAN_FLAG_FAST: another result, which adds up");
    iceemdan_destroy(d);

    printf("Many signals\n");
    d = with(40, 0, 0, 0, 0);
    iceemdan_t *alone = with(40, 0, 0, 0, 0);
    const int rows = iceemdan_decompose(d, x, COUNT, N);
    double *bimfs = malloc(sizeof(double) * COUNT * (rows > 0 ? rows : 1) * N);
    double *bres = malloc(sizeof(double) * COUNT * N);
    int counts[COUNT];
    check(rows > 0 && iceemdan_result(d, bimfs, bres, counts) == 0 && iceemdan_info(d, &info) == 0 &&
              info.signals == COUNT && info.imfs == rows,
          "a batch of 5 channels");
    int same = 1, most = 0;
    for (int c = 0; c < COUNT; ++c)
    {
        const int kc = one(alone, x + c * N, other, other_res);
        most = kc > most ? kc : most;
        same = same && kc == counts[c] && memcmp(other, bimfs + (size_t)c * rows * N, sizeof(double) * kc * N) == 0 &&
               memcmp(other_res, bres + c * N, sizeof(double) * N) == 0;
        for (int j = kc; j < rows; ++j)
            for (int i = 0; i < N; ++i)
                same = same && bimfs[((size_t)c * rows + j) * N + i] == 0.0;
    }
    check(same && most == rows, "every channel bit for bit as alone; zeros after its own IMFs");
    check(iceemdan_result(d, NULL, NULL, NULL) == 0, "iceemdan_result(): every output may be NULL");

    printf("The audit trail\n");
    iceemdan_signal_stats_t ss;
    iceemdan_imf_stats_t is;
    int stats_ok = 1;
    for (int c = 0; c < COUNT; ++c)
    {
        stats_ok = stats_ok && iceemdan_signal_stats(d, c, &ss) == 0 && ss.imfs == counts[c] && ss.orthogonality >= 0.0 &&
                   ss.repaired == 0;
        for (int j = 0; j < counts[c]; ++j)
            stats_ok = stats_ok && iceemdan_imf_stats(d, c, j, &is) == 0 && is.copies == 40 && is.iterations_max >= 0 &&
                       is.iterations_mean >= 0.0 && is.iterations_mean <= is.iterations_max && is.unconverged >= 0;
    }
    check(stats_ok, "every signal and every IMF: its facts");
    check(iceemdan_signal_stats(d, COUNT, &ss) == -1 && iceemdan_imf_stats(d, 0, counts[0], &is) == -1 &&
              iceemdan_error()[0] != '\0',
          "no such signal, no such IMF: refused, with a reason");

    printf("NaN and Inf\n");
    double *holed = malloc(sizeof(double) * N), *filled = malloc(sizeof(double) * N);
    memcpy(holed, x, sizeof(double) * N);
    memcpy(filled, x, sizeof(double) * N);
    holed[100] = NAN; /* one sample: the mean of its neighbours */
    filled[100] = x[99] + 0.5 * (x[101] - x[99]);
    holed[300] = INFINITY; /* two in a row: on the line from sample 299 to 302 */
    holed[301] = -INFINITY;
    filled[300] = x[299] + (1.0 / 3) * (x[302] - x[299]);
    filled[301] = x[299] + (2.0 / 3) * (x[302] - x[299]);
    const int k_holed = one(d, holed, imfs, residue);
    const int repaired = iceemdan_info(d, &info) == 0 ? info.repaired : -1;
    check(k_holed > 0 && repaired == 3 && iceemdan_signal_stats(d, 0, &ss) == 0 && ss.repaired == 3 &&
              sum_error(filled, imfs, residue, k_holed) < 1e-10,
          "3 bad samples: filled in by straight lines, counted; the result adds up to that");
    free(holed);
    free(filled);

    printf("Refused calls\n");
    iceemdan_config_t bad;
    memset(&bad, 0, sizeof(bad));
    bad.ensemble_size = -1;
    int refused = iceemdan_create(&bad) == NULL && strstr(iceemdan_error(), "ensemble_size") != NULL;
    memset(&bad, 0, sizeof(bad));
    bad.noise = -0.1;
    refused = refused && iceemdan_create(&bad) == NULL && strstr(iceemdan_error(), "noise") != NULL;
    bad.noise = NAN;
    refused = refused && iceemdan_create(&bad) == NULL;
    memset(&bad, 0, sizeof(bad));
    bad.max_imfs = 65;
    refused = refused && iceemdan_create(&bad) == NULL && strstr(iceemdan_error(), "max_imfs") != NULL;
    memset(&bad, 0, sizeof(bad));
    bad.nthreads = -2;
    refused = refused && iceemdan_create(&bad) == NULL && strstr(iceemdan_error(), "nthreads") != NULL;
    memset(&bad, 0, sizeof(bad));
    bad.flags = 1u << 7;
    refused = refused && iceemdan_create(&bad) == NULL && strstr(iceemdan_error(), "flag") != NULL;
    check(refused, "iceemdan_create(): a value out of range, NaN, an unknown flag");
    check(iceemdan_decompose(d, x, 1, 3) == -1 && strstr(iceemdan_error(), "samples") != NULL &&
              iceemdan_decompose(d, x, 0, N) == -1 && iceemdan_decompose(d, NULL, 1, N) == -1 &&
              iceemdan_decompose(NULL, x, 1, N) == -1 && iceemdan_result(NULL, imfs, residue, NULL) == -1 &&
              iceemdan_info(d, NULL) == -1,
          "iceemdan_decompose(): 3 samples, no signals, NULL; NULL decomposers");
    iceemdan_t *huge = with(1 << 30, 0, 0, 0, 0); /* a billion copies: far more noise than any memory */
    check(huge && iceemdan_decompose(huge, x, 1, N) == -1 && strstr(iceemdan_error(), "GiB") != NULL &&
              strstr(iceemdan_error(), "fast option") != NULL,
          "noise that does not fit in half the memory: refused, with the numbers");
    iceemdan_destroy(huge);
    check(one(d, x, imfs, residue) > 0 && iceemdan_error()[0] == '\0', "a served call leaves no reason behind");
    iceemdan_destroy(NULL);
    iceemdan_destroy(alone);
    iceemdan_destroy(d);
    free(bimfs);
    free(bres);

    printf("Hilbert-Huang analysis (no decomposer alive)\n");
    {
        /* a tone of 32 whole cycles in the window, inside a frequency band */
        const double fs = 1000.0, f = 32.0 * fs / N, pi = 3.14159265358979323846;
        double tone[N], amp[N], freq[N], phase[N];
        for (int i = 0; i < N; ++i)
            tone[i] = 2.0 * cos(2 * pi * f * i / fs);
        check(iceemdan_instantaneous(tone, N, fs, amp, freq, phase) == 0 && fabs(amp[N / 2] - 2.0) < 1e-6 &&
                  fabs(freq[N / 2] - f) < 1e-6 && phase[N - 1] > phase[0],
              "iceemdan_instantaneous(): amplitude, frequency and phase of a tone");
        check(iceemdan_instantaneous(tone, N, fs, NULL, NULL, NULL) == 0, "... every output may be NULL");

        /* NULL options: 128 bands to fs / 2, a column per sample */
        double *spectrum = malloc(sizeof(double) * 128 * N);
        double marginal[128];
        int peak = 0;
        int ok = iceemdan_spectrum(tone, 1, N, fs, NULL, spectrum, marginal) == 0;
        for (int b = 1; b < 128; ++b)
            if (marginal[b] > marginal[peak])
                peak = b;
        double column_sum = 0.0;
        for (int b = 0; b < 128; ++b)
            column_sum += spectrum[b * N + N / 2];
        check(ok && peak == (int)(f / (fs / 2 / 128)) && fabs(column_sum - 4.0) < 1e-6,
              "iceemdan_spectrum(NULL options): the tone's band; a column holds its energy");

        iceemdan_spectrum_t o;
        memset(&o, 0, sizeof(o));
        o.frequency_bins = 50;
        o.fmax = 250.0;
        o.time_step = 10;
        o.amplitude = 1;
        double small[50 * (N / 10)], m50[50];
        peak = 0;
        ok = iceemdan_spectrum(tone, 1, N, fs, &o, small, m50) == 0;
        for (int b = 1; b < 50; ++b)
            if (m50[b] > m50[peak])
                peak = b;
        check(ok && peak == (int)(f / 5.0) && fabs(m50[peak] - 2.0 * N) < 1e-6,
              "50 bands to 250 Hz, 10 samples per column, amplitude");

        o.fmin = 300.0;
        check(iceemdan_spectrum(tone, 1, N, fs, &o, small, m50) == -1 && strstr(iceemdan_error(), "fmin") != NULL &&
                  iceemdan_spectrum(tone, 0, N, fs, NULL, NULL, NULL) == -1 &&
                  iceemdan_instantaneous(tone, 1, fs, amp, NULL, NULL) == -1 &&
                  iceemdan_instantaneous(tone, N, 0.0, amp, NULL, NULL) == -1 &&
                  iceemdan_instantaneous(NULL, N, fs, amp, NULL, NULL) == -1,
              "refused: fmin above fmax, no IMFs, 1 sample, no sample rate, NULL");
        free(spectrum);
    }

    printf("EMD and EEMD (emd.h)\n");
    {
        emd_info_t facts;
        check(emd_info(NULL, &facts) == 0 && strcmp(facts.version, ICEEMDAN_VERSION_STRING) == 0 &&
                  facts.cpu_supported == 1 && facts.processors >= 1,
              "emd_info(NULL): the same library");
        emd_t *e = emd_create(NULL);
        check(e != NULL && emd_info(e, &facts) == 0 && facts.ensemble_size == 0 && facts.nthreads == facts.processors &&
                  facts.signals == 0,
              "emd_create(NULL): plain EMD, on every processor");
        check(emd_result(e, imfs, residue, NULL) == -1 && emd_error()[0] != '\0',
              "emd_result() before a decomposition: refused, with a reason");
        const int ke = emd_decompose(e, x, 1, N);
        check(ke > 0 && ke <= MAX_ROWS && emd_result(e, imfs, residue, NULL) == 0 && emd_error()[0] == '\0' &&
                  sum_error(x, imfs, residue, ke) < 1e-10,
              "EMD of one signal: the IMFs and the residue add up to it");
        check(emd_info(e, &facts) == 0 && facts.signals == 1 && facts.n == N && facts.imfs == ke && facts.repaired == 0,
              "emd_info(): 1 signal, its length and IMFs");

        const int erows = emd_decompose(e, x, COUNT, N);
        double *eimfs = malloc(sizeof(double) * COUNT * (erows > 0 ? erows : 1) * N);
        double *eres = malloc(sizeof(double) * COUNT * N);
        int ecounts[COUNT];
        int each = erows > 0 && emd_result(e, eimfs, eres, ecounts) == 0;
        emd_t *single = emd_create(NULL);
        for (int c = 0; each && c < COUNT; ++c)
        {
            const int kc = emd_decompose(single, x + c * N, 1, N);
            each = kc == ecounts[c] && emd_result(single, other, other_res, NULL) == 0 &&
                   memcmp(other, eimfs + (size_t)c * erows * N, sizeof(double) * kc * N) == 0 &&
                   memcmp(other_res, eres + c * N, sizeof(double) * N) == 0;
            for (int j = kc; j < erows; ++j)
                for (int i = 0; i < N; ++i)
                    each = each && eimfs[((size_t)c * erows + j) * N + i] == 0.0;
        }
        check(each, "a batch of 5 channels: every one bit for bit as alone; zeros after its own IMFs");
        emd_destroy(single);
        emd_destroy(e);
        free(eimfs);
        free(eres);

        emd_config_t ec;
        memset(&ec, 0, sizeof(ec));
        ec.ensemble_size = 20;
        emd_t *ee = emd_create(&ec);
        const int kk = ee ? emd_decompose(ee, x, 1, N) : -1;
        const double gap = kk > 0 && emd_result(ee, other, other_res, NULL) == 0 ? sum_error(x, other, other_res, kk) : -1.0;
        check(kk > 0 && emd_info(ee, &facts) == 0 && facts.ensemble_size == 20 && gap > 1e-6 && gap < 0.5,
              "ensemble_size 20: EEMD; IMFs + residue = the signal + the mean noise");
        ec.nthreads = 1;
        emd_t *ee1 = emd_create(&ec);
        check(ee1 && emd_decompose(ee1, x, 1, N) == kk && emd_result(ee1, imfs, residue, NULL) == 0 &&
                  memcmp(imfs, other, sizeof(double) * kk * N) == 0 && memcmp(residue, other_res, sizeof(double) * N) == 0,
              "nthreads 1: the same result, bit for bit");
        emd_destroy(ee1);
        emd_destroy(ee);

        memset(&ec, 0, sizeof(ec));
        ec.ensemble_size = -1;
        int refused_emd = emd_create(&ec) == NULL && strstr(emd_error(), "ensemble_size") != NULL;
        memset(&ec, 0, sizeof(ec));
        ec.max_imfs = 65;
        refused_emd = refused_emd && emd_create(&ec) == NULL && strstr(emd_error(), "max_imfs") != NULL;
        e = emd_create(NULL);
        refused_emd = refused_emd && emd_decompose(e, x, 1, 3) == -1 && strstr(emd_error(), "samples") != NULL &&
                      emd_decompose(e, NULL, 1, N) == -1 && emd_decompose(NULL, x, 1, N) == -1 &&
                      emd_info(e, NULL) == -1;
        emd_destroy(e);
        emd_destroy(NULL);
        check(refused_emd, "refused: a value out of range, 3 samples, NULL");
    }

    free(x);
    free(imfs);
    free(residue);
    free(other);
    free(other_res);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

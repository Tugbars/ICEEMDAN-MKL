/*
 * The C interface from C: include/iceemdan.h compiled as C, against the
 * shared library. Results, batches, settings, errors and Hilbert-Huang
 * analysis through the header alone.
 */
#include "iceemdan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char *what)
{
    printf("  %-66s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

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

/* decompose with h and copy the result out; returns the number of IMFs */
static int decompose(ICEEMDANHandle *h, const double *x, int n, double *imfs, double *residue)
{
    const int k = iceemdan_decompose(h, x, n);
    if (k > 0)
    {
        iceemdan_get_all_imfs(h, imfs);
        iceemdan_get_residue(h, residue);
    }
    return k;
}

int main(void)
{
    enum { N = 600, COUNT = 5, MAX_K = 32 };
    double *x = malloc(sizeof(double) * COUNT * N);
    double *imfs = malloc(sizeof(double) * MAX_K * N);
    double *residue = malloc(sizeof(double) * N);
    double *other = malloc(sizeof(double) * MAX_K * N);
    double *other_res = malloc(sizeof(double) * N);
    make_channels(x, COUNT, N);

    printf("C interface %s (library %s)\n", ICEEMDAN_VERSION_STRING, iceemdan_version());
    check(strcmp(iceemdan_version(), ICEEMDAN_VERSION_STRING) == 0, "the library is the version of this header");

    ICEEMDANHandle *d = iceemdan_create(ICEEMDAN_STANDARD);
    check(d != NULL, "iceemdan_create(ICEEMDAN_STANDARD)");
    iceemdan_set_ensemble_size(d, 40);

    /* one signal */
    const int k = decompose(d, x, N, imfs, residue);
    check(k > 0 && k <= MAX_K && iceemdan_get_num_imfs(d) == k && iceemdan_get_imf_length(d) == N,
          "decompose: IMF count and length");
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double sum = residue[i];
        for (int j = 0; j < k; ++j)
            sum += imfs[j * N + i];
        worst = fmax(worst, fabs(sum - x[i]));
    }
    check(worst < 1e-10, "the IMFs and the residue add up to the signal");
    int one_ok = 1;
    for (int j = 0; j < k; ++j)
    {
        iceemdan_get_imf(d, j, other);
        one_ok = one_ok && memcmp(other, imfs + j * N, sizeof(double) * N) == 0;
    }
    check(one_ok && !iceemdan_get_imf(d, k, other), "iceemdan_get_imf: each IMF, and none past the last");
    check(iceemdan_get_valid(d) == 1 && iceemdan_get_error(d)[0] == '\0' &&
              iceemdan_get_rng_seed_used(d) == 42u && iceemdan_get_noise_bank_reused(d) == 0,
          "diagnostics: valid, no error, the default seed, the noise built");
    int diag_ok = iceemdan_get_orthogonality_index(d) >= 0.0;
    for (int j = 0; j < k; ++j)
        diag_ok = diag_ok && iceemdan_get_imf_trials(d, j) == 40 && iceemdan_get_imf_sift_iterations(d, j) >= 0 &&
                  iceemdan_get_imf_mean_sift_iterations(d, j) >= 0.0 && iceemdan_get_imf_converged(d, j) >= 0 &&
                  iceemdan_get_imf_nonconverged_trials(d, j) >= 0;
    check(diag_ok && iceemdan_get_imf_trials(d, k) == -1, "per-IMF diagnostics, -1 past the last IMF");

    /* the same signal again: the kept noise, the same result */
    const int k2 = decompose(d, x, N, other, other_res);
    check(k2 == k && memcmp(other, imfs, sizeof(double) * k * N) == 0 && iceemdan_get_noise_bank_reused(d) == 1,
          "a second call keeps the noise and gives the same result");

    /* the boundary can be set back to the default */
    iceemdan_set_boundary_method(d, ICEEMDAN_BOUNDARY_MIRROR);
    decompose(d, x, N, other, other_res);
    iceemdan_set_boundary_method(d, ICEEMDAN_BOUNDARY_SYMMETRIC);
    const int k3 = decompose(d, x, N, other, other_res);
    check(k3 == k && memcmp(other, imfs, sizeof(double) * k * N) == 0,
          "ICEEMDAN_BOUNDARY_SYMMETRIC restores the default after MIRROR");

    /* the speed options change the result and still add up */
    iceemdan_set_stop_rule(d, ICEEMDAN_STOP_SD);
    iceemdan_set_local_mean_method(d, ICEEMDAN_LOCAL_MEAN_SINGLE_ENVELOPE);
    iceemdan_set_circular_bank(d, 1);
    iceemdan_set_antithetic(d, 1);
    const int k4 = decompose(d, x, N, other, other_res);
    worst = 0.0;
    for (int i = 0; i < N && k4 > 0; ++i)
    {
        double sum = other_res[i];
        for (int j = 0; j < k4; ++j)
            sum += other[j * N + i];
        worst = fmax(worst, fabs(sum - x[i]));
    }
    check(k4 > 0 && worst < 1e-10 && (k4 != k || memcmp(other, imfs, sizeof(double) * k * N) != 0),
          "stop rule, local mean, circular bank, +/- pairs: set and used");
    iceemdan_destroy(d);

    /* errors */
    d = iceemdan_create(ICEEMDAN_STANDARD);
    iceemdan_set_ensemble_size(d, 40);
    check(iceemdan_decompose(d, x, 3) == 0 && iceemdan_get_error(d)[0] != '\0', "3 samples: refused, with a reason");
    check(decompose(d, x, N, other, other_res) == k && iceemdan_get_error(d)[0] == '\0',
          "the next call that succeeds clears the reason");

    /* a batch: each channel as alone */
    const int most = iceemdan_decompose_batch(d, x, COUNT, N);
    check(most > 0 && iceemdan_batch_count(d) == COUNT && iceemdan_batch_num_imfs(d, COUNT) == -1,
          "decompose_batch: count, and -1 past the last signal");
    double *bimfs = malloc(sizeof(double) * COUNT * most * N);
    double *bres = malloc(sizeof(double) * COUNT * N);
    iceemdan_batch_get(d, bimfs, bres);
    ICEEMDANHandle *alone = iceemdan_create(ICEEMDAN_STANDARD);
    iceemdan_set_ensemble_size(alone, 40);
    int same = 1;
    for (int c = 0; c < COUNT; ++c)
    {
        const int kc = decompose(alone, x + c * N, N, other, other_res);
        same = same && kc == iceemdan_batch_num_imfs(d, c) &&
               memcmp(other, bimfs + (size_t)c * most * N, sizeof(double) * kc * N) == 0 &&
               memcmp(other_res, bres + c * N, sizeof(double) * N) == 0 &&
               iceemdan_batch_orthogonality_index(d, c) == iceemdan_get_orthogonality_index(alone);
        for (int j = kc; j < most; ++j)
            for (int i = 0; i < N; ++i)
                same = same && bimfs[((size_t)c * most + j) * N + i] == 0.0;
    }
    check(same, "every channel bit for bit as alone; rows past its own IMFs are zeros");
    iceemdan_destroy(alone);
    iceemdan_destroy(d);
    free(bimfs);
    free(bres);

    /* Hilbert-Huang analysis of a pure tone: 32 whole cycles in the window, inside a band */
    {
        const double fs = 1000.0, f = 32.0 * fs / N, pi = 3.14159265358979323846;
        double tone[N], amp[N], freq[N];
        for (int i = 0; i < N; ++i)
            tone[i] = 2.0 * cos(2 * pi * f * i / fs);
        check(iceemdan_instantaneous(tone, N, fs, amp, freq, NULL) == 1 && fabs(amp[N / 2] - 2.0) < 1e-6 &&
                  fabs(freq[N / 2] - f) < 1e-6,
              "instantaneous amplitude and frequency of a tone");
        check(fabs(iceemdan_mean_frequency(tone, N, fs) - f) < 0.5, "its mean frequency");
        const int bins = 100, cols = iceemdan_spectrum_columns(N, 10);
        double *spectrum = malloc(sizeof(double) * bins * cols);
        double marginal[100];
        check(cols == N / 10 && iceemdan_hilbert_spectrum(tone, 1, N, fs, bins, 0.0, 0.0, 10, 0, spectrum, marginal) == 1,
              "Hilbert spectrum: its size, and computed");
        int peak = 0;
        for (int b = 1; b < bins; ++b)
            if (marginal[b] > marginal[peak])
                peak = b;
        check(peak == (int)(f / (fs / 2 / bins)), "the marginal spectrum peaks at the tone");
        free(spectrum);
    }

    /* the library */
    iceemdan_set_num_threads(3);
    check(iceemdan_get_num_threads() == 3, "threads: set and read");
    iceemdan_set_num_threads(0);
    check(strcmp(iceemdan_fft_backend(1024), "VectorFFT") == 0, "the FFT is VectorFFT's");
    const double energy = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    check(fabs(iceemdan_compute_energy(x, 3) - energy) <= 1e-14 * energy, "energy: the sum of squares");

    free(x);
    free(imfs);
    free(residue);
    free(other);
    free(other_res);
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

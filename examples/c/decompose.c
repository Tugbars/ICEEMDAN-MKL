/*
 * Decompose a signal from C with the C interface (include/iceemdan.h).
 * Builds with the project (target c_example); in your own program, link the
 * iceemdan library and keep its DLLs (iceemdan, VectorFFT) beside the program.
 */
#include "iceemdan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* What a part of the decomposition holds: its mean frequency (the instantaneous
   frequency weighted by the energy) and its share of the signal's energy. */
static void describe(const char *name, const double *part, int n, double fs, double total_energy)
{
    double *amplitude = malloc(sizeof(double) * n);
    double *frequency = malloc(sizeof(double) * n);
    double weighted = 0.0, weight = 0.0, energy = 0.0;
    if (iceemdan_instantaneous(part, n, fs, amplitude, frequency, NULL) == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            weighted += amplitude[i] * amplitude[i] * frequency[i];
            weight += amplitude[i] * amplitude[i];
            energy += part[i] * part[i];
        }
    }
    printf("  %-8s mean frequency %5.1f Hz, %5.1f %% of the energy\n", name, weight > 0 ? weighted / weight : 0.0,
           100.0 * energy / total_energy);
    free(amplitude);
    free(frequency);
}

int main(void)
{
    /* 4 s at 250 samples per second: a 6 Hz rhythm, and a 40 Hz burst in the middle */
    const int n = 1000;
    const double fs = 250.0;
    const double pi = 3.14159265358979323846;
    double *x = malloc(sizeof(double) * n);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double t = i / fs;
        x[i] = sin(2 * pi * 6 * t) + ((t > 1.5 && t < 2.5) ? 0.5 * sin(2 * pi * 40 * t) : 0.0);
        total += x[i] * x[i];
    }

    /* NULL: the published algorithm with its usual settings */
    iceemdan_t *d = iceemdan_create(NULL);
    if (!d)
    {
        printf("no decomposer: %s\n", iceemdan_error());
        return 1;
    }

    /* one signal of n samples; for the channels of a recording, their number instead of 1 */
    const int k = iceemdan_decompose(d, x, 1, n);
    if (k < 0)
    {
        printf("not decomposed: %s\n", iceemdan_error());
        iceemdan_destroy(d);
        return 1;
    }

    double *imfs = malloc(sizeof(double) * (k > 0 ? k : 1) * n); /* IMF j: imfs[j * n + i] */
    double *residue = malloc(sizeof(double) * n);
    iceemdan_result(d, imfs, residue, NULL);

    printf("%d IMFs and the residue\n", k);
    for (int j = 0; j < k; ++j)
    {
        char name[16];
        snprintf(name, sizeof(name), "IMF %d:", j);
        describe(name, imfs + j * n, n, fs, total);
    }
    describe("residue:", residue, n, fs, total);

    /* the IMFs and the residue add up to the signal */
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double sum = residue[i];
        for (int j = 0; j < k; ++j)
            sum += imfs[j * n + i];
        worst = fmax(worst, fabs(sum - x[i]));
    }
    printf("largest difference of IMFs + residue from the signal: %.1e\n", worst);

    free(imfs);
    free(residue);
    free(x);
    iceemdan_destroy(d);
    return worst < 1e-9 ? 0 : 1;
}

/*
 * Decompose a signal from C: the C interface (include/iceemdan.h) in a few
 * lines. Builds with the project (target c_example); link with the iceemdan
 * library and keep its DLLs (iceemdan, VectorFFT) beside the program.
 */
#include "iceemdan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    /* 4 s at 250 samples per second: a 6 Hz rhythm and a 40 Hz burst in the middle */
    const int n = 1000;
    const double fs = 250.0;
    const double pi = 3.14159265358979323846;
    double *x = malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i)
    {
        const double t = i / fs;
        x[i] = sin(2 * pi * 6 * t) + ((t > 1.5 && t < 2.5) ? 0.5 * sin(2 * pi * 40 * t) : 0.0);
    }

    /* the published algorithm; every setting is optional */
    ICEEMDANHandle *d = iceemdan_create(ICEEMDAN_STANDARD);
    if (!d)
        return 1;
    iceemdan_set_ensemble_size(d, 100);
    iceemdan_set_rng_seed(d, 7);

    const int k = iceemdan_decompose(d, x, n);
    if (k == 0)
    {
        printf("decomposition failed: %s\n", iceemdan_get_error(d));
        iceemdan_destroy(d);
        return 1;
    }

    double *imfs = malloc(sizeof(double) * k * n); /* IMF j: imfs[j * n + i] */
    double *residue = malloc(sizeof(double) * n);
    iceemdan_get_all_imfs(d, imfs);
    iceemdan_get_residue(d, residue);

    /* what each part holds: its frequency (Hilbert-Huang) and its share of the energy */
    const double total = iceemdan_compute_energy(x, n);
    printf("VectorEMD %s: %d IMFs and the residue\n", iceemdan_version(), k);
    for (int j = 0; j < k; ++j)
        printf("  IMF %d:   mean frequency %5.1f Hz, %5.1f %% of the energy\n", j,
               iceemdan_mean_frequency(imfs + j * n, n, fs), 100.0 * iceemdan_compute_energy(imfs + j * n, n) / total);
    printf("  residue: mean frequency %5.1f Hz, %5.1f %% of the energy\n", iceemdan_mean_frequency(residue, n, fs),
           100.0 * iceemdan_compute_energy(residue, n) / total);

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

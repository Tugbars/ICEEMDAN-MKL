/**
 * The library side of the same-noise comparison (validation/same_noise/README.md):
 * the library and the authors' code decompose the same signal with the same
 * noise realizations, so that what differs is the algorithms, sample by sample.
 *
 *   same_noise noise <n> <count> <seed> <out.csv>
 *       the library's noise realizations 0 .. count-1 of the seed, one per row
 *       (the realizations ICEEMDAN uses without antithetic pairs)
 *   same_noise emd <n> <seed> <index> <out.csv> [--max-iters K]
 *       the EMD of realization <index>, as the noise bank makes it: the modes
 *       one per column, the residue last
 *   same_noise iceemdan <in.csv> <out.csv> --seed S [options]
 *       the ICEEMDAN of a signal (one value per line): the modes one per
 *       column, the residue last
 *
 * options of iceemdan:
 *   --antithetic        +/- noise pairs: ensemble/2 realizations, each used twice
 *   --max-iters K       sifting iterations at most, for the local means
 *   --ensemble M        realizations (default 100)
 */

#include "iceemdan_mkl.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<double> read_column(const char *path)
{
    std::vector<double> v;
    FILE *f = std::fopen(path, "r");
    if (!f)
    {
        std::perror(path);
        std::exit(1);
    }
    double x;
    while (std::fscanf(f, "%lf", &x) == 1)
        v.push_back(x);
    std::fclose(f);
    return v;
}

static void write_columns(const char *path, const std::vector<std::vector<double>> &cols)
{
    FILE *f = std::fopen(path, "w");
    if (!f)
    {
        std::perror(path);
        std::exit(1);
    }
    for (size_t i = 0; i < cols[0].size(); ++i)
        for (size_t c = 0; c < cols.size(); ++c)
            std::fprintf(f, "%.17g%c", cols[c][i], c + 1 < cols.size() ? ',' : '\n');
    std::fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: see the top of same_noise.cpp\n");
        return 2;
    }
    const std::string what = argv[1];

    if (what == "noise" && argc == 6)
    {
        const int n = std::atoi(argv[2]), count = std::atoi(argv[3]);
        const uint32_t seed = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
        FILE *f = std::fopen(argv[5], "w");
        if (!f)
        {
            std::perror(argv[5]);
            return 1;
        }
        std::vector<double> w(n);
        for (int i = 0; i < count; ++i)
        {
            eemd::trial_gaussian(seed, static_cast<uint32_t>(i), n, w.data());
            for (int j = 0; j < n; ++j)
                std::fprintf(f, "%.17g%c", w[j], j + 1 < n ? ',' : '\n');
        }
        std::fclose(f);
        return 0;
    }

    if (what == "emd" && argc >= 6)
    {
        const int n = std::atoi(argv[2]);
        const uint32_t seed = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
        const uint32_t index = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
        // the sifting of the noise bank: ICEEMDAN's defaults
        eemd::ICEEMDANConfig ic;
        eemd::EEMD e;
        e.config().max_imfs = 64;
        e.config().max_sift_iters = ic.max_sift_iters;
        e.config().boundary_extend = ic.boundary_extend;
        e.config().stop_rule = ic.sift_stop_rule;
        e.config().rilling = ic.rilling;
        e.config().symmetric_boundary = true;
        e.config().spline_method = ic.spline_method;
        for (int a = 6; a < argc; ++a)
            if (!std::strcmp(argv[a], "--max-iters") && a + 1 < argc)
                e.config().max_sift_iters = std::atoi(argv[++a]);
        std::vector<double> w(n), residue;
        eemd::trial_gaussian(seed, index, n, w.data());
        std::vector<std::vector<double>> modes;
        e.decompose_emd(w.data(), n, modes, residue);
        modes.push_back(residue);
        write_columns(argv[5], modes);
        return 0;
    }

    if (what == "iceemdan" && argc >= 4)
    {
        const std::vector<double> x = read_column(argv[2]);
        eemd::ICEEMDAN d;
        d.config().use_antithetic = false;
        for (int a = 4; a < argc; ++a)
        {
            if (!std::strcmp(argv[a], "--seed") && a + 1 < argc)
                d.config().rng_seed = static_cast<uint32_t>(std::strtoul(argv[++a], nullptr, 10));
            else if (!std::strcmp(argv[a], "--antithetic"))
                d.config().use_antithetic = true;
            else if (!std::strcmp(argv[a], "--max-iters") && a + 1 < argc)
                d.config().max_sift_iters = std::atoi(argv[++a]);
            else if (!std::strcmp(argv[a], "--ensemble") && a + 1 < argc)
                d.config().ensemble_size = std::atoi(argv[++a]);
            else
            {
                std::fprintf(stderr, "unknown option %s\n", argv[a]);
                return 2;
            }
        }
        std::vector<std::vector<double>> modes;
        std::vector<double> residue;
        eemd::DecompositionDiagnostics diag;
        if (!d.decompose_with_diagnostics(x.data(), static_cast<int>(x.size()), modes, residue, diag))
            return 1;
        modes.push_back(residue);
        write_columns(argv[3], modes);
        std::fprintf(stderr, "%zu IMFs; most sifting iterations per IMF:", modes.size() - 1);
        for (int it : diag.sift_iterations)
            std::fprintf(stderr, " %d", it);
        std::fprintf(stderr, "; trials stopped by the iteration limit:");
        for (int c : diag.nonconverged_trials)
            std::fprintf(stderr, " %d", c);
        std::fprintf(stderr, "\n");
        return 0;
    }

    std::fprintf(stderr, "usage: see the top of same_noise.cpp\n");
    return 2;
}

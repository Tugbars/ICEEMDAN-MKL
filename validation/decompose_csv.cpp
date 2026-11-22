/**
 * decompose_csv: decompose a one-column CSV and write the IMFs (one column
 * each, residue last). Used by the validation against the reference code.
 *
 *   decompose_csv <in.csv> <out.csv> [--seed S] [--sd] [--single] [--circular]
 *     --sd        SiftStopRule::SD instead of Rilling
 *     --single    LocalMeanMethod::SingleEnvelope instead of Sifted
 *     --circular  circular noise bank instead of independent realizations
 */
#include "iceemdan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <in.csv> <out.csv> [--seed S] [--sd] [--single] [--circular]\n", argv[0]);
        return 2;
    }
    std::vector<double> x;
    {
        FILE *f = std::fopen(argv[1], "r");
        if (!f)
        {
            std::perror(argv[1]);
            return 1;
        }
        double v;
        while (std::fscanf(f, "%lf", &v) == 1)
            x.push_back(v);
        std::fclose(f);
    }

    eemd::ICEEMDAN d; // the defaults: the published algorithm
    for (int a = 3; a < argc; ++a)
    {
        if (!std::strcmp(argv[a], "--seed") && a + 1 < argc)
            d.config().rng_seed = static_cast<uint32_t>(std::strtoul(argv[++a], nullptr, 10));
        else if (!std::strcmp(argv[a], "--sd"))
            d.config().sift_stop_rule = eemd::SiftStopRule::SD;
        else if (!std::strcmp(argv[a], "--single"))
            d.config().local_mean_method = eemd::LocalMeanMethod::SingleEnvelope;
        else if (!std::strcmp(argv[a], "--circular"))
            d.config().use_circular_bank = true;
    }

    std::vector<std::vector<double>> imfs;
    std::vector<double> res;
    if (!d.decompose(x.data(), static_cast<int32_t>(x.size()), imfs, res))
    {
        std::fprintf(stderr, "decomposition failed\n");
        return 1;
    }
    FILE *f = std::fopen(argv[2], "w");
    for (size_t i = 0; i < x.size(); ++i)
    {
        for (size_t k = 0; k < imfs.size(); ++k)
            std::fprintf(f, "%.17g,", imfs[k][i]);
        std::fprintf(f, "%.17g\n", res[i]);
    }
    std::fclose(f);
    return 0;
}

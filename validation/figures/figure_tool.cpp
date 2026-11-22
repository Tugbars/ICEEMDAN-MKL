// Data for the README figures: decompose a one-column CSV with this library, or time it.
//
//   figure_tool decompose <in.csv> <out.csv> [options]   IMFs one per column, residue last
//   figure_tool time <in.csv> <runs> [options]            prints: median ms, min ms, number of IMFs
//
// options: --seed S  --threads T  --sd  --single  --circular
//          --no-keep   the noise bank is not kept between calls: every timed decomposition
//                      costs what the first one of its length costs
#include "iceemdan_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: see the top of figure_tool.cpp\n");
        return 2;
    }
    const bool timing = !std::strcmp(argv[1], "time");
    std::vector<double> x;
    {
        FILE *f = std::fopen(argv[2], "r");
        if (!f)
        {
            std::perror(argv[2]);
            return 1;
        }
        double v;
        while (std::fscanf(f, "%lf", &v) == 1)
            x.push_back(v);
        std::fclose(f);
    }
    const int32_t n = static_cast<int32_t>(x.size());

    eemd::ICEEMDAN d; // the defaults: the published algorithm
    for (int a = 4; a < argc; ++a)
    {
        if (!std::strcmp(argv[a], "--seed") && a + 1 < argc)
            d.config().rng_seed = static_cast<uint32_t>(std::strtoul(argv[++a], nullptr, 10));
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc)
            eemd::set_num_threads(std::atoi(argv[++a]));
        else if (!std::strcmp(argv[a], "--sd"))
            d.config().sift_stop_rule = eemd::SiftStopRule::SD;
        else if (!std::strcmp(argv[a], "--single"))
            d.config().local_mean_method = eemd::LocalMeanMethod::SingleEnvelope;
        else if (!std::strcmp(argv[a], "--circular"))
            d.config().use_circular_bank = true;
        else if (!std::strcmp(argv[a], "--no-keep"))
            d.config().keep_noise_bank = false;
    }

    std::vector<std::vector<double>> imfs;
    std::vector<double> res;
    if (timing)
    {
        const int runs = std::atoi(argv[3]);
        d.decompose(x.data(), n, imfs, res); // warm-up: threads started, memory touched
        std::vector<double> t(runs);
        for (int r = 0; r < runs; ++r)
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (!d.decompose(x.data(), n, imfs, res))
                return 1;
            t[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        std::sort(t.begin(), t.end());
        std::printf("%d,%d,%.3f,%.3f,%d\n", n, eemd::get_num_threads(), t[runs / 2], t[0],
                    static_cast<int>(imfs.size()));
        return 0;
    }

    if (!d.decompose(x.data(), n, imfs, res))
    {
        std::fprintf(stderr, "decomposition failed\n");
        return 1;
    }
    FILE *f = std::fopen(argv[3], "w");
    for (int32_t i = 0; i < n; ++i)
    {
        for (size_t k = 0; k < imfs.size(); ++k)
            std::fprintf(f, "%.17g,", imfs[k][i]);
        std::fprintf(f, "%.17g\n", res[i]);
    }
    std::fclose(f);
    return 0;
}

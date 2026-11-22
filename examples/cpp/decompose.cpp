/**
 * Decompose a signal from C++ with the C++ interface (include/iceemdan.hpp).
 * Header-only: builds with the project (target cpp_example); in your own
 * program, link the CMake target iceemdan::iceemdan.
 */
#include "iceemdan.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

/* What a part of the decomposition holds: its mean frequency (the instantaneous
   frequency weighted by the energy) and its share of the signal's energy. */
static void describe(const char *name, const std::vector<double> &part, double fs, double total_energy)
{
    const iceemdan::Instantaneous a = iceemdan::instantaneous(part, fs);
    double weighted = 0.0, weight = 0.0, energy = 0.0;
    for (size_t i = 0; i < part.size(); ++i)
    {
        weighted += a.amplitude[i] * a.amplitude[i] * a.frequency[i];
        weight += a.amplitude[i] * a.amplitude[i];
        energy += part[i] * part[i];
    }
    std::printf("  %-8s mean frequency %5.1f Hz, %5.1f %% of the energy\n", name, weight > 0 ? weighted / weight : 0.0,
                100.0 * energy / total_energy);
}

int main()
{
    // 4 s at 250 samples per second: a 6 Hz rhythm, and a 40 Hz burst in the middle
    const int n = 1000;
    const double fs = 250.0;
    const double pi = 3.14159265358979323846;
    std::vector<double> x(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double t = i / fs;
        x[i] = std::sin(2 * pi * 6 * t) + ((t > 1.5 && t < 2.5) ? 0.5 * std::sin(2 * pi * 40 * t) : 0.0);
        total += x[i] * x[i];
    }

    iceemdan::Decomposer d; // the published algorithm with its usual settings
    const iceemdan::Decomposition r = d.decompose(x);

    std::printf("%d IMFs and the residue\n", static_cast<int>(r.imfs.size()));
    for (size_t j = 0; j < r.imfs.size(); ++j)
    {
        char name[16];
        std::snprintf(name, sizeof(name), "IMF %d:", static_cast<int>(j));
        describe(name, r.imfs[j], fs, total);
    }
    describe("residue:", r.residue, fs, total);

    // the IMFs and the residue add up to the signal
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double sum = r.residue[i];
        for (const std::vector<double> &imf : r.imfs)
            sum += imf[i];
        worst = std::fmax(worst, std::fabs(sum - x[i]));
    }
    std::printf("largest difference of IMFs + residue from the signal: %.1e\n", worst);
    return worst < 1e-9 ? 0 : 1;
}

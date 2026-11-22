/**
 * ICEEMDAN Performance Benchmark
 *
 * Tests scaling across signal lengths, ensemble sizes, thread counts,
 * and compares EEMD vs ICEEMDAN performance and quality.
 */

#define _USE_MATH_DEFINES
#include <memory>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "iceemdan.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>

using namespace eemd;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Test Signal Generators
// ============================================================================

// Multi-component synthetic signal (known structure)
void generate_synthetic_signal(double *signal, int32_t n, double dt)
{
    for (int32_t i = 0; i < n; ++i)
    {
        const double t = i * dt;
        // Trend (slow)
        const double trend = 2.0 * std::sin(2.0 * M_PI * 0.5 * t);
        // Mid-frequency with AM
        const double am = 1.0 + 0.3 * std::sin(2.0 * M_PI * 1.0 * t);
        const double mid = am * std::sin(2.0 * M_PI * 5.0 * t);
        // High frequency
        const double high = 0.5 * std::sin(2.0 * M_PI * 25.0 * t);
        // Chirp (frequency sweep)
        const double chirp = 0.3 * std::sin(2.0 * M_PI * (10.0 + 5.0 * t) * t);
        
        signal[i] = trend + mid + high + chirp;
    }
}

// ============================================================================
// Benchmark Structures
// ============================================================================

struct BenchResult
{
    double time_ms;
    double throughput_msamples;
    int32_t n_imfs;
    double noise_bank_time_ms; // ICEEMDAN-specific
};

struct QualityMetrics
{
    double orthogonality_index;  // How orthogonal are IMFs (0 = perfect)
    double reconstruction_error; // ||signal - sum(IMFs) - residue||
    double mode_mixing_score;    // Heuristic for mode mixing
};

// ============================================================================
// Quality Metrics
// ============================================================================

QualityMetrics compute_quality(
    const double *signal,
    int32_t n,
    const std::vector<std::vector<double>> &imfs,
    const std::vector<double> &residue)
{
    QualityMetrics q;
    
    // Orthogonality index: sum of |<IMF_i, IMF_j>| / (||IMF_i|| * ||IMF_j||)
    double orth_sum = 0.0;
    int32_t orth_count = 0;
    
    for (size_t i = 0; i < imfs.size(); ++i)
    {
        for (size_t j = i + 1; j < imfs.size(); ++j)
        {
            double dot = 0.0, norm_i = 0.0, norm_j = 0.0;
            
            for (int32_t k = 0; k < n; ++k)
            {
                dot += imfs[i][k] * imfs[j][k];
                norm_i += imfs[i][k] * imfs[i][k];
                norm_j += imfs[j][k] * imfs[j][k];
            }
            
            if (norm_i > 1e-10 && norm_j > 1e-10)
            {
                orth_sum += std::abs(dot) / std::sqrt(norm_i * norm_j);
                ++orth_count;
            }
        }
    }
    
    q.orthogonality_index = (orth_count > 0) ? orth_sum / orth_count : 0.0;
    
    // Reconstruction error
    std::vector<double> reconstructed(n, 0.0);
    for (const auto &imf : imfs)
    {
        for (int32_t i = 0; i < n; ++i)
        {
            reconstructed[i] += imf[i];
        }
    }
    for (int32_t i = 0; i < n; ++i)
    {
        reconstructed[i] += residue[i];
    }
    
    double err_sum = 0.0, sig_sum = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        double diff = signal[i] - reconstructed[i];
        err_sum += diff * diff;
        sig_sum += signal[i] * signal[i];
    }
    
    q.reconstruction_error = (sig_sum > 1e-10) ? std::sqrt(err_sum / sig_sum) : 0.0;
    
    // Mode mixing score: variance of instantaneous frequency within each IMF
    // Lower variance = less mode mixing
    double mm_score = 0.0;
    int32_t mm_count = 0;
    
    for (const auto &imf : imfs)
    {
        std::vector<double> inst_freq(n);
        if (compute_instantaneous_frequency(imf.data(), n, inst_freq.data(), 1.0))
        {
            // Compute variance of instantaneous frequency
            double mean = 0.0;
            for (int32_t i = 0; i < n; ++i)
            {
                mean += std::abs(inst_freq[i]);
            }
            mean /= n;
            
            double var = 0.0;
            for (int32_t i = 0; i < n; ++i)
            {
                double d = std::abs(inst_freq[i]) - mean;
                var += d * d;
            }
            var /= n;
            
            // Normalized by mean (coefficient of variation)
            if (mean > 1e-10)
            {
                mm_score += std::sqrt(var) / mean;
                ++mm_count;
            }
        }
    }
    
    q.mode_mixing_score = (mm_count > 0) ? mm_score / mm_count : 0.0;
    
    return q;
}

// ============================================================================
// ICEEMDAN Benchmarks
// ============================================================================

BenchResult run_iceemdan_benchmark(int32_t n, int32_t ensemble_size, int32_t n_trials)
{
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    ICEEMDANConfig config;
    config.max_imfs = 8;
    config.ensemble_size = ensemble_size;
    config.noise_std = 0.2;
    
    ICEEMDAN decomposer(config);
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    
    // Warmup (includes noise bank initialization)
    auto t_nb_start = Clock::now();
    decomposer.decompose(signal.data(), n, imfs, residue);
    auto t_nb_end = Clock::now();
    double noise_bank_time = std::chrono::duration<double, std::milli>(t_nb_end - t_nb_start).count();
    
    // Timed runs
    double total_time = 0.0;
    int32_t n_imfs = 0;
    
    for (int32_t t = 0; t < n_trials; ++t)
    {
        auto t0 = Clock::now();
        decomposer.decompose(signal.data(), n, imfs, residue);
        auto t1 = Clock::now();
        total_time += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_imfs = static_cast<int32_t>(imfs.size());
    }
    
    const double avg_time = total_time / n_trials;
    const double total_samples = static_cast<double>(ensemble_size) * n;
    const double throughput = (total_samples / 1e6) / (avg_time / 1000.0);
    
    return {avg_time, throughput, n_imfs, noise_bank_time};
}

void bench_signal_length()
{
    std::cout << "\n=== ICEEMDAN: Signal Length Scaling ===\n";
    std::cout << "Ensemble: 100, Threads: " << eemd::get_num_threads() << "\n\n";
    
    std::cout << std::setw(10) << "Length"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(8) << "IMFs"
              << std::setw(14) << "NoiseBnk(ms)" << "\n";
    std::cout << std::string(59, '-') << "\n";
    
    for (int32_t n : {256, 512, 1024, 2048, 4096, 8192})
    {
        auto r = run_iceemdan_benchmark(n, 100, 3);
        std::cout << std::setw(10) << n
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(8) << r.n_imfs
                  << std::setw(14) << std::setprecision(1) << r.noise_bank_time_ms << "\n";
    }
}

void bench_ensemble_size()
{
    std::cout << "\n=== ICEEMDAN: Ensemble Size Scaling ===\n";
    std::cout << "Signal: 1024, Threads: " << eemd::get_num_threads() << "\n\n";
    
    std::cout << std::setw(10) << "Ensemble"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "ms/trial" << "\n";
    std::cout << std::string(49, '-') << "\n";
    
    for (int32_t ens : {10, 25, 50, 100, 200, 500})
    {
        auto r = run_iceemdan_benchmark(1024, ens, 3);
        std::cout << std::setw(10) << ens
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(12) << std::setprecision(3) << r.time_ms / ens << "\n";
    }
}

void bench_thread_scaling()
{
    std::cout << "\n=== ICEEMDAN: Thread Scaling ===\n";
    std::cout << "Signal: 1024, Ensemble: 100\n\n";
    
    const int32_t max_threads = eemd::get_num_threads();
    
    std::cout << std::setw(8) << "Threads"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(47, '-') << "\n";
    
    double baseline_time = 0.0;
    
    for (int32_t threads = 1; threads <= max_threads; threads *= 2)
    {
        eemd::set_num_threads(threads);
        
        // Warm up the thread pool
        eemd::threads::run(eemd::threads::team_size(), [](int) {});
        
        auto r = run_iceemdan_benchmark(1024, 100, 5);
        
        if (threads == 1)
            baseline_time = r.time_ms;
        const double speedup = (r.time_ms > 0) ? baseline_time / r.time_ms : 0.0;
        
        std::cout << std::setw(8) << threads
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.time_ms
                  << std::setw(12) << std::setprecision(2) << r.throughput_msamples << " MS/s"
                  << std::setw(12) << std::setprecision(2) << speedup << "x";
        
        if (r.n_imfs == 0)
            std::cout << " [FAIL]";
        std::cout << "\n";
    }
    
    eemd::set_num_threads(max_threads);
}

void bench_eemd_vs_iceemdan()
{
    std::cout << "\n=== EEMD vs ICEEMDAN Comparison ===\n";
    std::cout << "Signal: 2048, Ensemble: 100\n\n";
    
    const int32_t n = 2048;
    std::vector<double> signal(n);
    generate_synthetic_signal(signal.data(), n, 0.01);
    
    // EEMD
    EEMDConfig eemd_config;
    eemd_config.max_imfs = 8;
    eemd_config.ensemble_size = 100;
    eemd_config.noise_std = 0.2;
    
    EEMD eemd_decomposer(eemd_config);
    std::vector<std::vector<double>> eemd_imfs;
    int32_t eemd_n_imfs;
    
    // Warmup
    eemd_decomposer.decompose(signal.data(), n, eemd_imfs, eemd_n_imfs);
    
    auto t0 = Clock::now();
    for (int i = 0; i < 5; ++i)
    {
        eemd_decomposer.decompose(signal.data(), n, eemd_imfs, eemd_n_imfs);
    }
    auto t1 = Clock::now();
    double eemd_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    
    // Build residue for EEMD (sum of what's left)
    std::vector<double> eemd_residue(n, 0.0);
    for (int32_t i = 0; i < n; ++i)
    {
        eemd_residue[i] = signal[i];
        for (const auto &imf : eemd_imfs)
        {
            eemd_residue[i] -= imf[i];
        }
    }
    
    // ICEEMDAN
    ICEEMDANConfig iceemdan_config;
    iceemdan_config.max_imfs = 8;
    iceemdan_config.ensemble_size = 100;
    iceemdan_config.noise_std = 0.2;
    
    ICEEMDAN iceemdan_decomposer(iceemdan_config);
    std::vector<std::vector<double>> iceemdan_imfs;
    std::vector<double> iceemdan_residue;
    
    // Warmup
    iceemdan_decomposer.decompose(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    
    t0 = Clock::now();
    for (int i = 0; i < 5; ++i)
    {
        iceemdan_decomposer.decompose(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    }
    t1 = Clock::now();
    double iceemdan_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    
    // Quality metrics
    auto eemd_quality = compute_quality(signal.data(), n, eemd_imfs, eemd_residue);
    auto iceemdan_quality = compute_quality(signal.data(), n, iceemdan_imfs, iceemdan_residue);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "                    EEMD        ICEEMDAN\n";
    std::cout << "                    ----        --------\n";
    std::cout << "Time (ms):          " << std::setw(8) << eemd_time
              << "    " << std::setw(8) << iceemdan_time << "\n";
    std::cout << "IMFs extracted:     " << std::setw(8) << eemd_n_imfs
              << "    " << std::setw(8) << iceemdan_imfs.size() << "\n";
    std::cout << std::setprecision(6);
    std::cout << "Recon error:        " << std::setw(8) << eemd_quality.reconstruction_error
              << "    " << std::setw(8) << iceemdan_quality.reconstruction_error << "\n";
    std::cout << std::setprecision(4);
    std::cout << "Orthogonality:      " << std::setw(8) << eemd_quality.orthogonality_index
              << "    " << std::setw(8) << iceemdan_quality.orthogonality_index << "\n";
    std::cout << "Mode mixing:        " << std::setw(8) << eemd_quality.mode_mixing_score
              << "    " << std::setw(8) << iceemdan_quality.mode_mixing_score << "\n";
    
    std::cout << "\n(Lower orthogonality & mode mixing = better)\n";
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    eemd_init_low_latency(8, true);
    
    std::cout << "\nICEEMDAN Performance Benchmark\n";
    std::cout << "==============================\n";
    std::cout << "Max threads: " << eemd::get_num_threads() << "\n";
    
    bench_signal_length();
    bench_ensemble_size();
    bench_thread_scaling();
    bench_eemd_vs_iceemdan();
    
    
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}
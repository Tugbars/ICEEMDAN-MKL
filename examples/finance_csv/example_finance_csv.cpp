/**
 * ICEEMDAN Finance Example - CSV Output
 * 
 * Decomposes a simulated financial signal and outputs CSV files
 * for visualization in Python/Jupyter.
 * 
 * Outputs:
 *   - signal.csv:      Original price series
 *   - imfs.csv:        All IMFs (columns: imf_0, imf_1, ..., residue)
 *   - diagnostics.csv: Decomposition metadata
 */

#include "iceemdan_mkl.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Financial Signal Generators
// ============================================================================

/**
 * Generate GARCH(1,1) price series with volatility clustering
 */
void generate_garch_price(double *price, double *returns, double *volatility,
                          int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> innovations(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, innovations.data(), 0.0, 1.0);
    
    // GARCH(1,1): sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
    const double omega = 0.00001;
    const double alpha = 0.1;
    const double beta = 0.85;
    
    double sigma2 = 0.0001;
    price[0] = 100.0;
    returns[0] = 0.0;
    volatility[0] = std::sqrt(sigma2);
    
    for (int32_t i = 1; i < n; ++i)
    {
        double eps_prev = innovations[i-1] * std::sqrt(sigma2);
        sigma2 = omega + alpha * eps_prev * eps_prev + beta * sigma2;
        
        volatility[i] = std::sqrt(sigma2);
        returns[i] = volatility[i] * innovations[i];
        price[i] = price[i-1] * std::exp(returns[i]);
    }
    
    vslDeleteStream(&stream);
}

/**
 * Generate regime-switching signal (low vol → high vol → low vol)
 */
void generate_regime_switch(double *price, double *returns, double *volatility,
                            int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> noise(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, noise.data(), 0.0, 1.0);
    
    price[0] = 100.0;
    returns[0] = 0.0;
    volatility[0] = 0.005;
    
    for (int32_t i = 1; i < n; ++i)
    {
        // Regime determination
        double vol;
        if (i < n / 4)
            vol = 0.005;           // Low vol
        else if (i < n / 2)
            vol = 0.025;           // High vol (5x)
        else if (i < 3 * n / 4)
            vol = 0.005;           // Low vol
        else
            vol = 0.015;           // Medium vol
        
        // Add slight trend
        double drift = 0.0001 * std::sin(2.0 * M_PI * i / n * 2);
        
        volatility[i] = vol;
        returns[i] = drift + vol * noise[i];
        price[i] = price[i-1] * (1.0 + returns[i]);
    }
    
    vslDeleteStream(&stream);
}

/**
 * Generate signal with embedded cyclical components + noise
 * (easier to see ICEEMDAN separation)
 */
void generate_multi_scale(double *price, int32_t n, uint32_t seed)
{
    VSLStreamStatePtr stream = nullptr;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    
    std::vector<double> noise(n);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, noise.data(), 0.0, 1.0);
    
    for (int32_t i = 0; i < n; ++i)
    {
        double t = static_cast<double>(i) / n;
        
        // Trend (very low frequency)
        double trend = 100.0 + 20.0 * t;
        
        // Seasonal/cyclical (medium frequency)
        double seasonal = 5.0 * std::sin(2.0 * M_PI * 8 * t);
        
        // Higher frequency oscillation
        double oscillation = 2.0 * std::sin(2.0 * M_PI * 32 * t);
        
        // High frequency (noise-like)
        double high_freq = 0.8 * std::sin(2.0 * M_PI * 128 * t);
        
        // Actual noise
        double noise_component = 0.5 * noise[i];
        
        price[i] = trend + seasonal + oscillation + high_freq + noise_component;
    }
    
    vslDeleteStream(&stream);
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void write_signal_csv(const std::string &filename,
                      const double *price, 
                      const double *returns,
                      const double *volatility,
                      int32_t n)
{
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(8);
    
    file << "index,price,returns,volatility\n";
    for (int32_t i = 0; i < n; ++i)
    {
        file << i << "," << price[i];
        if (returns) file << "," << returns[i];
        else file << ",0";
        if (volatility) file << "," << volatility[i];
        else file << ",0";
        file << "\n";
    }
    
    file.close();
    std::cout << "Wrote: " << filename << "\n";
}

void write_imfs_csv(const std::string &filename,
                    const std::vector<std::vector<double>> &imfs,
                    const std::vector<double> &residue,
                    int32_t n)
{
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(8);
    
    // Header
    file << "index";
    for (size_t k = 0; k < imfs.size(); ++k)
    {
        file << ",imf_" << k;
    }
    file << ",residue\n";
    
    // Data
    for (int32_t i = 0; i < n; ++i)
    {
        file << i;
        for (size_t k = 0; k < imfs.size(); ++k)
        {
            file << "," << imfs[k][i];
        }
        file << "," << residue[i] << "\n";
    }
    
    file.close();
    std::cout << "Wrote: " << filename << "\n";
}

void write_diagnostics_csv(const std::string &filename,
                           const eemd::DecompositionDiagnostics &diag,
                           const std::vector<std::vector<double>> &imfs,
                           int32_t n,
                           double elapsed_ms)
{
    std::ofstream file(filename);
    file << std::fixed;
    
    // General info
    file << "metric,value\n";
    file << "signal_length," << n << "\n";
    file << "num_imfs," << imfs.size() << "\n";
    file << "elapsed_ms," << std::setprecision(2) << elapsed_ms << "\n";
    file << "orthogonality_index," << std::setprecision(6) << diag.orthogonality_index << "\n";
    file << "rng_seed," << diag.rng_seed_used << "\n";
    file << "nan_count," << diag.nan_count << "\n";
    
    file.close();
    std::cout << "Wrote: " << filename << "\n";
}

void write_imf_stats_csv(const std::string &filename,
                         const std::vector<std::vector<double>> &imfs,
                         const std::vector<double> &residue,
                         int32_t n)
{
    std::ofstream file(filename);
    file << std::fixed;
    
    file << "imf,energy,energy_pct,mean,std,min,max\n";
    
    // Compute total energy first
    double total_energy = 0.0;
    for (const auto &imf : imfs)
    {
        for (int32_t i = 0; i < n; ++i)
            total_energy += imf[i] * imf[i];
    }
    for (int32_t i = 0; i < n; ++i)
        total_energy += residue[i] * residue[i];
    
    // IMF stats
    for (size_t k = 0; k < imfs.size(); ++k)
    {
        const auto &imf = imfs[k];
        
        double sum = 0, sum_sq = 0;
        double min_val = imf[0], max_val = imf[0];
        
        for (int32_t i = 0; i < n; ++i)
        {
            sum += imf[i];
            sum_sq += imf[i] * imf[i];
            min_val = std::min(min_val, imf[i]);
            max_val = std::max(max_val, imf[i]);
        }
        
        double mean = sum / n;
        double energy = sum_sq;
        double variance = sum_sq / n - mean * mean;
        double std_dev = std::sqrt(std::max(0.0, variance));
        double energy_pct = 100.0 * energy / total_energy;
        
        file << "imf_" << k << ","
             << std::setprecision(4) << energy << ","
             << std::setprecision(2) << energy_pct << ","
             << std::setprecision(6) << mean << ","
             << std::setprecision(6) << std_dev << ","
             << std::setprecision(6) << min_val << ","
             << std::setprecision(6) << max_val << "\n";
    }
    
    // Residue stats
    {
        double sum = 0, sum_sq = 0;
        double min_val = residue[0], max_val = residue[0];
        
        for (int32_t i = 0; i < n; ++i)
        {
            sum += residue[i];
            sum_sq += residue[i] * residue[i];
            min_val = std::min(min_val, residue[i]);
            max_val = std::max(max_val, residue[i]);
        }
        
        double mean = sum / n;
        double energy = sum_sq;
        double variance = sum_sq / n - mean * mean;
        double std_dev = std::sqrt(std::max(0.0, variance));
        double energy_pct = 100.0 * energy / total_energy;
        
        file << "residue,"
             << std::setprecision(4) << energy << ","
             << std::setprecision(2) << energy_pct << ","
             << std::setprecision(6) << mean << ","
             << std::setprecision(6) << std_dev << ","
             << std::setprecision(6) << min_val << ","
             << std::setprecision(6) << max_val << "\n";
    }
    
    file.close();
    std::cout << "Wrote: " << filename << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[])
{
    std::cout << "ICEEMDAN Finance Example - CSV Output\n";
    std::cout << "======================================\n\n";
    
    // Parameters
    const int32_t n = 2048;
    const uint32_t seed = 42;
    const std::string output_dir = (argc > 1) ? argv[1] : ".";
    
    std::cout << "Signal length: " << n << "\n";
    std::cout << "Output directory: " << output_dir << "\n\n";
    
    // Allocate
    std::vector<double> price(n), returns(n), volatility(n);
    
    // ========================================================================
    // Dataset 1: GARCH Price
    // ========================================================================
    std::cout << "--- Dataset 1: GARCH Price Series ---\n";
    
    generate_garch_price(price.data(), returns.data(), volatility.data(), n, seed);
    write_signal_csv(output_dir + "/garch_signal.csv", 
                     price.data(), returns.data(), volatility.data(), n);
    
    // Decompose
    eemd::ICEEMDAN decomposer(eemd::ProcessingMode::Finance);
    decomposer.config().ensemble_size = 100;
    decomposer.config().rng_seed = seed;
    
    std::vector<std::vector<double>> imfs;
    std::vector<double> residue;
    eemd::DecompositionDiagnostics diag;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    decomposer.decompose_with_diagnostics(price.data(), n, imfs, residue, diag);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "Decomposed in " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
    std::cout << "Extracted " << imfs.size() << " IMFs\n\n";
    
    write_imfs_csv(output_dir + "/garch_imfs.csv", imfs, residue, n);
    write_diagnostics_csv(output_dir + "/garch_diagnostics.csv", diag, imfs, n, elapsed);
    write_imf_stats_csv(output_dir + "/garch_imf_stats.csv", imfs, residue, n);
    
    // ========================================================================
    // Dataset 2: Regime Switching
    // ========================================================================
    std::cout << "\n--- Dataset 2: Regime-Switching Signal ---\n";
    
    generate_regime_switch(price.data(), returns.data(), volatility.data(), n, seed + 1);
    write_signal_csv(output_dir + "/regime_signal.csv",
                     price.data(), returns.data(), volatility.data(), n);
    
    t0 = std::chrono::high_resolution_clock::now();
    decomposer.decompose_with_diagnostics(price.data(), n, imfs, residue, diag);
    t1 = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "Decomposed in " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
    std::cout << "Extracted " << imfs.size() << " IMFs\n\n";
    
    write_imfs_csv(output_dir + "/regime_imfs.csv", imfs, residue, n);
    write_diagnostics_csv(output_dir + "/regime_diagnostics.csv", diag, imfs, n, elapsed);
    write_imf_stats_csv(output_dir + "/regime_imf_stats.csv", imfs, residue, n);
    
    // ========================================================================
    // Dataset 3: Multi-Scale (clear separation demo)
    // ========================================================================
    std::cout << "\n--- Dataset 3: Multi-Scale Synthetic ---\n";
    
    generate_multi_scale(price.data(), n, seed + 2);
    write_signal_csv(output_dir + "/multiscale_signal.csv", price.data(), nullptr, nullptr, n);
    
    // Use standard mode for comparison
    eemd::ICEEMDAN std_decomposer(eemd::ProcessingMode::Standard);
    std_decomposer.config().ensemble_size = 100;
    std_decomposer.config().rng_seed = seed;
    
    t0 = std::chrono::high_resolution_clock::now();
    std_decomposer.decompose_with_diagnostics(price.data(), n, imfs, residue, diag);
    t1 = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::cout << "Decomposed in " << std::fixed << std::setprecision(1) << elapsed << " ms\n";
    std::cout << "Extracted " << imfs.size() << " IMFs\n\n";
    
    write_imfs_csv(output_dir + "/multiscale_imfs.csv", imfs, residue, n);
    write_diagnostics_csv(output_dir + "/multiscale_diagnostics.csv", diag, imfs, n, elapsed);
    write_imf_stats_csv(output_dir + "/multiscale_imf_stats.csv", imfs, residue, n);
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n======================================\n";
    std::cout << "Output files created:\n";
    std::cout << "  - garch_signal.csv, garch_imfs.csv, garch_imf_stats.csv\n";
    std::cout << "  - regime_signal.csv, regime_imfs.csv, regime_imf_stats.csv\n";
    std::cout << "  - multiscale_signal.csv, multiscale_imfs.csv, multiscale_imf_stats.csv\n";
    std::cout << "\nRun the Jupyter notebook to visualize results.\n";
    
    return 0;
}

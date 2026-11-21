"""Signals for the README figure and timings, written to the working directory.

burst.csv        the artificial signal of Torres et al. (2011) and Colominas et al. (2014): a
                 sinusoid of 1000 samples plus a burst of a higher frequency on samples 501-750
burst_truth.csv  its two components (burst, sinusoid), one per column
tones_<N>.csv    the sum of tones of ../make_signals.py (8, 32 and 128 cycles plus a trend) at
                 N = 1024, 2048, 4096 and 8192, for the timings
"""
import numpy as np

n = np.arange(1, 1001)
sinusoid = np.sin(2 * np.pi * 0.065 * (n - 1))
burst = np.where((n >= 501) & (n <= 750), np.sin(2 * np.pi * 0.255 * (n - 501)), 0.0)
np.savetxt('burst.csv', burst + sinusoid, fmt='%.17g')
np.savetxt('burst_truth.csv', np.column_stack([burst, sinusoid]), fmt='%.17g', delimiter=',')

for length in (1024, 2048, 4096, 8192):
    t = np.arange(length) / length
    x = 100 + 10 * t + 5 * np.sin(2 * np.pi * 8 * t) + 2 * np.sin(2 * np.pi * 32 * t) + 0.5 * np.sin(2 * np.pi * 128 * t)
    np.savetxt(f'tones_{length}.csv', x, fmt='%.17g')

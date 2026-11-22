"""Write the validation signals: white noise (N = 4096), a sum of tones on a trend (N = 2048)
and a rhythm on a wandering baseline (N = 2048)."""
import numpy as np

rng = np.random.default_rng(2024)
np.savetxt('wn.csv', rng.standard_normal(4096), fmt='%.17g')
n = 2048
t = np.arange(n) / n
x = 100 + 10 * t + 5 * np.sin(2 * np.pi * 8 * t) + 2 * np.sin(2 * np.pi * 32 * t) + 0.5 * np.sin(2 * np.pi * 128 * t)
np.savetxt('sines.csv', x, fmt='%.17g')
# a rhythm of 40 cycles on a baseline that wanders (one and a half cycles) and drifts
x = np.sin(2 * np.pi * 40 * t) + 1.5 * np.sin(2 * np.pi * 1.5 * t + 0.7) + 2 * t * t
np.savetxt('baseline.csv', x, fmt='%.17g')

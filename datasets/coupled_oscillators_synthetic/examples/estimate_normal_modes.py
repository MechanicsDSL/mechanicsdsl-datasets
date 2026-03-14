"""
estimate_normal_modes.py
------------------------
Identifies normal mode frequencies from the coupled oscillator dataset
using FFT, then estimates spring constant k from the frequency ratio.

Usage:
    python examples/estimate_normal_modes.py

Expected output:
    omega1 (FFT)      : 3.1321 rad/s  (ground truth: 3.1321)
    omega2 (FFT)      : 3.2636 rad/s  (ground truth: 3.2636)
    k estimated       : 0.500 N/m     (ground truth: 0.500)
"""

import numpy as np
import pandas as pd
import json

data = pd.read_csv("../data.csv")
meta = json.load(open("../metadata.json"))

t    = data["t_s"].values
th1  = data["theta1_rad_noisy"].values
th2  = data["theta2_rad_noisy"].values
truth = meta["ground_truth"]
dt   = t[1] - t[0]

# Normal mode coordinates
q_plus  = (th1 + th2) / np.sqrt(2)
q_minus = (th1 - th2) / np.sqrt(2)

# FFT to find dominant frequencies
freqs = np.fft.rfftfreq(len(q_plus), dt) * 2 * np.pi
fft_p = np.abs(np.fft.rfft(q_plus))
fft_m = np.abs(np.fft.rfft(q_minus))

# Mask DC component
fft_p[0] = 0
fft_m[0] = 0

omega1_fft = freqs[np.argmax(fft_p)]
omega2_fft = freqs[np.argmax(fft_m)]

# Estimate k from: omega2^2 = g/l + 2k/m  =>  k = m*(omega2^2 - g/l) / 2
g = truth["g"]
l = truth["l"]
m = truth["m"]
k_est = m * (omega2_fft**2 - g/l) / 2.0

print(f"Normal mode identification from FFT")
print(f"  omega1 (FFT)      : {omega1_fft:.4f} rad/s  (ground truth: {meta['omega1_rad_s']:.4f})")
print(f"  omega2 (FFT)      : {omega2_fft:.4f} rad/s  (ground truth: {meta['omega2_rad_s']:.4f})")
print()
print(f"Spring constant estimation from omega2")
print(f"  k estimated       : {k_est:.3f} N/m  (ground truth: {truth['k']:.3f})")
print(f"  Error             : {abs(k_est-truth['k'])*1000:.2f} mN/m")
print()
print(f"Beat period")
omega_beat = (omega2_fft - omega1_fft) / 2
T_beat_est = 2*np.pi / omega_beat
print(f"  T_beat estimated  : {T_beat_est:.3f} s  (metadata: {meta['beat_period_s']:.3f} s)")

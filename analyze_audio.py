import wave
import struct
import math
import numpy as np
import soundfile as sf
import sys

filename = sys.argv[1]

try:
    data, samplerate = sf.read(filename)
    print(f"Sample Rate: {samplerate}")
    print(f"Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")
    
    # Check max/min
    max_val = np.max(data)
    min_val = np.min(data)
    print(f"Max Value: {max_val}")
    print(f"Min Value: {min_val}")
    
    # Check for clipping (values >= 0.999 or <= -0.999)
    clipping_ratio = np.mean((data >= 0.999) | (data <= -0.999))
    print(f"Clipping Ratio: {clipping_ratio:.5f}")
    
    # Check for wrap-around jumps (sudden huge differences between adjacent samples)
    diffs = np.abs(np.diff(data, axis=0))
    max_diff = np.max(diffs)
    wrap_around_jumps = np.sum(diffs > 1.5)
    print(f"Max adjacent difference: {max_diff}")
    print(f"Wrap-around jumps (>1.5): {wrap_around_jumps}")
    
    # Spectral energy check (is it just white noise?)
    # Simple FFT on a chunk
    chunk = data[samplerate:samplerate*2, 0] if len(data.shape) > 1 else data[samplerate:samplerate*2]
    fft_vals = np.abs(np.fft.rfft(chunk))
    # Ratio of high freq to low freq energy
    high_freq_energy = np.sum(fft_vals[len(fft_vals)//2:])
    low_freq_energy = np.sum(fft_vals[:len(fft_vals)//2])
    print(f"High-to-Low Freq Energy Ratio: {high_freq_energy / (low_freq_energy + 1e-9):.3f}")

except Exception as e:
    print(f"Error: {e}")


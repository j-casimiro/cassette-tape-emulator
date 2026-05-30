import argparse
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- DSP MODULES ---

def apply_bandpass(audio_data, sample_rate):
    order = 4
    nyquist = 0.5 * sample_rate
    low_hz = 80.0
    high_hz = 12000.0
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist - 1.0)
    b, a = signal.butter(order, [low, high], btype='bandpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0)

def apply_saturation(audio_data, drive=2.0):
    return np.tanh(audio_data * drive) / np.tanh(drive)

def add_hiss(audio_data, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio_data.shape)
    return audio_data + noise

# --- AUDIO ENGINE ---

def process_file(input_path):
    print(f"Reading {input_path}...")
    sample_rate, data = wavfile.read(input_path)
    
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    else:
        audio_float = data.astype(np.float32)

    print("Applying frequency limits...")
    processed = apply_bandpass(audio_float, sample_rate)
    
    print("Applying magnetic tape saturation...")
    processed = apply_saturation(processed, drive=2.0)
    
    print("Adding tape hiss...")
    processed = add_hiss(processed, noise_level=0.005)
    
    processed = np.clip(processed, -1.0, 1.0)
    final_audio = np.int16(processed * 32767.0)
    
    file_dir, file_name = os.path.split(input_path)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(file_dir, f"{name}_cassette{ext}")
    
    print(f"Saving to {output_path}...")
    wavfile.write(output_path, sample_rate, final_audio)
    print("Done!")

# --- CLI WRAPPER ---

def main():
    parser = argparse.ArgumentParser(description="Apply a cassette tape effect to WAV files.")
    parser.add_argument("input_file", help="Path to the input WAV file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    process_file(args.input_file)

if __name__ == "__main__":
    main()
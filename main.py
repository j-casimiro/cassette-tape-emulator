import argparse
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- PRESETS DEFINITIONS ---
# The cassette sound is defined by: gentle high-frequency roll-off, warm
# soft-knee compression, tape hiss, and very subtle speed instability.
# NOT by dramatic pitch wobbling.
PRESETS = {
    'hifi': {
        'high_shelf_freq': 12000.0,
        'high_shelf_db': -3.0,
        'low_cut_hz': 30.0,
        'drive': 1.3,
        'wow_depth': 0.00015,
        'flutter_depth': 0.00005,
        'drift_depth': 0.0001,
        'hiss_level': 0.003,
        'dropout_rate': 0.02,
        'dropout_depth': 0.08,
        'crosstalk': 0.05,
    },
    'vintage': {
        'high_shelf_freq': 7500.0,
        'high_shelf_db': -8.0,
        'low_cut_hz': 60.0,
        'drive': 2.0,
        'wow_depth': 0.0004,
        'flutter_depth': 0.00012,
        'drift_depth': 0.0003,
        'hiss_level': 0.008,
        'dropout_rate': 0.15,
        'dropout_depth': 0.2,
        'crosstalk': 0.12,
    },
    'lofi': {
        'high_shelf_freq': 5000.0,
        'high_shelf_db': -14.0,
        'low_cut_hz': 100.0,
        'drive': 3.0,
        'wow_depth': 0.0008,
        'flutter_depth': 0.0003,
        'drift_depth': 0.0006,
        'hiss_level': 0.016,
        'dropout_rate': 0.4,
        'dropout_depth': 0.35,
        'crosstalk': 0.18,
    }
}

# --- DSP MODULES ---

def apply_high_shelf_cut(audio_data, sample_rate, shelf_freq, shelf_db):
    """Gentle high-frequency roll-off — THE defining cassette characteristic."""
    nyquist = sample_rate / 2.0
    freq = min(shelf_freq, nyquist - 1.0)
    
    # First: a gentle shelf/slope above shelf_freq
    w0 = 2 * np.pi * freq / sample_rate
    A = 10 ** (shelf_db / 40.0)
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0/A) * (1.0/0.7 - 1) + 2)
    
    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    
    return signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)

def apply_low_cut(audio_data, sample_rate, cutoff_hz):
    """Remove sub-bass rumble that cassette heads can't reproduce."""
    nyquist = sample_rate / 2.0
    freq = min(cutoff_hz, nyquist - 1.0)
    b, a = signal.butter(2, freq, btype='highpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)

def apply_crosstalk(audio_data, crosstalk_factor):
    """Stereo crosstalk and slight azimuth misalignment."""
    if audio_data.ndim < 2 or audio_data.shape[1] < 2:
        return audio_data
        
    processed = audio_data.copy()
    left = processed[:, 0].copy()
    right = processed[:, 1].copy()
    
    processed[:, 0] = left + crosstalk_factor * right
    processed[:, 1] = right + crosstalk_factor * left
    
    # Azimuth error: 1-sample delay on right channel
    processed[1:, 1] = processed[:-1, 1].copy()
    
    return processed

def apply_wow_flutter(audio_data, sample_rate, wow_depth, flutter_depth, drift_depth):
    """Very subtle speed instability — should be barely perceptible."""
    if wow_depth == 0 and flutter_depth == 0 and drift_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    t = np.arange(n_samples, dtype=np.float64)
    
    # Wow: slow, irregular speed variation (belt/capstan imperfection)
    wow = wow_depth * (
        0.6 * np.sin(2 * np.pi * 1.3 * t / sample_rate) +
        0.25 * np.sin(2 * np.pi * 0.7 * t / sample_rate + 0.4) +
        0.15 * np.sin(2 * np.pi * 2.1 * t / sample_rate + 1.1)
    )
    
    # Flutter: faster micro-vibration from motor/belt
    flutter = flutter_depth * (
        0.7 * np.sin(2 * np.pi * 11.0 * t / sample_rate) +
        0.3 * np.sin(2 * np.pi * 17.5 * t / sample_rate + 0.8)
    )
    
    # Random drift
    num_drift_points = max(2, int(n_samples / (sample_rate * 0.8)))
    drift_vals = np.random.normal(0, 1.0, num_drift_points)
    x_drift = np.linspace(0, n_samples - 1, num_drift_points)
    drift = np.interp(t, x_drift, drift_vals) * drift_depth
    
    t_mod = t + (wow + flutter + drift) * sample_rate
    t_mod = np.clip(t_mod, 0, n_samples - 1)
    
    if audio_data.ndim == 1:
        return np.interp(t_mod, t, audio_data).astype(np.float32)
    else:
        processed = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            processed[:, ch] = np.interp(t_mod, t, audio_data[:, ch])
        return processed

def apply_tape_saturation(audio_data, drive):
    """Warm, gentle tape saturation — soft-knee compression, not harsh distortion."""
    if drive <= 1.0:
        return audio_data
    
    # Soft saturation using tanh with gentle drive
    x = audio_data * drive
    saturated = np.tanh(x)
    
    # Blend: keep some of the dry signal to preserve transients
    blend = min(0.7, (drive - 1.0) / 4.0)
    result = (1.0 - blend) * audio_data + blend * saturated
    
    # Normalize to match original peak level
    orig_peak = np.max(np.abs(audio_data))
    result_peak = np.max(np.abs(result))
    if result_peak > 0 and orig_peak > 0:
        result = result * (orig_peak / result_peak)
    
    return result.astype(np.float32)

def apply_dropouts(audio_data, sample_rate, dropout_rate, dropout_depth):
    """Brief volume dips from oxide loss or dust on the tape."""
    if dropout_rate == 0 or dropout_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    duration = n_samples / sample_rate
    
    n_dropouts = max(0, int(np.random.poisson(dropout_rate * duration)))
    if n_dropouts == 0:
        return audio_data
        
    processed = audio_data.copy()
    
    for _ in range(n_dropouts):
        idx = np.random.randint(0, n_samples)
        dropout_len = int(np.random.uniform(0.015, 0.08) * sample_rate)
        dip = 1.0 - dropout_depth * 0.5 * (1.0 + np.cos(np.linspace(-np.pi, np.pi, dropout_len)))
        
        end_idx = min(idx + dropout_len, n_samples)
        actual_len = end_idx - idx
        
        if processed.ndim == 1:
            processed[idx:end_idx] *= dip[:actual_len]
        else:
            processed[idx:end_idx] *= dip[:actual_len, np.newaxis]
            
    return processed

def add_tape_hiss(audio_data, sample_rate, hiss_level):
    """Bandpassed noise that sounds like real tape hiss (mid-high frequency)."""
    if hiss_level == 0:
        return audio_data
        
    white_noise = np.random.normal(0, hiss_level, audio_data.shape)
    
    # Tape hiss is concentrated in the 1kHz-8kHz range
    nyquist = sample_rate / 2.0
    high_cutoff = min(8000.0, nyquist - 1.0)
    b, a = signal.butter(2, [1000.0, high_cutoff], btype='bandpass', fs=sample_rate)
    
    if audio_data.ndim == 1:
        hiss = signal.filtfilt(b, a, white_noise)
    else:
        hiss = np.zeros_like(white_noise)
        for ch in range(white_noise.shape[1]):
            hiss[:, ch] = signal.filtfilt(b, a, white_noise[:, ch])
            
    return (audio_data + hiss).astype(np.float32)

def generate_button_click(sample_rate, is_play=True):
    """Generate a realistic cassette player button click.
    
    Play button: a sharp plastic 'click' followed by the head engaging
    Stop button: a sharper, snappier spring-loaded release 'clack'
    """
    duration = 0.08 if is_play else 0.06
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    
    # Sharp initial transient — short burst of filtered noise
    burst = np.random.normal(0, 1.0, n)
    
    if is_play:
        # Play: plastic click with brief resonance
        # Fast attack, medium decay
        envelope = np.exp(-120 * t)
        # Second smaller click 30ms later (head engaging onto tape)
        head_engage = np.exp(-200 * np.maximum(t - 0.03, 0)) * 0.4
        head_engage[:int(0.03 * sample_rate)] = 0
        envelope = envelope + head_engage
    else:
        # Stop: snappier, shorter click
        envelope = np.exp(-180 * t)
    
    click = burst * envelope
    
    # Bandpass to make it sound like a plastic mechanism (2kHz-10kHz)
    nyquist = sample_rate / 2.0
    high = min(10000.0, nyquist - 1.0)
    b, a = signal.butter(2, [2000.0, high], btype='bandpass', fs=sample_rate)
    click = signal.filtfilt(b, a, click)
    
    # Add a tiny low-frequency thump from the mechanical force
    thump = np.sin(2 * np.pi * 80 * t) * np.exp(-80 * t) * 0.15
    click = click + thump
    
    # Normalize
    peak = np.max(np.abs(click))
    if peak > 0:
        click = click / peak * 0.25
    
    return click.astype(np.float32)

def apply_startup_glide(audio_data, sample_rate):
    """Very brief speed ramp at the start (tape engaging), ~120ms."""
    ramp_duration = 0.12
    ramp_samples = int(ramp_duration * sample_rate)
    
    if ramp_samples >= len(audio_data):
        return audio_data
    
    # Create a speed curve that goes from 0.85 to 1.0
    speed_curve = np.linspace(0.85, 1.0, ramp_samples)
    
    # Build the time remapping for the ramp region
    t_ramp = np.cumsum(speed_curve)
    # Normalize so the end of the ramp maps to ramp_samples
    t_ramp = t_ramp / t_ramp[-1] * ramp_samples
    
    t_original = np.arange(ramp_samples, dtype=np.float64)
    
    if audio_data.ndim == 1:
        audio_data[:ramp_samples] = np.interp(t_ramp, t_original, audio_data[:ramp_samples])
    else:
        for ch in range(audio_data.shape[1]):
            audio_data[:ramp_samples, ch] = np.interp(t_ramp, t_original, audio_data[:ramp_samples, ch])
    
    # Fade in over the ramp to mask the pitch shift
    fade = np.linspace(0.0, 1.0, ramp_samples) ** 0.5
    if audio_data.ndim == 1:
        audio_data[:ramp_samples] *= fade
    else:
        audio_data[:ramp_samples] *= fade[:, np.newaxis]
    
    return audio_data

def apply_stop_glide(audio_data, sample_rate):
    """Brief slowdown at the end when stop is pressed, ~100ms."""
    ramp_duration = 0.10
    ramp_samples = int(ramp_duration * sample_rate)
    
    if ramp_samples >= len(audio_data):
        return audio_data
    
    start = len(audio_data) - ramp_samples
    
    # Speed curve from 1.0 down to 0.7
    speed_curve = np.linspace(1.0, 0.7, ramp_samples)
    
    t_ramp = np.cumsum(speed_curve)
    t_ramp = t_ramp / t_ramp[-1] * ramp_samples
    
    t_original = np.arange(ramp_samples, dtype=np.float64)
    
    if audio_data.ndim == 1:
        audio_data[start:] = np.interp(t_ramp, t_original, audio_data[start:])
    else:
        for ch in range(audio_data.shape[1]):
            audio_data[start:, ch] = np.interp(t_ramp, t_original, audio_data[start:, ch])
    
    # Fade out over the ramp
    fade = np.linspace(1.0, 0.0, ramp_samples) ** 0.5
    if audio_data.ndim == 1:
        audio_data[start:] *= fade
    else:
        audio_data[start:] *= fade[:, np.newaxis]
    
    return audio_data

# --- AUDIO ENGINE ---

def process_file(input_path, params, no_physical=False):
    print(f"Reading {input_path}...")
    sample_rate, data = wavfile.read(input_path)
    
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_float = data.astype(np.float32) / 2147483648.0
    else:
        audio_float = data.astype(np.float32)
        
    # --- Tape Processing Chain (whole mix, like a real cassette) ---

    # 1. Stereo crosstalk & azimuth error
    print("Applying stereo crosstalk...")
    processed = apply_crosstalk(audio_float, params['crosstalk'])

    # 2. Very subtle wow & flutter (barely perceptible speed instability)
    print("Applying subtle wow & flutter...")
    processed = apply_wow_flutter(
        processed, sample_rate,
        wow_depth=params['wow_depth'],
        flutter_depth=params['flutter_depth'],
        drift_depth=params['drift_depth']
    )

    # 3. High-frequency roll-off (THE cassette signature)
    print("Applying high-frequency roll-off...")
    processed = apply_high_shelf_cut(processed, sample_rate, params['high_shelf_freq'], params['high_shelf_db'])
    
    # 4. Low-frequency cut (tape heads can't reproduce deep sub-bass)
    print("Applying low-frequency cut...")
    processed = apply_low_cut(processed, sample_rate, params['low_cut_hz'])
    
    # 5. Warm tape saturation (gentle compression, not distortion)
    print("Applying warm tape saturation...")
    processed = apply_tape_saturation(processed, params['drive'])
    
    # 6. Tape dropouts
    print("Applying tape dropouts...")
    processed = apply_dropouts(processed, sample_rate, params['dropout_rate'], params['dropout_depth'])

    # 7. Tape hiss
    print("Adding tape hiss...")
    processed = add_tape_hiss(processed, sample_rate, params['hiss_level'])

    if not no_physical:
        # 8. Startup speed glide (brief pitch ramp, ~120ms)
        print("Applying startup speed glide...")
        processed = apply_startup_glide(processed, sample_rate)
        
        # 9. Stop slowdown glide at end (~100ms)
        print("Applying stop speed glide...")
        processed = apply_stop_glide(processed, sample_rate)
        
        # 10. Add play button click at start and stop click at end
        print("Adding play/stop button clicks...")
        play_click = generate_button_click(sample_rate, is_play=True)
        stop_click = generate_button_click(sample_rate, is_play=False)
        
        # Prepend silence + play click, append stop click + silence
        pre_silence = int(0.3 * sample_rate)
        post_silence = int(0.4 * sample_rate)
        click_offset = int(0.15 * sample_rate)  # click happens 150ms into the pre-silence
        gap_after_click = pre_silence - click_offset - len(play_click)
        
        if audio_float.ndim == 1:
            # Build: [silence...click...gap...processed_audio...stop_click...silence]
            pre = np.zeros(pre_silence, dtype=np.float32)
            pre[click_offset:click_offset+len(play_click)] += play_click
            
            post = np.zeros(post_silence, dtype=np.float32)
            stop_offset = int(0.05 * sample_rate)
            end_sc = min(stop_offset + len(stop_click), post_silence)
            post[stop_offset:end_sc] += stop_click[:end_sc - stop_offset]
            
            processed = np.concatenate([pre, processed, post])
        else:
            n_ch = processed.shape[1]
            pre = np.zeros((pre_silence, n_ch), dtype=np.float32)
            for ch in range(n_ch):
                pre[click_offset:click_offset+len(play_click), ch] += play_click
            
            post = np.zeros((post_silence, n_ch), dtype=np.float32)
            stop_offset = int(0.05 * sample_rate)
            end_sc = min(stop_offset + len(stop_click), post_silence)
            for ch in range(n_ch):
                post[stop_offset:end_sc, ch] += stop_click[:end_sc - stop_offset]
            
            processed = np.concatenate([pre, processed, post], axis=0)
    
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
    parser = argparse.ArgumentParser(
        description="Apply a realistic cassette tape effect to WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py song.wav                      # Default vintage preset
  python main.py song.wav --preset lofi         # Worn-out tape sound
  python main.py song.wav --preset hifi         # Clean, warm tape
  python main.py song.wav --hiss 0.02           # Extra hiss
  python main.py song.wav --no-physical         # No button clicks or speed glides
        """
    )
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("--preset", choices=['hifi', 'vintage', 'lofi'], default='vintage',
                        help="Cassette quality preset (default: vintage)")
    parser.add_argument("--drive", type=float, help="Tape saturation amount (1.0 = none, 3.0 = warm)")
    parser.add_argument("--wow", type=float, help="Wow depth (try 0.0004 for subtle)")
    parser.add_argument("--flutter", type=float, help="Flutter depth (try 0.0001 for subtle)")
    parser.add_argument("--hiss", type=float, help="Tape hiss level (try 0.008 for vintage)")
    parser.add_argument("--dropouts", type=float, help="Dropout rate (events per second)")
    parser.add_argument("--no-physical", action="store_true",
                        help="Skip button clicks and speed glides")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    params = PRESETS[args.preset].copy()
    
    if args.drive is not None:
        params['drive'] = args.drive
    if args.wow is not None:
        params['wow_depth'] = args.wow
    if args.flutter is not None:
        params['flutter_depth'] = args.flutter
    if args.hiss is not None:
        params['hiss_level'] = args.hiss
    if args.dropouts is not None:
        params['dropout_rate'] = args.dropouts
        
    process_file(args.input_file, params, no_physical=args.no_physical)

if __name__ == "__main__":
    main()
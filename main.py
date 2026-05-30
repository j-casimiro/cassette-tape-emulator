import argparse
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- PRESETS DEFINITIONS ---
PRESETS = {
    'hifi': {
        'low_hz': 40.0,
        'high_hz': 13000.0,
        'drive': 1.6,
        'asymmetry': 0.02,
        'crossover': 0.0,
        'wow_depth': 0.0004,
        'flutter_depth': 0.0001,
        'drift_depth': 0.0002,
        'hiss_level': 0.0015,
        'motor_level': 0.0008,
        'dropout_rate': 0.04,
        'dropout_depth': 0.1
    },
    'vintage': {
        'low_hz': 75.0,
        'high_hz': 9000.0,
        'drive': 3.0,
        'asymmetry': 0.07,
        'crossover': 0.005,
        'wow_depth': 0.0012,
        'flutter_depth': 0.0004,
        'drift_depth': 0.0008,
        'hiss_level': 0.007,
        'motor_level': 0.0025,
        'dropout_rate': 0.2,
        'dropout_depth': 0.3
    },
    'lofi': {
        'low_hz': 110.0,
        'high_hz': 6000.0,
        'drive': 4.5,
        'asymmetry': 0.12,
        'crossover': 0.015,
        'wow_depth': 0.0028,
        'flutter_depth': 0.0009,
        'drift_depth': 0.002,
        'hiss_level': 0.018,
        'motor_level': 0.004,
        'dropout_rate': 0.5,
        'dropout_depth': 0.45
    }
}

# --- DSP MODULES ---

def separate_vocals(audio_data, sample_rate):
    if audio_data.ndim < 2 or audio_data.shape[1] < 2:
        return audio_data, np.zeros_like(audio_data)
        
    left = audio_data[:, 0]
    right = audio_data[:, 1]
    
    nperseg = 2048
    noverlap = 1536
    
    # Compute STFT for both channels
    f, t, L_spec = signal.stft(left, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, _, R_spec = signal.stft(right, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Calculate similarity (coherence) based on magnitude & phase similarity
    numerator = 2.0 * np.real(L_spec * np.conj(R_spec))
    denominator = np.abs(L_spec)**2 + np.abs(R_spec)**2 + 1e-10
    similarity = numerator / denominator
    similarity = np.clip(similarity, 0.0, 1.0)
    
    # Generate masks: center-panned (vocals) vs side-panned (music)
    mask_center = similarity ** 3.0
    mask_side = 1.0 - mask_center
    
    # Apply masks and perform inverse STFT
    _, vocal_L = signal.istft(L_spec * mask_center, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, vocal_R = signal.istft(R_spec * mask_center, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    _, music_L = signal.istft(L_spec * mask_side, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, music_R = signal.istft(R_spec * mask_side, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    n_samples = len(audio_data)
    vocal = np.zeros((n_samples, 2), dtype=np.float32)
    music = np.zeros((n_samples, 2), dtype=np.float32)
    
    len_v = min(n_samples, len(vocal_L))
    vocal[:len_v, 0] = vocal_L[:len_v]
    vocal[:len_v, 1] = vocal_R[:len_v]
    
    len_m = min(n_samples, len(music_L))
    music[:len_m, 0] = music_L[:len_m]
    music[:len_m, 1] = music_R[:len_m]
    
    return music, vocal

def apply_bandpass(audio_data, sample_rate, low_hz, high_hz):
    order = 3
    nyquist = 0.5 * sample_rate
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist - 1.0)
    b, a = signal.butter(order, [low, high], btype='bandpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0)

def apply_tape_imperfections(audio_data, sample_rate):
    if audio_data.ndim < 2 or audio_data.shape[1] < 2:
        return audio_data
        
    processed = audio_data.copy()
    left = processed[:, 0]
    right = processed[:, 1]
    
    # 1. Channel Crosstalk (reduces stereo width)
    crosstalk_factor = 0.15
    new_left = left + crosstalk_factor * right
    new_right = right + crosstalk_factor * left
    
    # 2. Azimuth Error (slight delay in right channel, 2 samples at ~44.1kHz)
    delay_samples = 2
    new_right_delayed = np.zeros_like(new_right)
    new_right_delayed[delay_samples:] = new_right[:-delay_samples]
    new_right_delayed[:delay_samples] = new_right[0]
    
    processed[:, 0] = new_left
    processed[:, 1] = new_right_delayed
    
    # Re-normalize to prevent clipping
    peak = np.max(np.abs(processed))
    if peak > 0:
        processed = processed / peak
        
    return processed

def apply_wow_flutter(audio_data, sample_rate, wow_depth, flutter_depth, drift_depth):
    if wow_depth == 0 and flutter_depth == 0 and drift_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    t = np.arange(n_samples, dtype=np.float64)
    
    # Wow (mix of two slow organic mechanical friction speeds)
    wow = wow_depth * (
        0.7 * np.sin(2 * np.pi * 1.6 * t / sample_rate) + 
        0.3 * np.sin(2 * np.pi * 0.55 * t / sample_rate)
    )
    
    # Flutter (mix of two rapid belt/motor vibration frequencies)
    flutter = flutter_depth * (
        0.65 * np.sin(2 * np.pi * 14.5 * t / sample_rate) + 
        0.35 * np.sin(2 * np.pi * 22.0 * t / sample_rate)
    )
    
    # Random drift/walk (slow chaotic fluctuations)
    num_drift_points = max(2, int(n_samples / (sample_rate * 0.5))) # 1 point every 0.5 seconds
    drift_points = np.random.normal(0, 1.0, num_drift_points)
    x_drift = np.linspace(0, n_samples - 1, num_drift_points)
    drift = np.interp(t, x_drift, drift_points) * drift_depth
    
    # Total speed variation/displacement in samples
    t_mod = t + (wow + flutter + drift) * sample_rate
    t_mod = np.clip(t_mod, 0, n_samples - 1)
    
    # Interpolate
    if audio_data.ndim == 1:
        return np.interp(t_mod, t, audio_data).astype(np.float32)
    else:
        processed = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            processed[:, ch] = np.interp(t_mod, t, audio_data[:, ch])
        return processed

def apply_tape_speed_profile(audio_data, sample_rate, pad_start_samples, original_samples):
    n_samples = len(audio_data)
    
    t1_sec = 0.35
    t2_sec = 0.60
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.1
    t4_sec = t3_sec + 0.15
    
    t1 = t1_sec * sample_rate
    t2 = t2_sec * sample_rate
    t3 = t3_sec * sample_rate
    t4 = t4_sec * sample_rate
    
    L1 = t2 - t1
    L2 = t4 - t3
    
    t = np.arange(n_samples, dtype=np.float64)
    t_mod = np.zeros(n_samples, dtype=np.float64)
    
    # 1. Before t1: Tape is stationary
    mask1 = t <= t1
    t_mod[mask1] = 0.0
    
    # 2. Ramp up t1 to t2: Tape accelerates (startup glide)
    mask2 = (t > t1) & (t < t2)
    t_mod[mask2] = 0.5 * ((t[mask2] - t1) ** 2) / L1
    
    # 3. Constant speed t2 to t3: Normal playback
    mask3 = (t >= t2) & (t <= t3)
    p_t2 = 0.5 * L1
    t_mod[mask3] = p_t2 + (t[mask3] - t2)
    
    # 4. Ramp down t3 to t4: Tape decelerates (stop glide)
    mask4 = (t > t3) & (t < t4)
    p_t3 = p_t2 + (t3 - t2)
    t_mod[mask4] = p_t3 + (t[mask4] - t3) - 0.5 * ((t[mask4] - t3) ** 2) / L2
    
    # 5. After t4: Tape has stopped
    mask5 = t >= t4
    p_t4 = p_t3 + 0.5 * L2
    t_mod[mask5] = p_t4
    
    # Shift so that the constant speed region maps 1:1 to original audio
    offset = pad_start_samples - p_t2
    t_mod += offset
    
    t_mod = np.clip(t_mod, 0, n_samples - 1)
    
    # Interpolate
    if audio_data.ndim == 1:
        return np.interp(t_mod, t, audio_data).astype(np.float32)
    else:
        processed = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            processed[:, ch] = np.interp(t_mod, t, audio_data[:, ch])
        return processed

def pre_emphasis(audio_data, alpha=0.45):
    # y[n] = x[n] - alpha * x[n-1]
    b = [1.0, -alpha]
    a = [1.0]
    return signal.lfilter(b, a, audio_data, axis=0)

def de_emphasis(audio_data, alpha=0.45):
    # y[n] = x[n] + alpha * y[n-1]
    b = [1.0]
    a = [1.0, -alpha]
    return signal.lfilter(b, a, audio_data, axis=0)

def apply_saturation(audio_data, drive, asymmetry, crossover):
    # Scale input by drive, apply pre-emphasis to high frequencies
    emphasized = pre_emphasis(audio_data, alpha=0.45)
    
    # Apply asymmetric distortion
    x = emphasized * drive
    saturated = (np.tanh(x + asymmetry) - np.tanh(asymmetry))
    
    # Apply crossover distortion (simulates cheap preamp diode thresholds)
    if crossover > 0:
        saturated = saturated - crossover * np.tanh(saturated / crossover)
    
    # Apply de-emphasis to restore balance
    de_emphasized = de_emphasis(saturated, alpha=0.45)
    
    # Normalize peak back to original level (capped at 0.95 to avoid clipping)
    peak = np.max(np.abs(audio_data))
    sat_peak = np.max(np.abs(de_emphasized))
    if sat_peak > 0:
        return (de_emphasized / sat_peak) * min(max(peak, 0.5), 0.95)
    return audio_data

def apply_dropouts(audio_data, sample_rate, dropout_rate, dropout_depth):
    if dropout_rate == 0 or dropout_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    duration = n_samples / sample_rate
    
    n_dropouts = int(np.random.poisson(dropout_rate * duration))
    if n_dropouts == 0:
        return audio_data
        
    processed = audio_data.copy()
    indices = np.random.randint(0, n_samples, n_dropouts)
    
    for idx in indices:
        # Dropout duration: between 20ms and 120ms
        dropout_len = int(np.random.uniform(0.02, 0.12) * sample_rate)
        dip_shape = 1.0 - dropout_depth * 0.5 * (1.0 + np.cos(np.linspace(-np.pi, np.pi, dropout_len)))
        
        end_idx = min(idx + dropout_len, n_samples)
        actual_len = end_idx - idx
        
        if processed.ndim == 1:
            processed[idx:end_idx] *= dip_shape[:actual_len]
        else:
            # Dropout affects one channel more severely
            ch = np.random.randint(0, processed.shape[1])
            processed[idx:end_idx, ch] *= dip_shape[:actual_len]
            # Other channel gets a minor dip
            other_ch = 1 - ch
            processed[idx:end_idx, other_ch] *= (1.0 - (1.0 - dip_shape[:actual_len]) * 0.3)
            
    return processed

def generate_motor_noise(n_samples, sample_rate, level=0.005):
    t = np.arange(n_samples, dtype=np.float64)
    # AC Hum (60Hz + 120Hz + 180Hz)
    hum = (np.sin(2 * np.pi * 60.0 * t / sample_rate) +
           0.25 * np.sin(2 * np.pi * 120.0 * t / sample_rate) +
           0.08 * np.sin(2 * np.pi * 180.0 * t / sample_rate))
    
    # Low-frequency mechanical spindle rumble (pink-ish lowpassed noise)
    white = np.random.normal(0, 1.0, n_samples)
    b, a = signal.butter(1, 80.0 / (sample_rate / 2.0), btype='low')
    rumble = signal.filtfilt(b, a, white)
    rumble_mod = 1.0 + 0.2 * np.sin(2 * np.pi * 4.7 * t / sample_rate)
    rumble = rumble * rumble_mod * 0.5
    
    motor_noise = hum + rumble
    peak = np.max(np.abs(motor_noise))
    if peak > 0:
        motor_noise = (motor_noise / peak) * level
    return motor_noise

def add_hiss(audio_data, sample_rate, hiss_level):
    if hiss_level == 0:
        return audio_data
        
    n_samples = len(audio_data)
    white_noise = np.random.normal(0, hiss_level, audio_data.shape)
    
    # Tape hiss peaks around mid/high and rolls off in extreme low/high
    b, a = signal.butter(2, [350.0, 7000.0], btype='bandpass', fs=sample_rate)
    
    if audio_data.ndim == 1:
        hiss = signal.filtfilt(b, a, white_noise)
    else:
        hiss = np.zeros_like(white_noise)
        for ch in range(white_noise.shape[1]):
            hiss[:, ch] = signal.filtfilt(b, a, white_noise[:, ch])
            
    return audio_data + hiss

def generate_clunk(sample_rate, is_play=True):
    duration = 0.2
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Low-frequency mechanical thud
    thud_freq = 60.0 if is_play else 85.0
    thud_decay = 30.0 if is_play else 50.0
    thud = np.sin(2 * np.pi * thud_freq * t) * np.exp(-thud_decay * t)
    
    # High-frequency mechanical contact click
    noise = np.random.normal(0, 1.0, n_samples)
    b, a = signal.butter(2, [1800.0, 6000.0], btype='bandpass', fs=sample_rate)
    click = signal.filtfilt(b, a, noise)
    click_decay = 150.0 if is_play else 200.0
    click = click * np.exp(-click_decay * t) * 0.3
    
    if is_play:
        # Engagement latch click after 45ms
        delay_samples = int(0.045 * sample_rate)
        click2 = np.zeros_like(click)
        click2[delay_samples:] = click[:-delay_samples] * 0.6
        clunk = thud + click + click2
    else:
        clunk = thud + click
        
    clunk = clunk / np.max(np.abs(clunk)) * 0.35
    return clunk

# --- AUDIO ENGINE ---

def process_file(input_path, params, no_physical=False):
    print(f"Reading {input_path}...")
    sample_rate, data = wavfile.read(input_path)
    
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    else:
        audio_float = data.astype(np.float32)
        
    original_samples = len(audio_float)
    
    if no_physical:
        pad_start_samples = 0
        padded = audio_float.copy()
    else:
        # Pad the audio with silence at start and end for mechanical startup/stop
        pad_start_samples = int(1.0 * sample_rate)
        pad_end_samples = int(1.0 * sample_rate)
        
        if audio_float.ndim == 1:
            padded = np.zeros(pad_start_samples + original_samples + pad_end_samples, dtype=np.float32)
            padded[pad_start_samples:pad_start_samples+original_samples] = audio_float
        else:
            padded = np.zeros((pad_start_samples + original_samples + pad_end_samples, audio_float.shape[1]), dtype=np.float32)
            padded[pad_start_samples:pad_start_samples+original_samples, :] = audio_float

    # 1. Panning-based Center extraction to separate Vocals and Music
    print("Separating vocals (center) from music (sides) using panning extraction...")
    music, vocal = separate_vocals(padded, sample_rate)

    # 2. Process the music track with full wobbly tape parameters
    print("Processing music track (full tape wobble & saturation)...")
    music_proc = apply_tape_imperfections(music, sample_rate)
    music_proc = apply_wow_flutter(
        music_proc, 
        sample_rate, 
        wow_depth=params['wow_depth'], 
        flutter_depth=params['flutter_depth'],
        drift_depth=params['drift_depth']
    )
    music_proc = apply_bandpass(music_proc, sample_rate, params['low_hz'], params['high_hz'])
    music_proc = apply_saturation(music_proc, drive=params['drive'], asymmetry=params['asymmetry'], crossover=params['crossover'])
    music_proc = apply_dropouts(music_proc, sample_rate, params['dropout_rate'], params['dropout_depth'])

    # 3. Process the vocal track with a cleaner vintage character (strictly NO wow/flutter)
    print("Processing vocal track (clean, warm tape character)...")
    # Vocals get a vintage radio bandpass focus (150Hz to 6500Hz)
    vocal_low_hz = max(150.0, params['low_hz'])
    vocal_high_hz = min(6500.0, params['high_hz'])
    vocal_proc = apply_bandpass(vocal, sample_rate, vocal_low_hz, vocal_high_hz)
    
    # Mild saturation to warm up the voice
    vocal_proc = apply_saturation(vocal_proc, drive=1.8, asymmetry=0.04, crossover=params['crossover'] * 0.5)
    
    # Soft dropouts on vocals
    vocal_proc = apply_dropouts(vocal_proc, sample_rate, params['dropout_rate'] * 0.4, params['dropout_depth'] * 0.5)

    # 4. Re-combine music and vocals
    print("Mixing vocal and music tracks back together...")
    combined = music_proc + vocal_proc

    # 5. Global mechanical and physical noise additions on the combined mix
    if not no_physical:
        print("Simulating tape player mechanical startup and stop glides...")
        combined = apply_tape_speed_profile(combined, sample_rate, pad_start_samples, original_samples)

        print("Adding mechanical motor noise & hum...")
        motor_noise = generate_motor_noise(len(combined), sample_rate, level=params['motor_level'])
        if combined.ndim == 1:
            combined += motor_noise
        else:
            for ch in range(combined.shape[1]):
                combined[:, ch] += motor_noise
            
    print("Adding tape hiss...")
    combined = add_hiss(combined, sample_rate, params['hiss_level'])
    
    if not no_physical:
        print("Adding mechanical button play/stop clunks...")
        play_clunk = generate_clunk(sample_rate, is_play=True)
        stop_clunk = generate_clunk(sample_rate, is_play=False)
        
        # Inject play clunk at 0.15s
        play_idx = int(0.15 * sample_rate)
        if combined.ndim == 1:
            combined[play_idx:play_idx+len(play_clunk)] += play_clunk
        else:
            for ch in range(combined.shape[1]):
                combined[play_idx:play_idx+len(play_clunk), ch] += play_clunk
                
        # Inject stop clunk at t4_sec (where the tape stops accelerating)
        t4_sec = (pad_start_samples + original_samples + int(0.1 * sample_rate)) / sample_rate + 0.15
        stop_idx = int(t4_sec * sample_rate)
        
        # Truncate processed audio shortly after the stop clunk to avoid long silent tails
        end_idx = min(stop_idx + len(stop_clunk) + int(0.25 * sample_rate), len(combined))
        combined = combined[:end_idx]
        
        if combined.ndim == 1:
            combined[stop_idx:stop_idx+len(stop_clunk)] += stop_clunk
        else:
            for ch in range(combined.shape[1]):
                combined[stop_idx:stop_idx+len(stop_clunk), ch] += stop_clunk
            
    combined = np.clip(combined, -1.0, 1.0)
    final_audio = np.int16(combined * 32767.0)
    
    file_dir, file_name = os.path.split(input_path)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(file_dir, f"{name}_cassette{ext}")
    
    print(f"Saving to {output_path}...")
    wavfile.write(output_path, sample_rate, final_audio)
    print("Done!")

# --- CLI WRAPPER ---

def main():
    parser = argparse.ArgumentParser(description="Apply a cassette tape effect to WAV files with realistic analog modeling.")
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("--preset", choices=['hifi', 'vintage', 'lofi'], default='vintage',
                        help="Cassette simulation preset (default: 'vintage')")
    parser.add_argument("--drive", type=float, help="Tape saturation drive (higher = more distortion/compression)")
    parser.add_argument("--wow", type=float, help="Wow depth (slow pitch wobble)")
    parser.add_argument("--flutter", type=float, help="Flutter depth (fast pitch wobble)")
    parser.add_argument("--hiss", type=float, help="Tape hiss noise level")
    parser.add_argument("--dropouts", type=float, help="Rate of tape dropouts (average dropouts per second)")
    parser.add_argument("--crossover", type=float, help="Crossover distortion threshold (adds preamp grit)")
    parser.add_argument("--motor", type=float, help="Motor hum and whir noise level")
    parser.add_argument("--no-physical", action="store_true",
                        help="Disable physical cassette player elements (clunks, motor hum, and startup/stop speed glides)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    # Get parameters from preset
    params = PRESETS[args.preset].copy()
    
    # Override with manual arguments if provided
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
    if args.crossover is not None:
        params['crossover'] = args.crossover
    if args.motor is not None:
        params['motor_level'] = args.motor
        
    process_file(args.input_file, params, no_physical=args.no_physical)

if __name__ == "__main__":
    main()
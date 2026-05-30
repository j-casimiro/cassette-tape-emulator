import argparse
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- PRESETS DEFINITIONS ---
PRESETS = {
    'hifi': {
        'high_shelf_freq': 12000.0,
        'high_shelf_db': -2.0,
        'low_cut_hz': 35.0,
        'high_cut_hz': 15000.0,
        'treble_boost_freq': 10000.0,
        'treble_boost_gain': 1.5,
        'drive': 1.3,
        'threshold': 0.65,
        'wow_depth': 0.00015,
        'flutter_depth': 0.00005,
        'drift_depth': 0.0001,
        'hiss_level': 0.002,
        'motor_level': 0.0005,
        'dropout_rate': 0.02,
        'dropout_depth': 0.08,
        'crosstalk': 0.05,
    },
    'vintage': {
        'high_shelf_freq': 7500.0,
        'high_shelf_db': -6.0,
        'low_cut_hz': 65.0,
        'high_cut_hz': 8500.0,
        'treble_boost_freq': 6000.0,
        'treble_boost_gain': 3.0,
        'drive': 2.0,
        'threshold': 0.5,
        'wow_depth': 0.0004,
        'flutter_depth': 0.00012,
        'drift_depth': 0.0003,
        'hiss_level': 0.006,
        'motor_level': 0.0015,
        'dropout_rate': 0.15,
        'dropout_depth': 0.2,
        'crosstalk': 0.12,
    },
    'lofi': {
        'high_shelf_freq': 3200.0,
        'high_shelf_db': -10.0,
        'low_cut_hz': 150.0,
        'high_cut_hz': 4000.0,
        'treble_boost_freq': 3200.0,
        'treble_boost_gain': 4.5,
        'drive': 3.0,
        'threshold': 0.35,
        'wow_depth': 0.0008,
        'flutter_depth': 0.0003,
        'drift_depth': 0.0006,
        'hiss_level': 0.012,
        'motor_level': 0.003,
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
    
    # Gentle shelving filter
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
    """Steeper 4th-order low-cut filter to remove modern sub-bass rumble."""
    nyquist = sample_rate / 2.0
    freq = min(cutoff_hz, nyquist - 1.0)
    b, a = signal.butter(4, freq, btype='highpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)

def apply_high_cut(audio_data, sample_rate, cutoff_hz):
    """Sharp 4th-order high-cut filter to remove modern high sizzle and air."""
    nyquist = sample_rate / 2.0
    freq = min(cutoff_hz, nyquist - 1.0)
    b, a = signal.butter(4, freq, btype='lowpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)

def apply_peaking_eq(audio_data, sample_rate, center_freq, gain_db, Q=1.0):
    """Standard peaking biquad EQ. Used to emulate the physical low-frequency head bump."""
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    A = 10 ** (gain_db / 35.0)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    
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
    """Organic pitch instability containing both cyclic belt speeds and chaotic mechanical slip."""
    if wow_depth == 0 and flutter_depth == 0 and drift_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    t = np.arange(n_samples, dtype=np.float64)
    
    # 1. Periodic wow (slow belt cycles):
    wow_periodic = (
        0.6 * np.sin(2 * np.pi * 1.3 * t / sample_rate) +
        0.4 * np.sin(2 * np.pi * 0.55 * t / sample_rate + 0.8)
    )
    
    # 2. Chaotic wow (random capstan/pinch friction slips):
    noise_wow = np.random.normal(0, 1.0, n_samples)
    b_wow, a_wow = signal.butter(1, 3.0 / (sample_rate / 2.0), btype='low')
    wow_random = signal.filtfilt(b_wow, a_wow, noise_wow)
    wr_peak = np.max(np.abs(wow_random))
    if wr_peak > 0:
        wow_random = wow_random / wr_peak
        
    total_wow = wow_depth * (0.6 * wow_periodic + 0.4 * wow_random)
    
    # 3. Periodic flutter (motor speed hum/vibration):
    flutter_periodic = (
        0.7 * np.sin(2 * np.pi * 11.5 * t / sample_rate) +
        0.3 * np.sin(2 * np.pi * 18.2 * t / sample_rate + 0.5)
    )
    
    # 4. Chaotic flutter (scratchy speed instabilities):
    noise_flut = np.random.normal(0, 1.0, n_samples)
    b_flut, a_flut = signal.butter(1, [8.0 / (sample_rate / 2.0), 25.0 / (sample_rate / 2.0)], btype='bandpass')
    flutter_random = signal.filtfilt(b_flut, a_flut, noise_flut)
    fr_peak = np.max(np.abs(flutter_random))
    if fr_peak > 0:
        flutter_random = flutter_random / fr_peak
        
    total_flutter = flutter_depth * (0.6 * flutter_periodic + 0.4 * flutter_random)
    
    # 5. Slow organic drift (reel friction drag):
    num_drift_points = max(2, int(n_samples / (sample_rate * 1.0)))
    drift_vals = np.random.normal(0, 1.0, num_drift_points)
    x_drift = np.linspace(0, n_samples - 1, num_drift_points)
    drift = np.interp(t, x_drift, drift_vals) * drift_depth
    
    t_mod = t + (total_wow + total_flutter + drift) * sample_rate
    t_mod = np.clip(t_mod, 0, n_samples - 1)
    
    if audio_data.ndim == 1:
        return np.interp(t_mod, t, audio_data).astype(np.float32)
    else:
        processed = np.zeros_like(audio_data)
        for ch in range(audio_data.shape[1]):
            processed[:, ch] = np.interp(t_mod, t, audio_data[:, ch])
        return processed

def apply_tape_speed_profile(audio_data, sample_rate, pad_start_samples, original_samples):
    """Accurately maps the audio index to simulate snappy startup & shutdown glides."""
    n_samples = len(audio_data)
    
    t1_sec = 0.25
    t2_sec = 0.50
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
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
    
    # 2. Ramp up: Tape accelerates (startup glide)
    mask2 = (t > t1) & (t < t2)
    t_mod[mask2] = 0.5 * ((t[mask2] - t1) ** 2) / L1
    
    # 3. Constant speed: Normal playback
    mask3 = (t >= t2) & (t <= t3)
    p_t2 = 0.5 * L1
    t_mod[mask3] = p_t2 + (t[mask3] - t2)
    
    # 4. Ramp down: Tape decelerates (stop glide)
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

def get_tape_speed_envelope(n_samples, sample_rate, pad_start_samples, original_samples):
    """Generates the dynamic speed/volume multiplier for hiss and mechanical hum."""
    t1_sec = 0.25
    t2_sec = 0.50
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
    t4_sec = t3_sec + 0.30
    
    t1 = t1_sec * sample_rate
    t2 = t2_sec * sample_rate
    t3 = t3_sec * sample_rate
    t4 = t4_sec * sample_rate
    
    t = np.arange(n_samples, dtype=np.float64)
    speed = np.zeros(n_samples, dtype=np.float32)
    
    speed[t <= t1] = 0.0
    
    mask2 = (t > t1) & (t < t2)
    speed[mask2] = (t[mask2] - t1) / (t2 - t1)
    
    speed[(t >= t2) & (t <= t3)] = 1.0
    
    mask4 = (t > t3) & (t < t4)
    speed[mask4] = 1.0 - (t[mask4] - t3) / (t4 - t3)
    
    speed[t >= t4] = 0.0
    return speed

def apply_tape_saturation(audio_data, drive, threshold=0.5):
    """Dynamic soft-knee saturation: quiet sections remain completely clean.
    Only loud peaks over the headroom threshold are compressed and saturated.
    """
    if drive <= 1.0:
        return audio_data
        
    x = audio_data * drive
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    
    y = np.zeros_like(x)
    
    # 1. Linear region: No distortion below the knee threshold
    linear_mask = abs_x < threshold
    y[linear_mask] = x[linear_mask]
    
    # 2. Saturation region: Soft connection to a tanh ceiling
    sat_mask = ~linear_mask
    L = 1.0 - threshold
    if L <= 0:
        return audio_data
        
    y[sat_mask] = sign_x[sat_mask] * (threshold + L * np.tanh((abs_x[sat_mask] - threshold) / L))
    
    # Restore scale to match input peak
    orig_peak = np.max(np.abs(audio_data))
    result_peak = np.max(np.abs(y))
    if result_peak > 0 and orig_peak > 0:
        y = y * (orig_peak / result_peak)
        
    return y.astype(np.float32)

def apply_dropouts(audio_data, sample_rate, dropout_rate, dropout_depth):
    """Tape dropouts with Wallace Spacer Loss: 
    Instead of just dropping volume, high frequencies are heavily lowpass-filtered (at 1200Hz) 
    representing physical tape spacer lift-off, and blended back as the tape settles.
    """
    if dropout_rate == 0 or dropout_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    duration = n_samples / sample_rate
    
    n_dropouts = max(0, int(np.random.poisson(dropout_rate * duration)))
    if n_dropouts == 0:
        return audio_data
        
    # Generate smooth dropout envelope D(t)
    D = np.zeros(n_samples, dtype=np.float32)
    
    for _ in range(n_dropouts):
        idx = np.random.randint(0, n_samples)
        dropout_len = int(np.random.uniform(0.02, 0.08) * sample_rate) # 20ms to 80ms
        window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(dropout_len) / dropout_len))
        
        end_idx = min(idx + dropout_len, n_samples)
        actual_len = end_idx - idx
        D[idx:end_idx] = np.maximum(D[idx:end_idx], window[:actual_len])
        
    # Lowpass filter the audio to make a "muffled" layer
    b, a = signal.butter(1, 1200.0 / (sample_rate / 2.0), btype='low')
    muffled = signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)
    
    # Blend HF loss and apply overall volume attenuation
    vol_gain = 1.0 - D * dropout_depth
    
    if audio_data.ndim == 1:
        y = (1.0 - D) * audio_data + D * muffled
        y = y * vol_gain
    else:
        y = (1.0 - D[:, np.newaxis]) * audio_data + D[:, np.newaxis] * muffled
        y = y * vol_gain[:, np.newaxis]
        
    return y.astype(np.float32)

def generate_motor_noise(n_samples, sample_rate, level=0.005):
    """Generates low-frequency mechanical rumble and AC electrical ground loop hum."""
    t = np.arange(n_samples, dtype=np.float64)
    # AC Hum (60Hz + harmonics)
    hum = (np.sin(2 * np.pi * 60.0 * t / sample_rate) +
           0.25 * np.sin(2 * np.pi * 120.0 * t / sample_rate) +
           0.08 * np.sin(2 * np.pi * 180.0 * t / sample_rate))
    
    # Low-frequency mechanical spindle rumble
    white = np.random.normal(0, 1.0, n_samples)
    b, a = signal.butter(1, 80.0 / (sample_rate / 2.0), btype='low')
    rumble = signal.filtfilt(b, a, white)
    # Cyclic modulation at the spindle rotation rate
    rumble_mod = 1.0 + 0.2 * np.sin(2 * np.pi * 4.7 * t / sample_rate)
    rumble = rumble * rumble_mod * 0.5
    
    motor_noise = hum + rumble
    peak = np.max(np.abs(motor_noise))
    if peak > 0:
        motor_noise = (motor_noise / peak) * level
    return motor_noise.astype(np.float32)

def add_tape_hiss(audio_data, sample_rate, hiss_level, speed_envelope=None, hiss_file_path=None):
    """Blends a custom hiss WAV file, looping it to match the song duration,
    and applies organic 4.7Hz spindle-modulation and speed gating.
    """
    if hiss_level == 0:
        return audio_data
        
    n_samples = len(audio_data)
    t = np.arange(n_samples, dtype=np.float64)
    
    # Resolve the path to the hiss file (check direct path, then relative to script directory)
    actual_hiss_path = None
    if hiss_file_path is not None:
        paths_to_try = [
            hiss_file_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), hiss_file_path)
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                actual_hiss_path = p
                break
                
    # Fallback to generated noise if no file is provided or file doesn't exist
    if actual_hiss_path is None:
        if hiss_file_path is not None and hiss_file_path != "assets/my-real-tape-hiss.wav":
            print(f"Warning: Hiss file '{hiss_file_path}' not found. Falling back to generated noise.")
        custom_hiss = np.random.normal(0, hiss_level, audio_data.shape)
    else:
        # Load the custom hiss file
        hiss_sr, b_data = wavfile.read(actual_hiss_path)
        
        # Convert to float32 normalized
        if b_data.dtype == np.int16:
            b_float = b_data.astype(np.float32) / 32768.0
        elif b_data.dtype == np.int32:
            b_float = b_data.astype(np.float32) / 2147483648.0
        else:
            b_float = b_data.astype(np.float32)
            
        # Match channels (Mono to Stereo conversion if needed)
        if audio_data.ndim == 2 and b_float.ndim == 1:
            b_float = np.column_stack((b_float, b_float))
        elif audio_data.ndim == 1 and b_float.ndim == 2:
            b_float = np.mean(b_float, axis=1)
            
        # Loop the hiss file to cover the entire length of the audio
        repeats = int(np.ceil(n_samples / len(b_float)))
        custom_hiss = np.tile(b_float, (repeats, 1) if b_float.ndim == 2 else repeats)
        custom_hiss = custom_hiss[:n_samples] * hiss_level

    # Apply your organic spindle-modulation (4.7 Hz reel breathing effect)
    spindle_mod = 1.0 + 0.15 * np.sin(2 * np.pi * 4.7 * t / sample_rate)
    
    num_drift_points = max(2, int(n_samples / (sample_rate * 1.5)))
    drift_vals = np.random.uniform(0.85, 1.15, num_drift_points)
    x_drift = np.linspace(0, n_samples - 1, num_drift_points)
    drift = np.interp(t, x_drift, drift_vals)
    
    total_mod = spindle_mod * drift
    
    if speed_envelope is not None:
        total_mod = total_mod * speed_envelope
        
    if audio_data.ndim == 1:
        custom_hiss = custom_hiss * total_mod
    else:
        custom_hiss = custom_hiss * total_mod[:, np.newaxis]
        
    return (audio_data + custom_hiss).astype(np.float32)

def generate_button_click(sample_rate, is_play=True):
    """Generate a realistic, sharp plastic tactile button click.
    
    Play: plastic dual-stage click representing mechanical carriage lock-in.
    Stop: snappier, spring-loaded latch release.
    """
    duration = 0.08 if is_play else 0.06
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    
    burst = np.random.normal(0, 1.0, n)
    
    if is_play:
        envelope = np.exp(-120 * t)
        # Latched mechanical click 30ms later
        head_engage = np.exp(-200 * np.maximum(t - 0.03, 0)) * 0.4
        head_engage[t < 0.03] = 0
        envelope = envelope + head_engage
    else:
        envelope = np.exp(-180 * t)
        
    click = burst * envelope
    
    # Bandpass focus on plastic frequency response (2kHz - 10kHz)
    nyquist = sample_rate / 2.0
    high = min(10000.0, nyquist - 1.0)
    b, a = signal.butter(2, [2000.0, high], btype='bandpass', fs=sample_rate)
    click = signal.filtfilt(b, a, click)
    
    # Soft chassis resonance thump
    thump = np.sin(2 * np.pi * 80 * t) * np.exp(-80 * t) * 0.15
    click = click + thump
    
    peak = np.max(np.abs(click))
    if peak > 0:
        click = click / peak * 0.25
        
    return click.astype(np.float32)

def load_start_sound(target_sample_rate, target_channels, start_sound_path="assets/cassette-tape-start.wav"):
    """Loads, resamples, and formats the cassette-tape-start.wav file."""
    paths_to_try = [
        start_sound_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), start_sound_path)
    ]
    actual_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            actual_path = p
            break
            
    if actual_path is None:
        return None
        
    try:
        rate, data = wavfile.read(actual_path)
        
        # Convert to float32
        if data.dtype == np.int16:
            data_float = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data_float = data.astype(np.float32) / 2147483648.0
        else:
            data_float = data.astype(np.float32)
            
        # Match channel counts
        if target_channels == 1:
            if data_float.ndim > 1:
                data_float = np.mean(data_float, axis=1)
        else:
            if data_float.ndim == 1:
                data_float = np.column_stack((data_float, data_float))
            elif data_float.shape[1] > 2:
                data_float = data_float[:, :2]
                
        # Resample using linear interpolation if rates differ
        if rate != target_sample_rate:
            num_samples = int(len(data_float) * target_sample_rate / rate)
            t_original = np.arange(len(data_float))
            t_target = np.linspace(0, len(data_float) - 1, num_samples)
            
            if data_float.ndim == 1:
                resampled = np.interp(t_target, t_original, data_float)
            else:
                resampled = np.zeros((num_samples, 2), dtype=np.float32)
                for ch in range(2):
                    resampled[:, ch] = np.interp(t_target, t_original, data_float[:, ch])
        else:
            resampled = data_float
            
        return resampled.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading start sound file: {e}. Falling back to synthetic startup sounds.")
        return None

# --- AUDIO ENGINE ---

def process_file(input_path, params, output_path=None, no_physical=False):
    print(f"Reading {input_path}...")
    sample_rate, data = wavfile.read(input_path)
    
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_float = data.astype(np.float32) / 2147483648.0
    else:
        audio_float = data.astype(np.float32)
        
    original_samples = len(audio_float)
    n_channels = 1 if audio_float.ndim == 1 else audio_float.shape[1]
    
    # Load start sound if mechanical effects are enabled
    start_sound_data = None
    if not no_physical:
        start_sound_data = load_start_sound(sample_rate, n_channels)
    
    # Pad the audio with silence for mechanical operations
    if no_physical:
        pad_start_samples = 0
        pad_end_samples = 0
        processed = audio_float.copy()
    else:
        if start_sound_data is not None:
            pad_start_samples = int(4.0 * sample_rate)
        else:
            pad_start_samples = int(0.75 * sample_rate)
        pad_end_samples = int(0.5 * sample_rate)
        
        if audio_float.ndim == 1:
            processed = np.zeros(pad_start_samples + original_samples + pad_end_samples, dtype=np.float32)
            processed[pad_start_samples:pad_start_samples+original_samples] = audio_float
        else:
            processed = np.zeros((pad_start_samples + original_samples + pad_end_samples, audio_float.shape[1]), dtype=np.float32)
            processed[pad_start_samples:pad_start_samples+original_samples, :] = audio_float

    # Compute tape motion envelope to gate hiss & hum dynamically
    if not no_physical:
        speed_envelope = get_tape_speed_envelope(len(processed), sample_rate, pad_start_samples, original_samples)
    else:
        speed_envelope = np.ones(len(processed), dtype=np.float32)

    # 1. Stereo crosstalk & azimuth mismatch
    print("Applying stereo crosstalk...")
    processed = apply_crosstalk(processed, params['crosstalk'])

    # 2. Subtle organic wow & flutter (with chaotic slips)
    print("Applying subtle wow & flutter...")
    processed = apply_wow_flutter(
        processed, sample_rate,
        wow_depth=params['wow_depth'],
        flutter_depth=params['flutter_depth'],
        drift_depth=params['drift_depth']
    )

    # 3. Mechanical speed glides (startup/stop pitch sweep)
    if not no_physical:
        print("Applying mechanical startup and stop glides...")
        processed = apply_tape_speed_profile(processed, sample_rate, pad_start_samples, original_samples)

    # 4. High-frequency roll-off (shelf filter + brickwall high-cut)
    print("Applying high-frequency roll-off and sharp cuts...")
    processed = apply_high_shelf_cut(processed, sample_rate, params['high_shelf_freq'], params['high_shelf_db'])
    processed = apply_high_cut(processed, sample_rate, params['high_cut_hz'])
    
    # 5. Vintage Low End: low-cut + head bump EQ + treble/midrange boost EQ
    print("Applying vintage bass contour and midrange boost...")
    processed = apply_low_cut(processed, sample_rate, params['low_cut_hz'])
    processed = apply_peaking_eq(processed, sample_rate, center_freq=100.0, gain_db=2.5, Q=0.9)
    processed = apply_peaking_eq(processed, sample_rate, center_freq=params['treble_boost_freq'], gain_db=params['treble_boost_gain'], Q=0.8)
    
    # 6. Dynamic soft-knee tape saturation (headroom)
    print("Applying warm dynamic tape saturation...")
    processed = apply_tape_saturation(processed, params['drive'], threshold=params['threshold'])
    
    # 7. Tape dropouts with Wallace Spacer Loss
    print("Applying tape dropouts...")
    processed = apply_dropouts(processed, sample_rate, params['dropout_rate'], params['dropout_depth'])

    # 8. Add gated mechanical hum (hum turns off when tape is stopped)
    if not no_physical and params['motor_level'] > 0:
        print("Adding gated mechanical motor hum...")
        motor_noise = generate_motor_noise(len(processed), sample_rate, level=params['motor_level'])
        if processed.ndim == 1:
            processed += motor_noise * speed_envelope
        else:
            processed += motor_noise[:, np.newaxis] * speed_envelope[:, np.newaxis]

    # 9. Add gated tape hiss (spindle-modulated)
    print("Adding gated tape hiss...")
    processed = add_tape_hiss(processed, sample_rate, params['hiss_level'], 
                              speed_envelope=speed_envelope, 
                              hiss_file_path=params.get('hiss_file'))

    # 10. Add mechanical click button sounds or custom start WAV
    if not no_physical:
        if start_sound_data is not None:
            print("Overlaying physical tape start sound...")
            mix_len = min(len(start_sound_data), len(processed))
            if processed.ndim == 1:
                processed[:mix_len] += start_sound_data[:mix_len]
            else:
                processed[:mix_len, :] += start_sound_data[:mix_len, :]
        else:
            print("Adding mechanical button play clicks (synthetic)...")
            play_click = generate_button_click(sample_rate, is_play=True)
            play_idx = int(0.15 * sample_rate)
            if processed.ndim == 1:
                processed[play_idx:play_idx+len(play_click)] += play_click
            else:
                for ch in range(processed.shape[1]):
                    processed[play_idx:play_idx+len(play_click), ch] += play_click

        # Always inject the stop click at the end of the recording
        stop_click = generate_button_click(sample_rate, is_play=False)
        t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
        t4_sec = t3_sec + 0.15
        stop_idx = int((t4_sec + 0.05) * sample_rate)
        
        # Truncate array tails to save space
        end_idx = min(stop_idx + len(stop_click) + int(0.15 * sample_rate), len(processed))
        processed = processed[:end_idx]
        
        if processed.ndim == 1:
            processed[stop_idx:stop_idx+len(stop_click)] += stop_click
        else:
            for ch in range(processed.shape[1]):
                processed[stop_idx:stop_idx+len(stop_click), ch] += stop_click
    
    processed = np.clip(processed, -1.0, 1.0)
    final_audio = np.int16(processed * 32767.0)
    
    if output_path is None:
        file_dir, file_name = os.path.split(input_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(file_dir, f"{name}_cassette{ext}")
    
    print(f"Saving to {output_path}...")
    wavfile.write(output_path, sample_rate, final_audio)
    print("Done!")

# --- CLI WRAPPER ---

def main():
    parser = argparse.ArgumentParser(
        description="Apply a realistic cassette tape effect to WAV files with dynamic acoustics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py song.wav                      # Default vintage preset
  python main.py song.wav --preset lofi         # Worn-out tape sound
  python main.py song.wav --preset hifi         # Clean, warm tape
  python main.py song.wav --no-physical         # No button clicks, motor hum, or speed glides
        """
    )
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("--preset", choices=['hifi', 'vintage', 'lofi'], default='vintage',
                        help="Cassette quality preset (default: vintage)")
    parser.add_argument("--drive", type=float, help="Tape saturation amount (1.0 = none, 3.0 = heavy)")
    parser.add_argument("--wow", type=float, help="Wow depth (try 0.0004 for subtle)")
    parser.add_argument("--flutter", type=float, help="Flutter depth (try 0.0001 for subtle)")
    parser.add_argument("--hiss", type=float, help="Tape hiss level (try 0.008 for vintage)")
    parser.add_argument("--dropouts", type=float, help="Dropout rate (events per second)")
    parser.add_argument("--no-physical", action="store_true",
                        help="Skip button clicks, speed glides, and motor hum")
    parser.add_argument("--hiss-file", default="assets/my-real-tape-hiss.wav",
                        help="Path to your custom hiss WAV file (default: assets/my-real-tape-hiss.wav)")
    parser.add_argument("-o", "--output", help="Path to save the processed WAV file (default: input_dir/input_name_cassette.wav)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    params = PRESETS[args.preset].copy()

    params['hiss_file'] = args.hiss_file
    
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
        
    process_file(args.input_file, params, output_path=args.output, no_physical=args.no_physical)

if __name__ == "__main__":
    main()
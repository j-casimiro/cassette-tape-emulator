import argparse
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- DSP MODULES ---

def apply_bandpass(audio_data, sample_rate, low_hz, high_hz):
    """4th-order Butterworth bandpass filter to recreate cassette frequency cuts."""
    nyquist = 0.5 * sample_rate
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist - 1.0)
    b, a = signal.butter(4, [low, high], btype='bandpass', fs=sample_rate)
    return signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)

def apply_peaking_eq(audio_data, sample_rate, center_freq, gain_db, Q=1.0):
    """Standard peaking biquad EQ. Used to emulate the physical low-frequency head bump."""
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    A = 10 ** (gain_db / 40.0)
    
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
    x_drift = np.linspace(0, n_samples - 1, len(drift_vals))
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
    
    t2_sec = pad_start_samples / sample_rate
    t1_sec = max(0.0, t2_sec - 0.25)
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
    t2_sec = pad_start_samples / sample_rate
    t1_sec = max(0.0, t2_sec - 0.25)
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
    t4_sec = t3_sec + 0.15
    
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

def get_hiss_envelope(n_samples, sample_rate, pad_start_samples, original_samples):
    """Generates a continuous hiss envelope that starts immediately and fades out when the tape stops."""
    t = np.arange(n_samples, dtype=np.float64)
    envelope = np.ones(n_samples, dtype=np.float32)
    
    # Quick fade in at the very start (0.0 to 0.2 seconds)
    fade_in_samples = int(0.2 * sample_rate)
    if fade_in_samples > 0:
        envelope[:fade_in_samples] = (t[:fade_in_samples] / fade_in_samples).astype(np.float32)
        
    # Fade out at the end (between t3 and t4)
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
    t4_sec = t3_sec + 0.15
    t3 = int(t3_sec * sample_rate)
    t4 = int(t4_sec * sample_rate)
    
    if t4 > t3:
        mask = (t >= t3) & (t <= t4)
        envelope[mask] = (1.0 - (t[mask] - t3) / (t4 - t3)).astype(np.float32)
        envelope[t > t4] = 0.0
        
    return envelope

def pre_emphasis(audio_data, alpha=0.45):
    b = [1.0, -alpha]
    a = [1.0]
    return signal.lfilter(b, a, audio_data, axis=0)

def de_emphasis(audio_data, alpha=0.45):
    b = [1.0]
    a = [1.0, -alpha]
    return signal.lfilter(b, a, audio_data, axis=0)

def apply_tape_saturation(audio_data, drive, threshold=0.5):
    """Asymmetric waveshaping and hard-clip blending to emulate tape being driven completely 'into the red'."""
    if drive <= 1.0:
        return audio_data
        
    # Boost high frequencies before drive to saturate them first (pre-emphasis)
    emphasized = pre_emphasis(audio_data, alpha=0.5)
    
    x = emphasized * drive
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    
    y = np.zeros_like(x)
    
    # Asymmetry: creates warm even-order harmonics (bias misalignment)
    asymmetry = 0.12 * (drive - 1.0) / drive
    
    # 1. Soft saturation mapping
    linear_mask = abs_x < threshold
    y[linear_mask] = x[linear_mask]
    
    sat_mask = ~linear_mask
    L = 1.0 - threshold
    if L > 0:
        y[sat_mask] = sign_x[sat_mask] * (threshold + L * np.tanh((abs_x[sat_mask] - threshold + asymmetry) / L) - L * np.tanh(asymmetry / L))
        
    # 2. Harder saturation / limiting blend for signal peaks 'in the red' (> 1.05)
    clip_threshold = 1.05
    hot_mask = np.abs(y) > clip_threshold
    y[hot_mask] = np.sign(y[hot_mask]) * (clip_threshold + 0.1 * np.tanh((np.abs(y[hot_mask]) - clip_threshold) / 0.1))
    
    # Apply de-emphasis to restore balance
    result = de_emphasis(y, alpha=0.5)
    
    # Normalize with dynamic makeup gain (driven tape gets slightly louder/punchier)
    orig_peak = np.max(np.abs(audio_data))
    result_peak = np.max(np.abs(result))
    if result_peak > 0 and orig_peak > 0:
        boost = 1.0 + 0.05 * (drive - 1.0)
        target_peak = min(0.98, orig_peak * boost)
        result = result * (target_peak / result_peak)
        
    return result.astype(np.float32)

def apply_dropouts(audio_data, sample_rate, dropout_rate, dropout_depth):
    """Micro-Dropouts / Tangled Ribbon effect:
    Abruptly drops volume by 35-80% on a targeted channel and lowpasses it dynamically at 900 Hz
    (emulating Wallace spacer lift-off), creating sudden stereo imbalance cuts.
    """
    if dropout_rate == 0 or dropout_depth == 0:
        return audio_data
        
    n_samples = len(audio_data)
    duration = n_samples / sample_rate
    
    # Poisson-distributed triggers
    n_dropouts = max(0, int(np.random.poisson(dropout_rate * duration)))
    if n_dropouts == 0:
        return audio_data
        
    # Muffled track lowpassed at 900 Hz
    b, a = signal.butter(1, 900.0 / (sample_rate / 2.0), btype='low')
    muffled = signal.filtfilt(b, a, audio_data, axis=0).astype(np.float32)
    
    processed = audio_data.copy()
    
    for _ in range(n_dropouts):
        idx = np.random.randint(0, n_samples)
        # Abrupt fraction-of-a-second length (15ms to 90ms)
        dropout_len = int(np.random.uniform(0.015, 0.090) * sample_rate)
        # Random volume drop of 35% to 80%
        dip_depth = np.random.uniform(0.35, 0.80) * dropout_depth
        
        window = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(dropout_len) / dropout_len))
        end_idx = min(idx + dropout_len, n_samples)
        actual_len = end_idx - idx
        
        dip = window[:actual_len] * dip_depth
        
        if processed.ndim == 1:
            # Mono dip + muffle
            processed[idx:end_idx] = (1.0 - dip) * processed[idx:end_idx] + dip * muffled[idx:end_idx]
            processed[idx:end_idx] *= (1.0 - dip)
        else:
            # Stereo: Target L or R randomly to create a heavy panning imbalance
            ch = np.random.randint(0, 2)
            other_ch = 1 - ch
            
            # Primary channel: full volume dip + HF spacing loss
            processed[idx:end_idx, ch] = (1.0 - dip) * processed[idx:end_idx, ch] + dip * muffled[idx:end_idx, ch]
            processed[idx:end_idx, ch] *= (1.0 - dip)
            
            # Secondary channel: minor volume bleed drop
            other_dip = dip * 0.25
            processed[idx:end_idx, other_ch] *= (1.0 - other_dip)
            
    return processed.astype(np.float32)

def generate_motor_noise(n_samples, sample_rate, level=0.005):
    """Generates low-frequency mechanical rumble and AC electrical ground loop hum with random fluctuations."""
    t = np.arange(n_samples, dtype=np.float64)
    # AC Hum (60Hz + harmonics)
    hum = (np.sin(2 * np.pi * 60.0 * t / sample_rate) +
           0.25 * np.sin(2 * np.pi * 120.0 * t / sample_rate) +
           0.08 * np.sin(2 * np.pi * 180.0 * t / sample_rate))
    
    # Spindle mechanical rumble
    white = np.random.normal(0, 1.0, n_samples)
    b, a = signal.butter(1, 80.0 / (sample_rate / 2.0), btype='low')
    rumble = signal.filtfilt(b, a, white)
    
    # Low-pass filter random noise at 1.5 Hz to create chaotic mechanical flutter (no sine wave)
    rand_array = np.random.normal(0, 1.0, n_samples)
    b_env, a_env = signal.butter(1, 1.5 / (sample_rate / 2.0), btype='low')
    rumble_mod_rand = signal.filtfilt(b_env, a_env, rand_array)
    rm_min, rm_max = np.min(rumble_mod_rand), np.max(rumble_mod_rand)
    if rm_max - rm_min > 0:
        rumble_mod_rand = 0.8 + 0.4 * (rumble_mod_rand - rm_min) / (rm_max - rm_min)
        
    rumble = rumble * rumble_mod_rand * 0.5
    
    motor_noise = hum + rumble
    peak = np.max(np.abs(motor_noise))
    if peak > 0:
        motor_noise = (motor_noise / peak) * level
    return motor_noise.astype(np.float32)

def load_vcr_hiss(hiss_path, target_sample_rate, target_length, target_channels, hiss_level, input_path):
    """Loads, matches, and loops the custom VCR tape hiss file with search path logic."""
    # List of files to try: specified path, followed by fallback defaults
    files_to_try = [hiss_path]
    for fallback in ["my-real-tape-hiss.wav", "vcr-tape-hiss-sound-effect-vhs-camera-buzz-80-s-90-s-home-video-i-1-ogjq.wav"]:
        if fallback not in files_to_try:
            files_to_try.append(fallback)
            
    paths_to_try = []
    for f in files_to_try:
        paths_to_try.extend([
            f,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f),
            os.path.join(os.path.dirname(os.path.abspath(input_path)), f)
        ])
    
    actual_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            actual_path = p
            break
            
    if actual_path is None:
        print(f"Warning: Hiss file not found in search paths (checked fallbacks). Falling back to generated noise.")
        return None
        
    try:
        hiss_rate, hiss_data = wavfile.read(actual_path)
        
        # Convert to float32
        if hiss_data.dtype == np.int16:
            hiss_float = hiss_data.astype(np.float32) / 32768.0
        elif hiss_data.dtype == np.int32:
            hiss_float = hiss_data.astype(np.float32) / 2147483648.0
        else:
            hiss_float = hiss_data.astype(np.float32)
            
        # NORMALIZE the hiss file peak to 1.0 so that the low recording volume is corrected
        hiss_peak = np.max(np.abs(hiss_float))
        if hiss_peak > 0:
            hiss_float = hiss_float / hiss_peak
            
        # Match channel counts
        if target_channels == 1:
            if hiss_float.ndim > 1:
                hiss_float = np.mean(hiss_float, axis=1)
        else:
            if hiss_float.ndim == 1:
                hiss_float = np.column_stack((hiss_float, hiss_float))
            elif hiss_float.shape[1] > 2:
                hiss_float = hiss_float[:, :2]
                
        # Resample using linear interpolation if rates differ
        if hiss_rate != target_sample_rate:
            num_samples = int(len(hiss_float) * target_sample_rate / hiss_rate)
            t_original = np.arange(len(hiss_float))
            t_target = np.linspace(0, len(hiss_float) - 1, num_samples)
            
            if hiss_float.ndim == 1:
                hiss_resampled = np.interp(t_target, t_original, hiss_float)
            else:
                hiss_resampled = np.zeros((num_samples, 2), dtype=np.float32)
                for ch in range(2):
                    hiss_resampled[:, ch] = np.interp(t_target, t_original, hiss_float[:, ch])
        else:
            hiss_resampled = hiss_float
            
        # Loop / Tile the hiss to match target length
        hiss_len = len(hiss_resampled)
        if hiss_len < target_length:
            repeats = int(np.ceil(target_length / hiss_len))
            if hiss_resampled.ndim == 1:
                hiss_matched = np.tile(hiss_resampled, repeats)[:target_length]
            else:
                hiss_matched = np.tile(hiss_resampled, (repeats, 1))[:target_length]
        else:
            hiss_matched = hiss_resampled[:target_length]
            
        # Apply 2nd-order Butterworth high-pass filter at 500 Hz to lessen the bass on the hiss file
        nyquist = 0.5 * target_sample_rate
        cutoff = min(500.0, nyquist - 1.0)
        b, a = signal.butter(2, cutoff, btype='highpass', fs=target_sample_rate)
        hiss_matched = signal.filtfilt(b, a, hiss_matched, axis=0)

        # Mix the custom VCR hiss at an authentic, clearly audible level (hiss_level * 40.0)
        hiss_matched = hiss_matched * hiss_level * 40.0
        return hiss_matched.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading hiss file: {e}. Falling back to generated noise.")
        return None

def load_start_sound(target_sample_rate, target_channels, input_path):
    """Loads, resamples, and formats the cassette-tape-start.wav file."""
    hiss_path = "cassette-tape-start.wav"
    paths_to_try = [
        hiss_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), hiss_path),
        os.path.join(os.path.dirname(os.path.abspath(input_path)), hiss_path)
    ]
    actual_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            actual_path = p
            break
            
    if actual_path is None:
        print("Warning: cassette-tape-start.wav not found. Falling back to synthetic startup sounds.")
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

def add_tape_hiss(audio_data, sample_rate, hiss_level, speed_envelope=None, hiss_file_path=None, input_path=None):
    """Adds tape hiss (using custom VCR noise if provided, otherwise generated random-walk noise)."""
    if hiss_level == 0:
        return audio_data
        
    n_samples = len(audio_data)
    n_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
    
    # Try custom hiss
    if hiss_file_path is not None and input_path is not None:
        custom_hiss = load_vcr_hiss(hiss_file_path, sample_rate, n_samples, n_channels, hiss_level, input_path)
        if custom_hiss is not None:
            if speed_envelope is not None:
                if audio_data.ndim == 1:
                    custom_hiss = custom_hiss * speed_envelope
                else:
                    custom_hiss = custom_hiss * speed_envelope[:, np.newaxis]
            return (audio_data + custom_hiss).astype(np.float32)
            
    # Generated fallback
    nyquist = sample_rate / 2.0
    noise1 = np.random.normal(0, hiss_level, audio_data.shape)
    noise2 = np.random.normal(0, hiss_level, audio_data.shape)
    
    b_low, a_low = signal.butter(2, [800.0, 2200.0], btype='bandpass', fs=sample_rate)
    b_high, a_high = signal.butter(2, [3000.0, min(8500.0, nyquist - 1.0)], btype='bandpass', fs=sample_rate)
    
    if audio_data.ndim == 1:
        hiss_low = signal.filtfilt(b_low, a_low, noise1)
        hiss_high = signal.filtfilt(b_high, a_high, noise2)
    else:
        hiss_low = np.zeros_like(noise1)
        hiss_high = np.zeros_like(noise2)
        for ch in range(noise1.shape[1]):
            hiss_low[:, ch] = signal.filtfilt(b_low, a_low, noise1[:, ch])
            hiss_high[:, ch] = signal.filtfilt(b_high, a_high, noise2[:, ch])
            
    rand_array1 = np.random.normal(0, 1.0, n_samples)
    rand_array2 = np.random.normal(0, 1.0, n_samples)
    
    b_env1, a_env1 = signal.butter(1, 0.5 / nyquist, btype='low')
    b_env2, a_env2 = signal.butter(1, 0.8 / nyquist, btype='low')
    
    env_low = signal.filtfilt(b_env1, a_env1, rand_array1)
    env_high = signal.filtfilt(b_env2, a_env2, rand_array2)
    
    el_min, el_max = np.min(env_low), np.max(env_low)
    if el_max - el_min > 0:
        env_low = 0.7 + 0.6 * (env_low - el_min) / (el_max - el_min)
        
    eh_min, eh_max = np.min(env_high), np.max(env_high)
    if eh_max - eh_min > 0:
        env_high = 0.6 + 0.8 * (env_high - eh_min) / (eh_max - eh_min)
        
    if audio_data.ndim == 1:
        hiss = hiss_low * env_low + hiss_high * env_high
        if speed_envelope is not None:
            hiss = hiss * speed_envelope
    else:
        hiss = hiss_low * env_low[:, np.newaxis] + hiss_high * env_high[:, np.newaxis]
        if speed_envelope is not None:
            hiss = hiss * speed_envelope[:, np.newaxis]
            
    return (audio_data + hiss).astype(np.float32)

def generate_deck_sounds(sample_rate, is_start=True):
    """Synthesizes mechanical cassette deck noises:
    - Start: Deck door closing (thump + plastic latch) followed by head carriage engage.
    - End: Solenoid disengage click followed by deck door opening click.
    """
    if is_start:
        duration = 0.35
        n = int(duration * sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        
        # 1. Deck door closing (t=0.0s)
        thump1 = np.sin(2 * np.pi * 65 * t) * np.exp(-35 * t) * 0.25
        noise = np.random.normal(0, 1.0, n)
        b, a = signal.butter(2, [1500.0, 6000.0], btype='bandpass', fs=sample_rate)
        click1 = signal.filtfilt(b, a, noise) * np.exp(-120 * t) * 0.15
        
        # 2. Latching click (t=0.08s)
        t_latch = t - 0.08
        latch_mask = t_latch > 0
        click_latch = np.zeros_like(t)
        click_latch[latch_mask] = (
            np.random.normal(0, 1.0, np.sum(latch_mask)) * 
            np.exp(-220 * t_latch[latch_mask]) * 0.1
        )
        click_latch = signal.filtfilt(b, a, click_latch)
        
        # 3. Head carriage engage (t=0.25s)
        t_engage = t - 0.25
        engage_mask = t_engage > 0
        thump2 = np.zeros_like(t)
        thump2[engage_mask] = np.sin(2 * np.pi * 80 * t_engage[engage_mask]) * np.exp(-45 * t_engage[engage_mask]) * 0.3
        
        click2 = np.zeros_like(t)
        click2[engage_mask] = (
            np.random.normal(0, 1.0, np.sum(engage_mask)) * 
            np.exp(-150 * t_engage[engage_mask]) * 0.25
        )
        b_met, a_met = signal.butter(2, [2500.0, 8000.0], btype='bandpass', fs=sample_rate)
        click2 = signal.filtfilt(b_met, a_met, click2)
        
        # Secondary lock click (t=0.29s)
        t_lock = t - 0.29
        lock_mask = t_lock > 0
        click_lock = np.zeros_like(t)
        click_lock[lock_mask] = (
            np.random.normal(0, 1.0, np.sum(lock_mask)) * 
            np.exp(-250 * t_lock[lock_mask]) * 0.12
        )
        click_lock = signal.filtfilt(b_met, a_met, click_lock)
        
        sound = thump1 + click1 + click_latch + thump2 + click2 + click_lock
        
    else:
        duration = 0.25
        n = int(duration * sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        
        # 1. Stop button click / Solenoid disengage (t=0.0s)
        noise = np.random.normal(0, 1.0, n)
        b_release, a_release = signal.butter(2, [1800.0, 7500.0], btype='bandpass', fs=sample_rate)
        click1 = signal.filtfilt(b_release, a_release, noise) * np.exp(-140 * t) * 0.22
        thump1 = np.sin(2 * np.pi * 90 * t) * np.exp(-60 * t) * 0.18
        
        # 2. Deck door opening (t=0.15s)
        t_open = t - 0.15
        open_mask = t_open > 0
        click2 = np.zeros_like(t)
        click2[open_mask] = (
            np.random.normal(0, 1.0, np.sum(open_mask)) * 
            np.exp(-110 * t_open[open_mask]) * 0.08
        )
        b_open, a_open = signal.butter(2, [1200.0, 4500.0], btype='bandpass', fs=sample_rate)
        click2 = signal.filtfilt(b_open, a_open, click2)
        
        sound = click1 + thump1 + click2
        
    peak = np.max(np.abs(sound))
    if peak > 0:
        sound = sound / peak * 0.35
        
    return sound.astype(np.float32)

# --- AUDIO ENGINE ---

def process_file(input_path, params):
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
    
    # Pad the audio with silence for mechanical operations
    # Try loading custom cassette-tape-start.wav
    start_sound_data = load_start_sound(sample_rate, n_channels, input_path)
    if start_sound_data is not None:
        pad_start_samples = int(4.0 * sample_rate)
    else:
        pad_start_samples = int(0.75 * sample_rate)
        
    pad_end_samples = int(0.50 * sample_rate)
    
    if audio_float.ndim == 1:
        processed = np.zeros(pad_start_samples + original_samples + pad_end_samples, dtype=np.float32)
        processed[pad_start_samples:pad_start_samples+original_samples] = audio_float
    else:
        processed = np.zeros((pad_start_samples + original_samples + pad_end_samples, audio_float.shape[1]), dtype=np.float32)
        processed[pad_start_samples:pad_start_samples+original_samples, :] = audio_float

    # Compute tape motion envelope to gate hiss & hum dynamically
    speed_envelope = get_tape_speed_envelope(len(processed), sample_rate, pad_start_samples, original_samples)

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
    print("Applying mechanical startup and stop glides...")
    processed = apply_tape_speed_profile(processed, sample_rate, pad_start_samples, original_samples)

    # 4. Dynamic soft-knee tape saturation (headroom)
    print("Applying warm dynamic tape saturation...")
    processed = apply_tape_saturation(processed, params['drive'], threshold=params['threshold'])

    # 5. Add mechanical motor hum (before playback EQ so it gets bandpassed)
    if params['motor_level'] > 0:
        print("Adding gated mechanical motor hum...")
        motor_noise = generate_motor_noise(len(processed), sample_rate, level=params['motor_level'])
        if processed.ndim == 1:
            processed += motor_noise * speed_envelope
        else:
            processed += motor_noise[:, np.newaxis] * speed_envelope[:, np.newaxis]

    # 6. Tangled Ribbon Stereo-Skewed Dropouts (the cuts)
    print("Applying tape dropouts (the cuts)...")
    processed = apply_dropouts(processed, sample_rate, params['dropout_rate'], params['dropout_depth'])

    # 7. Playback Head EQ (Brickwall bandpass + head bump EQ + treble boost)
    print("Applying brickwall bandpass cuts (tape head limits)...")
    processed = apply_bandpass(processed, sample_rate, params['low_hz'], params['high_hz'])
    
    print("Applying head bump EQ...")
    processed = apply_peaking_eq(processed, sample_rate, center_freq=100.0, gain_db=2.5, Q=0.9)
    
    print("Applying treble boost EQ...")
    processed = apply_peaking_eq(processed, sample_rate, center_freq=params['treble_boost_freq'], gain_db=params['treble_boost_gain'], Q=0.8)

    # Compute hiss envelope (starts immediately, fades out at the end)
    hiss_envelope = get_hiss_envelope(len(processed), sample_rate, pad_start_samples, original_samples)

    # 8. Add gated tape hiss on top of the generated cassette music (unfiltered, full fidelity)
    print("Adding gated tape hiss on top of the music...")
    processed = add_tape_hiss(processed, sample_rate, params['hiss_level'], speed_envelope=hiss_envelope, hiss_file_path=params['hiss_file_path'], input_path=input_path)

    # 9. Add physical mechanical clicks and carriage sounds
    print("Adding mechanical deck sounds...")
    stop_sound = generate_deck_sounds(sample_rate, is_start=False)
    
    if start_sound_data is not None:
        # Overlay the cassette-tape-start.wav at the very beginning
        mix_len = min(len(start_sound_data), len(processed))
        if processed.ndim == 1:
            processed[:mix_len] += start_sound_data[:mix_len]
        else:
            processed[:mix_len, :] += start_sound_data[:mix_len, :]
    else:
        # Fallback to synthetic start sound
        start_sound = generate_deck_sounds(sample_rate, is_start=True)
        start_idx = int(0.15 * sample_rate)
        if processed.ndim == 1:
            processed[start_idx:start_idx+len(start_sound)] += start_sound
        else:
            for ch in range(processed.shape[1]):
                processed[start_idx:start_idx+len(start_sound), ch] += start_sound
            
    # Inject stop sound at t4_sec + 0.05s (release button + deck opening click)
    t3_sec = (pad_start_samples + original_samples) / sample_rate + 0.05
    t4_sec = t3_sec + 0.15
    stop_idx = int((t4_sec + 0.05) * sample_rate)
    
    if processed.ndim == 1:
        processed[stop_idx:stop_idx+len(stop_sound)] += stop_sound
    else:
        for ch in range(processed.shape[1]):
            processed[stop_idx:stop_idx+len(stop_sound), ch] += stop_sound
    
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
        description="Apply a realistic cassette tape effect to WAV files with dynamic acoustics."
    )
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("--preset", choices=['hifi', 'vintage', 'lofi'], default='vintage',
                        help="Cassette quality preset (default: vintage)")
    parser.add_argument("--drive", type=float, help="Tape saturation amount (1.0 = none, 3.0 = heavy)")
    parser.add_argument("--wow", type=float, help="Wow depth (try 0.0004 for subtle)")
    parser.add_argument("--flutter", type=float, help="Flutter depth (try 0.0001 for subtle)")
    parser.add_argument("--hiss", type=float, help="Tape hiss level (try 0.008 for vintage)")
    parser.add_argument("--dropouts", type=float, help="Dropout rate (events per second)")
    parser.add_argument("--hiss-file", type=str, default="my-real-tape-hiss.wav",
                        help="Path to WAV file to use for tape hiss")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    presets = {
        'hifi': {
            'low_hz': 40.0,
            'high_hz': 12500.0,
            'treble_boost_freq': 10000.0,
            'treble_boost_gain': 1.5,
            'drive': 1.4,
            'threshold': 0.6,
            'wow_depth': 0.00015,
            'flutter_depth': 0.00005,
            'drift_depth': 0.0001,
            'hiss_level': 0.002,
            'motor_level': 0.0005,
            'dropout_rate': 0.05,
            'dropout_depth': 0.1,
            'crosstalk': 0.05,
        },
        'vintage': {
            'low_hz': 65.0,
            'high_hz': 8000.0,
            'treble_boost_freq': 6000.0,
            'treble_boost_gain': 3.0,
            'drive': 2.5,
            'threshold': 0.4,
            'wow_depth': 0.0005,
            'flutter_depth': 0.00015,
            'drift_depth': 0.0004,
            'hiss_level': 0.009,
            'motor_level': 0.0018,
            'dropout_rate': 0.25,
            'dropout_depth': 0.35,
            'crosstalk': 0.12,
        },
        'lofi': {
            'low_hz': 140.0,
            'high_hz': 4200.0,
            'treble_boost_freq': 3200.0,
            'treble_boost_gain': 4.0,
            'drive': 4.5,
            'threshold': 0.18,
            'wow_depth': 0.001,
            'flutter_depth': 0.0004,
            'drift_depth': 0.0008,
            'hiss_level': 0.018,
            'motor_level': 0.0035,
            'dropout_rate': 0.6,
            'dropout_depth': 0.6,
            'crosstalk': 0.18,
        }
    }
    
    params = presets[args.preset].copy()
    params['hiss_file_path'] = args.hiss_file
    
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
        
    process_file(args.input_file, params)

if __name__ == "__main__":
    main()
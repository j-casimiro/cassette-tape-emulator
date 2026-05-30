// --- APP STATE & INITIALIZATION ---

// Audio Contexts
let audioCtx = null;
let liveAudioCtx = null;

// Showcase player elements
const audioOriginal = new Audio();
const audioEmulated = new Audio();
let isShowcasePlaying = false;
let currentTrackData = null;
let updateInterval = null;

// Web Audio API Nodes for Showcase Analyzer
let showcaseSourceOrig = null;
let showcaseSourceEmul = null;
let showcaseAnalyser = null;
let showcaseGainOrig = null;
let showcaseGainEmul = null;

// Web Audio API Nodes for Live DSP
let liveSource = null;
let liveAudioElement = null;
let liveAnalyser = null;
let liveMainGain = null;

// DSP Nodes
let dspLowCut = null;
let dspBassBump = null;
let dspSaturation = null;
let dspRollOff = null;
let dspDelay = null;
let dspWowLFO = null;
let dspWowGain = null;
let dspFlutterLFO = null;
let dspFlutterGain = null;
let dspHissSource = null;
let dspHissGain = null;
let dspHumSource = null;
let dspHumGain = null;

// --- SHOWCASE SECTION LOGIC ---

// DOM Elements
const deckPlayBtn = document.getElementById('deck-play');
const deckPauseBtn = document.getElementById('deck-pause');
const deckStopBtn = document.getElementById('deck-stop');
const abToggle = document.getElementById('ab-toggle');
const tapeTitle = document.getElementById('tape-title');
const tapeCounter = document.getElementById('tape-counter');
const reelL = document.getElementById('reel-l');
const reelR = document.getElementById('reel-r');
const vuL = document.querySelector('#vu-l .vu-fill');
const vuR = document.querySelector('#vu-r .vu-fill');
const trackItems = document.querySelectorAll('.track-item');

// Track selection
trackItems.forEach(item => {
    item.addEventListener('click', () => {
        trackItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        selectTrack(item);
    });
});

function selectTrack(trackItem) {
    // Stop current playback
    stopShowcase();
    
    currentTrackData = {
        title: trackItem.dataset.title,
        original: trackItem.dataset.original,
        emulated: trackItem.dataset.emulated
    };
    
    tapeTitle.textContent = currentTrackData.title;
    audioOriginal.src = currentTrackData.original;
    audioEmulated.src = currentTrackData.emulated;
    
    // Enable controls
    deckPlayBtn.disabled = false;
    deckPauseBtn.disabled = true;
    deckStopBtn.disabled = true;
    abToggle.disabled = false;
    abToggle.checked = false;
    updateABLabelStates();
    
    tapeCounter.textContent = "00:00";
}

function initShowcaseAudioContext() {
    if (audioCtx) return;
    
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
    
    showcaseAnalyser = audioCtx.createAnalyser();
    showcaseAnalyser.fftSize = 64;
    showcaseAnalyser.connect(audioCtx.destination);
    
    // Create elements and link
    showcaseSourceOrig = audioCtx.createMediaElementSource(audioOriginal);
    showcaseSourceEmul = audioCtx.createMediaElementSource(audioEmulated);
    
    showcaseGainOrig = audioCtx.createGain();
    showcaseGainEmul = audioCtx.createGain();
    
    showcaseSourceOrig.connect(showcaseGainOrig);
    showcaseSourceEmul.connect(showcaseGainEmul);
    
    showcaseGainOrig.connect(showcaseAnalyser);
    showcaseGainEmul.connect(showcaseAnalyser);
    
    // Initial gains
    showcaseGainOrig.gain.value = 1.0;
    showcaseGainEmul.gain.value = 0.0;
}

// Seamless A/B crossfader
abToggle.addEventListener('change', () => {
    initShowcaseAudioContext();
    updateABLabelStates();
    
    if (abToggle.checked) {
        // Crossfade to Cassette
        crossfade(showcaseGainOrig, showcaseGainEmul);
    } else {
        // Crossfade to Original
        crossfade(showcaseGainEmul, showcaseGainOrig);
    }
});

function updateABLabelStates() {
    const labels = document.querySelectorAll('.ab-label');
    if (abToggle.checked) {
        labels[0].classList.remove('active');
        labels[1].classList.add('active');
    } else {
        labels[0].classList.add('active');
        labels[1].classList.remove('active');
    }
}

function crossfade(gainOut, gainIn) {
    if (!audioCtx) return;
    const now = audioCtx.currentTime;
    gainOut.gain.setValueAtTime(gainOut.gain.value, now);
    gainOut.gain.linearRampToValueAtTime(0.0, now + 0.08); // 80ms crossfade
    
    gainIn.gain.setValueAtTime(gainIn.gain.value, now);
    gainIn.gain.linearRampToValueAtTime(1.0, now + 0.08);
}

// Play / Pause / Stop Handlers
deckPlayBtn.addEventListener('click', () => {
    if (!currentTrackData) return;
    
    // Resume context if suspended (browser security)
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    initShowcaseAudioContext();
    
    // Play both in sync
    audioOriginal.play();
    audioEmulated.play();
    
    // Keep them locked in sync
    audioEmulated.currentTime = audioOriginal.currentTime;
    
    isShowcasePlaying = true;
    deckPlayBtn.disabled = true;
    deckPlayBtn.classList.add('pressed');
    deckPauseBtn.disabled = false;
    deckPauseBtn.classList.remove('pressed');
    deckStopBtn.disabled = false;
    
    // Start reel animations
    reelL.classList.add('spinning');
    reelR.classList.add('spinning');
    
    // Start timers
    startShowcaseTimers();
});

deckPauseBtn.addEventListener('click', () => {
    audioOriginal.pause();
    audioEmulated.pause();
    
    isShowcasePlaying = false;
    deckPlayBtn.disabled = false;
    deckPlayBtn.classList.remove('pressed');
    deckPauseBtn.disabled = true;
    deckPauseBtn.classList.add('pressed');
    
    reelL.classList.remove('spinning');
    reelR.classList.remove('spinning');
    
    stopShowcaseTimers();
});

deckStopBtn.addEventListener('click', stopShowcase);

function stopShowcase() {
    audioOriginal.pause();
    audioOriginal.currentTime = 0;
    
    audioEmulated.pause();
    audioEmulated.currentTime = 0;
    
    isShowcasePlaying = false;
    deckPlayBtn.disabled = false;
    deckPlayBtn.classList.remove('pressed');
    deckPauseBtn.disabled = true;
    deckPauseBtn.classList.remove('pressed');
    deckStopBtn.disabled = true;
    
    reelL.classList.remove('spinning');
    reelR.classList.remove('spinning');
    
    tapeCounter.textContent = "00:00";
    
    stopShowcaseTimers();
    resetVUMeters();
}

function startShowcaseTimers() {
    updateInterval = setInterval(() => {
        // Keep in sync
        if (Math.abs(audioOriginal.currentTime - audioEmulated.currentTime) > 0.05) {
            audioEmulated.currentTime = audioOriginal.currentTime;
        }
        
        // Counter
        const minutes = Math.floor(audioOriginal.currentTime / 60).toString().padStart(2, '0');
        const seconds = Math.floor(audioOriginal.currentTime % 60).toString().padStart(2, '0');
        tapeCounter.textContent = `${minutes}:${seconds}`;
        
        // Check if finished
        if (audioOriginal.ended) {
            stopShowcase();
        }
        
        // VU Meter levels
        updateShowcaseVUMeters();
    }, 50);
}

function stopShowcaseTimers() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

function updateShowcaseVUMeters() {
    if (!showcaseAnalyser || !isShowcasePlaying) {
        resetVUMeters();
        return;
    }
    
    const bufferLength = showcaseAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    showcaseAnalyser.getByteFrequencyData(dataArray);
    
    // Compute peak/average level
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
    }
    const average = sum / bufferLength;
    
    // Scale value to match standard display
    const rawVal = Math.min(100, (average / 255) * 160); 
    
    // Add minor variation for Left and Right channels
    const leftVal = Math.min(100, rawVal * (0.9 + Math.random() * 0.2));
    const rightVal = Math.min(100, rawVal * (0.9 + Math.random() * 0.2));
    
    vuL.style.width = `${leftVal}%`;
    vuR.style.width = `${rightVal}%`;
}

function resetVUMeters() {
    vuL.style.width = '0%';
    vuR.style.width = '0%';
}

// Load initial track
selectTrack(trackItems[0]);


// --- LIVE WEB DSP DECK LOGIC ---

// Preset configs
const PRESET_PARAMS = {
    vintage: { drive: 2.0, wow: 0.0004, hiss: 0.006, cutoff: 7500, motor: 0.0015, bass: 2.5 },
    lofi: { drive: 3.2, wow: 0.0009, hiss: 0.015, cutoff: 4500, motor: 0.0035, bass: 4.5 },
    hifi: { drive: 1.3, wow: 0.0001, hiss: 0.002, cutoff: 13500, motor: 0.0005, bass: 1.0 }
};

// UI sliders
const sliderDrive = document.getElementById('param-drive');
const sliderWow = document.getElementById('param-wow');
const sliderHiss = document.getElementById('param-hiss');
const sliderRollOff = document.getElementById('param-roll-off');
const sliderMotor = document.getElementById('param-motor');
const sliderBass = document.getElementById('param-bass');

const valDrive = document.getElementById('val-drive');
const valWow = document.getElementById('val-wow');
const valHiss = document.getElementById('val-hiss');
const valRollOff = document.getElementById('val-roll-off');
const valMotor = document.getElementById('val-motor');
const valBass = document.getElementById('val-bass');

// File Upload UI
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const selectedFileName = document.getElementById('selected-file-name');
const removeFileBtn = document.getElementById('remove-file-btn');
const dspPlayBtn = document.getElementById('dsp-play');
const dspStopBtn = document.getElementById('dsp-stop');
const presetBtns = document.querySelectorAll('.btn-preset');

// Canvas visualizer elements
const canvas = document.getElementById('visualizer-canvas');
const canvasCtx = canvas.getContext('2d');
let animationFrameId = null;

// Setup Drag & Drop Handlers
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetLiveFile();
});

function handleFileSelect(file) {
    if (liveAudioElement) {
        liveAudioElement.pause();
    }
    
    // Update UI
    selectedFileName.textContent = file.name;
    document.querySelector('.drop-content').style.display = 'none';
    fileInfo.style.display = 'flex';
    
    // Create new audio element
    if (liveAudioElement) {
        liveAudioElement.src = URL.createObjectURL(file);
    } else {
        liveAudioElement = new Audio(URL.createObjectURL(file));
    }
    
    dspPlayBtn.disabled = false;
    dspStopBtn.disabled = true;
}

function resetLiveFile() {
    if (liveAudioElement) {
        liveAudioElement.pause();
        liveAudioElement = null;
    }
    
    fileInput.value = '';
    document.querySelector('.drop-content').style.display = 'flex';
    fileInfo.style.display = 'none';
    
    dspPlayBtn.disabled = true;
    dspStopBtn.disabled = true;
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}

// Preset button handlers
presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const preset = btn.dataset.preset;
        if (preset !== 'custom') {
            applyPresetValues(PRESET_PARAMS[preset]);
        }
    });
});

function applyPresetValues(params) {
    sliderDrive.value = params.drive;
    sliderWow.value = params.wow;
    sliderHiss.value = params.hiss;
    sliderRollOff.value = params.cutoff;
    sliderMotor.value = params.motor;
    sliderBass.value = params.bass;
    
    updateDisplays();
    updateLiveAudioParameters();
}

function updateDisplays() {
    valDrive.textContent = `${parseFloat(sliderDrive.value).toFixed(1)}x`;
    valWow.textContent = parseFloat(sliderWow.value).toFixed(4);
    valHiss.textContent = parseFloat(sliderHiss.value).toFixed(3);
    valRollOff.textContent = `${sliderRollOff.value} Hz`;
    valMotor.textContent = parseFloat(sliderMotor.value).toFixed(4);
    valBass.textContent = `+${parseFloat(sliderBass.value).toFixed(1)} dB`;
}

// Slider listeners to switch to custom preset
const sliders = [sliderDrive, sliderWow, sliderHiss, sliderRollOff, sliderMotor, sliderBass];
sliders.forEach(slider => {
    slider.addEventListener('input', () => {
        presetBtns.forEach(b => b.classList.remove('active'));
        document.querySelector('[data-preset="custom"]').classList.add('active');
        
        updateDisplays();
        updateLiveAudioParameters();
    });
});

// Update displays on launch
updateDisplays();


// --- LIVE AUDIO NODE NETWORK (DSP PIPELINE) ---

function initLiveAudioCtx() {
    if (liveAudioCtx) return;
    
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    liveAudioCtx = new AudioContextClass();
    
    // Source
    liveSource = liveAudioCtx.createMediaElementSource(liveAudioElement);
    
    // DSP nodes
    
    // 1. Low Cut highpass
    dspLowCut = liveAudioCtx.createBiquadFilter();
    dspLowCut.type = 'highpass';
    dspLowCut.frequency.value = 60.0;
    
    // 2. Bass Bump peaking
    dspBassBump = liveAudioCtx.createBiquadFilter();
    dspBassBump.type = 'peaking';
    dspBassBump.frequency.value = 100.0;
    dspBassBump.Q.value = 0.9;
    dspBassBump.gain.value = parseFloat(sliderBass.value);
    
    // 3. Saturation WaveShaper
    dspSaturation = liveAudioCtx.createWaveShaper();
    dspSaturation.curve = makeDistortionCurve(parseFloat(sliderDrive.value));
    dspSaturation.oversample = '4x';
    
    // 4. High-frequency roll-off (High Shelf)
    dspRollOff = liveAudioCtx.createBiquadFilter();
    dspRollOff.type = 'highshelf';
    dspRollOff.frequency.value = parseFloat(sliderRollOff.value);
    dspRollOff.gain.value = -8.0; // Steady gentle high shelf cut
    
    // 5. Wow & Flutter Pitch instability
    // DelayNode connected to LFO speed modulators
    dspDelay = liveAudioCtx.createDelay(1.0);
    dspDelay.delayTime.value = 0.005; // 5ms baseline delay
    
    // Wow LFO (Slow pitch variation)
    dspWowLFO = liveAudioCtx.createOscillator();
    dspWowLFO.type = 'sine';
    dspWowLFO.frequency.value = 1.3; // 1.3 Hz belt wobble
    
    dspWowGain = liveAudioCtx.createGain();
    dspWowGain.gain.value = parseFloat(sliderWow.value); // Maps to delay time fluctuation
    
    dspWowLFO.connect(dspWowGain);
    dspWowGain.connect(dspDelay.delayTime);
    dspWowLFO.start();
    
    // Flutter LFO (Fast mechanical vibrations)
    dspFlutterLFO = liveAudioCtx.createOscillator();
    dspFlutterLFO.type = 'sine';
    dspFlutterLFO.frequency.value = 14.5; // 14.5 Hz speed flutter
    
    dspFlutterGain = liveAudioCtx.createGain();
    // Flutter depth is generally 1/3 of wow depth
    dspFlutterGain.gain.value = parseFloat(sliderWow.value) * 0.3;
    
    dspFlutterLFO.connect(dspFlutterGain);
    dspFlutterGain.connect(dspDelay.delayTime);
    dspFlutterLFO.start();
    
    // 6. Gated Tape Hiss
    // Pre-build a 2-second loop of white noise
    const bufferSize = liveAudioCtx.sampleRate * 2;
    const noiseBuffer = liveAudioCtx.createBuffer(1, bufferSize, liveAudioCtx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
    }
    
    dspHissSource = liveAudioCtx.createBufferSource();
    dspHissSource.buffer = noiseBuffer;
    dspHissSource.loop = true;
    
    dspHissGain = liveAudioCtx.createGain();
    dspHissGain.gain.value = parseFloat(sliderHiss.value);
    
    dspHissSource.connect(dspHissGain);
    dspHissSource.start();
    
    // 7. Gated Motor Hum
    dspHumSource = liveAudioCtx.createOscillator();
    dspHumSource.type = 'sine';
    dspHumSource.frequency.value = 60.0; // 60Hz mains motor hum
    
    dspHumGain = liveAudioCtx.createGain();
    dspHumGain.gain.value = parseFloat(sliderMotor.value);
    
    dspHumSource.connect(dspHumGain);
    dspHumSource.start();
    
    // Output nodes
    liveMainGain = liveAudioCtx.createGain();
    liveAnalyser = liveAudioCtx.createAnalyser();
    liveAnalyser.fftSize = 256;
    
    // Connect pipeline
    liveSource.connect(dspLowCut);
    dspLowCut.connect(dspBassBump);
    dspBassBump.connect(dspSaturation);
    dspSaturation.connect(dspRollOff);
    dspRollOff.connect(dspDelay);
    dspDelay.connect(liveMainGain);
    
    // Blend Hiss and Hum at output
    dspHissGain.connect(liveMainGain);
    dspHumGain.connect(liveMainGain);
    
    liveMainGain.connect(liveAnalyser);
    liveAnalyser.connect(liveAudioCtx.destination);
}

function updateLiveAudioParameters() {
    if (!liveAudioCtx) return;
    
    const now = liveAudioCtx.currentTime;
    
    // Drive/Saturation
    dspSaturation.curve = makeDistortionCurve(parseFloat(sliderDrive.value));
    
    // Bass bump
    dspBassBump.gain.setValueAtTime(parseFloat(sliderBass.value), now);
    
    // Cutoff roll-off
    dspRollOff.frequency.setValueAtTime(parseFloat(sliderRollOff.value), now);
    
    // Wow and Flutter gains
    const wowVal = parseFloat(sliderWow.value);
    dspWowGain.gain.setValueAtTime(wowVal, now);
    dspFlutterGain.gain.setValueAtTime(wowVal * 0.3, now);
    
    // Hiss
    dspHissGain.gain.setValueAtTime(parseFloat(sliderHiss.value), now);
    
    // Motor hum
    dspHumGain.gain.setValueAtTime(parseFloat(sliderMotor.value), now);
}

// Generate Waveshaper distortion curves
function makeDistortionCurve(drive) {
    const n_samples = 44100;
    const curve = new Float32Array(n_samples);
    for (let i = 0; i < n_samples; ++i) {
        const x = (i * 2) / n_samples - 1;
        // Soft clipping formula: x * drive / (1 + |x * (drive - 1)|)
        const value = x * drive;
        curve[i] = value / (1 + Math.abs(x * (drive - 0.8)));
    }
    return curve;
}

// Play / Stop control handlers for DSP
dspPlayBtn.addEventListener('click', () => {
    if (!liveAudioElement) return;
    
    if (liveAudioCtx && liveAudioCtx.state === 'suspended') {
        liveAudioCtx.resume();
    }
    
    initLiveAudioCtx();
    
    liveAudioElement.play();
    
    dspPlayBtn.disabled = true;
    dspStopBtn.disabled = false;
    
    // Fade in hum and hiss
    const now = liveAudioCtx.currentTime;
    dspHissGain.gain.setValueAtTime(0, now);
    dspHissGain.gain.linearRampToValueAtTime(parseFloat(sliderHiss.value), now + 0.1);
    dspHumGain.gain.setValueAtTime(0, now);
    dspHumGain.gain.linearRampToValueAtTime(parseFloat(sliderMotor.value), now + 0.1);
    
    drawVisualizer();
});

dspStopBtn.addEventListener('click', stopLivePlayback);

function stopLivePlayback() {
    if (liveAudioElement) {
        liveAudioElement.pause();
        liveAudioElement.currentTime = 0;
    }
    
    dspPlayBtn.disabled = false;
    dspStopBtn.disabled = true;
    
    if (liveAudioCtx) {
        // Fade out hum and hiss
        const now = liveAudioCtx.currentTime;
        dspHissGain.gain.setValueAtTime(dspHissGain.gain.value, now);
        dspHissGain.gain.linearRampToValueAtTime(0.0, now + 0.15);
        dspHumGain.gain.setValueAtTime(dspHumGain.gain.value, now);
        dspHumGain.gain.linearRampToValueAtTime(0.0, now + 0.15);
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
}

// Live end monitoring
setInterval(() => {
    if (liveAudioElement && liveAudioElement.ended) {
        stopLivePlayback();
    }
}, 500);


// --- VISUALIZER DRAW LOOP ---

function drawVisualizer() {
    if (!liveAnalyser) return;
    
    animationFrameId = requestAnimationFrame(drawVisualizer);
    
    const bufferLength = liveAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    liveAnalyser.getByteTimeDomainData(dataArray);
    
    // Canvas sizing setup dynamically
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    
    canvasCtx.fillStyle = '#07090d';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw visual gridlines
    canvasCtx.strokeStyle = 'rgba(92, 225, 230, 0.08)';
    canvasCtx.lineWidth = 1;
    
    // Horizontal center gridline
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, height / 2);
    canvasCtx.lineTo(width, height / 2);
    canvasCtx.stroke();
    
    // Vertical spacing lines
    const gridCols = 8;
    for (let i = 1; i < gridCols; i++) {
        canvasCtx.beginPath();
        canvasCtx.moveTo((width / gridCols) * i, 0);
        canvasCtx.lineTo((width / gridCols) * i, height);
        canvasCtx.stroke();
    }
    
    // Draw the audio waveform trace
    canvasCtx.lineWidth = 2.5;
    canvasCtx.strokeStyle = '#5ce1e6';
    canvasCtx.shadowBlur = 8;
    canvasCtx.shadowColor = '#5ce1e6';
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // Range 0 to 2
        const y = (v * canvas.height) / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
    
    // Reset shadow values
    canvasCtx.shadowBlur = 0;
    canvasCtx.shadowColor = 'transparent';
}

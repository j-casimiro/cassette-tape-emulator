// --- LIVE WEB DSP TAPEDECK LOGIC ---

// Preset parameters
const PRESETS = {
    vintage: { drive: 2.0, wow: 0.0004, hiss: 0.006, cutoff: 7500 },
    lofi: { drive: 3.2, wow: 0.0009, hiss: 0.015, cutoff: 4500 },
    hifi: { drive: 1.3, wow: 0.0001, hiss: 0.002, cutoff: 13500 }
};

// UI Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadLabel = document.getElementById('upload-label');
const dspPlayBtn = document.getElementById('dsp-play');
const dspStopBtn = document.getElementById('dsp-stop');
const presetBtns = document.querySelectorAll('.btn-preset');

const sliderDrive = document.getElementById('param-drive');
const sliderWow = document.getElementById('param-wow');
const sliderHiss = document.getElementById('param-hiss');
const sliderRollOff = document.getElementById('param-roll-off');

const valDrive = document.getElementById('val-drive');
const valWow = document.getElementById('val-wow');
const valHiss = document.getElementById('val-hiss');
const valRollOff = document.getElementById('val-roll-off');

// Audio Context & Nodes
let audioCtx = null;
let audioSource = null;
let audioElement = null;
let sourceNode = null;

// DSP Nodes
let dspSaturation = null;
let dspRollOff = null;
let dspDelay = null;
let dspWowLFO = null;
let dspWowGain = null;
let dspHissSource = null;
let dspHissGain = null;

// Drag & Drop
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
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (audioElement) {
        audioElement.pause();
    }
    
    uploadLabel.textContent = `Selected: ${file.name}`;
    
    // Create new audio element with raw object URL
    audioElement = new Audio(URL.createObjectURL(file));
    
    dspPlayBtn.disabled = false;
    dspStopBtn.disabled = true;
}

// Preset Buttons
presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const presetName = btn.dataset.preset;
        if (presetName !== 'custom') {
            applyPreset(PRESETS[presetName]);
        }
    });
});

function applyPreset(params) {
    sliderDrive.value = params.drive;
    sliderWow.value = params.wow;
    sliderHiss.value = params.hiss;
    sliderRollOff.value = params.cutoff;
    
    updateDisplays();
    updateDSPNodes();
}

function updateDisplays() {
    valDrive.textContent = `${parseFloat(sliderDrive.value).toFixed(1)}x`;
    valWow.textContent = parseFloat(sliderWow.value).toFixed(4);
    valHiss.textContent = parseFloat(sliderHiss.value).toFixed(3);
    valRollOff.textContent = `${sliderRollOff.value} Hz`;
}

// Slider event listeners
const sliders = [sliderDrive, sliderWow, sliderHiss, sliderRollOff];
sliders.forEach(slider => {
    slider.addEventListener('input', () => {
        // Set preset to Custom
        presetBtns.forEach(b => b.classList.remove('active'));
        document.querySelector('[data-preset="custom"]').classList.add('active');
        
        updateDisplays();
        updateDSPNodes();
    });
});

// Initial displays
updateDisplays();


// --- DSP ENGINE (WEB AUDIO API) ---

function initDSP() {
    if (audioCtx) return;
    
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
    
    // Media Source
    sourceNode = audioCtx.createMediaElementSource(audioElement);
    
    // 1. Saturation WaveShaper
    dspSaturation = audioCtx.createWaveShaper();
    dspSaturation.curve = makeDistortionCurve(parseFloat(sliderDrive.value));
    dspSaturation.oversample = '4x';
    
    // 2. High Shelf Cut (frequency roll-off)
    dspRollOff = audioCtx.createBiquadFilter();
    dspRollOff.type = 'highshelf';
    dspRollOff.frequency.value = parseFloat(sliderRollOff.value);
    dspRollOff.gain.value = -8.0; // Standard cassette loss
    
    // 3. Delay Node (for speed/pitch instability)
    dspDelay = audioCtx.createDelay(1.0);
    dspDelay.delayTime.value = 0.005; // 5ms baseline
    
    // LFO to modulate delayTime (Wow)
    dspWowLFO = audioCtx.createOscillator();
    dspWowLFO.type = 'sine';
    dspWowLFO.frequency.value = 1.3; // 1.3 Hz speed fluctuation
    
    dspWowGain = audioCtx.createGain();
    dspWowGain.gain.value = parseFloat(sliderWow.value);
    
    dspWowLFO.connect(dspWowGain);
    dspWowGain.connect(dspDelay.delayTime);
    dspWowLFO.start();
    
    // 4. White noise generator (Tape Hiss)
    const bufferSize = audioCtx.sampleRate * 2;
    const noiseBuffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
    const channelData = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
        channelData[i] = Math.random() * 2 - 1;
    }
    
    dspHissSource = audioCtx.createBufferSource();
    dspHissSource.buffer = noiseBuffer;
    dspHissSource.loop = true;
    
    dspHissGain = audioCtx.createGain();
    dspHissGain.gain.value = parseFloat(sliderHiss.value);
    
    dspHissSource.connect(dspHissGain);
    dspHissSource.start();
    
    // Main Mix Node
    const mainMix = audioCtx.createGain();
    
    // Connections
    sourceNode.connect(dspSaturation);
    dspSaturation.connect(dspRollOff);
    dspRollOff.connect(dspDelay);
    dspDelay.connect(mainMix);
    
    // Add hiss
    dspHissGain.connect(mainMix);
    
    // Connect to outputs
    mainMix.connect(audioCtx.destination);
}

function updateDSPNodes() {
    if (!audioCtx) return;
    
    const now = audioCtx.currentTime;
    
    // Update Saturation Curve
    dspSaturation.curve = makeDistortionCurve(parseFloat(sliderDrive.value));
    
    // Update HF cut
    dspRollOff.frequency.setValueAtTime(parseFloat(sliderRollOff.value), now);
    
    // Update Wow amplitude
    dspWowGain.gain.setValueAtTime(parseFloat(sliderWow.value), now);
    
    // Update Hiss volume
    dspHissGain.gain.setValueAtTime(parseFloat(sliderHiss.value), now);
}

function makeDistortionCurve(drive) {
    const n_samples = 44100;
    const curve = new Float32Array(n_samples);
    for (let i = 0; i < n_samples; ++i) {
        const x = (i * 2) / n_samples - 1;
        const val = x * drive;
        curve[i] = val / (1 + Math.abs(x * (drive - 0.8)));
    }
    return curve;
}

// Play / Stop triggers
dspPlayBtn.addEventListener('click', () => {
    if (!audioElement) return;
    
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    initDSP();
    
    audioElement.play();
    dspPlayBtn.disabled = true;
    dspStopBtn.disabled = false;
    
    // Fade hiss back in
    const now = audioCtx.currentTime;
    dspHissGain.gain.setValueAtTime(0, now);
    dspHissGain.gain.linearRampToValueAtTime(parseFloat(sliderHiss.value), now + 0.15);
});

dspStopBtn.addEventListener('click', stopDSP);

function stopDSP() {
    if (audioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
    }
    
    dspPlayBtn.disabled = false;
    dspStopBtn.disabled = true;
    
    if (audioCtx) {
        // Fade out hiss
        const now = audioCtx.currentTime;
        dspHissGain.gain.setValueAtTime(dspHissGain.gain.value, now);
        dspHissGain.gain.linearRampToValueAtTime(0, now + 0.1);
    }
}

// Detect when audio finishes naturally
setInterval(() => {
    if (audioElement && audioElement.ended) {
        stopDSP();
    }
}, 500);

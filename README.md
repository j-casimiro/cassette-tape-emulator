# Cassette Tape Emulator

A Python script that emulates the sound of vintage cassette tapes. It simulates physical tape speed anomalies (wow and flutter), tape saturation, frequency limits, ribbon dropouts, and integrates authentic mechanical deck sounds and tape hiss.

## Audio Samples

You can play and compare the original high-fidelity tracks with their boxy cassette emulations below.

### 1. The Girl Is Mine

*   **Original (Before)**:  
    <audio src="the-girl-is-mine.wav" controls></audio>  
    ([the-girl-is-mine.wav](the-girl-is-mine.wav))
*   **Cassette Emulation (After)**:  
    <audio src="the-girl-is-mine_cassette.wav" controls></audio>  
    ([the-girl-is-mine_cassette.wav](the-girl-is-mine_cassette.wav))

### 2. Don't Stop 'Til You Get Enough

*   **Original (Before)**:  
    <audio src="dont-stop-til-you-get-enough.wav" controls></audio>  
    ([dont-stop-til-you-get-enough.wav](dont-stop-til-you-get-enough.wav))
*   **Cassette Emulation (After)**:  
    <audio src="dont-stop-til-you-get-enough_cassette.wav" controls></audio>  
    ([dont-stop-til-you-get-enough_cassette.wav](dont-stop-til-you-get-enough_cassette.wav))

### 3. Rock With You

*   **Original (Before)**:  
    <audio src="rock-with-you.wav" controls></audio>  
    ([rock-with-you.wav](rock-with-you.wav))
*   **Cassette Emulation (After)**:  
    <audio src="rock-with-you_cassette.wav" controls></audio>  
    ([rock-with-you_cassette.wav](rock-with-you_cassette.wav))

## Setup

Ensure you have Python 3 and the required dependencies installed:

```bash
pip install numpy scipy
```

## Usage

Run the script by passing the input WAV file:

```bash
python main.py <input.wav> [options]
```

### Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_file` | string | | Path to the WAV file to process. |
| `--preset` | string | `vintage` | Emulation preset: `hifi`, `vintage`, or `lofi`. |
| `--drive` | float | | Tape saturation drive level (1.0 to 4.0). |
| `--wow` | float | | Wow depth (slow pitch variations). |
| `--flutter`| float | | Flutter depth (fast pitch variations). |
| `--hiss` | float | | Noise floor level. |
| `--dropouts`| float | | Rate of dropouts (events per second). |

## Presets

The script comes with three built-in presets:

*   **hifi**: Wide frequency response (up to 12.5 kHz), minimal pitch wobble, and low noise. Mimics high-end studio tape recorders.
*   **vintage**: Medium frequency response (up to 8.0 kHz), moderate wow & flutter, and typical cassette noise. Mimics standard 80s tape decks.
*   **lofi**: Narrower, boxy frequency response (140 Hz to 4.2 kHz) with a peaking midrange resonance boost at 3.2 kHz. Simulates a stuffy, muffled ("makulob") cheap boombox or Shoebox recorder tape deck.

## Examples

**Default (Best):**
```bash
python main.py <your-file>.wav --preset lofi --hiss 0.012
```

**Nostalgic Vintage Emulation:**
```bash
python main.py <your-file>.wav --preset vintage --hiss 0.006 --drive 2.2 --wow 0.0004 --flutter 0.00015 --dropouts 0.15
```

**Aggressive Lofi (Heavy Saturation & Warble):**
```bash
python main.py <your-file>.wav --preset lofi --hiss 0.010 --drive 3.5 --wow 0.0008 --flutter 0.0003 --dropouts 0.40
```

**Clean Studio Sound:**
```bash
python main.py <your-file>.wav --preset hifi --hiss 0.002 --drive 1.3 --wow 0.0001 --flutter 0.00005 --dropouts 0.02
```

## Disclaimer

The audio samples used in this project are for demonstration purposes only. All music ownership, copyright, and rights belong to their respective artists, songwriters, and record labels. No copyright infringement is intended, and no claim of ownership is made over these materials.


# Cassette Tape Emulator

A Python tool that emulates the sound of vintage cassette tapes by simulating tape saturation, frequency roll-off, speed anomalies (wow and flutter), and noise/hiss.

👉 **[Open the Live Cassette Player Showcase](https://j-casimiro.github.io/cassette-tape-emulator/)**

## Setup

```bash
pip install numpy scipy
```

## Usage

```bash
python main.py <input.wav> [options]
```

### Options

- `--preset`: Emulation preset (`vintage`, `lofi`, `hifi`).
- `--drive`: Saturation level (1.0 to 4.0).
- `--wow`: Depth of wow (slow speed pitch variation).
- `--flutter`: Depth of flutter (fast speed pitch variation).
- `--hiss`: Noise floor level.
- `--dropouts`: Rate of signal dropouts.

### Examples

- **Vintage (Default):** `python main.py input.wav --preset vintage`
- **Lo-Fi:** `python main.py input.wav --preset lofi --hiss 0.012`
- **Hi-Fi:** `python main.py input.wav --preset hifi --hiss 0.002`

## Audio Samples

Listen to pre-rendered comparisons:
- **The Girl Is Mine:** [Original](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/original/the-girl-is-mine.wav) | [Cassette Emulation](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/emulated/the-girl-is-mine_cassette.wav)
- **Don't Stop 'Til You Get Enough:** [Original](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/original/dont-stop-til-you-get-enough.wav) | [Cassette Emulation](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/emulated/dont-stop-til-you-get-enough_cassette.wav)
- **Baby Come Back:** [Original](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/original/baby-come-back.wav) | [Cassette Emulation](https://github.com/j-casimiro/cassette-tape-emulator/raw/main/samples/emulated/baby-come-back_cassette.wav)

## Disclaimer

The audio samples used in this project are for demonstration purposes only. All music ownership, copyright, and rights belong to their respective artists, songwriters, and record labels. No copyright infringement is intended, and no claim of ownership is made over these materials.

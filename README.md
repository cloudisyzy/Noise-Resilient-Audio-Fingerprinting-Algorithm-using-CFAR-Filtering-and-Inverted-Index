# Noise-Resilient Audio Fingerprinting (CFAR + Inverted Index)

This project implements an audio fingerprinting pipeline that is robust to noise, based on:
- STFT-based anchor point selection with **Advanced CFAR filtering** (CA, OS, SO, TM)
- Pairwise hashing within target zones
- An inverted index for fast matching

## What's new (refactor)
- **Advanced CFAR Modes**:
  - **CA (Cell Averaging)**: Standard mode, best for Gaussian white noise (AWGN).
  - **OS (Ordered Statistic)**: Uses 75th percentile, robust to impulsive noise (Nature recordings).
  - **SO (Smallest Of)**: Uses min(left_window, right_window), best for dense music (avoids masking weak signals near strong beats).
  - **TM (Trimmed Mean)**: Removes top 20%/bottom 10% outliers, offering a robust middle-ground.
- **Refactored Structure**:
  - `audiofp/`: Core package (fingerprint, index).
  - `main.py`: Unified entry point with easy Scenario/CFAR configuration.
- **Multiprocessing**: Enabled by default for faster index building.

## Project structure

```
.
├── audiofp
│   ├── __init__.py
│   ├── fingerprint.py
│   └── index.py
├── main.py
├── Data/
│   └── GTZAN/                         # your dataset folder (example)
└── Inverted_Index/
    └── GTZAN_STFT_inverted_index_table.pkl   # where you save pickles
```

## Installation

Python 3.9+ recommended.

```
pip install numpy librosa matplotlib seaborn scikit-learn tqdm
```

## Quick start

### 1. Configure Experiment in `main.py`

Open `main.py` and edit the configuration block at the bottom:

```python
if __name__ == "__main__":
    # ==========================================
    #             EXPERIMENT CONFIG
    # ==========================================
    
    # Select Scenario: "AWGN" or "NATURE"
    SCENARIO = "NATURE" 

    # Select CFAR Algorithm: "CA", "OS", "SO", "TM"
    CFAR_MODE = "TM"  
```

### 2. Run the script

```
python main.py
```

It will automatically select the appropriate dataset paths (if configured) and run the matching experiment, printing the classification accuracy.

### 3. Use as a library (recommended)

```python
from audiofp import MusicFingerprint, inverted_index_table, music_to_folder_matching

# Build inverted index
ii = inverted_index_table("Data/GTZAN/", multiprocess=True, CFAR_mode="CA")

# Query with specific CFAR mode
best, counts, deltas = music_to_folder_matching(
    music_path="Data/query_snippet.wav",
    music_name="query.wav",
    folder_path="",
    inverted_index=ii,
    CFAR_mode="TM",  # Use Trimmed Mean for query
)

print("Best match:", best)
```

## Background & Inspiration

This project originates from a **Master's course project in Music Informatics**. 

The core innovation stems from the author's background in **Radar Signal Processing**. In radar systems, detecting a target against a complex background (clutter) is a classic problem, often solved using **CFAR (Constant False Alarm Rate)** algorithms. These algorithms dynamically adjust the detection threshold based on local noise statistics to maintain a stable false alarm rate.

Inspired by this, this project treats:
- **Audio Spectrogram Peaks** as "radar targets".
- **Background Music/Noise** as "clutter".

By cross-applying radar technology to audio information retrieval, we implement and evaluate multiple CFAR variants to robustly extract audio fingerprints under challenging conditions:
- **CA-CFAR**: The classic baseline, effective for uniform noise.
- **OS/SO/TM-CFAR**: Advanced variants designed to handle non-homogeneous acoustic environments, impulsive noise, and dense polyphonic textures.

## Authors

**Ziyue Yang**, **Yuqi Zhang**

## License

MIT

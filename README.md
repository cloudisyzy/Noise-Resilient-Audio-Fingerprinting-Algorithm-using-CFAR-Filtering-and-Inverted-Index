# Noise-Resilient Audio Fingerprinting (CFAR + Inverted Index)

This project implements an audio fingerprinting pipeline that is robust to noise, based on:
- STFT-based anchor point selection with optional CFAR filtering
- Pairwise hashing within target zones
- An inverted index for fast matching

What's new (refactor)
- New Python package `audiofp/` with clear modules:
  - `audiofp/fingerprint.py`: fingerprint extraction (STFT, anchors, hashes)
  - `audiofp/index.py`: inverted index build and matching (now using multiprocessing)
- `utils.py` is removed. Its API is available via:
  - `from audiofp import ...` (recommended), or
  - `from main import ...` (main also re-exports the same API)
- `main.ipynb` is converted to `main.py`, keeping all example usages.
- Index building is multiprocessing by default (instead of multithreading). Pass `multiprocess=False` to disable.

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

If you plan to open figures in headless environments, configure a non-interactive Matplotlib backend or save figures to disk.

## Quick start

### Run examples from main.py

Uncomment desired calls at the bottom of `main.py` and run:

```
python main.py
```

Examples included:
1. Build inverted index (default: multiprocessing)
2. Save/Load inverted index (pickled dict)
3. Query a single file against a folder (builds index if not provided)
4. Match a query folder against a reference folder
5. Timing demo

### Use as a library (recommended)

```python
from audiofp import MusicFingerprint, inverted_index_table, music_to_folder_matching

# Build inverted index for a folder of reference tracks
ii = inverted_index_table("Data/GTZAN/", multiprocess=True)

# Query a single file against the index (no re-build)
best, counts, deltas = music_to_folder_matching(
    music_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
    music_name="blues.00005-snippet-10-20.wav",
    folder_path="",             # not used when inverted_index is supplied
    inverted_index=ii,
    CFAR_flag=True,
)

print("Best match:", best)
```

### From main.py (compat import)

If you prefer the previous `utils`-style import, `main.py` re-exports the same names:

```python
from main import invertedIndexTable, MusicFingerPrint

ii = invertedIndexTable("Data/GTZAN/", multithread=True)  # deprecated alias, mapped to multiprocess=True
```

## API overview

- Fingerprinting
  - `MusicFingerprint(file_path, file_name, window_length_ms=100, hop_length_ms=20, zero_padding=4)`
  - `get_hash(...) -> List[Tuple[t1, (f1, f2, dt)]]`
  - Backward-compatible aliases: `MusicFingerPrint`, `getHash`, etc.

- Inverted Index & Matching
  - `inverted_index_table(folder_path, ..., multiprocess=True, num_workers=None)`
    - Deprecated alias: `invertedIndexTable(..., multithread=...)`
  - `music_to_folder_matching(music_path, music_name, folder_path, inverted_index=None, ...)`
  - `folder_to_folder_matching(folder_path, query_folder_path, inverted_index=None, ...)`
  - `query_from_table(query_music, inverted_index, ...)`

All functions support `CFAR_flag=True/False` to enable/disable CFAR filtering during anchor selection.

## Notes on multiprocessing

- Windows/macOS (spawn): Make sure execution is protected by:

```python
if __name__ == "__main__":
    # call functions that build the index here
```

- You can control workers with `num_workers` in `inverted_index_table`. By default it uses `os.cpu_count()`.

## Parameters

- `window_length_ms`, `hop_length_ms`:
  - Default 100 ms window, 20 ms hop; `n_fft = window_length * zero_padding`.
- `delta_T_ms`, `n_bands`:
  - Time/frequency tiling for anchor block maxima (default 200 ms and 20 bands).
- `target_zone_*`:
  - Define time/frequency target zone around anchor1 for pairing and hashing.
- `CFAR_flag`:
  - When True, applies local CFAR thresholding to suppress false anchors.

## Migration guide (from older code)

- `utils.py` has been removed.
  - Replace `from utils import ...` with `from audiofp import ...`.
  - If you must, `from main import ...` also works (same names are re-exported).
- `multithread` parameter is deprecated; use `multiprocess` instead. If provided, it is mapped for backward compatibility.

## Saving/Loading the inverted index

```python
import pickle

# Save
with open("Inverted_Index/GTZAN_STFT_inverted_index_table.pkl", "wb") as f:
    pickle.dump(ii, f)

# Load
with open("Inverted_Index/GTZAN_STFT_inverted_index_table.pkl", "rb") as f:
    ii_loaded = pickle.load(f)
```

## Authors

**Ziyue Yang**, **Yuqi Zhang**

## License

MIT (update if your repository uses a different license).

"""
Main script converted from main.ipynb (with integrated API exports).

This file now also re-exports the public API that used to be provided by utils.py,
so you can import from main.py if you want. Recommended usage is still importing
from the audiofp package directly.

Preserved examples:
1) Build inverted index (with/without multiprocessing)
2) Save/Load inverted index (with confirmation)
3) Query a single music from inverted index
4) Match a folder of queries against a reference folder
5) Build & Match separately with timing

Notes:
- Multiprocessing is used by default for building inverted index.
- On Windows, ensure this script runs under `if __name__ == "__main__":` guard.
"""

from __future__ import annotations

import pickle
import time

# Re-export the public API that used to be in utils.py
from audiofp import (
    MusicFingerprint,
    MusicFingerprintFromData,
    MusicFingerPrint,            # backward-compatible alias
    MusicFingerPrintFromData,    # backward-compatible alias
    inverted_index_table,
    query_from_table,
    music_to_folder_matching,
    folder_to_folder_matching,
    # backward-compatible aliases
    invertedIndexTable,
    queryFromTable,
    musicToFolderMatching,
    folderToFolderMatching,
)


def example_build_inverted_index():
    # Example 1: Build inverted index (multiprocess)
    inverted_index = inverted_index_table(
        "Data/GTZAN/",
        window_length_ms=100,
        hop_length_ms=20,
        zero_padding=4,
        delta_T_ms=200,
        n_bands=20,
        target_zone_time_s=0.5,
        target_zone_time_offset_s=0.1,
        target_zone_freq_factor=0.5,
        progress_bar=True,
        CFAR_flag=True,
        multiprocess=True,
        num_workers=None,  # None -> use default process count
    )
    return inverted_index


def example_build_inverted_index_single_process():
    # Example 1 (alternative): single-process (no multiprocessing)
    inverted_index = inverted_index_table(
        "Data/GTZAN/",
        multiprocess=False,
    )
    return inverted_index


def example_save_inverted_index(inverted_index):
    # Example 2: Save inverted index
    path = "Inverted_Index/GTZAN_STFT_inverted_index_table.pkl"
    confirmation = input(f"Are you sure you want to save the inverted index table to {path}? (y/n): ")
    if confirmation.lower() == "y":
        with open(path, "wb") as f:
            pickle.dump(inverted_index, f)
        print("Saved")
    else:
        print("Canceled")


def example_load_inverted_index():
    # Example 2: Load inverted index
    path = "Inverted_Index/GTZAN_STFT_inverted_index_table.pkl"
    confirmation = input(f"Are you sure you want to load the inverted index table from {path}? (y/n): ")
    if confirmation.lower() == "y":
        with open(path, "rb") as f:
            inverted_index = pickle.load(f)
        print("Loaded")
        return inverted_index
    else:
        print("Canceled")
        return None


def example_query_music(inverted_index=None):
    # Example 3: Query a single music
    # Case 1: Without pre-computed inverted index
    best_match, _, _ = music_to_folder_matching(
        music_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
        music_name="blues.00005-snippet-10-20.wav",
        folder_path="GTZAN/",
        inverted_index=None,
        window_length_ms=100,
        hop_length_ms=20,
        zero_padding=4,
        delta_T_ms=200,
        n_bands=30,
        target_zone_time_s=0.5,
        target_zone_time_offset_s=0.1,
        target_zone_freq_factor=0.5,
        progress_bar=True,
        CFAR_flag=True,
    )
    # Case 2: With pre-computed inverted index (if provided)
    if inverted_index is not None:
        best_match, _, _ = music_to_folder_matching(
            music_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
            music_name="blues.00011-snippet-10-20.wav",
            folder_path="",
            inverted_index=inverted_index,
        )


def example_match_folder(inverted_index=None):
    # Example 4: Match a folder of queries
    # Case 1: Build inverted index inside
    if inverted_index is None:
        predicted_labels, true_labels, inverted_index_built = folder_to_folder_matching(
            folder_path="Data/GTZAN/",
            inverted_index=None,
            query_folder_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
            progress_bar=True,
            CFAR_flag=True,
            accuracy_flag=True,
            report_flag=False,
            confusion_flag=True,
        )

    # Case 2: Use provided inverted index
    if inverted_index is not None:
        predicted_labels, true_labels, _ = folder_to_folder_matching(
            folder_path="",
            inverted_index=inverted_index,
            query_folder_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
            progress_bar=True,
            CFAR_flag=True,
            accuracy_flag=True,
            report_flag=False,
            confusion_flag=True,
        )


def example_timing_demo():
    # Example 5: Build and match separately, with timing
    start_t = time.time()
    inverted_index = inverted_index_table("Data/GTZAN/", multiprocess=True)
    built_ii_t = time.time()
    predicted_labels, true_labels, inverted_index = folder_to_folder_matching(
        folder_path="",
        inverted_index=inverted_index,
        query_folder_path="Data/mir-2013-GeorgeDataset_snippet(10sec)_1062/",
        progress_bar=True,
        CFAR_flag=True,
        accuracy_flag=True,
        report_flag=False,
        confusion_flag=True,
    )
    end_t = time.time()
    print(f"Time of Building Inverted Indexes: {(built_ii_t - start_t):.3f} sec")
    print(f"Time of Matching Musics` Fingerprints: {(end_t - built_ii_t):.3f} sec")
    print(f"Total Time of Execution: {(end_t - start_t):.3f} sec")


if __name__ == "__main__":
    # Run the examples selectively by uncommenting the lines below.

    # 1) Build inverted index (multiprocess)
    # ii = example_build_inverted_index()

    # 1b) Build inverted index (single-process)
    # ii = example_build_inverted_index_single_process()

    # 2) Save/Load inverted index
    # if ii is not None:
    #     example_save_inverted_index(ii)
    # ii_loaded = example_load_inverted_index()

    # 3) Query a music
    # example_query_music(inverted_index=ii_loaded)

    # 4) Match a folder
    # example_match_folder(inverted_index=ii_loaded)

    # 5) Timing demo
    # example_timing_demo()
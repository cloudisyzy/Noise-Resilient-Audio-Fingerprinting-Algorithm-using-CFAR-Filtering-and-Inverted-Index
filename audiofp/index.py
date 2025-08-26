from __future__ import annotations

import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from .fingerprint import MusicFingerprint


Hash = Tuple[float, float, float]
InvertedIndex = Dict[Hash, List[Tuple[str, float]]]


def _process_file_for_index(
    folder_path: str,
    file_name: str,
    window_length_ms: int,
    hop_length_ms: int,
    zero_padding: int,
    delta_T_ms: int,
    n_bands: int,
    target_zone_time_s: float,
    target_zone_time_offset_s: float,
    target_zone_freq_factor: float,
    CFAR_flag: bool,
) -> InvertedIndex:
    """
    Helper for multiprocessing. Must be top-level for pickling.
    """
    music = MusicFingerprint(
        file_path=folder_path,
        file_name=file_name,
        window_length_ms=window_length_ms,
        hop_length_ms=hop_length_ms,
        zero_padding=zero_padding,
    )
    song_id = file_name
    hashes = music.get_hash(
        delta_T_ms=delta_T_ms,
        n_bands=n_bands,
        target_zone_time_s=target_zone_time_s,
        target_zone_time_offset_s=target_zone_time_offset_s,
        target_zone_freq_factor=target_zone_freq_factor,
        plot_flag=False,
        CFAR_flag=CFAR_flag,
    )
    partial: InvertedIndex = {}
    for t_offset, h in hashes:
        if h not in partial:
            partial[h] = []
        partial[h].append((song_id, float(t_offset)))
    return partial


def inverted_index_table(
    folder_path: str,
    *,
    window_length_ms: int = 100,
    hop_length_ms: int = 20,
    zero_padding: int = 4,
    delta_T_ms: int = 200,
    n_bands: int = 20,
    target_zone_time_s: float = 0.5,
    target_zone_time_offset_s: float = 0.1,
    target_zone_freq_factor: float = 0.5,
    progress_bar: bool = True,
    CFAR_flag: bool = True,
    multiprocess: bool = True,
    num_workers: Optional[int] = None,
    # Backward-compatibility with old API:
    multithread: Optional[bool] = None,
) -> InvertedIndex:
    """
    Build inverted index for all audio files in folder_path.
    - Replaces multithreading with multiprocessing by default.
    - If 'multithread' is provided, it's treated as deprecated and mapped to 'multiprocess'.
      multithread=True  -> multiprocess=True
      multithread=False -> multiprocess=False
    """
    if multithread is not None:
        warnings.warn(
            "Parameter 'multithread' is deprecated. Use 'multiprocess' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        multiprocess = bool(multithread)

    all_files: List[str] = []
    for _, _, files in os.walk(folder_path):
        for file_name in files:
            all_files.append(file_name)
        break  # single-level folder

    inverted_index: InvertedIndex = {}

    if multiprocess:
        # ProcessPoolExecutor for CPU-bound workloads (librosa/STFT/hash).
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_file_for_index,
                    folder_path,
                    file_name,
                    window_length_ms,
                    hop_length_ms,
                    zero_padding,
                    delta_T_ms,
                    n_bands,
                    target_zone_time_s,
                    target_zone_time_offset_s,
                    target_zone_freq_factor,
                    CFAR_flag,
                ): file_name
                for file_name in all_files
            }

            iterator = as_completed(futures)
            if progress_bar:
                iterator = tqdm(iterator, total=len(futures), desc="Building Inverted Index (multiprocess)")

            for future in iterator:
                partial = future.result()
                for h, entries in partial.items():
                    if h not in inverted_index:
                        inverted_index[h] = []
                    inverted_index[h].extend(entries)
    else:
        iterator = all_files
        if progress_bar:
            iterator = tqdm(all_files, desc="Building Inverted Index (single process)")
        for file_name in iterator:
            partial = _process_file_for_index(
                folder_path,
                file_name,
                window_length_ms,
                hop_length_ms,
                zero_padding,
                delta_T_ms,
                n_bands,
                target_zone_time_s,
                target_zone_time_offset_s,
                target_zone_freq_factor,
                CFAR_flag,
            )
            for h, entries in partial.items():
                if h not in inverted_index:
                    inverted_index[h] = []
                inverted_index[h].extend(entries)

    return inverted_index


def query_from_table(
    query_music: MusicFingerprint,
    inverted_index: InvertedIndex,
    *,
    CFAR_flag: bool = True,
) -> Tuple[Dict[str, int], Dict[str, List[float]]]:
    """
    Search a query track using a pre-built inverted index.
    Returns:
        match_counts: song_id -> count at modal delta_t
        time_diffs:   song_id -> list of rounded delta_t's
    """
    query_hashes = query_music.get_hash(plot_flag=False, CFAR_flag=CFAR_flag)
    time_diffs: Dict[str, List[float]] = {}

    for t_query, h in query_hashes:
        if h in inverted_index:
            for song_id, t_song in inverted_index[h]:
                delta_t = round(float(t_song) - float(t_query), 1)
                time_diffs.setdefault(song_id, []).append(delta_t)

    match_counts: Dict[str, int] = {}
    for song_id, deltas in time_diffs.items():
        counts = Counter(deltas)
        _, N = counts.most_common(1)[0]
        match_counts[song_id] = int(N)

    return match_counts, time_diffs


def music_to_folder_matching(
    music_path: str,
    music_name: str,
    folder_path: str,
    inverted_index: Optional[InvertedIndex] = None,
    *,
    window_length_ms: int = 100,
    hop_length_ms: int = 20,
    zero_padding: int = 4,
    delta_T_ms: int = 200,
    n_bands: int = 20,
    target_zone_time_s: float = 0.5,
    target_zone_time_offset_s: float = 0.1,
    target_zone_freq_factor: float = 0.5,
    progress_bar: bool = True,
    CFAR_flag: bool = True,
) -> Tuple[Optional[str], Dict[str, int], Dict[str, List[float]]]:
    """
    Match a single query file against a folder (using or building an inverted index).
    """
    if inverted_index is None:
        inverted_index = inverted_index_table(
            folder_path,
            window_length_ms=window_length_ms,
            hop_length_ms=hop_length_ms,
            zero_padding=zero_padding,
            delta_T_ms=delta_T_ms,
            n_bands=n_bands,
            target_zone_time_s=target_zone_time_s,
            target_zone_time_offset_s=target_zone_time_offset_s,
            target_zone_freq_factor=target_zone_freq_factor,
            progress_bar=progress_bar,
            CFAR_flag=CFAR_flag,
            multiprocess=True,
        )

    query_music = MusicFingerprint(
        file_path=music_path,
        file_name=music_name,
        window_length_ms=window_length_ms,
        hop_length_ms=hop_length_ms,
        zero_padding=zero_padding,
    )

    match_counts, time_diffs = query_from_table(query_music, inverted_index, CFAR_flag=CFAR_flag)

    # aggregate before '-'
    match_counts_cleaned: Dict[str, int] = {}
    for song_id, cnt in match_counts.items():
        cleaned = song_id.split("-")[0]
        match_counts_cleaned[cleaned] = match_counts_cleaned.get(cleaned, 0) + cnt

    if match_counts_cleaned:
        best_match = max(match_counts_cleaned, key=match_counts_cleaned.get)
        print(f"Best match: {best_match} with {match_counts_cleaned[best_match]} matching hashes.")
    else:
        print("No match found.")
        best_match = None

    return best_match, match_counts_cleaned, time_diffs


def folder_to_folder_matching(
    folder_path: str,
    query_folder_path: str,
    inverted_index: Optional[InvertedIndex] = None,
    *,
    window_length_ms: int = 100,
    hop_length_ms: int = 20,
    zero_padding: int = 4,
    delta_T_ms: int = 200,
    n_bands: int = 20,
    target_zone_time_s: float = 0.5,
    target_zone_time_offset_s: float = 0.1,
    target_zone_freq_factor: float = 0.5,
    progress_bar: bool = True,
    CFAR_flag: bool = True,
    accuracy_flag: bool = True,
    report_flag: bool = True,
    confusion_flag: bool = True,
) -> Tuple[List[str], List[str], InvertedIndex]:
    """
    Match a folder of query tracks against a reference folder.
    Returns:
        predicted_labels, true_labels, inverted_index
    """
    if inverted_index is None:
        inverted_index = inverted_index_table(
            folder_path,
            window_length_ms=window_length_ms,
            hop_length_ms=hop_length_ms,
            zero_padding=zero_padding,
            delta_T_ms=delta_T_ms,
            n_bands=n_bands,
            target_zone_time_s=target_zone_time_s,
            target_zone_time_offset_s=target_zone_time_offset_s,
            target_zone_freq_factor=target_zone_freq_factor,
            progress_bar=progress_bar,
            CFAR_flag=CFAR_flag,
            multiprocess=True,
        )

    predicted_labels: List[str] = []
    true_labels: List[str] = []

    query_files = [f for f in os.listdir(query_folder_path) if f.lower().endswith(".wav")]
    if progress_bar:
        query_files = list(tqdm(query_files, desc="Matching Queries"))

    for query_file_name in query_files:
        query_music = MusicFingerprint(
            file_path=query_folder_path,
            file_name=query_file_name,
            window_length_ms=window_length_ms,
            hop_length_ms=hop_length_ms,
            zero_padding=zero_padding,
        )

        query_hashes = query_music.get_hash(plot_flag=False, CFAR_flag=CFAR_flag)
        time_diffs: Dict[str, List[float]] = {}

        for t_query, h in query_hashes:
            if h in inverted_index:
                for song_id, t_song in inverted_index[h]:
                    delta_t = round(float(t_song) - float(t_query), 1)
                    time_diffs.setdefault(song_id, []).append(delta_t)

        match_counts: Dict[str, int] = {}
        for song_id, deltas in time_diffs.items():
            counts = Counter(deltas)
            _, N = counts.most_common(1)[0]
            match_counts[song_id] = int(N)

        if match_counts:
            best_match = max(match_counts, key=match_counts.get)
            predicted_label = os.path.splitext(best_match.split("-")[0])[0]
        else:
            predicted_label = "Unknown"

        # keep only part before '-' for ground truth
        base = query_file_name.split("-")[0]
        # preserve extension if split-left part already contains extension not equal to '.wav'
        root, ext = os.path.splitext(base)
        true_label = root if ext == ".wav" else (root + ext)

        predicted_labels.append(predicted_label)
        true_labels.append(true_label)

    if confusion_flag:
        labels_sorted = sorted(set(true_labels + predicted_labels))
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels_sorted)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    if accuracy_flag:
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Classification Accuracy: {accuracy:.4f}")

    if report_flag:
        report = classification_report(true_labels, predicted_labels, labels=sorted(set(true_labels + predicted_labels)))
        print("\nClassification Report:")
        print(report)

    return predicted_labels, true_labels, inverted_index


# ---------------- Backward-compatible function name aliases ----------------
# Original camelCase API (imported by old utils/main)
invertedIndexTable = inverted_index_table
queryFromTable = query_from_table
musicToFolderMatching = music_to_folder_matching
folderToFolderMatching = folder_to_folder_matching
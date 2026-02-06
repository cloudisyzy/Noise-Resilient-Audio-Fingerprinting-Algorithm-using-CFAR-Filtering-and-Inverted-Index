from __future__ import annotations

import math
import os
from typing import Iterable, List, Sequence, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class MusicFingerprint:
    """
    Compute audio fingerprints using STFT, anchor selection (with optional CFAR), and hashing.
    Compatible with original API via camelCase aliases.
    """

    def __init__(
        self,
        file_path: str = "",
        file_name: str = "",
        window_length_ms: int = 100,
        hop_length_ms: int = 20,
        zero_padding: int = 4,
    ) -> None:
        """
        If both file_path and file_name are provided, load audio from file_path + file_name.
        If only file_path is provided and it's a file, load it directly.
        """
        if file_name:
            full_path = os.path.join(file_path, file_name)
        else:
            full_path = file_path

        if not full_path:
            raise ValueError("MusicFingerprint requires a valid audio path (file_path or file_path+file_name).")

        self.file_name = os.path.basename(full_path)
        self.file_path = full_path

        self.data, self.sample_rate = librosa.load(self.file_path, sr=None)
        # Downsample to 8 kHz if needed (band-limiting to speech-like bandwidth can help robustness/speed)
        if self.sample_rate > 8000:
            self.data = librosa.resample(self.data, orig_sr=self.sample_rate, target_sr=8000)
            self.sample_rate = 8000

        self.audio_length_s = len(self.data) / self.sample_rate
        self.window_length_ms = window_length_ms
        self.hop_length_ms = hop_length_ms
        # Convert ms to samples
        self.window_length = int((window_length_ms / 1000) * self.sample_rate)
        self.hop_length = int((hop_length_ms / 1000) * self.sample_rate)
        # FFT size with zero-padding
        self.n_fft = self.window_length * zero_padding

    # -------- basic operations --------
    def audio_clip(self, start_time_sec: float, len_sec: float) -> "MusicFingerprintFromData":
        """
        Return a new MusicFingerprintFromData instance cropped from start_time_sec with duration len_sec.
        """
        end_time_sec = start_time_sec + len_sec
        start_idx = int(start_time_sec * self.sample_rate)
        end_idx = int(end_time_sec * self.sample_rate)
        data_clipped = self.data[start_idx:end_idx]
        return MusicFingerprintFromData(
            data_clipped,
            self.sample_rate,
            window_length_ms=self.window_length_ms,
            hop_length_ms=self.hop_length_ms,
            zero_padding=self.n_fft // self.window_length,
        )

    def stft(self, plot_flag: bool = True):
        """
        Compute Short-Time Fourier Transform (STFT).
        """
        S = librosa.stft(
            self.data,
            n_fft=self.n_fft,
            win_length=self.window_length,
            hop_length=self.hop_length,
            window="hann",
        )

        if plot_flag:
            S_dB = librosa.amplitude_to_db(np.abs(S))
            plt.figure(figsize=(20, 6))
            librosa.display.specshow(
                S_dB,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="hz",
                cmap="inferno",
            )
            plt.ylim(0, 5000)
            plt.xlim(0, 10.1)
            plt.ylabel("Frequency (kHz)")
            plt.yticks(np.arange(0, 5001, 1000), np.arange(0, 6, 1))
            plt.xlabel("Time (s)")
            plt.title(f"Sonogram of {self.file_path} (log amplitude)")
            plt.show()
        return S

    # -------- anchor selection --------
    def anchors_filter(
        self,
        S_power: np.ndarray,
        anchors: List[Tuple[int, int, float]],
        delta_T: int,
        delta_F: int,
        CFAR_mode: str = "CA",
    ) -> np.ndarray:
        """
        CFAR-based false alarm suppression over anchors.
        modes:
         - 'CA': Cell Averaging (Standard, best for AWGN)
         - 'OS': Ordered Statistic (Robust, best for impulsive noise/nature)
         - 'SO': Smallest Of (Best for multi-target/dense music, avoids masking)
         - 'TM': Trimmed Mean (Robust compromise, rejects outliers then averages)
         - 'OFF': No filtering
        """
        if CFAR_mode == "OFF":
            # Just convert to physical units
            for i, (t, f, peak_value) in enumerate(anchors):
                freq_hz = f * self.sample_rate / self.n_fft
                time_s = t * self.hop_length / self.sample_rate
                anchors[i] = (time_s, freq_hz, float(peak_value))
            return np.array(anchors, dtype=object)

        fa_list: List[int] = []
        S_threshold = np.zeros((S_power.shape[0], S_power.shape[1]))
        P_FA = 1e-4

        for i, (t, f, peak_value) in enumerate(anchors):
            # Define outer window boundaries
            t_1 = max(int(t - delta_T), 0)
            t_2 = min(int(t + delta_T), S_power.shape[1])
            f_1 = max(int(f - delta_F), 0)
            f_2 = min(int(f + delta_F), S_power.shape[0])

            # Define inner (guard) window boundaries
            t0_1 = max(int(t - delta_T * 0.3), 0)
            t0_2 = min(int(t + delta_T * 0.3), S_power.shape[1])
            f0_1 = max(int(f - delta_F * 0.3), 0)
            f0_2 = min(int(f + delta_F * 0.3), S_power.shape[0])

            noise_level = 0.0
            n_train = 1 # Avoid division by zero if something goes wrong

            if CFAR_mode == "CA":
                # CA-CFAR: Mean of all reference cells
                # Sum(Outer) - Sum(Inner)
                sum_outer = np.sum(S_power[f_1:f_2, t_1:t_2])
                sum_inner = np.sum(S_power[f0_1:f0_2, t0_1:t0_2])
                n_outer = (f_2 - f_1) * (t_2 - t_1)
                n_inner = (f0_2 - f0_1) * (t0_2 - t0_1)
                n_train = n_outer - n_inner
                
                if n_train > 0:
                    noise_level = (sum_outer - sum_inner) / n_train
            
            elif CFAR_mode == "SO":
                # SO-CFAR (Smallest Of): Split time window into Left and Right
                # Left Block: [t_1, t0_1] x [f_1, f_2]
                # Right Block: [t0_2, t_2] x [f_1, f_2]
                # Note: This ignores the frequency-guard strip above/below the cell for simplicity,
                # focusing on temporal masking which is dominant in audio.
                
                # Left
                left_block = S_power[f_1:f_2, t_1:t0_1]
                # Right
                right_block = S_power[f_1:f_2, t0_2:t_2]
                
                mean_left = np.mean(left_block) if left_block.size > 0 else float('inf')
                mean_right = np.mean(right_block) if right_block.size > 0 else float('inf')
                
                # If we are at the very start/end, one might be empty/inf. Handle gracefully.
                if left_block.size == 0: noise_level = mean_right
                elif right_block.size == 0: noise_level = mean_left
                else: noise_level = min(mean_left, mean_right)
                
                # Estimate N for alpha calculation (approximate as half of total cells)
                n_train = (left_block.size + right_block.size) / 2

            elif CFAR_mode == "OS":
                # OS-CFAR: Percentile based
                # Extract full window
                full_window = S_power[f_1:f_2, t_1:t_2]
                # Create a mask for the guard region
                mask = np.ones(full_window.shape, dtype=bool)
                
                # Relative guard indices
                gt0_1, gt0_2 = t0_1 - t_1, t0_2 - t_1
                gf0_1, gf0_2 = f0_1 - f_1, f0_2 - f_1
                
                if gt0_2 > gt0_1 and gf0_2 > gf0_1:
                     mask[gf0_1:gf0_2, gt0_1:gt0_2] = False
                
                training_cells = full_window[mask]
                n_train = training_cells.size
                
                if n_train > 0:
                    # 75th percentile / 1.386 approx mean for exponential distribution
                    robust_noise_est = np.percentile(training_cells, 75)
                    noise_level = robust_noise_est / 1.386

            elif CFAR_mode == "TM":
                # TM-CFAR (Trimmed Mean): Sort, trim top/bottom, then average
                full_window = S_power[f_1:f_2, t_1:t_2]
                mask = np.ones(full_window.shape, dtype=bool)
                gt0_1, gt0_2 = t0_1 - t_1, t0_2 - t_1
                gf0_1, gf0_2 = f0_1 - f_1, f0_2 - f_1
                
                if gt0_2 > gt0_1 and gf0_2 > gf0_1:
                     mask[gf0_1:gf0_2, gt0_1:gt0_2] = False
                
                training_cells = full_window[mask]
                N = training_cells.size
                
                if N > 4: # Only trim if we have enough samples
                    # Sort
                    sorted_cells = np.sort(training_cells)
                    # Trim Top 20%, Bottom 10%
                    n1 = int(N * 0.2) # Top
                    n2 = int(N * 0.1) # Bottom
                    
                    # Remaining slice
                    trimmed_cells = sorted_cells[n2 : N - n1]
                    if trimmed_cells.size > 0:
                        noise_level = np.mean(trimmed_cells)
                        n_train = trimmed_cells.size
                    else:
                        noise_level = np.mean(sorted_cells) # Fallback
                        n_train = N
                elif N > 0:
                    noise_level = np.mean(training_cells)
                    n_train = N
                else:
                    n_train = 0

            # Calculate Threshold
            # Note: The 'alpha' formula technically depends on the CFAR type and N.
            # For simplicity in this demo, we reuse the CA alpha formula.
            # In rigorous radar systems, OS and SO have different alpha lookups.
            if n_train > 0:
                alpha = n_train * ((1 / P_FA) ** (1 / n_train) - 1)
                S_threshold[f, t] = alpha * noise_level

                if peak_value < S_threshold[f, t]:
                    fa_list.append(i)

            # convert to physical units
            freq_hz = f * self.sample_rate / self.n_fft
            time_s = t * self.hop_length / self.sample_rate
            anchors[i] = (time_s, freq_hz, float(peak_value))

        if fa_list:
            anchors = [item for i, item in enumerate(anchors) if i not in fa_list]

        anchors_array = np.array(anchors, dtype=object)
        return anchors_array

    def get_anchor(
        self,
        S,
        delta_T_ms: int = 200,
        n_bands: int = 20,
        plot_flag: bool = False,
        plot_duration: float = 10.0,
        CFAR_flag: bool = True,
        CFAR_mode: str = None,
    ) -> np.ndarray:
        """
        Tile time-frequency plane and pick block maxima as anchors, then filter by CFAR (optional).
        """
        # Resolve mode: CFAR_mode takes precedence, fallback to CFAR_flag logic
        if CFAR_mode is None:
            CFAR_mode = "CA" if CFAR_flag else "OFF"

        S_dB = librosa.amplitude_to_db(np.abs(S))
        S_power = np.square(np.abs(S))
        delta_T = int((delta_T_ms / 1000) * self.sample_rate / self.hop_length)
        delta_F = max(1, int(S_dB.shape[0] / n_bands))
        anchors: List[Tuple[int, int, float]] = []

        for f in range(0, S_power.shape[0], delta_F):
            for t in range(0, S_power.shape[1], delta_T):
                t_end = min(t + delta_T, S_power.shape[1])
                f_end = min(f + delta_F, S_power.shape[0])
                block = S_power[f:f_end, t:t_end]

                max_val = float(np.max(block))
                if max_val == 0.0:
                    continue
                max_idx = np.where(block == max_val)
                max_f = f + int(max_idx[0][0])
                max_t = t + int(max_idx[1][0])
                anchors.append((max_t, max_f, max_val))

        anchors_array = self.anchors_filter(S_power, anchors, delta_T, delta_F, CFAR_mode=CFAR_mode)

        if plot_flag:
            plt.figure(figsize=(20, 6))
            librosa.display.specshow(
                S_dB,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="hz",
                cmap="inferno",
            )
            plt.ylim(0, 5000)
            plt.xlim(0, plot_duration)
            plt.ylabel("Frequency (kHz)")
            plt.yticks(np.arange(0, 5001, 1000), np.arange(0, 6, 1))
            plt.xlabel("Time (s)")
            plt.title(f"Sonogram of {self.file_path} with Anchors ({CFAR_mode}-CFAR)")
            anchor_times = anchors_array[:, 0].astype(float)
            anchor_freqs = anchors_array[:, 1].astype(float)
            plt.scatter(anchor_times, anchor_freqs, color="cyan", marker="x", s=50)
            plt.show()

        return anchors_array

    # -------- hashing --------
    def compute_hash(
        self,
        anchors: Sequence[Tuple[float, float, float]],
        target_zone_time_s: float = 0.5,
        target_zone_time_offset_s: float = 0.1,
        target_zone_freq_factor: float = 0.5,
    ) -> List[Tuple[float, Tuple[float, float, float]]]:
        """
        Generate (t1, (f1, f2, delta_t12)) hashes for anchor pairs within a target zone.
        """
        hashes: List[Tuple[float, Tuple[float, float, float]]] = []

        anchors_array = np.array(anchors, dtype=object)
        anchor_times = anchors_array[:, 0].astype(float)
        anchor_freqs = anchors_array[:, 1].astype(float)

        for i, anchor1 in enumerate(anchors):
            t1, f1, _ = anchor1

            # time-frequency target zone
            min_time = float(t1) + float(target_zone_time_offset_s)
            max_time = min_time + float(target_zone_time_s)
            min_freq = float(f1) * (2 ** (-target_zone_freq_factor))
            max_freq = float(f1) * (2 ** (target_zone_freq_factor))

            time_mask = (anchor_times >= min_time) & (anchor_times <= max_time)
            anchors_in_time = anchors_array[time_mask]
            if anchors_in_time.size == 0:
                continue

            freqs_in_time = anchors_in_time[:, 1].astype(float)
            freq_mask = (freqs_in_time >= min_freq) & (freqs_in_time <= max_freq)
            anchors_in_zone = anchors_in_time[freq_mask]

            for anchor2 in anchors_in_zone:
                t2, f2, _ = anchor2
                delta_t12 = float(t2) - float(t1)
                hashes.append((float(t1), (float(f1), float(f2), float(delta_t12))))

        return hashes

    def get_hash(
        self,
        delta_T_ms: int = 200,
        n_bands: int = 20,
        target_zone_time_s: float = 0.5,
        target_zone_time_offset_s: float = 0.1,
        target_zone_freq_factor: float = 0.5,
        plot_flag: bool = False,
        plot_duration: float = 10.0,
        CFAR_flag: bool = True,
        CFAR_mode: str = None,
    ) -> List[Tuple[float, Tuple[float, float, float]]]:
        """
        Convenience wrapper: STFT -> anchors -> hashes.
        """
        S = self.stft(plot_flag=False)
        anchors = self.get_anchor(
            S,
            delta_T_ms=delta_T_ms,
            n_bands=n_bands,
            plot_flag=plot_flag,
            plot_duration=plot_duration,
            CFAR_flag=CFAR_flag,
            CFAR_mode=CFAR_mode,
        )
        hashes = self.compute_hash(
            anchors,
            target_zone_time_s=target_zone_time_s,
            target_zone_time_offset_s=target_zone_time_offset_s,
            target_zone_freq_factor=target_zone_freq_factor,
        )
        return hashes

    # ---------- Backward-compatible camelCase aliases ----------
    def audioClip(self, start_time_sec: float, len_sec: float) -> "MusicFingerprintFromData":
        return self.audio_clip(start_time_sec, len_sec)

    def getAnchor(self, S, delta_T_ms=200, n_bands=20, plot_flag=False, plot_duration=10, CFAR_flag=True, CFAR_mode=None):
        return self.get_anchor(
            S,
            delta_T_ms=delta_T_ms,
            n_bands=n_bands,
            plot_flag=plot_flag,
            plot_duration=plot_duration,
            CFAR_flag=CFAR_flag,
            CFAR_mode=CFAR_mode,
        )

    def computeHash(
        self,
        anchors,
        target_zone_time_s=0.5,
        target_zone_time_offset_s=0.1,
        target_zone_freq_factor=0.5,
    ):
        return self.compute_hash(
            anchors,
            target_zone_time_s=target_zone_time_s,
            target_zone_time_offset_s=target_zone_time_offset_s,
            target_zone_freq_factor=target_zone_freq_factor,
        )

    def getHash(
        self,
        delta_T_ms=200,
        n_bands=20,
        target_zone_time_s=0.5,
        target_zone_time_offset_s=0.1,
        target_zone_freq_factor=0.5,
        plot_flag=False,
        plot_duration=10,
        CFAR_flag=True,
        CFAR_mode=None,
    ):
        return self.get_hash(
            delta_T_ms=delta_T_ms,
            n_bands=n_bands,
            target_zone_time_s=target_zone_time_s,
            target_zone_time_offset_s=target_zone_time_offset_s,
            target_zone_freq_factor=target_zone_freq_factor,
            plot_flag=plot_flag,
            plot_duration=plot_duration,
            CFAR_flag=CFAR_flag,
            CFAR_mode=CFAR_mode,
        )


class MusicFingerprintFromData(MusicFingerprint):
    """
    Subclass for operating on in-memory audio arrays.
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        window_length_ms: int = 100,
        hop_length_ms: int = 20,
        zero_padding: int = 4,
    ) -> None:
        self.data = np.asarray(data)
        self.sample_rate = int(sample_rate)
        self.window_length_ms = int(window_length_ms)
        self.hop_length_ms = int(hop_length_ms)
        self.window_length = int((window_length_ms / 1000) * self.sample_rate)
        self.hop_length = int((hop_length_ms / 1000) * self.sample_rate)
        self.n_fft = self.window_length * int(zero_padding)
        self.file_name = "<from_data>"
        self.file_path = "<from_data>"

    # Backward-compatible alias
    def audioClip(self, start_time_sec: float, len_sec: float) -> "MusicFingerprintFromData":
        return super().audio_clip(start_time_sec, len_sec)


# Backward-compatible class name aliases
MusicFingerPrint = MusicFingerprint
MusicFingerPrintFromData = MusicFingerprintFromData
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import math
from tqdm import tqdm
from collections import Counter
import os, time, pickle
import concurrent.futures

params = {'figure.dpi': 300}
plt.rcParams.update(params)

class MusicFingerPrint:
    
    def __init__(self, file_path='', file_name='', window_length_ms=100, hop_length_ms=20, zero_padding=4):
        '''
        Parameter initialization, utilized throughout the whole class.
        '''
        self.file_name = file_name
        self.file_path = file_path + file_name
        self.data, self.sample_rate = librosa.load(self.file_path, sr=None)
        if self.sample_rate > 8000:
            self.data = librosa.resample(self.data, orig_sr=self.sample_rate, target_sr=8000)
            self.sample_rate=8000
        self.audio_length_s = len(self.data) / self.sample_rate
        self.window_length_ms = window_length_ms
        self.hop_length_ms = hop_length_ms
        self.window_length = int((window_length_ms / 1000) * self.sample_rate)  # Convert ms to samples
        self.hop_length = int((hop_length_ms / 1000) * self.sample_rate)       # Convert ms to samples
        self.n_fft = self.window_length * zero_padding  # Set FFT size with zero-padding

        
    def audioClip(self, start_time_sec, len_sec):
        """
        Clips the audio from start_time_sec to start_time_sec + len_sec and returns a new MusicFingerPrint object that contains the clipped audio.
        """
        end_time_sec = start_time_sec + len_sec
        start_time_idx = int(start_time_sec * self.sample_rate)
        end_time_idx = int(end_time_sec * self.sample_rate)
        
        # Clip the audio data
        data_clipped = self.data[start_time_idx:end_time_idx]
        
        # Create a new piece of MusicFingerPrint (sub-class) for the clipped data
        return MusicFingerPrintFromData(data_clipped, self.sample_rate, self.window_length_ms, self.hop_length_ms, self.n_fft // self.window_length)
    
    
    def stft(self, plot_flag=True):
        '''
        Compute Short-Time Fourier Transform (STFT), aka the sonogram.
        '''
        S = librosa.stft(self.data, n_fft=self.n_fft, win_length=self.window_length, hop_length=self.hop_length, window='hann')
        
        if plot_flag:
            S_dB = librosa.amplitude_to_db(np.abs(S))  # Convert amplitude to dB scale
            plt.figure(figsize=(20, 6))
            librosa.display.specshow(S_dB, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='hz', cmap='inferno')
            plt.ylim(0, 5000)
            plt.xlim(0, 10.1)
            plt.ylabel('Frequency (kHz)')
            plt.yticks(np.arange(0, 5001, 1000), np.arange(0, 6, 1))
            plt.xlabel('Time (s)')
            plt.title(f'Sonogram of {self.file_path} (logarithmic amplitude)')
            plt.show()
        return S
    
    def anchors_filter(self, S_power, anchors, delta_T, delta_F, CFAR_flag):
        '''
        Filtering anchors using Constant False Alarm Rate, inspired by radar signal processing
        '''
        fa_list=[]
        S_threshold = np.zeros((S_power.shape[0], S_power.shape[1]))
        for i, (t,f,peak_value) in enumerate(anchors):
            
            if CFAR_flag==True :
                
                t0_1 = max (int(t-delta_T*0.3), 0)
                t0_2 = min (int(t+delta_T*0.3), S_power.shape[1])
                f0_1 = max (int(f-delta_F*0.3), 0)
                f0_2 = min (int(f+delta_F*0.3), S_power.shape[0])

                t_1 = max (int(t-delta_T), 0)
                t_2 = min (int(t+delta_T), S_power.shape[1])
                f_1 = max (int(f-delta_F), 0)
                f_2 = min (int(f+delta_F), S_power.shape[0])

                n_train = np.size(S_power[f_1:f_2, t_1:t_2]) - np.size(S_power[f0_1:f0_2, t0_1:t0_2])
                noise_level = (np.sum(S_power[f_1:f_2, t_1:t_2])-np.sum(S_power[f0_1:f0_2, t0_1:t0_2]))/n_train
                P_FA=0.0001
                alpha = n_train*((1/P_FA)**(1/n_train)-1)
                S_threshold[f,t]= alpha*noise_level
                
                if peak_value < S_threshold[f,t] :
                    fa_list.append(i)
            
            freq_hz = f * self.sample_rate / self.n_fft  # Convert index to frequency in Hz
            time_s = t * self.hop_length / self.sample_rate # Convert index to time in sec
            anchors[i] = (time_s, freq_hz, peak_value)
                    
        if len(fa_list)!=0:
            anchors = [item for i, item in enumerate(anchors) if i not in fa_list]

        anchors_array = np.array(anchors, dtype=object)

        return anchors_array
    
    def getAnchor(self, S, delta_T_ms=200, n_bands=20, plot_flag=False, plot_duration=10, CFAR_flag=True):
        '''
        Divide time into delta_T segments, and frequency into n_bands, to find the maximum STFT value in each block, returns the coordinate, aka "anchor".
        Used after MusicFingerPrint.stft().
        '''
        S_dB = librosa.amplitude_to_db(np.abs(S))  # Convert amplitude to dB scale
        S_power = np.square(np.abs(S))
        delta_T = int((delta_T_ms / 1000) * self.sample_rate / self.hop_length)
        delta_F = int(S_dB.shape[0] / n_bands)
        anchors = []

        #Loop through frequency and time to find points of interest (anchors)
        for f in range(0, S_power.shape[0], delta_F):
            for t in range(0, S_power.shape[1], delta_T):
                t_end = min(t + delta_T, S_power.shape[1])
                f_end = min(f + delta_F, S_power.shape[0])
                block = S_power[f:f_end, t:t_end]  # Get block of STFT data

                max_val = np.max(block)  # Find maximum value in block
                if max_val==0:
                    continue
                max_idx = np.where(block == max_val)  # Get index of maximum value
                max_f = f + max_idx[0][0]
                max_t = t + max_idx[1][0]
                
                anchors.append((max_t, max_f, max_val))  # Store

        anchors_array = self.anchors_filter(S_power, anchors, delta_T, delta_F, CFAR_flag)
        
        if plot_flag==True:
            plt.figure(figsize=(20, 6))
            librosa.display.specshow(S_dB, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='hz', cmap='inferno')
            plt.ylim(0, 5000)
            plt.xlim(0, plot_duration)
            plt.ylabel('Frequency (kHz)')
            plt.yticks(np.arange(0, 5001, 1000), np.arange(0, 6, 1))
            plt.xlabel('Time (s)')
            plt.title(f'Sonogram of {self.file_path} with Anchors')
            anchor_times = anchors_array[:, 0].astype(float)  # Extract anchor times
            anchor_freqs = anchors_array[:, 1].astype(float)  # Extract anchor frequencies
            plt.scatter(anchor_times, anchor_freqs, color='cyan', marker='x', s=50)  # Plot anchors
            plt.show()
        return anchors_array
    
    
    def computeHash(self, anchors, target_zone_time_s=0.5, target_zone_time_offset_s=0.1, target_zone_freq_factor=0.5):
        """
        Computes the hashes from the given anchors, param 'anchors' can be computed via previous MusicFingerPrint.getAnchors() function.
        """
        hashes = []

        # Convert anchors to numpy array for efficient indexing
        anchors_array = np.array(anchors, dtype=object)
        anchor_times = anchors_array[:, 0].astype(float)
        anchor_freqs = anchors_array[:, 1].astype(float)

        for i, anchor1 in enumerate(anchors):
            t1, f1, _ = anchor1

            # Define the target zone
            min_time = t1 + target_zone_time_offset_s
            max_time = min_time + target_zone_time_s
            min_freq = f1 * (2 ** -target_zone_freq_factor)
            max_freq = f1 * (2 ** target_zone_freq_factor)

            # Filter the anchors within the time range
            time_mask = (anchor_times >= min_time) & (anchor_times <= max_time)
            anchors_in_time_range = anchors_array[time_mask]

            # Filter the anchors within the frequency range
            freq_mask = (anchors_in_time_range[:, 1].astype(float) >= min_freq) & (anchors_in_time_range[:, 1].astype(float) <= max_freq)
            anchors_in_target_zone = anchors_in_time_range[freq_mask]

            # Compute the hashes only for anchors within the target zone
            for anchor2 in anchors_in_target_zone:
                t2, f2, _ = anchor2
                delta_t12 = t2 - t1 
                hash_tuple = (f1, f2, delta_t12)
                hashes.append((t1, hash_tuple))

        return hashes
    
    
    def getHash(self, delta_T_ms=200, n_bands=20, target_zone_time_s=0.5, target_zone_time_offset_s=0.1, target_zone_freq_factor=0.5, plot_flag=False, plot_duration=10, CFAR_flag=True):
        """
        Computes the hashes from the given anchors, this is similar to computeHash, but you dont need to call previous funcs.
        CFAR_flag: Enable or not the CFAR(Constant False Alarm Rate) detector which is for reducing redundant anchors.
        plot_duration: the time duration of anchors map (if plotted).
        """
        S = self.stft(plot_flag=False)
        anchors = self.getAnchor(S, delta_T_ms, n_bands, plot_flag, plot_duration, CFAR_flag)
        #print(f"number of anchors = {len(anchors)}")
        hashes = self.computeHash(anchors, target_zone_time_s, target_zone_time_offset_s, target_zone_freq_factor)
        return hashes
    
    
class MusicFingerPrintFromData(MusicFingerPrint):
    """
    Sub-Class for clipped audios, for convenience.
    """
    def __init__(self, data, sample_rate, window_length_ms=100, hop_length_ms=20, zero_padding=4):
        self.data = data
        self.sample_rate = sample_rate
        self.window_length_ms = window_length_ms
        self.hop_length_ms = hop_length_ms
        self.window_length = int((window_length_ms / 1000) * self.sample_rate)
        self.hop_length = int((hop_length_ms / 1000) * self.sample_rate)
        self.n_fft = self.window_length * zero_padding
        
        

def invertedIndexTable(folder_path, window_length_ms=100, hop_length_ms=20, zero_padding=4,
                       delta_T_ms=200, n_bands=20, target_zone_time_s=0.5, target_zone_time_offset_s=0.1,
                       target_zone_freq_factor=0.5, progress_bar=True, CFAR_flag=True, 
                       multithread=True, num_workers=4):
    """
    Build the inverted index for the fingerprints of all songs in a given folder.

    Parameters:
        multithread (bool): Whether to use multithreading.
        num_workers (int): If multithread is True, the number of worker threads to use.
        ** For other params, see the comments in function `folderToFolderMatching`

    Returns:
        inverted_index (dict): A dictionary that maps hashes to a list of (song ID, time offset) tuples.
    """
    
    inverted_index = {}
    song_lengths = {}  # To store song lengths for future use

    # Collect all filenames in the folder
    all_files = []
    for _, _, files in os.walk(folder_path):
        for file_name in files:
            all_files.append(file_name)
        break  # Since all files are in a single folder, no need to recurse

    if multithread:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_file(folder_file_name):
            music_archive = MusicFingerPrint(folder_path, folder_file_name, window_length_ms, hop_length_ms, zero_padding)
            song_id = folder_file_name
            music_archive_hash = music_archive.getHash(delta_T_ms, n_bands, target_zone_time_s,
                                                       target_zone_time_offset_s, target_zone_freq_factor,
                                                       plot_flag=False, CFAR_flag=CFAR_flag)
            partial_inverted_index = {}
            for t_offset, h in music_archive_hash:
                if h not in partial_inverted_index:
                    partial_inverted_index[h] = []
                partial_inverted_index[h].append((song_id, t_offset))
            return partial_inverted_index

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_file, file_name): file_name for file_name in all_files}

            if progress_bar:
                futures_iter = tqdm(as_completed(futures), total=len(futures), desc="Building Inverted Index")
            else:
                futures_iter = as_completed(futures)

            for future in futures_iter:
                partial_inverted_index = future.result()
                # Merge partial inverted index into the main inverted index
                for h, entries in partial_inverted_index.items():
                    if h not in inverted_index:
                        inverted_index[h] = []
                    inverted_index[h].extend(entries)
    else:
        if progress_bar:
            all_files_iter = tqdm(all_files, desc="Building Inverted Index")
        else:
            all_files_iter = all_files

        for folder_file_name in all_files_iter:
            music_archive = MusicFingerPrint(folder_path, folder_file_name, window_length_ms, hop_length_ms, zero_padding)
            song_id = folder_file_name
            music_archive_hash = music_archive.getHash(delta_T_ms, n_bands, target_zone_time_s,
                                                       target_zone_time_offset_s, target_zone_freq_factor,
                                                       plot_flag=False, CFAR_flag=CFAR_flag)
            for t_offset, h in music_archive_hash:
                if h not in inverted_index:
                    inverted_index[h] = []
                inverted_index[h].append((song_id, t_offset))

    return inverted_index


def queryFromTable(query_music, inverted_index, CFAR_flag=True):
    """
    Searches for the query music using the inverted index.
    ** The function is used inside `musicToFolderMatching` **
    
    Parameters:
    query_music (str): Path to query music
    inverted_index (dict): A dictionary that maps hashes to a list of (song ID, time offset) tuples.
    ** For other params, see the comments in function `folderToFolderMatching`

    Returns:
        A dictionary with song IDs as keys and their respective match counts.
    """
    query_hashes = query_music.getHash(plot_flag=False, CFAR_flag=CFAR_flag)
    time_diffs = {}

    for t_query, h in query_hashes:
        if h in inverted_index:
            matches = inverted_index[h]
            for song_id, t_song in matches:
                delta_t = t_song - t_query
                delta_t = round(delta_t, 1)  # Round delta_t to one decimal place
                if song_id not in time_diffs:
                    time_diffs[song_id] = []
                time_diffs[song_id].append(delta_t)

    # Now, for each song_id, compute match_counts[song_id] as the count of the mode
    match_counts = {}
    
    for song_id, deltas in time_diffs.items():
        counts = Counter(deltas)
        mode_delta_t, N = counts.most_common(1)[0]  # mode and its count
        match_counts[song_id] = N

    return match_counts, time_diffs


def musicToFolderMatching(music_path, music_name, folder_path, inverted_index=None,
                          window_length_ms=100, hop_length_ms=20, zero_padding=4,
                          delta_T_ms=200, n_bands=20, target_zone_time_s=0.5,
                          target_zone_time_offset_s=0.1, target_zone_freq_factor=0.5,
                          progress_bar=True, CFAR_flag=True):
    """
    Uses the inverted index to match the query music with songs in the folder.
    
    Parameters:
    music_path (str): Path to the folder containing the query music
    music_name (str): File name of query music
    folder_path (str): Path to the folder containing original music files (clean).
    inverted_index (dict): A dictionary that maps hashes to a list of (song ID, time offset) tuples.
    ** For other params, see the comments in function `folderToFolderMatching`

    Returns:
        The best matching song filename and the match statistics.
        ......
    """
    # Build the inverted index
    if inverted_index == None:
        inverted_index = invertedIndexTable(folder_path, window_length_ms, hop_length_ms, zero_padding,
                                            delta_T_ms, n_bands, target_zone_time_s,
                                            target_zone_time_offset_s, target_zone_freq_factor,
                                            progress_bar, CFAR_flag, multithread=True, num_workers=4)

    # Create MusicFingerPrint object for the query music
    query_music = MusicFingerPrint(music_path, music_name, window_length_ms, hop_length_ms, zero_padding)

    # Search using the inverted index
    match_counts, time_diffs = queryFromTable(query_music, inverted_index)

    # Modify the matching process to keep only the part before '-'
    match_counts_cleaned = {}
    for song_id in match_counts:
        cleaned_song_id = song_id.split('-')[0]  # Keep the part before '-'
        if cleaned_song_id in match_counts_cleaned:
            match_counts_cleaned[cleaned_song_id] += match_counts[song_id]
        else:
            match_counts_cleaned[cleaned_song_id] = match_counts[song_id]

    # Determine the best match
    if match_counts_cleaned:
        best_match = max(match_counts_cleaned, key=match_counts_cleaned.get)
        print(f"Best match: {best_match} with {match_counts_cleaned[best_match]} matching hashes.")
    else:
        print("No match found.")
        best_match = None

    return best_match, match_counts_cleaned, time_diffs


def folderToFolderMatching(folder_path, query_folder_path, inverted_index=None,
                           window_length_ms=100, hop_length_ms=20, zero_padding=4,
                           delta_T_ms=200, n_bands=20, target_zone_time_s=0.5,
                           target_zone_time_offset_s=0.1, target_zone_freq_factor=0.5,
                           progress_bar=True, CFAR_flag=True, accuracy_flag=True,
                           report_flag=True, confusion_flag=True):
    """
    Matches all query songs in query_folder_path against songs in folder_path.

    Args:
        **folder_path (str): Path to the folder containing original music files (clean).
        **query_folder_path (str): Path to the folder containing query music files (noisy).
        **inverted_index (dict): 1. ='None', call `invertedIndexTable` to compute inverted indexes using Muti-Thread.
                                 2. != 'None', load the pre-computed inverted indexes table.
        window_length_ms (int): Window length in milliseconds, see LAB3
        hop_length_ms (int): Hop length in milliseconds, see LAB3
        zero_padding (int): Amount of zero padding, see LAB3
        delta_T_ms (int): Delta time parameter in milliseconds, see LAB3
        n_bands (int): Number of frequency bands, see LAB3
        target_zone_time_s (float): Target zone time in seconds, see LAB3
        target_zone_time_offset_s (float): Target zone time offset in seconds, see LAB3
        target_zone_freq_factor (float): Target zone frequency factor, see LAB3
        progress_bar (bool): Whether to display a progress bar.
        CFAR_flag (bool): Whether to apply Constant False Alarm Rate (CFAR), from Radar Signal Processing, by Yuqi.
        accuracy_flag (bool): Whether to print the classification accuracy.
        report_flag (bool): Whether to print the classification report.
        confusion_flag (bool): Whether to display the confusion matrix.

    Returns:
        predicted_labels (list): List of predicted song IDs (without file extensions).
        true_labels (list): List of true song IDs (without file extensions).
        inverted_index (dict): A dictionary that maps hashes to a list of (song ID, time offset) tuples for musics in 'folder_path'.
    """

    # Build the inverted index from the music folder
    if inverted_index == None:
        inverted_index = invertedIndexTable(folder_path, window_length_ms, hop_length_ms, zero_padding,
                                            delta_T_ms, n_bands, target_zone_time_s,
                                            target_zone_time_offset_s, target_zone_freq_factor,
                                            progress_bar, CFAR_flag, multithread=True, num_workers=4)

    predicted_labels = []
    true_labels = []

    # Get list of query files
    query_files = [f for f in os.listdir(query_folder_path) if f.endswith('.wav')]

    if progress_bar:
        query_files = tqdm(query_files, desc="Matching Queries")

    for query_file_name in query_files:
        # Create MusicFingerPrint object for the query music
        query_music = MusicFingerPrint(query_folder_path, query_file_name, window_length_ms, hop_length_ms, zero_padding)

        # Search using the inverted index
        query_hashes = query_music.getHash(plot_flag=False, CFAR_flag=CFAR_flag)
        time_diffs = {}

        for t_query, h in query_hashes:
            if h in inverted_index:
                matches = inverted_index[h]
                for song_id, t_song in matches:
                    delta_t = t_song - t_query
                    delta_t = round(delta_t, 1)  # Round delta_t to one decimal place
                    if song_id not in time_diffs:
                        time_diffs[song_id] = []
                    time_diffs[song_id].append(delta_t)

        # Now, for each song_id, compute match_counts[song_id] as the count of the mode
        match_counts = {}

        for song_id, deltas in time_diffs.items():
            counts = Counter(deltas)
            mode_delta_t, N = counts.most_common(1)[0]  # mode and its count
            match_counts[song_id] = N

        # Determine the best match
        if match_counts:
            best_match = max(match_counts, key=match_counts.get)
            predicted_label = os.path.splitext(best_match.split('-')[0])[0]  # Keep part before '-'
        else:
            predicted_label = "Unknown"

        # Modify true_label extraction to keep only the part before '-'
        if os.path.splitext(query_file_name.split('-')[0])[1] != '.wav':
            true_label = os.path.splitext(query_file_name.split('-')[0])[0] + os.path.splitext(query_file_name.split('-')[0])[1]
        else:
            true_label = os.path.splitext(query_file_name.split('-')[0])[0]

        predicted_labels.append(predicted_label)
        true_labels.append(true_label)

    # Confusion matrix, classification accuracy, and report generation
    if confusion_flag:
        confusion = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    # Compute and display classification accuracy if accuracy_flag is True
    if accuracy_flag:
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Classification Accuracy: {accuracy:.4f}')

    # Generate and display the classification report if report_flag is True
    if report_flag:
        report = classification_report(true_labels, predicted_labels, labels=sorted(set(true_labels + predicted_labels)))
        print("\nClassification Report:")
        print(report)

    return predicted_labels, true_labels, inverted_index
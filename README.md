# Noise Resilient Audio Fingerprinting Algorithm using CFAR Filtering and Inverted Index

## Overview
This project implements a music fingerprinting system, which allows for fast and accurate identification of music tracks from audio snippets, even when the audio is distorted or contains background noise. The core functionality is built upon **Short-Time Fourier Transform (STFT)** for fingerprint extraction, and it leverages an **inverted index** for efficient querying. Additionally, a **Constant False Alarm Rate (CFAR)** filter inspired by radar signal processing is used to enhance noise robustness.

## Features
- **Music Fingerprint Extraction**: Uses STFT to generate time-frequency representations of music, and creates unique fingerprints for each song.
- **Inverted Index**: Organizes fingerprints in an inverted index structure, improving query speed significantly compared to linear search.
- **Noise Robustness**: Employs a CFAR filter to reduce the impact of noise on anchor points, improving accuracy for noisy queries.
- **Efficient Querying**: Supports multithreaded inverted index construction and fast lookup of music fingerprints, allowing for near real-time music identification.

## Requirements
- **Python 3.7+**
- **Libraries**: 
  - `librosa`
  - `matplotlib`
  - `numpy`
  - `sklearn`
  - `tqdm`
  - `pickle`
  - `seaborn`

## Structure
- **Data/**: Contains the data used in this project.
- **Inverted_Index/**: Stores the pre-built inverted index file.
- **main.ipynb**: Jupyter notebook demonstrating the usage of our algorithm.
- **utils.py**: Contains the core classes and functions for music fingerprinting, index building, and querying.
- **README.md**: Project documentation (this file).

## Performance
The system has been tested on the **GTZAN dataset**, with various levels of noise (AWGN, MIREX noise). In real-world scenarios, the system achieves high accuracy even with noisy queries, and its query times have been reduced dramatically by using an inverted index.

For example:
- Without an inverted index, querying a 5-second clip took over 8 minutes.
- With the inverted index, the same query took just 0.073 seconds.

The system's robustness to noise has been improved by CFAR, resulting in more accurate matches even in challenging conditions.

## Contributors
Ziyue Yang, Yuqi Zhang, Erik Halme, Hao Wang

## References
1. Wang, A. L. C. "An industrial-strength audio search algorithm." *ISMIR*, 2003. [Link](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
2. Li, S. Pan, and H. Liu, “The 2020 netease audio fingerprint system,” in *Proceedings of the International Symposium on Music Information Retrieval (ISMIR)*, 2020. [Link](https://music-ir.org/mirex/abstracts/2020/LPL1.pdf)
3. A. Jalil, H. Yousaf, and M. I. Baig, “Analysis of CFAR techniques,” in *Proceedings of the 2016 13th International Bhurban Conference on Applied Sciences and Technology (IBCAST)*, 2016, pp. 654–659. [Link](https://ieeexplore.ieee.org/abstract/document/7429949)




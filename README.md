# Detecting Deepfake Audio

## Overview

This project focuses on the detection of deepfake audio, a critical issue that poses significant challenges in various domains, including communication, media, and security. Through thorough exploration and analysis, we have developed robust and efficient methods for accurate detection of deepfake voices.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Final Research Problem Assessment](#final-research-problem-assessment)
- [Conclusion](#conclusion)
- [Clone](#clone)

## Introduction

In recent years, the proliferation of deepfake technology has raised concerns about the authenticity of audio content. Deepfake audio can be used to manipulate trust, spread misinformation, and compromise security. This project aims to develop methods for accurately detecting deepfake voices to combat these risks.

## Dataset

We used a high-quality dataset provided by [Fraunhofer AISEC](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild), comprising bonafide and spoof audio collected from various sources, including public figures and politicians. The dataset contains 38 hours of audio data in .wav format, along with associated metadata. Through thorough exploration and preprocessing, we prepared the data for model building.

### Dataset Details

- *Source*: Fraunhofer AISEC
- *Format*: .wav
- *Size*: 8.16 GB
- *Number of Records*: 31,779
- *Content*: Audio of public figures, politicians, and common people, collected from social networks and other video streaming platforms.

## Exploratory Data Analysis (EDA)

During the EDA phase, we explored the dataset to gain insights into the distribution of real and deepfake audio samples. We visualized waveforms, spectrograms, and other audio features to better understand the characteristics of the data. This analysis guided our feature selection and preprocessing steps.

### EDA Process

- *Data Loading*: Loading the dataset and metadata.
- *Audio Visualization*: Visualizing waveforms, spectrograms, and other audio features.
- *Data Preprocessing*: Extracting and preprocessing audio features, such as MFCCs.

## MFCC Features

Mel-Frequency Cepstral Coefficients (MFCCs) are commonly used in audio processing and speech recognition tasks due to their effectiveness in capturing essential characteristics of the audio signal. MFCCs are extracted using the following steps:

1. *Pre-emphasis*: To enhance high-frequency components, the audio signal is passed through a pre-emphasis filter, typically implemented as a first-order high-pass filter.

   ![Pre-emphasis Formula](https://latex.codecogs.com/svg.latex?y(t)%20%3D%20x(t)%20-%20%5Calpha%20x(t-1))

   Where \(x(t)\) is the input audio signal, \(y(t)\) is the output of the pre-emphasis filter, and \(\alpha\) is the pre-emphasis coefficient.

2. *Framing*: The pre-emphasized signal is divided into short frames of typically 20-40 ms duration. Each frame is multiplied by a window function such as Hamming or Hanning to reduce spectral leakage.

3. *Fast Fourier Transform (FFT)*: The power spectrum of each frame is computed using the FFT.

4. *Mel Filtering*: The power spectrum is then passed through a bank of Mel filters spaced uniformly in the mel-frequency scale. The output of each filter is the log energy.

5. *Discrete Cosine Transform (DCT)*: Finally, the DCT is applied to the log filterbank energies to obtain the MFCCs. Typically, only the lower-order coefficients are retained as they contain the most relevant information.

In mathematical terms, the MFCC extraction process can be represented as follows:

![MFCC Formula](https://latex.codecogs.com/svg.latex?%5Ctext%7BMFCC%7D(n)%20%3D%20%5Csum_%7Bm%3D0%7D%5E%7BM-1%7D%20%5Clog%5Cleft(%5Csum_%7Bk%3D0%7D%5E%7BN-1%7D%20%7CX(k)%7C%5E2%20H_m(k)%20%5Ccos%5Cleft%5B%5Cfrac%7B%5Cpi%7D%7BM%7D(m&plus;0.5)k%5Cright%5D%5Cright%29%20%5Ccos%5Cleft%5B%5Cfrac%7B%5Cpi%7D%7BM%7D(n&plus;0.5)m%5Cright%5D)

Where:
- \(N\) is the number of FFT bins,
- \(M\) is the number of Mel filters,
- \(X(k)\) is the FFT of the framed signal,
- \(H_m(k)\) is the m-th Mel filter,
- \(n\) is the MFCC index.

Mel-Frequency Cepstral Coefficients (MFCCs) are commonly used in audio processing and speech recognition tasks due to their effectiveness in capturing essential characteristics of the audio signal. Compared to other audio features such as chromagrams and spectrograms, MFCCs often outperform in deepfake audio detection tasks for several reasons:

1. *Capture of Timbral Information*: MFCCs effectively capture the timbral characteristics of the audio signal, including spectral envelope information. This allows them to represent the unique characteristics of a speaker's voice, making them valuable for voice-based tasks like speaker recognition and voice authentication.

2. *Dimensionality Reduction*: MFCCs provide a compact representation of the audio signal by reducing the dimensionality of the feature space. This is achieved through the application of the Discrete Cosine Transform (DCT), which decorrelates the coefficients and retains the most relevant information. The reduced dimensionality helps in mitigating the curse of dimensionality and improves the efficiency of machine learning models.

3. *Robustness to Noise*: MFCCs are relatively robust to noise and other environmental factors, making them suitable for real-world applications where audio data may be subject to various sources of interference. By focusing on the spectral characteristics of the signal, MFCCs can effectively discriminate between genuine and manipulated audio even in noisy environments.

4. *Human Auditory Perception*: The extraction process of MFCCs is inspired by the human auditory system's response to sound frequencies. The Mel-filterbank mimics the frequency resolution of the human ear, leading to features that are perceptually relevant and aligned with human auditory perception. This ensures that the features captured by MFCCs are meaningful for tasks involving human speech and audio.

In contrast, other audio features like chromagrams and spectrograms may lack the compactness, robustness, or perceptual relevance of MFCCs, making them less suitable for tasks like deepfake audio detection. While these features may still be valuable in certain contexts, MFCCs are often preferred for their effectiveness in capturing the essential characteristics of the audio signal.

## Model Building

We experimented with various machine learning and deep learning models for deepfake audio detection. Models included traditional algorithms such as Naive Bayes, K-Nearest Neighbors, and Decision Trees, as well as deep learning architectures like Convolutional Neural Networks (CNN), Gated Recurrent Units (GRU), and Long Short-Term Memory (LSTM) networks. Through iterative model updates and performance evaluations, we identified LSTM as the best performing model.

### Model Building Process

- *Supervised Classification Models*: Naive Bayes, KNN, Decision Trees, CNN, GRU, LSTM.
- *Hyperparameter Tuning*: Tuning model parameters for optimal performance.
- *Evaluation Metrics*: Assessing model performance using accuracy, precision, recall, and F1-score.

## Final Research Problem Assessment

We addressed several research questions, including the effectiveness of different audio features, strategies for addressing class imbalance, generalization to unseen data, and optimization for real-time detection. Our findings underscored the importance of feature selection, preprocessing techniques, and model optimization in achieving robust detection capabilities.

### Research Questions

1. Which audio features exhibit the highest discriminatory power between real and deepfake voices?
2. How can class imbalance in the dataset be effectively addressed to prevent biased model training?
3. To what extent does the trained model generalize to unseen deepfake voices?
4. How can the model be optimized for real-time detection of deepfake voices?
5. What are the practical challenges and considerations for deploying deepfake voice detection systems?

## Conclusion

This project contributes to combating the spread of deepfake audio, safeguarding trust, security, and integrity in digital communication and media. The LSTM model demonstrated promising results in detecting deepfake voices with high accuracy and precision. Future research may focus on exploring advanced deep learning architectures, addressing class imbalance issues, and optimizing models for real-time detection.

## Clone

You can clone the repository using the following command:
git clone https://github.com/rajduttmamidi/Deep-Fake-Audio.git

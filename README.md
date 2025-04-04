# Preventing Malicious Audio Transformations via Watermarking

## Overview

This repository implements a novel audio watermarking scheme designed to protect digital audio against malicious transformations, such as deepfakes. Our approach embeds a secure watermark into the audio’s spectral domain using Fast Fourier Transform (FFT) techniques.
## Features

- **Secure Watermark Generation:**  
  - Uses SHA-256 hashing to generate a robust, fixed-length binary watermark.
  - Randomized embedding strategy to improve security.
  
- **Adaptive Watermark Embedding:**  
  - Embeds watermark bits into selected frequency bins of the FFT-transformed audio.
  - Two implementations are provided:
    - Core embedding with model-based parameter prediction in [`watermark_via_model.py`](watermark_via_model.py) [&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}].
    - Alternative randomized embedding in [`wm_gen_embed.py`](wm_gen_embed.py) [&#8203;:contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}].

- **Watermark Parameter Prediction Model:**  
  - Trains a Random Forest classifier on extracted audio features to predict the optimal frequency and amplitude for embedding [&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}].

- **Evaluation Metrics:**  
  - Provides utilities to compute Signal-to-Noise Ratio (SNR), Cross-Correlation (CC), PESQ, and STOI to assess watermark integrity and audio quality [&#8203;:contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}].

## Repository Structure

- **`watermark_via_model.py`**  
  Contains functions for generating a watermark, extracting audio features, predicting embedding parameters using a pre-trained model, and embedding the watermark into an audio file.

- **`wm_gen_embed.py`**  
  An alternative implementation that embeds watermarks using randomized frequency and amplitude selections.

- **`model_generation.py`**  
  Script for training a Random Forest classifier on audio features to predict optimal watermark embedding parameters.

- **`evaluation_criteria.py`**  
  Provides evaluation utilities to calculate SNR, Cross-Correlation, PESQ, and STOI, and to generate comparison plots for assessing watermark robustness.

## Requirements

- Python 3.7+
- NumPy, SciPy, Pandas
- scikit-learn, joblib
- cupy, librosa, soundfile, matplotlib
- Optional: pesq, pystoi (for PESQ and STOI calculations)

Install dependencies with:

```bash
pip install -r requirements.txt

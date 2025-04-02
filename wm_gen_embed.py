import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
import pandas as pd
import os
import hashlib

def generate_watermark(key, desired_length):
    np.random.seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))
    watermark = ""
    while len(watermark) < desired_length:
        random_bytes = np.random.bytes(16)
        hash_val = hashlib.sha256(random_bytes).hexdigest()  # 64-character hex string
        watermark += hash_val
    return watermark[:desired_length]

def embed_watermark(data, samplerate, key, freq_range, amp_range):
    desired_length = len(data) * data.dtype.itemsize * 8
    watermark = generate_watermark(key, desired_length)
    
    data_length = len(data)
    fft_data = fft(data)
    
    np.random.seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))
    
    for digit in watermark:
        bit_val = int(digit, 16) % 2
        freq = np.random.uniform(freq_range[0], freq_range[1])
        current_amp = np.random.uniform(amp_range[0], amp_range[1])
        index = int(freq / samplerate * data_length)
        if index >= len(fft_data):
            index = len(fft_data) - 1
        if bit_val == 1:
            fft_data[index] *= (1 + current_amp)
        else:
            fft_data[index] *= (1 - current_amp)
    
    modified_data = np.real(ifft(fft_data))
    return modified_data.astype(np.int16), watermark

def calculate_snr(original_data, watermarked_data):
    signal_power = np.mean(original_data**2)
    noise_power = np.mean((original_data - watermarked_data)**2)
    epsilon = 1e-10
    return 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))

def calculate_cross_correlation(original_data, watermarked_data):
    correlation = np.correlate(original_data, watermarked_data, 'full')
    max_correlation = np.max(correlation)
    norm = np.linalg.norm(original_data) * np.linalg.norm(watermarked_data)
    if norm == 0:
        return 0
    return max_correlation / norm

def process_audio_files(audio_files, output_folder, freq_range, amp_range):
    data = []
    for i, file_path in enumerate(audio_files):
        rate, audio_data = wav.read(file_path)
        key = f"example_key_{i+1}"
        original_data = np.copy(audio_data)
        watermarked_data, watermark = embed_watermark(audio_data, rate, key, freq_range, amp_range)
        snr = calculate_snr(original_data, watermarked_data)
        correlation = calculate_cross_correlation(original_data, watermarked_data)
        data.append([key, watermark, snr, correlation])
        output_file_path = os.path.join(output_folder, f'ex{i+1}_watermarked.wav')
        wav.write(output_file_path, rate, watermarked_data)
    df = pd.DataFrame(data, columns=['Key', 'Watermark', 'SNR', 'Correlation'])
    df.index = [f'ex{i+1}' for i in range(len(audio_files))]
    return df

def get_audio_files_from_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    return files

# Configuration
audio_folder = 'C:/Users/forcode/Desktop/audio/eldata'
audio_files = get_audio_files_from_folder(audio_folder)
output_folder = 'path to your file'
freq_range = (0, 10000)      # Frequency range: 0 to 10,000 Hz
amp_range = (0.01, 100)      # Amplitude range

os.makedirs(output_folder, exist_ok=True)

dataset = process_audio_files(audio_files, output_folder, freq_range, amp_range)


import numpy as np
import scipy.io.wavfile as wav
import joblib
import os
import warnings
import hashlib

model_path = ''
model = joblib.load(model_path)

def generate_watermark(key, desired_length):
    np.random.seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))
    watermark = ""
    while len(watermark) < desired_length:
        random_bytes = np.random.bytes(16)
        hash_val = hashlib.sha256(random_bytes).hexdigest()  # 64 hex characters
        watermark += hash_val
    return watermark[:desired_length]

def embed_watermark(data, samplerate, key, base_freq, base_amp):
    desired_length = len(data) * data.dtype.itemsize * 8
    watermark = generate_watermark(key, desired_length)
    print("Generated watermark length:", len(watermark))
    
    data_length = len(data)
    fft_data = np.fft.fft(data)

    delta = 10  
    for i, digit in enumerate(watermark):
        bit_val = int(digit, 16) % 2  # Convert hex digit (0-15) to binary decision (0 or 1)
        freq_to_embed = base_freq + i * delta
        index = int(freq_to_embed / samplerate * data_length)
        if index >= len(fft_data):
            index = len(fft_data) - 1
        if bit_val == 1:
            fft_data[index] *= (1 + base_amp)
        else:
            fft_data[index] *= (1 - base_amp)
    
    modified_data = np.real(np.fft.ifft(fft_data))
    return modified_data.astype(np.int16)

def extract_audio_features(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wav.WavFileWarning)
        rate, data = wav.read(file_path)
        data = data.astype(float)
    features = [
        np.mean(data), np.std(data), np.max(data), np.min(data),
        np.median(data), np.percentile(data, 25), np.percentile(data, 75),
        np.var(data), np.ptp(data), np.mean(np.abs(data)), np.std(np.abs(data)),
        np.max(np.abs(data)), np.min(np.abs(data)),
        np.median(np.abs(data)), np.percentile(np.abs(data), 25), np.percentile(np.abs(data), 75)
    ]
    return np.array(features).reshape(1, -1), rate, data

def apply_watermark(audio_file_path, output_folder, key):
    features, rate, data = extract_audio_features(audio_file_path)
    print("Extracted features:", features)
    
    # Obtain model predictions (assumed to be [base_freq, base_amp])
    predicted_params = model.predict(features)
    print("Model predicted parameters:", predicted_params)
    
    if isinstance(predicted_params[0], (list, np.ndarray)) and len(predicted_params[0]) >= 2:
        base_freq, base_amp = predicted_params[0][0], predicted_params[0][1]
    else:
        base_freq = predicted_params[0]
        base_amp = 0.01  # Default amplitude scaling factor if not provided
    
    modified_data = embed_watermark(data, rate, key, base_freq, base_amp)
    base_filename = os.path.basename(audio_file_path)
    output_path = os.path.join(output_folder, base_filename.replace('.WAV', '_watermarked.wav').replace('.wav', '_watermarked.wav'))
    wav.write(output_path, rate, modified_data)
    print(f"Watermarked audio saved to {output_path}")

# Configuration for file paths and key
audio_folder = ''
output_folder = ''
key = "secret_key"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each audio file in the specified folder
for i in range(1, 5245):
    audio_file_path = os.path.join(audio_folder, f'ex_speechocean_{i}.WAV')
    if os.path.exists(audio_file_path):
        apply_watermark(audio_file_path, output_folder, key)
    else:
        print(f"Audio file {audio_file_path} not found.")

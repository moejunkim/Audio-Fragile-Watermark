import os
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.signal import correlate as cp_correlate

from tqdm import tqdm

import librosa

try:
    from pesq import pesq
except ImportError:
    pesq = None
    print("[Warning] PESQ library not found. Please install via `pip install pesq` if needed.")

try:
    from pystoi.stoi import stoi
except ImportError:
    stoi = None
    print("[Warning] STOI library not found. Please install via `pip install pystoi` if needed.")


###############################################################################
# Utility: Calculate SNR
###############################################################################
def calculate_snr(original, watermarked):
    """
    Calculate Signal-to-Noise Ratio (SNR) between original and watermarked signals.
    """
    min_length = min(len(original), len(watermarked))
    if min_length == 0:
        return 0.0  # Avoid division by zero

    orig_gpu = cp.asarray(original[:min_length])
    water_gpu = cp.asarray(watermarked[:min_length])
    eps = 1e-12

    noise_gpu = orig_gpu - water_gpu
    signal_power = cp.sum(orig_gpu**2) + eps
    noise_power = cp.sum(noise_gpu**2) + eps
    snr_gpu = 10 * cp.log10(signal_power / noise_power)
    return float(snr_gpu.get())


###############################################################################
# Utility: Calculate Cross-Correlation
###############################################################################
def calculate_cross_correlation(original, watermarked):
    """
    Calculate the maximum cross-correlation between original and watermarked signals.
    """
    min_length = min(len(original), len(watermarked))
    if min_length == 0:
        return 0.0  # Avoid invalid operations

    orig_gpu = cp.asarray(original[:min_length])
    water_gpu = cp.asarray(watermarked[:min_length])
    corr_gpu = cp_correlate(orig_gpu, water_gpu, mode='valid')
    denom = cp.sqrt(cp.sum(orig_gpu**2) * cp.sum(water_gpu**2)) + 1e-12
    if denom == 0:
        return 0.0
    max_corr_gpu = cp.max(corr_gpu / denom)
    return float(max_corr_gpu.get())


###############################################################################
# Utility: Calculate PESQ
###############################################################################
def calculate_pesq(original, watermarked, sr):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality).
    - By ITU standard, valid at 8k or 16k sample rate only.
    - We will force a resample to 16k in this example.
    - If `pesq` library is not installed, returns np.nan.
    - We also check for minimum duration and non-silence to avoid
      "Buffer needs to be at least 1/4 second long" or "No utterances detected."
    """
    if pesq is None:
        return np.nan

    target_sr = 16000
    try:
        # === Resample if needed
        if sr != target_sr:
            orig_16k = librosa.core.resample(
                y=original.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr
            )
            water_16k = librosa.core.resample(
                y=watermarked.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr
            )
        else:
            orig_16k = original.astype(np.float32)
            water_16k = watermarked.astype(np.float32)

        # === Check length threshold (e.g., 0.25s)
        if (len(orig_16k) < target_sr * 0.25) or (len(water_16k) < target_sr * 0.25):
            return np.nan

        # === Check RMS amplitude to avoid "No utterances detected"
        rms_orig = np.sqrt(np.mean(orig_16k**2))
        rms_wm = np.sqrt(np.mean(water_16k**2))
        if (rms_orig < 1e-7) or (rms_wm < 1e-7):
            return np.nan

        # PESQ mode 'wb' = wide band (16k), 'nb' = narrow band (8k)
        pesq_val = pesq(
            fs=target_sr,
            ref=orig_16k,
            deg=water_16k,
            mode='wb'  # wide band
        )
        return pesq_val

    except Exception as e:
        print(f"[Warning] PESQ calculation failed: {e}")
        return np.nan


###############################################################################
# Utility: Calculate STOI
###############################################################################
def calculate_stoi_metric(original, watermarked, sr):
    """
    Calculate STOI (Short-Time Objective Intelligibility).
    - Typically used at 10 kHz or 16 kHz.
    - We'll use 16k here.
    - If `pystoi` library is not installed, returns np.nan.
    - Also check for minimum duration to avoid warnings about silent frames.
    - Also check RMS amplitude to avoid effectively silent signals.
    """
    if stoi is None:
        return np.nan

    target_sr = 16000
    try:
        # === Resample if needed
        if sr != target_sr:
            orig_16k = librosa.core.resample(
                y=original.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr
            )
            water_16k = librosa.core.resample(
                y=watermarked.astype(np.float32),
                orig_sr=sr,
                target_sr=target_sr
            )
        else:
            orig_16k = original.astype(np.float32)
            water_16k = watermarked.astype(np.float32)

        # === Check length threshold
        if (len(orig_16k) < target_sr * 0.25) or (len(water_16k) < target_sr * 0.25):
            return np.nan

        # === Check RMS amplitude
        rms_orig = np.sqrt(np.mean(orig_16k**2))
        rms_wm = np.sqrt(np.mean(water_16k**2))
        if (rms_orig < 1e-7) or (rms_wm < 1e-7):
            return np.nan

        # STOI returns a value in [0, 1]
        stoi_val = stoi(orig_16k, water_16k, target_sr, extended=False)
        return stoi_val

    except Exception as e:
        print(f"[Warning] STOI calculation failed: {e}")
        return np.nan

def main():
    # Define the structure for datasets, deepfake methods, and watermark methods
    #datasets

    # Define output directory
    output_dir = r'path to your output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = []
    print("==== STARTING COMPARISON ====")

    for ds_name, ds_content in datasets.items():
        print(f"\n[Dataset: {ds_name}] Processing...")

        for deepfake_method, df_content in ds_content['deepfakes'].items():
            print(f"  [Deepfake Method: {deepfake_method}]")

            # Original data configuration
            orig_cfg = df_content['original']
            orig_path = orig_cfg['path']
            orig_prefix = orig_cfg['prefix']
            orig_suffix = orig_cfg['suffix']
            orig_ext = orig_cfg['ext']
            orig_count = orig_cfg['count']
            orig_start_idx = orig_cfg['start_idx']

            # Iterate through each watermark method
            for wm_name, wm_content in watermark_methods.items():
                if ds_name not in wm_content:
                    continue
                if deepfake_method not in wm_content[ds_name]:
                    continue

                wm_cfg = wm_content[ds_name][deepfake_method]
                wm_path = wm_cfg['path']
                wm_prefix = wm_cfg['prefix']
                wm_suffix = wm_cfg['suffix']
                wm_ext = wm_cfg['ext']
                wm_count = wm_cfg['count']
                wm_start_idx = wm_cfg['start_idx']

                # Determine the range of indices to compare
                min_count = min(orig_count, wm_count)
                if min_count <= 0:
                    print(f"    [Warning] Invalid count for watermark method: {wm_name}")
                    continue

                # Process in chunks
                chunk_size = 500
                for chunk_start in range(orig_start_idx, orig_start_idx + min_count, chunk_size):
                    chunk_end = min(chunk_start + chunk_size - 1, orig_start_idx + min_count - 1)
                    file_indices = range(chunk_start, chunk_end + 1)

                    desc_str = f"{ds_name}-{deepfake_method}-{wm_name} [{chunk_start}~{chunk_end}]"
                    for i in tqdm(file_indices, desc=desc_str, leave=False):
                        orig_filename = f"{orig_prefix}{i}{orig_suffix}{orig_ext}"
                        orig_fullpath = os.path.join(orig_path, orig_filename)

                        wm_filename = f"{wm_prefix}{i}{wm_suffix}{wm_ext}"
                        wm_fullpath = os.path.join(wm_path, wm_filename)

                        if (not os.path.exists(orig_fullpath)) or (not os.path.exists(wm_fullpath)):
                            continue

                        try:
                            orig_data, orig_sr = sf.read(orig_fullpath)
                            wm_data, wm_sr = sf.read(wm_fullpath)
                        except Exception as e:
                            print(f"    [Warning] Could not read {orig_fullpath} or {wm_fullpath}, error: {e}")
                            continue
                        if orig_sr != wm_sr:
                            try:
                                wm_data = librosa.core.resample(
                                    y=wm_data.astype(np.float32),
                                    orig_sr=wm_sr,
                                    target_sr=orig_sr
                                )
                                wm_sr = orig_sr
                            except Exception as e:
                                print(f"    [Warning] Resample failed: {e}")
                                continue

                        if len(orig_data.shape) > 1:
                            orig_data = orig_data[:, 0]
                        if len(wm_data.shape) > 1:
                            wm_data = wm_data[:, 0]

                        min_length = min(len(orig_data), len(wm_data))
                        if min_length == 0:
                            print(f"    [Warning] One of the files {orig_fullpath} or {wm_fullpath} is empty after processing.")
                            continue
                        orig_data = orig_data[:min_length]
                        wm_data = wm_data[:min_length]

                        try:
                            snr_val = calculate_snr(orig_data, wm_data)
                            cross_val = calculate_cross_correlation(orig_data, wm_data)
                        except Exception as e:
                            print(f"    [Warning] Metric calculation failed for {orig_fullpath} and {wm_fullpath}, error: {e}")
                            continue

                        # === PESQ & STOI (Optional) ===
                        pesq_val = calculate_pesq(orig_data, wm_data, orig_sr)
                        stoi_val = calculate_stoi_metric(orig_data, wm_data, orig_sr)

                        row = {
                            'Dataset': ds_name,
                            'DeepfakeMethod': deepfake_method,
                            'WatermarkMethod': wm_name,
                            'FileIndex': i,
                            'SNR': snr_val,
                            'CrossCorrelation': cross_val,
                            'PESQ': pesq_val,
                            'STOI': stoi_val
                        }
                        rows.append(row)

            print(f"  [Dataset: {ds_name}, Deepfake Method: {deepfake_method}] Comparison complete.")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No data was processed. Please check file paths and prefixes.")
        return

    grouped = df.groupby(['Dataset', 'DeepfakeMethod', 'WatermarkMethod'])[['SNR', 'CrossCorrelation', 'PESQ', 'STOI']]
    summary_df = grouped.agg(['mean', 'min', 'max'])
    summary_df.columns = ['_'.join(col) for col in summary_df.columns]
    summary_df.reset_index(inplace=True)

    summary_csv_path = os.path.join(output_dir, "csv file name")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[INFO] Saved summary CSV to: {summary_csv_path}")

    # === CDF Plot (SNR, CrossCorrelation, PESQ, STOI) ===
    metrics = ['SNR', 'CrossCorrelation', 'PESQ', 'STOI']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for (ds, dfm, wm), subdf in df.groupby(['Dataset', 'DeepfakeMethod', 'WatermarkMethod']):
            values = subdf[metric].dropna().values
            if len(values) == 0:
                continue
            sorted_vals = np.sort(values)
            cdf_y = np.linspace(0, 1, len(sorted_vals))
            label_str = f"{ds}-{dfm}-{wm}"
            plt.plot(sorted_vals, cdf_y, label=label_str)

        plt.title(f"CDF of {metric}")
        plt.xlabel(metric)
        plt.ylabel("CDF")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        cdf_path = os.path.join(output_dir, f"CDF_{metric}.png")
        plt.savefig(cdf_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved CDF plot for {metric} to: {cdf_path}")

    raw_csv_path = os.path.join(output_dir, "deepfake_comparison_result_raw.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Saved raw data CSV to: {raw_csv_path}")

    print("\n==== ALL COMPARISONS COMPLETE ====")


if __name__ == "__main__":
    main()

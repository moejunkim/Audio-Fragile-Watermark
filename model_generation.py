import os
import warnings
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
dataset_path = ''
dataset = pd.read_csv(dataset_path)

# Function to extract audio features
def extract_audio_features(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wav.read(file_path)
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'max': np.max(data),
        'min': np.min(data),
    }

# Create a feature DataFrame
audio_folder = ''
watermarked_folder = ''
deepfake_folder = ''
watermarked_deepfake_folder = ''

features_list = []
for i in range(1, 10001):
    orig_path = os.path.join(audio_folder, f'ex{i}.wav')
    wm_path = os.path.join(watermarked_folder, f'ex{i}_watermarked.wav')
    df_path = os.path.join(deepfake_folder, f'ex{i}_df.wav')
    wdf_path = os.path.join(watermarked_deepfake_folder, f'ex{i}_watermarked_df.wav')
    
    if all(os.path.exists(p) for p in [orig_path, wm_path, df_path, wdf_path]):
        orig_features = extract_audio_features(orig_path)
        wm_features = extract_audio_features(wm_path)
        df_features = extract_audio_features(df_path)
        wdf_features = extract_audio_features(wdf_path)
        
        features = {
            'ID': f'ex{i}',
            'Orig_Mean': orig_features['mean'],
            'Orig_Std': orig_features['std'],
            'Orig_Max': orig_features['max'],
            'Orig_Min': orig_features['min'],
            'WM_Mean': wm_features['mean'],
            'WM_Std': wm_features['std'],
            'WM_Max': wm_features['max'],
            'WM_Min': wm_features['min'],
            'DF_Mean': df_features['mean'],
            'DF_Std': df_features['std'],
            'DF_Max': df_features['max'],
            'DF_Min': df_features['min'],
            'WDF_Mean': wdf_features['mean'],
            'WDF_Std': wdf_features['std'],
            'WDF_Max': wdf_features['max'],
            'WDF_Min': wdf_features['min'],
        }
        features_list.append(features)

features_df = pd.DataFrame(features_list)

# Merge with CSV dataset
full_data = pd.merge(features_df, dataset, on='ID', how='left')

# Handle missing values
imputer = SimpleImputer(strategy='median')
imputed_data = imputer.fit_transform(full_data.select_dtypes(include=[np.number]))
full_data_imputed = pd.DataFrame(imputed_data, columns=full_data.select_dtypes(include=[np.number]).columns)

# Define the target variable
full_data['Target'] = ((full_data['SNR'] > 25) & 
                       (full_data['Correlation'] > 0.9) &
                       (full_data['Deepfake_SNR'] < 25) & 
                       (full_data['Deepfake_Correlation'] < 0.9)).astype(int)

# Prepare data for training
X = full_data_imputed.drop('Target', axis=1)
y = full_data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Save the model
model_path = ''
joblib.dump(model, model_path)
print(f'Model saved to {model_path}')

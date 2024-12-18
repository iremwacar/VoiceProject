# app.py

from flask import Flask, render_template, request, jsonify
import os
import sounddevice as sd
import numpy as np
import threading
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
import librosa
import soundfile as sf

# Flask application
app = Flask(__name__, template_folder='../Frontend')

# Path configurations
DATA_DIR = "data"
FEATURES_CSV = "../Data/audio_features.csv"
MODEL_PATH = "../Model/random_forest_model.joblib"
SCALER_PATH = "../Model/scaler.joblib"
os.makedirs(DATA_DIR, exist_ok=True)

# Global variables
default_sr = 16000
segment_length_seconds = 1.5
classifier = None
scaler = None

# Helper functions
def record_audio(file_path, duration=5, samplerate=16000):
    print("Recording started...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Kayıt tamamlanana kadar bekle
    sf.write(file_path, audio, samplerate)
    print(f"Recording finished. Saved to {file_path}")

def extract_features(audio, sr=16000):
    """Extract features from audio signal."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    rms = librosa.feature.rms(y=audio)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features = np.concatenate((np.mean(mfccs, axis=1), np.mean(rms, axis=1), np.mean(chroma, axis=1)))
    return features

def segment_audio(file_path, segment_length=1.5):
    """Split audio file into segments of specified length."""
    audio, sr = librosa.load(file_path, sr=default_sr)
    segment_samples = int(segment_length * sr)
    return [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples) if len(audio[i:i + segment_samples]) == segment_samples]

def normalize_features(features_df, label_column='label'):
    """Normalize features and save the scaler."""
    global scaler
    scaler = MinMaxScaler()
    labels = features_df[label_column]
    features_df = features_df.drop(columns=[label_column])
    normalized_features = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(normalized_features, columns=features_df.columns)
    features_df[label_column] = labels
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    return features_df

def retrain_model():
    """Retrain the model with the updated dataset."""
    if os.path.exists(FEATURES_CSV):
        features_df = pd.read_csv(FEATURES_CSV)
    else:
        features_df = pd.DataFrame()

    X = features_df.iloc[:, :-1].values
    y = features_df['label'].values

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    global classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    joblib.dump(classifier, MODEL_PATH)
    print(f"Model retrained and saved to {MODEL_PATH}. Accuracy: {accuracy}")

# Flask routes
@app.route('/')
def index():
    return render_template('index5.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        user_name = request.json['name']
        if not user_name:
            return jsonify({"error": "User name is required."}), 400

        file_path = os.path.join(DATA_DIR, f"{user_name}.wav")
        record_audio(file_path, duration=90)

        segments = segment_audio(file_path, segment_length=segment_length_seconds)
        features = [extract_features(segment) for segment in segments]

        rows = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(len(features[0]))])
        rows['label'] = user_name

        if os.path.exists(FEATURES_CSV):
            features_df = pd.read_csv(FEATURES_CSV)
        else:
            features_df = pd.DataFrame()

        features_df = pd.concat([features_df, rows], ignore_index=True)
        features_df.to_csv(FEATURES_CSV, index=False)

        retrain_model()

        return jsonify({"message": "User added and model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_user', methods=['POST'])
def delete_user():
    try:
        user_name = request.json['name']
        if not user_name:
            return jsonify({"error": "User name is required."}), 400

        file_path = os.path.join(DATA_DIR, f"{user_name}.wav")
        if os.path.exists(file_path):
            os.remove(file_path)

        if os.path.exists(FEATURES_CSV):
            features_df = pd.read_csv(FEATURES_CSV)
            features_df = features_df[features_df['label'] != user_name]
            features_df.to_csv(FEATURES_CSV, index=False)

        retrain_model()

        return jsonify({"message": "User deleted and model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if classifier is None:
            return jsonify({"error": "Model not trained."}), 400

        duration = 5
        temp_file = "temp.wav"
        record_audio(temp_file, duration=duration)
        threading.Event().wait(duration + 1)

        segments = segment_audio(temp_file, segment_length=segment_length_seconds)
        if not segments:
            return jsonify({"error": "No valid audio segments found for prediction."}), 400

        features = [extract_features(segment) for segment in segments]
        predictions = [classifier.predict([feature])[0] for feature in features]
        most_common_prediction = max(set(predictions), key=predictions.count)

        return jsonify({"prediction": most_common_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

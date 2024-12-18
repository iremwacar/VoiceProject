from flask import Flask, render_template, request, jsonify
import os
import sounddevice as sd
import numpy as np
import threading
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

# Flask uygulaması
app = Flask(__name__, template_folder='../Frontend')

# Yol ayarları
DATA_DIR = "data"
FEATURES_CSV = "../Data/audio_features.csv"
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "../Model/scaler.joblib"
os.makedirs(DATA_DIR, exist_ok=True)

# Global değişkenler
default_sr = 16000
segment_length_seconds = 1.5
classifier = None
scaler = None

# Yardımcı Fonksiyonlar
def record_audio_async(file_path, duration=90):
    """Asenkron ses kaydı yapar."""
    def record():
        print("Recording started...")
        recording = sd.rec(int(duration * default_sr), samplerate=default_sr, channels=1, dtype='float32')
        sd.wait()
        np.save(file_path, recording)
        print(f"Recording finished. Saved to {file_path}")
    threading.Thread(target=record).start()

def extract_features(audio, sr=16000):
    """Ses sinyalinden özellik çıkarır."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    rms = librosa.feature.rms(y=audio)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features = np.concatenate((np.mean(mfccs, axis=1), np.mean(rms, axis=1), np.mean(chroma, axis=1)))
    return features

def segment_audio(file_path, segment_length=1.5):
    """Ses dosyasını belirtilen uzunlukta segmentlere ayırır."""
    audio = np.load(file_path).flatten()
    segment_samples = int(segment_length * default_sr)
    return [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples) if len(audio[i:i + segment_samples]) == segment_samples]

def normalize_features(features_df, label_column='label'):
    """Özellikleri normalize eder ve scaler'ı kaydeder."""
    scaler = MinMaxScaler()
    labels = features_df[label_column]
    features_df = features_df.drop(columns=[label_column])
    normalized_features = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(normalized_features, columns=features_df.columns)
    features_df[label_column] = labels
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    return features_df

# Flask Rotaları
@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    """Yeni kullanıcı kaydı için ses kaydı başlatır."""
    try:
        user_name = request.json['name']
        if not user_name:
            return jsonify({"error": "User name is required."}), 400

        file_path = os.path.join(DATA_DIR, f"{user_name}.npy")
        record_audio_async(file_path, duration=90)
        return jsonify({"message": "Recording started."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Modeli hem eski hem yeni ses kayıtlarıyla eğitir."""
    try:
        # Eski özellikleri yükle
        if os.path.exists(FEATURES_CSV):
            features_df = pd.read_csv(FEATURES_CSV)
        else:
            features_df = pd.DataFrame()

        # Yeni ses dosyalarını işle ve özellik çıkar
        for file in os.listdir(DATA_DIR):
            label = file.split('.')[0]
            file_path = os.path.join(DATA_DIR, file)
            segments = segment_audio(file_path, segment_length=segment_length_seconds)
            for segment in segments:
                features = extract_features(segment)
                row = pd.DataFrame([features.tolist() + [label]], columns=[f"feature_{i}" for i in range(len(features))] + ['label'])
                features_df = pd.concat([features_df, row], ignore_index=True)

        # Normalize Et ve Güncel Veri Setini Kaydet
        features_df = normalize_features(features_df, label_column='label')
        features_df.to_csv(FEATURES_CSV, index=False)
        print("Features saved to", FEATURES_CSV)

        # X ve y Ayrıştırma
        X = features_df.iloc[:, :-1].values
        y = features_df['label'].values

        # Veriyi Dengeleme
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        # Eğitim ve Test Bölümü
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Modeli Eğitme
        global classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        classifier.fit(X_train, y_train)

        # Doğruluk Hesaplama
        accuracy = classifier.score(X_test, y_test)
        joblib.dump(classifier, MODEL_PATH)
        print("Model saved to", MODEL_PATH)

        return jsonify({"message": "Model trained successfully.", "accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Anlık ses tahmini yapar."""
    try:
        if classifier is None:
            return jsonify({"error": "Model not trained."}), 400

        # Ses kaydı yap
        duration = 5
        file_path = "temp.npy"
        record_audio_async(file_path, duration=duration)
        threading.Event().wait(duration + 1)

        # Ses segmentasyonu ve tahmin
        segments = segment_audio(file_path, segment_length=segment_length_seconds)
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

from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

app = Flask(__name__, template_folder='../Frontend')

# Modeli yükleyin (eğitilmiş modelinizin yolunu belirtin)
model = joblib.load("../Model/random_forest_model.joblib")

# Scaler'ı yükleyin (MinMaxScaler veya StandardScaler)
scaler = joblib.load("../Model/scaler.joblib")

# Özellikleri normalleştirme fonksiyonu
def normalize_features(features):
    # Özellikleri yeniden şekillendir ve normalizasyonu uygula
    features_reshaped = np.reshape(features, (1, -1))  # [1, n] formatına getir
    normalized_features = scaler.transform(features_reshaped)
    
    # Normalize edilmiş özellikleri döndür
    return normalized_features[0]

# Özellik çıkarma fonksiyonu
def extract_features_from_audio(audio_data, sr=16000):
    try:
        # MFCC çıkarımı
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # MFCC ortalamasını al
        logging.debug(f"MFCC Mean: {mfcc_mean}")

        # Chroma çıkarımı
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        logging.debug(f"Chroma Mean: {chroma_mean}")

        # RMS Energy çıkarımı
        rms = librosa.feature.rms(y=audio_data)
        rms_mean = np.mean(rms)
        logging.debug(f"RMS Mean: {rms_mean}")

        # Zero-Crossing Rate çıkarımı
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr)
        logging.debug(f"ZCR Mean: {zcr_mean}")
        
        # Özellikleri birleştir
        features = np.hstack((mfcc_mean, chroma_mean, rms_mean, zcr_mean))
        logging.debug(f"Extracted Features: {features}")
        
        # Özellikleri normalleştir
        normalized_features = normalize_features(features)
        logging.debug(f"Normalized Features: {normalized_features}")
        
        return normalized_features
    
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return None

@app.route('/')
def index():
    # Ana sayfayı (HTML formunu) kullanıcıya göster
    return render_template('index.html')

logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    # Ses dosyasını alın
    audio_file = request.files['file']
    
    logging.debug(f"Dosya alındı: {audio_file.filename}")
    
    # Geçici bir dosyaya kaydedin
    temp_file_path = os.path.join('uploads', audio_file.filename)

    # Mevcut ses kaydını silin (önceki kayıttan kalma dosya varsa)
    for file in os.listdir('uploads'):
        file_path = os.path.join('uploads', file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logging.debug(f"Önceki dosya silindi: {file_path}")

    # Yeni dosyayı kaydedin
    audio_file.save(temp_file_path)
    
    logging.debug(f"Dosya kaydedildi: {temp_file_path}")

    # Ses dosyasını yükleyin
    try:
        audio_data, sr = librosa.load(temp_file_path, sr=None)
        logging.debug(f"Ses verisi yüklendi: {temp_file_path}")
        
        # Özellik çıkarımı
        features = extract_features_from_audio(audio_data)
        if features is None:
            return jsonify({"error": "Ses dosyası işlenemedi."}), 500

        logging.debug(f"Özellikler çıkarıldı: {features}")

        # Model ile tahmin yapma
        prediction = model.predict([features])
        logging.debug(f"Tahmin yapıldı: {prediction}")

        # Dosyayı silme
        os.remove(temp_file_path)
        logging.debug(f"Dosya silindi: {temp_file_path}")

        # Tahmin sonuçlarını döndürün
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        logging.error(f"Ses dosyası işlenemedi: {e}")
        return jsonify({"error": "Ses dosyası işlenemedi."}), 500

if __name__ == '__main__':
    app.run(debug=True)

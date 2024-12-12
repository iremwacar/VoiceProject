from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import joblib
import os
import logging

app = Flask(__name__, template_folder='../Frontend')

# Modeli yükleyin (eğitilmiş modelinizin yolunu belirtin)
model = joblib.load("../Model/random_forest_model.joblib")

# Özellik çıkarma fonksiyonu
def extract_features_from_audio(audio_data, sr=16000):
    try:
        # MFCC çıkarımı
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # MFCC ortalamasını al

        # Chroma çıkarımı
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # RMS Energy çıkarımı
        rms = librosa.feature.rms(y=audio_data)
        rms_mean = np.mean(rms)

        # Zero-Crossing Rate çıkarımı
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr)
        
        # Özellikleri birleştir
        features = np.hstack((mfcc_mean, chroma_mean, rms_mean, zcr_mean))
        return features
    
    except Exception as e:
        print(f"Error processing audio: {e}")
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
    audio_file.save(temp_file_path)
    
    logging.debug(f"Dosya kaydedildi: {temp_file_path}")

    # Özellikleri çıkarın
    y, sr = librosa.load(temp_file_path, sr=None)  # Ses dosyasını yükleyin
    features = extract_features_from_audio(y, sr)  # Doğru fonksiyonu çağırın
    if features is None:
        return jsonify({"error": "Ses dosyası işlenemedi."}), 500

    logging.debug(f"Özellikler çıkarıldı: {features}")

    # Özellikler ile tahmin yapın
    prediction = model.predict(np.array([features]))
    logging.debug(f"Tahmin yapıldı: {prediction}")

    # Tahmin sonuçlarını döndürün
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

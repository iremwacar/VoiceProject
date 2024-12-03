from fastapi import FastAPI, File, UploadFile
import joblib
import librosa
import numpy as np
import tempfile
import os
import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py'nin bulunduğu dizin
MODEL_DIR = os.path.join(BASE_DIR, '..', 'Model')      # Model klasörünün yolu
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'Frontend')

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))


app = FastAPI()

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


def extract_features_for_prediction(file_path):
    """Ses dosyasından özellikleri çıkar."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        features = np.hstack((mfcc_mean, chroma_mean, rms_mean, zcr_mean))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Geçici bir dosyada sesi kaydet
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(file.file.read())
            temp_path = temp_audio.name

        # Özellik çıkarımı
        features = extract_features_for_prediction(temp_path)
        if features is None:
            return {"error": "Özellik çıkarımı başarısız."}

        # Normalizasyon
        features = features.reshape(1, -1)
        normalized_features = scaler.transform(features)

        # Model tahmini
        prediction = model.predict(normalized_features)
        return {"prediction": prediction[0]}

    except Exception as e:
        return {"error": str(e)}

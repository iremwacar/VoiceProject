from flask import Flask, jsonify, render_template
import librosa
import numpy as np
import joblib
import sounddevice as sd
import queue
from flask_socketio import SocketIO, emit
import logging
import threading
import time
import pandas as pd
import noisereduce as nr  # Gürültü azaltma kütüphanesi
import wave
import io

app = Flask(__name__, template_folder='../Frontend')
socketio = SocketIO(app)

model = joblib.load("../Model/random_forest_model1.joblib")

# Mikrofon kaydını dinlemek için
last_prediction_time = 0  # Son tahminin yapıldığı zaman
prediction_interval = 1.5  # Her 1 saniyede bir tahmin yap

# Mikrofon kaydını başlatan fonksiyon
def audio_callback(indata, frames, time_info, status):
    global last_prediction_time
    
    if status:
        print(status)
    
    audio_data = indata[:, 0]  # Sadece tek kanal alıyoruz (mono)
    
    # Gürültü azaltma işlemi
    audio_data = nr.reduce_noise(y=audio_data, sr=16000)  # Gürültüyü azalt
    
    # WAV formatında işlemek
    wav_data = audio_data.astype(np.float32)  # Veriyi uygun formatta ayarlıyoruz
    wav_io = io.BytesIO()
    
    # 'wav' dosyasına kaydet
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono kanal
        wav_file.setsampwidth(2)  # 16 bit (2 byte) örnekleme genişliği
        wav_file.setframerate(16000)  # Örnekleme hızı
        wav_file.writeframes(wav_data.tobytes())  # Ses verisini kaydet
    
    wav_io.seek(0)  # Dosyayı başa sar
    
    # Librosa ile WAV verisini işle
    audio_data, sr = librosa.load(wav_io, sr=16000)  # 16 kHz örnekleme hızı ile yükle
    
    features = extract_features_from_audio(audio_data, sr)
    
    if features is not None:
        current_time = time.time()
        
        if current_time - last_prediction_time >= prediction_interval:
            # Tahmin yap
            prediction = model.predict([features])[0]
            socketio.emit('speaker_update', {'speaker': prediction})
            last_prediction_time = current_time  # Son tahmin zamanını güncelle

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
        
        return features
    
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return None

# Mikrofon kaydını başlatma
recording_thread = None
is_recording = False

@socketio.on('start_recording')
def start_recording():
    global recording_thread, is_recording
    if not is_recording:
        # Ses verisini sürekli olarak dinlemek için mikrofonu başlatıyoruz
        print("Recording started...")
        is_recording = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    else:
        print("Recording already in progress")

@socketio.on('stop_recording')
def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped")
    socketio.emit('speaker_update', {'speaker': 'None'})

def record_audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while is_recording:
            time.sleep(0.5)  # 1 saniye bekle, sonra ses verisini al ve tahmin yap

@app.route('/')
def index():
    return render_template('index2.html')  # Ana sayfa

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Debug logları görüntüle
    socketio.run(app, debug=True)  # Flask-SocketIO sunucusunu başlat

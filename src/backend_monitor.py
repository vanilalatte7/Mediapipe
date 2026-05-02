import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import keras
import time
from datetime import datetime

# ==========================================
# KONFIGURASI FIREBASE (Wajib diisi)
# ==========================================
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    print("ERROR: Modul firebase-admin belum diinstal.")
    print("Jalankan: pip install firebase-admin")
    exit()

# Ganti dengan nama file credential JSON Anda yang diunduh dari Firebase Console
FIREBASE_CREDENTIALS_FILE = '../config/firebase_credentials.json'


if not os.path.exists(FIREBASE_CREDENTIALS_FILE):
    print(f"===============================================================")
    print(f"⚠️ PERINGATAN FIREBASE ⚠️")
    print(f"File kredensial '{FIREBASE_CREDENTIALS_FILE}' tidak ditemukan!")
    print(f"Harap unduh Service Account Key dari Firebase Console Anda")
    print(f"dan simpan di folder ini dengan nama '{FIREBASE_CREDENTIALS_FILE}'.")
    print(f"===============================================================")
    print(f"Script berjalan dalam mode 'OFFLINE' (Hanya deteksi lokal).")
    print(f"===============================================================\n")
    db = None
else:
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_FILE)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Terhubung ke Firebase Firestore!")
    except Exception as e:
        print(f"Gagal terhubung ke Firebase: {e}")
        db = None


# ==========================================
# INISIALISASI MODEL AI
# ==========================================
def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)

MODEL_PATH = '../archive/asl_pro_final_efficient_new.h5'
LABEL_PATH = '../models/labels.npy'

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):

    print("ERROR: File model atau label tidak ditemukan!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = np.load(LABEL_PATH)
print(f"✅ Model AI Siap! ({len(labels)} kelas)")

# ==========================================
# MANAJEMEN STATE BACKEND & FIREBASE
# ==========================================
# ID Pasien/Caregiver simulasi (Sesuaikan dengan struktur database Anda)
USER_ID = "patient_001" 

# Konfigurasi Default (Jika Firebase offline/belum diset)
# Format: { "Gestur": "Kebutuhan" }
gesture_mapping = {
    "A": "Makan",
    "B": "Minum",
    "C": "Toilet",
    "D": "Tidur"
}

# Sistem Cooldown agar tidak spam database
# Mencegah pengiriman sinyal yang sama berturut-turut dalam waktu 60 detik
COOLDOWN_SECONDS = 60
last_signal_time = {}

def get_firebase_config():
    """Mengambil konfigurasi gestur terbaru dari Firebase (Katalog)"""
    global gesture_mapping
    if db:
        try:
            # Contoh path: users/{USER_ID}/config/gestures
            doc_ref = db.collection('users').document(USER_ID).collection('config').document('gestures')
            doc = doc_ref.get()
            if doc.exists:
                # Membalikkan mapping dari DB (Misal DB: Makan -> ['A', 'B']) menjadi (A -> Makan, B -> Makan)
                db_config = doc.to_dict()
                new_mapping = {}
                for kebutuhan, daftar_gestur in db_config.items():
                    for gestur in daftar_gestur:
                        new_mapping[gestur] = kebutuhan
                gesture_mapping = new_mapping
                print(f"🔄 Konfigurasi diperbarui dari Firebase: {gesture_mapping}")
        except Exception as e:
            print(f"Gagal mengambil konfigurasi: {e}")

def push_to_firebase(kebutuhan, gestur):
    """Mengirim sinyal ke Riwayat Aktivitas di Firebase"""
    current_time = time.time()
    
    # Cek Cooldown
    if kebutuhan in last_signal_time:
        if current_time - last_signal_time[kebutuhan] < COOLDOWN_SECONDS:
            print(f"⏳ Cooldown: Sinyal '{kebutuhan}' sudah dikirim baru-baru ini. Menunggu...")
            return

    print(f"🚀 MENGIRIM SINYAL: Pasien membutuhkan '{kebutuhan}' (Gestur: {gestur})")
    
    if db:
        try:
            doc_ref = db.collection('users').document(USER_ID).collection('history').document()
            doc_ref.set({
                'kategori': kebutuhan,
                'gestur_pemicu': gestur,
                'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'status': 'terkirim' # Bisa diubah frontend jadi 'terbaca'
            })
            print(f"✅ Berhasil disimpan ke Firebase Dashboard!")
            last_signal_time[kebutuhan] = current_time
        except Exception as e:
            print(f"❌ Gagal mengirim ke Firebase: {e}")
    else:
        # Simulasi jika Offline
        last_signal_time[kebutuhan] = current_time


# ==========================================
# LOOP DETEKSI KAMERA (ULTRA STABLE)
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

buffer = deque(maxlen=30)
pred_history = deque(maxlen=12) # Responsif tapi stabil
cap = cv2.VideoCapture(0)

# Ambil konfigurasi awal
get_firebase_config()

print("===============================================================")
print("🤖 BACKEND STROKE MONITOR AKTIF")
print("Tekan Q untuk Keluar. Tekan C untuk Sync Konfigurasi Firebase.")
print("===============================================================")

label = "Waiting..."
conf = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    current_label = "Uncertain..."
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            wrist = pts[0]
            centered = pts - wrist
            
            hand_size = np.linalg.norm(centered[0] - centered[9])
            if hand_size == 0: hand_size = 1.0
            final_norm = (centered / hand_size).flatten()
            
            buffer.append(final_norm)
            
            if len(buffer) == 30:
                pred = model.predict(np.expand_dims(list(buffer), axis=0), verbose=0)
                idx = np.argmax(pred[0])
                conf = pred[0][idx]
                
                if conf > 0.70:
                    if conf > 0.90:
                        pred_history.append(labels[idx])
                    pred_history.append(labels[idx])
                
                if len(pred_history) > 0:
                    label = max(set(pred_history), key=list(pred_history).count)
                    
                    # LOGIKA BACKEND: Cek apakah gestur ini memicu kebutuhan di katalog
                    if label in gesture_mapping:
                        kebutuhan_aktif = gesture_mapping[label]
                        # Trigger pengiriman ke Firebase
                        push_to_firebase(kebutuhan_aktif, label)
                else:
                    label = "Analyzing..."

    # Visualisasi UI Kamera (Untuk debugging lokal)
    status_text = f"Gestur: {label} "
    if label in gesture_mapping:
        status_text += f"-> Sinyal: {gesture_mapping[label]}"
        
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Backend Monitoring AI', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Sinkronisasi manual dari Firebase
        get_firebase_config()

cap.release()
cv2.destroyAllWindows()

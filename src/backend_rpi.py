import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime
import threading
import queue

# ==========================================
# KONFIGURASI STANDAR INDUSTRI IOT
# ==========================================
HEADLESS_MODE = True # Ubah ke False jika ingin melihat video di layar Raspi
COOLDOWN_SECONDS = 60
USER_ID = "patient_001"
MODEL_PATH = '../models/model.tflite'
LABEL_PATH = '../models/labels.npy'
FIREBASE_CRED_PATH = '../config/firebase_credentials.json'


# ==========================================
# 1. FIREBASE WORKER THREAD
# ==========================================
firebase_queue = queue.Queue()
gesture_mapping = {"A": "Makan", "B": "Minum", "C": "Toilet", "D": "Tidur"}
last_signal_time = {}

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    # Setup Firebase
    if os.path.exists(FIREBASE_CRED_PATH):
        cred = credentials.Certificate(FIREBASE_CRED_PATH)

        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase Terhubung!")
    else:
        print("⚠️ Berjalan OFFLINE (credentials tidak ditemukan)")
        db = None
except ImportError:
    print("⚠️ firebase-admin belum diinstal. Berjalan OFFLINE.")
    db = None

def firebase_worker():
    """Background thread khusus untuk mengelola pengiriman data agar tidak membuat lag kamera"""
    global gesture_mapping
    while True:
        task = firebase_queue.get()
        if task is None: break # Sinyal berhenti
        
        action = task.get('action')
        
        if action == 'sync_config' and db:
            try:
                doc = db.collection('users').document(USER_ID).collection('config').document('gestures').get()
                if doc.exists:
                    new_mapping = {}
                    for kebutuhan, daftar_gestur in doc.to_dict().items():
                        for gestur in daftar_gestur: new_mapping[gestur] = kebutuhan
                    gesture_mapping = new_mapping
                    print(f"🔄 Sync Firebase: {gesture_mapping}")
            except Exception as e:
                print(f"Gagal sync config: {e}")
                
        elif action == 'push_history':
            kebutuhan = task.get('kebutuhan')
            gestur = task.get('gestur')
            current_time = time.time()
            
            # Cek Cooldown
            if kebutuhan not in last_signal_time or (current_time - last_signal_time[kebutuhan] > COOLDOWN_SECONDS):
                print(f"🚀 MENGIRIM SINYAL: '{kebutuhan}' (Gestur: {gestur})")
                if db:
                    try:
                        db.collection('users').document(USER_ID).collection('history').document().set({
                            'kategori': kebutuhan, 'gestur_pemicu': gestur,
                            'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'timestamp': firestore.SERVER_TIMESTAMP, 'status': 'terkirim'
                        })
                        print("✅ Tersimpan di DB!")
                        last_signal_time[kebutuhan] = current_time
                    except Exception as e: print(f"❌ Gagal mengirim: {e}")
                else:
                    last_signal_time[kebutuhan] = current_time
        firebase_queue.task_done()

# Mulai worker Firebase
threading.Thread(target=firebase_worker, daemon=True).start()
firebase_queue.put({'action': 'sync_config'})

# ==========================================
# 2. INISIALISASI TFLITE INTERPRETER (Sangat Ringan)
# ==========================================
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("ERROR: model.tflite atau labels.npy tidak ditemukan!")
    exit(1)

labels = np.load(LABEL_PATH)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

# Mengatur ops khusus jika diperlukan (terutama jika ada SELECT_TF_OPS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"✅ TFLite Interpreter Siap! Input shape: {input_details[0]['shape']}")

def predict_tflite(input_data):
    """Menjalankan inferensi sangat cepat menggunakan TFLite"""
    input_data = np.array(input_data, dtype=np.float32)
    # TFLite memerlukan shape [1, 30, 63] (Contoh)
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ==========================================
# 3. THREADED CAMERA STREAM
# ==========================================
class VideoStream:
    """Membaca kamera di background thread agar tidak menghalangi AI"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Downscale agar ringan
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# ==========================================
# 4. MAIN LOOP (CORE AI)
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

buffer = deque(maxlen=30)
pred_history = deque(maxlen=12)

print("===============================================================")
print("🤖 BACKEND RASPBERRY PI IOT AKTIF")
print(f"Mode Headless: {'ON (Lebih Cepat)' if HEADLESS_MODE else 'OFF (Tampil Jendela)'}")
print("Tekan Ctrl+C untuk berhenti.")
print("===============================================================")

vs = VideoStream(0).start()
time.sleep(2.0) # Pemanasan kamera

try:
    while True:
        frame = vs.read()
        if frame is None: continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        label = "Waiting..."
        conf = 0.0
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if not HEADLESS_MODE:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
                wrist = pts[0]
                centered = pts - wrist
                hand_size = np.linalg.norm(centered[0] - centered[9])
                if hand_size == 0: hand_size = 1.0
                final_norm = (centered / hand_size).flatten()
                
                buffer.append(final_norm)
                
                if len(buffer) == 30:
                    pred = predict_tflite(list(buffer))
                    idx = np.argmax(pred)
                    conf = pred[idx]
                    
                    if conf > 0.70:
                        if conf > 0.90: pred_history.append(labels[idx])
                        pred_history.append(labels[idx])
                    
                    if len(pred_history) > 0:
                        label = max(set(pred_history), key=list(pred_history).count)
                        if label in gesture_mapping:
                            firebase_queue.put({
                                'action': 'push_history',
                                'kebutuhan': gesture_mapping[label],
                                'gestur': label
                            })
                    else:
                        label = "Analyzing..."

        # Tampilkan GUI jika Headless = False
        if not HEADLESS_MODE:
            status_text = f"{label} ({conf:.2f})"
            cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('IoT Edge Node', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Di mode headless, hanya print jika ada perubahan atau sinyal kuat
            if conf > 0.85:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] AI Mendeteksi: {label} ({conf:.2f})")

except KeyboardInterrupt:
    print("\nMenghentikan sistem IoT...")
finally:
    vs.stop()
    cv2.destroyAllWindows()

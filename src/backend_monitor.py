import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import keras
from collections import deque
import os
import time
from datetime import datetime
import threading
import queue
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn

# ==========================================
# KONFIGURASI FASTAPI & KREDENSIAL
# ==========================================
COOLDOWN_SECONDS = 60
USER_ID = "patient_001"
MODEL_PATH = '../archive/asl_pro_final_efficient_new.h5'
LABEL_PATH = '../models/labels.npy'
FIREBASE_CRED_PATH = '../config/firebase_credentials.json'

global_status = {
    "current_label": "Waiting...",
    "confidence": 0.0,
    "last_signal_sent": None
}

app = FastAPI(title="Stroke Monitor AI Server (PC Version)")

# ==========================================
# 1. FIREBASE WORKER THREAD
# ==========================================
firebase_queue = queue.Queue()
gesture_mapping = {"A": "Makan", "B": "Minum", "C": "Toilet", "D": "Tidur"}
last_signal_time = {}

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
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
    global gesture_mapping
    while True:
        task = firebase_queue.get()
        if task is None: break 
        
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
            
            if kebutuhan not in last_signal_time or (current_time - last_signal_time[kebutuhan] > COOLDOWN_SECONDS):
                print(f"🚀 MENGIRIM SINYAL: '{kebutuhan}' (Gestur: {gestur})")
                global_status["last_signal_sent"] = f"{kebutuhan} at {datetime.now().strftime('%H:%M:%S')}"
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

threading.Thread(target=firebase_worker, daemon=True).start()
firebase_queue.put({'action': 'sync_config'})

# ==========================================
# 2. INISIALISASI KERAS MODEL & MEDIAPIPE
# ==========================================
def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("ERROR: File model (.h5) atau label tidak ditemukan!")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = np.load(LABEL_PATH)
print(f"✅ Model Keras Siap! ({len(labels)} kelas)")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

buffer = deque(maxlen=30)
pred_history = deque(maxlen=12)

# ==========================================
# 3. THREADED CAMERA STREAM
# ==========================================
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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

vs = VideoStream(0).start()
time.sleep(2.0)

# ==========================================
# 4. CORE AI GENERATOR (Untuk Streaming)
# ==========================================
def generate_frames():
    global global_status
    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.1)
            continue
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        label = "Waiting..."
        conf = 0.0
        
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
                    conf = float(pred[0][idx])
                    
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

        # Update global status
        global_status["current_label"] = str(label)
        global_status["confidence"] = conf

        # Gambar teks di frame untuk stream video
        status_text = f"{label} ({conf:.2f})"
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Konversi ke format JPEG untuk web stream
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# ==========================================
# 5. FASTAPI ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "AI PC Backend is Running"}

@app.get("/status")
def get_status():
    """Mengembalikan status gestur terkini (JSON)"""
    return global_status

@app.post("/sync")
def sync_firebase():
    """Memaksa sinkronisasi katalog konfigurasi dari Firebase"""
    firebase_queue.put({'action': 'sync_config'})
    return {"status": "Sync request sent to Firebase worker"}

@app.get("/video_feed")
def video_feed():
    """Endpoint untuk live streaming video ke browser / frontend"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Mulai server FastAPI. Akses http://localhost:8000/video_feed untuk melihat kamera.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

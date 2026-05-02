import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import keras

# Patch untuk kompatibilitas model Keras 3
def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)

# 1. LOAD MODEL & LABELS
MODEL_PATH = '../archive/asl_pro_final.h5'
LABEL_PATH = '../models/labels.npy'


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File model '{MODEL_PATH}' tidak ditemukan!")
    exit()

if not os.path.exists(LABEL_PATH):
    print(f"ERROR: File label '{LABEL_PATH}' tidak ditemukan!")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = np.load(LABEL_PATH)
    print(f"Model Ready: {len(labels)} classes")
except Exception as e:
    print(f"ERROR saat memuat model/label: {e}")
    exit()

# 2. MEDIAPIPE HANDS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3. BUFFER & STABILITY
buffer = deque(maxlen=30)
pred_history = deque(maxlen=25) # Diperbesar menjadi 25 agar sangat stabil
cap = cv2.VideoCapture(0)

label = "Waiting..."
conf = 0

print("Starting Test... Press Q to Exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Reset state per frame jika perlu, tapi label dipertahankan oleh voting
    current_label = "Uncertain..."
    current_conf = 0
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Preprocessing: Centering & Scaling (Wajib sama dengan Training)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            
            # 1. Wrist Centering
            wrist = pts[0]
            centered = pts - wrist
            
            # 2. Hand Size Scaling (Normalisasi Jarak)
            # Menggunakan jarak pergelangan (0) ke pangkal jari tengah (9)
            hand_size = np.linalg.norm(centered[0] - centered[9])
            if hand_size == 0: hand_size = 1.0
            final_norm = (centered / hand_size).flatten()
            
            buffer.append(final_norm)
            
            if len(buffer) == 30:
                pred = model.predict(np.expand_dims(list(buffer), axis=0), verbose=0)
                idx = np.argmax(pred[0])
                conf = pred[0][idx]
                
                # Update riwayat prediksi jika confidence cukup tinggi
                if conf > 0.75:
                    pred_history.append(labels[idx])
                    if conf > 0.95:
                        pred_history.append(labels[idx])
                elif conf < 0.40:
                    pred_history.clear()

                
                # Voting Mechanism: Ambil huruf yang paling konsisten muncul
                if len(pred_history) > 0:
                    label = max(set(pred_history), key=list(pred_history).count)
                else:
                    label = "Analyzing..."

    # Simple Text (Menampilkan Label Hasil Voting)
    cv2.putText(frame, f"{label} {conf:.2%}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Test Mode', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
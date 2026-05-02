import tensorflow as tf
import keras
import os

print("Memulai proses konversi model ke TFLite...")

# 1. Terapkan patch kompatibilitas Keras 3 untuk model lama
def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)

model_path = '../archive/asl_pro_final.h5'
tflite_path = '../models/model.tflite'


if not os.path.exists(model_path):
    print(f"ERROR: Model asal '{model_path}' tidak ditemukan.")
    exit(1)

try:
    # 2. Muat model Keras
    print(f"Memuat model {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model berhasil dimuat!")

    # 3. Konversi via SavedModel (Lebih stabil untuk Bidirectional LSTM)
    print("Menyimpan sementara sebagai SavedModel...")
    saved_model_dir = "temp_saved_model"
    model.export(saved_model_dir)
    
    print("Mengonversi arsitektur ke TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Opsi optimasi dan dukungan operator kompleks (LSTM)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS    
    ]
    
    # Eksekusi konversi
    tflite_model = converter.convert()



    # 4. Simpan ke disk
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\n✅ SUKSES! Model berhasil dikonversi dan disimpan sebagai '{tflite_path}'")
    print(f"Ukuran model TFLite: {os.path.getsize(tflite_path) / 1024:.2f} KB")

except Exception as e:
    print(f"\n❌ TERJADI KESALAHAN SAAT KONVERSI: {e}")

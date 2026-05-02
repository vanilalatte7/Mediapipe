import keras
import numpy as np

def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)

m = 'asl_pro_final_efficient_new.h5'
try:
    model = keras.models.load_model(m, compile=False)
    print(f'{m} Output shape: {model.output_shape}')
except Exception as e:
    print(f'Error loading {m}: {e}')

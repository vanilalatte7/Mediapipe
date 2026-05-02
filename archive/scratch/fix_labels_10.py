import numpy as np

# 10 Contrastive labels that match the new model (output shape 10)
labels = ['A', 'B', 'C', 'D', 'F', 'I', 'L', 'V', 'W', 'Y']

np.save('labels.npy', np.array(labels))
print("labels.npy updated with 10 contrastive classes.")

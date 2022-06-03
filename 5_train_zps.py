import sys
import json
import os
import tensorflow as tf
import h5py
import numpy as np

from src.Config import Config
from src.Model import Model

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

dataset_file = h5py.File(dataset_file,'r')
X_train = dataset_file['X_train'][:,:,8:-8,8:-8]
X_test = dataset_file['X_test'][:,:,8:-8,8:-8]
Z_train = dataset_file['Z_train'][:]
Z_test = dataset_file['Z_test'][:]
dataset_file.close()

print(X_train.shape, X_test.shape, Z_train.shape, Z_test.shape)
print(Z_test)

model = Model(config).zps_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'zurich-pituitary-score'), save_best_only=True)

print(model.summary())


history = model.fit(x=X_train,y=Z_train,validation_data=(X_test,Z_test),epochs=config.EPOCHS_CLASSIFIERS,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'zurich-pituitary-score.json'), 'w') as outfile:
    json.dump(history.history, outfile)
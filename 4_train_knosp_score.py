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
K_train = dataset_file['K_train'][:]
K_test = dataset_file['K_test'][:]
dataset_file.close()

K_train = np.max(K_train, axis=-1)
K_test = np.max(K_test, axis=-1)

print(X_train.shape, X_test.shape, K_train.shape, K_test.shape)

model = Model(config).knosp_score_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'knosp-score'), save_best_only=True)

print(model.summary())


history = model.fit(x=X_train,y=K_train,validation_data=(X_test,K_test),epochs=config.EPOCHS_CLASSIFIERS,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-knosp-score.json'), 'w') as outfile:
    json.dump(history.history, outfile)
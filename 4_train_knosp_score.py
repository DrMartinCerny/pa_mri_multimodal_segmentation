import sys
import json
import os
import tensorflow as tf
import h5py
import numpy as np

from src.Config import Config
from src.Model import Model

config_file = sys.argv[1]
dataset_file = sys.argv[2]
model_folder = sys.argv[3]
if not os.path.exists(model_folder): os.mkdir(model_folder)

config = Config(config_file)

dataset_file = h5py.File(dataset_file,'r')
X_train = dataset_file['X_train'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
X_val = dataset_file['X_val'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
K_train = dataset_file['K_train'][:]
K_val = dataset_file['K_val'][:]
dataset_file.close()

# Augmenting the dataset with flipped versions of the images
X_train = np.concatenate([X_train,np.flip(X_train, axis=3)])
X_val = np.concatenate([X_val,np.flip(X_val, axis=3)])
K_train = np.concatenate([K_train,np.flip(K_train, axis=1)])
K_val = np.concatenate([K_val,np.flip(K_val, axis=1)])

model = Model(config).knosp_score_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'knosp-score.h5'), save_best_only=True, save_weights_only=True)

print(model.summary())

history = model.fit(x=X_train,y=K_train,validation_data=(X_val,K_val),epochs=config.EPOCHS_CLASSIFIERS,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-knosp-score.json'), 'w') as outfile:
    json.dump(history.history, outfile)
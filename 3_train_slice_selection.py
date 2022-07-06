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
X_test = dataset_file['X_test'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
N_train = dataset_file['N_train'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
N_test = dataset_file['N_test'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS]
dataset_file.close()

print(X_train.shape, X_test.shape, N_train.shape, N_test.shape)

y_train = np.concatenate([np.ones(len(X_train)),np.zeros(len(N_train))])
y_test = np.concatenate([np.ones(len(X_test)),np.zeros(len(N_test))])

X_train = np.concatenate([X_train,N_train])
del N_train
X_test = np.concatenate([X_test,N_test])
del N_test

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Model(config).slice_selection_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'slice-selection.h5'), save_best_only=True, save_weights_only=True)

print(model.summary())


history = model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=config.EPOCHS_CLASSIFIERS,callbacks=[model_checkpoint_callback])

with open(os.path.join(model_folder, 'train-history-slice-selection.json'), 'w') as outfile:
    json.dump(history.history, outfile)
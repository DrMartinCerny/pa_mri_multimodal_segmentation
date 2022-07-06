import sys
import numpy as np
import h5py
import cv2
from tensorflow import keras
import os

from src.Config import Config
from src.Visualization import Visualization
from src.Model import Model

config = Config(sys.argv[1])
model = Model(config)

dataset_file = sys.argv[2]
model_folder = sys.argv[3]

dataset_file = h5py.File(dataset_file,'r')

X_test = dataset_file['X_test'][:,:,int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),int(config.CROP_OFFSET/2):-int(config.CROP_OFFSET/2),:config.NUM_CHANNELS].astype(np.float64)
knospScoreGroundTruth = dataset_file['K_test'][:]
maskGroundTruth = dataset_file['y_test'][:,int(config.CROP_OFFSET/2)+1:-int(config.CROP_OFFSET/2)-1,int(config.CROP_OFFSET/2)+1:-int(config.CROP_OFFSET/2)-1]
dataset_file.close()

model = Model(config).segmentation_model(compile=False)
model.load_weights(os.path.join(model_folder, 'segmentation.h5'))
maskPredicted = np.argmax(model.predict(X_test),axis=-1)
keras.backend.clear_session()

model = Model(config).knosp_score_model(compile=False)
model.load_weights(os.path.join(model_folder, 'knosp-score.h5'))
knospScorePredicted = np.argmax(model.predict(X_test),axis=-1)
keras.backend.clear_session()

for sample in np.random.randint(0, len(X_test), size=2):
    img = Visualization.toBitmap(X_test[sample,1,1:-1,1:-1,0])
    Visualization.overlay(img,maskPredicted[sample])
    img = Visualization.upsample(img)
    Visualization.contours(img,maskGroundTruth[sample])
    Visualization.addKnospScore(img,knospScoreGroundTruth[sample],knospScorePredicted[sample])
    cv2.imshow('Segmentation visualization', img)
    cv2.waitKey(0)
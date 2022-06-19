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

model = keras.models.load_model(os.path.join(model_folder, 'segmentation'), compile=False, custom_objects={'dice_coef_total': model.dice_coef_total, 'dice_coef_tumor': model.dice_coef_tumor, 'dice_coef_ica': model.dice_coef_ica, 'dice_coef_normal_gland': model.dice_coef_normal_gland})
maskPredicted = np.argmax(model.predict(X_test),axis=-1)

model = keras.models.load_model(os.path.join(model_folder, 'knosp-score'))
knospScorePredicted = np.argmax(model.predict(X_test),axis=-1)

for sample in np.random.randint(0, len(X_test), size=2):
    img = Visualization.toBitmap(X_test[sample,1,1:-1,1:-1,0])
    Visualization.overlay(img,maskPredicted[sample])
    img = Visualization.upsample(img)
    Visualization.contours(img,maskGroundTruth[sample])
    Visualization.addKnospScore(img,knospScoreGroundTruth[sample],knospScorePredicted[sample])
    cv2.imshow('Segmentation visualization', img)
    cv2.waitKey(0)
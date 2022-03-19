import sys
import numpy as np
import h5py
import cv2
from tensorflow import keras
import os

from src.Config import Config
from src.Visualization import Visualization
from src.KnospScore import KnospScore
from Model import Model

config = Config()
model = Model(config)

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

dataset_file = h5py.File(dataset_file,'r')

X_test = dataset_file['X_test'][:,:,8:-8,8:-8]
N_test = dataset_file['N_test'][:,:,8:-8,8:-8]
K_test = dataset_file['K_test'][:]
y_test = dataset_file['y_test'][:,9:-9,9:-9]
dataset_file.close()

model = keras.models.load_model(os.path.join(model_folder, 'segmentation'), compile=False,custom_objects={'dice_coef_total': model.dice_coef_total, 'dice_coef_tumor': model.dice_coef_tumor, 'dice_coef_ica': model.dice_coef_ica, 'dice_coef_normal_gland': model.dice_coef_normal_gland})
predicted = model.predict(X_test)
predicted = np.argmax(predicted,axis=-1)

print(X_test.shape)
print(N_test.shape)
print(y_test.shape)
print(predicted.shape)

for sample in np.random.randint(0, len(X_test), size=40):
    cor_t1_c = Visualization.toBitmap(X_test[sample,1,1:-1,1:-1,0])
    cor_t1 = Visualization.toBitmap(X_test[sample,1,1:-1,1:-1,1])
    cor_t2 = Visualization.toBitmap(X_test[sample,1,1:-1,1:-1,2])
    maskGroundTruth = y_test[sample]
    knospScoreGroundTruth = KnospScore(None,K_test[sample])
    knospScoreOriginalGeo = KnospScore(maskGroundTruth)
    maskPrediction = predicted[sample]
    knospScorePredictionGeo = KnospScore(maskPrediction)
    img = np.concatenate([
        # RAW
        np.concatenate([
            Visualization.upsample(cor_t1_c),
            Visualization.upsample(cor_t1),
            Visualization.upsample(cor_t2),
        ],axis=1),
        # GROUND TRUTH
        np.concatenate([
            Visualization.drawKnospLines(Visualization.overlay(cor_t1_c,maskGroundTruth),knospScoreGroundTruth,knospScoreOriginalGeo),
            Visualization.drawKnospLines(Visualization.overlay(cor_t1,maskGroundTruth),knospScoreGroundTruth,knospScoreOriginalGeo),
            Visualization.drawKnospLines(Visualization.overlay(cor_t2,maskGroundTruth),knospScoreGroundTruth,knospScoreOriginalGeo),
        ],axis=1),
        # PREDICTION
        np.concatenate([
            Visualization.drawKnospLines(Visualization.overlay(cor_t1_c,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo),
            Visualization.drawKnospLines(Visualization.overlay(cor_t1,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo),
            Visualization.drawKnospLines(Visualization.overlay(cor_t2,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo),
        ],axis=1),
    ])
    cv2.imshow('Segmentation visualization', img)
    cv2.waitKey(0)
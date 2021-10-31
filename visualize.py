import sys
import numpy as np
import h5py
import cv2
from tensorflow import keras

from src.Config import Config
from src.Visualization import Visualization
from src.KnospScore import KnospScore

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

dataset_file = h5py.File(dataset_file,'r')

X_test = dataset_file['X_test'][:]
N_test = dataset_file['N_test'][:]
y_test = dataset_file['y_test'][:]

model = keras.models.load_model(model_folder)
predicted = np.argmax(model.predict(X_test)[0],axis=-1)

print(['{:1.2f}'.format(x[0]) for x in model.predict(X_test)[1]])
print(['{:1.2f}'.format(x[0]) for x in model.predict(N_test)[1]])

print(X_test.shape, y_test.shape, predicted.shape)

for sample in np.random.randint(0, len(X_test), size=5):
    try:
        cor_t1_c = Visualization.toBitmap(X_test[sample,:,:,0])
        cor_t1 = Visualization.toBitmap(X_test[sample,:,:,1])
        ax_t2 = Visualization.toBitmap(X_test[sample,:,:,2])
        dwi = Visualization.toBitmap(X_test[sample,:,:,3])
        maskGroundTruth = y_test[sample]
        knospScoreGroundTruth = KnospScore(maskGroundTruth)
        maskPrediction = predicted[sample]
        knospScorePrediction = KnospScore(maskGroundTruth)
        img = np.concatenate([
            # RAW
            np.concatenate([
                Visualization.upsample(cor_t1_c),
                Visualization.upsample(cor_t1),
                Visualization.upsample(ax_t2),
                Visualization.upsample(dwi),
            ],axis=1),
            # GROUND TRUTH
            np.concatenate([
                Visualization.drawKnospLines(Visualization.overlay(cor_t1_c,maskGroundTruth),knospScoreGroundTruth),
                Visualization.drawKnospLines(Visualization.overlay(cor_t1,maskGroundTruth),knospScoreGroundTruth),
                Visualization.drawKnospLines(Visualization.overlay(ax_t2,maskGroundTruth),knospScoreGroundTruth),
                Visualization.drawKnospLines(Visualization.overlay(dwi,maskGroundTruth),knospScoreGroundTruth),
            ],axis=1),
            # PREDICTION
            np.concatenate([
                Visualization.drawKnospLines(Visualization.overlay(cor_t1_c,maskPrediction),knospScorePrediction),
                Visualization.drawKnospLines(Visualization.overlay(cor_t1,maskPrediction),knospScorePrediction),
                Visualization.drawKnospLines(Visualization.overlay(ax_t2,maskPrediction),knospScorePrediction),
                Visualization.drawKnospLines(Visualization.overlay(dwi,maskPrediction),knospScorePrediction),
            ],axis=1),
        ])
        cv2.imshow('Segmentation visualization', img)
        cv2.waitKey(0)
    except: pass
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

X_test = dataset_file['X_test'][:,8:-8,8:-8]
N_test = dataset_file['N_test'][:,8:-8,8:-8]
K_test = dataset_file['K_test']
y_test = dataset_file['y_test'][:,8:-8,8:-8]

model = keras.models.load_model(model_folder)
predicted = model.predict(X_test)
predicted_knosp_scores = np.argmax(predicted[2],axis=-1)
predicted = np.argmax(predicted[0],axis=-1)

print(['{:1.2f}'.format(x[0]) for x in model.predict(X_test)[1]])
print(['{:1.2f}'.format(x[0]) for x in model.predict(N_test)[1]])

print(X_test.shape, N_test.shape, y_test.shape, predicted.shape, K_test.shape, predicted_knosp_scores.shape)

for sample in np.random.randint(0, len(X_test), size=5):
    cor_t1_c = Visualization.toBitmap(X_test[sample,:,:,0])
    cor_t1 = Visualization.toBitmap(X_test[sample,:,:,1])
    cor_t2 = Visualization.toBitmap(X_test[sample,:,:,2])
    maskGroundTruth = y_test[sample]
    knospScoreGroundTruth = KnospScore(None,K_test[sample])
    knospScoreOriginalGeo = KnospScore(maskGroundTruth)
    maskPrediction = predicted[sample]
    knospScorePredictionGeo = KnospScore(maskPrediction)
    knospScorePredictionBlackbox = KnospScore(None,predicted_knosp_scores[sample])
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
            Visualization.drawKnospLines(Visualization.overlay(cor_t1_c,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo,knospScorePredictionBlackbox),
            Visualization.drawKnospLines(Visualization.overlay(cor_t1,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo,knospScorePredictionBlackbox),
            Visualization.drawKnospLines(Visualization.overlay(cor_t2,maskPrediction),knospScoreGroundTruth,knospScorePredictionGeo,knospScorePredictionBlackbox),
        ],axis=1),
    ])
    cv2.imshow('Segmentation visualization', img)
    cv2.waitKey(0)
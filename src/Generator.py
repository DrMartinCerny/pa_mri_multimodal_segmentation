import tensorflow as tf
import numpy as np
import h5py

class Generator(tf.keras.utils.Sequence):

    def __init__(self, dataset_file, isTrain, batchSize):
        dataset_file = h5py.File(dataset_file,'r')
        self.X = dataset_file['X_train' if isTrain else 'X_test'][:]
        self.y = dataset_file['y_train' if isTrain else 'y_test'][:]
        self.N = dataset_file['N_train' if isTrain else 'N_test'][:]
        self.numSamples = len(self.X)
        self.numNegatSamples = len(self.N)
        self.batchSize = batchSize
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.reshape(np.random.permutation(np.arange(self.numSamples))[:len(self)*self.batchSize], (len(self),self.batchSize))
        self.negative_indices = np.reshape(np.random.permutation(np.arange(self.numNegatSamples))[:len(self)*self.batchSize], (len(self),self.batchSize))
    
    def __getitem__(self, index):
        X = self.X[self.indices[index]]
        y = self.y[self.indices[index]]
        N = self.N[self.negative_indices[index]]
        positive_slice_relevances = np.ones((self.batchSize,1))
        negative_slice_relevances = np.zeros((self.batchSize,1))
        return [X,N], [y, positive_slice_relevances, negative_slice_relevances]
    
    def __len__(self):
        return self.numSamples // self.batchSize
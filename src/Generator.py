import tensorflow as tf
import numpy as np
import h5py

class Generator(tf.keras.utils.Sequence):

    def __init__(self, dataset_file, isTrain, config):
        dataset_file = h5py.File(dataset_file,'r')
        self.X = dataset_file['X_train' if isTrain else 'X_test'][:]
        self.y = dataset_file['y_train' if isTrain else 'y_test'][:]
        self.N = dataset_file['N_train' if isTrain else 'N_test'][:]
        self.numSamples = len(self.X)
        self.numNegatSamples = len(self.N)
        self.config = config
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.reshape(np.random.permutation(np.arange(self.numSamples))[:len(self)*self.config.BATCH_SIZE], (len(self),self.config.BATCH_SIZE))
        self.negative_indices = np.reshape(np.random.permutation(np.arange(self.numNegatSamples))[:len(self)*self.config.BATCH_SIZE], (len(self),self.config.BATCH_SIZE))
    
    def __getitem__(self, index):
        X = self.X[self.indices[index]]
        y = self.y[self.indices[index]]
        N = self.N[self.negative_indices[index]]
        X, y = self.dataAugmentation(X, y)
        N, _ = self.dataAugmentation(N, None)
        positive_slice_relevances = np.ones((self.config.BATCH_SIZE,1))
        negative_slice_relevances = np.zeros((self.config.BATCH_SIZE,1))
        return [X,N], [y, positive_slice_relevances, negative_slice_relevances]
    
    def __len__(self):
        return self.numSamples // self.config.BATCH_SIZE
    
    def dataAugmentation(self, X, y):
        cropOffsets = np.random.randint(0, high=self.config.CROP_OFFSET, size=(self.config.BATCH_SIZE,2))
        return self.crop(X, cropOffsets), self.crop(y, cropOffsets) if y is not None else None
    
    def crop(self, batch, cropOffsets):
        if len(batch.shape) == 4:
            cropped = np.zeros((self.config.BATCH_SIZE,self.config.IMG_SIZE,self.config.IMG_SIZE,self.config.NUM_CHANNELS))
        else:
            cropped = np.zeros((self.config.BATCH_SIZE,self.config.IMG_SIZE,self.config.IMG_SIZE))
        for sample in range(self.config.BATCH_SIZE):
            cropped[sample] = batch[sample,cropOffsets[sample,0]:cropOffsets[sample,0]+self.config.IMG_SIZE,cropOffsets[sample,1]:cropOffsets[sample,1]+self.config.IMG_SIZE]
        return cropped
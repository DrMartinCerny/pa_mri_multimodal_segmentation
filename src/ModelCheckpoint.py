'''
CUSTOM MODEL CHECKPOINT IS USED TO SAVE THE INNER MODEL INSTEAD OF THE SCAFFOLD MODEL BEING TRAINED
'''

import tensorflow as tf

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    
    def __init__(self, innerModel, modelPath):
        self.minimum_validation_loss = 99999.0
        self.innerModel = innerModel
        self.modelPath = modelPath

    def on_epoch_end(self, epoch, logs):
        if (logs['val_loss']<self.minimum_validation_loss):
            self.innerModel.save(self.modelPath)
            self.minimum_validation_loss = logs['val_loss']
            print('New minimum validation loss of {} achieved after epoch {}, saving model version'.format(self.minimum_validation_loss, epoch+1))
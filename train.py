import sys
import h5py
import matplotlib.pyplot as plt

from src.Config import Config
from src.Model import Model

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

dataset_file = h5py.File(dataset_file,'r')

X_train = dataset_file['X_train'][:]
y_train = dataset_file['y_train'][:]
X_test = dataset_file['X_test'][:]
y_test = dataset_file['y_test'][:]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Model(config)

print(model.model.summary())

history = model.model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=4,epochs=10)

model.model.save(model_folder)

fx,ax = plt.subplots(1,2)
ax[0].plot(history.history['loss'], label='Loss')
ax[0].plot(history.history['val_loss'], label='Validation loss')
ax[1].plot(history.history['accuracy'], label='Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation accuracy')
ax[0].legend(loc='upper center')
ax[1].legend(loc='lower center')
plt.savefig('data/train_history.png')
plt.close()
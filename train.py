import sys
import numpy as np
import matplotlib.pyplot as plt

from src.Config import Config
from src.Generator import Generator
from src.Model import Model

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

generator = Generator(dataset_file,True,config)
val_generator = Generator(dataset_file,False,config)
model = Model(config)

print(model.model.summary())

history = model.scaffold_model.fit(generator,validation_data=val_generator,epochs=10)

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
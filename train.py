import sys
import numpy as np
import matplotlib.pyplot as plt
import json

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

history = model.scaffold_model.fit(generator,validation_data=val_generator,epochs=20)

model.model.save(model_folder)

with open('data/train-history.json', 'w') as outfile:
    json.dump(history.history, outfile)

variable_names = ['Segmentation', 'Positive slice relevance', 'Knosp score', 'Negative slice relevance']
source_variables = ['functional_3', 'functional_3_1', 'functional_3_2', 'functional_3_3']
fx,ax = plt.subplots(2,4)
for i in range(4):
    ax[0,i].plot(history.history[source_variables[i]+'_loss'], label='Loss')
    ax[0,i].plot(history.history['val_'+source_variables[i]+'_loss'], label='Validation loss')
    ax[1,i].plot(history.history[source_variables[i]+'_accuracy'], label='Accuracy')
    ax[1,i].plot(history.history['val_'+source_variables[i]+'_accuracy'], label='Validation accuracy')
    ax[0,i].set_xlabel(variable_names[i])
    ax[1,i].set_xlabel(variable_names[i])
    ax[0,i].legend(loc='upper center')
    ax[1,i].legend(loc='lower center')
plt.tight_layout()
plt.savefig('data/train-history.png')
plt.close()
from gc import callbacks
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

from src.Config import Config
from src.Generator import Generator
from src.Model import Model
from src.ModelCheckpoint import CustomModelCheckpoint

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

generator = Generator(dataset_file,True,config)
val_generator = Generator(dataset_file,False,config)
model = Model(config)
modelCheckpoint = CustomModelCheckpoint(model.model, model_folder)

print(model.model.summary())

history = model.scaffold_model.fit(generator,validation_data=val_generator,epochs=20,callbacks=[modelCheckpoint])

with open('data/train-history.json', 'w') as outfile:
    json.dump(history.history, outfile)
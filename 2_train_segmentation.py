import sys
import json
import os
import tensorflow as tf

from src.Config import Config
from src.Generator import Generator
from src.Model import Model

config = Config()

dataset_file = sys.argv[1]
model_folder = sys.argv[2]

generator = Generator(dataset_file,True,config)
val_generator = Generator(dataset_file,False,config)
model = Model(config).segmentation_model()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_folder, 'segmentation'), save_best_only=True)

print(model.summary())

history = model.fit(generator,validation_data=val_generator,epochs=100,callbacks=[model_checkpoint_callback])

with open('data/train-history-segmentation.json', 'w') as outfile:
    json.dump(history.history, outfile)
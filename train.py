import sys
import numpy as np
import h5py
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

from src.Config import Config
from src.Visualization import Visualization
from src.KnospScore import KnospScore

config = Config()

dataset_file = sys.argv[1]

dataset_file = h5py.File(dataset_file,'r')

X_train = dataset_file['X_train'][:]
y_train = dataset_file['y_train'][:]
X_test = dataset_file['X_test'][:]
y_test = dataset_file['y_test'][:]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

batch_size=32
nb_classes=3
nb_epoch=300
img_channels=1
nb_filters=5
nb_pool=2
nb_conv=5
nb_channels = 16

smooth = 1.

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS])
  embedding = tf.keras.layers.Dense(3)(inputs)

  # Downsampling through the model
  skips = down_stack(embedding)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(output_channels=config.LABEL_CLASSES+1)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

history = model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=4,epochs=10)

model_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("data/model.h5")
print('Model weights saved')

print(history.history)
fx,ax = plt.subplots(1,2)
ax[0].plot(history.history['loss'], label='Loss')
ax[0].plot(history.history['val_loss'], label='Validation loss')
ax[1].plot(history.history['accuracy'], label='Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation accuracy')
ax[0].legend(loc='upper center')
ax[1].legend(loc='lower center')
plt.savefig('data/train_history.png')
plt.close()
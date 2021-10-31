import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

class Model:

    def __init__(self, config):

        base_model = tf.keras.applications.MobileNetV2(input_shape=[config.IMG_SIZE, config.IMG_SIZE, 3], include_top=False)

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
            embedding = tf.keras.layers.LeakyReLU(alpha=0.3)(embedding)

            # Downsampling through the model
            skips = down_stack(embedding)
            x = skips[-1]
            skips = reversed(skips[:-1])

            # Fully connected layers to predict per-sample parameters
            fully_connected = x
            fully_connected = tf.keras.layers.Flatten()(fully_connected)
            fully_connected = tf.keras.layers.Dense(128)(fully_connected)
            fully_connected = tf.keras.layers.LeakyReLU(alpha=0.2)(fully_connected)

            slice_relevance = fully_connected
            slice_relevance = tf.keras.layers.Dense(32)(slice_relevance)
            slice_relevance = tf.keras.layers.LeakyReLU(alpha=0.2)(slice_relevance)
            slice_relevance = tf.keras.layers.Dense(1, activation='sigmoid')(slice_relevance)

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

            return tf.keras.Model(inputs=inputs, outputs=[x, slice_relevance])

        self.model = unet_model(output_channels=config.LABEL_CLASSES+1)
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # Scaffold model to allow for training with negative samples too
        scaffold_segmentation_input = tf.keras.layers.Input(shape=[config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS])
        scaffold_negative_input = tf.keras.layers.Input(shape=[config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS])
        scaffold_segmentation_output, scaffold_positive_slice_relevance = self.model(scaffold_segmentation_input)
        _, scaffold_negative_slice_relevance = self.model(scaffold_negative_input)
        self.scaffold_model = tf.keras.Model(inputs=[scaffold_segmentation_input,scaffold_negative_input], outputs=[scaffold_segmentation_output, scaffold_positive_slice_relevance, scaffold_negative_slice_relevance])
        self.scaffold_model.compile(optimizer='adam',
                    loss=[
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.losses.BinaryCrossentropy(),
                        tf.keras.losses.BinaryCrossentropy(),
                    ],
                    metrics=[['accuracy'],['accuracy'],['accuracy']])
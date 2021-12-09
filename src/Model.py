import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from src.KnospScore import KnospScore

class Model:

    def __init__(self, config):
        
        input_shape = [
            1+config.ADJACENT_SLICES*2,
            config.IMG_SIZE+config.ADJACENT_SLICES*2,
            config.IMG_SIZE+config.ADJACENT_SLICES*2,
            config.NUM_CHANNELS
        ]

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
            inputs = tf.keras.layers.Input(shape=input_shape)
            embedding = tf.keras.layers.Conv3D(64,3)(inputs)
            embedding = tf.keras.layers.Reshape([config.IMG_SIZE, config.IMG_SIZE, 64])(embedding)
            embedding = tf.keras.layers.LeakyReLU(alpha=0.3)(embedding)
            embedding = tf.keras.layers.Dense(3)(embedding)
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
            slice_relevance = tf.keras.layers.Dense(1, activation='sigmoid', name='slice_relevance')(slice_relevance)
            
            knosp_score = fully_connected
            knosp_score = tf.keras.layers.Dense(64)(knosp_score)
            knosp_score = tf.keras.layers.LeakyReLU(alpha=0.2)(knosp_score)
            knosp_score = tf.keras.layers.Reshape((2,32))(knosp_score)
            knosp_score = tf.keras.layers.Dense(len(KnospScore.knospGrades), name='knosp_score')(knosp_score)

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                concat = tf.keras.layers.Concatenate()
                x = concat([x, skip])

            # This is the last layer of the model
            last = tf.keras.layers.Conv2DTranspose(
                filters=output_channels, kernel_size=3, strides=2,
                padding='same', name='predicted_segmentation')  #64x64 -> 128x128

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=[x, slice_relevance, knosp_score])

        self.model = unet_model(output_channels=config.LABEL_CLASSES+1)
        
        # Scaffold model to allow for training with negative samples too
        scaffold_segmentation_input = tf.keras.layers.Input(shape=input_shape)
        scaffold_negative_input = tf.keras.layers.Input(shape=input_shape)
        scaffold_segmentation_output, scaffold_positive_slice_relevance, scaffold_knosp_score = self.model(scaffold_segmentation_input)
        _, scaffold_negative_slice_relevance, _ = self.model(scaffold_negative_input)
        self.scaffold_model = tf.keras.Model(inputs=[scaffold_segmentation_input,scaffold_negative_input], outputs=[scaffold_segmentation_output, scaffold_positive_slice_relevance, scaffold_knosp_score, scaffold_negative_slice_relevance])
        self.scaffold_model.compile(optimizer='adam',
                    loss=[
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.losses.BinaryCrossentropy(),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.losses.BinaryCrossentropy(),
                    ],
                    metrics=[['accuracy'],['accuracy'],['accuracy'],['accuracy']])
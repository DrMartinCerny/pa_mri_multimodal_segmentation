import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_examples.models.pix2pix import pix2pix

from src.KnospScore import KnospScore
from src.PretrainedModel import unet

class Model:

    def __init__(self, config):
        
        self.config = config
        
        input_shape = [
            1+config.ADJACENT_SLICES*2,
            config.IMG_SIZE+config.ADJACENT_SLICES*2,
            config.IMG_SIZE+config.ADJACENT_SLICES*2,
            config.NUM_CHANNELS
        ]
        
        pretrained_model = unet()
        pretrained_model.load_weights("data/pretrained_weights.h5")

        inputs_pretrained = tf.keras.layers.Input(shape=[128,128,32])
        block_1_convolution_1 = pretrained_model.get_layer('conv2d_2')(inputs_pretrained)
        block_1_activation_1 = pretrained_model.get_layer('activation_2')(block_1_convolution_1)
        block_1_convolution_2 = pretrained_model.get_layer('conv2d_3')(block_1_activation_1)
        block_1_activation_2 = pretrained_model.get_layer('activation_3')(block_1_convolution_2)
        block_1_max_pooling = pretrained_model.get_layer('max_pooling2d_1')(block_1_activation_2)
        block_2_convolution_1 = pretrained_model.get_layer('conv2d_4')(block_1_max_pooling)
        block_2_activation_1 = pretrained_model.get_layer('activation_4')(block_2_convolution_1)
        block_2_convolution_2 = pretrained_model.get_layer('conv2d_5')(block_2_activation_1)
        block_2_activation_2 = pretrained_model.get_layer('activation_5')(block_2_convolution_2)
        block_2_max_pooling = pretrained_model.get_layer('max_pooling2d_2')(block_2_activation_2)
        block_3_convolution_1 = pretrained_model.get_layer('conv2d_6')(block_2_max_pooling)
        block_3_activation_1 = pretrained_model.get_layer('activation_6')(block_3_convolution_1)
        block_3_convolution_2 = pretrained_model.get_layer('conv2d_7')(block_3_activation_1)
        block_3_activation_2 = pretrained_model.get_layer('activation_7')(block_3_convolution_2)
        block_3_max_pooling = pretrained_model.get_layer('max_pooling2d_3')(block_3_activation_2)

        pretrained_downsampling_stack = tf.keras.Model(inputs=inputs_pretrained, outputs=[block_1_max_pooling,block_2_max_pooling,block_3_max_pooling])
        pretrained_downsampling_stack.trainable = False

        inputs = tf.keras.layers.Input(shape=[128,128,32])
        pretrained_output = pretrained_downsampling_stack(inputs)
        block_4_convolution_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(pretrained_output[2])
        block_4_activation_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(block_4_convolution_1)
        block_4_convolution_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(block_4_activation_1)
        block_4_activation_2 = tf.keras.layers.Activation('relu')(block_4_convolution_2)
        block_4_max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block_4_activation_2)
        block_5_convolution_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(block_4_max_pooling)
        block_5_activation_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(block_5_convolution_1)
        block_5_convolution_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(block_5_activation_1)
        block_5_activation_2 = tf.keras.layers.Activation('relu')(block_5_convolution_2)
        block_5_max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(block_5_activation_2)

        down_stack = tf.keras.Model(inputs=inputs, outputs=[pretrained_output[0],pretrained_output[1],pretrained_output[2],block_4_max_pooling,block_5_max_pooling])

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
            embedding = tf.keras.layers.Dense(32)(embedding)
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
                    metrics=[['accuracy', self.dice_coef_total, self.dice_coef_tumor, self.dice_coef_ica, self.dice_coef_normal_gland],['accuracy'],['accuracy'],['accuracy']])
    
    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        return (2. * intersection + self.config.DICE_COEF_SMOOTH) / (K.sum(y_true) + K.sum(y_pred) + self.config.DICE_COEF_SMOOTH)

    def dice_coef_by_class(self, y_true, y_pred, classId):
        y_pred = K.argmax(y_pred)
        y_pred = K.cast(y_pred == classId, dtype='float32')
        y_true = K.cast(y_true == classId, dtype='float32')
        return self.dice_coef(y_true, y_pred)

    def dice_coef_total(self, y_true, y_pred):
        y_pred = K.argmax(y_pred)
        y_pred = K.one_hot(K.cast(y_pred, dtype='int32'), self.config.LABEL_CLASSES+1)
        y_true = K.one_hot(K.cast(y_true, dtype='int32'), self.config.LABEL_CLASSES+1)
        return self.dice_coef(y_true, y_pred)

    def dice_coef_tumor(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 1)

    def dice_coef_ica(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 2)

    def dice_coef_normal_gland(self, y_true, y_pred):
        return self.dice_coef_by_class(y_true, y_pred, 3)
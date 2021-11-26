class Config:

    def __init__(self):
        self.IMG_SIZE = 128
        self.CROP_OFFSET = 16
        self.IMG_SIZE_UNCROPPED = self.IMG_SIZE + self.CROP_OFFSET
        self.NUM_CHANNELS = 3
        self.LABEL_CLASSES = 2
        self.BATCH_SIZE = 4
        self.TRAIN_VALIDATION_SPLIT = 0.8
        self.IMAGE_REGISTRATION_EPOCHS = 15
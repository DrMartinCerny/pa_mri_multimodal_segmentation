class Config:

    def __init__(self):
        self.IMG_SIZE = 192
        self.NUM_CHANNELS = 3
        self.ADJACENT_SLICES = 1
        self.CROP_OFFSET = 16
        self.IMG_SIZE_PADDED = self.IMG_SIZE + self.ADJACENT_SLICES*2
        self.IMG_SIZE_UNCROPPED = self.IMG_SIZE_PADDED + self.CROP_OFFSET
        self.LABEL_CLASSES = 3
        self.BATCH_SIZE = 16
        self.TRAIN_VALIDATION_SPLIT = 0.8
        self.IMAGE_REGISTRATION_EPOCHS = 15
        self.DICE_COEF_SMOOTH = 1.0
        self.USE_CLASS_WEIGHTS = False
import sys
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.Config import Config
from src.ImageRegistration import ImageRegistration
from src.Model import Model

config_file = sys.argv[1]
model_folder = sys.argv[2]
source_folder = sys.argv[3]
target_file = sys.argv[4]

config = Config(config_file)
model = Model(config)
imageRegistration = ImageRegistration(config)

#mask = os.path.join(source_folder, 'mask.nii')
cor_t1_c = os.path.join(source_folder, 'COR_T1_C.nii')
cor_t1 = os.path.join(source_folder, 'COR_T1.nii')
cor_t2 = os.path.join(source_folder, 'COR_T2.nii')
if os.path.exists(cor_t1_c):
    # LOAD IMAGES
    cor_t1_c = sitk.ReadImage(cor_t1_c, sitk.sitkFloat32)
    cor_t1 = sitk.ReadImage(cor_t1, sitk.sitkFloat32) if os.path.exists(cor_t1) and config.NUM_CHANNELS >= 2 else None
    cor_t2 = sitk.ReadImage(cor_t2, sitk.sitkFloat32) if os.path.exists(cor_t2) and config.NUM_CHANNELS >= 3 else None

    # REGISTER IMAGES TO T COR C
    cor_t1_transform = imageRegistration.findTransformation(cor_t1_c, cor_t1) if cor_t1 is not None else None
    cor_t2_transform = imageRegistration.findTransformation(cor_t1_c, cor_t2) if cor_t2 is not None else None

    # TRANSFORM IMAGES TO COR T1 SPACE
    cor_t1 = sitk.Resample(cor_t1, cor_t1_c, cor_t1_transform, sitk.sitkBSpline, 0) if cor_t1 is not None else None
    cor_t2 = sitk.Resample(cor_t2, cor_t1_c, cor_t2_transform, sitk.sitkBSpline, 0) if cor_t2 is not None else None

    # GET VOXEL ARRAY DATA
    cor_t1_c = sitk.GetArrayFromImage(cor_t1_c)
    cor_t1 = sitk.GetArrayFromImage(cor_t1) if cor_t1 is not None else None
    cor_t2 = sitk.GetArrayFromImage(cor_t2) if cor_t2 is not None else None

    # CROP IMAGES
    original_shape = cor_t1_c.shape
    centerX = int(cor_t1_c.shape[1]/2)
    centerY = int(cor_t1_c.shape[2]/2)
    top = int(centerY-config.IMG_SIZE_PADDED/2)
    bottom = int(centerY+config.IMG_SIZE_PADDED/2)
    left = int(centerX-config.IMG_SIZE_PADDED/2)
    right = int(centerX+config.IMG_SIZE_PADDED/2)

    # CROP IMAGES
    cor_t1_c = cor_t1_c[:,left:right,top:bottom]
    cor_t1 = cor_t1[:,left:right,top:bottom] if cor_t1 is not None else None
    cor_t2 = cor_t2[:,left:right,top:bottom] if cor_t2 is not None else None

    # NORMALIZE CROPPED IMAGES TO ZERO MEAN AND UNIT VARIANCE
    cor_t1_c = StandardScaler().fit_transform(cor_t1_c.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_PADDED,config.IMG_SIZE_PADDED))
    cor_t1 = StandardScaler().fit_transform(cor_t1.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_PADDED,config.IMG_SIZE_PADDED)) if cor_t1 is not None else None
    cor_t2 = StandardScaler().fit_transform(cor_t2.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_PADDED,config.IMG_SIZE_PADDED)) if cor_t2 is not None else None
    
    # CONCATENATE CHANNELS
    X = np.zeros((len(cor_t1_c)-2,1+config.ADJACENT_SLICES*2,config.IMG_SIZE_PADDED,config.IMG_SIZE_PADDED,config.NUM_CHANNELS))
    for i in range(len(X)):
        X[i,:,:,:,0] = cor_t1_c[i:i+1+config.ADJACENT_SLICES*2]
        if cor_t1 is not None : X[i,:,:,:,1] = cor_t1[i:i+1+config.ADJACENT_SLICES*2]
        if cor_t2 is not None : X[i,:,:,:,2] = cor_t2[i:i+1+config.ADJACENT_SLICES*2]

    # PREDICT SEGMENTATION
    model = keras.models.load_model(os.path.join(model_folder, 'segmentation'), compile=False, custom_objects={'dice_coef_total': model.dice_coef_total, 'dice_coef_tumor': model.dice_coef_tumor, 'dice_coef_ica': model.dice_coef_ica, 'dice_coef_normal_gland': model.dice_coef_normal_gland})
    predicted = model.predict(X)
    predicted = np.argmax(predicted,axis=-1)

    # REMOVE IRRELEVANT SLICES FROM SEGMENTATION
    model = keras.models.load_model(os.path.join(model_folder, 'slice-selection'), compile=False)
    irrelevant_slices = [x[0] < 0.5 for x in model.predict(X)]
    predicted[irrelevant_slices,:,:] = 0
    
    # SAVE MASK
    mask = np.zeros(original_shape, dtype=np.int16)
    mask[1:-1,left+1:right-1,top+1:bottom-1] = predicted
    mask = np.swapaxes(mask,0,2)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), target_file)
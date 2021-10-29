import sys
import glob
import os
import numpy as np
import SimpleITK as sitk
import h5py
from sklearn.preprocessing import StandardScaler

from src.Config import Config

config = Config()
dataset_source_folder = sys.argv[1]
dataset_target_file = sys.argv[2]

def readNifti(filename):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(filename)
    return reader.Execute()

dataset_X = []
dataset_y = []

for subject in glob.glob(os.path.join(dataset_source_folder, '**'))[:2]:
    mask = os.path.join(subject, 'mask.nii')
    cor_t1_c = os.path.join(subject, 'COR_T1_c.nii')
    cor_t1 = os.path.join(subject, 'COR_T1.nii')
    ax_t2 = os.path.join(subject, 'AX_t2.nii')
    dwi = os.path.join(subject, 'DWI.nii')
    if os.path.exists(mask) and os.path.exists(cor_t1_c) and os.path.exists(cor_t1) and os.path.exists(ax_t2) and os.path.exists(dwi):
        mask = sitk.GetArrayFromImage(readNifti(mask))
        if np.sum(mask) > 0:
            # LOAD IMAGES
            cor_t1_c = readNifti(cor_t1_c)
            cor_t1 = readNifti(cor_t1)
            ax_t2 = readNifti(ax_t2)
            dwi = readNifti(dwi)

            # FIND TRANSFORMATIONS TO COR T SPACE
            # TODO: register transformations to account for patient shifts
            cor_t1_transform = sitk.TranslationTransform(cor_t1_c.GetDimension())
            ax_t2_transform = sitk.TranslationTransform(cor_t1_c.GetDimension())
            dwi_transform = sitk.TranslationTransform(cor_t1_c.GetDimension())

            # TRANSFORM IMAGES TO COR T1 SPACE
            cor_t1 = sitk.Resample(cor_t1, cor_t1_c, cor_t1_transform, sitk.sitkNearestNeighbor, 0)
            ax_t2 = sitk.Resample(ax_t2, cor_t1_c, ax_t2_transform, sitk.sitkNearestNeighbor, 0)
            dwi = sitk.Resample(dwi, cor_t1_c, dwi_transform, sitk.sitkNearestNeighbor, 0)
            cor_t1_c = sitk.GetArrayFromImage(cor_t1_c)
            cor_t1 = sitk.GetArrayFromImage(cor_t1)
            ax_t2 = sitk.GetArrayFromImage(ax_t2)
            dwi = sitk.GetArrayFromImage(dwi)

            # FIND AREA OF INTEREST WITHIN THE IMAGE
            # TODO: image center computed as center of tumor label, can induce bias, switch to atlas registration later
            top = np.min(np.argwhere(np.sum(mask==1,axis=(0,1))>0))
            bottom = np.max(np.argwhere(np.sum(mask==1,axis=(0,1))>0))
            left = np.min(np.argwhere(np.sum(mask==1,axis=(0,2))>0))
            right = np.max(np.argwhere(np.sum(mask==1,axis=(0,2))>0))
            centerX = int((left+right)/2)
            centerY = int((top+bottom)/2)
            top = int(centerY-config.IMG_SIZE/2)
            bottom = int(centerY+config.IMG_SIZE/2)
            left = int(centerX-config.IMG_SIZE/2)
            right = int(centerX+config.IMG_SIZE/2)

            # CROP IMAGES
            mask = mask[:,left:right,top:bottom]
            cor_t1_c = cor_t1_c[:,left:right,top:bottom]
            cor_t1 = cor_t1[:,left:right,top:bottom]
            ax_t2 = ax_t2[:,left:right,top:bottom]
            dwi = dwi[:,left:right,top:bottom]

            # ADD TO DATASET
            labeledSlices = np.sum(mask, axis=(1,2)) > 0
            for slice, isLabeled in enumerate(labeledSlices):
                if isLabeled:
                    dataset_y.append(mask[slice])
                    dataset_X.append(np.stack([cor_t1_c[slice],cor_t1[slice],ax_t2[slice],dwi[slice]],axis=-1))

dataset_X = np.stack(dataset_X).astype(np.float64)
dataset_y = np.stack(dataset_y)

# STANDART SCALING
dataset_X[:,:,:,0] = StandardScaler().fit_transform(dataset_X[:,:,:,0].flatten().reshape(-1,1)).reshape((len(dataset_X),config.IMG_SIZE,config.IMG_SIZE))
dataset_X[:,:,:,1] = StandardScaler().fit_transform(dataset_X[:,:,:,1].flatten().reshape(-1,1)).reshape((len(dataset_X),config.IMG_SIZE,config.IMG_SIZE))
dataset_X[:,:,:,2] = StandardScaler().fit_transform(dataset_X[:,:,:,2].flatten().reshape(-1,1)).reshape((len(dataset_X),config.IMG_SIZE,config.IMG_SIZE))
dataset_X[:,:,:,3] = StandardScaler().fit_transform(dataset_X[:,:,:,3].flatten().reshape(-1,1)).reshape((len(dataset_X),config.IMG_SIZE,config.IMG_SIZE))

# TODO: train/test split

print(dataset_X.shape, dataset_y.shape)

dataset_target_file = f = h5py.File(dataset_target_file, "w")
dataset_target_file.create_dataset("X_train", data=dataset_X)
dataset_target_file.create_dataset("X_test", data=dataset_X)
dataset_target_file.create_dataset("y_train", data=dataset_y)
dataset_target_file.create_dataset("y_test", data=dataset_y)
dataset_target_file.close()
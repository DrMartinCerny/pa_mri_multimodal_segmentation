import sys
import glob
import os
import yaml
import numpy as np
import SimpleITK as sitk
import h5py
from sklearn.preprocessing import StandardScaler

from src.Config import Config
from src.ImageRegistration import ImageRegistration
from src.KnospScore import KnospScore

config = Config()
imageRegistration = ImageRegistration(config)
dataset_source_folder = sys.argv[1]
dataset_target_file = sys.argv[2]

dataset_X = []
dataset_y = []
negative_samples = []
knosp_scores = []
for subject in glob.glob(os.path.join(dataset_source_folder, '**')):
    mask = os.path.join(subject, 'mask.nii')
    cor_t1_c = os.path.join(subject, 'COR_T1_c.nii')
    cor_t1 = os.path.join(subject, 'COR_T1.nii')
    ax_t2 = os.path.join(subject, 'AX_t2.nii')
    dwi = os.path.join(subject, 'DWI.nii')
    knosp = os.path.join(subject, 'knosp.yaml')
    if os.path.exists(mask) and os.path.exists(cor_t1_c) and os.path.exists(cor_t1) and os.path.exists(ax_t2) and os.path.exists(dwi) and os.path.exists(knosp):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask, sitk.sitkInt16))
        if np.sum(mask) > 0:
            # LOAD IMAGES
            cor_t1_c = sitk.ReadImage(cor_t1_c, sitk.sitkFloat32)
            cor_t1 = sitk.ReadImage(cor_t1, sitk.sitkFloat32)
            ax_t2 = sitk.ReadImage(ax_t2, sitk.sitkFloat32)
            dwi = sitk.ReadImage(dwi, sitk.sitkFloat32)

            # REGISTER IMAGES TO T COR C
            cor_t1_transform = imageRegistration.findTransformation(cor_t1_c, cor_t1)
            ax_t2_transform = imageRegistration.findTransformation(cor_t1_c, ax_t2)
            dwi_transform = imageRegistration.findTransformation(cor_t1_c, dwi)

            # TRANSFORM IMAGES TO COR T1 SPACE
            cor_t1 = sitk.Resample(cor_t1, cor_t1_c, cor_t1_transform, sitk.sitkBSpline, 0)
            ax_t2 = sitk.Resample(ax_t2, cor_t1_c, ax_t2_transform, sitk.sitkBSpline, 0)
            dwi = sitk.Resample(dwi, cor_t1_c, dwi_transform, sitk.sitkBSpline, 0)

            # GET VOXEL ARRAY DATA
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
            top = int(centerY-config.IMG_SIZE_UNCROPPED/2)
            bottom = int(centerY+config.IMG_SIZE_UNCROPPED/2)
            left = int(centerX-config.IMG_SIZE_UNCROPPED/2)
            right = int(centerX+config.IMG_SIZE_UNCROPPED/2)

            # CROP IMAGES
            mask = mask[:,left:right,top:bottom]
            cor_t1_c = cor_t1_c[:,left:right,top:bottom]
            cor_t1 = cor_t1[:,left:right,top:bottom]
            ax_t2 = ax_t2[:,left:right,top:bottom]
            dwi = dwi[:,left:right,top:bottom]

            # NORMALIZE CROPPED IMAGES TO ZERO MEAN AND UNIT VARIANCE
            cor_t1_c = StandardScaler().fit_transform(cor_t1_c.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED))
            cor_t1 = StandardScaler().fit_transform(cor_t1.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED))
            ax_t2 = StandardScaler().fit_transform(ax_t2.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED))
            dwi = StandardScaler().fit_transform(dwi.flatten().reshape(-1,1)).reshape((len(cor_t1_c),config.IMG_SIZE_UNCROPPED,config.IMG_SIZE_UNCROPPED))

            # LOAD KNOSP SCORE GROUND TRUTH
            with open(knosp) as file:
                knosp = yaml.load(file, Loader=yaml.FullLoader)
            
            # ADD TO DATASET
            labeledSlices = np.sum(mask, axis=(1,2)) > 0
            for slice in [x[0] for x in np.argwhere(labeledSlices)]:
                dataset_y.append(mask[slice])
                dataset_X.append(np.stack([cor_t1_c[slice],cor_t1[slice],ax_t2[slice],dwi[slice]],axis=-1))
                knosp_scores.append(np.array([KnospScore.knospGrades.index(knosp[str(slice)]['left']), KnospScore.knospGrades.index(knosp[str(slice)]['right'])]))
            for slice in np.random.permutation([x[0] for x in np.argwhere(np.invert(labeledSlices))])[:np.count_nonzero(labeledSlices)]:
                negative_samples.append(np.stack([cor_t1_c[slice],cor_t1[slice],ax_t2[slice],dwi[slice]],axis=-1))

dataset_X = np.stack(dataset_X)
negative_samples = np.stack(negative_samples)
dataset_y = np.stack(dataset_y)
knosp_scores = np.stack(knosp_scores)

# TODO: train/test split

print(dataset_X.shape, dataset_y.shape, negative_samples.shape, knosp_scores.shape)

dataset_target_file = f = h5py.File(dataset_target_file, "w")
dataset_target_file.create_dataset("X_train", data=dataset_X)
dataset_target_file.create_dataset("X_test", data=dataset_X)
dataset_target_file.create_dataset("y_train", data=dataset_y)
dataset_target_file.create_dataset("y_test", data=dataset_y)
dataset_target_file.create_dataset("N_train", data=negative_samples)
dataset_target_file.create_dataset("N_test", data=negative_samples)
dataset_target_file.create_dataset("K_train", data=knosp_scores)
dataset_target_file.create_dataset("K_test", data=knosp_scores)
dataset_target_file.close()
# pa_mri_multimodal_segmentation
Pituitary Adenoma Segmentation From Multimodal MRI Images

Usage:

pip install -r requirements.txt

python 1_create_dataset.py data/config/default.yaml data/example-dataset data/dataset.h5

python 2_train_segmentation.py data/config/default.yaml data/dataset.h5 data/model

python 3_train_slice_selection.py data/config/default.yaml data/dataset.h5 data/model

python 4_train_knosp_score.py data/config/default.yaml data/dataset.h5 data/model

python visualize.py data/config/default.yaml data/dataset.h5 data/model

python 6_predict.py data/config/default.yaml data/model data/example-dataset/train/1 data/example-dataset/train/1/mask_predicted.nii
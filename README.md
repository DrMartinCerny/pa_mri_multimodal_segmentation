# pa_mri_multimodal_segmentation
Pituitary Adenoma Segmentation From Multimodal MRI Images

Usage:

pip install requirements.txt

python create_dataset.py data/example-dataset data/dataset.h5

python train.py data/dataset.h5 data/model

python visualize.py data/dataset.h5 data/model
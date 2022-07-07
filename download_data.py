import wget
import zipfile
import os

response = wget.download('https://storage.googleapis.com/pa-mri-multimodal-segmentation/pa_mri_multimodal_segmentation.zip', 'data/pa_mri_multimodal_segmentation.zip')
with zipfile.ZipFile('data/pa_mri_multimodal_segmentation.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
os.remove('data/pa_mri_multimodal_segmentation.zip')
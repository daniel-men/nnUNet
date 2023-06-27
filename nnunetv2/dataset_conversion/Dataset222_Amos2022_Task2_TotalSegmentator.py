from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import io
import os
import contextlib
import sys
import random
import json
import time
import string
import shutil
import zipfile
from pathlib import Path

import requests
import numpy as np
import nibabel as nib
from glob import glob

amos_to_ts_mapping = {0: 0,
 1: 1,
 2: 2,
 3: 3,
 4: 4,
 6: 5,
 7: 6,
 8: 7,
 9: 8,
 10: 10,
 11: 11,
 12: 12,
 5: 42,
 13: 56,
 14: 104,
 15: 105}

def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)

def convert_task(amos_base_dir: str, total_segmentator_base_dir: str, nnunet_dataset_id: int = 222):
    task_name = "AMOS2022_postChallenge_task2_TotalSegmentator"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    dataset_json_source = load_json(join(amos_base_dir, 'dataset.json'))

    training_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['training']]
    for tr in training_identifiers:
        shutil.copy(join(amos_base_dir, 'imagesTr', tr + '.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
        shutil.copy(join(amos_base_dir, 'labelsTr', tr + '.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    test_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['test']]
    for ts in test_identifiers:
        shutil.copy(join(amos_base_dir, 'imagesTs', ts + '.nii.gz'), join(imagests, f'{ts}_0000.nii.gz'))

    val_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['validation']]
    for vl in val_identifiers:
        shutil.copy(join(amos_base_dir, 'imagesVa', vl + '.nii.gz'), join(imagestr, f'{vl}_0000.nii.gz'))
        shutil.copy(join(amos_base_dir, 'labelsVa', vl + '.nii.gz'), join(labelstr, f'{vl}.nii.gz'))

    # Map AMOS22 class labels to TotalSegmentator
    amos_label_files = glob(join(labelstr, "*.nii.gz"))
    for amos_label_file in amos_label_files:
        mask_orig = nib.load(amos_label_file).get_fdata()
        mask_ts = np.zeros_like(mask_orig)

        for amos_class in amos_to_ts_mapping:
            mask_ts[np.where(mask_orig == amos_class)] = amos_to_ts_mapping[amos_class]

        nib.save(nib.Nifti1Image(mask_ts, mask_orig.affine), amos_label_file)

    with open(join(total_segmentator_base_dir, "CTs.lst", 'r')) as f:
        image_paths = f.readlines()

    for img_path in image_paths:
        identifier = img_path.split(os.path.sep)[0]
        img_path = join(total_segmentator_base_dir, img_path)
        shutil.copy(img_path, join(imagestr, f"{identifier}_0000.nii.gz"))

        masks_dir = join(total_segmentator_base_dir, join(identifier, "segmentations"))
        label_path = join(labelstr, f'{identifier}.nii.gz')
        combine_masks_to_multilabel_file(masks_dir=masks_dir, multilabel_file=label_path)



    
    generate_dataset_json(out_base, {0: "either_CT_or_MR"}, labels={v: int(k) for k,v in dataset_json_source['labels'].items()},
                          num_training_cases=len(training_identifiers) + len(val_identifiers)+len(image_paths), file_ending='.nii.gz',
                          dataset_name=task_name, reference='https://amos22.grand-challenge.org/',
                          release='https://zenodo.org/record/7262581',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="This is the dataset as released AFTER the challenge event. It has the "
                                      "validation set gt in it! We just use the validation images as additional "
                                      "training cases because AMOS doesn't specify how they should be used. nnU-Net's"
                                      " 5-fold CV is better than some random train:val split.")


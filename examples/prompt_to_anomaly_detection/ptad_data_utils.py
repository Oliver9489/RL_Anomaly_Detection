import os
import re
from PIL import Image

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
from torchvision.transforms import ToTensor
from torchvision import transforms


class PromptToAnomalyDetectionDataset(Dataset):
    def __init__(self, source_img, GT_img, file_names):
        assert len(source_img) == len(GT_img)
        self.source_img = source_img
        self.GT_img = GT_img
        self.file_names = file_names

    def __len__(self):
        return len(self.source_img)

    def __getitem__(self, idx):
        to_tensor = ToTensor()
        item = {'source_img': self.source_img[idx],
                'GT_img': self.GT_img[idx],
                'file_names': self.file_names[idx]
                }
        return item


def load_anomaly_detection_dataset(
        dataset: str,
        split: str,
        base_path: str,
        # dataset_seed: Optional[int],
        # max_size: Optional[int],
        # max_length: Optional[int],
        # max_length_tokenizer: Optional[str]
) -> Tuple[List[np.ndarray], List[str]]:
    assert dataset in ['ksdd2']
    assert split in ['train', 'test', 'dev']
    if dataset == 'ksdd2':
        filepath = f'{dataset}/{split}/'
        full_filepath = os.path.join(base_path, filepath)
        file_names = [f for f in os.listdir(full_filepath) if not f.endswith('_GT.png')]
        # gt_file_names = [f for f in os.listdir(full_filepath) if f.endswith('_GT.png')]
        source_img = []
        GT_img = []
        for name in file_names:
            img_name = os.path.join(full_filepath, name)
            gt = f'{name.split(".")[0]}_GT.png'
            # print(gt)
            gt_name = os.path.join(full_filepath, gt)
            # get img data and its gt
            img = Image.open(img_name).convert("RGB")
            img = transforms.Resize((400, 400))(img)
            source_img.append(np.array(img))
            gt_img = Image.open(gt_name).convert('L')
            gt_img = transforms.Resize((400, 400))(gt_img)
            GT_img.append(np.array(gt_img))
            # if name.split('.')[0] != gt.split('_GT')[0]:
            #     print("------------------------------------")
            #     print(name.split('.')[0])
            #     print(gt.split('_GT')[0])

    return source_img, GT_img, file_names

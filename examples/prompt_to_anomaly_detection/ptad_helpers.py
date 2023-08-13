from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
from ptad_reward import PromptedToAnomalyDetectionReward
from ptad_data_utils import (PromptToAnomalyDetectionDataset,
                             load_anomaly_detection_dataset)


def make_anomaly_detection_datasets(
        config: "DictConfig") -> Tuple[PromptToAnomalyDetectionDataset]:
    # assert config.direction in ['0_to_1', '1_to_0']
    # label = int(config.direction[0])
    data_dict = {}
    for split in ['train', 'dev', 'test']:
        # Hack - Only use 16 examples for Yelp validation to save time
        # max_size = 16
        # if config.dataset == "yelp" and split == 'dev':
        #     max_size = 16
        # else:
        #     max_size = config.max_size

        source_img, GT_img, file_names = load_anomaly_detection_dataset(
            config.dataset, split, config.base_path)
        ptad_dataset = PromptToAnomalyDetectionDataset(source_img, GT_img, file_names)
        data_dict[split] = ptad_dataset

    return data_dict['train'], data_dict['dev'], data_dict['test']


# def load_text_style_transfer_test_data(config: "DictConfig"):
#     label = int(config.direction[0])
#     source_texts, target_labels = load_anomaly_detection_dataset(
#         config.dataset, label, 'test',
#         config.dataset_seed, config.base_path, config.max_size,
#         config.max_length, config.max_length_tokenizer)
#
#     if config.dataset == 'yelp': # Separate human-written reference
#         ref_texts, _ = load_anomaly_detection_dataset(
#             config.dataset, label, 'ref',
#             config.dataset_seed, config.base_path, None,
#             None, None)
#     elif config.dataset == 'shakespeare': # Opposite test data are references
#         target_label = int(config.direction[1])
#         ref_texts, _ = load_anomaly_detection_dataset(
#             config.dataset, target_label, 'test',
#             config.dataset_seed, config.base_path, None,
#             None, None)
#
#     return source_texts, target_labels, ref_texts

# Key: (dataset, dataset_seed, split)
style_classifier_dict = {
    ('yelp', None, 'train'): './style_classifiers/yelp-bert-base-uncased-train/',
    ('yelp', None, 'test'): './style_classifiers/yelp-bert-base-uncased-test/',
    ('shakespeare', 0, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-0/',
    ('shakespeare', 1, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-1/',
    ('shakespeare', 2, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-2/',
    ('shakespeare', None, 'test'): './style_classifiers/shakespeare-bert-base-uncased-test-all/',
}


# def get_style_classifier(
#     split: str,
#     config: "DictConfig"
# ) -> str:
#     dataset_seed = config.dataset_seed if split != 'test' else None
#     return style_classifier_dict[(config.dataset, dataset_seed, split)]
def get_style_classifier(
        split: str,
        config: "DictConfig"
) -> str:
    dataset_seed = config.dataset_seed if split != 'test' else None
    return style_classifier_dict[('shakespeare', dataset_seed, split)]


def make_prompted_to_anomaly_detection_reward(
        config: "DictConfig", num_repeats: int = -1) -> PromptedToAnomalyDetectionReward:
    if num_repeats != -1:
        config.num_repeats = num_repeats
    return PromptedToAnomalyDetectionReward(
        config.task_lm, config.task_top_k,  # config.style_classifier,
        config.style_tokenizer,
        config.style_batch_size, config.pad_token, config.num_repeats,
        config.num_samples, config.num_bootstraps, config.compute_zscore,
        config.lower_outputs, config.control_output_length,
        config.template, config.end_punct)


@dataclass
class TextStyleTransferDatasetConfig:
    dataset: str = "???"
    dataset_seed: Optional[int] = None
    direction: str = "???"
    base_path: str = './data'
    max_size: Optional[int] = None
    max_length: Optional[int] = None
    max_length_tokenizer: Optional[str] = None


@dataclass
class PromptedTextStyleTransferRewardConfig:
    task_lm: str = 'distilgpt2'
    task_top_k: int = 10
    style_classifier: str = '???'
    style_tokenizer: Optional[str] = None
    # style_batch_size: int = 32
    style_batch_size: int = 8
    pad_token: str = '<|endoftext|>'
    num_repeats: int = 4
    num_samples: int = 1
    num_bootstraps: int = 2
    compute_zscore: bool = True  # Whether to compute z-score of rewards
    lower_outputs: bool = False  # Whether to convert all outputs to lower case
    control_output_length: bool = False
    template: str = '{prompt} "{sentence_1}" "'
    end_punct: str = '"'

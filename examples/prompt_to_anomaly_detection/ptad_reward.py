import os.path
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
from collections import defaultdict
from ptad_modules import PromptedGenerator, TextStyleTransferOutputSelector

from rlprompt.rewards import BaseReward

from PIL import Image
from lang_sam import LangSAM

# Magic variable
SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']


class PromptedToAnomalyDetectionReward(BaseReward):
    def __init__(
            self,
            task_lm: str,
            task_top_k: int,  # Top-k sampling for text generation
            style_classifier: str,
            style_tokenizer: Optional[str],
            style_batch_size: int,
            pad_token: str,
            num_repeats: int,  # Num of repetitions for each example
            num_samples: int,  # Num of samples from which to take the output
            num_bootstraps: int,  # Num of bootstraps to reduce reward randomness
            compute_zscore: bool,  # Whether to compute z-score of rewards
            lower_outputs: bool,  # Whether to convert all outputs to lower case
            control_output_length: bool,  # Control output length for speedup
            template: str,  # Template for prompt generation
            end_punct: str,  # End punctuation to cut off after generation

    ):
        generator_device = 0  # TODO
        reward_device = 0  # TODO

        # Loading generator model
        assert task_lm in SUPPORTED_LMS
        print('Task LM:', task_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(task_lm)
        self.generator = PromptedGenerator(task_lm, template, end_punct,
                                           pad_token, generator_device,
                                           lower_outputs, control_output_length)
        self.top_k = task_top_k
        self.top_p = 1.0
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps

        # Loading reward models
        if style_tokenizer is None:
            style_tokenizer = style_classifier
        self.selector = TextStyleTransferOutputSelector(style_classifier,
                                                        style_tokenizer,
                                                        style_batch_size,
                                                        reward_device)

        # Misc. training details
        self.num_repeats = num_repeats
        self.compute_zscore = compute_zscore
        self._counter = 0
        self.tokens_explored = set()
        self.model = LangSAM()

    def forward(
            self,
            source_img: List[Image.Image],
            GT_img: List[Image.Image],
            file_names: List[str],
            source_texts: List[str],
            # target_labels: List[str],
            output_tokens: List[List[str]],
            to_tensor: bool,
            mode: str,
            dataset: Optional[str],
            tst_active: int = 0,
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == 'train':
            self._counter += 1
            source_strs = self._repeat_texts(source_texts)
            # target_labels = self._repeat_texts(target_labels)
        elif mode == "infer":
            source_strs = source_texts
        else:
            raise ValueError

        if dataset == None:
            dataset = '-1'

        prompt_tokens = output_tokens
        # convert the output tokens to string. meaning this is the output prompt that we can use
        # and its a list consist of string
        prompt_strs = self._convert_tokens_to_string(prompt_tokens)
        assert len(prompt_strs) == len(source_strs)

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward

        # input_rewards: Dict[str, List[float]] = defaultdict(list)
        reward_list = []
        for i, (prompt, src) in enumerate(zip(prompt_strs, source_strs)):
            hypos_rewards_list = []
            hypos = []

            # whether acitive tst
            if tst_active:
                if mode == 'train':
                    hypos = self.generator.sample_generate(prompt, src, N,
                                                           self.top_k, self.top_p)
                elif mode == 'infer':
                    hypos = self.generator.sample_generate(prompt, src, 1,
                                                           self.top_k, self.top_p)
            else:
                hypos = [prompt]

            for hypo in hypos:
                # print("input prompt :", src)
                # print("new prompt :", hypo)
                for (source, GT, file_name) in zip(source_img, GT_img, file_names):
                    # iou = self.prompts_to_seg(text_prompt=hypo,
                    #                           src=src,
                    #                           image_pil=source,
                    #                           image_gt=GT,
                    #                           file_name=file_name,
                    #                           dataset=dataset,
                    #                           mode=mode)
                    f1 = self.calculate_f1_score(text_prompt=hypo,
                                                 src=src,
                                                 image_pil=source,
                                                 image_gt=GT,
                                                 file_name=file_name,
                                                 dataset=dataset,
                                                 mode=mode)
                    hypos_rewards_list.append(round(f1 * 100, 2))
                # reward = sum(iou_list) / len(iou_list)
            reward = sum(hypos_rewards_list) / len(hypos_rewards_list)
            if reward == 0:
                reward = random.random()
            reward_list.append(reward)
            if mode == 'infer':
                print("output prompt to be:", hypo)
                print("this prompt predict res :", hypos_rewards_list)
                print("avg_score :", reward_list)
                return torch.tensor(reward_list)
                # sum_rewards, content_scores, style_probs = \
                #     self.selector.compute_sample_rewards(src, hypos, label)4
            # avg_reward_to_this_img = sum(reward_list) / len(reward_list)
            # if avg_reward_to_this_img == 0:
            #     avg_reward_to_this_img = random.random()
            # rewards.append(avg_reward_to_this_img)
        rewards = reward_list
        rewards_tensor_list = [torch.tensor(scalar) for scalar in rewards]

        rewards_tensor = torch.stack(rewards_tensor_list)
        rewards_tensor = rewards_tensor.float()

        mean_value = rewards_tensor.mean()
        std_value = rewards_tensor.std()

        normalized_rewards = (rewards_tensor - mean_value) / std_value
        if to_tensor is True:
            return normalized_rewards
        else:
            return normalized_rewards.tolist()

    # def _compute_iou_reward(self,
    #                         rewards_tensor:torch.Tensor
    # ) -> torch.Tensor:
    #
    #     return rewards_tensor

    def _repeat_texts(
            self,
            texts: List[str],
            num_repeats: Optional[int] = None
    ) -> List[str]:
        if num_repeats is None:
            num_repeats = self.num_repeats
        return list(itertools.chain(*[[s for _ in range(num_repeats)]
                                      for s in texts]))

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    # calculate of IoU
    def convert_to_binary_mask(self, goroundtrue):
        # 将 goroundtrue 图片转换为二维数组，以 bool 值表示是否异常
        binary_mask = np.any(goroundtrue != [255, 255, 255], axis=-1)
        return binary_mask

    def calculate_iou(self, goroundtrue, mask_array):
        goroundtrue_array = goroundtrue

        # mask_array = mask.squeeze().numpy()
        # mask_array = ~mask_array
        false_ratio = np.mean(mask_array == False)
        if np.all(goroundtrue_array == False) and false_ratio > 0.99:
            return 1

        intersection = np.logical_and(goroundtrue_array, mask_array).sum()
        union = np.logical_or(goroundtrue_array, mask_array).sum()
        iou = intersection / union
        return iou

    # do promt to segment ,and return the iou between groundtrue and predict image
    # def prompt_to_seg(self,text_prompt,image_pil,image_gt):
    #     model = LangSAM()
    #     array = np.array(image_gt)
    #     gt_mask = array > 128
    #
    #     masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    #     iou = self.calculate_iou(gt_mask, masks)
    #     return iou

    # do promt to segment ,and return the iou between groundtrue and predict image
    # def prompts_to_seg(self,
    #                    text_prompts: List[str],
    #                    image_pil: torch.Tensor,
    #                    image_gt: torch.Tensor
    #                    ) -> List[float]:
    #     # res = []
    #     # for i in range(len(text_prompts)):
    #     #     res.append(random.random())
    #     # return res
    #     array = np.array(image_gt)
    #     gt_mask = array > 128
    #     max_iou = 0
    #     reward_prompt = text_prompts[0]
    #     image = Image.fromarray(image_pil.numpy().astype('uint8'))
    #     iou_list = []
    #     for text_prompt in text_prompts:
    #         masks, _, _, _ = self.model.predict(image, text_prompt)
    #         if masks.numel() == 0:
    #             iou_list.append(0)
    #             continue
    #         iou = self.calculate_iou(gt_mask, masks)
    #         iou_list.append(round(iou*100, 2))
    #         # if iou >max_iou:
    #         #     max_iou = iou
    #         #     reward_prompt = text_prompt
    #
    #     return iou_list

    def prompts_to_seg(self,
                       text_prompt: str,
                       src: str,
                       image_pil: torch.Tensor,
                       image_gt: torch.Tensor,
                       file_name: str,
                       dataset: Optional[str],
                       mode: str,
                       ) -> float:

        array = np.array(image_gt)
        gt_mask = array > 128
        image = Image.fromarray(image_pil.numpy().astype('uint8'))
        # iou_list = []
        masks, _, _, _ = self.model.predict(image, text_prompt)

        # situation of lang_sam predict can't generate any mask
        if masks.numel() == 0:
            print("predict fialled in generate mask,text prompt is ", text_prompt)
            print("image name is ", file_name)
            text_prompt = "find anomaly" + src
            return self.prompts_to_seg(text_prompt, src, image_pil, image_gt, file_name, dataset, mode)

        # situation of lang_sam predict generate multi masks
        if masks.dim() == 3:
            masks = masks[0:1]
        int_tensor_data = masks.to(dtype=torch.uint8) * 255
        mask_array = int_tensor_data.squeeze().numpy()
        mask_array = ~mask_array

        # saving predict img
        try:
            if dataset != '-1':
                save_path = os.path.join('./data', dataset, 'predict_folder', mode + '_res')
                temp = Image.fromarray(mask_array, mode='L')
                temp.save(os.path.join(save_path, file_name))
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An exception occurred when saving output image: {e}")
            print("masks shape to be :", masks.shape)
            print("text_prompt to be :", text_prompt)
            print("img_name to be :", file_name)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        iou = self.calculate_iou(gt_mask, mask_array)
        return round(iou * 100, 2)

    # def prompts_to_seg_infer(self,
    #                          text_prompt: str,
    #                          image_pil: torch.Tensor,
    #                          image_gt: torch.Tensor,
    #                          file_name: str,
    #                          ) -> float:
    #     save_path = './data/ksdd2/predict_folder/infer_res'
    #     array = np.array(image_gt)
    #     gt_mask = array > 128
    #     image = Image.fromarray(image_pil.numpy().astype('uint8'))
    #     masks, _, _, _ = self.model.predict(image, text_prompt)
    #     if masks.numel() == 0:
    #         print("predict fialled in generate mask,text prompt is ", text_prompt)
    #         print("image name is ", file_name)
    #         return 0
    #     try:
    #         if masks.dim() == 3:
    #             masks = masks[0:1]
    #         int_tensor_data = masks.to(dtype=torch.uint8) * 255
    #         mask_array = int_tensor_data.squeeze().numpy()
    #         mask_array = ~mask_array
    #         temp = Image.fromarray(mask_array, mode='L')
    #         # text_prompt = text_prompt.replace('?', ' question').replace('.', ' dot')
    #         # f_name = file_name.split('.')[0] + text_prompt + '.png'
    #         temp.save(os.path.join(save_path, file_name))
    #     except Exception as e:
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #         print(f"An exception occurred when saving output image: {e}")
    #         print("masks shape to be :", masks.shape)
    #         print("text_prompt to be :", text_prompt)
    #         print("img_name to be :", file_name)
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #
    #     goroundtrue_array = gt_mask
    #     mask_array = masks.squeeze().numpy()
    #     mask_array = ~mask_array
    #     false_ratio = np.mean(mask_array==False)
    #     if false_ratio > 0.98 and np.all(goroundtrue_array == False):
    #         return 100.00
    #     intersection = np.logical_and(goroundtrue_array, mask_array).sum()
    #     union = np.logical_or(goroundtrue_array, mask_array).sum()
    #     iou = intersection / union
    #
    #     return round(iou*100, 2)

    def calculate_f1_score(self,
                           text_prompt: str,
                           src: str,
                           image_pil: torch.Tensor,
                           image_gt: torch.Tensor,
                           file_name: str,
                           dataset: Optional[str],
                           mode: str,
                           ) -> float:
        image_array = image_pil.numpy().astype('uint8')
        image = Image.fromarray(image_array)
        # iou_list = []
        masks, _, _, _ = self.model.predict(image, text_prompt)

        if masks.numel() == 0:
            print("predict fialled in generate mask,text prompt is ", text_prompt)
            print("image name is ", file_name)
            text_prompt = "find anomaly" + src
            return self.prompts_to_seg(text_prompt, src, image_pil, image_gt, file_name, dataset, mode)

        # situation of lang_sam predict generate multi masks
        if masks.dim() == 3:
            masks = masks[0:1]
        int_tensor_data = masks.to(dtype=torch.uint8) * 255
        mask_array = int_tensor_data.squeeze().numpy()
        mask_array = ~mask_array
        mask_array[mask_array != 0] = 1
        gt_array = image_gt.numpy().astype('uint8')
        gt_array[gt_array != 0] = 1

        assert gt_array.shape == mask_array.shape, "Shapes of y_true and y_pred must be the same"
        # Calculate True Positive (TP)
        TP = np.sum(np.logical_and(gt_array, mask_array))
        # Calculate False Positive (FP)
        FP = np.sum(np.logical_and(np.logical_not(gt_array), mask_array))
        # Calculate False Negative (FN)
        FN = np.sum(np.logical_and(gt_array, np.logical_not(mask_array)))
        # Calculate Precision
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        # Calculate Recall
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        # Calculate F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        try:
            if dataset != '-1':
                save_path = os.path.join('./data', dataset, 'predict_folder', mode + '_res')
                temp = Image.fromarray(mask_array, mode='L')
                temp.save(os.path.join(save_path, file_name))
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An exception occurred when saving output image: {e}")
            print("masks shape to be :", masks.shape)
            print("text_prompt to be :", text_prompt)
            print("img_name to be :", file_name)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return f1_score

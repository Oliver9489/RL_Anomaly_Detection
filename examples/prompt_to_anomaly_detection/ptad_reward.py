import os.path
import random
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
from collections import defaultdict

from examples.prompt_to_anomaly_detection.ptad_modules.metrics import metric_cal
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
            # style_classifier: str,
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
        # if style_tokenizer is None:
        # style_tokenizer = style_classifier
        # self.selector = TextStyleTransferOutputSelector(style_classifier,
        #                                                 style_tokenizer,
        #                                                 style_batch_size,
        #                                                 reward_device)

        # Misc. training details
        self.num_repeats = num_repeats
        self.compute_zscore = compute_zscore
        self._counter = 0
        self.tokens_explored = set()

        # extend config
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
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == 'train':
            self._counter += 1
            source_strs = self._repeat_texts(source_texts)
        elif mode == "infer":
            return self.evaluate(source_img, GT_img, file_names, source_texts, output_tokens,
                                 to_tensor, mode, dataset)
        else:
            raise ValueError

        assert len(output_tokens) == len(source_strs)

        # predict_mask_list is a list contain all predict mask of mutil prompt to one image value [0,1]
        predict_mask_list = [[] for _ in output_tokens]
        rewards = []
        for (source, gt, file_name) in zip(source_img, GT_img, file_names):
            one_prompt_predict_masks = self.generate_predict_mask(output_tokens,
                                                                  source_strs,
                                                                  source,
                                                                  gt,
                                                                  file_name,
                                                                  mode
                                                                  )
            for (l, m) in zip(predict_mask_list, one_prompt_predict_masks):
                l.append(m)
        gt_label = []
        for gt in GT_img:
            if np.all(gt.numpy().astype('uint8') == 0):
                gt_label.append(0)
            else:
                gt_label.append(1)

        for predict_mask in predict_mask_list:
            result_dict = metric_cal(np.array(predict_mask), gt_label, np.array(GT_img))
            # print("cal_res,f1:", result_dict['p_f1'])
            if result_dict['p_f1'] == 0.0 or result_dict['p_f1'] is None:
                rewards.append(random.random())
            else:
                rewards.append(result_dict['p_f1'])
        # loss = (sum(rewards)/len(rewards))*2-100.0
        # loss = torch.tensor(loss, requires_grad=True).float()
        # in order to prevent nothing return back
        # rewards[rewards == 0.0] = random.random()
        if dataset == None:
            dataset = '-1'
        else:
            ex_path = os.path.join("./data", dataset, "tokens_reward.csv")
            existing_data = pd.read_csv(ex_path)
            assert len(output_tokens) == len(rewards)
            new_data = {
                "name": [file_names[0]] * len(output_tokens),
                "tokens": output_tokens,
                "reward": rewards
            }
            new_data_df = pd.DataFrame(new_data)
            combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)
            combined_data.to_csv(ex_path, index=False)

        print(f"epoch {self._counter} generate res :", rewards)
        rewards_tensor_list = [torch.tensor(scalar) for scalar in rewards]

        rewards_tensor = torch.stack(rewards_tensor_list)
        rewards_tensor = rewards_tensor.float()

        if len(rewards_tensor) > 1:
            mean_value = rewards_tensor.mean()
            std_value = rewards_tensor.std()
            if std_value == 0:
                normalized_rewards = rewards_tensor - 50.0 / 50.0
            else:
                normalized_rewards = (rewards_tensor - mean_value) / std_value
        else:
            normalized_rewards = rewards_tensor - 50.0 / 50.0
        print(f"epoch {self._counter} return :", normalized_rewards)
        if to_tensor is True:
            return normalized_rewards
        else:
            return normalized_rewards.tolist()
        # return loss

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
        converted_strings = []
        for token_list in tokens:
            readable_str = self.tokenizer.convert_tokens_to_string(token_list)
            converted_strings.append(readable_str)
        return converted_strings
        # return [self.tokenizer.convert_tokens_to_string(s)
        #         for s in tokens]

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

        masks = masks[0:1]
        int_tensor_data = masks.to(dtype=torch.uint8) * 255
        mask_array = int_tensor_data.squeeze().numpy()

        # saving predict img
        self.save_predict_array(dataset, mode, mask_array, file_name)

        iou = self.calculate_iou(gt_mask, mask_array)
        return round(iou * 100, 2)

    def calculate_accuracy(self,
                           text_prompt: str,
                           src: str,
                           image_pil: torch.Tensor,
                           image_gt: torch.Tensor,
                           file_name: str,
                           dataset: Optional[str],
                           mode: str,
                           is_regenerate: bool = False
                           ) -> float:
        image_array = image_pil.numpy().astype('uint8')
        image = Image.fromarray(image_array)
        # iou_list = []
        masks, _, _, _ = self.model.predict(image, text_prompt)

        if masks.numel() == 0:
            print("predict fialled in generate mask,text prompt is ", text_prompt)
            print("image name is ", file_name)
            text_prompt = "find anomaly" + src
            if not is_regenerate:
                return self.calculate_accuracy(text_prompt, src, image_pil, image_gt, file_name, dataset, mode, True)
            else:
                return 0.0
        # situation of lang_sam predict generate multi masks
        if masks.dim() == 3:
            masks = masks[0:1]
        int_tensor_data = masks.to(dtype=torch.uint8) * 255
        mask_array = int_tensor_data.squeeze().numpy()

        gt_array = image_gt.numpy().astype('uint8')
        gt_array[gt_array != 0] = 255

        # saving predict img
        self.save_predict_array(dataset, mode, gt_array, mask_array, file_name)

        # mask_array = ~mask_array
        mask_array[mask_array != 0] = 1
        gt_array[gt_array != 0] = 1

        assert gt_array.shape == mask_array.shape, "Shapes of y_true and y_pred must be the same"
        # Calculate True Positive (TP)
        TP = np.sum(np.logical_and(gt_array, mask_array))
        # Calculate False Positive (TN)
        TN = np.sum(np.logical_and(np.logical_not(gt_array), np.logical_not(mask_array)))

        total_samples = gt_array.size
        # Calculate Accuracy
        accuracy = (TP + TN) / total_samples
        return accuracy

    def save_predict_array(self, dataset, mode, gt_array, mask_array, file_name):
        try:
            if dataset != '-1':
                save_path = os.path.join('./data', dataset, 'predict_folder', mode + '_res')
                gt_save_path = os.path.join('./data', dataset, 'predict_folder', mode + '_gt')
                temp = Image.fromarray(mask_array, mode='L')
                temp.save(os.path.join(save_path, file_name))
                gt_temp = Image.fromarray(gt_array, mode='L')
                gt_temp.save(os.path.join(gt_save_path, file_name))

        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An exception occurred when saving output image: {e}")
            # print("masks shape to be :", masks.shape)
            # print("text_prompt to be :", text_prompt)
            print("img_name to be :", file_name)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def generate_predict_mask(self,
                              prompt_strs: List[List[str]],
                              source_strs: List[str],
                              source_img: torch.Tensor,
                              GT_img: torch.Tensor,
                              file_name: str,
                              mode: str
                              ) -> List[np.ndarray]:
        save_path = './data/ksdd2/predict_folder'
        predict_mask_list = []
        img = Image.fromarray(source_img.numpy().astype('uint8'))
        GT = GT_img
        for (prompt_list, src) in zip(prompt_strs, source_strs):
            prompt = ' '.join(prompt_list)
            masks, _, _, _ = self.model.predict(img, prompt)
            if masks.numel() == 0:
                # deal with mask no generate situation
                log_info = "no anomaly detect from " + file_name + ".prompt:" + prompt
                print(log_info)
                mask_array = np.zeros_like(GT_img)
            else:
                masks = masks[0:1]
                int_tensor_data = masks.to(dtype=torch.int) * 255
                mask_array = int_tensor_data.squeeze().numpy()
                if np.count_nonzero(mask_array == 255) / mask_array.size > 0.9:
                    log_info = "no anomaly detect from " + file_name + ".prompt:" + prompt
                    print(log_info)
                    mask_array = np.zeros_like(GT_img)
                else:
                    log_info = "anomaly detect from " + file_name + ".prompt:" + prompt
                    print(log_info)
            predict_mask_list.append(mask_array)

        res_path = os.path.join(save_path, mode + "_res")
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        gt_path = os.path.join(save_path, mode + "_gt")
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        # save predict mask
        for i, mask in enumerate(predict_mask_list):
            path = os.path.join(res_path, file_name.split('.')[0] + f'_{i}' + ".png")
            Image.fromarray(mask).convert('L').save(path)
        path = os.path.join(gt_path, file_name)
        Image.fromarray(GT.numpy().astype('uint8')).save(path)

        predict_mask_list = [(arr.astype(float) / 255) for arr in predict_mask_list]
        return predict_mask_list

    def evaluate(
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
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == "infer":
            source_strs = [source_texts[0] for _ in range(len(output_tokens))]
        else:
            raise ValueError

        prompt_strs = output_tokens
        assert len(prompt_strs) == len(source_strs)

        predict_mask_list = []

        for (source, gt, file_name) in zip(source_img, GT_img, file_names):
            one_prompt_predict_masks = self.generate_predict_mask(prompt_strs,
                                                                  source_strs,
                                                                  source,
                                                                  gt,
                                                                  file_name,
                                                                  mode
                                                                  )
            predict_mask_list.append(one_prompt_predict_masks)

        gt_label = []
        for gt in GT_img:
            if np.all(gt.numpy().astype('uint8') == 0):
                gt_label.append(0)
            else:
                gt_label.append(1)
        rewards = []

        result_dict = metric_cal(np.array(predict_mask_list), gt_label, np.array(GT_img))
        if result_dict['p_f1'] == 0.0 or result_dict['p_f1'] is None:
            rewards.append(random.random())
        else:
            rewards.append(result_dict['p_f1'])

        if dataset == None:
            dataset = '-1'
        else:
            ex_path = os.path.join("./data", dataset, "eval_tokens_reward.csv")
            existing_data = pd.read_csv(ex_path)
            assert len(prompt_strs) == len(rewards)
            new_data = {
                "name": file_names,
                "tokens": prompt_strs * len(file_names),
                "gt_label": gt_label,
            }
            new_data_df = pd.DataFrame(new_data)
            combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)
            combined_data.to_csv(ex_path, index=False)

        rewards_tensor_list = [torch.tensor(scalar) for scalar in rewards]
        rewards_tensor = torch.stack(rewards_tensor_list)
        rewards_tensor = rewards_tensor.float()
        print("this epoch return :", prompt_strs, rewards)
        return rewards_tensor

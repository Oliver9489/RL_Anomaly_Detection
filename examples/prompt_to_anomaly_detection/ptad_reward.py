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
        source_texts: List[str],
        # target_labels: List[str],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == 'train':
            self._counter += 1
            source_strs = self._repeat_texts(source_texts)
            # target_labels = self._repeat_texts(target_labels)
        elif mode == "infer":
            source_strs = source_texts
        else:
            raise ValueError

        prompt_tokens = output_tokens
        # convert the output tokens to string. meaning this is the output prompt that we can use
        # and its a list consist of string
        prompt_strs = self._convert_tokens_to_string(prompt_tokens)
        assert len(prompt_strs) == len(source_strs)

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward

        rewards: List[float] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        for (source, GT) in zip(source_img, GT_img):
            reward_list = []
            best_prompt_list = []
            # target_labels = ['0', '0']
            for i, (prompt, src) in enumerate(zip(prompt_strs, source_strs)):
                hypos = self.generator.sample_generate(prompt, src, N,
                                                       self.top_k, self.top_p)
                # iou = self.prompt_to_seg(hypos,source_imgs,GT_imgs)
                max_iou,reward_prompt = self.prompts_to_seg(hypos,source,GT)
                reward_list.append(max_iou)
                best_prompt_list.append(reward_prompt)
                # sum_rewards, content_scores, style_probs = \
                #     self.selector.compute_sample_rewards(src, hypos, label)
            rewards.append(sum(reward_list) / len(reward_list))
        rewards_tensor_list = [torch.tensor(scalar) for scalar in rewards]
        print(rewards)
        print(rewards_tensor_list)
        rewards_tensor = torch.stack(rewards_tensor_list)
        if to_tensor is True:
            return rewards_tensor
        else:
            return rewards_tensor.tolist()

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
    def convert_to_binary_mask(self,goroundtrue):
        # 将 goroundtrue 图片转换为二维数组，以 bool 值表示是否异常
        binary_mask = np.any(goroundtrue != [255, 255, 255], axis=-1)
        return binary_mask
    def calculate_iou(self,goroundtrue, mask):
        goroundtrue_array = goroundtrue
        print(type(mask))
        mask_array = mask.squeeze().numpy()
        mask_array = ~mask_array
        # m = Image.fromarray(mask_array)
        # m.save("output.png")
        if np.all(goroundtrue_array == False) and np.all(mask_array == False):
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
    def prompts_to_seg(self,
                       text_prompts: List[str],
                       image_pil: torch.Tensor,
                       image_gt: torch.Tensor
                       ) -> float:
        array = np.array(image_gt)
        gt_mask = array > 128
        max_iou = 0
        reward_prompt = text_prompts[0]
        image = Image.fromarray(image_pil.numpy().astype('uint8'))
        for text_prompt in text_prompts:
            masks, _, _, _ = self.model.predict(image, text_prompt)
            if masks.numel() == 0:
                continue
            print(masks)
            print(type(masks))

            #将布尔值的Tensor转换为0和255的整数Tensor
            int_tensor_data = masks.to(dtype=torch.uint8) * 255
            # 将整数Tensor转换为PIL Image
            mask = Image.fromarray(int_tensor_data.squeeze().numpy(), mode='L')
            inverted_image = mask.point(lambda x: 255 - x)
            inverted_image.save("output2.png")
            mask.save("output.png")

            # image.show()
            image.save("image.png")

            print(gt_mask)
            print(type(gt_mask))

            gt = Image.fromarray(gt_mask)
            # gt.show()
            gt.save("gt.png")

            iou = self.calculate_iou(gt_mask, masks)
            if iou >max_iou:
                max_iou = iou
                reward_prompt = text_prompt

        return round(max_iou*100, 2),reward_prompt
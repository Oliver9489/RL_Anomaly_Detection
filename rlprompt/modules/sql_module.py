import os

import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.losses import sql_loss_with_sparse_rewards
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anomaly_detection_dataset = ["ksdd2"]

class SQLModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        target_model: Optional[BaseModel],
        reward: Optional[BaseReward],
        sql_loss_impl: str,
        training_mode: str,
        mix_strategy: Optional[str],
        target_update_method: str,
        target_update_steps: Optional[int],
        target_learning_rate: float,
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        dataset: Optional[str],
        activate: int = 0,
    ):
        super().__init__()
        # Initialize self._model and self._reward
        assert target_update_method in ["copy", "polyak"]
        assert not (top_k is not None and top_p < 1.0), \
               "Only one of top_k or top_p should be selected"

        self._model = model
        if target_model is None:
            self._target_model = copy.deepcopy(self._model)
        else:
            self._target_model = target_model
        # for p1, p2 in zip(self._model.parameters(), self._target_model.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         print(False)
        #     print(True) 
        self._reward = reward

        self._sql_loss_impl = sql_loss_impl
        self._training_mode = training_mode
        self._mix_strategy = mix_strategy
        self._forward_modes = _get_forward_modes(training_mode, mix_strategy)
        self._target_update_method = target_update_method
        self._target_update_steps = target_update_steps
        self._target_learning_rate = target_learning_rate
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams
        self._dataset = dataset
        self._activate = activate
        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

    def _sync_target_model(self) -> None:
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if self._target_update_method == "copy":
            self._target_model.load_state_dict(self._model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._target_update_method == "polyak":
            for param_, param in zip(self._target_model.parameters(),
                                     self._model.parameters()):
                param_.data.copy_((1 - self._target_learning_rate) * param_
                                  + self._target_learning_rate * param)

    def _pre_steps(self, step: int) -> None:
        if self._target_update_method == "polyak":
            self._sync_target_model()
        elif self._target_update_method == "copy" \
                and step % self._target_update_steps == 0:
            self._sync_target_model()


    def forward(self, batch: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        # loss_log_list = []
        if self._dataset in anomaly_detection_dataset:
            full_filepath = os.path.join("./data", self._dataset, self._dataset+".tsv")
            with open(full_filepath) as f:
                source_text = [line.strip() for line in f]
            for mode in self._forward_modes:
                _loss = self._forward(mode=mode, batch=batch, source_text=source_text)
                loss_list.append(_loss)
        else:
            for mode in self._forward_modes:
                _loss = self._forward(mode=mode, batch=batch)
                loss_list.append(_loss)
            # loss_log_list.append(_loss_log)

        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        # loss_log = utils.unionize_dicts(loss_log_list)

        return loss

    def _forward(
        self,
        mode: ForwardMode,
        batch: Dict[str, Any],
        source_text: Optional[List[str]],
    ) -> Tuple[torch.Tensor, Dict]:
        if mode != ForwardMode.SQL_ON and mode != ForwardMode.INFER:
            # TODO: Enable training modes other than on-policy
            raise NotImplementedError('Only on-policy sampling and greedy '
                                      'inference is supported now')

        if mode == ForwardMode.SQL_ON :
            if source_text != None :
                (logits, logits_, output_tokens, output_ids, sequence_lengths) = \
                    self._decode_sampling(batch=batch, source_text=source_text)
            else:
                (logits, logits_, output_tokens, output_ids, sequence_lengths) = \
                    self._decode_sampling(batch=batch)

            raw_rewards = \
                self.compute_rewards(batch=batch, source_texts=source_text, output_tokens=output_tokens, mode="train")

        shaped_rewards = self._reward_shaping_func(raw_rewards)

        sql_loss, sql_loss_log = sql_loss_with_sparse_rewards(
            implementation=self._sql_loss_impl,
            logits=logits,
            logits_=logits_,
            actions=output_ids,
            sampled_actions=None,
            rewards=shaped_rewards,
            sequence_length=sequence_lengths)

        # utils.add_prefix_to_dict_keys_inplace(
        #     rewards_log, prefix=f"{mode.value}/rewards/")
        # utils.add_prefix_to_dict_keys_inplace(
        #     sql_loss_log, prefix=f"{mode.value}/")
        # sql_loss_log = utils.unionize_dicts([
        #     rewards_log,
        #     sql_loss_log,
        #     {
        #         f"{mode.value}/rewards/raw": raw_rewards.mean(),
        #         f"{mode.value}/rewards/shaped": shaped_rewards.mean(),
        #     },
        # ])

        return sql_loss

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        source_texts: List[str],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # output_tokens is a list that consist of the output texts
        rewards_tensor = self._reward(
            **batch,
            source_texts=source_texts,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode,
            dataset=self._dataset,
            tst_active=self._activate,
        )

        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor

    def infer(
        self,
        batch: Dict[str, Any],
        source_texts: List[str],
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:

        return self._model.generate(**batch,
                                    source_texts = source_texts,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    def _decode_sampling(
        self,
        batch: Dict[str, Any],
        source_text: Optional[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        if source_text != None:
            outputs = self._model.generate(**batch,
                                           source_texts=source_text,
                                           do_sample=True,
                                           top_k=self._top_k,
                                           top_p=self._top_p,
                                           num_beams=self._num_beams)
        else:
            outputs = self._model.generate(**batch,
                                           do_sample=True,
                                           top_k=self._top_k,
                                           top_p=self._top_p,
                                           num_beams=self._num_beams)

        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)

        # in text_style_transfer target_model refer to None,in the other word,outputs come from self._model.generate()
        outputs_ = self._target_model.teacher_forcing(**batch_,
                                                      source_texts=source_text,
                                                      )

        return (outputs['sample_logits'].contiguous(),
                outputs_['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())


def _get_forward_modes(
    training_mode: str,
    mix_strategy: Optional[str]
) -> List[ForwardMode]:
    if training_mode == "sql-mixed":
        candidate_modes = [
            ForwardMode.SQL_OFF_GT,
            ForwardMode.SQL_ON]

        if mix_strategy == "alternate":
            modes = [candidate_modes[step % len(candidate_modes)]]
        elif mix_strategy == "mix":
            modes = candidate_modes

    else:
        training_mode_map = {"sql-onpolicy": ForwardMode.SQL_ON,
                             "sql-offpolicy": ForwardMode.SQL_OFF_GT}

        modes = [training_mode_map[training_mode]]

    return modes

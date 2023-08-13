import math
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List
import os
import wandb
import json
import click

from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import get_default_train_op, set_random_seed


class Trainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
            self,
            module: BaseModule,
            dataset: str,
            # Train params
            train_dataset: Optional[Dataset],
            train_batch_size: int,
            train_shuffle: bool,
            train_drop_last: bool,
            num_train_epochs: int,
            max_train_steps: int,
            # Eval params
            do_eval: bool,
            eval_dataset: Optional[Dataset],
            eval_batch_size: int,
            eval_steps: int,
            # Save params
            do_save: bool,
            save_dir: str,
            save_steps: int,
            # Optimizer params
            learning_rate: float,
            gradient_clip: bool,
            gradient_clip_norm: float,
            # Checkpoint params
            checkpoint_path: Optional[str],
            # Random seed
            random_seed: Optional[int],
            # Wandb reporting
            report_to_wandb: bool,
            project_name: Optional[str],
            run_name: Optional[str]
    ):
        assert do_eval == False or eval_dataset is not None, \
            "Need to have eval_dataset if do_eval is True"
        self.module = module
        self._dataset = dataset

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps

        self.train_op = get_default_train_op(self.module._model,
                                             learning_rate,
                                             gradient_clip,
                                             gradient_clip_norm)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        self.run_name = run_name

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    # @torch.no_grad
    def _train_step(
            self,
            step: int,
            batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.module.train()
        model._pre_steps(step)
        # modify = math.ceil(100.0 / (step+1))
        # where sql_module were called
        loss = model(batch)
        # loss = loss * modify
        print(step + 1, '-------loss:', loss)
        loss[torch.isnan(loss)] = random.random() * 1000
        # nan_mask = torch.isnan(loss)
        # loss = torch.where(nan_mask, torch.tensor(1.0), loss)
        loss.backward()
        self.train_op()

        return

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None,
              train_activate: int = 0,
              eval_activate: int = 0,
              ) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None:
            run_name = self.run_name
        if config is not None:
            config = eval(str(config))
        if report_to_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
            wandb.watch(self.module, log=None)

        # Create saving path
        # output path for pth file and json file
        # record the trainning process
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        train_dataloader = self._get_train_dataloader()

        # do training
        # here we only let model training through the entire train dataset and eval in test dataset
        # and didn't set any other limited
        if train_activate:
            print("start training.......................................")
            if self.num_train_epochs == 1:
                for step, batch in enumerate(train_dataloader):
                    self._train_step(step, batch)
                print("################training Finished####################\n")
                print("saving model.........................................")
                torch.save({"steps": 1,
                            "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir,
                                        f"ckpt.model_train.pth"))
                print("model_savepath:", os.path.join(ckpt_save_dir, f"ckpt.model_train.pth"))
                print("##################save Finished######################")
            else:  # more epoch trainning
                for epoch in range(self.num_train_epochs):
                    for step, batch in enumerate(train_dataloader):
                        self._train_step(step, batch)
                    print(f"################Finished epoch {epoch}####################\n")
                    if epoch % 10 == 0:
                        print("saving model.........................................")
                        torch.save({"steps": epoch,
                                    "model_state_dict": self.module.state_dict()},
                                   os.path.join(ckpt_save_dir,
                                                f"ckpt.model_train_{epoch}.pth"))
                        print("model_savepath:", os.path.join(ckpt_save_dir, f"ckpt.model_train_{epoch}.pth"))
                        print("##################save Finished######################")

        if eval_activate:
            output_save_path = "./result"
            print("do eval.......................................")
            eval_log = self.evaluate(output_save_path=output_save_path)
            output_save_path = os.path.join(eval_save_dir,
                                            f'outputs.eval.json')
            print("eval_log_savepath:", output_save_path)
            eval_log = self.evaluate(output_save_path=output_save_path)
            wandb.log(eval_log)
            print("##################eval Finished######################")

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset,
                          batch_size=self.eval_batch_size)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            output_save_path: Optional[str] = None,
            compute_scores: bool = True
    ) -> Dict[str, np.number]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        full_filepath = os.path.join("./data", self._dataset, self._dataset + '.tsv')
        with open(full_filepath) as f:
            source_text = [line.strip() for line in f]
            # source_text = [source_text[0] for _ in range(self.eval_batch_size)]
        eval_dataloader = self._get_eval_dataloader(eval_dataset)

        model = self.module.eval()
        hypos = []
        scores: List[List[str]] = []
        for batch in eval_dataloader:
            infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
            infer_outputs = model.infer(batch, source_texts=source_text)
            hypos += infer_outputs['sample_tokens']

            score = model.compute_rewards(
                batch=batch,
                source_texts=source_text,
                output_tokens=infer_outputs['sample_tokens'])
            scores += score.detach().tolist()
        # print('avg accuracy:', sum(scores)/len(scores))
        print(scores)
        if output_save_path is not None:
            print("output_save_path:", output_save_path)
            json.dump({'output_tokens': hypos,
                       'scores': scores},
                      open(output_save_path, 'w'))

        # score = score.mean().item()

        # utils.add_prefix_to_dict_keys_inplace(
        #     score_log,
        #     prefix=f"eval/rewards/")

        return utils.unionize_dicts([
            # gem_scores_dict,
            {
                f"eval/score": score,
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ])

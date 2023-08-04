from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from ptad_helpers import (PromptedTextStyleTransferRewardConfig,
                          TextStyleTransferDatasetConfig,
                          make_prompted_text_style_transfer_reward,
                          make_anomaly_detection_datasets,
                          get_style_classifier)

# Compose default config
config_list = [PromptedTextStyleTransferRewardConfig,
                TextStyleTransferDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_ptad', config_list)


@hydra.main(version_base=None, config_path="./", config_name="ptad_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    # the output dir "./outputs/2023-07-27(date)/21-36-52(time)"
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_anomaly_detection_datasets(config)
    # print('Train Size:', len(train_dataset))
    # print(type(train_dataset))
    # print('Examples:', train_dataset[:5])
    # print('Val Size', len(val_dataset))
    # print('Examples:', val_dataset[:5])
    # print('Examples:', test_dataset[:5])
    # print('Val Size:', len(test_dataset))
    # creat a adaptor,place to edit MLP
    policy_model = make_lm_adaptor_model(config)

    # prompt model for algo_module input ,dont know of its function yet
    # update: t
    prompt_model = make_single_prompt_model(policy_model, config)

    # a classifier can get from helpers.
    # using shakespeare to train for prompt style transfer
    config.style_classifier = get_style_classifier('train', config)

    # this function return class of a BaseReward,
    # and its the class that we rewrite the mechanism of reward
    reward = make_prompted_text_style_transfer_reward(config)

    # this is where the reward were called
    # (in function compute_rewards().it called forward() of reward )
    # algo_modeul is BaseModule
    algo_module = make_sql_module(prompt_model, reward, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module,train_dataset,test_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()

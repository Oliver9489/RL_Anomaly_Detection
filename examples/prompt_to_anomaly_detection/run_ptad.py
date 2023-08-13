import torch
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
                          make_prompted_to_anomaly_detection_reward,
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
    print('Train Size:', len(train_dataset))
    print(type(train_dataset))
    # print('Examples:', train_dataset[:3])
    # print('Examples:', test_dataset[:3])
    print('Val Size:', len(test_dataset))
    # step 1 multi to multi
    # creat a adaptor,place to edit MLP
    # policy_model = make_lm_adaptor_model(config)
    # # prompt model for algo_module input ,dont know of its function yet
    # prompt_model = make_single_prompt_model(policy_model, config, prompt_train_batch_size=10)
    #
    # # a classifier can get from helpers.
    # # using shakespeare to train for prompt style transfer
    # # not using right now
    # config.style_classifier = get_style_classifier('train', config)
    #
    # # this function return class of a BaseReward,
    # # and its the class that we rewrite the mechanism of reward
    # reward = make_prompted_to_anomaly_detection_reward(config, num_repeats=10)
    #
    # # this is where the reward were called
    # # (in function compute_rewards().it called forward() of reward )
    # algo_module = make_sql_module(prompt_model, reward, config)
    #
    # config.num_train_epochs = 1
    # config.save_dir = os.path.join(output_dir, config.save_dir)
    # trainer = make_trainer(algo_module, train_dataset, test_dataset, config, train_batch_size=12)
    # trainer.train(config=config, train_activate=1)


    # step 2 import nomaly image.   multi prompt  to multi image
    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config, prompt_train_batch_size=7)
    reward = make_prompted_to_anomaly_detection_reward(config, num_repeats=7)
    algo_module = make_sql_module(prompt_model, reward, config)
    ckpt_path = "./result/ckpt.model_train_multi_multi.pth"
    checkpoint = torch.load(ckpt_path)
    model_state_dict = checkpoint["model_state_dict"]
    algo_module.load_state_dict(model_state_dict)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    config.num_train_epochs = 1
    trainer = make_trainer(algo_module, train_dataset, test_dataset, config, train_batch_size=111)
    trainer.train(config=config, train_activate=1)




if __name__ == "__main__":
    main()

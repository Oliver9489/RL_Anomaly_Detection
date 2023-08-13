import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# from examples.prompt_to_anomaly_detection.data.ksdd2.predict_folder import ksdd2_eval
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
    # print('Train Size:', len(train_dataset))
    # print(type(train_dataset))
    # print('Examples:', train_dataset[:5])
    # print('Val Size', len(val_dataset))
    # print('Examples:', val_dataset[:5])
    # print('Examples:', test_dataset[:5])
    # print('Val Size:', len(test_dataset))
    # creat a adaptor,place to edit MLP

    policy_model = make_lm_adaptor_model(config)

    prompt_model = make_single_prompt_model(policy_model, config,
                                            prompt_infer_batch_size=1)

    config.style_classifier = get_style_classifier('train', config)

    reward = make_prompted_to_anomaly_detection_reward(config, num_repeats=1)

    # this is where the reward were called
    # (in function compute_rewards().it called forward() of reward )
    # algo_modeul is BaseModule
    algo_module = make_sql_module(prompt_model, reward, config, eval_activate=1)
    # load train pth
    # ckpt_path = "./outputs/2023-08-06/20-13-26/outputs/ckpt/ckpt.model_train.pth"

    # mutil to mutil
    # ckpt_path = "./outputs/2023-08-09/23-32-26/outputs/ckpt/ckpt.model_train.pth"
    # ckpt_path = "./result/ckpt.model_train_multi_multi.pth.pth"
    ckpt_path = "./result/ckpt.model_train_multi_multi.pth"
    checkpoint = torch.load(ckpt_path)
    model_state_dict = checkpoint["model_state_dict"]
    algo_module.load_state_dict(model_state_dict)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, test_dataset, config, eval_batch_size=251)
    trainer.train(config=config, eval_activate=1)
    # ksdd2_eval.eval()

if __name__ == "__main__":
    main()

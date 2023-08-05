## classifier download

classifier here is the text_style_transfer method we combine code in rl_prompt ,the following code is a example we use:

```bash
python scripts/download_tst_classifiers.py --model_name  "shakespeare-train-100-0"
```

use other classifier

```bash
python scripts/download_tst_classifiers.py \
    --model_name [yelp-train,
                  shakespeare-train-100-0,
                  shakespeare-train-100-1,
                  shakespeare-train-100-2]
```



## run training code

```
python run_ptad.py dataset="ksdd2" dataset_seed=0 direction="0_to_1" prompt_length=5 task_lm="distilgpt2"
```

### run in terminal

cd to the RL_Anomaly_Detection Folder first,then run the code as below:

```bash
python run_ptad.py dataset="ksdd2" dataset_seed=0 direction="0_to_1" prompt_length=5 task_lm="distilgpt2"
```

### or run in PyCharm

you can set parameters as below before training:

```python
dataset="ksdd2" dataset_seed=0 direction="0_to_1" prompt_length=5 task_lm="distilgpt2"
```


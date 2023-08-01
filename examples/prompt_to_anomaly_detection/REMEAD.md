classifier download

```python
python scripts/download_tst_classifiers.py --model_name  "shakespeare-train-100-0"
```

```python
python scripts/download_tst_classifiers.py \
    --model_name [yelp-train,
                  shakespeare-train-100-0,
                  shakespeare-train-100-1,
                  shakespeare-train-100-2]
```

run training code

```python
python run_ptad.py dataset="ksdd2" dataset_seed=0 direction="0_to_1" prompt_length=5 task_lm="distilgpt2"
```


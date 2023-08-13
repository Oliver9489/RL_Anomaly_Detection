# full_filepath = "data/one_shot/ksdd2.tsv"
# with open(full_filepath) as f:
#     source_text = [line.strip() for line in f]
#
# print(source_text)

# import torch
# t = torch.tensor([34.2580, -182.0170, 33.6741, 34.3752, 34.2580, 33.0934, -182.0170, 34.3752], device='cuda:0')
# random_tensor = torch.randn(4, 5, 50257)
#
# print(t.ndim)
# print(t.shape[0])
# print(random_tensor.shape[0])
#
# tensor = torch.randn(4, 5, 50259)
# print(tensor.dim())
# 将第一维扩展成8
# expanded_tensor = tensor.repeat(2, 1, 1)
#
# # 输出结果的形状为 (8, 5, 50259)
# print(expanded_tensor.size())
# import os
# import pandas as pd
#
# ex_path = os.path.join("./data", "ksdd2", "tokens_reward.csv")
# existing_data = pd.read_csv(ex_path)
# print(existing_data)

l1 = [9]
l = [1,2]*3
l1.extend(l)
print(l1)
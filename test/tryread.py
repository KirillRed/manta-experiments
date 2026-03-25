import numpy as np
import torch

tensor = torch.load("./test/sliced_sample.pt")

unique_values, counts = np.unique(tensor[1], return_counts=True)

counts_dict = dict(zip(unique_values, counts))

for i in range(25):
    print(i, tensor[i][tensor[i] == 11].sum() / 11)
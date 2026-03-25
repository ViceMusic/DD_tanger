# 该文件用于各种意义上的蒸馏检查

import numpy as np

path = "./datasets/distilled/mnist_simple_DM/syn_ipc=800.npz"

data = np.load(path)

print("Keys:", data.files)

for k in data.files:
    arr = data[k]
    print(f"\n[{k}]")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    print("  min/max:", arr.min(), arr.max())
# 蒸馏配置如下
# 模型：simple_cnn
# 数据获取：mnist_data
# 蒸馏类（DM）：simple_distill
# 蒸馏结果保存：datasets/distilled/{该实验名称}/ipc={n}.npz

"""Minimal distiller setup example."""
from __future__ import annotations
from torch.utils.data import DataLoader
from src.models import SimpleMNISTCNN
from src.utils.mnist_data import mnist_tensor
from src.distillers.simple_distill import DM
import numpy as np

# 代码配置如下
model = SimpleMNISTCNN()
batch_size=32
train_dataset, test_dataset, shape_info = mnist_tensor()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

ipcs=[10,20,30]

# 
for ipc_num in ipcs:
    dm = DM(
        model=model, train_dataset=train_dataset,
        num_classes=10, ipc=10,
        image_shape=(1, 28, 28),
        device=None,
        lr_img=1.0,
        batch_real=64,
        iters=1000,
    )

    result = dm.run()

    syn_images = result["images"]
    syn_labels = result["labels"]


    np.savez(
        f"./datasets/distilled/mnist_simple_DM/syn_ipc={ipc_num}.npz",# 以工作运行目录为标准
        images=syn_images.cpu().numpy(),
        labels=syn_labels.cpu().numpy(),
    )
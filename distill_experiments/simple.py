# 蒸馏配置如下
# 模型：simple_cnn
# 数据获取：mnist_data
# 蒸馏类（DM）：simple_distill
# 蒸馏结果保存：datasets/distilled/{该实验名称}/ipc={n}.npz

# 数据集：mnist

"""Minimal distiller setup example."""
from __future__ import annotations

import os
import time
import numpy as np
from torch.utils.data import DataLoader

from src.models import SimpleMNISTCNN
from src.utils.mnist_data import mnist_tensor
from src.distillers.simple_distill import DM


# =========================
# 基础配置
# =========================
model = SimpleMNISTCNN()
batch_size = 64

train_dataset, test_dataset, shape_info = mnist_tensor()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 大批量 IPC 循环：50 到 3000，间隔 50
ipcs = list(range(50, 3001, 50))

# 保存目录
save_dir = "./datasets/distilled/mnist_simple_DM"
os.makedirs(save_dir, exist_ok=True)


# =========================
# 工具函数：秒 -> 时分秒
# =========================
def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# 主循环
# =========================
total_start_time = time.time()

for idx, ipc_num in enumerate(ipcs, start=1):
    case_start_time = time.time()

    print(f"\n========== [{idx}/{len(ipcs)}] IPC={ipc_num} 开始 ==========")

    dm = DM(
        model=model,
        train_dataset=train_dataset,
        num_classes=10,
        ipc=ipc_num,
        image_shape=(1, 28, 28),
        device=None,
        lr_img=1.0,
        batch_real=64,
        iters=1000,
    )

    result = dm.run()

    syn_images = result["images"]
    syn_labels = result["labels"]

    save_path = os.path.join(save_dir, f"syn_ipc={ipc_num}.npz")

    np.savez(
        save_path,
        images=syn_images.detach().cpu().numpy(),
        labels=syn_labels.detach().cpu().numpy(),
    )

    case_end_time = time.time()
    case_elapsed = case_end_time - case_start_time

    print(f"[完成] IPC={ipc_num} 已结束")
    print(f"[耗时] IPC={ipc_num} 用时：{format_seconds(case_elapsed)}")
    print(f"[保存] 文件已保存到：{save_path}")

total_end_time = time.time()
total_elapsed = total_end_time - total_start_time

print("\n========== 全部实验结束 ==========")
print(f"总耗时：{format_seconds(total_elapsed)}")
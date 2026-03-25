from __future__ import annotations

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import SimpleMNISTCNN


# =========================
# 配置
# =========================
data_dir = "./datasets/distilled/mnist_simple_DM"
save_dir = "./results/mnist_random_baseline"
os.makedirs(save_dir, exist_ok=True)

batch_size = 64
epochs = 30
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

ipcs = list(range(50, 3001, 50))
random_repeat = 5
num_classes = 10


# =========================
# 时间格式
# =========================
def fmt(t):
    t = int(t)
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"


# =========================
# 测试集
# =========================
test_data = np.load(os.path.join(data_dir, "test_original.npz"))
test_x = torch.tensor(test_data["images"], dtype=torch.float32)
test_y = torch.tensor(test_data["labels"], dtype=torch.long)
if test_x.ndim == 3:
    test_x = test_x.unsqueeze(1)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =========================
# 原始训练集
# =========================
train_data = np.load(os.path.join(data_dir, "train_original.npz"))
train_x_all = torch.tensor(train_data["images"], dtype=torch.float32)
train_y_all = torch.tensor(train_data["labels"], dtype=torch.long)
if train_x_all.ndim == 3:
    train_x_all = train_x_all.unsqueeze(1)

# 分层索引
class_indices = {}
for c in range(num_classes):
    class_indices[c] = torch.where(train_y_all == c)[0]


# =========================
# 主循环
# =========================
all_results = []
total_start = time.time()

for i, ipc in enumerate(ipcs, start=1):
    case_start = time.time()

    print(f"\n========== [{i}/{len(ipcs)}] random ipc={ipc} ==========")

    repeat_results = []

    for r in range(random_repeat):
        print(f"  ---- repeat [{r+1}/{random_repeat}] ----")

        # 分层抽样
        sampled_idx = []
        for c in range(num_classes):
            idx = class_indices[c]
            perm = idx[torch.randperm(len(idx))[:ipc]]
            sampled_idx.append(perm)
        sampled_idx = torch.cat(sampled_idx, dim=0)

        train_x = train_x_all[sampled_idx]
        train_y = train_y_all[sampled_idx]

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = SimpleMNISTCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ===== 训练 =====
        for epoch in range(epochs):
            epoch_start = time.time()

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_correct += (out.argmax(dim=1) == y).sum().item()
                train_total += y.size(0)

            model.eval()
            test_loss, test_correct, test_total = 0.0, 0, 0

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)

                    test_loss += loss.item() * x.size(0)
                    test_correct += (out.argmax(dim=1) == y).sum().item()
                    test_total += y.size(0)

            print(
                f"    Epoch [{epoch+1}/{epochs}] | "
                f"train_acc={train_correct/train_total:.4f} | "
                f"test_acc={test_correct/test_total:.4f} | "
                f"time={fmt(time.time()-epoch_start)}"
            )

        repeat_results.append({
            "train_acc": train_correct / train_total,
            "test_acc": test_correct / test_total,
        })

    # ===== 统计 =====
    case_time = time.time() - case_start

    result = {
        "ipc": ipc,
        "type": "random_stratified",
        "repeat_times": random_repeat,
        "train_size": int(num_classes * ipc),
        "test_size": len(test_dataset),
        "train_acc_mean": float(np.mean([x["train_acc"] for x in repeat_results])),
        "test_acc_mean": float(np.mean([x["test_acc"] for x in repeat_results])),
        "train_acc_std": float(np.std([x["train_acc"] for x in repeat_results])),
        "test_acc_std": float(np.std([x["test_acc"] for x in repeat_results])),
        "time": case_time,
    }

    all_results.append(result)

    print(
        f"[完成] ipc={ipc} | "
        f"test_acc={result['test_acc_mean']:.4f} ± {result['test_acc_std']:.4f} | "
        f"time={fmt(case_time)}"
    )


# =========================
# 保存
# =========================
json_path = os.path.join(save_dir, "random_baseline.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

total_time = time.time() - total_start
print(f"\n全部完成 | 总耗时: {fmt(total_time)}")
print(f"结果保存到: {json_path}")
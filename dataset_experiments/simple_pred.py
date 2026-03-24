from __future__ import annotations

import os
import json
import time   # ✅ 新增
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import SimpleMNISTCNN


# =========================
# 基础配置
# =========================
data_dir = "./datasets/distilled/mnist_simple_DM"
save_dir = "./results/mnist_simple_train"
os.makedirs(save_dir, exist_ok=True)

batch_size = 64
epochs = 30
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

ipcs = list(range(50, 3001, 50))


# =========================
# 时间格式化
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
# cases
# =========================
cases = [{
    "name": "train_original.npz",
    "path": os.path.join(data_dir, "train_original.npz"),
    "ipc": "original",
}]

for ipc in ipcs:
    cases.append({
        "name": f"syn_ipc={ipc}.npz",
        "path": os.path.join(data_dir, f"syn_ipc={ipc}.npz"),
        "ipc": ipc,
    })


# =========================
# 主循环
# =========================
all_results = []
total_start = time.time()   # ✅ 总计时

for i, case in enumerate(cases, start=1):
    if not os.path.exists(case["path"]):
        print(f"[跳过] 文件不存在: {case['path']}")
        continue

    case_start = time.time()   # ✅ case计时

    print(f"\n========== [{i}/{len(cases)}] {case['name']} ==========")

    train_data = np.load(case["path"])
    train_x = torch.tensor(train_data["images"], dtype=torch.float32)
    train_y = torch.tensor(train_data["labels"], dtype=torch.long)
    if train_x.ndim == 3:
        train_x = train_x.unsqueeze(1)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_start = time.time()   # ✅ epoch计时

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

        # ✅ epoch进度输出
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"train_acc={train_correct/train_total:.4f} | "
            f"test_acc={test_correct/test_total:.4f} | "
            f"time={fmt(time.time()-epoch_start)}"
        )

    case_time = time.time() - case_start   # ✅ case耗时

    result = {
        "train_file": case["name"],
        "ipc": case["ipc"],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "final_train_loss": train_loss / train_total,
        "final_train_acc": train_correct / train_total,
        "final_test_loss": test_loss / test_total,
        "final_test_acc": test_correct / test_total,
        "time": case_time,   # ✅ 新增
    }
    all_results.append(result)

    print(
        f"[完成] ipc={case['ipc']} | "
        f"train_acc={result['final_train_acc']:.4f} | "
        f"test_acc={result['final_test_acc']:.4f} | "
        f"time={fmt(case_time)}"
    )


# =========================
# 保存
# =========================
json_path = os.path.join(save_dir, "results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

total_time = time.time() - total_start   # ✅ 总耗时
print(f"\n全部完成 | 总耗时: {fmt(total_time)}")
print(f"结果保存到: {json_path}")
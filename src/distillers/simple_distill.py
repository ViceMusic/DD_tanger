"""Minimal distiller setup example."""
import os
import numpy as np
import torch
import torch.nn as nn


class DM:
    def __init__(
        self,
        model,
        train_dataset,
        num_classes,
        ipc,
        image_shape,
        device=None,
        lr_img=1.0,
        batch_real=64,
        iters=1000,
        save_path="./distilled.npz",
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.ipc = ipc
        self.image_shape = image_shape
        self.device = device
        self.lr_img = lr_img
        self.batch_real = batch_real
        self.iters = iters
        self.save_path = save_path

        self.images_all, self.labels_all = self._load_dataset()
        self.class_indices = self._build_class_indices()

        self.syn_images = nn.Parameter(
            torch.randn(num_classes * ipc, *image_shape, device=device).sigmoid()
        )
        self.syn_labels = torch.tensor(
            [c for c in range(num_classes) for _ in range(ipc)],
            dtype=torch.long,
            device=device,
        )

        self.optimizer = torch.optim.SGD([self.syn_images], lr=lr_img, momentum=0.5)

    def _load_dataset(self):
        images, labels = [], []
        for i in range(len(self.train_dataset)):
            x, y = self.train_dataset[i]
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            else:
                x = x.float()
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.long)
            else:
                y = y.long()
            images.append(x)
            labels.append(y)

        images = torch.stack(images)
        labels = torch.stack(labels).view(-1)
        return images, labels

    def _build_class_indices(self):
        class_indices = {}
        for c in range(self.num_classes):
            idx = torch.where(self.labels_all == c)[0]
            class_indices[c] = idx
        return class_indices

    def get_real_batch(self, class_id):
        idx = self.class_indices[class_id]
        rand_idx = idx[torch.randint(0, len(idx), (self.batch_real,))]
        return self.images_all[rand_idx].to(self.device)

    def get_syn_batch(self, class_id):
        start = class_id * self.ipc
        end = (class_id + 1) * self.ipc
        return self.syn_images[start:end]

    def match_loss(self, feat_real, feat_syn):
        feat_real = feat_real.view(feat_real.size(0), -1)
        feat_syn = feat_syn.view(feat_syn.size(0), -1)

        mean_real = feat_real.mean(dim=0)
        mean_syn = feat_syn.mean(dim=0)

        return ((mean_real - mean_syn) ** 2).sum()

    def distill(self):
        self.model.eval()

        for it in range(1, self.iters + 1):
            loss_total = 0.0

            for c in range(self.num_classes):
                real_batch = self.get_real_batch(c)
                syn_batch = self.get_syn_batch(c)

                feat_real = self.model.embed(real_batch).detach()
                feat_syn = self.model.embed(syn_batch)

                loss = self.match_loss(feat_real, feat_syn)
                loss_total += loss

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.syn_images.clamp_(0.0, 1.0)

            if it % 50 == 0 or it == 1:
                print(f"iter {it}/{self.iters}, loss = {loss_total.item():.6f}")



    def run(self):
        self.distill()
        return {
            "images": self.syn_images.detach().cpu(),
            "labels": self.syn_labels.detach().cpu(),
        }



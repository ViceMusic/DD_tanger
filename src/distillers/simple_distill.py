import time
import torch
import torch.nn as nn


class DM:
    def __init__(
        self,
        model_fn,            # 传入“建模函数”，而不是固定 model 实例
        train_dataset,
        num_classes,
        ipc,
        image_shape,
        device=None,
        lr_img=1.0,
        batch_real=256,
        iters=2000,
        init="real",         # "real" or "noise"
    ):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.ipc = ipc
        self.image_shape = image_shape
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_img = lr_img
        self.batch_real = batch_real
        self.iters = iters
        self.init = init

        self.images_all, self.labels_all = self._load_dataset()
        self.class_indices = self._build_class_indices()

        self.syn_images = nn.Parameter(self._init_syn_images())
        self.syn_labels = torch.tensor(
            [c for c in range(num_classes) for _ in range(ipc)],
            dtype=torch.long,
            device=self.device,
        )

        self.optimizer_img = torch.optim.SGD(
            [self.syn_images], lr=self.lr_img, momentum=0.5
        )

    def _load_dataset(self):
        images, labels = [], []
        for i in range(len(self.train_dataset)):
            x, y = self.train_dataset[i]
            x = x.float() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
            y = y.long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
            images.append(x)
            labels.append(y)

        images = torch.stack(images).to(self.device)
        labels = torch.stack(labels).view(-1).to(self.device)
        return images, labels

    def _build_class_indices(self):
        class_indices = {}
        for c in range(self.num_classes):
            idx = torch.where(self.labels_all == c)[0]
            if len(idx) == 0:
                raise ValueError(f"class {c} has no samples in train_dataset")
            class_indices[c] = idx
        return class_indices

    def _init_syn_images(self):
        if self.init == "real":
            syn = []
            for c in range(self.num_classes):
                idx = self.class_indices[c]
                perm = torch.randperm(len(idx), device=self.device)[:self.ipc]
                syn.append(self.images_all[idx[perm]])
            syn = torch.cat(syn, dim=0).clone().detach()
            return syn.requires_grad_(True)
        else:
            syn = torch.randn(
                self.num_classes * self.ipc, *self.image_shape, device=self.device
            ).sigmoid()
            return syn.requires_grad_(True)

    def get_real_batch(self, class_id, n=None):
        n = n or self.batch_real
        idx = self.class_indices[class_id]
        rand_idx = idx[torch.randint(0, len(idx), (n,), device=self.device)]
        return self.images_all[rand_idx]

    def get_syn_batch(self, class_id):
        start = class_id * self.ipc
        end = (class_id + 1) * self.ipc
        return self.syn_images[start:end]

    def augment(self, x):
        # 这里先留空，默认不增强
        # 你后面可以替换成 DiffAugment(x, ...)
        return x

    def match_loss(self, feat_real, feat_syn):
        feat_real = feat_real.reshape(feat_real.size(0), -1)
        feat_syn = feat_syn.reshape(feat_syn.size(0), -1)

        mean_real = feat_real.mean(dim=0)
        mean_syn = feat_syn.mean(dim=0)

        real_centered = feat_real - mean_real
        syn_centered = feat_syn - mean_syn

        cov_real = real_centered.T @ real_centered / max(feat_real.size(0) - 1, 1)
        cov_syn = syn_centered.T @ syn_centered / max(feat_syn.size(0) - 1, 1)

        loss_mean = ((mean_real - mean_syn) ** 2).mean()
        loss_cov = ((cov_real - cov_syn) ** 2).mean()

        return loss_mean + 0.1 * loss_cov

    def distill(self):
        for it in range(1, self.iters + 1):
            net = self.model_fn().to(self.device)   # 每轮重新采样随机网络
            net.train()

            for p in net.parameters():
                p.requires_grad = False

            embed = net.embed
            loss_total = torch.tensor(0.0, device=self.device)

            for c in range(self.num_classes):
                real_batch = self.get_real_batch(c, self.batch_real)
                syn_batch = self.get_syn_batch(c)

                real_batch = self.augment(real_batch)
                syn_batch = self.augment(syn_batch)

                feat_real = embed(real_batch).detach()
                feat_syn = embed(syn_batch)

                loss_total = loss_total + self.match_loss(feat_real, feat_syn)

            self.optimizer_img.zero_grad()
            loss_total.backward()
            self.optimizer_img.step()

            with torch.no_grad():
                self.syn_images.clamp_(0.0, 1.0)

            if it % 50 == 0 or it == 1:
                print(f"iter {it:04d}/{self.iters}, loss = {loss_total.item():.6f}")

    def run(self):
        self.distill()
        return {
            "images": self.syn_images.detach().cpu(),
            "labels": self.syn_labels.detach().cpu(),
        }
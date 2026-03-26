import time
import torch
import torch.nn as nn


class DM:
    """
    尽量按 VICO-UoE / DatasetCondensation 的 main_DM.py 逻辑复刻
    核心点：
    1. 每个 iteration 都随机初始化一个 net
    2. 冻结 net 参数
    3. 用 net.embed 提特征
    4. real / syn 使用同一个 seed 做 DiffAugment
    5. 非 BN: 逐类匹配 mean(feature)
    6. BN: 先把所有类拼起来一起 forward，再按类 reshape 后匹配 mean(feature)
    7. 只优化 synthetic images
    8. 优化器用 SGD(momentum=0.5)，和官方一致
    """

    def __init__(
        self,
        model_fn,
        train_dataset,
        num_classes,
        ipc,
        image_shape,
        device=None,
        lr_img=1.0,
        batch_real=256,
        iters=20000,
        init="real",
        dsa=True,
        dsa_fn=None,
        dsa_strategy="color_crop_cutout_flip_scale_rotate",
        dsa_param=None,
        model_name="ConvNet",
        log_every=10,
    ):
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.ipc = ipc
        self.image_shape = image_shape  # (C,H,W)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.lr_img = lr_img
        self.batch_real = batch_real
        self.iters = iters
        self.init = init

        self.dsa = dsa
        self.dsa_fn = dsa_fn
        self.dsa_strategy = dsa_strategy
        self.dsa_param = dsa_param

        self.model_name = model_name
        self.log_every = log_every

        self.images_all, self.labels_all = self._load_dataset()
        self.class_indices = self._build_class_indices()

        self.syn_images = nn.Parameter(self._init_syn_images())
        self.syn_labels = torch.tensor(
            [c for c in range(num_classes) for _ in range(ipc)],
            dtype=torch.long,
            device=self.device,
        )

        # 官方 main_DM.py 用的是 SGD + momentum=0.5
        self.optimizer_img = torch.optim.SGD(
            [self.syn_images], lr=self.lr_img, momentum=0.5
        )

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
        C, H, W = self.image_shape
        if self.init == "real":
            syn = []
            for c in range(self.num_classes):
                idx = self.class_indices[c]
                if len(idx) < self.ipc:
                    rand_idx = idx[
                        torch.randint(0, len(idx), (self.ipc,), device=self.device)
                    ]
                else:
                    perm = torch.randperm(len(idx), device=self.device)[: self.ipc]
                    rand_idx = idx[perm]
                syn.append(self.images_all[rand_idx])
            syn = torch.cat(syn, dim=0).clone().detach()
            return syn.requires_grad_(True)
        else:
            syn = torch.randn(
                self.num_classes * self.ipc, C, H, W,
                dtype=torch.float,
                device=self.device,
            )
            return syn.requires_grad_(True)

    def get_real_batch(self, class_id, n=None):
        n = n or self.batch_real
        idx = self.class_indices[class_id]
        rand_idx = idx[torch.randint(0, len(idx), (n,), device=self.device)]
        return self.images_all[rand_idx]

    def get_syn_batch(self, class_id):
        start = class_id * self.ipc
        end = (class_id + 1) * self.ipc
        C, H, W = self.image_shape
        return self.syn_images[start:end].reshape(self.ipc, C, H, W)

    def _has_batchnorm(self):
        # 官方代码是通过 if 'BN' not in args.model 判断
        return "BN" in self.model_name

    def _diff_augment(self, x, seed):
        if (not self.dsa) or (self.dsa_fn is None):
            return x
        return self.dsa_fn(
            x,
            self.dsa_strategy,
            seed=seed,
            param=self.dsa_param,
        )

    def _get_embed(self, net):
        if hasattr(net, "module") and hasattr(net.module, "embed"):
            return net.module.embed
        if hasattr(net, "embed"):
            return net.embed
        raise AttributeError(
            "官方 DM 逻辑要求网络提供 embed()；你的 model_fn 返回的模型没有 embed 方法。"
        )

    def distill(self):
        for it in range(self.iters + 1):
            # ===== 官方 DM：每轮新建一个随机网络 =====
            net = self.model_fn().to(self.device)
            net.train()

            for p in net.parameters():
                p.requires_grad = False

            embed = self._get_embed(net)

            if not self._has_batchnorm():
                # ===== 官方 main_DM.py 非 BN 分支 =====
                loss = torch.tensor(0.0, device=self.device)

                for c in range(self.num_classes):
                    img_real = self.get_real_batch(c, self.batch_real)
                    img_syn = self.get_syn_batch(c)

                    if self.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = self._diff_augment(img_real, seed)
                        img_syn = self._diff_augment(img_syn, seed)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss = loss + torch.sum(
                        (
                            torch.mean(output_real, dim=0)
                            - torch.mean(output_syn, dim=0)
                        ) ** 2
                    )

            else:
                # ===== 官方 main_DM.py BN 分支 =====
                images_real_all = []
                images_syn_all = []

                for c in range(self.num_classes):
                    img_real = self.get_real_batch(c, self.batch_real)
                    img_syn = self.get_syn_batch(c)

                    if self.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = self._diff_augment(img_real, seed)
                        img_syn = self._diff_augment(img_syn, seed)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                output_real = output_real.reshape(self.num_classes, self.batch_real, -1)
                output_syn = output_syn.reshape(self.num_classes, self.ipc, -1)

                loss = torch.sum(
                    (
                        torch.mean(output_real, dim=1)
                        - torch.mean(output_syn, dim=1)
                    ) ** 2
                )

            self.optimizer_img.zero_grad()
            loss.backward()
            self.optimizer_img.step()

            if it % self.log_every == 0:
                print(f"iter = {it:05d}, loss = {loss.item():.4f}")

    def run(self):
        self.distill()
        return {
            "images": self.syn_images.detach().cpu(),
            "labels": self.syn_labels.detach().cpu(),
        }
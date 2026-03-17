import os
import csv
import numpy as np
from torch.utils.data import Dataset

from .reader import FeatureReader


class VSLDataset(Dataset):
    """
    Dataset chuẩn cho CTR-GCN Early Fusion 9-channel.
    Đọc fused features (.npy), áp dụng augmentation, normalize.
    """

    def __init__(
        self,
        csv_path,
        feature_dir,
        is_train=True,
        max_frames=64,
        random_frame_drop=0.0,
        random_joint_drop=0.0,
    ):
        self.csv_path = csv_path
        self.feature_dir = feature_dir
        self.is_train = is_train
        self.max_frames = max_frames
        self.random_joint_drop = random_joint_drop

        # Load file_name, label_id
        self.samples = self._load_csv()

        # FeatureReader dùng Random Frame Drop
        self.reader = FeatureReader(
            feature_dir=feature_dir,
            max_frames=max_frames,
            random_drop=random_frame_drop,
        )

    # ===========================
    # LOAD CSV
    # ===========================
    def _load_csv(self):
        items = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append((row["file_name"].strip(),
                              int(row["label_id"])))
        return items

    def __len__(self):
        return len(self.samples)

    # ===========================
    # AUGMENTATION
    # ===========================
    def joint_drop(self, x):
        """
        Drop ngẫu nhiên joints → tạo robustness.
        x: (C, T, V)
        """
        if self.random_joint_drop <= 0:
            return x

        _, _, V = x.shape
        drop_num = int(V * self.random_joint_drop)
        drop_joints = np.random.choice(V, drop_num, replace=False)
        x[:, :, drop_joints] = 0
        return x

    def normalize(self, x):
        """
        Normalize skeleton như CTR-GCN:
        Remove mean theo joints axis.
        """
        mean = x.mean(axis=2, keepdims=True)
        return x - mean

    # ===========================
    # GET ITEM
    # ===========================
    def __getitem__(self, idx):
        file_name, label_id = self.samples[idx]

        # Load fused feature → (C=9, T, V)
        x = self.reader.load_feature(file_name)

        # Augment CHỈ khi train
        if self.is_train:
            x = self.joint_drop(x)

        # Normalize
        x = self.normalize(x)

        return x, label_id

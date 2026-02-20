"""
Sign Language DataModule for Light-T2M
Compatible with Hydra config system
"""
import os
import os.path as osp
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from src.utils.feats2joints import feats2joints_smplx

from .signlang import (
    SignMotionDataset,
    SignText2MotionDataset,
    SignText2MotionDatasetEval,
    sign_collate,
)


class SignDataModule(LightningDataModule):
    """DataModule for Sign Language datasets (H2S, CSL, Phoenix)"""
    
    def __init__(
        self,
        data_root: str,
        csl_root: str = None,
        phoenix_root: str = None,
        # 528D npy 경로 (None이면 기존 data_root 사용)
        npy_root: str = None,
        csl_npy_root: str = None,
        phoenix_npy_root: str = None,
        mean_path: str = None,
        std_path: str = None,
        # CSL 전용 mean/std
        csl_mean_path: str = None,
        csl_std_path: str = None,
        batch_size: int = 64,
        val_batch_size: int = -1,
        test_batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = False,
        njoints: int = 55,
        nfeats: int = 120,
        fps: int = 25,
        max_motion_length: int = 300,
        min_motion_length: int = 40,
        unit_length: int = 4,
        dataset_name: str = 'how2sign_csl_phoenix',
        stage: str = 'lm',
        motion_dim: int = 120,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_root = data_root
        self.csl_root = csl_root
        self.phoenix_root = phoenix_root
        self.npy_root = npy_root
        self.csl_npy_root = csl_npy_root
        self.phoenix_npy_root = phoenix_npy_root
        self.njoints = njoints
        self.nfeats = nfeats
        self.name = dataset_name
        self.stage = stage
        
        # H2S/Phoenix용 mean/std (기본)
        if mean_path and osp.exists(mean_path):
            self.mean = torch.load(mean_path)[:nfeats]
        else:
            self.mean = torch.zeros(nfeats)
            
        if std_path and osp.exists(std_path):
            self.std = torch.load(std_path)[:nfeats]
        else:
            self.std = torch.ones(nfeats)
        
        # CSL 전용 mean/std
        if csl_mean_path and osp.exists(csl_mean_path):
            self.csl_mean = torch.load(csl_mean_path)[:nfeats]
        else:
            self.csl_mean = self.mean  # fallback to general
            
        if csl_std_path and osp.exists(csl_std_path):
            self.csl_std = torch.load(csl_std_path)[:nfeats]
        else:
            self.csl_std = self.std  # fallback to general
        
        self.dataloader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "collate_fn": sign_collate,
        }
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def get_mean_std(self):
        return {"mean": self.mean, "std": self.std}
    
    def feats2joints(self, features):
        """Convert features to 55 joints via SMPLX"""
        _, joints = feats2joints_smplx(features, self.mean, self.std)
        return joints
    
    def normalize(self, motion):
        mean = self.mean.to(motion.device)
        std = torch.clamp(self.std.to(motion.device), min=1e-8)
        return (motion - mean) / std
    
    def denormalize(self, motion):
        mean = self.mean.to(motion.device)
        std = self.std.to(motion.device)
        return motion * std + mean
    
    def setup(self, stage=None):
        common_kwargs = {
            'data_root': self.data_root,
            'csl_root': self.csl_root,
            'phoenix_root': self.phoenix_root,
            'npy_root': self.npy_root,
            'csl_npy_root': self.csl_npy_root,
            'phoenix_npy_root': self.phoenix_npy_root,
            'dataset_name': self.name,
            'nfeats': self.nfeats,
            'max_motion_length': self.hparams.max_motion_length,
            'min_motion_length': self.hparams.min_motion_length,
            'unit_length': self.hparams.unit_length,
            'mean': self.mean,
            'std': self.std,
            # CSL 전용 mean/std 전달
            'csl_mean': self.csl_mean,
            'csl_std': self.csl_std,
        }
        
        DatasetClass = SignMotionDataset if self.stage == 'vae' else SignText2MotionDataset
        
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(split='train', **common_kwargs)
            self.val_dataset = DatasetClass(split='val', **common_kwargs)
        
        if stage == 'test' or stage is None:
            if self.stage == 'vae':
                self.test_dataset = SignMotionDataset(split='test', **common_kwargs)
            else:
                self.test_dataset = SignText2MotionDatasetEval(split='test', **common_kwargs)
    
    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup('fit')
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.batch_size
        return DataLoader(dataset=self.train_dataset, shuffle=True, drop_last=True, **options)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup('fit')
        options = self.dataloader_options.copy()
        bs = self.hparams.val_batch_size
        options["batch_size"] = bs if bs > 0 else self.hparams.batch_size
        return DataLoader(dataset=self.val_dataset, shuffle=False, drop_last=False, **options)
    
    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        options = self.dataloader_options.copy()
        options["batch_size"] = self.hparams.test_batch_size
        return DataLoader(dataset=self.test_dataset, shuffle=False, drop_last=False, **options)
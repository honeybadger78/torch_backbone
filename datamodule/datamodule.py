import torch
import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from datamodule.dataset import CustomDataset


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/workspace/torch_backbone/data/tiny-imagenet-200",
        batch_size: int = 32,
        num_workers=torch.cuda.device_count() * 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_transforms = A.Compose(
            [
                A.Resize(64, 64),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.val_transforms = A.Compose(
            [
                A.Resize(64, 64),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            data_path=self.hparams.data_dir,
            transforms=self.train_transforms,
            is_train=True,
        )
        self.val_dataset = CustomDataset(
            data_path=self.hparams.data_dir,
            transforms=self.val_transforms,
            is_train=False,
        )
        self.test_dataset = CustomDataset(
            data_path=self.hparams.data_dir,
            transforms=self.val_transforms,
            is_train=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, is_train=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, is_train=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, is_train=False)

    def _loader(self, dataset, is_train):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=is_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

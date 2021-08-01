# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MNIST backbone image classifier example.

To run:
python backbone_image_classifier.py --trainer.max_epochs=50
"""
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from adas_dataset import AdasDataset

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    import torchvision.models as models


class ChargerKpts(pl.LightningModule):

    def __init__(
        self,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.num_points = 4
        self.backbone = models.resnet50(pretrained=True)
        self.kpt_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, self.num_points*2, kernel_size=3),
        )

    def forward(self, x):
        # use forward for inference/predictions
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        return hm

    def training_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        assert hm.shape() == y.shape(), f"hm and target shape mismatch. hm:{hm.shape()}, y:{y.shape()}"
        loss = F.binary_cross_entropy(hm, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        loss = F.binary_cross_entropy(hm, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        loss = F.binary_cross_entropy(hm, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self.kpt_head(self.backbone(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class ChargerData(pl.LightningDataModule):

    def __init__(
        self,
        ds_path: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_ds = AdasDataset(ds_path, "train")
        self.val_ds = AdasDataset(ds_path, "val")
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


if __name__ == '__main__':
    dataset = ChargerData("/path/to/ds")
    ch_call = pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=2)
    trainer = pl.Trainer(checkpoint_callback=True, callbacks=[ch_call])
    trainer.fit(ChargerKpts, dataset.train_dataloader)

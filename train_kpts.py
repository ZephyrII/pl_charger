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
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ChargerKpts(pl.LightningModule):

    def __init__(
        self,
        ds_path,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.ds_path = ds_path
        self.batch_size = 1 #FIXME
        self.save_hyperparameters(ignore=['backbone'])
        self.num_points = 4
        # self.backbone = models.resnet50(pretrained=True)
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.kpt_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            # torch.nn.Conv2d(256, 64, 3, padding=1),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.Upsample(scale_factor=2),
            # torch.nn.Conv2d(256, 128, 3, padding=1),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.Upsample(scale_factor=2),
            # torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, self.num_points, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # use forward for inference/predictions
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        return hm

    def training_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)['0']
        assert fm.shape[-2:] == (240, 240), f"incorrect fm shape:{fm.shape}"
        hm = self.kpt_head(fm)
        assert hm.shape == y.shape, f"hm and target shape mismatch. hm:{hm.shape()}, y:{y.shape()}"
        loss = F.binary_cross_entropy(hm, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)['0']
        assert fm.shape[-2:] == (240, 240), f"incorrect fm shape:{fm.shape}"
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

    def train_dataloader(self):
        return DataLoader(AdasDataset(self.ds_path, "train"), batch_size=self.batch_size, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(AdasDataset(self.ds_path, "val"), batch_size=self.batch_size, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(AdasDataset(self.ds_path, "val"), batch_size=self.batch_size, num_workers=2, pin_memory=True)

if __name__ == '__main__':
    # dataset = ChargerData("/root/share/tf/dataset/4_point_final/")
    ch_call = pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_last=True, dirpath='/root/share/tf/pl_checkpoints/res_50/')
    trainer = pl.Trainer(gpus=1, checkpoint_callback=True, callbacks=[ch_call], accumulate_grad_batches=4)
    model = ChargerKpts("/root/share/tf/dataset/final_localization/tape_1.0/")
    trainer.fit(model)

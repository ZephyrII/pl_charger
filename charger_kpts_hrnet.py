import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import cv2
import numpy as np
from torch.utils.data import DataLoader
from adas_dataset import AdasDataset
import timm


class ChargerKptsHrnet(pl.LightningModule):

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
        self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
        self.kpt_head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
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
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, self.num_points, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # use forward for inference/predictions
        with torch.no_grad():
            fm = self.backbone(x)[1]
            hm = self.kpt_head(fm)
        return hm

    def training_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)[1]
        assert fm.shape[-2:] == (240, 240), f"incorrect fm shape:{fm.shape}"
        hm = self.kpt_head(fm)
        assert hm.shape == y.shape, f"hm and target shape mismatch. hm:{hm.shape()}, y:{y.shape()}"
        loss = F.binary_cross_entropy(hm, y)
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)[1]
        assert fm.shape[-2:] == (240, 240), f"incorrect fm shape:{fm.shape}"
        hm = self.kpt_head(fm)
        loss = F.binary_cross_entropy(hm, y)
        self.log('valid_loss', loss, on_epoch=True, logger=True)

        if batch_idx==0:
            input = x[0].cpu().numpy()
            input = np.transpose(input, [1,2,0])

            res = np.sum(hm[0].cpu().numpy(), axis=0)
            res = res-np.min(res)
            res = res/np.max(res)
            mask = np.zeros_like(input)
            mask[:,:,0] = res
            alpha = 0.5
            out_img = cv2.addWeighted(input, alpha, mask, 1-alpha, 0)

            self.logger.experiment.log_image("test_img", out_img)

    def test_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)
        assert fm.shape()[-2:] == (240, 240), f"incorrect fm shape:{fm.shape()}"
        hm = self.kpt_head(fm)
        res = np.sum(hm, axis=0)
        alpha = 0.5
        out_img = cv2.addWeighted(res, alpha, x, 1-alpha, 0)

        self.logger.experiment.log_image("test_img", out_img)

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

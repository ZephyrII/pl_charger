import torch
import pytorch_lightning as pl
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.nn import functional as F
import cv2
import numpy as np
from torch.utils.data import DataLoader
from adas_dataset import AdasDataset
from sklearn.cluster import DBSCAN, KMeans
from pck import keypoint_pck_accuracy
# from .adas_dataset import AdasDataset


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
        # self.backbone = resnet_fpn_backbone('resnext50_32x4d', pretrained=True)
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
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, self.num_points, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # use forward for inference/predictions
        with torch.no_grad():
            fm = self.backbone(x)['0']
            hm = self.kpt_head(fm)
        return hm

    def training_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)['0']
        hm = self.kpt_head(fm)
        loss = F.binary_cross_entropy(hm, y)
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fm = self.backbone(x)['0']
        assert fm.shape[-2:] == (240, 240), f"incorrect fm shape:{fm.shape}"
        hm = self.kpt_head(fm)
        loss = F.binary_cross_entropy(hm, y)
        self.log('valid_loss', loss, on_epoch=True, logger=True)
        with torch.no_grad():
            hm = hm[0].cpu().detach().numpy()

            if batch_idx==0:
                input = x[0].cpu().numpy()
                input = np.transpose(input, [1,2,0])
                res = np.sum(hm, axis=0)
                res = res-np.min(res)
                res = res/np.max(res)
                mask = np.zeros_like(input)
                mask[:,:,0] = res
                alpha = 0.5
                out_img = cv2.addWeighted(input, alpha, mask, 1-alpha, 0)

                self.logger.experiment.log_image("test_img", out_img)

            preds = []
            for i, kp in enumerate(hm):
                # subtract all other heatmaps from current heatmap. Use when more than one detection is on the same point on image
                raw_kp = kp

                kp = (kp - np.min(kp)) / (np.max(kp) - np.min(kp))
                h, w = kp.shape
                # Binarize kp heatmaps
                ret, kp = cv2.threshold(kp, 0.2, 1.0, cv2.THRESH_BINARY)
                # Get coordinates of white pixels
                X = np.argwhere(kp == 1)
                if X.shape[0] == 0:
                    preds.append((0, 0))
                    continue
                # Init algorithm for clustering
                clustering = DBSCAN(eps=3, min_samples=2, n_jobs=8)
                # clustering = KMeans(n_clusters=4)
                clustering.fit(X)
                cluster_scores = []
                # Get labels of all clusters

                unique_labels = np.unique(clustering.labels_)

                # for each cluster calculate their "score" by summing values of all pixels in cluster
                for id in np.unique(clustering.labels_):
                    cluster = X[np.where(clustering.labels_ == id)]
                    cluster_scores.append(np.sum(raw_kp[cluster[:, 0], cluster[:, 1]]))
                # Get pixels of cluster with max score
                cluster = X[np.where(clustering.labels_ == unique_labels[np.argmax(cluster_scores)])]
                mask = np.zeros_like(kp)
                mask[cluster[:, 0], cluster[:, 1]] = raw_kp[cluster[:, 0], cluster[:, 1]]

                if np.sum(mask) == 0:
                    center = (0, 0)
                else:
                    center = np.average(np.sum(mask, axis=1) * np.arange(w)) / np.sum(mask) * w, \
                            np.average(np.sum(mask, axis=0) * np.arange(h)) / np.sum(mask) * h
                preds.append(center)

            gt = []
            for i, kp in enumerate(y.cpu().detach().numpy()[0]):
                # subtract all other heatmaps from current heatmap. Use when more than one detection is on the same point on image
                h, w = kp.shape
                # Get coordinates of white pixels
                X = np.argwhere(kp == 1)
                if X.shape[0] == 0:
                    gt.append((0, 0))
                    continue
                # Init algorithm for clustering
                clustering = DBSCAN(eps=3, min_samples=2, n_jobs=8)
                clustering.fit(X)
                cluster_scores = []
                # Get labels of all clusters

                unique_labels = np.unique(clustering.labels_)

                # for each cluster calculate their "score" by summing values of all pixels in cluster
                for id in np.unique(clustering.labels_):
                    cluster = X[np.where(clustering.labels_ == id)]
                    cluster_scores.append(np.sum(raw_kp[cluster[:, 0], cluster[:, 1]]))
                # Get pixels of cluster with max score
                cluster = X[np.where(clustering.labels_ == unique_labels[np.argmax(cluster_scores)])]
                mask = np.zeros_like(kp)
                mask[cluster[:, 0], cluster[:, 1]] = raw_kp[cluster[:, 0], cluster[:, 1]]

                if np.sum(mask) == 0:
                    center = (0, 0)
                else:
                    center = np.average(np.sum(mask, axis=1) * np.arange(w)) / np.sum(mask) * w, \
                            np.average(np.sum(mask, axis=0) * np.arange(h)) / np.sum(mask) * h
                gt.append(center)
            
            acc, avg_acc, cnt = keypoint_pck_accuracy(np.array([preds]), np.array([gt]), 2/960, np.broadcast_to(x.shape[-2:], (4,2)))

            self.log('avg_acc', avg_acc, on_epoch=True, logger=True)


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
        return DataLoader(AdasDataset(self.ds_path, "train"), batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(AdasDataset(self.ds_path, "val"), batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(AdasDataset(self.ds_path, "val"), batch_size=self.batch_size, num_workers=4, pin_memory=False)

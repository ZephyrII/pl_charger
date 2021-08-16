import torch
import pandas as pd
import cv2
import numpy as np
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class AdasDataset(Dataset):
    def __init__(self, dataset_dir, subset, num_points=4):
        self.num_points = num_points
        self.increase_bbox_percent = 0.05
        self.img_labels = []
        if subset == "val":
            dataset_dir = os.path.join(dataset_dir, 'val')
        annotations = os.listdir(os.path.join(dataset_dir, 'annotations'))

        # Add images
        for a in annotations:
            image_path = os.path.join(dataset_dir, 'images', a[:-4]+'.png')
            self.img_labels.append({
                "img_id":a,  # use file name as a unique image id
                "img_path":image_path,
                "annotation":os.path.join(dataset_dir, 'annotations', a)})

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx]["img_path"]
        image = read_image(img_path).float()/255
        label = self.generate_target(idx)
        return image, label

    def generate_target(self, image_id):
        ann_path = self.img_labels[image_id]['annotation']
        #TODO: consider using xmltodict
        tree = ET.parse(ann_path)
        root = tree.getroot()
        keypoints = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        kp_maps = np.zeros((self.num_points, w, h), dtype=np.float32)
        obj = root.findall('object')[0]
        kps = obj.find('keypoints')
        for i in range(self.num_points):
            kp = kps.find('keypoint' + str(i))
            point_size = 5
            point_center = (
                int((float(kp.find('x').text) * w)),
                int((float(kp.find('y').text) * h)))
            keypoints.append(point_center)
            cv2.circle(kp_maps[i], point_center, point_size, 255, -1)
            # TODO: verify label corrrectness
            image = read_image(self.img_labels[image_id]['img_path'])
            # kap = kp_maps[i].astype(np.float32) / 255
            # kap = cv2.cvtColor(kap, cv2.COLOR_GRAY2BGR)
            # alpha = 0.8
            # out = cv2.addWeighted(image, alpha, kap, 1 - alpha, 0.0)
            # cv2.imshow('xddlol', out)
            # cv2.waitKey(0)

        return torch.Tensor(kp_maps/255)#, torch.Tensor(keypoints)
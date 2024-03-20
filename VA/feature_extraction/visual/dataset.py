# *_*coding:utf-8 *_*
import os
import cv2
import glob
import numpy as np
from PIL import Image
from skimage import io
from skimage import img_as_float
import torch.utils.data as data
import tqdm


class FaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = []
        print('reading img from ', self.path)
        for img in tqdm.tqdm(sorted(os.listdir(self.path))):
            if not img.endswith('.jpg'):
                continue
            img_path = os.path.join(self.path, img)
            with Image.open(img_path) as img:
                # 读取图像数据
                img_data = np.array(img)
                # 将图像数据添加到列表中
                frames.append(img_data)
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        ## image format
        # path = self.frames[index]
        # img = Image.open(path)
        # name = os.path.basename(path)[:-4]

        ## npy format [cv2 -> Image]
        img = self.frames[index]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        name = '%08d' %(index)

        if self.transform is not None:
            img = self.transform(img)
        return img, name


class FaceDatasetForEmoNet(data.Dataset):
    def __init__(self, vid, face_dir, transform=None, augmentor=None):
        super(FaceDatasetForEmoNet, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.augmentor = augmentor
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = []
        print('reading img from ', self.path)
        for img in tqdm.tqdm(sorted(os.listdir(self.path))):
            if not img.endswith('.jpg'):
                continue
            img_path = os.path.join(self.path, img)
            with Image.open(img_path) as img:
                # 读取图像数据
                img_data = np.array(img)
                # 将图像数据添加到列表中
                frames.append(img_data)
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        ## image format
        # path = self.frames[index]
        # img = io.imread(path)
        # name = os.path.basename(path)[:-4]

        ## npy format [cv2 -> skimage]
        img = self.frames[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = '%08d' %(index)

        if self.augmentor is not None:
            img = self.augmentor(img)[0]
        if self.transform is not None:
            img = self.transform(img)
        
        return img, name
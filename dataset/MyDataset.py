#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2024/11/21  09:58
@Author:   WWD
@File:     databse.py
@Software: VSCode
"""

import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

IMAGE_SHAPE = (120, 50)

transform = T.Compose([
    T.Resize(IMAGE_SHAPE),
    T.ToTensor(),
    T.Normalize((0.647, 0.654, 0.653), (0.251, 0.250, 0.249))
])


class MyDataset(Dataset):
    def __init__(self, data_path, label_map, label_map_len):
        super(MyDataset, self).__init__()
        # split
        self.data = [(os.path.join(data_path, file), file.split('.')[0]) for file in os.listdir(data_path)]
        self.label_map = [char for char in label_map]
        self.label_map_len = len(self.label_map)
        self.label_map_len = label_map_len

    def __getitem__(self, index):
        file = self.data[index][0]
        label = self.data[index][1]
        raw_len = len(label)

        img = Image.open(file).convert('RGB')
        img = transform(img)

        # 将标签字符转换为对应的索引
        label = [self.label_map.index(i) for i in label]
        label = torch.as_tensor(label, dtype=torch.int64)
        
        # 二分类标签计算方式
        # label = torch.tensor(self.label_map.index(label), dtype=torch.float32)

        return img, label, raw_len

    def __len__(self):
        return len(self.data)
    

    
def get_train_test(train_path, test_path, label_map, label_map_len):
    train = DataLoader(
        dataset=MyDataset(train_path, label_map=label_map, label_map_len=label_map_len),
        batch_size=32, shuffle=True,
        num_workers=3)
    test = DataLoader(
        dataset=MyDataset(test_path, label_map=label_map, label_map_len=label_map_len),
        batch_size=32, shuffle=True,
        num_workers=0)
    
    return train, test
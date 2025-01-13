#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2025/01/13  09:07
@Author:   WWD
@File:     CRNN.py
@Software: VSCode
"""

import torch
import torch.nn as nn



class CNNLayer(nn.Module):
    def __init__(self, filters, kernel_size, strides, flag_pool = 1):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=filters[0], out_channels=filters[1],
            kernel_size=kernel_size, stride=strides[0],
            padding=(kernel_size - 1) // 2,  # Assuming 'same' padding
            bias=False  # We will use BatchNorm, so no bias is needed
        )
        self.bn = nn.BatchNorm2d(num_features=filters[1], momentum=0.9)
        self.relu = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=strides[1], padding=(1, 1))  # Assuming 'same' padding
        self.flag = flag_pool
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CRNN(nn.Module):
    def __init__(self, charset_len):
        super(CRNN, self).__init__()
        self.charset_len = charset_len
        self.cnn_layer1 = CNNLayer(kernel_size=7, filters=[3, 32], strides=(1, 1))
        self.cnn_layer2 = CNNLayer(kernel_size=5, filters=[32, 64], strides=(1, 2))
        self.cnn_layer3 = CNNLayer(kernel_size=3, filters=[64, 128], strides=(1, 2))
        self.cnn_layer4 = CNNLayer(kernel_size=3, filters=[128, 128], strides=(1, 2))
        self.cnn_layer5 = CNNLayer(kernel_size=3, filters=[128, 64], strides=(1, 2))

        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True,
                                dropout=0.4)
        self.fc = nn.Linear(in_features=128, out_features=len(self.charset_len))
        
    def forward(self, input):
        x = self.cnn_layer1(input)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        
        x = torch.reshape(x, (x.shape[0], x.shape[2] * x.shape[3], x.shape[1]))
        x, _ = self.rnn(x)
        x = torch.reshape(x, (-1, 128))
        x = self.fc(x)
        x = torch.reshape(x, (input.shape[0], -1, len(self.charset_len)))

        
        predict = torch.transpose(x, 1, 0)
        out = predict.max(2)[1].transpose(0, 1)
        
        return predict, out


loss_func = nn.CTCLoss(blank=0, reduction='mean')


def Train(model, inputs, labels, label_length, optimizer):
    model.train()
    predict, _ = model(inputs)
    
    loss = get_loss(predict, labels, label_length)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lr = optimizer.param_groups[0]['lr']

    return loss.detach().cpu().numpy(), lr

def Test(model, inputs, labels, label_length):
    model.eval()
    predict, _ = model(inputs)
    
    loss = get_loss(predict, labels, label_length)
    outputs = predict.max(2)[1].transpose(0, 1)
    pred_decode_labels = []
    for pred_labels in outputs:
        decoded = []
        for item in pred_labels:
            if item != 0:
                decoded.append(item.item())
        pred_decode_labels.append(decoded)
    labels_list = []
    labels = labels.tolist()
    i = 0
    for idx in label_length.tolist():
        labels_list.append(labels[i: i + idx])
        i += idx

    correct_list = []
    error_list = []
    for ids in range(len(labels_list)):
        if labels_list[ids] == pred_decode_labels[ids]:
            correct_list.append(ids)
        else:
            error_list.append(ids)

        # 计算准确率
    total_num = len(correct_list) + len(error_list)
    accuracy = len(correct_list) / total_num if total_num > 0 else 0  # 防止除数为 0 的情况
    
    return loss, accuracy

def get_loss(pred, labels, label_length):
    log_pred = pred.log_softmax(2)
    seq_len = torch.IntTensor([log_pred.shape[0]] * log_pred.shape[1])
    loss = loss_func(log_pred.cpu(), labels, seq_len, label_length)

    return loss
import torch
from torch import nn
import torch.nn.functional as F


class Net4o6(nn.Module):
    def __init__(self):
        super(Net4o6, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1500, 1)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))

        x = x.reshape(x.shape[0], -1)
        # 二分类问题，值在(0,1)
        x = torch.sigmoid(self.fc1(x))
        return x

# 定义损失函数
loss_func = nn.BCELoss()
            
def Train(model, x, label, optimizer):
    net_output = model(x)
    
    label = label.view(-1, 1)

    loss = loss_func(net_output, label)

    # 计算损失
    loss = loss_func(net_output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lr = optimizer.param_groups[0]['lr']

    return loss.detach().cpu().numpy(), lr



def Test(model, x, label):
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        net_output = model(x)
        label = label.view(-1, 1)
        loss = loss_func(net_output, label)
        
        # 使用sigmoid的输出与阈值比较来获取预测结果
        predict = (net_output >= 0.5).float()
        
        # 确保label是浮点类型，以便与predict进行比较
        label = label.float()
        
        # 计算准确率
        total += label.numel()
        correct += (predict == label).sum().item()
        acc = correct / total

    return loss.detach().cpu().numpy(), acc



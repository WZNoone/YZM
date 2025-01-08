import torch
from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self, label_len, label_map):
        super(Net, self).__init__()
        self.label_len = label_len
        self.num_classes = len(label_map)
        self.resnet = models.resnet18(num_classes=label_len * len(label_map))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, self.label_len, self.num_classes)
        # [batchsize, length, class]

        return x

# 定义损失函数
loss_func = nn.CrossEntropyLoss()
            
def Train(model, x, label, optimizer):
    net_output = model(x)
    
    net_output = net_output.view(-1, net_output.size(-1))
    label = label.view(-1)

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

        net_output = net_output.view(-1, net_output.size(-1))
        label = label.view(-1)
        loss = loss_func(net_output, label)

        # 获取预测结果
        predict = torch.argmax(net_output, dim=-1)
        # 确保label和predict的数据类型一致
        if label.dtype != predict.dtype:
            label = label.type(predict.dtype)

        # 计算准确率
        total += label.numel()
        # 使用.item()来获取正确预测的数量
        correct += (predict == label).sum().item()
        acc = correct / total

    return loss.detach().cpu().numpy(), acc



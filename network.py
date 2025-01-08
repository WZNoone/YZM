import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class Net(nn.Module):
    def __init__(self, BasicBlock = BasicBlock, label_len = 6, label_map = 0) -> None:
        super(Net, self).__init__()
        self.label_len = label_len
        self.in_channels = 64
        self.numclass = len(label_map)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1,1],[1,1]])
        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, label_len * len(label_map))

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        
        # 展平特征图，二维张量，其中第一维是批次大小，第二维是所有特征的总数
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = out.view(-1, self.label_len, self.numclass)

        return out

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


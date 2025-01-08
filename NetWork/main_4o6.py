import sys
sys.path.append('d:\\S\\yzm\\CBcap')
import torch
from tqdm import tqdm
from devide4or6 import Net4o6, Train, Test
from torch.optim.lr_scheduler import StepLR
from dataset.MyDataset import get_train_test


LABEL_MAP = [i for i in '01']

Max_label_len = 1

train_path = r'D:/S/yzm/Mark/train_all'

test_path = r'D:/S/yzm/Mark/test_all'


if __name__ == '__main__':
    # 加载数据集
    train, test = get_train_test(train_path, test_path, LABEL_MAP, len(LABEL_MAP))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net4o6()
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.9)


    for epoch in range(0, 30):
        # Train
        train_bar = tqdm(train, 'Training')
        for x, label, _ in train_bar:
            x, label = x.to(DEVICE), label.to(DEVICE)
            loss, lr = Train(model, x, label, optimizer)
            train_bar.set_description("Train epoch %d, loss %.4f, lr %.6f" % (
                epoch, loss, lr))

        # Test
        test_bar = tqdm(test, 'Test')
        # 初始化准确率和损失总和变量
        total_acc, total_loss, total_samples = 0, 0, 0

        # 测试循环
        for x, label, _ in test_bar:
            x, label = x.to(DEVICE), label.to(DEVICE)
            loss, acc = Test(model, x, label)
            total_acc += acc * x.size(0)  # 累加准确率
            total_loss += loss * x.size(0)  # 累加损失
            total_samples += x.size(0)  # 累加样本数量
            test_bar.set_description("Eval epoch %d, batch acc %.4f, batch loss %.4f" % (
                epoch, acc, loss))

        # 计算整体准确率和损失
        average_acc = total_acc / total_samples
        average_loss = total_loss / total_samples

        # 保存整体准确率
        torch.save(model.state_dict(), "pth/4o6%.4f.pth" % average_acc)

        scheduler.step()


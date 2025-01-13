import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

IMAGE_SHAPE = (120, 50)

transform = T.Compose([
    T.Resize(IMAGE_SHAPE),
    T.ToTensor(),
    T.Normalize((0.647, 0.654, 0.653), (0.251, 0.250, 0.249))
])

# 假设LABEL_MAP是一个字符串，包含了所有可能的字符
LABEL_MAP = '23456789ABCDEFGHIJKLMNPQRSTUVWXYZ\
    出毕例足析存炎枚吉汕色岸欣那佻函沃办孙画或妮只汾取宇忍邵向入同吃夕忽个兄佳以弘丰们戊伽羽委日庚幻江仓物有舟过空形告序讦三别卧板米执戎许古丹呈幺垂居青达金卷沛六池考忠纤皮枕果壮乙些门亢佩沈宠来后夜自伯长刺局朵巫飞风去抒打言弦切直扶两坤刀宙农宏杏侗左击与乳京宝和什杉邦伸我兔投冶欢竹供奈甸屉朋妻然始放此昌仪负共何力制昆史迅布代旦助冈牙朱昏百处即夫父邓廷往身中礼尽耳伺巾甲二枇必开全岩犬刷岐一杖芒广念奇昂住汉关常侄先动改分审尺互兵亿政设仗央岱邑固佐示甘幼典决佬吏券宗欠汰且材用明记弛叶臣实闪对厂书李快义岳仇凡又未访沙引弋才卓讯付瓜月忝妲至伶光你吝系卯帙呢吴友更利彤吻于到目杷豆坦兆少九坐丸圻裏昀房功咏采冽气究优低阪肋攻训弓凤文扮均侍忘术沐币伏孝府手冰字无仙企火姑杯松徇圣戏电年抑味承侏丛状皂奉化让列井妞斗季汪龙君舌弟亟受予土岭扑技妆为讼杞议虫也亘行内乞旧沁吞生妯军田虎尤寻妓卦刑瓦亚旺床官估侈阜纠孕的寸定十店帖位合所尚七务佯吟坡宕北吸洽末台使扒牛汁帛享炉爷呆方公判卒早纪毛辰驰亡役昔久坎托升宛禾佛么初冲佰驭会岢艾兴式良干协其效斥权武在鸟红炊刻四版具八服作并屈巧上姒幸成车盲仟弄千冷母完尾仕忏男肉岑戈立云坪步届今枝夭死非杜巨侃习匹束场句贞仔人刹汇赤回争问占杼伴吾他尧传华林斤扣闭外卫工姓角户地勾东衣午炙希由贝孟私当血研妾羊劣万失灰依西发孔征区玉巡含事忙肌白宜表舍心兰汐讲子杀木乡尸产机命佟易页汝见杰业志则约拈佃杭水歹王儿冬乎老归村抗伟酉吵姐扔丑秀下玖灭底坊竺众专及五本宋齐周吹而知努姗丁丈牧彼杆邪社'
LABEL_MAP_LEN = len(LABEL_MAP)

class MyDataset(Dataset):
    def __init__(self, data_path, label_map):
        super(MyDataset, self).__init__()
        self.data = [(os.path.join(data_path, file), file.split('.')[0]) for file in os.listdir(data_path)]
        self.label_map = label_map

    def __getitem__(self, index):
        file, label = self.data[index]
        raw_len = len(label)

        img = Image.open(file).convert('RGB')
        img = transform(img)

        # 将标签字符转换为对应的索引
        label = [self.label_map.index(char) for char in label]
        label = torch.as_tensor(label, dtype=torch.int64)

        return img, label, raw_len

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    images, labels, raw_lengths = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=LABEL_MAP_LEN)  # 使用LABEL_MAP_LEN作为填充值
    raw_lengths = torch.tensor(raw_lengths, dtype=torch.int64)
    return images, labels, raw_lengths

def get_train_test(train_path, test_path, label_map):
    train_dataset = MyDataset(train_path, label_map)
    test_dataset = MyDataset(test_path, label_map)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32, shuffle=True,
        num_workers=3, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    
    return train_loader, test_loader

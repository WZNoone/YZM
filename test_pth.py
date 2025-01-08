from PIL import Image
import os
import torch
import torchvision.transforms as T
from tqdm import tqdm
from network import Net
from NetWork.devide4or6 import Net4o6


IMAGE_SHAPE = (120, 50)

transform = T.Compose([
    T.Resize(IMAGE_SHAPE),
    T.ToTensor(),
    T.Normalize((0.647, 0.654, 0.653), (0.251, 0.250, 0.249))
])

LABEL_MAP = [i for i in '23456789ABCDEFGHIJKLMNPQRSTUVWXYZ\
    出毕例足析存炎枚吉汕色岸欣那佻函沃办孙画或妮只汾取宇忍邵向入同吃夕忽个兄佳以弘丰们戊伽羽委日庚幻江仓物有舟过空形告序讦三别卧板米执戎许古丹呈幺垂居青达金卷沛六池考忠纤皮枕果壮乙些门亢佩沈宠来后夜自伯长刺局朵巫飞风去抒打言弦切直扶两坤刀宙农宏杏侗左击与乳京宝和什杉邦伸我兔投冶欢竹供奈甸屉朋妻然始放此昌仪负共何力制昆史迅布代旦助冈牙朱昏百处即夫父邓廷往身中礼尽耳伺巾甲二枇必开全岩犬刷岐一杖芒广念奇昂住汉关常侄先动改分审尺互兵亿政设仗央岱邑固佐示甘幼典决佬吏券宗欠汰且材用明记弛叶臣实闪对厂书李快义岳仇凡又未访沙引弋才卓讯付瓜月忝妲至伶光你吝系卯帙呢吴友更利彤吻于到目杷豆坦兆少九坐丸圻裏昀房功咏采冽气究优低阪肋攻训弓凤文扮均侍忘术沐币伏孝府手冰字无仙企火姑杯松徇圣戏电年抑味承侏丛状皂奉化让列井妞斗季汪龙君舌弟亟受予土岭扑技妆为讼杞议虫也亘行内乞旧沁吞生妯军田虎尤寻妓卦刑瓦亚旺床官估侈阜纠孕的寸定十店帖位合所尚七务佯吟坡宕北吸洽末台使扒牛汁帛享炉爷呆方公判卒早纪毛辰驰亡役昔久坎托升宛禾佛么初冲佰驭会岢艾兴式良干协其效斥权武在鸟红炊刻四版具八服作并屈巧上姒幸成车盲仟弄千冷母完尾仕忏男肉岑戈立云坪步届今枝夭死非杜巨侃习匹束场句贞仔人刹汇赤回争问占杼伴吾他尧传华林斤扣闭外卫工姓角户地勾东衣午炙希由贝孟私当血研妾羊劣万失灰依西发孔征区玉巡含事忙肌白宜表舍心兰汐讲子杀木乡尸产机命佟易页汝见杰业志则约拈佃杭水歹王儿冬乎老归村抗伟酉吵姐扔丑秀下玖灭底坊竺众专及五本宋齐周吹而知努姗丁丈牧彼杆邪社']

Max_label_len = 4



# 是否使用GPU
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
model = Net(label_len = Max_label_len, label_map = LABEL_MAP)

model4o6 = Net4o6()
model.to(DEVICE)
# 载入训练模型
model.load_state_dict(torch.load("./pth/6ori0.9408.pth", map_location=torch.device('cpu')))
model.eval()


def captcha(im):
    im = transform(im)
    im = im.to(DEVICE)
    im = im.unsqueeze(0)
    out = model(im)
    out = out.view(-1, Max_label_len, len(LABEL_MAP))
    predict = torch.argmax(out, dim=2)
    label = predict.cpu().detach().numpy().tolist()[0]

    return ''.join(LABEL_MAP[x] for x in label)

local_path = r'D:\Program Files\Aisino\365Server\FPCY_img\01'
for filename in tqdm(os.listdir(local_path)):
    filepath = os.path.join(local_path, filename)
    # 调用方法
    im = Image.open(filepath)
    ret = captcha(im)
    ret = ret + '.jpg'
    newfilename = os.path.join(local_path, ret)
    try:
        os.rename(filepath, newfilename)
    except:
        os.remove(filepath)
        # print(filepath)


import onnx
import torch
import onnxsim
import numpy as np
import torchvision
from PIL import Image
import onnxruntime as ort
from torchvision.transforms import Compose
from NetWork.devide4or6 import Net4o6




def pth2onnx(pth):
    # 是否使用GPU
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cpu')
    model = Net4o6()
    model.to(DEVICE)

    # 载入训练模型
    model.load_state_dict(torch.load(pth))
    model.eval()

    # 创建一个随机张量作为示例
    x = torch.randn(1, 3, 120, 50, requires_grad=True)

    torch.onnx.export(model, x,
                    "6ori.onnx", # where to save the model (can be a file or file-like object)
                    #   opset_version=10,    # the ONNX version to export the model to
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']) # the model's output names)

    # 重新加载并检查模型，自动化简模型
    # model_onnx = onnx.load("model.onnx")
    # model_onnx, check = onnxsim.simplify(model_onnx,
    #                 dynamic_input_shape=False,
    #                 input_shapes=None)
    # onnx.save(model_onnx, "model.onnx")



def ort_test(image_path, onnx_model_path):
    
    transform_pipeline = Compose([
    torchvision.transforms.Resize((120, 50)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.6, 0.6, 0.6), (0.2, 0.2, 0.2))
    ])

    session = ort.InferenceSession(onnx_model_path)

    # 加载图片
    image = Image.open(image_path)
    image = transform_pipeline(image)

    # 获取模型输入节点的信息
    input_name = 'input'
    # 将图片转换为numpy数组
    image_data = np.array(image).astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)  # 添加批次维度


    # 运行模型并获取输出
    outputs = session.run(None, {input_name: image_data})
    output_data = outputs[0]
    print(output_data)



ort_test(r'224E.jpg', r'NetWork\devide4o6.onnx')
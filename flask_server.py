# 导入常用的库
import time
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,models
# 导入flask库的Flask类和request对象
from flask import request, Flask
import torch.nn.functional as F
from flask_cors import CORS
import torch.nn as nn
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #当前文件地址
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # 定义字典className_list，把种类索引转换为种类名称
classes = ['damaged','good']  #标签序号对应类名

#------------------------------------------------------1.加载模型--------------------------------------------------------------
num_classes = 2
class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):   # num_classes
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   # 从预训练模型加载mobilenet_v2网络参数
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1280, 1000),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = torchvision.models.vgg16_bn(pretrained=False)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=2)

# here
checkpoint = torch.load('epoch_40.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

#------------------------------------------------------2.获取测试图片--------------------------------------------------------------

# 根据图片文件路径获取图像数据矩阵
def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image


#------------------------------------------------------3.定义图片预处理--------------------------------------------------------------
# 模型预测前必要的图像处理
def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_chw = preprocess(input_image)
    return img_chw  # chw:channel height width

#------------------------------------------------------4.模型预测--------------------------------------------------------------
# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称

def predict(imageFilePath):
    img = Image.open(imageFilePath)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(img)

    model.eval()

    image = torch.reshape(image, (1, 3, 224, 224))  # 原图片是三维的，网络中图片要求是四维的

    with torch.no_grad():
        out = model(image)
    out = F.softmax(out, dim=1)
    out = out.data.cpu().numpy()  # 转换输出数据的类型
    print(out)
    a = int(out.argmax(1))
    plt.figure()
    list = ['damaged', 'good']
    print("Classes:{}:{:.1%}".format(list[a], out[0, a]))
    return list[a]



#------------------------------------------------------5.服务返回--------------------------------------------------------------
# 定义回调函数，接收来自/的post请求，并返回预测结果
@app.route("/", methods=['POST'])
def return_result():
    startTime = time.time()
    received_file = request.files['file']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = './resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        usedTime = usedTime * 1000
        print('接收图片并保存，总共耗时%.2fms' % usedTime)
        startTime = time.time()
        print(imageFilePath)
        result = predict(imageFilePath)
        usedTime = time.time() - startTime
        print('完成对接收图片的检测，总共耗时%.2fms' % usedTime)
        print("testtest",result)
        result = result + str(' ') + str('%.2fms'%usedTime)
        return result

    else:
        return 'failed'


# 主函数

CORS(app, resources=r'/*')
if __name__ == "__main__":
    print('在开启服务前，先测试predict_image函数')
    print('\n')
    app.run("127.0.0.1", port=4399)



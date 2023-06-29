import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from models import *
from src.train import net

from src.models import HighPassFilter

device = torch.device('cpu')



# 测试函数
def validate(test_loader, net, threshold=1.0):
    data_buf = [] #存储从测试数据加载器 test_loader 中获取的数据项，每个数据项代表一个图像样本及其标签
    y_true = [] #保存真实的标签值，在每次对比两个图像时，根据它们的标签是否相同，将相应的标签值（0或1）添加到 y_true 列表中。
    y_pred = [] #保存预测的标签值。根据计算得到的特征距离与阈值的比较结果，将相应的预测标签值（0或1）添加到 y_pred 列表中。
    for it in test_loader:
        data_buf.append(it)
        # max_rounds = 50;
    for i, it1 in enumerate(data_buf):
        # if i >=max_rounds:
        #     break
        for j, it2 in enumerate(data_buf):
            if i != j:
                y = int(it1[1] == it2[1])
                x0, x1 = it1[0], it2[0]
                output1, output2 = net(x0.to(device), x1.to(device))
                euclidean_distance = F.pairwise_distance(output1, output2)
                pred = int(euclidean_distance < threshold)
                y_true.append(y)
                y_pred.append(pred)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    loss = F.binary_cross_entropy(torch.tensor(y_pred, dtype=torch.float), torch.tensor(y_true, dtype=torch.float))
    #二元交叉熵损失

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Loss:", loss.item())


testing_dir = "../data/orl_faces/test/"  # 测试集地址
transform_test = transforms.Compose([#图像预处理
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.4515), (0.1978)),
    transforms.GaussianBlur(3),
    HighPassFilter()
])
dataset_test = torchvision.datasets.ImageFolder(testing_dir, transform=transform_test)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=1)

model_path = './model/t.pt'
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()
validate(test_loader, net, threshold=0.8)
import torch
import torchvision
import torch.nn as nn
from matplotlib import MatplotlibDeprecationWarning
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
# 消除警告关于Matplotlib的支持和未来版本的兼容性的提示。
import warnings

from src.models import ResBlock, ResNet

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from models import *

# 定义一些超参
train_batch_size = 8  # 训练时batch_size
train_number_epochs = 50  # 训练的epoch


# 这个函数用于展示一幅图像。它接收一个tensor格式的图像作为输入，并将其转换为ndarray并显示出来。可以通过设置text参数添加文本说明，
# should_save参数用于判断是否保存图像，path参数指定保存图像的路径。
def imshow(img, text=None, should_save=False, path=None):
    # 展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy()  # 将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换为(H,W,C)
    if path:
        plt.savefig(path)
    plt.show()


# 这个函数用于绘制损失变化图。它接收两个列表作为输入，分别是迭代次数和损失值。函数会将损失值随迭代次数的变化绘制成图像，并可以通过path参数保存为文件。
def show_plot(iteration, loss, path=None):
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.ylabel("loss")
    plt.xlabel("batch")
    if path:
        plt.savefig(path)
    plt.show()


# 定义文件dataset
training_dir = "../data/orl_faces/train/"  # 训练集地址
# ImageFolder对象，用于加载训练集图像数据
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

# 定义图像dataset
# Compose对象，用于定义图像的预处理操作。
transform = transforms.Compose([transforms.Resize((100, 100)),
                                # 有坑，传入int和tuple有区别，如果传入一个整数作为参数，例如 transforms.Resize(100)，那么该函数会将图像的短边缩放到指定大小，然后保持原始图像的宽高比进行等比例缩放。这意味着图像的长边可能会超过指定的大小。
                                transforms.ToTensor(),  # 将 PIL 图像或 NumPy 数组转换为 PyTorch 的张量（Tensor）格式。
                                transforms.Normalize((0.4515), (0.1978)),  # 标准化，均值，标准差
                                transforms.GaussianBlur(3),  # 高斯模糊
                                HighPassFilter()])  # 高通滤波器
# SiameseNetworkDataset对象，用于定义图像数据集。它接收一个imageFolderDataset和transform作为参数，并可以设置是否翻转图像以及正样本的比例。
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False, pos_rate=0.5)

# 定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,  # 是否打乱数据和参数的顺序
                              batch_size=train_batch_size)


# 自定义ContrastiveLoss对比损失函数
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()  # 调用父类（torch.nn.Module）的构造函数，以确保正确地初始化 ContrastiveLoss 类。
        self.margin = margin

    # 通过计算两个样本间的欧氏距离F.pairwise_distance来衡量它们之间的相似度，并根据标签来计算损失值。
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


device = torch.device('cpu')

# net = SiameseNetwork().to(device) #定义模型
# net = leNet().to(device)
# net = mcnn().to(device)
net = ResNet(ResBlock).to(device)
criterion = ContrastiveLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器

counter = []  # 存储迭代次数
loss_history = []  # 存储损失值
iteration_number = 0

# 开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)  # 数据移至CPU
        optimizer.zero_grad()  # 清空梯度信息，准备进行反向传播
        output1, output2 = net(img0, img1)  # 对img0和img1进行前向传播
        loss_contrastive = criterion(output1, output2, label)  # 调用损失函数criterion计算输出和标签之间的对比损失loss_contrastive
        loss_contrastive.backward()  # 反向传播，计算参数的梯度
        optimizer.step()  # 根据梯度更新模型的参数
        if i % 10 == 0:
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))

show_plot(counter, loss_history, 'output/loss.jpg')

model_path = './model/t.pt'
torch.save(net.state_dict(), model_path)
import sys
import os
import random
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage

# 加载模型
from src.models import HighPassFilter, ResBlock
from src.models import ResNet

model_path = './model/t.pt'
device = torch.device('cpu')
net = ResNet(ResBlock)
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

# 创建应用程序窗口
class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face Recognition App')
        self.resize(800, 400)  # 调整窗口大小
        self.layout = QVBoxLayout()

        self.image_layout = QHBoxLayout()  # 图片布局
        self.label1 = QLabel()
        self.label2 = QLabel()
        self.image_layout.addWidget(self.label1)
        self.image_layout.addWidget(self.label2)
        self.layout.addLayout(self.image_layout)

        self.select_button = QPushButton('Select Image')
        self.select_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_button)

        self.next_button = QPushButton('Next Image')
        self.next_button.clicked.connect(self.next_image)
        self.layout.addWidget(self.next_button)

        self.compare_button = QPushButton('Compare')
        self.compare_button.clicked.connect(self.start_comparison)
        self.layout.addWidget(self.compare_button)

        self.result_label = QLabel()  # 添加结果标签
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)
        self.show()

        self.selected_image = None  # 存储用户选择的图片路径
        self.current_image_path = None  # 当前展示的测试集图片路径

        self.worker = ComparisonWorker()
        self.worker.resultReady.connect(self.update_result)

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg *.pgm)')#没啥用
        file_dialog.setOptions(options)
        if file_dialog.exec_():
            try:
                filenames = file_dialog.selectedFiles()
                self.selected_image = filenames[0]
                image = Image.open(self.selected_image)#Image.open() 方法打开图像文件，并将其转换为 PIL 图像对象
                qimage = self.convert_pil_to_qimage(image)
                self.show_image(qimage, self.label1)
            except Exception as e:
                print("An error occurred:", str(e))

    def next_image(self):
        if dataset_test.imgs:
            try:
                self.current_image_path = random.choice(dataset_test.imgs)
                image = Image.open(self.current_image_path)
                qimage = self.convert_pil_to_qimage(image)
                self.show_image(qimage, self.label2)
                self.result_label.clear()  # 清除结果标签
            except Exception as e:
                print("An error occurred while loading the next image:", str(e))

    def start_comparison(self):
        if self.selected_image is not None and self.current_image_path is not None:
            try:
                self.worker.set_images(self.selected_image, self.current_image_path)
                self.worker.start()
            except Exception as e:
                print("An error occurred while starting the comparison:", str(e))

    def update_result(self, similarity):#比较线程完成比较后，会发射一个信号，并将比较结果（相似性）传递给该函数。根据相似性的值，该函数会更新结果标签 result_label 的文本内容。
        if similarity:
            result_text = "Same person"
        else:
            result_text = "Different persons"
        self.result_label.setText(result_text)

    def show_image(self, image, label):
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)

    def convert_pil_to_qimage(self, pil_image):#对象转换，方便显示
        if pil_image.mode == 'RGB':
            image_format = QImage.Format_RGB888
        elif pil_image.mode == 'L':
            image_format = QImage.Format_Grayscale8
        else:
            raise ValueError("Unsupported image mode: {}".format(pil_image.mode))
        qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, image_format)
        return qimage

class ComparisonWorker(QThread):
    resultReady = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_image_path = None
        self.current_image_path = None

    def set_images(self, selected_image_path, current_image_path):
        self.selected_image_path = selected_image_path
        self.current_image_path = current_image_path

    def run(self):
        try:
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.4515,), (0.1978,)),
                transforms.GaussianBlur(3),
                HighPassFilter()
            ])

            selected_image = Image.open(self.selected_image_path).resize((100, 100))
            current_image = Image.open(self.current_image_path).resize((100, 100))

            selected_image_tensor = transform(selected_image).unsqueeze(0).to(device)
            current_image_tensor = transform(current_image).unsqueeze(0).to(device)

            threshold = 0.8 #相似性阈值参数(欧氏距离)
            output1, output2 = net(selected_image_tensor, current_image_tensor)
            euclidean_distance = F.pairwise_distance(output1, output2)
            similarity = euclidean_distance.item() < threshold
            self.resultReady.emit(similarity)
        except Exception as e:
            print("An error occurred in the worker thread:", str(e))


# 加载测试集数据
testing_dir = "../data/orl_faces/test/"
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4515,), (0.1978,)),
    transforms.GaussianBlur(3),
    HighPassFilter()
])

# print("Testing directory:", testing_dir)
# 读取测试集图片路径
image_paths = []
for root, dirs, files in os.walk(testing_dir):
    for file in files:
        if file.endswith(".pgm"):
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
            # print("Image path:", image_path)

dataset_test = torchvision.datasets.DatasetFolder(testing_dir, loader=lambda x: Image.open(x).convert("L"), extensions=('.pgm',),
                                                  transform=transform)
dataset_test.imgs = image_paths

# 启动应用程序
app = QApplication(sys.argv)
window = FaceRecognitionApp()
sys.exit(app.exec_())
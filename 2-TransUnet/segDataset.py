from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

"""
将16位图像转为8位图像
"""
def transfer_16bit_to_8bit(image_path):
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # print(max_16bit, min_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    # 或者下面一种写法
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    # print(image_16bit.dtype)
    # print('16bit dynamic range: %d - %d' % (min_16bit, max_16bit))
    # print(image_8bit.dtype)
    # print('8bit dynamic range: %d - %d' % (np.min(image_8bit), np.max(image_8bit)))
    return image_8bit

"""
依次读取图像的文件路径，并放到array中返回
"""
def get_dataPath(root, isTraining):
    if isTraining:
        root = os.path.join(root + "/Mito/train_imgs50")
        imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    else:
        root = os.path.join(root + "/Mito/test_imgs2")
        imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    return imgPath

"""
定义分割数据集
"""
class segDataset(Dataset):

    def __init__(self, root, isTraining=True):
        self.root = root
        self.isTraining = isTraining
        self.imgPath = get_dataPath(root, isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        filename = imgPath.split('/')[-1]
        image_8bit = transfer_16bit_to_8bit(imgPath)  # 16位转8位
        pil_image = Image.fromarray(image_8bit)  # numpy数组转PIL格式图像
        #pil_image = Image.open(imgPath)
        inputImage = pil_image.convert('RGB')
        #1模式:转化为为二值图像，非黑即白，每个像素用8个bit表示，0表示黑，255表示白。
        #L模式:转化为为灰色图像，每个像素用8个bit表示，0表示黑，255表示白，0~255代表不同的灰度。
        #需要注意的是，在PIL中，RGB是通过以下公式转化为L的：L = R * 299/1000 + G * 587/1000 + B * 114/1000
        if self.isTraining:
            labelPath = self.root + '/Mito/train_mask50/' + filename
            labelImage = Image.open(labelPath)
            """
            Data augmentation for training date, if needed.
            """
            p1 = np.random.randint(-45, 45)
            p2 = np.random.randint(0, 1)
            p3 = np.random.randint(0, 1)
            inputImage = inputImage.rotate(p1)
            label = labelImage.rotate(p1)
            resize_transform = transforms.Compose([
                # transforms.RandomCrop(512),
                transforms.Resize([512, 512]),
                transforms.RandomHorizontalFlip(p2),
                transforms.RandomVerticalFlip(p3),
                transforms.ToTensor()  # 将PIL.Image转化为tensor，即归一化。注：shape 会从(H，W，C)变成(C，H，W)
            ])
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            inputImage = resize_transform(inputImage)
            label = resize_transform(label)

        else:
            labelPath = self.root + '/Mito/test_mask2/' + filename
            labelImage = Image.open(labelPath)
            simple_transform = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor()
            ])
            inputImage = simple_transform(inputImage)
            label = simple_transform(labelImage)
        return inputImage, label

    def __len__(self):
        """
        返回总的图像数量
        """
        return len(self.imgPath)

# root = ".\\Mito\\test_imgs2"
# imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
# image_8bit = transfer_16bit_to_8bit(imgPath[0]) # 16位转8位
# pil_image=Image.fromarray(image_8bit) # numpy数组转PIL格式图像
# inputImage = pil_image.convert('RGB')
# #1模式:转化为为二值图像，非黑即白，每个像素用8个bit表示，0表示黑，255表示白。
# #L模式:转化为为灰色图像，每个像素用8个bit表示，0表示黑，255表示白，0~255代表不同的灰度。
# #需要注意的是，在PIL中，RGB是通过以下公式转化为L的：L = R * 299/1000 + G * 587/1000 + B * 114/1000
# label_root = ".\\Mito\\test_mask2"
# labelPath = list(map(lambda x: os.path.join(label_root, x), os.listdir(label_root)))
# label_8bit = transfer_16bit_to_8bit(labelPath[0]) # 16位转8位
# pil_label=Image.fromarray(label_8bit) # numpy数组转PIL格式图像
#
# p1 = np.random.randint(-45,45)
# p2 = np.random.randint(0,1)
# p3 = np.random.randint(0,1)
# print(p1,p2,p3)
# inputImage = inputImage.rotate(p1)
# label = pil_label.rotate(p1)
# resize_transform = transforms.Compose([
#     #transforms.RandomCrop(512),
#     transforms.Resize([512, 512]),
#     transforms.RandomHorizontalFlip(p2),
#     transforms.RandomVerticalFlip(p3),
#     transforms.ToTensor()#将PIL.Image转化为tensor，即归一化。注：shape 会从(H，W，C)变成(C，H，W)
# ])
# seed = np.random.randint(2147483647)
# np.random.seed(seed)
# simple_transform = transforms.ToTensor()
# inputImage = resize_transform(inputImage)
# label = resize_transform(label)
# transforms.ToPILImage()(inputImage).show()
# transforms.ToPILImage()(label).show()

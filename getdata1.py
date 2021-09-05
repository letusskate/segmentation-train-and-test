import time
import copy
import torch
from torch import optim, nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
# from IPython import display
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") # 忽略警告
import os,glob,csv,random

#划分数据集，将文件名（不包括后缀）存入train.txt val.txt test.txt
def load_txt(root="../iccv09Data",trainprop=0.8,valprop=0.2,testprop=0.2):
    """
    :param filename:
    :return:
    """
    if not os.path.exists(os.path.join(root,'Segmentation')):
        os.mkdir(os.path.join(root,'Segmentation'))
    # 是否已经存在了txt文件
    if not os.path.exists(os.path.join(root,"Segmentation", "test.txt")):
        imgs = []
        imgs += glob.glob(os.path.join(root,"images","*.png"))
        imgs += glob.glob(os.path.join(root,"images","*.jpg"))
        imgs += glob.glob(os.path.join(root,"images","*.jpeg"))
        # labels = []
        # labels += glob.glob(os.path.join(root, "labels", "*.xml"))[20:-4]
        # labels += glob.glob(os.path.join(root, "labels", "*.json"))[20:-5]
        #
        # 将元素打乱
        random.shuffle(imgs)
        begin={"train":0,"val":trainprop,"test":trainprop+valprop}
        end={"train":trainprop,"val":trainprop+valprop,"test":1}
        for mode in ["train","val","test"]:
            with open(os.path.join(root,"Segmentation",mode+".txt"), mode="w", newline="") as f:
                for img in imgs[int(begin[mode]*len(imgs)):int(end[mode]*len(imgs))]:  # 'pokemon/pikachu/00000058.png'
                    img=os.path.split(os.path.splitext(img)[0])[-1]#最外层分离路径和文件名，内层分离文件后缀和文件
                    f.write(img+"\n")
                print("writen into csv file:{} ".format(mode+".txt"))

#从划分的数据集的txt中读取 图片 和 标签
def read_voc_images(root="../iccv09Data", is_train=True, max_num=None):
    txt_fname = '%s/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split() # 拆分成一个个名字组成list
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # 读入数据并且转为RGB的 PIL image
        try:
            features[i] = Image.open('%s/images/%s.jpg' % (root, fname)).convert("RGB")
        except:
            try:
                features[i] = Image.open('%s/images/%s.png' % (root, fname)).convert("RGB")
            except:
                try:
                    features[i] = Image.open('%s/images/%s.jpeg' % (root, fname)).convert("RGB")
                except:
                    print("Can't read the picture")
                    continue
        try:
            labels[i] = Image.open('%s/labels/%s.png' % (root, fname)).convert("RGB")
        except:
            print("Can't read the label")
    return features, labels # PIL image 0-255

# 这个函数可以不需要
def set_figsize(figsize=(3.5, 2.5)):
    """在jupyter使用svg显示"""
    # display.set_matplotlib_formats('svg')
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]  # colormap 映射 到colormaplabel中计算的下标


# y = voc_label_indices(train_labels[0], colormap2label)
# print(y[100:110, 130:140])  # 打印结果是一个int型tensor，tensor中的每个元素i表示该像素的类别是VOC_CLASSES[i]


# 预处理数据
def voc_rand_crop(feature, label, height, width):
    """
	随机裁剪feature(PIL image) 和 label(PIL image).
	为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
	Get parameters for ``crop`` for a random crop.
	Args:
		img (PIL Image): Image to be cropped.
		output_size (tuple): Expected output size of the crop.
	Returns:
		tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
	"""
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


# 显示n张随机裁剪的图像和标签，前面的n是5
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# show_images(imgs[::2] + imgs[1::2], 2, n);


class KiwiDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, kiwidir, colormap2label, max_num=None):
        """
		crop_size: (h, w)
		"""
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.5, 0.5, 0.5])
        self.rgb_std = np.array([0.5, 0.5, 0.5])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        self.crop_size = crop_size  # (h, w)
        load_txt(root=kiwidir, trainprop=0.8, valprop=0.2, testprop=0)
        features, labels = read_voc_images(root=kiwidir, is_train=is_train, max_num=max_num)
        # 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features)  # PIL image
        self.labels = self.filter(labels)  # PIL image
        self.colormap2label = colormap2label
        print('{} read '.format('Trainset' if is_train else 'Valset') + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
                img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # float32 tensor           uint8 tensor (b,h,w)
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

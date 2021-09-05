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
import os
from PIL import Image

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
COLORMAP = [[0,0,0],[128,0,0]]
pthfile=r"models/bestmodel.pth"
net=torch.load(pthfile).to(device)
# print(net)


def label2image(pred):
    # pred: [320,480]
    colormap = torch.tensor(COLORMAP,device=device,dtype=int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()



imgpath=r'C:\Users\linbin\PycharmProjects\王宇博\yjyyfg\data\finalouter\images'
for file in tqdm(os.listdir(imgpath),total=len(os.listdir(imgpath))):
    n=4
    img=os.path.join(imgpath,file)
    labelpath = img.replace('images','labels')
    img=Image.open(img).convert("RGB")
    label=Image.open(labelpath).convert("RGB")
    # i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=(1000, 1000))
    # img = torchvision.transforms.functional.crop(img, i, j, h, w)
    tsf=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=np.array([0.5, 0.5, 0.5]),\
                                         std=np.array([0.5, 0.5, 0.5]))])
    input=tsf(img).unsqueeze(0).to(device)
    output=torch.argmax(net(input),dim=1)
    pred=label2image(output).squeeze(0)
    _, axes = plt.subplots(3)
    axes[0].imshow(img)
    axes[1].imshow(label)
    axes[2].imshow(pred)
    plt.show()
    # imgs=[img,pred]
    # show_images(img,1,2)
    # show_images(imgs,3,n)
    
def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize,return_type='dict')
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


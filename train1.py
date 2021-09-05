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
warnings.filterwarnings("ignore")  # 忽略警告
from getdata1 import KiwiDataset
from deeplab_xception import DeepLabv3_plus
import os

# 根据自己存放数据集的路径修改voc_dir
wikidir = r"C:\Users\linbin\PycharmProjects\王宇博\yjyyfg\data\finalouter"
# train_features, train_labels = read_voc_images(voc_dir, max_num=10)
# n = 5  # 展示几张图像
# imgs = train_features[0:n] + train_labels[0:n]  # PIL image
# show_images(imgs, 2, n)

# 标签中每个RGB颜色的值
COLORMAP = [[0,0,0],[128,0,0]]
# 标签其标注的类别
CLASSES = ['background', 'outer']

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)  # torch.Size([16777216])
for i, colormap in enumerate(COLORMAP):
    # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

batch_size = 2  # 实际上我的小笔记本不允许我这么做！哭了（大家根据自己电脑内存改吧）
crop_size = (1000, 1000)  # 指定随机裁剪的输出图像的形状为(320,480)
max_num = 20000  # 最多从本地读多少张图片，我指定的这个尺寸过滤完不合适的图像之后也就只有1175张~

# 创建训练集和测试集的实例
voc_train = KiwiDataset(True, crop_size, wikidir, colormap2label, max_num)
voc_test = KiwiDataset(False, crop_size, wikidir, colormap2label, max_num)

# 设批量大小为32，分别定义【训练集】和【测试集】的数据迭代器
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                         drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                                        num_workers=num_workers)

# 方便封装，把训练集和验证集保存在dict里
dataloaders = {'train': train_iter, 'val': test_iter}
dataset_sizes = {'train': len(voc_train), 'val': len(voc_test)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(COLORMAP)  # 21分类，1个背景，20个物体

#建立网络
model_ft = DeepLabv3_plus(nInputChannels=3, n_classes=num_classes, os=16, pretrained=True, _print=True)  # 设置True，表明要加载使用训练好的参数

# # 特征提取器
# for param in model_ft.parameters():
#     param.requires_grad = False
#
# model_ft = nn.Sequential(*list(model_ft.children())[:-2],  # 去掉最后两层
#                          nn.Conv2d(512, num_classes, kernel_size=1),  # 用大小为1的卷积层改变输出通道为num_class
#                          nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)).to(
#     device)  # 转置卷积层使图像变为输入图像的大小
#
# # net = model_ft  # 自己加的
# # # 对model_ft做一个测试
# # x = torch.rand((2, 3, 320, 480), device=device)  # 构造随机的输入数据
# # print(net(x).shape)  # 输出依然是 torch.Size([2, 21, 320, 480])
#
#
# # 打印第一个小批量的类型和形状。不同于图像分类和目标识别，这里的标签是一个三维数组
# # for X, Y in train_iter:
# #     print(X.dtype, X.shape)
# #     print(Y.dtype, Y.shape)
# #     break
#
# # 双线性插值的上采样，用来初始化转置卷积层的卷积核
# def bilinear_kernel(in_channels, out_channels, kernel_size):
#     factor = (kernel_size + 1) // 2
#     if kernel_size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:kernel_size, :kernel_size]
#     filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
#     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
#     weight[range(in_channels), range(out_channels), :, :] = filt
#     weight = torch.Tensor(weight)
#     weight.requires_grad = True
#     return weight
#
#
# nn.init.xavier_normal_(model_ft[-2].weight.data, gain=1)
# model_ft[-1].weight.data = bilinear_kernel(num_classes, num_classes, 64).to(device)

#训练
def train_model(model: nn.Module, criterion, optimizer, scheduler, num_epochs=20):
    print("Start Training!")
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('records'):
        os.mkdir('records')
    if not os.path.exists('models'):
        os.mkdir('models')
    with open(r"records/acc.txt", "w") as f:
        with open(r"records/log.txt", "w")as f2:
            # 每个epoch都有一个训练和验证阶段
            for epoch in range(num_epochs):
                batch_cnt = len(dataloaders['train'])
                print('Epoch {}/{} together {} batchs'.format(epoch+1, num_epochs,batch_cnt))
                print('-' * 30)
                for phase in ['train', 'val']:
                    if phase == 'train':
                        scheduler.step()
                        model.train()
                    else:
                        model.eval()
                    runing_loss = 0.0
                    runing_corrects = 0.0


                    # 迭代一个epoch
                    for i,(inputs,labels) in enumerate(dataloaders[phase],0):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()  # 零参数梯度
                        # 前向，只在训练时跟踪参数
                        with torch.set_grad_enabled(phase == 'train'):
                            logits = model(inputs)  # [5, 21, 320, 480]
                            loss = criterion(logits, labels.long())
                            # 后向，只在训练阶段进行优化
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        # 统计loss和correct
                        thisloss = loss.item() * inputs.size(0)
                        thiscorrect = torch.sum((torch.argmax(logits.data, 1)) == labels.data) / (crop_size[0]*crop_size[1])
                        runing_loss += thisloss
                        runing_corrects += thiscorrect
                        if phase == 'train':
                            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f '% (epoch + 1, (i + 1 + epoch * batch_cnt),
                                                                                   thisloss/batch_size , thiscorrect/batch_size))
                            f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f '% (epoch + 1, (i + 1 + epoch * batch_cnt),
                                                                              thisloss/batch_size, thiscorrect/batch_size))
                            f2.write('\n')
                            f2.flush()

                    epoch_loss = runing_loss / dataset_sizes[phase]
                    epoch_acc = runing_corrects.double() / dataset_sizes[phase]
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    f.write("EPOCH=%03d,Accuracy= %.3f" % (epoch + 1, epoch_acc))
                    f.write('\n')
                    f.flush()
                    # 深度复制model参数
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model,'models/bestmodel.pth')
                        f3 = open("records/best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f" % (epoch+1, best_acc))
                        f3.close()
                torch.save(model,'models/epoch{}.pth'.format(epoch+1))
                print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # 加载最佳模型权重
    # model.load_state_dict(best_model_wts)
    model=model.load('models/bestmodel.pth')
    return model


epochs = 2000  # 训练5个epoch
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
# 每3个epochs衰减LR通过设置gamma=0.1
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device)
# 开始训练
model_ft = train_model(model_ft, criteon, optimizer, exp_lr_scheduler, num_epochs=epochs)

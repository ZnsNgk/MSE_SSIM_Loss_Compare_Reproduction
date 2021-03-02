import os, sys
import cv2
import torch
import math
import numpy
from model import net
from SSIM_loss import SSIM_loss
import matplotlib.pyplot as plt
import torch.nn as nn

input_pic = cv2.imread("input.bmp")
input_pic = numpy.array(input_pic,dtype='float32')
input_pic = torch.from_numpy(input_pic)
input_pic = input_pic.permute(2,0,1).unsqueeze(0)
input_pic = input_pic.contiguous()
ori_pic = cv2.imread("test.png")
ori_pic = numpy.array(ori_pic,dtype='float32')
ori_pic = torch.from_numpy(ori_pic)
ori_pic = ori_pic.permute(2,0,1).unsqueeze(0)
ori_pic = ori_pic.contiguous()

output_folder = "./SSIM_Loss/"  #在不输入参数的情况下默认选择SSIM损失
loss_func = SSIM_loss()
normal = 1.0

args = sys.argv[1:]
for arg in args:
    arg = arg[1:]
    if arg == "SSIM":
        output_folder = "./SSIM_Loss/"
        loss_func = SSIM_loss()
        normal = 1.0
    elif arg == "MSE":
        output_folder = "./MSE_Loss/"
        loss_func = nn.MSELoss()
        normal = 65025.0
    elif arg == "L1":
        output_folder = "./L1_Loss/"
        loss_func = nn.L1Loss()
        normal = 255.0

model = net()
lr = 5e-2   #学习率设置为0.05
optim = torch.optim.Adam(model.parameters(),lr) #定义优化器为Adam
loss_value = math.inf   #初始化损失值为无穷

isExists = os.path.exists(output_folder)    #判断文件夹是否存在
if isExists:
    del_list = os.listdir(output_folder)    #运行前先清理目标文件夹内数据
    for file in del_list:
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(output_folder)

for step in range(1500):    #迭代1500次
    optim.zero_grad()
    out = model(input_pic)
    loss = loss_func(out, ori_pic)
    loss.backward()
    optim.step()
    out = out.permute(0,2,3,1).squeeze(0).detach()
    out = numpy.array(out)
    filename = str(step).zfill(4)   #将输出文件名扩展为4位(方便整理成视频时作为图像序列)
    cv2.imwrite(output_folder + filename + ".png", out)     #先写入再读取防止float转uint8时溢出
    out = cv2.imread(output_folder + filename + ".png")
    out = out[:,:,::-1]     #BGR转RGB
    loss_value = float(loss)
    loss_value = loss_value / normal
    s = 'Step ' + str(step) + ', Loss = ' + str(round(loss_value,4))
    print(s)
    plt.title(s)
    plt.imshow(out)
    plt.savefig(output_folder + filename + ".png",dpi=150)
    plt.close()

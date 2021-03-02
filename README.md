# SSIM损失收敛性实验

---

本实验基于PyTorch，对[**compare_ssim_mse**](https://github.com/CharlesNord/pytorch-ssim/blob/master/compare_ssim_mse)进行了复现，输入随机噪声，使用MSE、L1Loss和SSIM_Loss三种损失函数对噪声进行优化，通过反向传播和梯度下降法，最终重建出原始图像

[原文链接](https://www.cnblogs.com/king-lps/p/12248912.html)

---

## 原理解释

本实验通过构建一个深度学习网络实现，该网络仅包含一个LinearWise层，通过对输入图像求哈达玛积然后输出，其参数通过反向传播和梯度下降法进行优化，最终使得输出图像尽可能接近原始图像

---

## 使用方法

本实验使用cpu运行，不需要支持cuda的独立显卡

若运行示例图片则可直接跳转到第四步

1) 首次运行时将原始图片名称改为“test.png”（**彩色图像**）放入根目录，然后在属性查看其**宽度**和**高度**

2) 进入made_mosaic.py文件，将`pic = numpy.random.rand(321, 481, 3)`中rand内参数改为**高度**(h)、**宽度**(w)、通道数(c)，即`pic = numpy.random.rand(h, w, c)`，彩色图像的通道数为3，修改后运行该文件

3) 进入model.py文件，将`self.lw = LinearWise(321*481*3,bias=False)`修改为`self.lw = LinearWise(h*w*c,bias=False)`，再将`x = x.view(1, 3, 321, 481)`修改为`x = x.view(1, 3, h, w)`

4) 运行“main.py”文件，该文件需要在运行时添加后缀以进行不同损失函数的选择，格式为：

`python main.py -lossFunction`

lossFunction：损失函数，可选择MSE、SSIM和L1三种，默认为SSIM，即

```
python main.py -MSE   #选择MSE作为损失函数
python main.py -SSIM  #选择SSIM_Loss作为损失函数
python main.py -L1    #选择L1_Loss作为损失函数
```

main.py文件运行时会执行1500次迭代，每次迭代的输出会以png的格式保存在相关文件夹内

5) 若要做成视频或者与原图进行对比，则可以生成与输出图片格式相同的原图，在命令行输入`python made_origin.py`即可，其会在根目录生成“Origin.png”图片

---

## 文件构成

L1_Loss文件夹：存放L1损失产生的输出文件

MSE_Loss文件夹：存放MSE损失产生的输出文件

SSIM_Loss文件夹：存放SSIM损失产生的输出文件

input.bmp：随机生成的噪声图片

made_mosaic.py：生成随机噪声图片代码，运行后生成“input.bmp”

made_origin.py：生成与输出图片格式相同的原始图片，运行后生成“Origin.png”

main.py：主文件

model.py：模型文件

SSIM_loss.py：SSIM损失函数文件，来自于[pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py)

test.png：原始图片

---

## 运行环境

```
python                        3.7.6
opencv-python                 4.2.0.34
matplotlib                    3.2.2
torch                         1.5.1
torchvision                   0.6.1
numpy                         1.19.5
```


import GetData
from MyDataSetClass import *
import torch
import torchvision
from torchvision import transforms
import os
import time
import MyNeuralNetwork
import PIL
path_file=GetData.GetImgPath(r'C:\Users\46501\source\repos\LeNet\LeNet\data','path.txt')#第一个参数为图片数据库所存的位置，第二个参数为存储文件信息的文件的名称，返回值为存储文件信息的文件的路径
print(path_file)
trasnforms1=transforms.Compose(
    [   transforms.Resize((32,32)),#图片全部转换为32*32格
        #做数据增强
        transforms.RandomHorizontalFlip(0.5),#0.5的概率左右交换
        transforms.RandomVerticalFlip(0.5),#0.5的概率上下交换
        transforms.RandomGrayscale(0.5),#0.5的概率改变亮度
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#数据标准化
        ]
    )
DataSet1=MyDataSet1(path_file,transforms=trasnforms1) #建立一个Dataset实例，path_file为存储数据信息的文件的位置
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader=torch.utils.data.DataLoader(DataSet1,batch_size=2,shuffle=True)
Net1=MyNeuralNetwork.VGGNet16()
Net1=Net1.to(device)
Net1.train_sgd(device,trainloader,'weights.pth',100)
img1=trasnforms1(PIL.Image.fromarray(cv2.cvtColor(cv2.imread(r'C:\Users\46501\source\repos\LeNet\LeNet\data\02\u=3598895450,1009063269&fm=26&gp=0.jpg'),cv2.COLOR_BGR2RGB)))
print()
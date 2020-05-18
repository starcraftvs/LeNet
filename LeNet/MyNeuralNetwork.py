import torch
import torchvision                                                                                                                                           
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import msvcrt

#定义一个父类Net，以减少代码复用率
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
    #定义训练
    def train_sgd(self,device,trainloader,model_path,epoches):
        #初始化部分
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) #定义优化器：种类为Adam，学习率lr=0.0001
        path = model_path   #用一个变量规定模型参数的存储文件名及路径
        initepoch = 0           #初始化epoch参数，从0开始
        
        #新建或载入模型
        if os.path.exists(path) is not True:   #若没有已保存的现有模型参数，则新建
            loss = nn.CrossEntropyLoss()       #新建loss参数，选择为交叉熵类

        else:
            checkpoint = torch.load(path)      #若已有，则导入模型，存在checkpoint变量里，该变量为一个dict
            self.load_state_dict(checkpoint['model_state_dict'])     #load模型结构参数
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    #load模型状态参数
            initepoch = checkpoint['epoch']        #load现在已经运行了多少个epoch
            loss = checkpoint['loss']              #load现在的loss（看上面，是一个类，而不是一个数值）

        #训练模型
        for epoch in range(initepoch,epoches):  #跑定义的epoches数量次epoches
            timestart = time.time()#记录时间
            running_loss = 0.0 #用于记录running_loss的数值，用以判断模型精度
            total = 0          #记录总共的统计数据
            correct = 0        #记录预测正确的统计数据
            for i, data in enumerate(trainloader, 0): #跑一遍epoch，跑i次batches
                #数据获取
                inputs, labels = data   #data是从trainloader中返回的一个btach，包括输入以及标签
                inputs, labels = inputs.to(device),labels.to(device) #用.to()函数，将结果输入device（device是自己定义的可以看出这个函数是要从外部输入device参数的）

                #每一次batch要将梯度清0，否则每一个batch的数据不一样，上一个batch的梯度对这个batch是没有意义的,因为上面已经创建了实例优化器，此处利用其进行操作即可
                optimizer.zero_grad()

                # 前向传播，反向传播以及优化
                outputs = self(inputs)  #调用类自身就相当于函数，可以跑出outputs?具体需要看nn.Module
                l = loss(outputs, labels)#用定义为交叉熵类的loss来计算loss的具体值，根据输出和自己定义的标签来确定差值自带反向传播函数
                l.backward()
                #用optimzer.step()后模型会更新，每个mini_batch之后最好都要调用，不然这个batch就是白跑了
                optimizer.step()
                running_loss =(running_loss+l.item())/(i+1) #计算统计数据running_loss，其为500个batches后的loss和，下面利用其计算均值
            # print("i ",i)
                total+=labels.size()[0]
                _,predicted=torch.max(outputs.data,1)
                correct+=(predicted==labels).sum().item()
                if i%2==1: 
                    torch.save({'epoch':epoch,
                                'model_state_dict':self.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':loss
                                },path)                 #保存模型参数，到path
            print("Accuracy is %3f" %(100*correct/total))
            print("loss is %d" %(running_loss))
            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))    #跑了多少epoches及跑一个epoch需要的时间

        print('Finished Training')        #跑完100遍epoches，输出完成训练
    #跑测试集
    def test(self,device,testloader):                #定义测试网络效果
        correct = 0                       #正确的数量为0
        total = 0                         #总数量为0
        with torch.no_grad():                #with语句相当于try-finally
            for data in testloader:          #同上，跑一遍测试
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the all test images: %.3f %%' % (
                100.0 * correct / total))                    #输出测试结果  


#定义神经网络
#VGGNet-16
class VGGNet16(Net):
    def __init__(self):
        #nn.Module是一个特殊的类，它的子类必须在构造函数中执行父类的构造函数（具体原因要以后看）
        super(VGGNet16,self).__init__()#等价于nn.Module.__init__(self)，但推荐使用第一种写法
        #接下里定义神经网络，使用函数nn.Conv2d(此时输入的都是规范的32*32*3的图片)    
        self.conv1=nn.Conv2d(3,64,3,padding=1)      
        #nn.Conv2d(in_channels:输入特征矩阵的通道数Cin，out_channels:输出特征矩阵的通道数Cout,kernal_size：卷积核的大小，padding:边缘的扩充，默认
        #值为0，代表使用0进行扩充，dilation:内核间的距离，默认为1，groups：组数，默认为1，bias:要不要加偏差，默认为True
        self.conv2=nn.Conv2d(64,64,3,padding=1)
        self.pool1=nn.MaxPool2d(2,2)#池化层的矩阵大小
        #nn.MaxPool2d(kernal_size:窗口大小（可以是int或者tuple）,stride:(int/tuple)，步长，默认为kernal_size，padding:输入的每一条边补充0的层数？
        #真的？，dilation:内核间的距离，也有叫控制步幅的参数，return_indices：如果True，会返回输出最大值的序号，默认False，ceil_mode:True的话计算输出信号
        #大小的时候向上取整，默认为False，向下取整)
        self.bn1=nn.BatchNorm2d(64) #用于标准化输出参数，比较复杂，以后慢慢看，目前姑且认为输入为feature_map的数量
        self.relu1=nn.ReLU()#激活函数为ReLU函数，具体参数以后看
        #往下按照VGG-16Net继续往下搭
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        #对得出的矩阵做线性变换 明天再细看
        self.fcl4=nn.Linear(512*4*4,1024)#真的吗？那图片应该不是32*32的把？
        self.drop1=nn.Dropout2d()#效果上来讲，使得得到的矩阵一部分行随机为0。另一个角度，即使得部分特征随机被消除，
        #也即抑制了部分神经节点，使得结果不容易过拟合
        self.fcl5=nn.Linear(1024,1024)#再次线性转换
        self.drop2=nn.Dropout2d()#同上
        self.fcl6=nn.Linear(1024,10)#因为结果有10分类，最后得到的是对于10种分类的不同概率，下面取最大概率即为预测结果
    def forward(self,x):#定义前向传播，即把上面定义的神经网络全部跑一遍
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fcl4(x))
        x = self.drop1(x)
        x = F.relu(self.fcl5(x))
        x = self.drop2(x)
        x = F.relu(self.fcl6(x))

        return x
    def train_sgd(self,device,trainloader,model_path,epoches):
        super(VGGNet16,self).train_sgd(device,trainloader,model_path,epoches)
    def test(self,device,testloader):
        super(VGGNet16,self).test(device,testloader)

#定义LeNet5
class LeNet5(Net):
    #定义结构
    def __init__(self):
        super(LeNet5,self).__init__()
        #第一卷积层
        self.conv1=nn.Conv2d(3,6,2,padding=2)
        self.pool1=nn.MaxPool2d(2,2)
        self.bn1=nn.BatchNorm2d(6)
        self.relu1=nn.ReLU()
        #第二卷积层
        self.conv2=nn.Conv2d(6,16,2)
        self.pool2=nn.MaxPool2d(2,2)
        self.bn2=nn.BatchNorm2d(16)
        self.relu2=nn.ReLU()
        #全连接层
        self.fcl1=nn.Linear(16*5*5,120)
        self.fcl2=nn.Linear(120,84)
        self.fcl3=nn.Linear(84,10)
    #定义一个函数用于计算卷积层跑完后的特征矩阵的元素个数
    def num_flat_features(self,x):
        s=x.size[1:] #x的size是图片数*16*5*5，因此，每个图片的特征矩阵数为16*5*5
        n=1
        for i in s:
            n*=i
        return n
    #定义前向函数
    def forward(self, x):
        #跑卷积层
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        #转换为一维
        x=x.view(-1,self.num_flat_features(x))
        #跑全连接层
        x=nn.ReLU(self.fcl1(x))
        x=nn.ReLU(self.fcl2(x))
        x=nn.self.fcl3(x)
        return x #即为该图片对十种不同classes的概率
    def train_sgd(self,device,trainloader,model_path,epoches):
        super(LeNet5,self).train_sgd(device,trainloader,model_path,epoches)
    def test(self,device,testloader):
        super(LeNet5,self).test(device,testloader)
        






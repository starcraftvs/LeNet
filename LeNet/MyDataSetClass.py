import torch
import torchvision
import GetData
import cv2
import PIL
class MyDataSet1(torch.utils.data.Dataset):
    def __init__(self,path_file,transforms=None,target_transforms=None):
        self.imgs=GetData.GetImgInfo(path_file)
        self.transforms=transforms
        self.target_transforms=target_transforms
    def __getitem__(self,index):
        label=self.imgs[index]['label']
        img=cv2.imread(self.imgs[index]['path'])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=PIL.Image.fromarray(img)
        img=self.transforms(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

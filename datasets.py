import os
import zipfile
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageDraw
import torch.nn as nn
import random
# import datasets

class ASL_C_Dataset(data.Dataset):
    def __init__(self, mode='train', filename='archive_c', img_size=640):
        super(ASL_C_Dataset, self).__init__()
        
        #Initialize variablesd
        self.filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
        # imgpath=self.filepath
        self.filedir=self.filepath
        self.img_size = img_size
        self.class_dict={}
        self.inv_class_dict={}
        self.imagelist = []
        self.labellist = []
        self.zsl_list = []
        
        alphabetlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for iter,clas in enumerate(alphabetlist):
            self.class_dict.update({clas:iter})
            self.inv_class_dict.update({iter:clas})
        temp_num_classes = len(self.class_dict.keys())
        
        # Add classes that are not alphabets
        for clas in os.listdir(self.filedir):
            classpath = os.path.join(self.filedir,clas)
            for imgname in os.listdir(classpath):
                self.imagelist.append(os.path.join(classpath,imgname))
                self.labellist.append(clas)
                
                if(clas not in alphabetlist):
                    # if(clas=='asl-alphabet-test'):
                    #     continue
                    # else:
                    self.class_dict.update({clas:temp_num_classes})
                    temp_num_classes+=1
                    
            
                
         
        self.transform = transforms.Compose([
        transforms.Resize((self.img_size,self.img_size)),
        transforms.CenterCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                            std=[0.224, 0.224, 0.225] )
        ])
        
    def load_img(self,index):
        im = Image.open(self.imagelist[index])
        im = self.transform(im)
        
        return im
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, index):
        return self.load_img(index), torch.as_tensor(self.class_dict[self.labellist[index]], dtype=torch.int64)
    
    def visualise(self,index):
        im = Image.open(self.imagelist[index])
        print('Current label is: ', self.labellist[index])
        im.show()
        return
    
    
# test = ASL_C_Dataset()
# print(len(test))
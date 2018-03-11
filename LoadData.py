#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms

#orl face dataset
class FaceData(Dataset):
    def __init__(self,imglist,class_number,dataparams,mode,transform=None,should_invert=True):
        super(FaceData,self).__init__()
        #self.imglist=imglist
        self.class_number=class_number
        self.mode=mode#switch mode
        self.tupleimglist=self.loadimglist(imglist)
        self.should_invert=should_invert
        self.transform=transform
        self.dataparams=dataparams
          
    def loadimglist(self,imglist):
        #return image name list
        tmpimg_list=list()
        if self.mode=='mlptrain':
            for i in range(self.class_number):#Traverse all categories
                for j in  self.dataparams.train_number:
                    tmpimg_list.append(imglist[i*self.dataparams.all_number+j])
        elif self.mode=='mlpvalidate':
            for i in range(self.class_number):#Traverse all categories
                for j in  self.dataparams.validate_number:
                    tmpimg_list.append(imglist[i*self.dataparams.all_number+j])
        elif self.mode=='mlptest':
            for i in range(self.class_number):#Traverse all categories
                for j in  self.dataparams.test_number:
                    tmpimg_list.append(imglist[i*self.dataparams.all_number+j])
        else:   #variational autoencoder 
            tmpimg_list=imglist
        
        return tmpimg_list
    
    def __getitem__(self,item):

        img=self.tupleimglist[item][0]#PIL image 
        label=self.tupleimglist[item][1]#image label   
        if self.transform is not None:
            img=self.transform(img)#use transform
        return [img,label]#img is a torch.tensor object
        
    def __len__(self):#image list
        return len(self.tupleimglist)

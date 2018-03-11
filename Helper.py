#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import os 
from PIL import Image,ImageDraw
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import visdom

class plotfunc(object):
    def __init__(self):
        super(plotfunc,self).__init__()  
        #self.vis=visdom.Visdom()
        
    def save_Loss(self,iteration,loss):
        #plt.figure
        plt.plot(iteration,loss)
        plt.savefig('iter-loss.png')
    
    def visdom_realtime_show(self,iteration,loss):
        iteration=torch.FloatTensor(iteration)
        loss=torch.FloatTensor(loss)
        self.vis.line(iteration,loss)

    def imshow(self,img,text,should_save=False):
        npimg=img.numpy()
        plt.axis("off")
        if text:
            plt.text(75,8,text,style='italic',fontweight='bold',bbox={'facecolor':'white','alpha':0.8,'pad':10})
            plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()

    def save_TSNE(self,X_tsne,label):
        '''show two dimension TSNE'''
        #plt.figure
        plt.scatter(X_tsne[:,0],X_tsne[:,1],c=label)
        plt.savefig('TSNE.png')

class  TraverseDataset(object):
    def __init__(self,dataset_dir,n,a=1,b=1):
        super(TraverseDataset,self).__init__()
        self.dataset_dir=dataset_dir
        self.imgScrambling= ImageScrambling(a,b,n)
    
    def preprocessYaleDataset(self,ScrambleType):
        """
        Args:
            dataset:dataset directory
            n:number of arnold transforms 
        Returns:
            image list:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            class_numbner :40
        """
        img_list=list()
        tmp_imglist=list()
        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):
            for d in dirnames:
                subpath=os.path.join(self.dataset_dir,d)
                classname=int(d)-1#子文件夹就是类别名称,这个后续还得改
                class_number=len(dirnames)#类别的总数
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):
                    for f in subfilenames:#获得子文件夹下所有的文件
                        img_name=os.path.join(subpath,f)#image name                      
                        name_class=(img_name,classname)#组成一个tuple类型，包含了(文件绝对路径名，类别名)
                        tmp_imglist.append(name_class)                     
        sorted_list=sorted(tmp_imglist)#sort the image name list
        for imagename,classname in sorted_list:
            img=Image.open(imagename)#PIL image
            trans_img=self.imgScrambling.ChooseTransform(ScrambleType,img)
            print(" filename : ",imagename,"-> class : ",classname,"-> Scramble :", ScrambleType)
            img_list.append([trans_img,classname])
        #print(img_list[0])
        return img_list,class_number#
    
    def preprocessORLDataset(self,ScrambleType):
        """
        Args:
            dataset:dataset directory
            n:number of arnold transforms 
        Returns:
            image list:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            class_numbner :40
        """
        img_list=list()
        tmp_imglist=list()
        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):
            for d in dirnames:
                subpath=os.path.join(self.dataset_dir,d)
                classname=int(d)-1#子文件夹就是类别名称,这个后续还得改
                class_number=len(dirnames)#类别的总数
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):
                    for f in subfilenames:#获得子文件夹下所有的文件
                        img_name=os.path.join(subpath,f)#image name                      
                        name_class=(img_name,classname)#组成一个tuple类型，包含了(文件绝对路径名，类别名)
                        tmp_imglist.append(name_class)                     
        sorted_list=sorted(tmp_imglist)#sort the image name list
        for imagename,classname in sorted_list:
            img=Image.open(imagename)#PIL image
            trans_img=self.imgScrambling.ChooseTransform(ScrambleType,img)# use different transforms
            print(" filename : ",imagename,"-> class : ",classname,'-> Scramble :', ScrambleType)
            img_list.append([trans_img,classname])
        #print(img_list[0])
        return img_list,class_number#

    def preprocessCMUPIEDataset(self,ScrambleType):
        """
        Args:
            Pose05_64x64_files:68个人，每个人49张，总的3332
            Pose07_64x64_files:68个人，每个人24张，总的1632，
            Pose09_64x64_files:68个人，每个人24张，总的1632
            Pose27_64x64_files:68个人，每个人49张，总的3332，
            Pose29_64x64_files:68个人，每个人24张，总的1632
        Returns:
            imglist:[class 1 (10 per class ),class 2(10 per class),...,class n( 10 per class )]
            classnumber:68
        """
        img_list=list()

        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):#获得根目录下的文件
            for d in dirnames:#子文件夹，包含不同的姿势的命名
                subpath=os.path.join(self.dataset_dir,d)#构造出路径
                tmp_namelist=list()
                if "Pose05_64x64_files" in subpath:#仅仅找其中的一个姿势
                    for subdirpath,subdirnames,subfilenames in os.walk(subpath):#
                        for f in subfilenames:#遍历子文件下的所有数据
                            img_name=os.path.join(subpath,f)#获得每个子文件夹下的图片名
                            tmp_namelist.append(img_name)
                
        sorted_imglist=sorted(tmp_namelist)
        count=0
        for name in sorted_imglist:
            img=Image.open(name)
            trans_img=self.imgScrambling.ChooseTransform(ScrambleType,img)# use different transforms
            count=count+1
            classname=int(count/49)
            img_list.append((trans_img,classname))
            print(" filename : ",name,"-> class : ",classname,'-> Scramble :', ScrambleType)
        
        classnumber=int(count/49)
        return img_list,classnumber  

    def preprocessPUBFIG83Dataset(self,ScrambleType):
        """
        Args:
            dataset:dataset directory
            n:number of arnold transforms 
        Returns:
            image list:[class 1 (50 per class ),class 2(50 per class),...,class n( 50 per class )]
            class_number :83
        """
        classname=-1    
        img_list=list()
        tmp_imglist=list()
        for dirpath,dirnames,filenames in os.walk(self.dataset_dir):
            for d in dirnames:#获得所有的人名文件夹
                file_number=-1
                subpath=os.path.join(self.dataset_dir,d)
                classname=classname+1#子文件夹就是类别名称,这个后续还得改
                class_number=len(dirnames)#类别的总数
                for subdirpath,subdirnames,subfilenames in os.walk(subpath):
                    for f in subfilenames:#获得子文件夹下所有的文件
                        file_number=file_number+1
                        if file_number<50:#控制每个类中的个数为50张
                            img_name=os.path.join(subpath,f)#image name                      
                            name_class=(img_name,classname)#组成一个tuple类型，包含了(文件绝对路径名，类别名)
                            tmp_imglist.append(name_class)                     
        sorted_list=sorted(tmp_imglist)#sort the image name list

        for imagename,classname in sorted_list:
            img=Image.open(imagename)#PIL image
            w,h=img.size #get the size of image
            img.thumbnail((w//2,h//2))#(250,250)->(125,125)
            trans_img=self.imgScrambling.ChooseTransform(ScrambleType,img)# use different transforms
            print(" filename : ",imagename,"-> class : ",classname,'-> Scramble :', ScrambleType)
            img_list.append([trans_img,classname])

        return img_list,class_number#   

class ImageScrambling(object):
    def __init__(self,a,b,n):
        super(ImageScrambling,self).__init__()
        self.a=a
        self.b=b
        self.n=n

    def ChooseTransform(self,ScramleName,img):
        if ScramleName=='Arnold':
            image=self.ArnoldTransform(self,img)
        if ScramleName=='Fabonacci':
            image=self.FibonacciTransform(self,img)
        if ScramleName=='Magic':
            image=self.MagicTransform(self,img)

    def ArnoldTransform(self,img):
        a=self.a 
        b=self.b 
        n=self.n 
        #cpu model
        width,height=img.size#获得输入图像的大小(宽度，高度)
        if width<height:
            N=width
        else:
            N=height
        #接着对图片进行设置长宽
        img = img.resize((N, N), Image.ANTIALIAS)
        img=img.convert('L')  # 把输入的图片转化为灰度图
        image=Image.new('L',(N,N),(255))#空白的图
        draw=ImageDraw.Draw(image)

        #填充每个像素
        for inc in range(n):
            for y in range(N):
                for x in range(N):
                    xx=(x+b*y)%N
                    yy=(a*x+(a*b+1)*y)%N
                    temp=img.getpixel((x,y))
                    draw.point((xx,yy),fill=img.getpixel((x,y)))       
            img=image
        
        return image
        
    def FibonacciTransform(self,img):
        a=self.a 
        b=self.b 
        n=self.n 
        #cpu model
        width,height=img.size#获得输入图像的大小(宽度，高度)
        if width<height:
            N=width
        else:
            N=height
        #接着对图片进行设置长宽
        img = img.resize((N, N), Image.ANTIALIAS)
        img=img.convert('L')  # 把输入的图片转化为灰度图
        image=Image.new('L',(N,N),(255))#空白的图
        draw=ImageDraw.Draw(image)

        #填充每个像素
        for inc in range(n):
            for y in range(N):
                for x in range(N):
                    xx=(x+b*y)%N
                    yy=(a*x+0*y)%N
                    temp=img.getpixel((x,y))
                    draw.point((xx,yy),fill=img.getpixel((x,y)))       
            img=image
        
        return image

    def MagicTransform(self,img):
        a=self.a 
        b=self.b 
        n=self.n 
        #cpu model
        width,height=img.size#获得输入图像的大小(宽度，高度)
        if width<height:
            N=width
        else:
            N=height
        #接着对图片进行设置长宽
        img = img.resize((N, N), Image.ANTIALIAS)
        img=img.convert('L')  # 把输入的图片转化为灰度图
        image=Image.new('L',(N,N),(255))#空白的图
        draw=ImageDraw.Draw(image)

        #填充每个像素
        for inc in range(n):
            for y in range(N):
                for x in range(N):
                    xx=(x+b*y)%N
                    yy=(a*x+(a*b+1)*y)%N
                    temp=img.getpixel((x,y))
                    draw.point((xx,yy),fill=img.getpixel((x,y)))       
            img=image
        
        return image
#! /usr/bin/env python
#-*- coding:UTF-8 -*-
from __future__ import print_function
import os
import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import time
import shutil
import torch
from sklearn.manifold import TSNE
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#import self-built function
from LoadData import  FaceData
from Net import MLP,VAE  #import network
from Train import ModelTrain,AverageMeter#import trainer
from Helper import plotfunc,TraverseDataset #import some helper function
from args import get_parser#import args
#==================================================
mp=get_parser()
opts=mp.parse_args()
#==================================================

def test(test_loader,MLPModel,AEModel):
    losses=AverageMeter()
    top1=AverageMeter()
    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    MLPModel.load_state_dict(checkpoint['MLPstate_dict'])
    AEModel.load_state_dict(checkpoint['AEstate_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))
    
    # switch to evaluate mode
    AEModel.eval()
    MLPModel.eval()
    counter=0
    before_output=[]
    label_1=[]
    after_output=[]
    for i,testdata in enumerate(test_loader,0):
        counter=counter+1
        input_var=list()
        [img,label]=testdata
        opts.cuda=torch.cuda.is_available()
        img_var,label_var=Variable(img,volatile=True),Variable(label,volatile=True)
        if opts.cuda:
            img_var,label_var=img_var.cuda(),label_var.cuda()
        
        inputimg=img_var.view(img_var.size()[0],-1)       
        encoded,decoded=AEModel(inputimg)
        output=MLPModel(encoded)  
       
        output_1=output.cpu().data.numpy()
        if counter==1:
            before_output=img_var.numpy()
            after_output=output_1
            label_1=label.numpy()
        elif counter<=2:
            after_output=np.concatenate((after_output,output_1),axis=0)
            label_1=np.concatenate((label_1,label.numpy()),axis=0)
        #use euclidean distance of two image
                    
        loss_=F.cross_entropy(output,label_var)
        losses.update(loss_.data[0], img.size(0))

        prec1=accuracy(output.cpu().data.numpy(),label.numpy())
        top1.update(prec1, img.size(0))
    print('test Prec@1 {0:.2f}\t validation Loss {1:.2f}'.format(float(top1.avg), float(losses.avg)))
    Y_tsne=TSNE(n_components=2,learning_rate=1000.0).fit_transform(before_output)
    X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(after_output)
    ori_label=np.transpose(label_1)
    return top1.avg,X_tsne,Y_tsne, ori_label
   
def plot_TSNE(data_loader,DeidenNetModel):
    '''data_loader:all data'''
    losses=AverageMeter()
    top1=AverageMeter()
    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    DeidenNetModel.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # switch to evaluate mode
    DeidenNetModel.eval()
    counter=0
    before_output=[]
    label_1=[]
    after_output=[]
    for i,data in enumerate(data_loader,0):
        counter=counter+1
        input_var=list()
        [img,label]=data
        opts.cuda=torch.cuda.is_available()
        img_var,label_var=Variable(img,volatile=True),Variable(label,volatile=True)
        if opts.cuda:
            img_var,label_var=img_var.cuda(),label_var.cuda()

        output=DeidenNetModel(img_var)  
       
        output_1=img.view(img.size()[0],-1).numpy()
        output_2=output.cpu().data.numpy()
        if counter==1:
            before_output=output_1
            after_output=output_2
            label_1=label.numpy()
        elif counter<=2:
            before_output=np.concatenate((before_output,output_1),axis=0)
            after_output=np.concatenate((after_output,output_2),axis=0)
            label_1=np.concatenate((label_1,label.numpy()),axis=0)
        #use euclidean distance of two image
                    
        loss_=F.cross_entropy(output,label_var)
        losses.update(loss_.data[0], img.size(0))

        prec1=accuracy(output.cpu().data.numpy(),label.numpy())
        top1.update(prec1, img.size(0))
    print('test Prec@1 {0:.2f}\t validation Loss {1:.2f}'.format(float(top1.avg), float(losses.avg)))
    #print(before_output.shape)
    #print(after_output.shape)
    Y_tsvimne=TSNE(n_components=3,learning_rate=100.0).fit_transform(before_output)
    X_tsne = TSNE(n_components=3,learning_rate=100).fit_transform(after_output)
    return top1.avg,X_tsne,Y_tsne, label_1

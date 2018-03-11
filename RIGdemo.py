#! /usr/bin/env python
#-*- coding:UTF-8 -*-
from __future__ import print_function 
import os 
import sys
import argparse  
import torchvision 
import torchvision.datasets as dset 
from torchvision.transforms import transforms
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
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
from LoadData import  FaceData#import face data
from Net import MLP,VAE  #import network
from RIBTrain import ModelTrain,AverageMeter#import trainer
from Helper import plotfunc,TraverseDataset,ImageScrambling #import some helper function
from args import get_parser#import args
from test import test #import test 
from staticParams import params

#=======================================================================================================
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def main(argv):
    mp=get_parser(argv[0])  #argv[0]:dataset. argv[1]:scramble type
    opts=mp.parse_args()
    #pltfunc=plotfunc()#plot image 

###################################### 0.Image Scrambling and preprocess  images ###########################
    if opts.dataset_name=='Yale':
        dataparams=params('Yale')
        Scrambleimglist,class_number=TraverseDataset(dataparams.dataset_dir,opts.n).preprocessYaleDataset(opts.scramble_method) # Yale
    elif opts.dataset_name=='ORL':
        dataparams=params('ORL')
        Scrambleimglist,class_number=TraverseDataset(dataparams.dataset_dir,opts.n).preprocessORLDataset(opts.scramble_method)   #ORL
    elif opts.dataset_name=='CMUPIE':  
        dataparams=params('CMUPIE')
        Scrambleimglist,class_number=TraverseDataset(dataparams.dataset_dir,opts.n).preprocessCMUPIEDataset(opts.scramble_method)   #CMUPIE
    else:      
        dataparams=params('PUBFIG')     
        Scrambleimglist,class_number=TraverseDataset(dataparams.dataset_dir,opts.n).preprocessPUBFIG83Dataset(opts.scramble_method) # PUBFIG
   
    opts.num_classes=class_number
    print('Data Image Scrambling transforms done. Class number is :{0}',class_number)

###################################### 1.Data Loader#################################################

    MLP_traindata=FaceData(Scrambleimglist,class_number,dataparams,mode='mlptrain',
        transform=transforms.Compose([              
            transforms.ToTensor(),
            transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
        ]),should_invert=False)
    print('MLP train data loaded done')

    MLP_valdata=FaceData(Scrambleimglist,class_number,opts,mode='mlpvalidate',
        transform=transforms.Compose([       
            transforms.ToTensor(),
            transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
        ]),should_invert=False)
    print('MLP validate data loaded done')

    MLP_testdata=FaceData(Scrambleimglist,class_number,dataparams,mode='mlptest',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
        ]),should_invert=False)
    print('MLP test data loaded done')
    
    MLP_train_loader=DataLoader(MLP_traindata,shuffle=True,num_workers=opts.num_workers,batch_size=opts.MLP_training_batch_size)
    MLP_val_loader=DataLoader(MLP_valdata,shuffle=True,num_workers=opts.num_workers,batch_size=opts.MLP_validate_batch_size)
    MLP_test_loader=DataLoader(MLP_testdata,shuffle=True,num_workers=opts.num_workers,batch_size=opts.MLP_test_batch_size)
###################################### 2. Model ############################################################
    opts.cuda=torch.cuda.is_available()
    inputsize=arnoldimglist[0][0].size[0]#image size
    MLPModel=MLP(inputsize,dataparams.num_classes)#Yale is 15,ORL face class number is 40,CMUPIE class number is 68,PUBFIG is 83

    torch.manual_seed(opts.seed)
    if opts.cuda:
        MLPModel= MLPModel.cuda()
        torch.cuda.manual_seed(opts.seed)

    MLPModel.apply(weights_init)#weight initial  

    print('model done.')
####################################### 3.Optimizer ################################################################
    MLPOptimizer=optim.Adam(MLPModel.parameters(),lr=opts.lr)# optimzer4mlp  
    best_val=0   
    print('optimizer done.')
        
#################################################### 4.training#####################################################
    modelTrain=ModelTrain(opts,MLPModel, MLP_train_loader,MLP_val_loader,MLP_test_loader,MLPOptimizer)
     
    VAEcount=modelTrain.VAEtrain()
    mlpcount=modelTrain.mlptrain()
    #pltfunc.show_plot(mlpcount.counter,mlpcount.loss_history)
    #_,after_tsne,before_tsne, ori_label=calculate_TSNE( MLP_test_loader,MLPModel)
    ##_,X_tsne,Y_tsne, ori_label=test(opts,MLP_test_loader,MLPModel,VAEModel)
    #pltfunc.show_TSNE(before_tsne,ori_label)
    #pltfunc.show_TSNE(after_tsne,ori_label)
 
def calculate_TSNE(data_loader,MLPModel):
    '''data_loader:all data'''
    losses=AverageMeter()
    top1=AverageMeter()
    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    self.MLPModel.load_state_dict(checkpoint['MLPstate_dict'])
    self.VAEModel.load_state_dict(checkpoint['VAEstate_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # switch to evaluate mode
    MLPModel.eval()
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

        ip1,output=MLPModel(img_var)  
       
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
    print(before_output.shape)
    print(after_output.shape)
    Y_tsvimne=TSNE(n_components=3,learning_rate=100.0).fit_transform(before_output)
    X_tsne = TSNE(n_components=3,learning_rate=100).fit_transform(after_output)
    return top1.avg,X_tsne,Y_tsne, label_1   

if __name__=="__main__":
    main(sys.argv)
    
    
   

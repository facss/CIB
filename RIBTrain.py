#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import time
import torch
import shutil
import numpy as np 
import torch.nn as nn
import torchvision 
import torch.utils.data 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from args import get_parser
from Loss import VAE_loss_fun,MLP_loss_fun

#==========================================================
class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def Update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

class FunctionsAndStatistics(object):
    def __init__(self):
        super(FunctionsAndStatistics,self).__init__()

    def save_checkpoint(state,is_best,filename='checkpoint.path.tar'):
        """save file to checkpoint"""
        torch.save(state,filename)
        if is_best:
            shutil.copy(filename,opts.model_path)

    def accuracy(output, label):
        #top 1 precision
        """Computes the precision@k for the specified values of k"""
        count=0
        reses=np.argmax(output,axis=1)
        lengths=output.shape[0]
        for i in range(lengths):
            if reses[i] == label[i]:
                count=count+1
        
        result=count/lengths
        return result

    def ROC(self):
        pass

class Counter(object):
    def __init__(self):
        super(Counter,self).__init__()
        self.iteration_number=0
        self.counter=list()
        self.loss_history=list()
    def Add(self,loss):
        self.iteration_number+=10
        self.counter.append(self.iteration_number)
        self.loss_history.append(loss)

class ModelTrain(object):
    def __init__(self,opts,VAEModel,MLPModel,VAETrainLoader,MLPTrainLoader,MLPValLoader,MLPTestLoader,VAEOptim,MLPOptim):
        super(ModelTrain,self).__init__()
        self.opts=opts
        self.MLPModel=MLPModel
        self.MLPTrainLoader=MLPTrainLoader
        self.MLPValLoader=MLPValLoader
        self.MLPTestLoader=MLPTestLoader
        self.MLPOptim=MLPOptim
        self.FAS=FunctionsAndStatistics()

    def mlptrain(self,best_val):
        #train for one epoch
        mlpcount=Counter() 
        #self.FAS.CounterInitialize()
        for epoch in range(self.opts.MLP_Start_epoch,self.opts.MLP_train_number_epochs):       
            mlptrain_epoch(epoch,mlpcount)

            if (epoch+1)%opts.valfre ==0 and epoch !=0:#save the model
                val_res=mlpvalidate(MLP_val_loader,MLPModel)
                #save the best model
                is_best=val_res>best_val
                best_val=max(val_res,best_val)
                save_checkpoint({
                    'epoch':epoch+1,
                    'MLPstate_dict':MLPModel.state_dict(),
                    'AEstate_dict':AEModel.state_dict(),
                    'best_val':best_val,
                    'MLPOptimizer':MLPOptimizer,
                    'AEOptimer':AEOptimizer,
                    'curr_val':val_res,
                },is_best)
                print('** Validation : %f (best) '%(best_val))

        return mlpcount
        
    def mlptrain_epoch(self,epoch,mlpcount):
        #switch to train model
        self.MLPModel.train()
        
        for i,traindata in enumerate(self.MLPTrainLoader,0):
            #measure data loading time       
            [img,label]=traindata#load original training data
            opts.cuda=torch.cuda.is_available()
            if opts.cuda:
                img,label=Variable(img).cuda(),Variable(label).cuda()
            
            #load data into models     
            inputimg=img.view(img.size()[0],-1)  
            output=self.MLPModel(inputimg)
            loss =MLP_loss_fun(output,label)

            self.MLPOptim.zero_grad() 
            
            loss.backward()

            self.MLPOptim.step()
            
            if i%1000 ==0:
                print('Epochs number {}\n Current loss:{}\n'.format(epoch,loss.data[0]))                
                mlpcount.Add(loss.data[0])

    def mlpvalidate(self):
        top1=AverageMeter()
        losses=AverageMeter()
        
        #switch to evaluate mode
        self.MLPModel.eval()

        for i,valdata in enumerate(self.MLPValLoader,0):
            input_var=list()
            [img,label]=valdata
            opts.cuda=torch.cuda.is_available()
            img_var,label_var=Variable(img,volatile=True),Variable(label,volatile=True)
            if opts.cuda:
                img_var,label_var=img_var.cuda(),label_var.cuda()
            
            img_var=img_var.view(img_var.size()[0],-1)
            output=self.MLPModel(img_var)  
            
            loss_=MLP_loss_fun(output,label_var)          
            losses.update(loss_.data[0], img.size(0))

            prec1=self.FAS.accuracy(output.cpu().data.numpy(),label.numpy())
            top1.update(prec1, img.size(0))
        print('Prec@1 {0:.2f}\t validation Loss {1:.2f}'.format(float(top1.avg), float(losses.avg)))
        return top1.avg
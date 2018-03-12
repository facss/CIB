#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

#naive VAE model
class VAE (nn.Module):
    def __init__(self,inputsize):
        '''
         all parameters need to be initialize in the beginning,or the model will be empty in the training parameters
        '''
        super(VAE,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(inputsize*inputsize,inputsize*(inputsize/4)),
            nn.LeakyReLU(.2),
            nn.Linear(inputsize*int(inputsize)/4,2*inputsize*int(inputsize/16) ),#2 for mean and variance.        
        )
        self.decoder=nn.Sequential(
            nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize)/4),
            nn.ReLU(),
            nn.Linear(inputsize*int(inputsize/4),inputsize),
            nn.Sigmoid()
        )    

    def reparameterize(self,mu,logvar):
        '''
        z=mean+eps*sigma where eps is sampled from N(0,1)
        '''
        std=logvar.mul(.5).exp_()
        eps=Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std).add_(mu)

    def forward(self,x):
        h=self.encoder(x)
        mu,logvar=torch.chunk(h,2,dim=1)
        z=self.reparameterize(mu,logvar)
        decoded=self.decoder(z)
        return mu,logvar,decoded

class MLP(nn.Module):
    def __init__(self,inputsize,bits_len):
        super(MLP,self).__init__()
        self.fc1=nn.Sequential(
            nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/2)),
            nn.ReLU(),
            nn.BatchNorm1d(inputsize*int(inputsize/2)),

            nn.Linear(inputsize*int(inputsize/2),1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512,bits_len),
            nn.ReLU(),
            nn.BatchNorm1d(bits_len),
        ) 

    def forward(self,input):           
        #full connect network
        #input=input.view(input.size()[0],-1)
        output=self.fc1(input)
        output=F.log_softmax(output)
        
        return output



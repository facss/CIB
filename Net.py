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
        super(VAE,self).__init__()
        self.fc11=nn.Linear()
        self.fc12=nn.Linear()
        self.fc3=nn.Linear()
        self.fc4=nn.Linear()
        
    def encoder(self,x):
        self.fc1=nn.Sequential(
            nn.Linear(inputsize*inputsize,inputsize*int(inputsize/4)),
            nn.ReLU(),
            nn.Linear(inputsize*int(inputsize/4),inputsize*int(inputsize/16))
        )
        self.fc21=nn.Linear(inputsize*int(inputsize/16),20)
        self.fc22=nn.Linear(inputsize*int(inputsize/16),20)
        h1=self.fc1(x)
       
        return self.fc21(h1),self.fc22(h1) 

    def decoder(self,z):
        self.fc3=nn.Sequential(
            nn.Linear(20,inputsize*int(inputsize/16)),
            nn.Linear(inputsize*int(inputsize/16),inputsize*int(inputsize/4)),
            nn.ReLU(),
            nn.Linear(inputsize*int(inputsize/4),inputsize*inputsize),
        )
        self.sigmoid=nn.Sigmoid()
        h3=self.sigmoid(self.fc3(z))
        return h3 

    def reparameterize(self,mu,logvar):
        std=logvar.mul(.5).exp_()
        eps=Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self,x):
        mu,logvar=self.encoder(x)
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
        input=input.view(input.size()[0],-1)
        output=self.fc1(input)
        output=F.log_softmax(output)
        
        return output



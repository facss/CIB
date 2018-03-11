#! /usr/bin/env python
#-*- coding:UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def VAE_loss_fun(recon_x,x,mu,logvar):
    reconstruction_function=nn.BCELoss()
    reconstruction_function.size_average=False
    BCE=reconstruction_function(recon_x,x )

    KLD_element=mu.pow(2).add_(logvar.exp()).mul_(-1).add_(logvar)
    KLD=torch.sum(KLD_element).MUL_(-0.5)

    return BCE+KLD 

def MLP_loss_fun(input,label):
    loss_val=F.cross_entropy(input,label)
    return loss_val
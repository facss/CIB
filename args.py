#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import argparse


def get_parser():
        
    #parser.add_argument('-shortname','--fullname',type=?,default=?)
    ################################# Data Loader #############################################
    
    parser=argparse.ArgumentParser(description='pytorch Scrambled Domain face recognition via CIB.')
    parser.add_argument('--dataset_name','-dn',type=str,default='ORL')
    parser.add_argument('--scramble_method','-sm',type=str,default='Arnold')

    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--VAE_training_batch_size',type=int,default=64)
    parser.add_argument('--MLP_training_batch_size',type=int,default=64,help='batch size of reidentifynetwork training set')
    parser.add_argument('--MLP_test_batch_size',type=int,default=16,help='batch size of reidentifynetwork testing set')
    parser.add_argument('--MLP_validate_batch_size',type=int ,default=16,help='batch size of reidentifynetwork validate set')

    ################################# Model #############################################
    parser.add_argument('--cuda',type=bool,default=False,help='if the GPU is available')
    parser.add_argument('--seed',type=int,default=1,help='manual seed ') 

    ################################# Train & Validate #############################################
    parser.add_argument('--valfre',type=int,default=10,help='frequency of validate')
    parser.add_argument('--model_path',type=str,default='./model_best.path.tar')
    parser.add_argument('--momentum',type=float,default=0.9,help='SGD momentum')
    parser.add_argument('--weight_decay',type=float,default=0.0005,help='SGD weight decay')

    ################################ Arnold transform #############################################
    parser.add_argument('--n',type=int,default=3,help='arnold transform n')
    parser.add_argument('--a',type=int,default=1,help='arnold transform a')
    parser.add_argument('--b',type=int,default=1,help='arnold transform b')

    ################################# Optimizer #############################################
    parser.add_argument('--MLP_Start_epoch',type=int,default=0,help='the number of reidentify checkpoint start epoch')
    parser.add_argument('--resume',type=str,default='./')
    parser.add_argument('--MLP_train_number_epochs',type=int,default=300)
    parser.add_argument('--VAE_train_number_epochs',type=int,default=500)
    parser.add_argument('--lr',type=int,default=0.00001)
    parser.add_argument('--bit_length',type=int,default=64)

    return parser
    


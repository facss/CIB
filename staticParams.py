#! /usr/bin/env python
#-*- coding:UTF-8 -*-

class YaleParams:
    dataset_dir='/media/Dataset/Yale/'
    train_number=[0,1,4,5,6,7,8,9,10]
    validate_number=[2]
    test_number=[3]
    all_number=11
    num_classes=15

class ORLParams:
    dataset_dir='/media/Dataset/ORL/'
    train_number=[0,1,2,3,4,5,6]
    validate_number=[7,8]
    test_number=[9]
    all_number=10
    num_classes=40

class CMUPIEParams:
    dataset_dir='/media/Dataset/CMUPIE/'
    train_number=range(0,30)
    validate_number=range(30,40)
    test_number=range(40,49)
    all_number=49
    num_classes=68

class PUBFIGParams:
    dataset_dir='/media/Dataset/PUBFIG/'
    train_number=range(0,30)
    validate_number=range(30,40)
    test_number=range(40,50)
    all_number=50
    num_classes=83


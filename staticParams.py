#! /usr/bin/env python
#-*- coding:UTF-8 -*-

class params(object):
    dataset_dir='' #
    train_number=[] #
    validate_number=[] #
    test_number=[] #
    all_number= 0 #
    num_classes= 0 #

    def __init__(self,dataset_name):
        self.dataset_dir=dataset_dir

        if self.dataset_name=='Yale':
            dataset_dir='/media/Dataset/Yale/'
            train_number=[0,1,4,5,6,7,8,9,10]
            validate_number=[2]
            test_number=[3]
            all_number=11
            num_classes=15
        elif self.dataset_name=='ORL':
            dataset_dir='/media/Dataset/ORLface/'
            train_number=6
            validate_number=8
            test_number=10
            all_number=10
            num_classes=40
        elif  self.dataset_name=='CMUPIE':
            dataset_dir='/media/Dataset/CMUPIE/'
            train_number=30
            validate_number=40
            test_number=49
            all_number=49
            num_classes=68
        else:
            dataset_dir='/media/Dataset/PUBFIG/'
            train_number=30
            validate_number=40
            test_number=50
            all_number=50
            num_classes=83

        

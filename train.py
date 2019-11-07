
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sb
import json
import methods1


def get_input_args():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = 'flowers', 
                    help = 'path to the folder flowers')
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'choose model between vgg16 or vgg13')
    parser.add_argument('--inputs', type = int, default = 2098, 
                    help = 'hidden inputs for classifier')
    parser.add_argument('--learningrate', type = float, default = 0.001, 
                    help = 'choose learning rate')
    parser.add_argument('--device', type = str, default = 'gpu', 
                    help = 'choose device between gpu or cpu')
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'choose number of epochs')
    parser.add_argument('--checkpoint', type = str,  
                    help = 'save model')
    return parser.parse_args()

def main():
    in_args=get_input_args()
    #print("command line inputs :\n dir:", in_args.dir, "\n arch:", in_args.arch , '\n inputs :', in_args.inputs, ' \n lr', in_args.learningrate,'\n device',in_args.device,'\n epochs', in_args.epochs)
    data_dir=in_args.dir
    arch=in_args.arch
    hidden_inputs=in_args.inputs
    lr=in_args.learningrate
    device=in_args.device
    epochs=in_args.epochs
    checkpoint=in_args.checkpoint
    trainloader,testloader,validloader,train_datasets,test_datasets,valid_datasets= methods1.load_data(data_dir)
    
    classifier,model,criterion,optimizer=methods1.archticture(arch,hidden_inputs,lr)
    
    #print(model)
    
    methods1.train(classifier,model,criterion,optimizer,device,epochs,trainloader,validloader,train_datasets,checkpoint,arch,hidden_inputs,lr)
    methods1.test( model, testloader, criterion, device)
    
    
if __name__ == "__main__":
    main()
    
   
    
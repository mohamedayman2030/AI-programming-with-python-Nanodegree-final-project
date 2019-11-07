
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
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', 
                    help = 'path to the model')
    parser.add_argument('--imagepath', type = str, default = 'flowers/test/10/image_07090.jpg', 
                    help = 'path to image to know its type')
    parser.add_argument('--topk', type=int, default=5,
                        help='the number of most probabilities')
    parser.add_argument('--flowersnamesfile', type = str, default = 'cat_to_name.json', 
                    help = 'flowers names file')
    parser.add_argument('--device', type = str, default = 'gpu', 
                    help = 'choose device between gpu or cpu')
    return parser.parse_args()

def main():
    in_args=get_input_args()
    
    checkpoint=in_args.checkpoint
    image_path=in_args.imagepath
    topk=in_args.topk
    flowers_name=in_args.flowersnamesfile
    device=in_args.device
    
    with open(flowers_name, 'r') as f:
        cat_to_name = json.load(f)
    
    model=methods1.load_checkpoint(checkpoint)
    #print(model)
    probs=methods1.predict(image_path,model,topk,device,flowers_name)
    probabilities = np.array(probs[0][0])
    names=[cat_to_name[str(i+1)] for i in np.array(probs[1][0])]
    print( '{} are the probabilities for top {} predicted flowers'.format(probabilities*100,topk))
    print( '{} are the names for top {}  predicted flowers'.format(names,topk))
    
if __name__ == "__main__":
    main()    
    
    

    
    
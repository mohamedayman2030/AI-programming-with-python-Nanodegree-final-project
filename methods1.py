
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




def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets =  datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=32,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets,batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_datasets,batch_size=32)
    return trainloader,testloader,validloader,train_datasets,test_datasets,valid_datasets
def archticture(model,hidden_units,learning_rate):
    if model=='vgg16':
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        classifier=nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, hidden_units)),
                           ('relu',nn.ReLU()),
                           ('dropout',nn.Dropout(0.5)),
                           ('fc2',nn.Linear(hidden_units,102)),
                           ('output',nn.LogSoftmax(dim=1))]))
        model.classifier= classifier
    elif model=='vgg13':
        model=models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        classifier=nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, hidden_units)),
                           ('relu',nn.ReLU()),
                           ('dropout',nn.Dropout(0.5)),
                           ('fc2',nn.Linear(hidden_units,102)),
                           ('output',nn.LogSoftmax(dim=1))]))
        model.classifier= classifier
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return classifier,model,criterion,optimizer
def validation(model,validloader,criterion,device):
    model.eval()
    valid_loss=0
    accuracy=0
    if device == 'gpu' and torch.cuda.is_available():
        model.to('cuda')
    for images,labels in validloader:
        if device == 'gpu' and torch.cuda.is_available():
                images,labels = images.to('cuda'), labels.to('cuda')
        else:
                images,labels = images.to('cpu'), labels.to('cpu')
        output=model.forward(images)
        valid_loss+=criterion(output,labels).item()
        ps=torch.exp(output)
        equality=(labels.data==ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
            
    return valid_loss,accuracy
def train(classifier,model,criterion,optimizer,device,epochno,trainloader,validloader,train_datasets,cp,arch,hiddenunits,lr):
    epochs=epochno
    steps=0
    print_every=10
    if device == 'gpu' and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    for e in range(epochs):
        model.train()
        for ii, (inputs,labels) in enumerate(trainloader):
            if device == 'gpu' and torch.cuda.is_available():
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs,labels = inputs.to('cpu'), labels.to('cpu')
            
            steps+=1
            optimizer.zero_grad()
            
        
            outputs=model.forward(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if steps%print_every==0:
                model.eval()
                with torch.no_grad():
                    validloss, accuracy=validation(model,validloader,criterion,device)
                print("Epoch:{}/{}..".format(e+1,epochs),
                 "valid loss:{:.3f}..".format(validloss/len(validloader)),
                 "accuracy:{:.3f}..".format(accuracy/len(validloader))
                 ) 
                model.train()
    #save the model
    model.class_to_idx=train_datasets.class_to_idx
    checkpoint = { 
              'class_to_index':model.class_to_idx,
              'epochs':epochno,
              'arch': arch,
              'hidden_layers':hiddenunits,
              'learning_rate':lr,
              'optimizer_state':optimizer.state_dict(),
              'state_dict': model.state_dict()
             }
    torch.save(checkpoint,cp)

def test(model,testloader,criterion,device):
    model.eval()
    valid_loss=0
    accuracy=0
   
    for images,labels in testloader:
        if device == 'gpu' and torch.cuda.is_available():
                images,labels = images.to('cuda'), labels.to('cuda')
        else:
                images,labels = images.to('cpu'), labels.to('cpu')
        output=model.forward(images)
        test_loss+=criterion(output,labels).item()
        ps=torch.exp(output)
        equality=(labels.data==ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print((accuracy/len(testloader))*100)    
    
def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    _,model, _, _=archticture(checkpoint['arch'],
                      checkpoint['hidden_layers'],
                      checkpoint['learning_rate']
                     )
    
    model.class_to_idx =checkpoint['class_to_index']
    
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    im = Image.open(image)
    width, height = im.size
    
    if width > height:
        ratio=width/height
        
        im.thumbnail((ratio*256,256))
    elif height > width:
        im.thumbnail((256,height/width*256))
        
    
    new_width, new_height = im.size
    
    
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width +224)/2
    bottom = (new_height+ 224)/2
    im=im.crop((left, top, right, bottom))
    
    np_image=np.array(im)
    np_image = np_image / 255


    means=np.array([0.485, 0.456, 0.406])
    std= np.array([0.229, 0.224, 0.225]) 
    
    np_image=(np_image-means)/std
    np_image = np_image.transpose((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk , device,namesfile): 
    with open(namesfile, 'r') as f:
        cat_to_name = json.load(f)
    if device == 'gpu' and torch.cuda.is_available():
        model.to('cuda')
    
    img = process_image(image_path)
    img=torch.from_numpy(img)

    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        if device == 'gpu' and torch.cuda.is_available():
            output = model.forward(img.cuda())
        else:
            output = model.forward(img)
       
        
    probs = torch.exp(output)
    probs=probs.topk(topk)
    return probs

def check_sanity(image_path,model,device,topk):
    
    with open(namesfile, 'r') as f:
        cat_to_name = json.load(f)
    
    
    
    
    im = process_image(image_path)
    probs = predict(image_path, model,topk,device)
    
    
    

    image = imshow(im)
    
     
    probabilities = np.array(probs[0][0])
    
    flowersname=[cat_to_name[str(i+1)] for i in np.array(probs[1][0])]
    
    print(flowersname)
    fig=plt.figure(figsize=(10,10))
    axis1=fig.add_subplot(2,1,2)
    
    axis1.barh(flowersname,probabilities)
    plt.show()
            

        

    
    


        
            
        
        
    

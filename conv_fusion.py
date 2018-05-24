from __future__ import print_function, division
import pickle
import torch
import torch.nn as nn
import glob
from PIL import Image
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torchvision
import dataloader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import re
import time
import os
import copy


# for computing confusion matrix
def compute_confusion_matrix(y_actu, y_pred, class_names):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_actu, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    #uncomment below lines if you want to plot non-normalized matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(20,20))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()

    
#for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
#function for getting weights of first layer of pre-trained model and first layer of model defined on scratch
def get_weights(trained_model, untrained_model):
    i = 0
    for param in untrained_model.parameters():
        if(i == 0):
            untrained_weights = param.data
        i += 1
        
    i = 0
    for param in trained_model.parameters():
        if(i == 0):
            trained_weights = param.data
        i += 1 
        
    return [trained_weights, untrained_weights]
    

#function for appending the trained model weights to the first layer of model defined on scratch
def concat_trained_weights(trained_weights, untrained_weights):
    #averaging of trained filter weights
    temp           = torch.sum(trained_weights[:,:], dim= 1)[0] / 3

    filter_dim     = untrained_weights.size(1)
    filter_weights = temp.view(1,3,3)
    for i in range(filter_dim-1):
        filter_weights = torch.cat((filter_weights, temp.view(1,3,3)), dim=0)

    for i in range(untrained_weights.size(0)):
        untrained_weights[i,:,:,:] = filter_weights
    
    return untrained_weights

#function for showing the inputs at the time of loading of data
def show_images(inputs,label):
    for i in range(inputs.shape[0]):
        a = inputs[i:i+1,:]
        a = a.view([int(inputs.size(1)/3),3,224,224])

        out = torchvision.utils.make_grid(a)
        out = out.numpy().transpose((1, 2, 0))
        out = [0.229, 0.224, 0.225] * out + [0.485, 0.456, 0.406]
        out = np.clip(out, 0, 1)
        plt.imshow(out)
        plt.pause(0.001) 
        print("Patches of RGB frames for batch %d for class %d" % (i,label[i]+1))
   
'''
Function for training of model
This function can be modified accordingly to include validation set
'''
    
def train_model(model, criterion, optimizer,scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            j = 0
            scheduler.step()
            model.train()

            epoch_loss = 0.0
            correct_samples = 0
            
            # Iterate over data.
            for data in fullloader[phase]:
#                 print("loading the data")
                spat_data, temp_data, labels = data

                # wrap them in Variable
                if use_gpu:
                    spat_data = Variable(spat_data.cuda())
                    temp_data = Variable(temp_data.cuda())
                    labels    = Variable(labels.cuda())
                else:
                    spat_data = Variable(spat_data)
                    temp_data = Variable(temp_data)
                    labels    =  Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(spat_data, temp_data)
                #_,preds = torch.max(outputs.data, 1)
                                         

                # backward and optimize for training mode
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # finding epoch losses and accuracies for fusion
                epoch_loss += loss.data[0] * spat_data.size(0)
                
                #correct_samples += torch.sum(preds == labels.data)
            #print("Correct Samples = ", correct_samples)
                                         
            phase_size = len(fullloader[phase])*fullloader[phase].batch_size
#             print("phase size = ", phase_size)
            #epoch_acc = correct_samples / phase_size 
            epoch_loss = epoch_loss / phase_size

            #print("=================== Training mode ====================")
            print('Loss: {:.4f} '.format(
                epoch_loss))
         
       del(spat_data)
       del(temp_data)
       del(labels)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model




#loading the train and test data
'''
change here for loading data for spatial loader, temporal loader
'''
data_loader = dataloader.spatio_temporal_dataloader(BATCH_SIZE=16, num_workers=1, in_channel = 50,
                                spatial_path='../link_to_conv_fusion/data/jpegs_256/',         
                                temp_path='../link_to_tvl/tvl1_flow/',
                                ucf_list='./UCF_list/',
                                ucf_split='02',
                                train_transform = transforms.Compose([
                                                   transforms.RandomCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                               ]),
                                val_transform = transforms.Compose([
                                                     transforms.Resize([224,224]),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])
                                                 ]))
train_loader, test_loader, test_video = data_loader.run()
'''
appending train-loader and test loader for training the model
'''
fullloader = {}
fullloader['train'] = train_loader
fullloader['val'] = test_loader

'''
loading classes name for computing confusion matrix
'''
temp = []
with open("./UCF_list/classInd.txt", "r") as f:
    for line in f:
        abc = line.split(" ")[1]
        temp.append(abc.rstrip("\n"))
f.close()


'''
Defining a model model which will do convolution fusion of both stream and 3D pooling 
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spat_feature = spat_model #feature_size =   Nx512x7x7
        self.temp_feature = temp_model #feature_size = Nx512x7x7
        self.layer1       = nn.Sequential(nn.Conv3d(1024, 512, 1, stride=1, padding=1, dilation=1,bias=True),
                                   nn.ReLU(),nn.MaxPool3d(kernel_size=2,stride=2))
        self.fc           = nn.Sequential(nn.Linear(8192,2048), nn.ReLU(), nn.Dropout(p=0.85),
                                        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.85),
                                        nn.Linear(512, 101))
        
    def forward(self,spat_data,temp_data):
        x1       = self.spat_feature(spat_data)
        x2       = self.temp_feature(temp_data)
        
        y        = torch.cat((x1,x2), dim= 1)
        for i in range(x1.size(1)):
            y[:,(2*i),:,:]   = x1[:,i,:,:]
            y[:,(2*i+1),:,:] = x2[:,i,:,:]
            
        y        = y.view(y.size(0), 1024, 1, 7, 7)
        cnn_out  = self.layer1(y)
        cnn_out  = cnn_out.view(cnn_out.size(0),-1)
        out      = self.fc(cnn_out)
        return out

'''
using pre-trained weights on ImageNet for both streams
'''
spat_model = models.vgg16(pretrained=True)
temp_model = models.vgg16(pretrained=True)

#changing input filter dimension of spatial model 
feat_     = list(spat_model.features.children())
class_    = list(spat_model.classifier.children())
feat_[0]  = nn.Conv2d(30, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
class_[6] = nn.Linear(in_features=4096, out_features=101, bias=True)

#temporary model 
abc       = nn.Sequential(*feat_)


'''
first average the weight value across the RGB channels 
and replicate this average by the channel number of model
'''
[trained_weights, untrained_weights] = get_weights(spat_model, abc)
trained_weights                      = concat_trained_weights(trained_weights, untrained_weights)

i = 0
for param in abc.parameters():
    if(i == 0):
        param.data = trained_weights
    i += 1 
    
spat_model.features   = abc
spat_model.classifier = nn.Sequential(*class_) 


#changing input filter dimension of temporal model 
feat_     = list(temp_model.features.children())
class_  = list(temp_model.classifier.children())
feat_[0]  = nn.Conv2d(100, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
class_[6] = nn.Linear(in_features=4096, out_features=101, bias=True)
#temporary model 
abc = nn.Sequential(*feat_)

[trained_weights, untrained_weights] = get_weights(temp_model, abc)
trained_weights                      = concat_trained_weights(trained_weights, untrained_weights)

i = 0
for param in abc.parameters():
    if(i == 0):
        param.data = trained_weights
    i += 1 
    
temp_model.features = abc
temp_model.classifier = nn.Sequential(*class_)


'''
if system has cuda 
'''
use_gpu = torch.cuda.is_available()

model   = Net()

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


model = train_model(model, criterion, optimizer,scheduler, num_epochs=20)

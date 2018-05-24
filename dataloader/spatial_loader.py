import pickle
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure
#for loading the dataset in dataloader
class spatial_dataset(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

#loading image from path and frame number
    def load_ucf_image(self,video_name, index):
        path = self.root_dir +'v_'+video_name+'/frame'
        a=str(index)
        b=a.zfill(6)
        img = Image.open(path +str(b)+'.jpg')
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

    def __getitem__(self, idx):
        if self.mode == 'train' or 'val':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            #taking 5 spatial frames that are 10 framses apart
            clips = [5,15,25,35,45]
        else:
            raise ValueError('There are only train and val mode')

        #getting labels of the video
        label = self.values[idx]
        label = int(label)-1
        
        #returing all the 5 frames 
        if self.mode=='train' or 'val':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                temp = self.load_ucf_image(video_name, index)
                #temp = temp.view([1,3,224,224])
                if(i == 0):
                    data = temp
                else:
#                     print("data shape ", data.shape)
                    data = torch.cat((data,temp))
            sample = (data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample
    
    
class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        #loading the frame count for videos
        with open('./dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()
#         print("Now in loadframe_count ")

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            self.frame_count[videoname]=dic_frame[line]

    def run(self):
#         print("Now in run ")
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.validate()
        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
#         print 'Now in training dict'
        #making a training dictionary i.e. a list of video number with no_of_frames
        self.dic_training={}
        for video in self.train_video:
            nb_frame = self.frame_count[video]
            if(nb_frame > 50):
                key = video+' '+ str(nb_frame)
                self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self):
#         print 'Now in val_sample'

        #similarly making a validation dictionary
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]
            #choosing inly those videos with frames > 50
            if(nb_frame > 50):
                key = video+ ' '+str(nb_frame)
                self.dic_testing[key] = self.test_video[video] 

    def train(self):
#         print("Now in train")
        #applying trabsformation on training videos 
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('Eligible videos for training :',len(training_set),'videos')
        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        #         print("Now in Validate")
        #applying transformation for validation videos 
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('Eligible videos for validation:',len(validation_set),'videos')
        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader




if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=2, num_workers=1, 
                                path='../data/link_to_jpegs_256_1/', 
                                ucf_list='../UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()


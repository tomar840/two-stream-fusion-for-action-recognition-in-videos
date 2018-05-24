import pickle
from PIL import Image
import numpy as np
import pickle
#import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
#from skimage import io, color, exposure

#for loading the dataset in dataloader
class spatio_temporal_dataset(Dataset):
    def __init__(self, dic, spatial_path, temp_path, in_channel, mode, train_transform, val_transform):
        self.keys = dic.keys()
        self.values=dic.values()
        self.spatial_path = spatial_path
        self.temp_path = temp_path
        self.mode = mode
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

        
    def __len__(self):
        return len(self.keys)
    
    def stackopf(self, video):
        self.video = video
        name = 'v_'+self.video
        u = self.temp_path+ 'u/' + name
        v = self.temp_path+ 'v/'+ name
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + 2*j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.val_transform(imgH)
            V = self.val_transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow


#loading image from path and frame number
    def load_ucf_image(self,video_name, index, mode):
        path = self.spatial_path +'v_'+video_name+'/frame'
        a=str(index)
        b=a.zfill(6)
        img = Image.open(path +str(b)+'.jpg')
        if(mode == 'train'):
            transformed_img = self.train_transform(img)
        else:
            transformed_img = self.val_transform(img)
        img.close()
        return transformed_img

    
    def __getitem__(self, idx):
        if self.mode == 'train' or 'val':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            self.clips_idx = 1
            #taking 10 spatial frames that are 10 framses apart
            clips = [10,20,30,40,50,60,70,80,90,99]
        else:
            raise ValueError('There are only train and val mode')

        #getting labels of the video
        label = self.values[idx]
        label = int(label)-1
        
        #returing all the 5 frames 
        if self.mode=='train':
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                temp = self.load_ucf_image(video_name, index,mode='train')
                #temp = temp.view([1,3,224,224])
                if(i == 0):
                    spatial_data = temp
                else:
#                     print("data shape ", data.shape)
                    spatial_data = torch.cat((spatial_data,temp))
            
            temp_data = self.stackopf(video=video_name)
            sample = (spatial_data, temp_data, label)
        
        elif self.mode=='val':
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                temp = self.load_ucf_image(video_name, index, mode='val')
                #temp = temp.view([1,3,224,224])
                if(i == 0):
                    spatial_data = temp
                else:
#                     print("data shape ", data.shape)
                    spatial_data = torch.cat((spatial_data,temp))
            
            temp_data = self.stackopf(video=video_name)
            sample = (spatial_data, temp_data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample
    
    
class spatio_temporal_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, spatial_path, temp_path, ucf_list, ucf_split, train_transform, val_transform):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.spatial_path=spatial_path
        self.temp_path=temp_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.in_channel = in_channel
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        #loading the frame count for videos
        with open('./dataloader/dic/frame_count.pickle','rb') as file:
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
            if(nb_frame > 100):
                key = video+' '+ str(nb_frame)
                self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self):
#         print 'Now in val_sample'

        #similarly making a validation dictionary
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]
            #choosing inly those videos with frames > 50
            if(nb_frame > 100):
                key = video+ ' '+str(nb_frame)
                self.dic_testing[key] = self.test_video[video] 

    def train(self):
#         print("Now in train")
        #applying trabsformation on training videos 
        training_set = spatio_temporal_dataset(dic=self.dic_training, spatial_path=self.spatial_path,
                                               temp_path=self.temp_path, in_channel=self.in_channel,
                                               mode='train', train_transform = self.train_transform,
                                              val_transform = self.val_transform)
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
        validation_set = spatio_temporal_dataset(dic=self.dic_testing, spatial_path=self.spatial_path, 
                                                 temp_path=self.temp_path, in_channel=self.in_channel,
                                                 mode='val', train_transform = self.train_transform, 
                                                 val_transform = self.val_transform)
        
        print('Eligible videos for validation:',len(validation_set),'videos')
        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

if __name__ == '__main__':
    
    dataloader = spatio_temporal_dataloader(BATCH_SIZE=3, num_workers=1, in_channel = 50,
                                spatial_path='../data/link_to_jpegs_256_1/',         
                                temp_path='../data/link_to_tvl1_flow/',
                                ucf_list='../UCF_list/',
                                ucf_split='01',
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
    train_loader, val_loader, test_video = dataloader.run()

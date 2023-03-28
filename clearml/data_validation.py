#THis code is to check if the video is corrupted or not..
#If the video is corrupted delete the video.
import glob
import json
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import copy
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from config import AppConfig


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])


#Check if the file is corrupted or not
def validate_video(vid_path,train_transforms):
    transform = train_transforms
    count = 20
    video_path = vid_path
    frames = []
    a = int(100/count)
    first_frame = np.random.randint(0,a)
    temp_video = video_path.split('\\')[-1]
    for i,frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if(len(frames) == count):
            break
    frames = torch.stack(frames)
    frames = frames[:count]
    return frames


#extract a from from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


#get labels from csv
def get_labels():
    path = "datasets/meta_short.csv"
    header_list = ["file","label"]
    return pd.read_csv(path,names=header_list)


# load the video name and labels from csv
class video_dataset(Dataset):
    def __init__(self,video_names,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.labels = get_labels()
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        temp_video = video_path.split('\\')[-1]
        label = self.labels.iloc[(self.labels.loc[self.labels["file"] == temp_video].index.values[0]),1]
        if(label == 'fake'):
            label = 0
        if(label == 'real'):
            label = 1
        for i,frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames,label
    def frame_extract(self,path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


#count the number of fake and real videos
def number_of_real_and_fake_videos(data_list):
    labels = get_labels()
    fake = 0
    real = 0
    for i in data_list:
        temp_video = i.split('\\')[-1]
        label = labels.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
        if(label == 'fake'):
            fake+=1
        if(label == 'real'):
            real+=1
    return real,fake


def main_actions(config: AppConfig):
    video_files = [str(x) for x in config.dataset_path.glob("**/*.mp4")]
    print("Total no of videos :" , len(video_files))

    count = 0
    for i in video_files:
        try:
            count+=1
            validate_video(i,train_transforms)
        except:
            print("Number of video processed: " , count ," Remaining : " , (len(video_files) - count))
            print("Corrupted video is : " , i)
            continue
    print((len(video_files) - count))

    #to load preprocessod video to memory
    random.shuffle(video_files)
    frame_count = []
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Total no of video: " , len(frame_count))
    print('Average frame per video:',np.mean(frame_count))

    # load the labels and video in data loader
    train_videos = video_files[:int(0.8*len(video_files))]
    valid_videos = video_files[int(0.8*len(video_files)):]
    print("train : " , len(train_videos))
    print("test : " , len(valid_videos))
    print("TRAIN: ", "Real:",number_of_real_and_fake_videos(train_videos)[0]," Fake:",number_of_real_and_fake_videos(train_videos)[1])
    print("TEST: ", "Real:",number_of_real_and_fake_videos(valid_videos)[0]," Fake:",number_of_real_and_fake_videos(valid_videos)[1])
    
    train_data = video_dataset(train_videos,sequence_length = 10,transform = train_transforms)
    val_data = video_dataset(valid_videos,sequence_length = 10,transform = train_transforms)
    train_loader = DataLoader(train_data,batch_size = 4,shuffle = True,num_workers = 4)
    valid_loader = DataLoader(val_data,batch_size = 4,shuffle = True,num_workers = 4)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
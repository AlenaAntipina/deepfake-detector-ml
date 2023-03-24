import json
import glob
import numpy as np
import cv2
import copy

from pathlib import Path
import dlib
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
import face_recognition
from tqdm.autonotebook import tqdm


def getAverageFrameCount():

    video_files = getVideoFiles()
    
    frame_count = []
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))<150):
            video_files.remove(video_file)
            continue
        frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("frames" , frame_count)
    print("Total number of videos: " , len(frame_count))
    print('Average frame per video:',np.mean(frame_count))

def getVideoFiles():
    video_files = []
    data_path = Path(f'datasets/T_deepfake')
    
    fake_frame_filenames =  _find_filenames(data_path / 'manipulated_sequences/Deepfakes/c23/videos/', '*.mp4')
    fake_frame_filenames += _find_filenames(data_path / 'manipulated_sequences/Face2Face/c23/videos/', '*.mp4')
    fake_frame_filenames += _find_filenames(data_path / 'manipulated_sequences/FaceSwap/c23/videos/', '*.mp4')
    fake_frame_filenames += _find_filenames(data_path / 'manipulated_sequences/NeuralTextures/c23/videos/', '*.mp4')

    real_frame_filenames = _find_filenames(data_path / 'original_sequences/youtube/c23/videos/', '*.mp4')
    
  #  video_files += real_frame_filenames
   # video_files += fake_frame_filenames
    

    video_files += glob.glob('datasets/T_deepfake/manipulated_sequences/Deepfakes/c23/videos/*.mp4')
    video_files += glob.glob('datasets/T_deepfake/manipulated_sequences/Face2Face/c23/videos/*.mp4')
    video_files += glob.glob('datasets/T_deepfake/manipulated_sequences/FaceSwap/c23/videos/*.mp4')
    video_files += glob.glob('datasets/T_deepfake/manipulated_sequences/NeuralTextures/c23/videos/*.mp4')
    video_files += glob.glob('datasets/T_deepfake/original_sequences/youtube/c23/videos/*.mp4')
    print(len(video_files))

    return video_files

def _find_filenames(file_dir_path, file_pattern): 
    return list(file_dir_path.glob(file_pattern))

# to extract frame
def frame_extract(path):
  vidObj = cv2.VideoCapture(path) 
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image

# process the frames
def create_face_videos(path_list,out_dir):
  already_present_count =  glob.glob(out_dir+'*.mp4')
  print("No of videos already present " , len(already_present_count))
  for path in tqdm(path_list):
    out_path = os.path.join(out_dir,path.split('/')[-1])
    file_exists = glob.glob(out_path)
    if(len(file_exists) != 0):
      print("File Already exists: " , out_path)
      continue
    frames = []
    flag = 0
    face_all = []
    frames1 = []
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (112,112))
    for idx,frame in enumerate(frame_extract(path)):
      #if(idx % 3 == 0):
      if(idx <= 150):
        frames.append(frame)
        if(len(frames) == 4):
          faces = face_recognition.batch_face_locations(frames)
          for i,face in enumerate(faces):
            if(len(face) != 0):
              top,right,bottom,left = face[0]
            try:
              out.write(cv2.resize(frames[i][top:bottom,left:right,:],(112,112)))
            except:
              pass
          frames = []
    try:
      del top,right,bottom,left
    except:
      pass
    out.release()

def main():
  print("start preprocessing")
  create_face_videos(getVideoFiles(),'datasets/face_only')
  #getAverageFrameCount()

if __name__ == "__main__":
    main()
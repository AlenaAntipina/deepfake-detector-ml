import json
import glob
import numpy as np
import cv2
import copy


def getAverageFrameCount():

  #  [file.filename for file in dataset/data/manipulated_sequences]


    video_files =  'dataset/data/manipulated_sequences'
    video_files += 'dataset/data/original_sequences'
    
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
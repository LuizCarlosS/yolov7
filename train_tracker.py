# #entity tracking
# self.window = window #frames
# self.entities = {}#entity_id: class
# self.memory = [[] for i in range(self.window)]#keeps track of entities from up to window frames. Each list contains all entities found inside the latest 10 frames.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import PIL

import pandas as pd
import numpy as np

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

class MOT17Dataset(Dataset):
    def __init__(self, ds_path: str = r'K:\Users\krish\Downloads\MOT17\train') -> None:
        super().__init__()
        self.base_path = ds_path
        gt_path = ds_path + r'\<seqname>\gt\gt.txt'
        self.img_path = ds_path + r'\<seqname>\img1'
        file_list = os.listdir(ds_path)
        self.df = pd.DataFrame(columns=['seqname', 'frame', 'ID',
                                        'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z'])
        for file in file_list:
            df = pd.read_csv(gt_path.replace('<seqname>', file), names=[
                'frame', 'ID', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z'])
            df['seqname'] = file
            df = df.sort_values('frame')
            self.df = pd.concat([self.df, df])
        self.df = self.df.reset_index(drop=True)


    
    def __len__(self):
        return self.df.seqname.nunique()

    def __get_item__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seqname = self.df.seqname.nunique()[idx]
        seqdata = self.df[self.df.seqname==seqname]
        
        #load images
        images = []
        for img_file in os.listdir(self.img_path):
            images.append(load_image(os.path.join(self.img_path, img_file)))

        #create heatmap
        
        ht_mp = torch.zeros(images[0].shape)

        #parse labels
        for i, elem in seqdata.iterrows():
            ht_mp[elem.centery][elem.centerx] = 1

            #list of bounding_boxes

        


ds = MOT17Dataset()
print(ds.df.head())

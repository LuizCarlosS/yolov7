# #entity tracking
# self.window = window #frames
# self.entities = {}#entity_id: class
# self.memory = [[] for i in range(self.window)]#keeps track of entities from up to window frames. Each list contains all entities found inside the latest 10 frames.
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.tracker import YoloTracker
from utils.general import labels_to_class_weights
from utils.loss import ComputeLossOTA

import logging
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.DEBUG)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# class IDModule():
#     def __init__(self, memory_max: int = 5) -> None:
#         self.memory_max = memory_max #max ammount of past frames to keep ID's
#         self.memory = [] #list of dicts, each dict corresponds to a frame. Each key is an unique ID and each value is a list of the points of the detection
#         self.latest_id = -1
#     def save_frame(self, frame_detections, frame_heatmap):
#         if not len(self.memory): #first frame
#             data = {}
#             for det in frame_detections:
#                 data[self.latest_id+1] = frame_detections
            

class MOT17Dataset(Dataset):
    def __init__(self, ds_path: str = r'K:\Users\krish\Downloads\MOT17\train', transforms = None, subset='train') -> None:
        super().__init__()
        self.base_path = ds_path
        gt_path = ds_path + r'\<seqname>\gt\gt.txt'
        self.img_path = ds_path + r'\<seqname>\img1'
        file_list = os.listdir(ds_path)
        self.transforms = transforms
        self.df = pd.DataFrame(columns=['seqname', 'frame', 'ID',
                                        'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z'])
        
        for file in file_list:
            df = pd.read_csv(gt_path.replace('<seqname>', file), names=[
                'frame', 'ID', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'z'])
            df['seqname'] = file
            df = df.sort_values('frame')
            self.df = pd.concat([self.df, df])
        self.df = self.df.reset_index(drop=True)
        self.df['centery'] = (self.df['bb_top'] + self.df['bb_height'])/2
        self.df['centerx'] = (self.df['bb_left'] + self.df['bb_width'])/2
        tr, ts = train_test_split(self.df, train_size=0.8, random_state=42)
        if subset == 'train':
            self.df = tr
        elif subset == 'test':
            self.df = ts
        else:
            raise NameError
    
    def __len__(self):
        return self.df.seqname.nunique()

    def __get_item__(self, idx, img_size = 640):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seqname = self.df.seqname.unique()[idx]
        seqdata = self.df[self.df.seqname==seqname]
        
        #load images (frames)
        rescalers = []
        samples = []
        for img_file in os.listdir(self.img_path.replace('<seqname>', seqname)):
            img = load_image(os.path.join(self.img_path.replace('<seqname>', seqname), img_file))
            h, w, c = img.shape
            resized = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
            rescalers.append([resized.shape[0]/h, resized.shape[1]/w]) #multiply the coordinates with these to obtain rescaled values
            samples.append(resized)

        y_cols = ['bb_top', 'bb_height', 'centery']
        x_cols = ['bb_left', 'bb_width', 'centerx']
        for col in y_cols:
            seqdata[col] = seqdata[col].apply(lambda x: x*rescalers[0][0]/h)
        for col in x_cols:
            seqdata[col] = seqdata[col].apply(lambda x: x*rescalers[0][1]/w)
        
        if self.transforms is not None:
            samples = [self.transforms(image=img)['image'] for img in samples]
        
        #label
        seq_heatmaps = []
        seq_bboxes = []
        seq_ids = []
        for frame_id in seqdata.frame.unique():
            frame_data = seqdata[seqdata.frame == frame_id]
            #parse labels
            bboxes = []
            ids = []
            ht_mp = torch.zeros((img_size, img_size))
            for i, elem in frame_data.iterrows():
                
                ht_mp[int(elem.centery)][int(elem.centerx)] = 1
                #list of bounding_boxes
                bboxes.append([0, elem.centerx, elem.centery, elem.bb_width, elem.bb_height])
                ids.append(elem.ID)
            seq_heatmaps.append(ht_mp)
            seq_bboxes.append(bboxes)
            seq_ids.append(ids)
        
        assert len(seq_heatmaps) == len(samples)
        
        return samples, seq_bboxes, seq_ids, seq_heatmaps

# def create_batches(data, batch_size):
#     return torch.cat([data[int(i*batch_size):int((i+1)*batch_size)].unsqueeze(0) for i in range(int(len(data)/batch_size))])

def train(ds_path = r'K:\Users\krish\Downloads\MOT17\train', batch_size=1, 
                cfg = './cfg/training/yolov7.yaml',
                hyp = './data/hyp.scratch.p5.yaml',
                nc = 80,
                epochs=30,
                device='cuda'):

    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
        ])
    train_dataset = MOT17Dataset(ds_path=ds_path, transforms=transform, subset='train')
    test_dataset = MOT17Dataset(ds_path=ds_path, transforms=transform, subset='test')
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
    # test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
    # nb = 
    #instantiate model
    full_model = YoloTracker()
    total_batch_size = 32
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in full_model.detection_branch.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)
    
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2
    
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    gs = max(int(full_model.detection_branch.stride.max()), 32)  # grid size (max stride)
    nl = full_model.detection_branch.model[-1].nl
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (640 / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = False
    full_model.detection_branch.nc = nc  # attach number of classes to model
    full_model.detection_branch.hyp = hyp  # attach hyperparameters to model
    full_model.detection_branch.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    full_model.detection_branch.names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = 0 - 1  # do not move
    scaler = amp.GradScaler(enabled=device)
    criterion_tracker = torch.nn.SmoothL1Loss()
    compute_loss = ComputeLossOTA(full_model.detection_branch)
    logging.debug(f'Starting training of {epochs} epochs.')
    for epoch in range(0, epochs):  # epoch ------------------------------------------------------------------
        print('train started')
        full_model.train()

        mloss = torch.zeros(3, device=device)  # mean losses
        # pbar = enumerate(train_loader)
        
        # pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        
        logging.debug(f'Starting dataset loading and training')
        for index in range(len(train_dataset)):
            samples, bboxes, ids, heatmaps = train_dataset.__get_item__(index) #grab a sequene of images
            
            # samples = create_batches(samples, batch_size=total_batch_size)
            # bboxes = create_batches(bboxes, batch_size=total_batch_size)
            # ids = create_batches(ids, batch_size=total_batch_size)
            # heatmaps = create_batches(heatmaps, batch_size=total_batch_size)

            for i, (seq_samples, seq_bboxes, seq_ids, seq_heatmaps) in enumerate(zip(samples, bboxes, ids, heatmaps)):
                
                seq_samples = seq_samples.to(device, non_blocking=True).float() / 255.0
                seq_samples = seq_samples.unsqueeze(0)
                print(seq_samples.shape)
                # Forward
                with amp.autocast(enabled=True):
                    pred_yolo, pred_track = full_model(seq_samples)  # forward
                    print(len(pred_yolo))
                    for i in pred_yolo:
                        print(i.shape)
                    loss_yolo, loss_items_yolo = compute_loss(pred_yolo, torch.as_tensor(seq_bboxes).to(device), seq_samples)  # loss scaled by batch_size

                    loss_heatmap = criterion_tracker(seq_heatmaps, pred_track)
                
                complete_loss = (3*loss_yolo + loss_heatmap)/4
                # Backward
                scaler.scale(complete_loss).backward()
                logging.debug(f'Yolo Loss: {loss_yolo}')
                logging.debug(f'Tracker Loss: {loss_heatmap}')
                # logging.debug(f'Shape of prediction)
                #reset feature extraction from prev frame
                full_model.reset_run()
        break

if __name__ == '__main__':
    train()
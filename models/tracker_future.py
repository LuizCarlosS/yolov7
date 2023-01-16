import torch
import torch.nn as nn
import yaml

from models.yolo import Model
from utils.torch_utils import intersect_dicts


class LinearTransformer(nn.Module):
    def __init__(self, linear_path) -> None:
        super().__init__()
        self.model = torch.load(linear_path)
    
    def forward(self, x):
        return self.model(x)

class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return

class TrackingHead(nn.Module):
    '''
    This module takes the features extracted from the yolo model and uses them to track 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return

class YoloTracker(nn.Module):
    def __init__(self,
                cutoff_layer=50, 
                weights = 'yolov7.pt',
                device = 'cuda',
                cfg = '../cfg/training/yolov7.yaml',
                hyp = '../data/hyp.scratch.p5.yaml',
                nc = 80,
                window = 10
                ) -> None:
        super().__init__()
        with open(hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        
        tracking_classes = [0]
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
        
        self.detection_branch = model
        self.yolo_features = {}
        self.detection_branch.model[cutoff_layer].register_forward_hook(self.get_features('features'))
        self.detection_branch = self.detection_branch['model']
        #entity tracking
        self.window = window #frames
        self.entities = {}#entity_id: class
        self.memory = [[] for i in range(self.window)]#keeps track of entities from up to window frames. Each list contains all entities found inside the latest 10 frames.
        self.fuse = FusionModel()
        self.track = TrackingHead()
        self.embed = LinearTransformer(in_chans=4) #4th channel is the detection center map
        
    def forward(self, t, t_1, h_t_1):
        pred = self.detection_branch(t)
        prev_frame = self.embed(torch.stack([t_1, h_t_1], dim=4))
        self.track(self.fusion(self.yolo_features['features'], prev_frame))
    
    def get_features(self, name):
        def hook(self, model, input, output):
            self.yolo_features[name] = output.detach()
        return hook
import torch
import torch.nn as nn
import yaml
from torch import nn

from models.yolo import Model
from utils.torch_utils import intersect_dicts

yolo_features = {}
def get_features(name):
    def hook(model, input, output):
        yolo_features[name] = output.detach()
    return hook

class TrackingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.conv1_3 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(512)
        # self.conv2_3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.flatten = nn.Flatten()
        # self.maxpool = nn.MaxPool2d(2)
        # self.linear1 = nn.Linear(64*4*8, 640*640*1)

        self.convt1 = nn.ConvTranspose2d(2048, 512, 4, stride=4)
        self.convt2 = nn.ConvTranspose2d(512, 64, 4, stride=4)
        self.convt3 = nn.ConvTranspose2d(64, 1, 2, stride=2)
    # def forward(self, x):
    #     print(x.size())
    #     x = self.bn1(self.relu(self.conv1_3(x)))
    #     x = self.bn2(self.relu(self.conv2_3(x)))
    #     x = self.maxpool(x)
    #     x = self.flatten(x)
    #     print(x.size())
    #     x = self.sigmoid(self.linear1(x))
    #     return x

    def forward(self, x):
        x = self.bn1(self.relu(self.convt1(x)))
        x = self.bn2(self.relu(self.convt2(x)))
        x = self.sigmoid(self.convt3(x))
        return x

class YoloTracker(nn.Module):
    def __init__(self,
                cutoff_layer=50, 
                weights = 'yolov7.pt',
                device = 'cuda',
                cfg = './cfg/training/yolov7.yaml',
                hyp = './data/hyp.scratch.p5.yaml',
                nc = 80,
                window = 2,
                ) -> None:
        super().__init__()
        with open(hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        
        tracking_classes = [0]
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = ckpt['model'].float()
            # model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            # exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []  # exclude keys
            # state_dict = ckpt['model'].float().state_dict()  # to FP32
            # state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            # model.load_state_dict(state_dict, strict=False)  # load
        
        self.detection_branch = model
        self.detection_branch.model[cutoff_layer].register_forward_hook(get_features('features'))
        # self.detection_branch = self.detection_branch['model']
        self.track_head = TrackingHead().to(device)
        
    def forward(self, current_frame):
        to_track = False
        if 'features' in yolo_features.keys():
            last_feats = yolo_features['features']
            to_track = True
            trk_p = None
        yolo_p = self.detection_branch(current_frame)
        if to_track:
            trk_p = self.track_head(torch.cat([last_feats, yolo_features['features']], dim=1))
            return yolo_p, trk_p
        return yolo_p, None
    


# def main():


# if __name__ == '__main__':
#     main()
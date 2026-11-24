from .common import PREVIOUS
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, resnet50


class ImageConceptModel(nn.Module):
    def __init__(self, architecture='resnet18', count=310, pretrained=False):
        super().__init__()
        self.architecture = architecture
        self.backbone = resnet18(weights=None) if architecture == 'resnet18' else resnet50(weights=None)
        dimension = self.backbone.fc.in_features
        if pretrained:
            name = 'resnet18-f37072fd.pth' if architecture == 'resnet18' else 'resnet50-11ad3fa6.pth'
            self.backbone.load_state_dict(torch.load('C:/Users/Administrator/.cache/torch/hub/checkpoints/'+name, map_location='cpu', weights_only=True))
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(dimension, count)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None,:,None,None])
        self.register_buffer('feature_mean', torch.zeros(dimension))
        self.register_buffer('feature_std', torch.ones(dimension))
        self.standardize_features = False

    def forward(self, rgb, return_spatial=False):
        b = self.backbone
        z = (rgb-self.mean)/self.std
        z = b.maxpool(b.relu(b.bn1(b.conv1(z))))
        z = b.layer4(b.layer3(b.layer2(b.layer1(z))))
        features = z.mean((-2,-1))
        value = features
        if self.standardize_features:
            value = ((value-self.feature_mean)/self.feature_std).clamp(-8,8)
        logits = self.head(value)
        return (logits, features, z) if return_spatial else logits


def load_existing():
    source = PREVIOUS/'解释机制复筛_2026-11-14/concepts200/seed0/concept_model.pt'
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model = ImageConceptModel('resnet50', len(saved['manifest']['attribute_names']), pretrained=True)
    model.head.load_state_dict(saved['state_dict'])
    model.feature_mean.copy_(torch.as_tensor(saved['mean']))
    model.feature_std.copy_(torch.as_tensor(saved['std']))
    model.standardize_features = True
    return model, saved['manifest'], str(source)


def load_meta():
    return dict(np.load(PREVIOUS/'可替换部件参照初步验证_2026-11-14/metadata.npz'))


def attribute_region(name):
    family = name.split('::')[0]
    if 'bill' in family: return 0
    if any(w in family for w in ('forehead','nape','crown','head')): return 1
    if 'eye' in family: return 2
    if 'wing' in family: return 3
    if 'tail' in family: return 4
    if 'back' in family or 'upperparts' in family: return 5
    if 'breast' in family: return 6
    if 'belly' in family or 'underparts' in family: return 7
    if 'throat' in family: return 8
    if 'leg' in family: return 9
    return 10


REGIONS = ['bill','head','eye','wing','tail','back','breast','belly','throat','leg']
PARTS = [[1],[4,5,9],[6,10],[8,12],[13],[0],[3],[2],[14],[7,11]]

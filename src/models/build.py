from torchvision import models
from torch import nn
import sys
sys.path.append('./models')
sys.path.append('./models/extractors')
sys.path.append('../src')
from convnext import build_extractor
from head import Classifier


class Model(nn.Module):
    def __init__(self, extractor, head):
        super(Model, self).__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, input_tensor):
        features = self.extractor(input_tensor)
        categorical_probs = self.head(features)

        return features, categorical_probs

def resnet():
    model = models.resnet34(pretrained=True)
    return model

def build_model(cfgs, pretrained=False, **kwargs):
    extractor = build_extractor(cfgs, pretrained, **kwargs)
    head = Classifier(cfgs)

    return Model(extractor, head)


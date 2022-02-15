import torch.nn as nn
import torchvision.models as models


class ColeNet(nn.Module):

    def __init__(self, backbone):
        super(ColeNet, self).__init__()
        self.model = self.get_backbone(backbone=backbone)

    def get_backbone(self, backbone):
        if backbone == "vgg":
            model = models.vgg16(pretrained=True)
            model.classifier = nn.Linear(model.classifier[0].in_features, 3)
        elif backbone == "resnet":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 3)
        elif backbone == "alexnet":
            model = models.alexnet(pretrained=True)
            model.classifier = nn.Linear(model.classifier[1].in_features, 3)
        elif backbone == "densenet":
            model = models.densenet169(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, 3)
        elif backbone == "inception":
            model = models.inception_v3(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 3)
        else:
            raise NotImplementedError("Unknown model backbone")

        return model

    def forward(self, x):
        x = self.model(x)

        return x

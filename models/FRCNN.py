
from .base import BaseModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch.nn as nn


class Model(BaseModel, nn.Module):
    def __init__(self):
        BaseModel.__init__(self)
        nn.Module.__init__(self)
        self.name = "FRCNN"
        self.model = None
        self._create_model()

    def _create_model(self):
        # Load the pre-trained Faster R-CNN model
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Freeze the backbone layers
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Freeze the RPN layers
        for param in self.model.rpn.parameters():
            param.requires_grad = False

        # Unfreeze the box predictor and ROI heads
        for param in self.model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        return self.model(x)

    def parameters(self):
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        return self.model.parameters()

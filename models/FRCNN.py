
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class Model():
    # Load the pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Freeze the backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Freeze the RPN layers
    for param in model.rpn.parameters():
        param.requires_grad = False

    # Unfreeze the box predictor and ROI heads
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True

    for param in model.roi_heads.parameters():
        param.requires_grad = True

    # Verify the requires_grad status for each component
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

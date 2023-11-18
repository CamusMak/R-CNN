from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


def faster_rccn(freeze=False):

    rcnn = fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # freeze all layers

    if not freeze:
        for p in rcnn.parameters():
            p.requires_grad = True

    

    return rcnn



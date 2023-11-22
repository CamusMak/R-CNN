from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def faster_rccn(freeze=False,trainable_backbone_layers = 3):

    """
    Returns custom R-CNN model

    --------------------------

    freeze:
        whether freeze layers or not
    
    trainable_backbone_layers:
        how many layers' weigths set trainable from the end
    """

    rcnn = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT,trainable_backbone_layers=trainable_backbone_layers)

    # freeze all layers

    if not freeze:
        for p in rcnn.parameters():
            p.requires_grad = True

    

    return rcnn



import torch
from torch.utils.data import DataLoader

from icecream import ic

import sys
sys.path.append("../utils")







def validate(model,batch,optimizer,device):

    """
    
    Validatas model for given batch

    ------------------------------------------------------


    model:
        model to train, in this case R-CNN

    batch:
        input data for current batch
    
    optimizer:
        optimizer to use to update model parameters
    
    device:
        cuda if available else cpu


    --------------------------------------------------------

    split batch into X and Y (image, label)

    X is a list of images
    Y is a list of boxes and labels for images


    """


    X,Y = batch
    X = [x.to(device) for x in X]
    Y = [{k:v.to(device) for k,v in t.items()} for t in Y]

    model.to(device)

    model.train()


    optimizer.zero_grad()
    losses = model(X,Y)
    loss = sum(los for los in losses.values())

    return loss




    


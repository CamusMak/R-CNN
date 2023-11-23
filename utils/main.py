import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("../utils")

from dataset import CatDogDataset
from model import faster_rccn
from split_roots import split
from helper_functions import collate_fn
from icecream import ic
from train_rcnn import train_model

TRAIN_ROOT = "../DATA/train"
VALIDATION_ROOT = '../DATA/validation'
TEST_ROOT = '../DATA/test'

## split files into several roots
split("../data/images",source_annoation_root='../data/annotations',train_root=TRAIN_ROOT,validation_root=VALIDATION_ROOT,test_root=TEST_ROOT)



quit()

BATCH_SIZE = 1
LR = 0.01



train_set = CatDogDataset(TRAIN_ROOT)
validataion_set = CatDogDataset(VALIDATION_ROOT)


train_dataloader = DataLoader(train_set,batch_size=BATCH_SIZE,collate_fn=collate_fn)
validation_dataloader = DataLoader(validataion_set,batch_size=BATCH_SIZE,collate_fn=collate_fn)




# for ind, batch in enumerate(train_dataloader):
    
#     print(batch)
#     break

# quit()

## split files into several roots
# split("../data/images",source_annoation_root='../data/annotations',train_root=TRAIN_ROOT,validation_root=VALIDATION_ROOT,test_root=TEST_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = faster_rccn()

optimizer = torch.optim.SGD(lr=LR,params=[param for param in model.parameters() if param.requires_grad])

train_model(model=model,train_dataloader=train_dataloader,
      validataion_dataloader=validation_dataloader,n_of_iterations=1,
      optimizer=optimizer,device=device)


# ((None,3,250,250))
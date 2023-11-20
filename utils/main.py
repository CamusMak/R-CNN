from torch.utils.data import DataLoader

import sys
sys.path.append("../utils")

from dataset import CatDogDataset
from model import faster_rccn
from split_roots import split
from train import train_model
from icecream import ic

TRAIN_ROOT = "../DATA/train"
VALIDATION_ROOT = '../DATA/validation'
TEST_ROOT = '../DATA/test'



## split files into several roots
# split("../data/images",source_annoation_root='../data/annotations',train_root=TRAIN_ROOT,validation_root=VALIDATION_ROOT,test_root=TEST_ROOT)




model = faster_rccn()

train_model(model=model,n_of_iterations=1)


# ((None,3,250,250))
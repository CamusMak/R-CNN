from torch.utils.data import DataLoader

import sys
sys.path.append("../utils")

from dataset import CatDogDataset
from model import faster_rccn
from split_roots import split
from icecream import ic

TRAIN_ROOT = "../DATA/train"
VALIDATION_ROOT = '../DATA/validation'
TEST_ROOT = '../DATA/test'



## split files into several roots
# split("../data/images",source_annoation_root='../data/annotations',train_root=TRAIN_ROOT,validation_root=VALIDATION_ROOT,test_root=TEST_ROOT)



# image_root = "../data/"

train_set = CatDogDataset(root=TRAIN_ROOT)
validataion_set = CatDogDataset(root=VALIDATION_ROOT)
test_set = CatDogDataset(root=TEST_ROOT)


# why batch size > 1 does not work???

train_dataloader = DataLoader(train_set,batch_size=1)
# train_dataloader = DataLoader(train_set,batch_size=1)
# train_dataloader = DataLoader(train_set,batch_size=1)


def train_model(model,n_of_iterations=10):

    for _ in range(n_of_iterations):

        for X,Y in train_dataloader:


            X = [X]
            Y = [Y]

            print(X)
            print(Y)

            # for k,v in Y.items():
            #     print(k,v)
            # quit()

            loss = model(X,Y)
            quit()
            # print(loss)


    







# model
# model = faster_rccn()
# losses = model(X,y)
# loss 
# # history = model.
# model.compile()
# model.forward()
# model.train(True)

# ic("Done")

# print(train_set[90])
# 
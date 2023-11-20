import torch


images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)


print(images[0].shape)
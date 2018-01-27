import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

resnet = models.resnet18(pretrained=True)

model = resnet.state_dict()

for k,v in model.items():
    nv = v.numpy()
    print ("name: {0} \t data: {1} ".format(k,nv.shape))
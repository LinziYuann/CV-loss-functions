import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

## refer: Dynamic filter networks - theano implementation
class DynamicFilter(nn.Module):
    def __init__(self, Cin, k_size=3):
        super(DynamicFilter, self).__init__()
        self.k_size = k_size
        # define filter_geneM for filter computation
        self.filter_geneM = torch.nn.Sequential(
            nn.Conv2d(Cin, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(32, self.k_size*self.k_size, kernel_size=5, stride=1, padding=2))

    def forward(self, tensorInput):
        image = tensorInput[0]
        refer = tensorInput[1]
        tensorIn = torch.cat([image, refer], dim=1)
        filters = self.filter_geneM(tensorIn) # filters (B, k_size*k_size, H, W)
        filter_size = filters.size() 
        filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size), filter_size[2], filter_size[0], filter_size[1])) 
        filter_localexpand = torch.from_numpy(filter_localexpand_np).float() 
        input_localexpanded = F.conv2d(image, filter_localexpand, padding=1) 
        output = input_localexpanded * filters 
        output = torch.sum(output, dim=1, keepdim=True)
        return output
        

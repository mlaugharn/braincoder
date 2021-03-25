import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange

class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x

# encoder = nn.Sequential(
#     nn.Conv2d(3, 10, kerneil_size=3),
#     nn.MaxPool2d(kernel_size=2),
#     nn.ReLU(),
#     nn.Conv2d(10, 10, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2),
#     nn.ReLU(),
#     nn.Dropout2d(),
#     nn.Conv2d(10, 10, kernel_size=3),
# #     PrintSize(),
#     Rearrange('b c h w -> b (c h w)'),
#     nn.Linear(10*24*24, NUM_VOXELS)
# ) # hand-rolled

import torchvision.models
# https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf

from torch_cluster import knn_graph
import torch_geometric
import torch
import numpy as np

def voxels_to_graph_template(xyz, k=16):
    x,y,z = xyz
    x,y,z = x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)
    pos = torch.from_numpy(np.hstack((x,y,z)))
    sparse = knn_graph(pos, k=k)
    return torch_geometric.utils.to_dense_adj(sparse)[0]

class AdjacencyMatrixLayer(nn.Module):
    def __init__(self, xyz, NUM_VOXELS=4643):
        super(AdjacencyMatrixLayer, self).__init__()
        k = NUM_VOXELS
        adj_matrix = voxels_to_graph_template(xyz, k=16)
        # self.adj_matrix = nn.Parameter(data=self.adj_matrix, requires_grad = False)
        with torch.no_grad():
            self.adj_matrix = nn.Linear(NUM_VOXELS, NUM_VOXELS, bias=False)
            self.adj_matrix.weight.data = torch.full((NUM_VOXELS, NUM_VOXELS), 1/NUM_VOXELS) + adj_matrix
            self.adj_matrix.requires_grad = False
        self.W1 = nn.Linear(512, NUM_VOXELS)
        self.W2 = nn.Linear(NUM_VOXELS, NUM_VOXELS)

    def forward(self, x):
        h1 = self.W1(x)
        h2 = h1 + self.adj_matrix(h1)
        out = h1 + self.W2(h2)
        return out


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, xyz, num_classes = 'unused', NUM_VOXELS=-1):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, NUM_VOXELS),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(NUM_VOXELS, NUM_VOXELS)
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(512),
                nn.Linear(512, NUM_VOXELS)
                # AdjacencyMatrixLayer(xyz)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, NUM_VOXELS),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(NUM_VOXELS, NUM_VOXELS),
                nn.ReLU(inplace=True),
                nn.Linear(NUM_VOXELS, NUM_VOXELS),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

# original_model = torchvision.models.resnet34(pretrained=True)
# arch = 'resnet34'
# encoder = FineTuneModel(original_model, arch, None)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class NCD(nn.Module):
    def __init__(self, mode ='rc'):
        super(NCD, self).__init__()
        self.mode = mode
        net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature = nn.Sequential(*list(net.children())[0:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(net.fc.in_features, 1))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def feat(self, x):
        b = x.shape[0]

        x = x.view((b*10, -1) + x.shape[3:])
        x = self.feature(x)
        x = self.avgpool(x)

        x = x.view(b, 10, -1)
        x = x - 0.5 * (x[:, 0:1] + x[:, 1:2])
        return x
    
    def forward(self, x):
        b = x.shape[0]


        choices = x[:, 8:].unsqueeze(dim=2)
        row1 = x[:, 0:3].unsqueeze(1)
        row2 = x[:, 3:6].unsqueeze(1)
        row3_p = x[:, 6:8].unsqueeze(1).repeat(1, 8, 1, 1, 1)
        row3 = torch.cat((row3_p, choices), dim = 2)
        rows = torch.cat((row1, row2, row3), dim = 1)


        if self.mode == 'r':
            x = self.feat(rows)
        
        elif self.mode == 'rc':
            col1 = x[:, 0:8:3].unsqueeze(1)
            col2 = x[:, 1:8:3].unsqueeze(1)
            col3_p = x[:, 2:8:3]. unsqueeze(dim = 1).repeat(1, 8, 1, 1, 1)
            col3 = torch.cat((col3_p, choices), dim=2)

            cols= torch.cat((col1, col2, col3), dim =1)

            x = self.feat (rows) + self.feat(cols)

        x = self.fc(x.view(b*10, -1 ))
        
        return x.view(b, 10)






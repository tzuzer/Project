from torch import nn
import torch
import torch.nn.functional as F
# Bilinear Attention Pooling


class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)

        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)
            if i == 0:
                feature_matrix = AiF
            else:
                feature_matrix = torch.cat([feature_matrix, AiF], dim=1)
        return feature_matrix


class mymodel(nn.Module):
    def __init__(self, cfg, model):
        super(mymodel, self).__init__()

        # 预训练网络 Resnet50
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.classifier = nn.Sequential(*list(model.children())[-2:-1])

        # Attention Maps
        self.attentions = nn.Conv2d(2048, 50, kernel_size=1, bias=False)
        self.dropout2 = nn.Dropout(0.3)
        self.bap = BAP(pool='GAP')
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048*50, 2048)
        self.fc1 = nn.Linear(2048, 17)

    def forward(self, x):
       # x [10, 3, 224, 224]
        w = x.size()
        xx = self.features(x)
        L = self.attentions(xx)


        feature_matrix = self.bap(xx, L)

        feat_linear = feature_matrix.view(w[0], -1)  # 10 x parts x 2048
        feats = self.dropout2(self.fc(feat_linear))  # 10 x 2048
        feats1 = self.dropout2(self.fc1(feats))  # 10 x17

        return feats1



import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PredictorNetwork(nn.Module):

    def __init__(self, input_size, output_size):

        super(PredictorNetwork, self).__init__()
    
        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU()
        )

        self.regression = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, output_size)
        )

        total_param = 0
        for n,m in self.feature.named_modules():
            if hasattr(m,'weight'):
                total_param += m.weight.numel()
        self.feature_param_count = total_param

        # initial
        for n,p in self.named_modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()

        for i in range(len(self.regression)):
            if type(self.regression[i]) == nn.Linear:
                init.orthogonal_(self.regression[i].weight, 0.01)
                self.regression[i].bias.data.zero_()


    def forward(self, state):
        x = self.feature(state)
        predict = self.regression(x)
        return predict


class DistancePredictor():

    def __init__(self, input_size, output_size, use_cuda, learning_rate):
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.predictor = PredictorNetwork(input_size, output_size).to(self.device)
        self.forward_mse = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.predictor.parameters()), lr=learning_rate)
        self.p_loss = []


    def fit(self, X, Y):
        X = torch.FloatTensor(X).to(self.device)
        Y = torch.FloatTensor(Y).to(self.device)
        predict = self.predictor(X)
        self.optimizer.zero_grad()
        predictor_loss = F.mse_loss(predict, Y)
        loss = predictor_loss.item()
        self.p_loss.append(loss)
        predictor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return loss
        

    def predict(self, X):
        return self.predictor(X).detach()
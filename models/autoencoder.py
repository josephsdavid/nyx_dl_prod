import torch.nn as nn
import torch
import torch.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim = 256, input_dim = 28038) :
        super().__init__()

        self.encoder1 = nn.Linear(input_dim, 500)
        self.encoder2 = nn.Linear(500, 500)
        self.encoder3 = nn.Linear(500, 2000)
        self.encoder4 = nn.Linear(2000, latent_dim)

        self.decoder1 = nn.Linear(latent_dim,2000)
        self.decoder2 = nn.Linear(2000, 500)
        self.decoder3 = nn.Linear(500, 500)
        self.decoder4 = nn.Linear(500, input_dim)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.encoder1(x)
        x = self.relu(x)
        x = self.encoder2(x)
        x = self.relu(x)
        x = self.encoder3(x)
        x = self.relu(x)
        x = self.encoder4(x)
        x = self.relu(x)
        encoded = x
        decoded = encoded
        decoded = self.decoder1(decoded)
        decoded = self.relu(decoded)
        decoded = self.decoder2(decoded)
        decoded = self.relu(decoded)
        decoded = self.decoder3(decoded)
        decoded = self.relu(decoded)
        decoded = self.decoder4(decoded)


        return encoded, decoded # worry about the sign stuff in the training and validation steps



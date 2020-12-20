import torch.nn as nn
import torch
import torch.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim = 256, input_dim = 28038) :
        super().__init__()

        self.dropout = nn.Dropout(0.25)
        self.encoder1 = nn.Linear(input_dim, 2048)
        self.encoder2 = nn.Linear(2048, 1024)
        self.encoder3 = nn.Linear(1024, 512)
        self.encoder4 = nn.Linear(512, latent_dim)

        self.decoder1 = nn.Linear(latent_dim,512)
        self.decoder2 = nn.Linear(512, 1024)
        self.decoder3 = nn.Linear(1024, 2048)
        self.decoder4 = nn.Linear(2048, input_dim)

        self.relu = nn.SELU()


    def forward(self, x):
        x = self.encoder1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.encoder2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.encoder3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.encoder4(x)
        x = self.relu(x)
        encoded = x
        decoded = encoded
        decoded = self.decoder1(decoded)
        decoded = self.relu(decoded)
        x = self.dropout(x)
        decoded = self.decoder2(decoded)
        decoded = self.relu(decoded)
        x = self.dropout(x)
        decoded = self.decoder3(decoded)
        decoded = self.relu(decoded)
        x = self.dropout(x)
        decoded = self.decoder4(decoded)


        return encoded, decoded # worry about the sign stuff in the training and validation steps



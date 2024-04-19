import torch
import torch.nn as nn
import torch.nn.functional as F


class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2,padding=1),  #,stride=2
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3,stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, 2, stride=2),
            nn.ReLU()
            nn.MaxPool3d(2)
        )
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose3d(32, 32, 2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3,stride=2,output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3,stride=2,output_padding=1, padding=1),
            nn.ReLU())
        
        
    def forward(self, x):
    
        # Conv3d expect input [batch_size, channels, depth, height, width].
        x = x.permute(0, 3, 2, 1).unsqueeze(1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.squeeze(1).permute(0, 3, 2, 1)

        return encoded, decoded

    
    def loss(self, pred, gt):
        return self.loss_func(pred, gt)
  
 
class cnn_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.cnn1 = nn.Conv3d(32, 24, 2)
        self.maxpool = nn.MaxPool3d(2)
        self.batchnorm = nn.BatchNorm1d(6912)

        self.fc1 = nn.Linear(6912, 1024) 
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)  
        self.dropout = nn.Dropout(p=0.2) 
       
    
    def forward(self, x):
        x = self.leaky_relu(self.cnn1(x))       
        x = self.maxpool(x)
        
        x = x.view(x.shape[0], -1) 
        
        x = self.batchnorm(x)
        x = self.dropout(x)

        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)  

        return x.squeeze()
    
    def loss(self, pred, gt):
        return self.loss_func(pred, gt)
        
      
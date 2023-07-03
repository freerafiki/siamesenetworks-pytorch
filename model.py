import torch.nn as nn
import torch
import pdb 

class MiniSiameseNetwork(nn.Module):
    def __init__(self):
        super(MiniSiameseNetwork, self).__init__()
        # input_shape = (512, 512, 3)
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_block1 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64*7*7, 1024),
            nn.ReLU()
        )
        self.fc_block3 = nn.Sequential(
            nn.Linear(1024, 64)
        )

    def forward_once(self, x):
        # B is batch size # x shape is like: torch.Size([B, 3, 512, 512])
        output = self.cnn_block1(x) 
        output = self.cnn_block2(output)
        output = self.cnn_block3(output) # torch.Size([B, 64, 7, 7])
        output = output.reshape(output.size(0), -1) # torch.Size([B, 3136])
        output = self.fc_block1(output) # torch.Size([B, 1024])
        output = self.fc_block3(output) # torch.Size([B, 64])

        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1,output2
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # input_shape = (512, 512, 3)
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.cnn_block4 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.cnn_block5 = nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.fc_block1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*14*14, 4096),
            nn.ReLU()
        )
        self.fc_block2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc_block3 = nn.Sequential(
            nn.Linear(4096, 128)
        )

     
    def forward_once(self, x):
        # B is batch size
        output = self.cnn_block1(x)
        # print(output.shape) # torch.Size([B, 16, 255, 255])
        output = self.cnn_block2(output)
        # print(output.shape) # torch.Size([B, 32, 126, 126])
        output = self.cnn_block3(output)
        # print(output.shape) # torch.Size([B, 32, 62, 62])
        output = self.cnn_block4(output)
        # print(output.shape) # torch.Size([B, 64, 30, 30])
        output = self.cnn_block5(output)                
        # print(output.shape) # torch.Size([B, 64, 14, 14])
        output = output.reshape(output.size(0), -1)
        # print(output.shape) # torch.Size([B, 12544])
        output = self.fc_block1(output) 
        # print(output.shape) # torch.Size([B, 4096])
        output = self.fc_block2(output)
        # print(output.shape) # torch.Size([B, 4096])
        output = self.fc_block3(output)
        # print(output.shape) # torch.Size([B, 128])

        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # out = self.out(torch.abs(output1-output2))
        # return out.view(out.size())

        return output1,output2

import torch.nn as nn
from torchvision.models import resnet18

class SampleClassifier(nn.Module):
    def __init__(self):
        super(SampleClassifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256* 32* 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        # Extract features by convolutional layers.
        z1 = self.cnn_layer1(x)
        a1 = self.relu(z1)

        z2 = self.cnn_layer2(a1)
        z2_res = a1 + z2
        a2 = self.relu(z2_res)

        z3 = self.cnn_layer3(a2)
        a3 = self.relu(z3)

        z4 = self.cnn_layer4(a3)
        z4_res = a3 + z4
        a4 = self.relu(z4_res)

        z5 = self.cnn_layer5(a4)
        a5 = self.relu(z5)

        z6 = self.cnn_layer6(a5)
        z6_res = a5 + z6
        a6 = self.relu(z6_res)

        xout = a6.flatten(1)
        xout = self.fc_layer(xout)
        return xout

class ResNet18Classifier(nn.Module):
    def __init__(self):
        super(ResNet18Classifier, self).__init__()
        self.resnet18 = resnet18(pretrained=False, num_classes=11)
    
    def forward(self, x):
        out = self.resnet18(x)
        return out

model_mapping = {
    "SampleClassifier": SampleClassifier,
    "ResNet18Classifier": ResNet18Classifier,
    "Residual_Network": Residual_Network
}
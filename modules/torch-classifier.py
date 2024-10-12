import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  

train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=24,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=24,
    shuffle=False,
)


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = Model().cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=0.001)
epochs = 60

for epoch in range(epochs):
    total_loss = 0
    step = len(train_loader)
    min_loss = 100
    for i, (input, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        input = input.cuda()
        target = target.cuda()
        logits = model(input)
        loss = loss_func(logits, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    cur_loss = total_loss/step
    if cur_loss < min_loss:
        min_loss = cur_loss
        torch.save(model.state_dict(), ".cache/models/model.pt")
    
    print(f"EPOCH: {epoch+1} ===> Loss: {cur_loss}")
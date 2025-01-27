import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据集路径
path = ""
train_dataset = datasets.ImageFolder(root=path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 训练
def train_model(num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), '.pth')

if __name__ == '__main__':
    print(f"Total Images: {len(train_dataset)}")
    train_model()

import torch
import torchvision
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms,models
import torch.utils.data


model = models.resnet18(pretrained=True)
device = "cuda:0"

def add_gpu_noise(device, tensor):
    a = torch.randn_like(tensor).to(device)
    return tensor + a

train_data_path = "." # Add correct path here!
model.to(device)
image_transforms = torchvision.transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=image_transforms)
batch_size=32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, loss_fn,  train_loader, val_loader, epochs=20, device='cuda:0'):
    model.to(device)
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ww, target = batch
            ww = ww.to(device)
            ww = add_gpu_noise(device,ww)
            target= target.to(device)
            output = model(ww)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            ww, target = batch
            ww = ww.to(device)
            target= target.to(device)
            output = model(ww)
            correct = torch.eq(torch.max(output, dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        print("Epoch {}, accuracy = {:.2f}".format(epoch, num_correct / num_examples))

train(model,optimizer,criterion,train_data_loader,train_data_loader,epochs=1)

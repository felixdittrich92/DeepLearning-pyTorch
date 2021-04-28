import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import time

class NoisyDataset(Dataset):
    
    def __init__(self, in_path, mode='train', img_size=(180, 180), sigma=30):
        super(NoisyDataset, self).__init__()

        self.mode = mode #train or test
        self.in_path = in_path # ./BSDS300/images
        self.img_size = img_size # (180, 180)


        self.img_dir = os.path.join(in_path, mode)
        self.imgs = os.listdir(self.img_dir)
        self.sigma = sigma

    def __len__(self):
        return len(self.imgs)
    
    def __repr__(self):
        return "Dataset Parameters: mode={}, img_size={}, sigma={}".format(self.mode, self.img_size, self.sigma)
        
    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.imgs[idx])
        clean_img = Image.open(img_path).convert('RGB')
        left = np.random.randint(clean_img.size[0] - self.img_size[0])
        top = np.random.randint(clean_img.size[1] - self.img_size[1])
        # .crop(left, upper, right, lower)
        cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        
        ground_truth = transform(cropped_clean)

        noisy = ground_truth + 2 / 255 * self.sigma * torch.randn(ground_truth.shape)
        
        return noisy, ground_truth
  
def dataset_imshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    
    return h

class DnCNN(pl.LightningModule):
    
    def __init__(self):
        super(DnCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=1)
        self.conv8 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64, 64)
        self.bn2 = nn.BatchNorm2d(64, 64)
        self.bn3 = nn.BatchNorm2d(64, 64)
        self.bn4 = nn.BatchNorm2d(64, 64)
        self.bn5 = nn.BatchNorm2d(64, 64)
        self.bn6 = nn.BatchNorm2d(64, 64)

        self.dataset_dir = "./BSDS300/images/"
    
    def forward(self, x):
        in_data = F.relu(self.conv1(x))
        in_data = F.relu(self.bn1(self.conv2(in_data)))
        in_data = F.relu(self.bn2(self.conv3(in_data)))
        in_data = F.relu(self.bn3(self.conv4(in_data)))
        in_data = F.relu(self.bn4(self.conv5(in_data)))
        in_data = F.relu(self.bn5(self.conv6(in_data)))
        in_data = F.relu(self.bn6(self.conv7(in_data)))
        residual = self.conv8(in_data)
        
        y = residual + x
        
        return y
    
    def train_dataloader(self):
        return DataLoader(NoisyDataset(self.dataset_dir), batch_size=20)
    
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'train_loss': loss}
        return {'loss' : loss, 'log' : tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
denoising_model = DnCNN()

checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/',  
    save_top_k=1,  
    monitor='loss',
    verbose=True)


trainer = pl.Trainer(gpus=1, max_epochs=20, checkpoint_callback=checkpoint_callback)

trainer.fit(denoising_model)

pretrained_model = DnCNN.load_from_checkpoint("./checkpoints/epoch=29.ckpt")

test_set = NoisyDataset("./BSDS300/images", mode='test', img_size=(320, 320))

with torch.no_grad():
    out = pretrained_model(test_set[2][0].unsqueeze(0))


fig, axes = plt.subplots(ncols=2)
dataset_imshow(test_set[2][0], ax=axes[0])
axes[0].set_title('Noisy')
dataset_imshow(out[0], ax=axes[1])
axes[1].set_title('Clean')
print(f'image size is {out[0].shape}.')
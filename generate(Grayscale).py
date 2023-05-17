import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import trange


class AnimeHeadImg_dataset(Dataset):

    def __init__(self, img_paths, labels, transform) -> None:
        super().__init__()
        self.imgs, self.labels, self.transform = img_paths, labels, transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        r_img = self.transform(Image.open(img).convert('RGB')) 
        # @r_img -shape: [1, 64, 64] // should squeeze zero dimension when preview

        return r_img, label
    
    def __len__(self):
        return len(self.imgs)


class Generator(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Generator, self).__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64 * 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, 64, 64, 1)
        return img
    

class Discriminator(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Discriminator, self).__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Linear(64 * 64 * 1, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 64 * 64 * 1)
        x = self.main(x)
        return x
    

PATH = {
    'data': './data/',
    'save': './generated/',
}
TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])
DATALENGTH = len(os.listdir(PATH['data']))
BATCHSIZE = 16

# make step result visualization
def generate_and_save_image(model, epoch, test_input):
    predictions = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for idx in range(predictions.shape[0]):
        plt.subplot(4, 4, idx + 1)
        plt.imshow((predictions[idx] + 1) / 2, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(PATH['save'], 'image_epoch_%d.png'%(epoch)))

waitting_to_train_images = [os.path.join(PATH['data'], image) for image in os.listdir(PATH['data'])]
train_ds = AnimeHeadImg_dataset(waitting_to_train_images, torch.ones(DATALENGTH), TRANSFORM)
train_dl = DataLoader(train_ds, BATCHSIZE, shuffle=True)

# instantiate generator and discriminator
gen, dis = Generator(), Discriminator()

# define loss function
loss_fn = nn.BCELoss()

# define optimizer
optimizer = {
    'g': torch.optim.Adam(gen.parameters(), lr=0.0001),
    'd': torch.optim.Adam(dis.parameters(), lr=0.0001)
}

# define random seed
test_seed = torch.randn(16, 100)

# let's trainning ✌
loss = {
    'd': [],
    'g': [],
}
# trainning epochs
epochs = 120
for epoch in trange(epochs):
    
    # initiate result calculation parameters
    D_epoch_loss, G_epoch_loss, count = 0, 0, len(train_dl)
    
    for step, (img, _) in enumerate(train_dl):
        
        size = img.shape[0]
        random_seed = torch.randn(size, 100)

        # judge real image
        optimizer['d'].zero_grad()
        real_output = dis(img)
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()

        # judge image generated
        generated_img = gen(random_seed)
        fake_output = dis(generated_img.detach())
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()

        disc_loss = d_real_loss + d_fake_loss
        optimizer['d'].step()

        # calculate generater loss
        optimizer['g'].zero_grad()
        fake_output = dis(generated_img)
        gen_loss = loss_fn(fake_output, torch.ones_like(fake_output))
        gen_loss.backward()
        optimizer['g'].step()

        # record
        with torch.no_grad():
            D_epoch_loss += disc_loss
            G_epoch_loss += gen_loss

    with torch.no_grad():
        D_epoch_loss /= count
        G_epoch_loss /= count
        loss['d'].append(D_epoch_loss)
        loss['g'].append(G_epoch_loss)
        generate_and_save_image(gen, epoch, test_seed)

print('♥ train end ♥')

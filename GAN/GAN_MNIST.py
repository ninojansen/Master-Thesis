import torch
from torch.optim.optimizer import Optimizer
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

n_epochs = 100
batch_size_train = 64

g_lr = 0.0001
d_lr = 0.0004

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),

    batch_size=batch_size_train, shuffle=True)


class Generator(nn.Module):
    def __init__(self, ngf=32, z_dim=100):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.fc1 = nn.Linear(z_dim, 7*7*ngf*4, bias=True)
        self.conv1 = nn.Conv2d(ngf*4, ngf*2, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(ngf*2, ngf, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(ngf, 1, 3, 1, padding=1)

    def forward(self, x):
        x = self.fc1(x)

        x = F.leaky_relu(x)

        x = x.view(-1, self.ngf*4, 7, 7)

        x = self.upsample(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x.view(-1, 28, 28)
        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, ndf, 4, 2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, padding=1)

        self.fc1 = nn.Linear(7*7*ndf*2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# with torch.no_grad():
#     generator = Generator()
#     discriminator = Discriminator()
#     discriminator.eval()
#     generator.eval()
#     random_data = torch.rand((1, 1, 100))
#     result = generator(random_data)

#     plt.imshow(result[0][0], cmap="gray")
#     plt.show()

#     pred = discriminator(result)
#     print(pred)


g = Generator()
g.cuda()
d = Discriminator()
d.cuda()
g_opt = optim.Adam(g.parameters(), lr=g_lr)
d_opt = optim.Adam(d.parameters(), lr=d_lr)

criterion = nn.BCELoss().cuda()


def train(epoch):
    g.train()
    d.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        g_opt.zero_grad()
        d_opt.zero_grad()

        noise = torch.rand((data.shape[0], 1, 100)).cuda()
        fake_img = g(noise)

        fake_pred = d(fake_img)
        real_pred = d(data)

        real_labels = torch.ones((data.shape[0], 1)).cuda()
        fake_labels = torch.zeros((data.shape[0], 1)).cuda()

        d_loss_real = criterion(real_pred, real_labels)
        d_loss_fake = criterion(fake_pred, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_opt.step()

        g_loss = criterion(fake_pred, real_labels)
        g_loss.backward()
        g_opt.step()


for epoch in range(5):
    print(f"Epoch {epoch}")
    train(epoch)

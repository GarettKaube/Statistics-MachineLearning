import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.fc1 = nn.Linear(self.latent_dim,384)

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=192,  kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2)
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2)
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2)
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, z, y):
        z = z.view(-1,self.latent_dim)
        z = self.fc1(z)
        z = z.view(-1,384,1,1)
        z = self.tconv1(z)
        z = self.tconv2(z)
        z = self.tconv3(z)
        z = self.tconv4(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
            )
        self.fc1 = nn.Linear(4*4*512,1)
        self.fc2 = nn.Linear(4*4*512,10)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1,4*4*512)
        dis = self.sig(self.fc1(x))
        aux = self.soft(self.fc2(x))
        return dis, aux
        

# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 256
workers = 6
latent_dim = 128
lr = 0.001
num_epochs = 300
validate_every = 1
print_every = 100


if not os.path.exists(os.path.join(os.path.curdir, "./visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "./visualize", "gan"))

ckpt_path = 'acgan.pt'
save_path = './visualise'

# -----
# Dataset
tfms = transforms.Compose([
    transforms.ToTensor(), 
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)


# -----
# Model

generator = Generator(latent_dim, 3)

discriminator = Discriminator()

# -----
# Losses


adv_loss = nn.BCELoss()
aux_loss = nn.NLLLoss()


if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

generator = generator.to(device)
discriminator = discriminator.to(device)
# Optimizers for Discriminator and Generator, separate

beta1 = 0.5
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr,betas=(beta1, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=lr,betas=(beta1, 0.999))


# -----
# Train loop

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """
        
    minimum = torch.min(x)
    maximum = torch.max(x)
    
    x = x*((255)/(maximum - minimum)) + 0
    x = x.cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1)) 
    return x.astype(np.uint8)


# For visualization part
# Generate 20 random sample for visualization


random_z = torch.FloatTensor(20, latent_dim, 1, 1)
random_z = Variable(random_z)
random_z.data.resize_(20, latent_dim, 1, 1).normal_(0, 1)
random_y = np.random.randint(0, 10, 20)
random_z_ = np.random.normal(0, 1, (20, latent_dim))
random_y_oh = np.zeros((20, 10))
random_y_oh[np.arange(20), random_y] = 1
random_z_[np.arange(20), :10] = random_y_oh[np.arange(20)]
random_z_ = (torch.from_numpy(random_z_))
random_z.data.copy_(random_z_.view(20, latent_dim, 1, 1))
random_z = random_z.to(device)


def train_step(x, y):
    """One train step for AC-GAN.
    You should return loss_g, loss_d, acc_d, a.k.a:
        - average train loss over batch for generator
        - average train loss over batch for discriminator
        - auxiliary train accuracy over batch
    """
    

    real_label = 1
    fake_label = 0

    optimizer_D.zero_grad()
    
    #optimize discriminator
    real_lab = torch.full(size = (x.shape[0],1), fill_value = real_label, dtype=torch.float16)
    if torch.cuda.is_available():
        real_lab = real_lab.cuda()
        
    dis_output, aux_output = discriminator(x)
    dis_loss = adv_loss(dis_output.float(), real_lab.float())
    aux_real_loss = aux_loss(aux_output,y)
    total_dis_real_loss = aux_real_loss + dis_loss
    total_dis_real_loss.backward()
    
    # calculate accuracy
    aux_pred = aux_output.max(1)[1]
    correct = torch.sum(aux_pred == y)
    aux_acc = (float(correct) / float(len(x)))
    

    noise = torch.FloatTensor(x.shape[0], latent_dim, 1, 1)
    noise = Variable(noise)
    noise.data.resize_(x.shape[0], latent_dim, 1, 1).normal_(0, 1)
    label = np.random.randint(0, 10, x.shape[0])
    noisenp = np.random.normal(0, 1, (x.shape[0], latent_dim))
    class_onehot = np.zeros((x.shape[0], 10))
    class_onehot[np.arange(x.shape[0]), label] = 1
    noisenp[np.arange(x.shape[0]), :10] = class_onehot[np.arange(x.shape[0])]
    noise_and_labels = (torch.from_numpy(noisenp))
    noise.data.copy_(noise_and_labels.view(x.shape[0], latent_dim, 1, 1))
    noise =noise.to(device)

    # fake output
    fake_gen_output = generator(noise, None)

    # discriminator output from fake data
    dis_output, aux_output = discriminator(fake_gen_output.detach())
    fake_lab = torch.full(size = (x.shape[0],1), fill_value = fake_label, dtype=torch.float16).to(device)

    # generator  
    dis_error_fake = adv_loss(dis_output.to(device), fake_lab.float())
    aux_error_fake = aux_loss(aux_output, y)
    
    total_dis_fake_loss = dis_error_fake + aux_error_fake
    total_dis_fake_loss.backward()

    dis_loss = total_dis_real_loss + total_dis_fake_loss

    optimizer_D.step()
    
    # optimise generator
    optimizer_G.zero_grad()
    real_lab = torch.full((x.shape[0], 1), real_label).to(device)
    dis_output, aux_output = discriminator(fake_gen_output)
    dis_error_fake_G = adv_loss(dis_output.to(device), real_lab.float())
    aux_error_fake_G = aux_loss(aux_output, y)
    generator_loss = dis_error_fake_G + aux_error_fake_G
    generator_loss.backward()

    optimizer_G.step()

    return generator_loss.detach().cpu().numpy()/x.shape[0], dis_loss.detach().cpu().numpy()/ x.shape[0], aux_acc

def test(
    test_loader,
    ):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    discriminator.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            # Forward only
            _, outputs = discriminator(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g * x.shape[0]
        avg_loss_d += loss_d * x.shape[0]

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_g, loss_d, acc_d))

    g_losses.append(avg_loss_g / len(train_dataset))
    d_losses.append(avg_loss_d / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


        # Do some reconstruction
        generator.eval()
        with torch.no_grad():
            # Forward
            
            xg = generator(random_z.detach(), None)
            xg = denormalize(xg)
            

            # Plot 20 randomly generated images
            plt.figure(figsize=(10, 5))
            for p in range(20):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')

        # Plot losses
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([1, epoch])
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

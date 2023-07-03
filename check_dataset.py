import torch
import torchvision
from model import SiameseNetwork
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import RePAIRFragments
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import pdb, os
import numpy as np
from torch.autograd import Variable

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    #plt.show()

def show(img,ax,dist):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    ax.set_title(f"dist: {dist.item():.06f}", fontweight = "bold", size = 24)
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    #path = '/media/lucap/big_data/datasets/ATAT/'
    path = '/media/lucap/big_data/datasets/repair/triplets/siamese_repair/data/'
    
    # net = SiameseNetwork()
    # net.load_state_dict(torch.load(os.path.join(path,'cp_19_5_img512_bs32_epochs50.pth')))
    # net.eval().cuda()

    bs = 4
    lr = 1e-3
    threshold = 0.3
    margin = 1.5
    epochs = 40
    img_size = 512
    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                              std=(0.229, 0.224, 0.225))
        ])
    test_ds = RePAIRFragments(ImageFolder(root = path + 'train',transform=test_transforms))
    vis_dataloader = DataLoader(test_ds,
                        shuffle=True,
                        num_workers=8,
                        batch_size=bs)

    dataiter = iter(vis_dataloader)


    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))#, example_batch[2].numpy())
    print(example_batch[2].numpy())
    plt.show()
    pdb.set_trace()

if __name__ == '__main__':
    main()
import torch
import torchvision
from model import SiameseNetwork, MiniSiameseNetwork
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import RePAIRFragments
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import pdb, os
import numpy as np
from torch.autograd import Variable
import sys
import yaml 
import argparse 

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

def show(img, ax, dist, label):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    ax.set_title(f"label={np.round(label).astype(int)}\ndist: {dist.item():.06f}", fontweight = "bold", size = 24)
    ax.set_xticks([])
    ax.set_yticks([])

def main(args):

    with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    print(f'Initializing the {cfg["model"]} model from configuration file..')
    ## Initialize network
    if cfg['model'] == 'mini':
        net = MiniSiameseNetwork()
    else:
        net = SiameseNetwork()
    net.load_state_dict(torch.load(args.weights))
    net.eval().cuda()


    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg['img_size']),
            torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                              std=(0.229, 0.224, 0.225))
        ])
    test_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + args.on, transform=test_transforms))
    test_dl = DataLoader(test_ds, batch_size=1) #cfg['bs'])
    dataiter = iter(test_dl)
    x0,_,_ = next(dataiter)
    #fig,ax = plt.subplots(3, 5, figsize = (30,30))


    fig,ax = plt.subplots(3, 5, figsize = (30,30))
    #fig.suptitle(title, fontsize=32)
    for j in range(3):
        _, _, _ = next(dataiter)

    for i in range(15):
        for j in range(1):
            _, _, _ = next(dataiter)
        _, x1, label = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        grid = torchvision.utils.make_grid(concatenated)
        show(grid, ax[i//5, i%5], euclidean_distance.cpu().detach().numpy(), label.item())

    plt.tight_layout()
    plt.show()
    pdb.set_trace()


    for i in range(15):
        for j in range(1):
            _, _, _ = next(dataiter)
        _, x1, label = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        #pdb.set_trace()
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        grid = torchvision.utils.make_grid(concatenated)
        # print(i%5, i//5)
        show(grid, ax[i//5, i%5], euclidean_distance.cpu().detach().numpy(), label.item())
        # imshow(t,'Dissimilarity: {:.6f}'.format(euclidean_distance.item()))

    plt.show()
    pdb.set_trace()
    # fig,ax = plt.subplots(5,1,figsize = (30,30))



    # for j,(input1,input2,target) in enumerate(test_dl):
    
    #     idx = np.random.randint(0,len(test_ds))
    #     #pdb.set_trace()
    #     output1,output2 = net(input1.cuda(),input2.cuda())
        
    #     for i in range(input1.shape[0]):
    #         grid = torchvision.utils.make_grid([input1[i,:,:,:],input2[i,:,:,:]])
    #         euclidean_distance = F.pairwise_distance(output1,output2)
        
    #         show(grid,ax[i],euclidean_distance.cpu().detach().numpy())

    #     plt.show()
    #     pdb.set_trace()
    #     if i > 1:
    #         break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a siamese network')
    parser.add_argument('-cfg', type=str, default='', help='config file with the parameters')
    parser.add_argument('-weights', type=str, default='', help='weights to continue the training')
    parser.add_argument('-on', type=str, default='test', help='train, val, or test - show results on that set (default test)')
    args = parser.parse_args()
    main(args)
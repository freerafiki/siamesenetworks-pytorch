import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from utils import *
from model import SiameseNetwork, MiniSiameseNetwork
from data_augmentation import get_data_augmentation_transforms, get_preproc_transforms
from loss import ContrastiveLoss
from dataset import RePAIRFragments
from torchsummary import summary
from matplotlib import pyplot as plt
import os 
import pdb 
from datetime import datetime
import yaml
import argparse 
import torchvision.transforms as T

def main(args):
    with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    print(f'Initializing the {cfg["model"]} model from configuration file..')
    ## Initialize network
    if cfg['model'] == 'mini':
        model = MiniSiameseNetwork()
    elif cfg['model'] == 'default':
        model = SiameseNetwork()
    else:
        model = SiameseNetwork()

    train_until_epoch = cfg['epochs']
    last_epoch_trained = 0
    if args.weights != '':
        print('Continuing training from checkpoint', args.weights)
        model.load_state_dict(torch.load(args.weights))
        last_epoch_trained = int(args.weights[args.weights.index('epochs')+6:args.weights.index('epochs')+11])
        print(f'Continuing from epoch {last_epoch_trained}')
        train_until_epoch += last_epoch_trained

    date_dm = f"{datetime.now().day}_{datetime.now().month:02d}"
    dset_name = cfg['dataset_root'].split('/')[-2]
    mdl_name = f"{cfg['model']}_ds_{dset_name}_trained_on{date_dm}_imgsize{cfg['img_size']}_daug{cfg['data_augmentation']}_bs{cfg['bs']}"

    if args.out != '':
        output_dir = args.out
    else:
        output_dir = os.path.join(cfg['ckp_dir'], f"run_{mdl_name}")
    os.makedirs(output_dir, exist_ok=True)

    # model
    model = model.cuda()
    print(summary(model,[(3, cfg['img_size'], cfg['img_size']), (3, cfg['img_size'], cfg['img_size'])], batch_size = cfg['bs']))

    ## Initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    ## Initialize loss
    criterion = ContrastiveLoss(cfg['margin'])

    # mean and variance 
    cfg['mean']  = torch.Tensor(np.loadtxt(os.path.join(cfg['dataset_root'], cfg['mean'])))
    cfg['std'] = torch.Tensor(np.loadtxt(os.path.join(cfg['dataset_root'], cfg['std'])))
    
    train_transforms = get_data_augmentation_transforms(cfg)
    basic_transforms = get_preproc_transforms(cfg)

    train_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + 'train', transform=train_transforms, target_transform = basic_transforms))
    valid_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + 'val', transform=basic_transforms, target_transform = basic_transforms))

    train_dl = DataLoader(train_ds, batch_size=cfg['bs'])
    valid_dl = DataLoader(valid_ds, batch_size=cfg['bs'])
    
    print('\nStarting the training..\n')
    train_loss = []
    valid_loss = []
    for epoch in range(last_epoch_trained, train_until_epoch):
        train_epoch_loss = 0
        model.train()
        print()
        for i,(input1,input2,target) in enumerate(train_dl):
            
            optim.zero_grad()
            output1,output2 = model(input1.cuda(),input2.cuda())
            loss = criterion(output1,output2,torch.squeeze(target, axis=1).cuda())
            train_epoch_loss += loss.item()
            print(f"# Epoch {epoch+1} / batch {i+1}: batch average loss {(loss.item()/cfg['bs']):.03f}".ljust(55) + f"{i+1}/{len(train_dl)} = {(i+1)/len(train_dl)*100:.02f}% completed", end='\r')
            loss.backward()
            optim.step()

        print()
        train_epoch_loss /= len(train_ds)
        train_loss.append(train_epoch_loss)
        
        print("# Finished epoch [{}/{}]".format(epoch+1, train_until_epoch))
        print("# Training\nTraining loss : {:.6f}".format(train_epoch_loss))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir,f"partial_ckp_{mdl_name}_epochs{(epoch+1):05d}.pth"))

        valid_epoch_loss = 0
        val_pos_accuracy = 0
        val_neg_accuracy = 0
        num_pos = 0
        num_neg = 0
        model.eval()

        for i,(input1,input2,target) in enumerate(valid_dl):
            output1,output2 = model(input1.cuda(),input2.cuda())
            loss = criterion(output1,output2,target.cuda())
            valid_epoch_loss += loss.item()
            pos_acc,pos_sum,neg_acc,neg_sum = evaluate_pair(output1, output2, target.cuda(), cfg['threshold'])
            val_pos_accuracy+=pos_acc
            val_neg_accuracy+=neg_acc
            num_pos+=pos_sum
            num_neg+=neg_sum

        valid_epoch_loss /= len(valid_ds)
        val_pos_accuracy /= num_pos
        val_neg_accuracy /= num_neg

        valid_loss.append(valid_epoch_loss)
        print("# Validation \nValidation loss :{:.06f} \t\t\t Acc (same): {:.03f}, Acc (diff): {:.03f} (threshold={})\n".format(valid_epoch_loss,val_pos_accuracy,val_neg_accuracy, cfg['threshold']))

    torch.save(model.state_dict(), os.path.join(output_dir,f"ckp_{mdl_name}_epochs{train_until_epoch:05d}.pth"))

    plt.figure(figsize = (10,5))
    plt.plot(train_loss,label = 'train')
    plt.plot(valid_loss,label = 'valid')
    plt.xlabel("Epochs",size = 20)
    plt.ylabel("Loss", size = 20)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"ckp_{mdl_name}_epochs{train_until_epoch:05d}.jpg"))
    print(f'\nFinished training {cfg["model"]} until {train_until_epoch} epochs, saved in {output_dir} checkpoints and plot of the losses.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a siamese network')
    parser.add_argument('-cfg', type=str, default='', help='config file with the parameters')
    parser.add_argument('-weights', type=str, default='', help='weights to continue the training')
    parser.add_argument('-out', type=str, default='', help='output folder')
    args = parser.parse_args()
    main(args)
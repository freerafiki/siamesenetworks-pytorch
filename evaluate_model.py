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
import sys, json
import yaml 
import argparse 
from data_augmentation import get_preproc_transforms

def save_visual(dataiter, net, title, image_path):

    fig,ax = plt.subplots(3, 5, figsize = (30,30))
    fig.suptitle(title, fontsize=32)
    
    for i in range(15):
        for j in range(1):
            _, _, _ = next(dataiter)
        x0, x1, label = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        grid = torchvision.utils.make_grid(concatenated)
        show(grid, ax[i//5, i%5], euclidean_distance.cpu().detach().numpy(), label.item())

    plt.tight_layout()
    plt.savefig(image_path)

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

    dset_name = cfg['dataset_root'].split('/')[-2]
    epochs_weights = int(args.weights[args.weights.index('epochs')+6:args.weights.index('epochs')+11])
    if args.out == '':
        output_dir = f"results/{cfg['model']}_trained_on{dset_name}_for{epochs_weights}epochs"
    else:
        output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on training set
    thr = cfg['threshold']
    correct = 0
    acc_same_class = 0
    acc_diff_class = 0
    num_prediction = 0
    num_prediction_same_class = 0
    num_prediction_diff_class = 0
    preproc_transforms = get_preproc_transforms(cfg)

    print("#" * 30)
    print(f"Evaluation {cfg['model']} model")
    print("#" * 30)

    results = {}

    #### 
    
    train_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + 'train', transform=preproc_transforms))
    train_dl = DataLoader(train_ds, batch_size=1)
    #print("Estimating distances for training set..")
    for x0, x1, label in iter(train_dl):
        
        output1, output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(f"# {num_prediction+1}/{len(train_dl)} = {(num_prediction+1)/len(train_dl)*100:.02f}% completed", end='\r')
        if label == 1:
            num_prediction_diff_class += 1
            if euclidean_distance > thr:
                acc_diff_class += 1
                correct += 1
        else: # if label == 0:
            num_prediction_same_class += 1
            if euclidean_distance < thr:
                acc_same_class += 1
                correct += 1
        num_prediction += 1
    
    print()

    print("#" * 30)
    print('# Evaluation on the training set')
    train_acc = correct/num_prediction*100
    results['train_acc'] = train_acc
    train_acc_same = acc_same_class/num_prediction_same_class*100
    results['train_acc_same'] = train_acc_same
    train_acc_diff = acc_diff_class/num_prediction_diff_class*100
    results['train_acc_diff'] = train_acc_diff
    print(f"Accuracy  : {(train_acc):02.02f}% ({correct:04d} of {num_prediction:04d})")
    print(f"Same class: {(train_acc_same):.02f}% ({acc_same_class:04d} of {num_prediction_same_class:04d})")
    print(f"Diff class: {(train_acc_diff):.02f}% ({acc_diff_class:04d} of {num_prediction_diff_class:04d})")
    print("#" * 30)

    #### 
    correct = 0
    acc_same_class = 0
    acc_diff_class = 0
    num_prediction = 0
    num_prediction_same_class = 0
    num_prediction_diff_class = 0
    val_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + 'val', transform=preproc_transforms))
    val_dl = DataLoader(val_ds, batch_size=1)
    #print("Estimating distances for validation set..")
    for x0, x1, label in iter(val_dl):
        print(f"# {num_prediction+1}/{len(val_dl)} = {(num_prediction+1)/len(val_dl)*100:.02f}% completed", end='\r')
        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        if label == 1:
            num_prediction_diff_class += 1
            if euclidean_distance > thr:
                acc_diff_class += 1
                correct += 1
        else: # if label == 0:
            num_prediction_same_class += 1
            if euclidean_distance < thr:
                acc_same_class += 1
                correct += 1
        num_prediction += 1
    
    print()
    print("#" * 30)
    print('# Evaluation on the validation set')
    val_acc = correct/num_prediction*100
    results['val_acc'] = val_acc
    val_acc_same = acc_same_class/num_prediction_same_class*100
    results['val_acc_same'] = val_acc_same
    val_acc_diff = acc_diff_class/num_prediction_diff_class*100
    results['val_acc_diff'] = val_acc_diff
    print(f"Accuracy  : {(val_acc):02.02f}% ({correct:04d} of {num_prediction:04d})")
    print(f"Same class: {(val_acc_same):.02f}% ({acc_same_class:04d} of {num_prediction_same_class:04d})")
    print(f"Diff class: {(val_acc_diff):.02f}% ({acc_diff_class:04d} of {num_prediction_diff_class:04d})")
    print("#" * 30)

    #### 
    correct = 0
    acc_same_class = 0
    acc_diff_class = 0
    num_prediction = 0
    num_prediction_same_class = 0
    num_prediction_diff_class = 0
    test_ds = RePAIRFragments(ImageFolder(root = cfg['dataset_root'] + 'test', transform=preproc_transforms))
    test_dl = DataLoader(test_ds, batch_size=1)
    #print("Estimating distances for test set..")
    for x0, x1, label in iter(test_dl):
        print(f"# {num_prediction+1}/{len(test_dl)} = {(num_prediction+1)/len(test_dl)*100:.02f}% completed", end='\r')
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        if label == 1:
            num_prediction_diff_class += 1
            if euclidean_distance > thr:
                acc_diff_class += 1
                correct += 1
        else: # if label == 0:
            num_prediction_same_class += 1
            if euclidean_distance < thr:
                acc_same_class += 1
                correct += 1
        num_prediction += 1
    
    print()
    print("#" * 30)
    print('# Evaluation on the test set')
    test_acc = correct/num_prediction*100
    results['test_acc'] = test_acc
    test_acc_same = acc_same_class/num_prediction_same_class*100
    results['test_acc_same'] = test_acc_same
    test_acc_diff = acc_diff_class/num_prediction_diff_class*100
    results['test_acc_diff'] = test_acc_diff
    print(f"Accuracy  : {(test_acc):02.02f}% ({correct:04d} of {num_prediction:04d})")
    print(f"Same class: {(test_acc_same):.02f}% ({acc_same_class:04d} of {num_prediction_same_class:04d})")
    print(f"Diff class: {(test_acc_diff):.02f}% ({acc_diff_class:04d} of {num_prediction_diff_class:04d})")
    print("#" * 30)

    # Saving results on .json file
    with open(os.path.join(output_dir, f"results_{cfg['model']}_{dset_name}_{epochs_weights}.json"), 'w') as rj:
        json.dump(results, rj, indent=3)

    # printing for the markdown benchmark 
    print("| Model Name | Dataset | Epochs | DA | TestAcc | Test (same) | Test (diff) | ValAcc | Val (same) | Val (diff) | TrainAcc | Train (same) | Train (diff) |")
    print(f"| {cfg['model']} | {dset_name} | {epochs_weights} | {cfg['data_augmentation']} | {test_acc:.02f} | {test_acc_same:.02f} | {test_acc_diff:.02f} | {val_acc:.02f} | {val_acc_same:.02f} | {val_acc_diff:.02f} | {train_acc:.02f} | {train_acc_same:.02f} | {train_acc_diff:.02f} |")

    # saving visual results as images
    if args.savefig:
        print("Creating images..")
        print("#" * 30)
        print("On training set..")
        train_dataiter = iter(train_dl)
        train_fig_path = os.path.join(output_dir, f"visual_results_training_set_{cfg['model']}_{dset_name}_{epochs_weights}")
        train_fig_title = f"{cfg['model']} on training set"
        save_visual(train_dataiter, net, train_fig_title, train_fig_path)

        print("#" * 30)
        print("On validation set..")
        val_dataiter = iter(val_dl)
        val_fig_path = os.path.join(output_dir, f"visual_results_validation_set_{cfg['model']}_{dset_name}_{epochs_weights}")
        val_fig_title = f"{cfg['model']} on validation set"
        save_visual(val_dataiter, net, val_fig_title, val_fig_path)

        print("#" * 30)
        print("On test set..")
        test_dataiter = iter(test_dl)
        test_fig_path = os.path.join(output_dir, f"visual_results_test_set_{cfg['model']}_{dset_name}_{epochs_weights}")
        test_fig_title = f"{cfg['model']} on test set"
        save_visual(test_dataiter, net, test_fig_title, test_fig_path)
        print("#" * 30)

    #pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a siamese network')
    parser.add_argument('-cfg', type=str, default='', help='config file with the parameters')
    parser.add_argument('-weights', type=str, default='', help='weights to continue the training')
    parser.add_argument('-out', type=str, default='', help='output folder')
    parser.add_argument('--savefig', action="store_true", default=False, help='save a folder with predictions')
    args = parser.parse_args()
    main(args)
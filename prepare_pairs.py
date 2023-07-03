import os
import shutil
import numpy as np
import pdb 
import torchvision 
import torch 
import torch.nn.functional as F
from utils import read_PIL_image
from model import SiameseNetwork, MiniSiameseNetwork
import argparse 
import yaml


def generate_sim_matrix(train_classes, train_folder, cfg):

    # load the pretrained model 
    with open(cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    
    output_dir = os.path.join('results', f'recognition_{cfg["model"]}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Initializing the {cfg["model"]} model from configuration file..')
    ## Initialize network
    if cfg['model'] == 'mini':
        net = MiniSiameseNetwork()
    else:
        net = SiameseNetwork()
    net.load_state_dict(torch.load(args.weights))
    net.eval().cuda()

    preproc_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg['img_size']),
            torchvision.transforms.ToTensor()
    ])

    # read images
    dist_matrix = np.zeros((len(train_classes), len(train_classes)))
    top_pose_img = {}
    print("we have", len(train_classes), "fragments")

    for train_class in train_classes:
        folder_path = os.path.join(train_folder, train_class)
        imgs_paths = os.listdir(folder_path)
        #pdb.set_trace()
        top_view = [fff for fff in imgs_paths if 'cam20' in fff]
        img_path = os.path.join(folder_path, top_view[0])
        img = read_PIL_image(img_path)
        x = preproc_transforms(img).unsqueeze(0)
        top_pose_img[train_class] = x.cuda()
        print(train_class, ":", top_view[0])

    # compute similarity using pretrained model
    for i, class_i in enumerate(train_classes):
        for j, class_j in enumerate(train_classes):
            if not i == j:
                feats1, feats2 = net(top_pose_img[class_i], top_pose_img[class_j])
                euclidean_distance = F.pairwise_distance(feats1, feats2)
                dist_matrix[i, j] = euclidean_distance.item()
            print(f"dist_matrix[{i}, {j}] = {dist_matrix[i, j]}", end="\r")

    return dist_matrix

def split_triplets_difficulty(train_classes, dist_matrix):

    max_val = np.max(dist_matrix)
    min_val = np.min(dist_matrix[dist_matrix>0])
    # small distances are hard references!
    hard_threshold = min_val + (max_val - min_val) / 4
    semihard_threshold = min_val + (max_val - min_val) / 2
    triplets_diff = {}
    for i, train_class_source in enumerate(train_classes):
        easy = []
        semihard = []
        hard = []
        for j, train_class_target in enumerate(train_classes):
            if dist_matrix[i, j] < semihard_threshold:
                if dist_matrix[i, j] < hard_threshold:
                    hard.append(train_class_target)
                else:
                    semihard.append(train_class_target)
            else:
                easy.append(train_class_target)
        triplets_diff[train_class_source] = {
            'easy': easy,
            'semihard': semihard,
            'hard': hard
        }
        #print(triplets_diff[train_class_source])
    return triplets_diff

def get_pairs(triplets_info, pairs_cfg, dset_path):

    pairs_list = []
    pdb.set_trace()
    for obj_key in triplets_info.keys():
        triplet = triplets_info[obj_key]
        object_folder = os.path.join(dset_path, obj_key)
        # imgs_paths = np.sort(os.listdir(object_folder))
        # append positive pairs
        pairs_list_nums = []
        for j in range(pairs_cfg['pos_pairs_per_objects']):
            pairs_list.append((object_folder, object_folder, 0))
            # pair = np.random.choice(pairs_cfg['pos_pairs_per_objects'], 2, replace=False)
            # if pair not in pairs_list_nums:
            #     pairs_list_nums.append(pair)
            #     pairs_list.append((imgs_paths[pair[0]], imgs_paths[pair[1]]))

        # append negative pairs
        # pairs_list_nums = []
        pdb.set_trace()
        neg_num = pairs_cfg['neg_pairs_per_objects']
        hard_k = pairs_cfg['perc_hard_triplets'] * neg_num
        semi_h_k = pairs_cfg['perc_semihard_triplets'] * neg_num
        easy_k = pairs_cfg['perc_easy_triplets'] * neg_num
        for k in range(neg_num):
            if k < easy_k:
                easy_choice = np.random.choice(triplet['easy'])
                pairs_list_nums.append((object_folder, easy_choice, 1))
            elif k < semi_h_k:
                semihard_choice = np.random.choice(triplet['semihard'])
                pairs_list_nums.append((object_folder, semihard_choice, 1))
            else: # k > hard_k
                hard_choice = np.random.choice(triplet['hard'])
                pairs_list_nums.append((object_folder, hard_choice, 1))
    return pairs_list

def main(args):

    # load the pretrained model 
    with open(args.pairs_cfg, 'r') as yaml_file:
        pairs_cfg = yaml.safe_load(yaml_file)

    dset_path = os.path.join(pairs_cfg['dset_root_path'], pairs_cfg['dset_v']) 

    train_folder = os.path.join(dset_path, 'train')
    val_folder = os.path.join(dset_path, 'val')
    test_folder = os.path.join(dset_path, 'test')
    
    # each object will have a folder in the training set
    train_objects = np.sort(os.listdir(train_folder))

    # calculate distance matrix 
    dist_matrix_path = os.path.join(dset_path, 'dist_matrix.txt')
    if os.path.exists(dist_matrix_path):
        print("reading the matrix")
        dist_matrix = np.loadtxt(dist_matrix_path)
    else:
        print("calculating the matrix")
        dist_matrix = generate_sim_matrix(train_objects, train_folder, args.net_cfg)
        np.savetxt(dist_matrix_path, dist_matrix)

    # and have a list for each fragment which are the easy, semi-hard and hard ones
    triplets_info = split_triplets_difficulty(train_objects, dist_matrix)
    

    
    # perc_hard_triplets = pairs_cfg['perc_hard_triplets'] 
    # perc_semihard_triplets = pairs_cfg['perc_semihard_triplets']
    # perc_easy_triplets = pairs_cfg['perc_easy_triplets']
    # assert(np.isclose(perc_easy_triplets + perc_semihard_triplets + perc_hard_triplets, 1.0)), 'please assign proper percentage, they should sum up to 1'
    # neg_triplets_cfg = [perc_easy_triplets, perc_semihard_triplets, perc_hard_triplets]

    train_pairs = []
    val_pairs = []
    test_pairs = []

    #for train_object in train_objects:
    pdb.set_trace()
    pairs_list = get_pairs(triplets_info, pairs_cfg, dset_path)
    pdb.set_trace()
    pairs_path = os.path.join(dset_path, 'pairs.txt')
    np.savetxt(pairs_path, pairs_list)
        #neg_pairs = get_pairs(triplets_info, pairs_cfg, 'negative')

    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare pairs for the siamese network')
    parser.add_argument('-net_cfg', type=str, default='', help='config file with the parameters of the network')
    parser.add_argument('-weights', type=str, default='', help='trained weights to create embeddings')
    parser.add_argument('-pairs_cfg', type=str, default='', help='config file with the parameters for the pairs')
    args = parser.parse_args()
    main(args)
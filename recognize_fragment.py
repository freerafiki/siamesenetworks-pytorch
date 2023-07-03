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
from PIL import Image
import pandas as pd
import cv2 

def read_PIL_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    return img

def update_probability(sim_mat, prob_vector):

    numerator = np.dot(sim_mat, prob_vector)
    denominator = np.dot(prob_vector.transpose(), numerator)
    updated = np.multiply(prob_vector, np.dot(numerator, 1/denominator))
    return updated, denominator

def initialize_prob_vector(size, init_mode='uniform'):

    if init_mode == 'uniform':
        pv = np.ones((size)) / size
    else:
        pv = np.zeros((size))

    return pv

def main(args):

    # init the network
    with open(args.cfg, 'r') as yaml_file:
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
    feats_frag = {}
    dataset_folder = '/media/lucap/big_data/datasets/repair/cds'
    db_name = 'Decor1_Decor2_singleView'
    images_folder = os.path.join(dataset_folder, 'db', f"{db_name}")
    sources_path = os.path.join(dataset_folder,  'db', f"{db_name}.json")
    ds_type = 'classic'
    query_folder_path = os.path.join(dataset_folder, 'query', db_name, ds_type)
    K = 5
    with open(sources_path, 'r') as jf:
        sources_list = json.load(jf)
    imgs = {}
    feat_frag_paths = [feat_path for feat_path in os.listdir(images_folder)]
    for k, img_folder_path in enumerate(feat_frag_paths):
        print(f'\n{img_folder_path}')        
        img_path = [images for images in os.listdir(os.path.join(images_folder, img_folder_path))][0]
        img = cv2.imread(os.path.join(images_folder, img_folder_path, img_path))
        imgs[img_folder_path] = img
        feats_frag[img_folder_path] = read_PIL_image(os.path.join(images_folder, img_folder_path, img_path))

    acc_query_name = []
    acc_query_group = []
    acc_query_id = []
    acc_position = []
    max_clique_sizes = []
    acc_clique = []
    for query_fragment in sources_list:
    
         # 2) add the "new" image (search)
        query_img_name = query_fragment['img_name']
        cur_out_folder = os.path.join(output_dir, query_img_name)
        os.makedirs(cur_out_folder, exist_ok=True)
        query_image_path = os.path.join(query_folder_path, f"{query_img_name}.png")
        print("\nquery:", query_img_name)
        query_image = read_PIL_image(query_image_path)
        query_image_cv = cv2.imread(query_image_path)


        # nearest neighbours
        likelihoods = []
        for frag_key in feats_frag.keys():
            x_db = preproc_transforms(feats_frag[frag_key]).unsqueeze(0)
            x_q = preproc_transforms(query_image).unsqueeze(0)
            feats1, feats2 = net(x_db.cuda(), x_q.cuda())
            euclidean_distance = F.pairwise_distance(feats1, feats2)
            likelihoods.append(euclidean_distance.item())
            # plt.figure()
            # plt.title(f"{frag_key} vs {query_img_name}")
            # plt.subplot(121)
            # plt.imshow(feats_frag[frag_key])
            # plt.subplot(122)
            # plt.imshow(query_image)
            # pdb.set_trace()
            
        # 6) cluster
        # print("\noutput probabilities")
        #likelihoods = updated_x[:-1]
        max_sim = 100000000000
        best_candidate = ''
        with open(os.path.join(cur_out_folder, f'results_cds_for_{query_img_name}.txt'), 'w') as rw:
            for candidate, prob in zip(feats_frag.keys(), likelihoods):
                #print(candidate, ":", prob)
                candidate_formatted = candidate.ljust(20)
                prob_formatted = f"{prob:.05f}".rjust(12)
                rw.write(f"{candidate_formatted}:{prob_formatted}\n")
                if prob < max_sim:
                    max_sim = prob
                    best_candidate = candidate

        res_df = pd.DataFrame()
        names = []
        rpf_ids = []
        gr_nums = []
        for k in feats_frag.keys():
            g_ind = k.index('Group')+5
            group_num = k[g_ind:g_ind+2]
            id_ind = k.index('RPf')+4
            rpf_id = k[id_ind:id_ind+5]
            gr_nums.append(group_num)
            rpf_ids.append(rpf_id)
            names.append(k)
        # names.append(f"query_{query_img_name}")
        # gr_nums.append(query_fragment['group'])
        # rpf_ids.append(query_fragment['id'])
        res_df['names'] = names
        res_df['group'] = gr_nums
        res_df['id'] = rpf_ids
        res_df['probs'] = likelihoods
        res_df.to_csv(os.path.join(cur_out_folder, f'results_cds_for_{query_img_name}.csv'))
        sorted_res = res_df.sort_values('probs', ascending=True)
        sorted_res.to_csv(os.path.join(cur_out_folder, f'results_cds_for_{query_img_name}_sorted.csv'))
        
        winner_pos = 0
        for j, ll in enumerate(sorted_res['id']): 
            if ll == query_fragment['id']: 
                winner_pos = j+1                
        acc_query_name.append(query_fragment['img_name'])
        acc_query_group.append(query_fragment['group'])
        acc_query_id.append(query_fragment['id'])
        acc_position.append(winner_pos)

        # clique 
        ratio_cut = 1.5
        max_clique_found = False 
        max_clique_size = 0
        ccc = 0
        while not max_clique_found:
            ccc += 1
            if ccc >= len(sorted_res['probs']) - 1:
                max_clique_size = len(sorted_res['probs'])
                max_clique_found = True
            if sorted_res['probs'][ccc-1] / sorted_res['probs'][ccc] > ratio_cut:
                max_clique_size = ccc 
                max_clique_found = True
        
        max_clique_sizes.append(max_clique_size)
        found_within_the_clique = int(winner_pos <= max_clique_size)
        acc_clique.append(found_within_the_clique)

        #pdb.set_trace()
        top_K_res = sorted_res.head(K)
        top_K_res.to_csv(os.path.join(cur_out_folder, f'results_cds_for_{query_img_name}_top_{K}.csv'))
        print(top_K_res)
        plt.figure(figsize=(32,12))
        plt.subplot(2,K,1)
        plt.title(f'Query: {query_img_name}', fontsize=24)
        plt.imshow(cv2.cvtColor(query_image_cv, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        top_k_images_name = top_K_res['names']
        top_k_images_id = top_K_res['id']
        top_k_images_gr = top_K_res['group']
        top_k_probs = top_K_res['probs']
        for kk, (name, k_id, k_gr, prob) in enumerate(zip(top_k_images_name, top_k_images_id, top_k_images_gr, top_k_probs)):
            plt.subplot(2, K, K+1+kk)
            # print(f'{name}: {prob:.05f}')
            plt.title(f'RPf{k_id}_G{k_gr}: {prob:.04f}', fontsize=22)
            plt.imshow(cv2.cvtColor(imgs[name], cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.savefig(os.path.join(cur_out_folder, f'results_{query_img_name}.jpg'))
        # plt.show()
        # pdb.set_trace()
        print('finished, saved in', os.path.join(cur_out_folder))

    accuracy_df = pd.DataFrame()
    accuracy_df['acc_query_name'] = acc_query_name
    accuracy_df['acc_query_group'] = acc_query_group
    accuracy_df['acc_query_id'] = acc_query_id
    accuracy_df['acc_position'] = acc_position
    accuracy_df['max_clique_sizes'] = max_clique_sizes
    accuracy_df['acc_clique'] = acc_clique
    accuracy_df.to_csv(os.path.join(output_dir, f'accuracy_cds_{db_name}.csv'))

    # STATS 
    print("#" * 35)
    print(f"# Features: {cfg['model']}".ljust(34) + "#")
    q_size = len(feats_frag.keys())
    print(f"# DB size: {q_size}".ljust(34) + "#")
    acc_np = np.asarray(acc_position)
    top1 = np.sum(acc_np < 2)
    top5 = np.sum(acc_np < 6)
    avg_pos = np.mean(acc_position)
    top_clique = np.sum(acc_clique)
    print(f"# TOP 1: {top1} / {q_size}".ljust(34) + "#")
    print(f"# TOP 5: {top5} / {q_size}".ljust(34) + "#")
    print(f"# Within the clique: {top_clique} / {q_size}".ljust(34) + "#")
    print(f"# Position (avg): {avg_pos:.02f}".ljust(34) + "#")
    avg_clique_size = np.mean(max_clique_sizes)
    print(f"# Max Clique Size (avg): {avg_clique_size:.02f}".ljust(34) + "#")
    print("#" * 35)
    pdb.set_trace()
    top1_perc = top1/q_size*100
    top5_perc = top5/q_size*100
    top_clique_perc = top_clique/q_size*100
    print(f"| {(cfg['model']).ljust(20)} | {('NN').ljust(4)} | {(ds_type).ljust(7)} | {top1_perc:.01f} % | {top5_perc:.01f} % | {top_clique_perc:.01f} % | {avg_pos:.02f} | {avg_clique_size:.02f} |")
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognition by nearest neighbour in the feature space using siamese network embeddings')
    parser.add_argument('-cfg', type=str, default='', help='config file with the parameters of the network')
    parser.add_argument('-weights', type=str, default='', help='trained weights to create embeddings')
    args = parser.parse_args()
    main(args)
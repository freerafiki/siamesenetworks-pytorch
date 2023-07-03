import torch
import torchvision
from model import SiameseNetwork, MiniSiameseNetwork
import torch.nn.functional as F
from PIL import Image
import pdb, os
import yaml 
import argparse 
import random 

def main(args):

    with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    print(f'Initializing the {cfg["model"]} model from configuration file..')
    ## Initialize network
    if cfg['model'] == 'mini':
        net = MiniSiameseNetwork()
    else:
        net = SiameseNetwork()
    net.load_state_dict(torch.load(os.path.join(cfg['dataset_root'], args.weights)))
    net.eval().cuda()

    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(cfg['img_size']),
            torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                              std=(0.229, 0.224, 0.225))
    ])

    imgs_path = os.listdir(cfg['raw_data_path'])
    img0_path = random.choice(imgs_path)
    img0 = Image.open(os.path.join(cfg['raw_data_path'], img0_path))
    img1_path = random.choice(imgs_path)
    img1 = Image.open(os.path.join(cfg['raw_data_path'], img1_path))
    img0 = img0.convert("RGB")
    img1 = img1.convert("RGB")
    x0 = test_transforms(img0).unsqueeze(0)
    x1 = test_transforms(img1).unsqueeze(0)

    output1,output2 = net(x0.cuda(),x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    print(f"Distance between {img0_path} and {img1_path} is {euclidean_distance.cpu().detach().item():.05f}")
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a siamese network')
    parser.add_argument('-cfg', type=str, default='', help='config file with the parameters')
    parser.add_argument('-weights', type=str, default='', help='weights to continue the training')
    args = parser.parse_args()
    main(args)
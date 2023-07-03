import torch 
import torchvision

def get_data_augmentation_transforms(cfg):
    
    tr_idx = cfg['data_augmentation']
    if tr_idx == 0:
        da_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(cfg['img_size']),
                    torchvision.transforms.ToTensor(),
                ])
    elif tr_idx == 1:

        da_transforms = torchvision.transforms.Compose([ #.nn.Sequential(
            torchvision.transforms.Resize(cfg['img_size']),
            torchvision.transforms.ToTensor(),
            # transform (scale, crop, flip, rotate)
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(180),
            # color-based
            torchvision.transforms.ColorJitter(brightness=0.35, contrast=0.1, saturation=0.08, hue=0.08), 
            #brightness=[0.9,1], contrast=[0.8,1], saturation=[0.7,1], hue=[0.3,0.5]), # we can give [min, max] values for bright, hue, sat..
            #torchvision.transforms.RandomEqualize(p=cfg['p_eq']),
            torchvision.transforms.RandomAutocontrast(p=cfg['p_auto_contrast']),
            torchvision.transforms.RandomGrayscale(p=0.1),
            # normalization
            #torchvision.transforms.Normalize(cfg['mean'], cfg['std'])
        ])

    elif tr_idx == 2:

        print("# WARNING: it requires torchvision >= 0.15 for v2.transforms")

        da_transforms = torchvision.transforms.Compose([ #.nn.Sequential(
            torchvision.transforms.v2.Resize(cfg['img_size']),
            torchvision.transforms.ToTensor(),
            # transform (scale, crop, flip, rotate)
            torchvision.transforms.v2.RandomRotation(180),
            torchvision.transforms.v2.RandomPerspective(),
            # color-based
            torchvision.transforms.v2.RandomPhotometricDistort(),
            # normalization
            #torchvision.transforms.Normalize(cfg['mean'], cfg['std'])
        ])
    
    return da_transforms

def get_preproc_transforms(cfg): 

    trasnf =  torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg['img_size']),
        torchvision.transforms.ToTensor(),
        # normalization
        #torchvision.transforms.Normalize(cfg['mean'], cfg['std'])
    ])
    return trasnf
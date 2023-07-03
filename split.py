import os
import shutil
import numpy as np
import pdb 
import torchvision 
import torch 
from PIL import Image 

path = '/media/lucap/big_data/datasets/repair/triplets/siamese_repair/data/'
dset_root_path = '/media/lucap/big_data/datasets/repair/triplets/siamese_repair/dataset/'
dset_v = 'v3'
dset_path = os.path.join(dset_root_path, dset_v) 
if os.path.exists(dset_path):
    print("\n\nWARNING: dataset path {dset_path} exists already! You really want to continue?\n\n")
    pdb.set_trace()

os.makedirs(dset_path, exist_ok=True)

classes_list = os.listdir(path)
np.random.shuffle(classes_list)

val_split = np.round(.8 * len(classes_list)).astype(int)
test_split = np.round(.9 * len(classes_list)).astype(int)
train,val,test = np.split(classes_list,[val_split,test_split])

train_mean_vals = np.zeros((len(train), 3))
train_std_vals = np.zeros((len(train), 3))

preproc_transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(512),
                        torchvision.transforms.ToTensor(),
                    ])

print(f"{len(train)} classes for training, {len(val)} for validation and {len(test)} for testing")

train_images = 0
val_images = 0
test_images = 0

train_counter = 0
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        folder_full_path = os.path.join(path, folder)
        if folder in train:
            imgs_names = os.listdir(folder_full_path)
            fragm_mean = np.zeros((len(imgs_names), 3))
            fragm_stds = np.zeros((len(imgs_names), 3))
            for j, img_path in enumerate(imgs_names):
                img = Image.open(os.path.join(folder_full_path, img_path))
                img = img.convert("RGB")
                timg = preproc_transform(img)
                for k in range(3):
                    fragm_mean[j, k] = torch.mean(timg[k,:,:])
                    fragm_stds[j, k] = torch.std(timg[k,:,:])
            folder_mean = np.mean(fragm_mean, axis=0)
            folder_std = np.mean(fragm_stds, axis=0)
            train_mean_vals[train_counter, :] = folder_mean
            train_std_vals[train_counter, :] = folder_std
            train_counter += 1
            shutil.copytree(path+folder,dset_path+'/train/'+folder)
            print(f"added {folder} to train")
            train_images += len(os.listdir(folder_full_path))
        elif folder in val:
            print(f"added {folder} to val")
            shutil.copytree(path+folder,dset_path+'/val/'+folder)
            val_images += len(os.listdir(folder_full_path))
        else:
            print(f"added {folder} to test")
            shutil.copytree(path+folder,dset_path+'/test/'+folder)
            test_images += len(os.listdir(folder_full_path))

stats_file = os.path.join(dset_path, 'stats.txt')
f = open(stats_file, "w")
f.write("# STATS")
f.write("\n# TRAINING")
f.write(f"\n{len(train)} classes")
f.write(f"\n{train_images} images")
f.write("\n# VALIDATION")
f.write(f"\n{len(val)} classes")
f.write(f"\n{val_images} images")
f.write("\n# TEST")
f.write(f"\n{len(test)} classes")
f.write(f"\n{test_images} images")
f.write("\n#######")
f.close()

print('saving mean and variance..')
stats_mean_path = os.path.join(dset_path, 'mean_vals.txt')
np.savetxt(stats_mean_path, train_mean_vals)
stats_std_path = os.path.join(dset_path, 'std_vals.txt')
np.savetxt(stats_std_path, train_std_vals)
stats_mean_path = os.path.join(dset_path, 'mean_single_val.txt')
np.savetxt(stats_mean_path, np.mean(train_mean_vals, axis=0))
stats_std_path = os.path.join(dset_path, 'std_single_val.txt')
np.savetxt(stats_std_path, np.mean(train_std_vals, axis=0))

print('\nfinished\n')
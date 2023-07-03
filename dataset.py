import torch
import torchvision
import numpy as np
from PIL import Image
import pdb 
import random 

class RePAIRFragments(torch.utils.data.Dataset):
    # """
    #     A custom class used to sample positive and negative pairs of images with equal probability, the outputs of which are
    #     fed to a contrastive loss function.
    # """
    def __init__(self, repair_dataset):
        ## cub_dataset IS AN ImageFolder DATASET,
        ## THIS FUNCTION MERELY COPIES ITS RELEVANT ATTRIBUTES
        self.classes = repair_dataset.classes
        self.imgs = repair_dataset.imgs
        self.transform = repair_dataset.transform
        if repair_dataset.target_transform:
            self.transform_pair = repair_dataset.target_transform
        else:
            self.transform_pair = repair_dataset.transform
        
        

    def __getitem__(self,index):

        #pdb.set_trace()
        img0_tuple = random.choice(self.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_tuple = random.choice(self.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform_pair(img1)

        #pdb.set_trace()
        #print(img0_tuple[1]==img1_tuple[1], torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)))
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

        # ## CHOOSE EITHER POSITIVE PAIR (0) OR NEGATIVE PAIR (1)
        # self.target = np.random.randint(0,2)
        # ## HERE THE FIRST IMAGE IS CHOSEN BY VIRTUE OF INDEX ITSELF
        # img1,label1 = self.imgs[index]
        # ## CREATE NEW LIST OF IMAGES TO AVOID RE-SELECTING ORIGINAL IMAGE
        # new_imgs = list(set(self.imgs) - set(self.imgs[index]))
        # length = len(new_imgs)
        # # print(length)
        # random = np.random.RandomState(42)
        # if self.target == 1:
        #     ## GET NEGATIVE COUNTERPART
        #     label2 = label1
        #     while label2 == label1:
        #         choice = random.choice(length)
        #         img2,label2 = new_imgs[choice]
        # else:
        #     ## GET POSITIVE COUNTERPART
        #     label2 = label1 + 1
        #     while label2 != label1:
        #         choice = random.choice(length)
        #         img2,label2 = new_imgs[choice]

        # img1 = Image.open(img1).convert('RGB')
        # img2 = Image.open(img2).convert('RGB')
        # if self.transform:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)

        # return (img1,img2,self.target)

    def __len__(self):
        return(len(self.imgs))


class ATATContrast(torch.utils.data.Dataset):
# """
#     A custom class used to sample positive and negative pairs of images with equal probability, the outputs of which are
#     fed to a contrastive loss function.
# """
    def __init__(self,atat_dataset):
        ## cub_dataset IS AN ImageFolder DATASET,
        ## THIS FUNCTION MERELY COPIES ITS RELEVANT ATTRIBUTES
        self.classes = atat_dataset.classes
        self.imgs = atat_dataset.imgs
        self.transform = atat_dataset.transform

    def __getitem__(self,index):
        ## CHOOSE EITHER POSITIVE PAIR (0) OR NEGATIVE PAIR (1)
        self.target = np.random.randint(0,2)
        ## HERE THE FIRST IMAGE IS CHOSEN BY VIRTUE OF INDEX ITSELF
        img1,label1 = self.imgs[index]
        ## CREATE NEW LIST OF IMAGES TO AVOID RE-SELECTING ORIGINAL IMAGE
        new_imgs = list(set(self.imgs) - set(self.imgs[index]))
        length = len(new_imgs)
        # print(length)
        random = np.random.RandomState(42)
        if self.target == 1:
            ## GET NEGATIVE COUNTERPART
            label2 = label1
            while label2 == label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]
        else:
            ## GET POSITIVE COUNTERPART
            label2 = label1 + 1
            while label2 != label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            #img2 = self.transform(img2)

        return (img1,img2,self.target)

    def __len__(self):
        return(len(self.imgs))

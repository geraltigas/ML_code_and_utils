from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from typing import List
def get_all_file_in_dir(dir:str):
    import os
    return [os.path.join(dir,file) for file in os.listdir(dir)]

def get_rgb_mean_and_std(dataset:Dataset|List[Dataset]):
    if isinstance(dataset,Dataset):
        print("a dataset")
        mean = [0, 0, 0]
        std = [0, 0, 0]
        num_imgs = len(dataset)
        for k in range(num_imgs):
            img, tar = dataset[k]
            for i in range(3):
                mean[i] += img[i, :, :].mean()
                std[i] += img[i, :, :].std()
        mean = np.array(mean) / num_imgs
        std = np.array(std) / num_imgs
        return mean, std
    else:
        print("a list of dataset")
        mean = [0, 0, 0]
        std = [0, 0, 0]
        num_imgs = 0
        for dataset_ in dataset:
            mean_, std_ = get_rgb_mean_and_std(dataset_)
            mean = mean*num_imgs + mean_*len(dataset_)
            std = std*num_imgs + std_*len(dataset_)
            num_imgs += len(dataset_)
        return mean, std


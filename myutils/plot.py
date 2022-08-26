import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor
from torchvision import transforms
import matplotlib.pyplot as plt

def show_scale(array:list,label:str = None):
    array.sort()
    fig, ax = plt.subplots()
    x = [i for i in range(len(array))]
    y = np.asarray(array)
    ax.plot(x, y, label=label)
    ax.legend()
    plt.show()

def show_distribution(array:list,value_count_num:int = 100):
    array = np.asarray(array)
    fig, ax = plt.subplots()
    x = [i for i in range(len(array))]
    y = np.asarray(array)
    ax.plot(x, y, label="distribution")
    ax.legend()
    plt.show()

def draw_all(dataframe:pd.DataFrame):
    for i in dataframe.columns:
        if dataframe[i].dtype != object:
            show_scale(list(dataframe[i]),i)

def show_tensor(tensor:Tensor,title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.pause(0.001)


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

def plot_y(y:list[float]|list[list],title:str = None):
    import  numpy as np
    import matplotlib.pyplot as plt
    plt.title(title)
    if isinstance(y[0],float):
        x = np.linspace(0,len(y),len(y))
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y)
        ax.legend()
        plt.show()
    else:
        fig, ax = plt.subplots()
        for y_ in y:
            x = np.linspace(0,len(y_),len(y_))
            line1, = ax.plot(x, y_)
        ax.legend()
        plt.show()
    plt.title(title)
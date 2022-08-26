import pandas as pd
import matplotlib.pyplot as plt
from myutils.plot import show_scale,show_distribution

def show_dataframe_info(dataframe:pd.DataFrame,value_count_nun:int = 100):
    print(dataframe.info(),"\n")
    for i in dataframe.columns:
        print("column {}:\nmax: {},\nmin: {},\nmean: {},\nstd: {}\n".format(i, dataframe[i].max(), dataframe[i].min(),
                                                                    dataframe[i].mean(), dataframe[i].std()))
        if dataframe[i].dtype != object:
            show_scale(list(dataframe[i]),i)
            show_distribution(list(dataframe[i]),value_count_nun)

def show_dataframe_columns(dataframe:pd.DataFrame):
    print(dataframe.columns)
    for i in dataframe.columns:
        print("column {}:\nmax: {},min: {},mean: {},std: {}".format(i,dataframe[i].max(),dataframe[i].min(),dataframe[i].mean(),dataframe[i].std()))
    return dataframe.columns


import pandas as pd
import matplotlib.pyplot as plt
from MyUtils.plot import showScale,showDistribution
def show_dataframe_info(dataframe:pd.DataFrame,value_count_nun:int = 100):
    print(dataframe.info(),"\n")
    for i in dataframe.columns:
        print("column {}:\nmax: {},\nmin: {},\nmean: {},\nstd: {}\n".format(i, dataframe[i].max(), dataframe[i].min(),
                                                                    dataframe[i].mean(), dataframe[i].std()))
        if dataframe[i].dtype != object:
            showScale(list(dataframe[i]),i)
            showDistribution(list(dataframe[i]),value_count_nun)

def show_dataframe_columns(dataframe:pd.DataFrame):
    print(dataframe.columns)
    for i in dataframe.columns:
        print("column {}:\nmax: {},min: {},mean: {},std: {}".format(i,dataframe[i].max(),dataframe[i].min(),dataframe[i].mean(),dataframe[i].std()))
    return dataframe.columns

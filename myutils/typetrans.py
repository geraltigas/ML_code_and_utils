import pandas as pd
def series_to_list(series:pd.Series):
    return list(series)

def dataframe_to_list(dataframe:pd.DataFrame,column:str|int):
    return dataframe[column].tolist()

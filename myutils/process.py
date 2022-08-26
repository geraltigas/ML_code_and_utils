import pandas as pd

def dataframe_column_process(dataframe:pd.DataFrame,column:str|int,process:callable):
    return process(dataframe[column])
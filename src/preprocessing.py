import pandas as pd

def timestamp_parsing(df : pd.DataFrame, col : str = "Time", is_parsing : bool = True):
    df[col] = df[col].apply(lambda x : pd.to_datetime(x))
    
    if is_parsing:
        df["year"] = df[col].apply(lambda x : x.year)
        df["month"] = df[col].apply(lambda x : x.month)
        df["day"] = df[col].apply(lambda x : x.day)
        df["hour"] = df[col].apply(lambda x : x.hour)
        df["minute"] = df[col].apply(lambda x : x.minute)
        df["second"] = df[col].apply(lambda x : x.second)
        
def clean_nan(df : pd.DataFrame):
    if sum(df.isna().sum())==0 and sum(df.isnull().sum())==0:
        pass
    else:
        print("Process for eliminating row data containing NaN / Null values")
        df = df.dropna(axis = 0)
   
"""
    utility code for online learning algorithms
"""

import numpy as np
import pandas as pd
import random 
import torch
import os
from typing import Union, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

def read_file(path : str = "./dataset/data.csv", format = None)->pd.DataFrame:
    if format is None:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, format = format)
    return df

def seed_everything(seed_num : int = 42, use_nn : bool = False)->None:
    
    print("# initialize random seed number")
    random.seed(seed_num)
    np.random.seed(seed_num)
    
     # os environment seed num fixed
    try:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
    except:
        pass
    
    if use_nn:
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        
        # torch seed num fixed
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        except:
            pass
    return

def MSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = True)

def RMSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = False)

def MAE(gt: np.array, pt: np.array):
    return np.mean(np.abs((gt - pt)))

def compute_metrics(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List], algorithm : str):
    
    mse = MSE(gt, pt)
    rmse = RMSE(gt, pt)
    mae = MAE(gt, pt)
    
    print("# {}, mse : {:.3f}, rmse : {:.3f}, mae : {:.3f}".format(algorithm, mse, rmse, mae))
"""
    utility code for online learning algorithms
"""

import numpy as np
import pandas as pd
import random 
import torch
import os
from typing import Union, List

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

def compute_metric(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List]):
    pass
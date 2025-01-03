import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    mse = mean_squared_error(target, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(target, pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
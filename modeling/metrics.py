import numpy as np
from scipy.stats import pearsonr
import torch
from math import sqrt

def forecasting_metrics(loss, preds, labels):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    score_dict = {}
    score_dict['val_loss'] = loss 
    score_dict['MSE'] = loss 
    score_dict['MAPE'] = torch.tensor(mean_absolute_percentage_error(labels, preds)).to(loss.device)
    score_dict['RMSE'] = torch.tensor(root_mean_square_error(loss)).to(loss.device)
    score_dict['sMAPE'] = torch.tensor(sym_mean_absolute_percentage_error(labels, preds)).to(loss.device)
    score_dict['pears'] = torch.tensor(pearsonr(labels, preds)[0])
    
    return score_dict

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((np.subtract(y_true, y_pred)) / y_true)) * 100

def root_mean_square_error(loss):
    return (sqrt(loss.item()))

def sym_mean_absolute_percentage_error(y_true, y_pred):
    # https://robjhyndman.com/hyndsight/smape/
    return np.mean(2*np.abs(np.subtract(y_true,y_pred))/( (np.abs(y_true)+np.abs(y_pred))) ) * 100

def gather_score_dict(outputs, multi_gpu):
    metric_vals = {'val_loss':0, 'MSE':0, 'MAPE':0, 'RMSE':0, 'sMAPE':0, 'pears': 0}
        
    tqdm_dict = {}
    for batch in outputs:
        for metric_name in batch.keys():
            val = batch[metric_name]
            
            # average across each GPU per batch
            if multi_gpu:
                val = torch.mean(val)

            metric_vals[metric_name] += val
            
    for metric_name in metric_vals.keys():
        tqdm_dict[metric_name] = metric_vals[metric_name]/len(outputs) # avg over all batches
    
    try:
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'MSE': tqdm_dict['MSE']}
    except:
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'MSE': tqdm_dict['MSE']}

    return result

from collections.abc import Iterable
import sys
import os
import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr,ttest_ind,mannwhitneyu
from sklearn.metrics import mean_squared_error
def ssim(im1,im2,M=1):
    im1, im2 = im1/im1.max(), im2/im2.max()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

def rmse(x1,x2):
    return mean_squared_error(x1,x2,squared=False)
def mae(x1,x2):
    return np.mean(np.abs(x1-x2))
    
def compare_results(gd, result_df, metric='pcc', columns=None, axis=1):
    if metric == 'pcc':
        func = pearsonr
        r_ind = 0
    elif metric == 'mae':
        func = mae
        r_ind = None
    elif metric == 'jsd':
        func = jensenshannon
        r_ind = None
    elif metric == 'rmse':
        func = rmse
        r_ind = None
    elif metric == 'ssim':
        func = ssim
        r_ind = None
    else:
        raise ValueError('Invalid metric: {}'.format(metric))

    c_list = []
    if axis == 1:
        for i, c in enumerate(gd.columns):
            r = func(gd.iloc[:, i].values, np.clip(result_df.iloc[:, i], 0, 1))
            if isinstance(result_df, Iterable) and r_ind is not None:
                r = r[r_ind]
            c_list.append(r)
        df = pd.DataFrame(c_list, index=gd.columns, columns=columns)
    else:
        for i, c in enumerate(gd.index):
            r = func(gd.iloc[i, :].values, np.clip(result_df.iloc[i, :], 0, 1))
            if isinstance(result_df, Iterable) and r_ind is not None:
                r = r[r_ind]
            c_list.append(r)
        df = pd.DataFrame(c_list, index=gd.index, columns=columns)

    return df
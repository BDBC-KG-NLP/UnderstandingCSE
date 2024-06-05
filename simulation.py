import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from tqdm import tqdm
from typing import Tuple, List, Dict
import matplotlib.font_manager as fm

def roman_font(size: int):
    return fm.FontProperties(family='serif', style='normal', 
        size=size, weight='normal', stretch='normal'
    )

def sample_batch(
    batch_size: int,
    pos_mean: float, neg_mean: float,
    pos_std: float, neg_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    assert batch_size > 1
    
    pos_theta = np.random.normal(loc=pos_mean, scale=pos_std, size=1)
    neg_thetas = np.random.normal(loc=neg_mean, scale=neg_std, size=(batch_size - 1))
    
    return pos_theta, neg_thetas

def sample_multiple_batches(
    num: int, batch_size: int,
    pos_mean: float, neg_mean: float,
    pos_std: float, neg_std: float
)-> List[Tuple[np.ndarray, np.ndarray]]:
    # if neg_mean - pos_mean > 0.25 * np.pi:
    #     pdb.set_trace()
    return [sample_batch(batch_size, pos_mean, neg_mean, pos_std, neg_std) for _ in range(num)]

def cal_gd(
    loss_type: str, batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs: float
) -> float:
    
    def _gd_infonce(batch: Tuple[np.ndarray, np.ndarray], temp: float) -> float:
        # if batch[1].min() - batch[0] > 0.25 * np.pi:
        #     pdb.set_trace()
        return 1 / (1 + (np.exp(np.cos(batch[0]) / temp) \
                         / np.exp(np.cos(batch[1]) / temp).sum()).item())
    
    def _gd_arccon(batch: Tuple[np.ndarray, np.ndarray], temp: float, u: float) -> float:
        return 1 / (1 + (np.exp(np.cos(batch[0] + u) / temp) \
                         / np.exp(np.cos(batch[1]) / temp).sum()).item())

    def _gd_mpt(batch: Tuple[np.ndarray, np.ndarray], m: float) -> float:
        pos_cos = np.cos(batch[0])
        neg_cos = np.cos(batch[1])

        return 1. if pos_cos - neg_cos.max() < m else 0.

    def _gd_met(batch: Tuple[np.ndarray, np.ndarray], m: float) -> float:
        pos_dis = np.sqrt(2 - 2 * np.cos(batch[0]) + 1e-8)
        neg_dis = np.sqrt(2 - 2 * np.cos(batch[1]) + 1e-8)

        return 1. if neg_dis.min() - pos_dis < m else 0.

    
    def _gd_mat(batch: Tuple[np.ndarray, np.ndarray], m: float) -> float:
        return 1. if batch[1].min() - batch[0] < m else 0.
    
    
    if loss_type == 'infonce':
        gd_fn = _gd_infonce
    elif loss_type == 'arccon':
        gd_fn = _gd_arccon
    elif loss_type == 'mpt':
        gd_fn = _gd_mpt
    elif loss_type == 'met':
        gd_fn = _gd_met
    elif loss_type == 'mat':
        gd_fn = _gd_mat
    else:
        raise NotImplementedError
    
    gds = [gd_fn(batch, **kwargs) for batch in batches]

    return sum(gds) / len(gds)

def gd_grid(
    pos_mean_range: Tuple[float, float],
    neg_mean_range: Tuple[float, float],
    pos_std: float, pos_cal_num: int,
    neg_std: float, neg_cal_num: int,
    num_batches: int, batch_size: int,
    loss_type: str, **kwargs: float
) -> dict:
    
    pos_cal_step = (pos_mean_range[1] - pos_mean_range[0]) / (pos_cal_num - 1)
    neg_cal_step = (neg_mean_range[1] - neg_mean_range[0]) / (neg_cal_num - 1)
    res = {
        'pos_mean(pi)': [],
        'neg_mean(pi)': [],
        'gd': []
    }
    for pos_i in tqdm(range(pos_cal_num)):
        for neg_i in range(neg_cal_num):
            pos_mean = pos_mean_range[0] + pos_i * pos_cal_step
            neg_mean = neg_mean_range[0] + neg_i * neg_cal_step
            batches = sample_multiple_batches(
                num_batches, batch_size, pos_mean, neg_mean, pos_std, neg_std
            )
            gd = cal_gd(loss_type, batches, **kwargs)
            res['pos_mean(pi)'].append(round(pos_mean / np.pi, 3))
            res['neg_mean(pi)'].append(round(neg_mean / np.pi, 3))
            res['gd'].append(round(gd, 4))

    return res

def cal_w(
    batches: List[Tuple[np.ndarray, np.ndarray]], 
    temp: float, gd_threshould: float = float('-inf') # 1e-2
) -> float:  
    res = []
    for pos_theta, neg_thetas in batches:
        negs = np.exp(np.cos(neg_thetas) / temp)
        pos = np.exp(np.cos(pos_theta) / temp)
        if 1 / (1 + pos / negs.sum()) >= gd_threshould:
            res.append(np.max(negs) / np.sum(negs))
    if res:
        return sum(res) / len(res)
    else:
        return 1.

def w_grid(
    neg_mean_range: Tuple[float, float], neg_std: float, neg_cal_num: int, 
    temps: List[float], num_batches: int, batch_size: int, gd_threshould: float = float('-inf')
):
    res = {
        'neg_mean(pi)': [],
        'temp': [],
        'w': []
    }

    neg_cal_step = (neg_mean_range[1] - neg_mean_range[0]) / (neg_cal_num - 1)
    for neg_i in tqdm(range(neg_cal_num)):
        neg_mean = neg_mean_range[0] + neg_i * neg_cal_step
        batches = sample_multiple_batches(
            num_batches, batch_size, 0.5, neg_mean, 0.05, neg_std
        )
        for temp in temps:
            w = cal_w(batches, temp, gd_threshould)
            res['neg_mean(pi)'].append(round(neg_mean / np.pi, 3))
            res['temp'].append(round(temp, 3))
            res['w'].append(round(w, 4))
    
    return res

def plot_pcolor_reverse(
    value_fn, A, B,
    title, xlabel, ylabel,
    show=True, save_path=None,
    out_size=24, in_size=16, cmap='tab20c',
    show_y_label=True, mask=False
):
    value_mesh, value_mask = value_fn(B, A)
    
    plt.pcolormesh(
        A, B, value_mesh, cmap=cmap,# alpha=0.5, 
        shading='auto', vmin=0, vmax=5, zorder=1
    )
    cbar = plt.colorbar(ticks=np.arange(0, 5.1, 1.))
    cbar.ax.tick_params(labelsize=in_size)
    if mask:
        rgba_colors = np.zeros((value_mask.shape[0], value_mask.shape[1], 4))
        rgba_colors[:, :, 3] = np.where(np.flipud(value_mask), 0.6, 0.)
        plt.imshow(rgba_colors, extent=(A.min(), A.max(), B.min(), B.max()), 
        interpolation='nearest', aspect='auto', zorder=2)

    plt.title(title, fontproperties=roman_font(out_size))
    plt.xlabel(xlabel, fontproperties=roman_font(out_size))
    plt.ylabel(ylabel, fontproperties=roman_font(out_size))
    plt.xticks(fontsize=in_size)
    plt.yticks(fontsize=in_size)
    ax.set_yticks([np.pi / 20, np.pi / 2])
    if not show_y_label:
        ax.tick_params(axis='y', labelleft=False)
    else:
        ax.set_yticklabels([r'$\frac{\pi}{20}$', r'$\frac{\pi}{2}$'])
    
    ax.set_xticks([np.pi / 20, np.pi])
    ax.set_xticklabels([r'$\frac{\pi}{20}$', r'$\pi$'])


    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def r_arccon(
    theta_ii, theta_ij, u=(10 / 180 * np.pi), temp=5e-2, 
    batch_size=128, threshould=0.01 #, apply_mask=True
):
    value = np.sin(theta_ii + u) / np.sin(theta_ii)
    # mask = theta_ii > theta_ij
    mask = np.logical_or(
        theta_ii > theta_ij,
        1 / (1 + (np.exp(np.cos(theta_ii + u) / temp) \
            / (np.exp(np.cos(theta_ij) / temp) * (batch_size - 1)))) < threshould
    )
    return value, mask #np.ma.array(value, mask=mask) if apply_mask else value

def r_met(
    theta_ii, theta_ij, 
    m=0.45 # , apply_mask=True
):
    pos_dis = np.sqrt(2 - 2 * np.cos(theta_ii) + 1e-8)
    neg_dis = np.sqrt(2 - 2 * np.cos(theta_ij) + 1e-8)
    # mask = theta_ii > theta_ij
    mask = np.logical_or(
        theta_ii > theta_ij,
        neg_dis - pos_dis >= m
    )
    value = neg_dis / pos_dis
    return value, mask # np.ma.array(value, mask=mask) if apply_mask else value

def r_mat(
    theta_ii, theta_ij, 
    m=(0.15 * np.pi) # , apply_mask=True
):
    # mask = theta_ii > theta_ij
    mask = np.logical_or(
        theta_ii > theta_ij,
        theta_ij - theta_ii >= m
    )
    value = np.sqrt((1 - np.cos(theta_ij)**2) / (1 - np.cos(theta_ii)**2))
    return value, mask # np.ma.array(value, mask=mask) if apply_mask else value

if __name__ == '__main__':
    np.random.seed(0)
    
    ''' Gradient Dissipation
    pos_mean_range = (0.05 * np.pi, 0.5 * np.pi)
    neg_mean_range = (0.05 * np.pi, 1 * np.pi)
    pos_std = 0.05 
    neg_std = 0.10 
    pos_cal_num = 100
    neg_cal_num = 100

    num_batches = 1000
    batch_size = 128

    type2kwarg = {
        'infonce': { 'temp': 5e-2},
        'arccon': { 'temp': 5e-2, 'u': 10 / 180 * np.pi },
        'mpt': { 'm': 0.23 },
        'met': { 'm': 0.45 }
    }

    for loss_type, kwargs in tqdm(type2kwarg.items()):
        save_path = rf'statistics\theoretical\simulation\gd\{loss_type}.csv'
        res = gd_grid(
            pos_mean_range, neg_mean_range, 
            pos_std, pos_cal_num, neg_std, neg_cal_num,
            num_batches, batch_size, loss_type, **kwargs
        )
        pd.DataFrame(res).to_csv(save_path, index=False)
    '''    

    ''' Weight
    neg_mean_range = (np.pi / 20, 1 * np.pi)
    neg_std = 0.10
    neg_cal_num = 100
    
    num_batches = 1000
    batch_size = 128
    temps = [3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 5e-3]

    save_path = rf'statistics\theoretical\simulation\w\exp_weight.csv'
    res = w_grid(
        neg_mean_range, neg_std, neg_cal_num, temps, num_batches, batch_size,
        gd_threshould=1e-2
    )
    pd.DataFrame(res).to_csv(save_path, index=False)
    '''

    ''' Ratio
    fig = plt.figure(figsize=(10, 3.30))
    mask = True
    save_path = rf'figure\theoretical\r_simulate.pdf'

    # Define the ranges for a and b
    theta_ii = np.linspace(0.05 * np.pi, 0.5 * np.pi, 100)
    theta_ij = np.linspace(0.05 * np.pi, 1 * np.pi, 100)

    # Create a meshgrid for the combination of a and b
    ij, ii = np.meshgrid(theta_ij, theta_ii)

    ax = fig.add_subplot(121)
    plot_pcolor_reverse(
        r_arccon, ij, ii, 
        'ArcCon', r"$\mu_\mathrm{neg}$", r"$\mu_\mathrm{pos}$", #r"$\theta_{ij'}$", r"$\theta_{ii'}$",
        show=False, mask=mask
    )

    ax = fig.add_subplot(122)
    plot_pcolor_reverse(
        r_met, ij, ii, 
        'MET', r"$\mu_\mathrm{neg}$", r"", # r"$\theta_{ij'}$", r"",
        show=True, save_path=save_path, 
        show_y_label=False, mask=mask
    )
    '''
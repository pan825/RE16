import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy import stats     

_eip_rg = {
    'original': [
        [1, 9, 0, 8],
        [1, 9, 10],
        [2, 10, 1],
        [2, 10, 11],
        [3, 11, 2],
        [3, 11, 12],
        [4, 12, 3],
        [4, 12, 13],
        [5, 13, 4],
        [5, 13, 14],
        [6, 14, 5],
        [6, 14, 15],
        [7, 15, 6],
        [7, 15, 16],
        [8, 16, 7],
        [8, 16, 17, 9]],
    'symmetry': [
        [0, 1, 17],
        [1, 17, 10],
        [1, 2, 10],
        [2, 10, 11],
        [2, 3, 11],
        [3, 11, 12],
        [3, 4, 12],
        [4, 12, 13],
        [4, 5, 13],
        [5, 13, 14],
        [5, 6, 14],
        [6, 14, 15],
        [6, 7, 15],
        [7, 15, 16],
        [7, 0, 16],
        [0, 16, 17]]
}

_WEIGHT_MATRICES = {}
for circuit, groups in _eip_rg.items():
    W = np.zeros((18, len(groups)), dtype=float)
    for j, idx_list in enumerate(groups):
        w = 1.0 / len(idx_list)
        W[idx_list, j] = w
    _WEIGHT_MATRICES[circuit] = W

def conv(fr, kernel_size=1000, stride=1000, in_channels=18) -> tuple[np.ndarray, np.ndarray]:
    """
    fr: (N, t_eval)
    """
    # if fr is a pandas DataFrame, convert it to a numpy array
    if isinstance(fr, pd.DataFrame):
        fr = fr.to_numpy()
    
    fr_torch = torch.from_numpy(fr).float()
    
    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        groups=in_channels
    )

    with torch.no_grad():
        conv.weight.fill_(1 / kernel_size)
        conv.bias.fill_(0)

    data = fr_torch.unsqueeze(0)

    output = conv(data)

    output = output.squeeze(0)
    
    fr_conv = output.detach().numpy()
    t_conv = np.linspace(0, fr_conv.shape[1]/10, fr_conv.shape[1])
    return fr_conv, t_conv

@njit(parallel=True, fastmath=True)
def conv_numba(fr: np.ndarray, kernel_size: int = 1000, stride: int = 1000):
    """
    fr: shape = (C, T)
    回傳 fr_conv: shape = (C, M) 以及 t_conv: shape = (M,)
    """
    C, T = fr.shape
    M = (T - kernel_size) // stride + 1
    fr_conv = np.empty((C, M), dtype=fr.dtype)

    for c in prange(C):
        # 維度 c 的累積和
        csum = np.empty(T+1, dtype=fr.dtype)
        csum[0] = 0.0
        for t in range(T):
            csum[t+1] = csum[t] + fr[c, t]

        # sliding window
        m = 0
        for start in range(0, T - kernel_size + 1, stride):
            end = start + kernel_size
            fr_conv[c, m] = (csum[end] - csum[start]) / kernel_size
            m += 1

    # 時間軸（假設原頻率 10Hz，可自行調整）
    t_conv = np.linspace(0, M*stride/10, M)
    return fr_conv, t_conv


def eip_to_eb(eip_fr, circuit:str='symmetry'):
    """
    input: fr.shape = (N, 18)
    
    circuit: 'original' or 'symmetry'
    """

    
    # Pre-compute mapping once
    mapping = _eip_rg[circuit]
    
    # Pre-allocate output array
    N = len(eip_fr)
    eb_fr = np.zeros((N, 16))
    
    # Use vectorized operations where possible
    for j in range(16):
        indices = mapping[j]
        # Calculate mean across specified indices for all time points at once
        eb_fr[:, j] = np.mean(eip_fr[:, indices], axis=1)
    
    return eb_fr

def eip_to_eb_fast(eip_fr: np.ndarray, circuit: str = 'symmetry') -> np.ndarray:
    """
    將 eip_fr (shape=(T,18)) 乘上預先計算好的 weight matrix 一次得到 eb_fr (shape=(T,16))
    """
    eip = np.asarray(eip_fr)
    W = _WEIGHT_MATRICES[circuit]
    eb_fr = eip.dot(W)
    return eb_fr

@njit(parallel=True, fastmath=True)
def eip_to_eb_numba(eip_fr: np.ndarray):
    """
    eip_fr: shape = (T, 18)
    回傳 eb_fr: shape = (T, 16)
    """
    T, _ = eip_fr.shape
    eb_fr = np.empty((T, 16), dtype=eip_fr.dtype)
    for t in prange(T):
        for j in range(16):
            idxs = _eip_rg[j]
            s = 0.0
            for k in range(len(idxs)):
                s += eip_fr[t, idxs[k]]
            eb_fr[t, j] = s / len(idxs)
    return eb_fr

def gau_fit(t: np.ndarray, fr: np.ndarray, step=1):
    """
    Performs Gaussian fitting on firing rate data.
    
    Parameters:
    t (ndarray): Time data with shape (N,)
    fr (ndarray): Firing rate data with shape (16, N)
    step (int): Step size for processing data
    
    Returns:
    tuple of (g_t, g_x, g_w) as numpy arrays, where g_t matches the time points
    """
    def func(_x, a, b, c):
        return a * np.exp(-(_x - b)**2 / c)

    def is_continuous(data):
        # Vectorized check for continuity pattern
        nonzero_indices = np.where(data != 0)[0]
        if len(nonzero_indices) == 0:
            return True
        return np.max(nonzero_indices) - np.min(nonzero_indices) + 1 == len(nonzero_indices)

    def to_continuous(data):
        data = np.array(data)
        nonzero_indices = np.where(data != 0)[0]
        if len(nonzero_indices) == 0:
            return data, 0
        
        first_nonzero = np.min(nonzero_indices)
        last_nonzero = np.max(nonzero_indices)
        
        # Calculate shift needed
        index = len(data) - last_nonzero - 1
        
        # Create continuous array
        result = np.concatenate((data[last_nonzero+1:], data[:last_nonzero+1]))
        
        return result, index

    def gaussian_fit(data):
        data = np.asarray(data)
        x = np.arange(len(data))
        
        if is_continuous(data):
            try:
                popt, _ = curve_fit(func, x, data, bounds=([0.1, 0.1, 0.1], [1000, len(data), 1000]))
            except RuntimeError:
                popt, _ = curve_fit(func, x, data, bounds=([0.1, 0.1, 0.1], [1000, len(data), 200]))
        else:
            data, moving_num = to_continuous(data)
            try:
                popt, _ = curve_fit(func, x, data, bounds=([0.1, 0.1, 0.1], [1000, len(data), 1000]))
            except RuntimeError:
                popt, _ = curve_fit(func, x, data, bounds=([0.1, 0.1, 0.1], [1000, len(data), 200]))
            
            popt[1] -= moving_num
            popt[1] %= len(data)
            if popt[1] >= (len(data) - 0.5):
                popt[1] -= len(data)

        return popt

    # Ensure fr is in the right format (N, 16)
    fr_transposed = fr.T if fr.shape[0] == 16 else fr
    
    # Pre-allocate lists with estimated capacity
    est_capacity = len(fr_transposed) // step + 1
    g_t_list = np.zeros(est_capacity)
    g_y_list = np.zeros(est_capacity)
    g_x_list = np.zeros(est_capacity)
    g_c_list = np.zeros(est_capacity)
    
    # Track actual number of valid fits
    valid_count = 0
    
    # Handle both DataFrame and ndarray inputs
    is_dataframe = hasattr(fr_transposed, 'iloc')
    
    # Create a range object once
    indices = range(0, len(fr_transposed), step)
    
    for i in indices:
        # Get current data row based on input type
        if is_dataframe:
            current_fr = fr_transposed.iloc[i, :].values  # Convert to numpy for faster operations
        else:
            current_fr = fr_transposed[i]
        
        # Fast checks for invalid data
        if np.all(current_fr == 0) or np.all(current_fr < 10) or np.count_nonzero(current_fr) == 2:
            continue
        
        try:
            popt = gaussian_fit(current_fr).round(5)
            # Use actual time point from t array instead of index
            g_t_list[valid_count] = t[i] if i < len(t) else i
            g_y_list[valid_count] = popt[0]
            g_x_list[valid_count] = popt[1]
            g_c_list[valid_count] = popt[2]
            valid_count += 1
        except:
            continue
    
    # Trim arrays to actual size
    g_t = g_t_list[:valid_count]
    g_x = g_x_list[:valid_count]
    g_c = g_c_list[:valid_count]
    g_y = g_y_list[:valid_count]
    g_w = 2 * np.sqrt(g_c / 2)
    
    return g_t, g_x, g_y, g_w
    
def translate_gau(g_t, g_x, g_y, g_w):
    """
    Translates the Gaussian fit data to handle discontinuities in angular data.
    
    Parameters:
    g_t (ndarray): Time points
    g_x (ndarray): Angular positions (0-16)
    g_w (ndarray): Widths of the Gaussian fits
    
    Returns:
    tuple: (g_t, g_x, g_w) with g_x adjusted for continuity
    """
    if len(g_x) <= 1:
        return g_t, g_x, g_y, g_w
        
    g_x_translated = g_x.copy()
    offset = 0
    
    # Threshold for detecting jumps (half of the total range)
    threshold = 8
    
    for i in range(1, len(g_x)):
        diff = g_x[i] - g_x[i-1]
        
        # If there's a large positive jump (e.g., 1 to 15)
        if diff < -threshold:
            offset += 16
        # If there's a large negative jump (e.g., 15 to 1)
        elif diff > threshold:
            offset -= 16
            
        g_x_translated[i] += offset
    
    return g_t, g_x_translated, g_y, g_w


def add_array(existing_array, new_data, axis=0):
    """
    usage:
    t = add_array(t, t_new, axis=0)
    fr = add_array(fr, fr_new, axis=1)
    """
    
    new_array = np.array(new_data)
    if existing_array is None:
        return new_array
    else:
        return np.concatenate((existing_array, new_array), axis=axis) 



def neuronID2rad(neuron):
    """
    neuron: from 0 to 16
    angle: rad  
    """
    return neuron * 2 * np.pi / 16

def rad2neuronID(rad):
    return rad * 16 / (2 * np.pi)

def plot_original(f, gau=False):
    data = np.loadtxt(f)
    t = data[:, 0]
    fr = data[:, 1:]
    fr_with_zeros = np.zeros((fr.shape[0], fr.shape[1] + 2))
    fr_with_zeros[:, :8] = fr[:, :8]
    fr_with_zeros[:, 10:] = fr[:, 8:]
    eb_fr = eip_to_eb(fr_with_zeros)
    eb_fr = eb_fr.T
    if gau:
        g_t, g_x, g_y, g_w = gau_fit(t, eb_fr)
        eb_fr = eb_fr.T
        plt.pcolormesh(t, np.arange(eb_fr.shape[0]), eb_fr, cmap='Blues', shading='nearest')
        plt.colorbar(label='Firing Rate [Hz]')
        plt.show()
    else:
        plt.pcolormesh(t, np.arange(eb_fr.shape[0]), eb_fr, cmap='Blues', shading='nearest')
        plt.colorbar(label='Firing Rate [Hz]')
        plt.show()
        
        
def get_parameters_from(f:str):
    pattern = r'w_EE_(?P<w_EE>[\d\.]+)_w_EI_(?P<w_EI>[\d\.]+)_w_IE_(?P<w_IE>[\d\.]+)_w_II_(?P<w_II>[\d\.]+)_w_EP_(?P<w_EP>[\d\.]+)_w_PE_(?P<w_PE>[\d\.]+)_w_PP_(?P<w_PP>[\d\.]+)\.*'
    match = re.search(pattern, f)
    if match:
        params = match.groupdict()
        print(f'{f} \nparameters = {params}')
        return params
    else:
        raise Warning(f"Invalid file name: {f}")
        return 0
    
def euler_to_quat(yaw, pitch, roll):
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = sy*cp*sr + cy*sp*cr
    z = sy*cp*cr - cy*sp*sr
    return np.array([w, x, y, z])

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_mult(a, b):
    w0,x0,y0,z0 = a; w1,x1,y1,z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def so3_control(q_cur, q_tgt, kp=4.0):
    # 誤差四元數
    q_err = quat_mult(q_tgt, quat_conj(q_cur))
    qw = np.clip(q_err[0], -1.0, 1.0)
    theta = 2 * np.arccos(qw)
    if theta < 1e-6:
        return np.zeros(3)
    axis = q_err[1:] / np.sin(theta/2)
    return kp * theta * axis  # body-frame 角速度
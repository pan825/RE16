import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

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

def fit_slope(g_t, g_x, t_threshold=0.7, t_end=None):
    """
    Fit a linear regression to g_x vs g_t data after a specified time threshold
    and return the slope, correlation coefficient, and standard error.
    
    Parameters:
    g_t (ndarray): Time points
    g_x (ndarray): Angular positions
    t_threshold (float): Time point after which to fit the data (default: 0.7)
    
    Returns:
    tuple: (slope, r, std_err) where:
        slope: Slope of the fitted line
        r: Correlation coefficient
        std_err: Standard error of the slope
    """
    # Convert inputs to numpy arrays if they aren't already
    g_t = np.array(g_t)
    g_x = np.array(g_x)
    
    # Filter data after t_threshold
    if t_end is None:
        mask = g_t >= t_threshold
    else:
        mask = (g_t >= t_threshold) & (g_t <= t_end)
    t_filtered = g_t[mask]
    x_filtered = g_x[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_filtered, x_filtered)
    mean_x = np.mean(x_filtered)
    CV = np.std(x_filtered) / mean_x
    
    return slope, r_value**2, std_err, CV
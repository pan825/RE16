import io
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from brian2 import *
import RE16
import utils

@dataclass
class NetworkParameters:
    """Parameters for the ring attractor network model.
    
    All weights represent connection strengths between neural populations.
    """
    w_EE: float = 0.719  # EB <-> EB 
    w_EI: float = 0.143  # R -> EB 
    w_IE: float = 0.74   # EB -> R 
    w_II: float = 0.01   # R <-> R 
    w_EP: float = 0.012  # EB -> PEN 
    w_PE: float = 0.709  # PEN -> EB 
    w_PP: float = 0.01   # PEN <-> PEN 
    sigma: float = 0.0001  # Noise level


class RE16Simulator:
    """Implementation of a ring attractor neural network model.
    
    This class handles simulation, data processing, analysis, and visualization
    of a ring attractor network based on the Drosophila central complex.
    """
    
    def __init__(self, parameters: NetworkParameters = None):
        """Initialize the ring attractor network with given parameters.
        
        Args:
            parameters: Configuration parameters for the network
        """
        self.parameters = parameters or NetworkParameters()
        
        # Simulation results
        self.time: Optional[np.ndarray] = None
        self.fr: Optional[np.ndarray] = None
        
        # Processed results
        self.processed_time: Optional[np.ndarray] = None
        self.processed_fr: Optional[np.ndarray] = None
        
        # Gaussian fit results
        self.gt: Optional[np.ndarray] = None
        self.gx: Optional[np.ndarray] = None
        self.gfr: Optional[np.ndarray] = None
        self.gw: Optional[np.ndarray] = None
        
        # Analysis results
        self.slope: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.std_err: Optional[float] = None
        self.coefficient_variation: Optional[float] = None
        self.angular_velocity: Optional[float] = None
        self.rotations_per_second: Optional[float] = None
        
        # Simulation parameters
        self.t_epg_open: Optional[int] = None
        self.t_epg_close: Optional[int] = None
        self.t_pen_open: Optional[int] = None
        self.stimulus_strength: Optional[float] = None
        self.stimulus_location: Optional[float] = None
        self.shifter_strength: Optional[float] = None
        self.half_PEN: Optional[str] = None
        
    def setup(self, 
            t_epg_open: int = 200, 
            t_epg_close: int = 500, 
            t_pen_open: int = 5000,
            stimulus_strength: float = 0.05,
            stimulus_location: float = 0.0,
            shifter_strength: float = 0.015,
            half_PEN: str = 'right'):
        """Setup the network simulation with specified parameters.
        
        Args:
            t_epg_open: Time (ms) when the stimulus starts
            t_epg_close: Time (ms) when the stimulus ends
            t_pen_open: Time (ms) when the PEN neurons become active
            stimulus_strength: Strength of the external stimulus
            stimulus_location: Angular location of stimulus (0 to π radians)
            shifter_strength: Strength of the shifter input
            half_PEN: Which half of PEN neurons to activate ('left' or 'right')
        """
        self.t_epg_open = t_epg_open
        self.t_epg_close = t_epg_close
        self.t_pen_open = t_pen_open
        self.stimulus_strength = stimulus_strength
        self.stimulus_location = stimulus_location
        self.shifter_strength = shifter_strength
        self.half_PEN = half_PEN
        return 
        
    def run(self, 
            t_epg_open: int = None, 
            t_epg_close: int = None, 
            t_pen_open: int = None,
            stimulus_strength: float = None,
            stimulus_location: float = None,
            shifter_strength: float = None,
            half_PEN: str = None):

        if t_epg_open is not None:
            self.t_epg_open = t_epg_open
        if t_epg_close is not None:
            self.t_epg_close = t_epg_close
        if t_pen_open is not None:
            self.t_pen_open = t_pen_open
        if stimulus_strength is not None:
            self.stimulus_strength = stimulus_strength
        if stimulus_location is not None:
            self.stimulus_location = stimulus_location
        if shifter_strength is not None:
            self.shifter_strength = shifter_strength
        if half_PEN is not None:
            self.half_PEN = half_PEN
        
        t, fr = RE16.simulator(
            **self.parameters.__dict__,
            stimulus_strength=self.stimulus_strength,
            stimulus_location=self.stimulus_location,
            shifter_strength=self.shifter_strength,
            half_PEN=self.half_PEN,
            t_epg_open=self.t_epg_open,
            t_epg_close=self.t_epg_close,
            t_pen_open=self.t_pen_open,
        )
        
        
        # Store results (append if multiple simulations)
        self.time = utils.add_array(self.time, t, axis=0) 
        self.fr = utils.add_array(self.fr, fr, axis=1)

    def process_data(self):
        """Process raw simulation data for analysis.
        
        Transforms the raw firing rates into a format suitable for analysis,
        including conversion to ellipsoid body (EB) representation.
        """
        t, fr = self.get_raw_results()
        
        # Insert zeros at positions 8 and 9 (missing neurons in the circuit)
        expanded_rates = np.insert(fr, 8, 0, axis=0)
        expanded_rates = np.insert(expanded_rates, 9, 0, axis=0)
        
        # Apply temporal convolution and convert to EB representation
        conv_rates, conv_time = utils.conv(expanded_rates)
        eb_fr = utils.eip_to_eb_fast(conv_rates.T)
    
        self.processed_time = conv_time
        self.processed_fr = eb_fr.T
    
    def save(self, file_path='simulation_results.dat', folder=None):
        """Save simulation results to a file.
        
        Args:
            file_path: Name of the file to save results to
            folder: Optional folder path for the file
        """
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
            file_path = os.path.join(folder, file_path)
    
        t = self.time
        fr = self.fr
            
        with open(file_path, 'w') as file:
            for i in range(len(t)):
                row = f'{t[i]} '
                row += ' '.join([f'{fr[j,i]}' for j in range(fr.shape[0])])
                file.write(row + '\n')
                
        print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")}: file saved as {file_path}')

    def load(self, file_path='simulation_results.dat'):
        """Load simulation results from a file.
        
        Args:
            file_path: Path to the file containing simulation results
        """
        # Use numpy for efficient file loading
        data = np.loadtxt(file_path)
        t = data[:, 0]
        fr = data[:, 1:]  # EIP0 - EIP17
        
        # Insert zeros at positions 8 and 9 (missing neurons)
        fr_with_zeros = np.zeros((fr.shape[0], fr.shape[1] + 2))
        fr_with_zeros[:, :8] = fr[:, :8]
        fr_with_zeros[:, 10:] = fr[:, 8:]
        
        # Process the data
        fr_conv, t_conv = utils.conv(fr_with_zeros.T)
        eb_fr = utils.eip_to_eb_fast(fr_conv.T)
        
        self.time = t
        self.fr = fr.T
        self.processed_time = t_conv
        self.processed_fr = eb_fr.T

    def fit_gaussian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a Gaussian to the processed firing rate data.
        
        Returns:
            Tuple containing time points, positions, amplitudes, and widths
        """
        t, fr = self.get_processed_results()
        gt, gx, gfr, gw = utils.gau_fit(
            t, fr
        )
        
        # Translate the Gaussian parameters to a consistent coordinate system
        gt, gx, gfr, gw = utils.translate_gau(
            gt, gx, gfr, gw
        )
        
        self.gx = gx
        self.gt = gt
        self.gw = gw
        self.gfr = gfr
        
        return gt, gx, gfr, gw
        
    def fit_velocity(self, time_threshold=None, time_end=None):
        """Calculate the angular velocity from Gaussian position data.
        
        Args:
            time_threshold: Starting time for the fit (defaults to stimulus close time)
            time_end: Ending time for the fit (defaults to end of simulation)
            
        Returns:
            Tuple of (slope, r_squared, std_err, CV)
        """
        if time_threshold is None and self.t_epg_close is not None:
            time_threshold = (self.t_epg_close + self.t_epg_open)/1000
            
        self._ensure_gaussian_fit()
            
        slope, r_value, std_err, CV = utils.fit_slope(
            self.gt, 
            self.gx, 
            time_threshold, 
            time_end
        )
        
        self.slope = slope
        self.r_squared = r_value**2
        self.std_err = std_err
        self.coefficient_variation = CV
        self.angular_velocity = self.slope * 2 * np.pi / 16
        
        # Rotations per second
        self.rotations_per_second = self.angular_velocity / (2 * np.pi)
        
        return slope, r_value**2, std_err, CV
        

    
    def reset(self):
        """Clear all simulation and analysis results."""
        self.fr = None
        self.time = None
        self.processed_time = None
        self.processed_fr = None
        self.gx = None
        self.gt = None
        self.gw = None
        self.gfr = None
        self.slope = None
        self.r_value = None
        self.std_err = None
        self.angular_velocity = None
        
    def _ensure_gaussian_fit(self):
        """Ensure that Gaussian fit has been performed."""
        if self.gx is None:
            self.fit_gaussian()
            
    def _ensure_processed_data(self):
        """Ensure that data processing has been performed."""
        if self.processed_time is None:
            self.process_data()

    def plot(self, title=None, file_name=None, region='EB', y_label='Time (s)', 
                      cmap='Blues', save=False, folder='figures', plot_gaussian=True, 
                      figsize=(10, 2.5)):
        """Visualize the neural activity as a color plot.
        
        Args:
            title: Plot title
            file_name: Filename for saving the plot
            region: Brain region to label ('EB' or other)
            y_label: Label for y-axis
            cmap: Colormap for the plot
            save: Whether to save the plot
            folder: Folder for saving
            plot_gaussian: Whether to plot the Gaussian fit
            figsize: Figure size
        """
        t, fr = self.get_processed_results()
        plot_results(t, fr, title, file_name, region, 
                    y_label, cmap, save, folder, plot_gaussian, figsize)
        
    def plot_raw(self, title=None, file_name=None, region='EB', y_label='Time (s)', 
                         cmap='Blues', save=False, folder='figures', plot_gaussian=True, 
                         figsize=(10, 2.5), eip2eb=True):
        """Visualize the raw neural activity.
        
        Args:
            title: Plot title
            file_name: Filename for saving the plot
            region: Brain region to label ('EB' or other)
            y_label: Label for y-axis
            cmap: Colormap for the plot
            save: Whether to save the plot
            folder: Folder for saving
            plot_gaussian: Whether to plot the Gaussian fit
            figsize: Figure size
            eip2eb: Whether to convert EIP to EB coordinates
        """
        t = self.time
        fr = self.fr.T
        
        if eip2eb:
            fr_with_zeros = np.zeros((fr.shape[0], fr.shape[1] + 2))
            fr_with_zeros[:, :8] = fr[:, :8]
            fr_with_zeros[:, 10:] = fr[:, 8:]
            eb_fr = utils.eip_to_eb_fast(fr_with_zeros)
            eb_fr = eb_fr.T
        else:
            eb_fr = fr
            
        plt.figure(figsize=figsize)
        plt.pcolormesh(t, range(eb_fr.shape[0]), eb_fr, cmap=cmap, shading='nearest')
        plt.colorbar(label='Firing Rate [Hz]')    
        plt.title(title)
        plt.xlabel(y_label)
        plt.ylabel('EB region' if region == 'EB' else 'Neuron ID')
        plt.yticks([0, 4, 11, 15], ['R8', 'R4', 'L4', 'L8'] if region == 'EB' else [0, 5, 10, 15])
        
        if plot_gaussian:
            processed_time, processed_fr = self.get_processed_results()
            g_t, g_x, g_y, g_w = utils.gau_fit(processed_time, processed_fr)
            plt.plot(g_t, g_x, 'r', linewidth=3)
        
        if save:
            plt.savefig(os.path.join(folder, file_name))
            plt.close()
    
    def summary(self):
        """Print a summary of the simulation and analysis results."""
        self._ensure_gaussian_fit()
        
        speed_rad = f'{self.angular_velocity:.3f}'
        speed_deg = f'{np.rad2deg(self.angular_velocity):.3f}'
        err_rad = f'{self.std_err*2*np.pi/16:.3f}'
        err_deg = f'{np.rad2deg(self.std_err*2*np.pi/16):.3f}'
        
        print('='* 40)
        print(f'Angular velocity: {speed_rad:>8} ± {err_rad:>8} [rad/s]')
        print(f'                  {speed_deg:>8} ± {err_deg:>8} [deg/s]')
        print(f'Rotations/sec:    {self.rotations_per_second:>8} ± {self.std_err/16:>8} [Hz]')
        
        # Color code r_squared based on quality
        if abs(self.r_value) >= 0.95:
            print('\033[92m' + f'R-squared: {self.r_squared:.3f}' + '\033[0m')
        elif np.isnan(self.r_value):
            print('\033[91m' + f'R-squared: {self.r_squared:.3f}' + '\033[0m')
        else: 
            print(f'R-squared: {self.r_squared:.3f}')
        
        # Color code bump width based on quality
        mean_bump_width = np.rad2deg(np.mean(self.gw*np.pi/8))
        std_bump_width = np.rad2deg(np.std(self.gw*np.pi/8))
        
        if mean_bump_width >= 360:
            print('\033[91m' + f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]' + '\033[0m')
        else:
            print(f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]')
        
        print(f'Average firing rate: {np.mean(self.gfr):.3f} ± {np.std(self.gfr):.3f} [Hz]')
        
        # Print the parameters
        print('Network parameters:')
        for param, value in self.parameters.__dict__.items():
            print(f'  {param}: {value}')
            
        print(f'Stimulus open time: {self.t_epg_open} ms')
        print(f'Stimulus close time: {self.t_epg_close} ms')
        print(f'PEN open time: {self.t_pen_open} ms')
        print(f'Stimulus strength: {self.stimulus_strength}')
        print(f'Stimulus location: {self.stimulus_location} rad')
        print(f'Shifter strength: {self.shifter_strength}')
        print(f'Half PEN: {self.half_PEN}')
        print('='*40)
        
        
def plot_results(t, fr, title=None, file_name=None, region='EB', 
              y_label='Time (s)', cmap='Blues', save=False, folder='figures', 
              plot_gaussian=True, figsize=(10, 2.5)):
    """Plot neural activity data.
    
    Args:
        t: Array of time points
        fr: Matrix of firing rates (neurons x time)
        title: Plot title
        file_name: Filename for saving
        region: Brain region label
        y_label: Y-axis label
        cmap: Colormap
        save: Whether to save the plot
        folder: Folder for saving
        plot_gaussian: Whether to plot the Gaussian fit
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.pcolormesh(t, range(fr.shape[0]), fr, 
                  cmap=cmap, shading='nearest')   
    plt.colorbar(label='Firing Rate [Hz]')
    plt.xlabel(y_label)
    plt.ylabel('EB region' if region == 'EB' else 'Neuron ID')
    plt.yticks([0, 4, 11, 15], ['R8', 'R4', 'L4', 'L8'] if region == 'EB' else [0, 5, 10, 15])
    plt.title(title)

    if plot_gaussian:
        g_t, g_x, g_y, g_w = utils.gau_fit(t, fr)
        plt.plot(g_t, g_x, 'r', linewidth=3)
        
    if save:
        plt.savefig(os.path.join(folder, file_name))
        plt.close()
        
if __name__ == '__main__':
    network = RE16Simulator()
    network.run(stimulus_strength=0.05, stimulus_location=0.0, shifter_strength=0.015, half_PEN='right')
    network.process_data()
    network.save(file_path='simulation_results.dat', folder='results')
    network.plot(title='Activity', file_name='activity.png', region='EB', save=True, folder='figures')
    

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

# Configure Brian2 preferences
prefs.codegen.target = "numpy"
set_device('runtime', build_on_run=False)


@dataclass
class NetworkParameters:
    """Parameters for the ring attractor network model.
    
    All weights represent connection strengths between neural populations.
    """
    w_EE: float = 0.719  # EB <-> EB connections
    w_EI: float = 0.143  # R -> EB connections
    w_IE: float = 0.74   # EB -> R connections
    w_II: float = 0.01   # R <-> R connections
    w_EP: float = 0.012  # EB -> PEN connections
    w_PE: float = 0.709  # PEN -> EB connections
    w_PP: float = 0.01   # PEN <-> PEN connections
    sigma: float = 0.0001  # Noise level


class RingAttractorNetwork:
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
        self.time_points: Optional[np.ndarray] = None
        self.firing_rates: Optional[np.ndarray] = None
        
        # Processed results
        self.processed_time: Optional[np.ndarray] = None
        self.processed_firing_rates: Optional[np.ndarray] = None
        
        # Gaussian fit results
        self.gaussian_time: Optional[np.ndarray] = None
        self.gaussian_position: Optional[np.ndarray] = None
        self.gaussian_amplitude: Optional[np.ndarray] = None
        self.gaussian_width: Optional[np.ndarray] = None
        
        # Analysis results
        self.slope: Optional[float] = None
        self.r_value: Optional[float] = None
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
        
    def run(self, 
            t_epg_open: int = 200, 
            t_epg_close: int = 500, 
            t_pen_open: int = 5000,
            stimulus_strength: float = 0.05,
            stimulus_location: float = 0.0,
            shifter_strength: float = 0.015,
            half_PEN: str = 'right'):
        """Run the network simulation with specified parameters.
        
        Args:
            t_epg_open: Time (ms) when the stimulus starts
            t_epg_close: Time (ms) when the stimulus ends
            t_pen_open: Time (ms) when the PEN neurons become active
            stimulus_strength: Strength of the external stimulus
            stimulus_location: Angular location of stimulus (0 to π radians)
            shifter_strength: Strength of the shifter input
            half_PEN: Which half of PEN neurons to activate ('left' or 'right')
            
        Returns:
            None: Results stored in instance variables
        """
        time_points, firing_rates = RE16.simulator(
            **self.parameters.__dict__,
            stimulus_strength=stimulus_strength,
            stimulus_location=stimulus_location,
            shifter_strength=shifter_strength,
            half_PEN=half_PEN,
            t_epg_open=t_epg_open,
            t_epg_close=t_epg_close,
            t_pen_open=t_pen_open,
        )
        
        # Store simulation parameters
        self.t_epg_open = t_epg_open
        self.t_epg_close = t_epg_close
        self.t_pen_open = t_pen_open
        self.stimulus_strength = stimulus_strength
        self.stimulus_location = stimulus_location
        self.shifter_strength = shifter_strength
        self.half_PEN = half_PEN
        
        # Store results (append if multiple simulations)
        self.time_points = utils.add_array(self.time_points, time_points, axis=0) 
        self.firing_rates = utils.add_array(self.firing_rates, firing_rates, axis=1)

    def process_data(self):
        """Process raw simulation data for analysis.
        
        Transforms the raw firing rates into a format suitable for analysis,
        including conversion to ellipsoid body (EB) representation.
        """
        time_points, firing_rates = self.get_raw_results()
        
        # Insert zeros at positions 8 and 9 (missing neurons in the circuit)
        expanded_rates = np.insert(firing_rates, 8, 0, axis=0)
        expanded_rates = np.insert(expanded_rates, 9, 0, axis=0)
        
        # Apply temporal convolution and convert to EB representation
        conv_rates, conv_time = utils.conv(expanded_rates)
        eb_firing_rates = utils.eip_to_eb_fast(conv_rates.T)
    
        self.processed_time = conv_time
        self.processed_firing_rates = eb_firing_rates.T
    
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
    
        time_points = self.time_points
        firing_rates = self.firing_rates
            
        with open(file_path, 'w') as file:
            for i in range(len(time_points)):
                row = f'{time_points[i]} '
                row += ' '.join([f'{firing_rates[j,i]}' for j in range(firing_rates.shape[0])])
                file.write(row + '\n')
                
        print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")}: file saved as {file_path}')

    def load(self, file_path='simulation_results.dat'):
        """Load simulation results from a file.
        
        Args:
            file_path: Path to the file containing simulation results
        """
        # Use numpy for efficient file loading
        data = np.loadtxt(file_path)
        time_points = data[:, 0]
        firing_rates = data[:, 1:]  # EIP0 - EIP17
        
        # Insert zeros at positions 8 and 9 (missing neurons)
        fr_with_zeros = np.zeros((firing_rates.shape[0], firing_rates.shape[1] + 2))
        fr_with_zeros[:, :8] = firing_rates[:, :8]
        fr_with_zeros[:, 10:] = firing_rates[:, 8:]
        
        # Process the data
        fr_conv, t_conv = utils.conv(fr_with_zeros.T)
        eb_fr = utils.eip_to_eb_fast(fr_conv.T)
        
        self.time_points = time_points
        self.firing_rates = firing_rates.T
        self.processed_time = t_conv
        self.processed_firing_rates = eb_fr.T

    def fit_gaussian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a Gaussian to the processed firing rate data.
        
        Returns:
            Tuple containing time points, positions, amplitudes, and widths
        """
        time_points, firing_rates = self.get_processed_results()
        gaussian_time, gaussian_position, gaussian_amplitude, gaussian_width = utils.gau_fit(
            time_points, firing_rates
        )
        
        # Translate the Gaussian parameters to a consistent coordinate system
        gaussian_time, gaussian_position, gaussian_amplitude, gaussian_width = utils.translate_gau(
            gaussian_time, gaussian_position, gaussian_amplitude, gaussian_width
        )
        
        self.gaussian_position = gaussian_position
        self.gaussian_time = gaussian_time
        self.gaussian_width = gaussian_width
        self.gaussian_amplitude = gaussian_amplitude
        
        return gaussian_time, gaussian_position, gaussian_amplitude, gaussian_width
        
    def fit_velocity(self, time_threshold=None, time_end=None):
        """Calculate the angular velocity from Gaussian position data.
        
        Args:
            time_threshold: Starting time for the fit (defaults to stimulus close time)
            time_end: Ending time for the fit (defaults to end of simulation)
            
        Returns:
            Tuple of (slope, r_value, std_err, CV)
        """
        if time_threshold is None and self.t_epg_close is not None:
            time_threshold = (self.t_epg_close + self.t_epg_open)/1000
            
        if self.gaussian_position is None:
            self.fit_gaussian()
            
        slope, r_value, std_err, CV = utils.fit_slope(
            self.gaussian_time, 
            self.gaussian_position, 
            time_threshold, 
            time_end
        )
        
        self.slope = slope
        self.r_value = r_value
        self.r_squared = r_value**2
        self.std_err = std_err
        self.coefficient_variation = CV
        self.angular_velocity = self.slope * 2 * np.pi / 16
        
        # Rotations per second
        self.rotations_per_second = self.angular_velocity / (2 * np.pi)
        
        return slope, r_value, std_err, CV
        
    def get_raw_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw simulation results.
        
        Returns:
            Tuple of (time_points, firing_rates)
        """
        return self.time_points, self.firing_rates

    def get_processed_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get processed simulation results.
        
        Returns:
            Tuple of (processed_time, processed_firing_rates)
        """
        if self.processed_time is None:
            self.process_data()
        return self.processed_time, self.processed_firing_rates
    
    def get_gaussian_fit_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the results of the Gaussian fit.
        
        Returns:
            Tuple of (gaussian_time, gaussian_position, gaussian_amplitude, gaussian_width)
        """
        if self.gaussian_position is None:
            self.fit_gaussian()
        return self.gaussian_time, self.gaussian_position, self.gaussian_amplitude, self.gaussian_width
    
    def get_velocity_parameters(self, time_threshold=None, time_end=None) -> Tuple[float, float, float]:
        """Get the parameters describing the velocity fit.
        
        Args:
            time_threshold: Starting time for the fit
            time_end: Ending time for the fit
            
        Returns:
            Tuple of (slope, r_value, std_err)
        """
        try:
            if self.gaussian_position is None:
                self.fit_gaussian()
            self.fit_velocity(time_threshold=time_threshold, time_end=time_end)
            return self.slope, self.r_value, self.std_err
        except:
            return 0, 0, 0
    
    def get_angular_velocity(self, time_threshold=None, time_end=None) -> float:
        """Get the angular velocity in radians per second.
        
        Args:
            time_threshold: Starting time for the fit
            time_end: Ending time for the fit
            
        Returns:
            Angular velocity in radians per second
        """
        try:
            if self.gaussian_position is None:
                self.fit_gaussian()
            self.fit_velocity(time_threshold=time_threshold, time_end=time_end)
            return self.angular_velocity
        except:
            return 0
        
    def get_velocity_std_error(self, time_threshold=None, time_end=None) -> float:
        """Get the standard error of the angular velocity fit.
        
        Args:
            time_threshold: Starting time for the fit
            time_end: Ending time for the fit
            
        Returns:
            Standard error of the velocity fit
        """
        try:
            if self.gaussian_position is None:
                self.fit_gaussian()
            self.fit_velocity(time_threshold=time_threshold, time_end=time_end)
            return self.std_err
        except:
            return 0
        
    def get_final_position(self) -> float:
        """Get the final angular position of the bump in radians.
        
        Returns:
            Final angular position in radians
        """
        try:
            if self.gaussian_position is None:
                self.fit_gaussian()
            return utils.neuronID2rad(self.gaussian_position[-1])
        except:
            return 0
        
    def get_bump_width(self) -> float:
        """Get the average width of the activity bump in degrees.
        
        Returns:
            Average bump width in degrees
        """
        return np.rad2deg(np.mean(self.gaussian_width*np.pi/8))
    
    def get_bump_width_std(self) -> float:
        """Get the standard deviation of the bump width in degrees.
        
        Returns:
            Standard deviation of bump width in degrees
        """
        return np.rad2deg(np.std(self.gaussian_width*np.pi/8))
    
    def get_fit_quality(self) -> float:
        """Get the R-squared value of the velocity fit.
        
        Returns:
            R-squared value of the velocity fit
        """
        return self.r_squared
    
    def reset(self):
        """Clear all simulation and analysis results."""
        self.firing_rates = None
        self.time_points = None
        self.processed_time = None
        self.processed_firing_rates = None
        self.gaussian_position = None
        self.gaussian_time = None
        self.gaussian_width = None
        self.gaussian_amplitude = None
        self.slope = None
        self.r_value = None
        self.std_err = None
        self.angular_velocity = None
        
    def _ensure_gaussian_fit(self):
        """Ensure that Gaussian fit has been performed."""
        if self.gaussian_position is None:
            self.fit_gaussian()
            self.fit_velocity()
            
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
        time_points, firing_rates = self.get_processed_results()
        plot_results(time_points, firing_rates, title, file_name, region, 
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
        time_points = self.time_points
        firing_rates = self.firing_rates.T
        
        if eip2eb:
            fr_with_zeros = np.zeros((firing_rates.shape[0], firing_rates.shape[1] + 2))
            fr_with_zeros[:, :8] = firing_rates[:, :8]
            fr_with_zeros[:, 10:] = firing_rates[:, 8:]
            eb_fr = utils.eip_to_eb_fast(fr_with_zeros)
            eb_fr = eb_fr.T
        else:
            eb_fr = firing_rates
            
        plt.figure(figsize=figsize)
        plt.pcolormesh(time_points, range(eb_fr.shape[0]), eb_fr, cmap=cmap, shading='nearest')
        plt.colorbar(label='Firing Rate [Hz]')    
        plt.title(title)
        plt.xlabel(y_label)
        plt.ylabel('EB region' if region == 'EB' else 'Neuron ID')
        plt.yticks([0, 4, 11, 15], ['R8', 'R4', 'L4', 'L8'] if region == 'EB' else [0, 5, 10, 15])
        
        if plot_gaussian:
            processed_time, processed_firing_rates = self.get_processed_results()
            g_t, g_x, g_y, g_w = utils.gau_fit(processed_time, processed_firing_rates)
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
        mean_bump_width = np.rad2deg(np.mean(self.gaussian_width*np.pi/8))
        std_bump_width = np.rad2deg(np.std(self.gaussian_width*np.pi/8))
        
        if mean_bump_width >= 360:
            print('\033[91m' + f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]' + '\033[0m')
        else:
            print(f'Average bump width: {mean_bump_width:.3f} ± {std_bump_width:.3f} [deg]')
        
        print(f'Average firing rate: {np.mean(self.gaussian_amplitude):.3f} ± {np.std(self.gaussian_amplitude):.3f} [Hz]')
        
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
        
        
def plot_results(time_points, firing_rates, title=None, file_name=None, region='EB', 
              y_label='Time (s)', cmap='Blues', save=False, folder='figures', 
              plot_gaussian=True, figsize=(10, 2.5)):
    """Plot neural activity data.
    
    Args:
        time_points: Array of time points
        firing_rates: Matrix of firing rates (neurons x time)
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
    plt.pcolormesh(time_points, range(firing_rates.shape[0]), firing_rates, 
                  cmap=cmap, shading='nearest')
    plt.colorbar(label='Firing Rate [Hz]')
    plt.xlabel(y_label)
    plt.ylabel('EB region' if region == 'EB' else 'Neuron ID')
    plt.yticks([0, 4, 11, 15], ['R8', 'R4', 'L4', 'L8'] if region == 'EB' else [0, 5, 10, 15])
    plt.title(title)

    if plot_gaussian:
        g_t, g_x, g_y, g_w = utils.gau_fit(time_points, firing_rates)
        plt.plot(g_t, g_x, 'r', linewidth=3)
        
    if save:
        plt.savefig(os.path.join(folder, file_name))
        plt.close()
        
if __name__ == '__main__':
    network = RingAttractorNetwork()
    network.run_simulation(stimulus_strength=0.05, stimulus_location=0.0, shifter_strength=0.015, half_PEN='right')
    network.process_data()
    network.save_results(file_path='simulation_results.dat', folder='results')
    network.visualize_activity(title='Activity', file_name='activity.png', region='EB', save=True, folder='figures')
    

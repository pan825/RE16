import matplotlib.pyplot as plt
import fit
import os

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
        g_t, g_x, g_y, g_w = fit.gau_fit(t, fr)
        plt.plot(g_t, g_x, 'r', linewidth=3)
        
    if save:
        plt.savefig(os.path.join(folder, file_name))
        plt.close()
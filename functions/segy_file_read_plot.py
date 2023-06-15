
import segyio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from segysak.segy import segy_header_scan
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib


def segy_file(file_name='Synth_seismic_1.segy', color_map='seismic'):
    """The function reads SEGY file of 2D cross-section, makes a plot for Seismic data, its attributes, Facies.
    In addition, defines Extent of the cross-section

    Parameters:
        file_name (str): Specify the name of the SEGY-file. Defaults to 'Synth_seismic_1.segy'.
        color_map (str): Specify the color_map. Defaults to 'seismic'.

    Returns:
        data_file (2D array): 2D numpy array of cross-section
        extent (list): List with depth and trace numbers for the plots
    """


    with segyio.open(file_name, ignore_geometry=True) as file:
        # Get basic attributes
        n_traces = file.tracecount
        sample_rate = segyio.tools.dt(file) / 1000
        n_samples = file.samples.size
        twt = file.samples
        data_file = file.trace.raw[:]  # Get all data into memory (could cause on big files)
        # Load headers
        bin_headers = file.bin
    f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
    d = data_file.flatten()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    extent = [1, n_traces, -twt[-1], -twt[0]]  # define extent
    
    # Choose color_map
    if color_map =='seismic':
        # Customise color map for sesimic
        custom_norm=colors.TwoSlopeNorm(vmin=min(d), vcenter=0, vmax=max(d))
        im = ax.imshow(data_file.T, origin='upper', cmap="seismic", extent=extent, aspect='auto', norm=custom_norm)
        fig.colorbar(im)
        #ax.set_title('Seismic section')
        
        
    elif color_map == 'facies':
        # Customise color map for facies
        facies_name = ['','Coarse Sand', 'Sand', 'Fine Sand', 'Shale', 'Carbonate']
        facies_color = ['#FFFFFF' ,'#E69076', '#FFFF00', '#FFCC00', '#A6A6A6', '#8080FF']
        cmap = matplotlib.colors.ListedColormap(facies_color)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(data_file.T, cmap=cmap, aspect='auto', extent=extent, norm=norm, vmin=0-1.5, vmax=4+0.5)
        fig.colorbar(im, ticks=np.arange(0, 5))
        #ax.set_title('Facies section')
       
        
    else:    
        im = ax.imshow(data_file.T, cmap="jet", vmin=min(d), vmax=max(d), aspect='auto', extent=extent)
        fig.colorbar(im)
        ax.set_title('')
    
    colormap1 = fig.axes[1] 
    colormap1.tick_params(labelsize=12)    
    ax.set_xlabel('trace number', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylabel('TWT [ms]', fontsize=14)
    plt.show()

    return data_file, extent
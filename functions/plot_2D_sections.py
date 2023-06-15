

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib




def plot_2D_section(data_file, extent_plot,  color_map='seismic', number_of_facies=5, list_of_wells=None):
    """This function plot 2D cross-section after non-existing  changed by NaN

    Args:
        data_file (DataFrame): cleaned 2D section
        extent_plot (list): List with depth and trace numbers for the plots
        color_map (str): Specify the color_map. Defaults to 'seismic'. Defaults to 'seismic'.
        number_of_facies (int): Specify the number of Facies. Defaults to 5.
        
    Returns:
        None 
    """
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    # d = np.array(data_file)
    
    data = data_file
    extent = extent_plot
    d = np.array(data_file).flatten()
    
    if color_map =='seismic':
        custom_norm=colors.TwoSlopeNorm(vmin=min(d), vcenter=0, vmax=max(d))
        im = ax.imshow(data.T, origin='upper', cmap="seismic", extent=extent, aspect='auto', norm=custom_norm)
        fig.colorbar(im)
        
        
    elif color_map == 'facies':
        facies_name = ['','Coarse Sand', 'Sand', 'Fine Sand', 'Shale', 'Carbonate']
        facies_color = ['#FFFFFF' ,'#E69076', '#FFFF00', '#FFCC00', '#A6A6A6', '#8080FF']
        # cmap = matplotlib.colors.ListedColormap(['black', 'fuchsia', 'yellow', 'cyan', 'orange', 'red'])
        cmap = matplotlib.colors.ListedColormap(facies_color)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(data_file.T, cmap=cmap, aspect='auto', extent=extent, norm=norm, vmin=0-1.5, vmax=4+0.5)
        # plt.colorbar(im, cmap=cmap, norm=norm, boundaries=bounds)
        fig.colorbar(im, ticks=np.arange(0, number_of_facies))
        # ax.set_title('Facies', fontsize=18)
        
    elif color_map == 'seis_inv':
        im = ax.imshow(data, cmap="gist_rainbow", aspect='auto', extent=extent)
        fig.colorbar(im)
        
        
        
    else:    
        im = ax.imshow(data, cmap="jet", aspect='auto', extent=extent)
        fig.colorbar(im)
        
    if list_of_wells==None:
        pass
    else:    
        for well in list_of_wells:
            ax.axvline(x = well, linewidth = 1, color ='black')
        
    
    colormap1 = fig.axes[1] 
    colormap1.tick_params(labelsize=12)
    
    ax.set_xlabel('trace number', fontsize=14)
    ax.set_ylabel('TWT [ms]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.show()
    
    
    
    
    

# Difference map

def difference_map(df_facies_comparison, facies_predicted, extent, list_of_wells=None):
    """ Function to plot difference map between the ground-truth and predicted facies.

    Args:
        df_facies_comparison (2D array): ground truth facies
        facies_predicted (2D array): predicted facies from ML models
        extent (list): List with depth and trace numbers for the plots 
        list_of_wells (list): the list with wells location. Defaults to None.
        
    Returns:
        nothing

    """
     
    
    import matplotlib.pyplot as plt
    import matplotlib.colors  
     
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    extent = extent # define extent
    # The difference
    np_facies_fact = df_facies_comparison
    np_facies_predicted = facies_predicted
    
    facies_difference = np.subtract(np_facies_fact, np_facies_predicted)
    
    df1 = pd.DataFrame(facies_difference)
    df1 = df1.apply(np.sign).replace({-4:1, -3:1, -2:1, -1:1, 0:0, 
                                1:1, 2:1, 3:1, 4:1
                                })
    facies_difference_result = np.array(df1)
    
    

    cmap = matplotlib.colors.ListedColormap(['green', 'red'])
    im = ax.imshow(facies_difference_result, cmap=cmap, vmin=0, vmax=1, aspect='auto', extent=extent)
    
    if list_of_wells==None:
        pass
    else:    
        for well in list_of_wells:
            ax.axvline(x = well, linewidth = 1, color ='black')

    ax.set_xlabel('trace number', fontsize=14)
    ax.set_ylabel('TWT [ms]', fontsize=14)
    ax.set_title('Difference map', fontsize=16)
    colormap1 = fig.axes[0] 
    colormap1.tick_params(labelsize=12)
    plt.colorbar(im, ticks=[True, False])
    plt.legend()
    plt.show()


from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd




def plot_3D_cube(data, color_map='facies', number_of_facies=5, number_of_wells=7):
    x = np.indices(data.shape)[0]
    y = np.indices(data.shape)[1]
    z = np.indices(data.shape)[2]
    col = data.flatten()
    
    # 3D Plot
    fig = plt.figure(figsize=(10, 6))
    ax3D = fig.add_subplot(projection='3d')
    # ax3D = plt.axes(projection='3d')
    

        
    # We will randomly choose 7 wells for training
    # It should not cross faults
    start_x = [10, 60, 30, 25, 65, 50, 21]
    start_y = [15, 15, 20, 58, 60, 50, 35]
    start_z = [700, 700, 700, 700, 700, 700, 700]

    end_x = [10, 60, 30, 25, 65, 50, 21]
    end_y = [15, 15, 20, 58, 60, 50, 35]
    end_z  =[0, 0, 0, 0, 0, 0, 0]
    
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')
    
    if number_of_wells == 7:
        for well in range(number_of_wells):
            ax3D.plot([start_x[well], end_x[well]], [start_y[well], end_y[well]],  zs=[start_z[well], end_z[well]], color='black', linewidth = 1)
    
    elif number_of_wells == None:
        pass
    
    
    if color_map == 'facies':
        
        if number_of_facies == 5: 
            facies_name = ['','Coarse Sand', 'Sand', 'Fine Sand', 'Shale']
            facies_color = ['#FFFFFF' ,'#E69076', '#FFFF00', '#FFCC00', '#A6A6A6', '#8080FF']
            cmap = matplotlib.colors.ListedColormap(facies_color)
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            # # 3D Plot
            # fig = plt.figure()
            # ax3D = fig.add_subplot(projection='3d')
            # p3d = ax3D.scatter(x, y, z)   
            # p3d = ax3D.scatter(x, y, z, c =data, cmap=cmap)        
            p3d = ax3D.scatter(x, y, z, c=data, cmap=cmap)                                                                        
            # fig.colorbar(p3d, ticks=np.arange(0, 5))
            fig.colorbar(p3d)
            
        # im = ax.imshow(data.T, cmap=cmap, aspect='auto', extent=extent, norm=norm, vmin=0-1.5, vmax=4+0.5)
        # # plt.colorbar(im, cmap=cmap, norm=norm, boundaries=bounds)
        # fig.colorbar(im, ticks=np.arange(0, 5))
        # ax.set_title('Facies section')
        
        if number_of_facies == 4: 
            facies_name = ['Coarse Sand', 'Sand', 'Fine Sand', 'Shale']
            facies_color = ['#E69076', '#FFFF00', '#FFCC00', '#A6A6A6', '#8080FF']
            cmap = matplotlib.colors.ListedColormap(facies_color)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            # # 3D Plot
            # fig = plt.figure()
            # ax3D = fig.add_subplot(projection='3d')
            # p3d = ax3D.scatter(x, y, z)   
            # p3d = ax3D.scatter(x, y, z, c =data, cmap=cmap)        
            p3d = ax3D.scatter(x, y, z, c=data, cmap=cmap)                                                                        
            fig.colorbar(p3d, ticks=np.arange(0, 5))
                   
    elif color_map == 'relai':
        # # 3D Plot
        # fig = plt.figure()
        # ax3D = fig.add_subplot(projection='3d')
        # p3d = ax3D.scatter(x, y, z)   
        # p3d = ax3D.scatter(x, y, z, c =data, cmap=cmap)        
        p3d = ax3D.scatter(x, y, z, c=data, cmap='jet')                                                                        
        # fig.colorbar(p3d, ticks=np.arange(0, 5))
        fig.colorbar(p3d)
    
    else:
        # fig = plt.figure()
        # ax3D = fig.add_subplot(projection='3d')
        # p3d = ax3D.scatter(x, y, z)   
        # p3d = ax3D.scatter(x, y, z, c =data, cmap=cmap)        
        p3d = ax3D.scatter(x, y, z, c=data, cmap='jet')                                                                        
        # fig.colorbar(p3d, ticks=np.arange(0, 5))
        fig.colorbar(p3d)
             
    plt.show()
    
    
    
    
    
    
    
    
#############################################################################################
def difference_map_3D(df_facies_comparison, facies_predicted, number_of_wells=7):
    d, e, f = np.shape(df_facies_comparison)
    facies_difference_map = np.zeros((d, e, f))
    facies_difference_map[:] = np.NaN
    
    
    # The difference map
    np_facies_fact = df_facies_comparison
    np_facies_predicted = facies_predicted
    
    
    facies_difference = np.subtract(np_facies_fact, np_facies_predicted)
    # np.unique(facies_difference)
    facies_difference_map = np.where(((facies_difference >= 1) | (facies_difference <= -1)),  1, facies_difference)
    
    
    x = np.indices(df_facies_comparison.shape)[0]
    y = np.indices(df_facies_comparison.shape)[1]
    z = np.indices(df_facies_comparison.shape)[2]
    col = df_facies_comparison.flatten()
    
    # 3D Plot
    fig = plt.figure(figsize=(8, 8))
    #fig1=plt.figure(figsize=(8,5))
    ax3D = fig.add_subplot(projection='3d')

    

    start_x = [10, 60, 30, 25, 65, 50, 21]
    start_y = [15, 15, 20, 58, 60, 50, 35]
    start_z = [700, 700, 700, 700, 700, 700, 700]

    end_x = [10, 60, 30, 25, 65, 50, 21]
    end_y = [15, 15, 20, 58, 60, 50, 35]
    end_z  =[0, 0, 0, 0, 0, 0, 0]
    
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')
    
    for well in range(number_of_wells):
        ax3D.plot([start_x[well], end_x[well]], [start_y[well], end_y[well]],  zs=[start_z[well], end_z[well]], color='black', linewidth = 1)
    
    
    cmap = matplotlib.colors.ListedColormap(['green', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)       
    p3d = ax3D.scatter(x, y, z, c=facies_difference_map, cmap=cmap)                                                                        
    # fig.colorbar(p3d, ticks=np.arange(0, 5))
    fig.colorbar(p3d, ticks=np.arange(0, 2))   
    plt.show()
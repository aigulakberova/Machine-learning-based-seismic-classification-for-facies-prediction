
import numpy as np
import pandas as pd



def replace_nonexisting_data_with_NaN(df, nonexisting_data):
    """This function removes data that are out of reservoir (for example, for depths it is data that =250)
    and replaces with NaN.

    Args:
        df (DataFrame): DataFrame 
        nonexisting_data (float): nonexisting data that have to be replaced by NaN

    Returns:
        df (DataFrame): DataFrame without nonexisting data
    """
    df = df.replace(nonexisting_data, np.NaN)
    return df





def take_data_that_exist(df, df2):
    """THis function removes data that are out of reservoir by filtering by another data

    Args:
        df (DataFrame): Data that need to be filtered
        df2 (DataFrame): Filter

    Returns:
        map_df (DataFrame): Filtered DataFrame that contains data within the reservoir
    """
    r, c = np.shape(df)
    map_df = np.zeros((r, c))
    map_df[:] = np.NaN

    # for every trace (column)
    for every_col in range(len(df.columns)):
        non_empty_rows = df2.iloc[:,every_col].dropna(axis=0)

        # Filter 
        map_df[non_empty_rows.index, every_col] = df.iloc[non_empty_rows.index, every_col]

    return pd.DataFrame(map_df)   # returns df




def standartization(df, value_to_drop=True):

    '''
    Implement standartization to a dataset.
    When calculating MEAN and STD excludes the values that belong to empty cells (in case, they were not removed earlier)

    Parameters:
        df (DataFrame)
        value_to_drop (float): specify the value that is outside reservoir
        

    Returns:
        df_standard (DataFrame): data after implementing Standartization
    '''
    df_array = df.values.flatten()
    # assign 'values_to_drop' a value
    if value_to_drop == True: 
        to_drop = df.iloc[0,0]
        # calculate mean of data after excluding values_to_drop from the data
        df_mean = np.mean(df_array[(df_array != to_drop)])
        # calculate STD
        df_STD = np.std(df_array[(df_array != to_drop)])
        # standartization
        df_standard = pd.DataFrame((df.values - df_mean) / df_STD)
    else:
        df_standard = pd.DataFrame((df - np.nanmean(df.values)) / np.nanstd(df.values))

    # return df_standard.add_prefix(str(feature_name))
    return df_standard




###########################################################################################################
def standartization_3D(feature_3D_array):

    '''
    Implement standartization to a dataset.
    When calculating MEAN and STD excludes the values that belong to empty cells.

    Parameters:

    Returns:
    '''
    # Flatten 3D array (to 1D)
    feature_array_1d = feature_3D_array.reshape(-1)
    
    # Calculate STD
    std_all = np.nanstd(feature_array_1d)
    
    # Calculate Mean
    mean_all = np.nanmean(feature_array_1d)
    
    # Apply Standartization
    feature_std_3D = (feature_3D_array - mean_all) / std_all
    
    return feature_std_3D



def df_wells_from_section(df, wells_list):

    '''
    returns DataFrame with particular columns
    
    Parameters:
        df (DataFrame)
        wells_list (list): list of traces (wells)
    
    Returns:
        DataFrame with particular columns
    '''
    df_wells = df.iloc[:,wells_list]
    return df_wells





################################################################################################################
# Extract properties for wells
# We have wells coordinates
def extract_wells_with_data_3D(data_3D, x_coord_wells, y_coord_wells):
    wells_list = []
    
    for well_x in x_coord_wells:
        for well_y in y_coord_wells:
        
            each_well = data_3D[x_coord_wells, y_coord_wells, :]
            
    wells_list.append(each_well)
        
    # Convert to list and Remove nan from well_all
    well_3d_array = np.asarray(wells_list)
    well_2d_array = np.reshape(well_3d_array, (len(x_coord_wells), data_3D.shape[2]))
    # wells_1d_array = well_2d_array.flatten()
    
    return well_2d_array
        

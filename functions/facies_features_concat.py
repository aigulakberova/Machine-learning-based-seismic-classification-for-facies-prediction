

import pandas as pd


def concat_features_RelAI_Seis_SeisInv(df_facies_wells, feature_list_RelAI_Seis_SeisInv):
    """The function concatenate features (RelAI, Seismic, Seismic Inversion) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_SeisInv (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 3 features (Relative AI, Seismic, Seis_inversion)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2 = create_empty_lists(len(feature_list_RelAI_Seis_SeisInv))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_SeisInv[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_SeisInv[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_SeisInv[2].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'seis_inv'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features




#######################################################################################################################
def concat_features_Seis_SeisInv(df_facies_wells, feature_list_Seis_SeisInv):
    """The function concatenate features (RelAI, Seismic, Seismic Inversion) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_SeisInv (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 2 features (Seismic and Seismic Inversion)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1 = create_empty_lists(len(feature_list_Seis_SeisInv))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_Seis_SeisInv[0].iloc[:,every_well]
        X1 = feature_list_Seis_SeisInv[1].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1], axis=1)

    features = features.rename(columns = {0:'seis', 
                                          1:'seis_inv'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features






def concat_features_RelAI_Seis_Envel_InstFreq(df_facies_wells, feature_list_RelAI_Seis_Envel_InstFreq):
    """The function concatenate features (RelAI, Seismic, Envelope, InstFreq) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_Envel_InstFreq (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 4 features (Relative AI, 
        Seismic, Envelope, Instantaneous Frequency)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2, XX3 = create_empty_lists(len(feature_list_RelAI_Seis_Envel_InstFreq))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_Envel_InstFreq[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_Envel_InstFreq[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_Envel_InstFreq[2].iloc[:,every_well]
        X3 = feature_list_RelAI_Seis_Envel_InstFreq[3].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)
        XX3.append(X3)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)
    XX3 = pd.concat(XX3, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2, XX3], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'envel',
                                          3: 'inst_freq'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features





##########################################################################################

def concat_features_RelAI_Seis_SeisInv_Depth(df_facies_wells, feature_list_RelAI_Seis_SeisInv_Depth):
    """The function concatenate features (RelAI, Seismic, Seismic Inversion) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_SeisInv (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 4 features (Relative AI,
        Seismic, Seismic Inversion, Geological Time)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2, XX3 = create_empty_lists(len(feature_list_RelAI_Seis_SeisInv_Depth))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_SeisInv_Depth[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_SeisInv_Depth[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_SeisInv_Depth[2].iloc[:,every_well]
        X3 = feature_list_RelAI_Seis_SeisInv_Depth[3].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)
        XX3.append(X3)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)
    XX3 = pd.concat(XX3, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2, XX3], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'seis_inv',
                                          3: 'depth'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features



################################################################################################

def concat_features_RelAI_Seis_Envel_InstFreq_Depth(df_facies_wells, feature_list_RelAI_Seis_Envel_InstFreq_Depth):
    """The function concatenate features (RelAI, Seismic, Envelope, InstFreq) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_Envel_InstFreq (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 5 features (Relative AI, 
        Seismic, Envelope, Instant Frequency, Geological Time)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2, XX3, XX4 = create_empty_lists(len(feature_list_RelAI_Seis_Envel_InstFreq_Depth))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_Envel_InstFreq_Depth[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_Envel_InstFreq_Depth[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_Envel_InstFreq_Depth[2].iloc[:,every_well]
        X3 = feature_list_RelAI_Seis_Envel_InstFreq_Depth[3].iloc[:,every_well]
        X4 = feature_list_RelAI_Seis_Envel_InstFreq_Depth[4].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)
        XX3.append(X3)
        XX4.append(X4)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)
    XX3 = pd.concat(XX3, ignore_index=False, axis=0)
    XX4 = pd.concat(XX4, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2, XX3, XX4], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'envel',
                                          3: 'inst_freq',
                                          4: 'depth'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features







################################################################################################

def concat_features_RelAI_Seis_SeisInv_SpecDec(df_facies_wells, feature_list_RelAI_Seis_SeisInv_SpecDec):
    """The function concatenate features (RelAI, Seismic, Envelope, InstFreq) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_Envel_InstFreq (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 5 features (Relative AI, 
        Seismic, Envelope, Instant Frequency, Geological Time)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2, XX3 = create_empty_lists(len(feature_list_RelAI_Seis_SeisInv_SpecDec))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_SeisInv_SpecDec[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_SeisInv_SpecDec[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_SeisInv_SpecDec[2].iloc[:,every_well]
        X3 = feature_list_RelAI_Seis_SeisInv_SpecDec[3].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)
        XX3.append(X3)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)
    XX3 = pd.concat(XX3, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2, XX3], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'seis_inv',
                                          3: 'spec'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features






################################################################################################

def concat_features_RelAI_Seis_Envel_InstFreq_SeisInv(df_facies_wells, feature_list_RelAI_Seis_Envel_InstFreq_SeisInv):
    """The function concatenate features (RelAI, Seismic, Envelope, InstFreq) with Facies

    Args:
        df_facies_wells (DataFrame): Facies
        feature_list_RelAI_Seis_Envel_InstFreq (lisr): list of features 
        
    Returns:
        facies_and_features (DataFrame): concatenated dataframe with facies and 5 features (Relative AI, 
        Seismic, Envelope, Instant Frequency, Geological Time)
    """
    def create_empty_lists(a):
        for i in range(a):
            yield []

    XX0, XX1, XX2, XX3, XX4 = create_empty_lists(len(feature_list_RelAI_Seis_Envel_InstFreq_SeisInv))
    YY1 = []

    for every_well in range(len(df_facies_wells.columns)):

        X0 = feature_list_RelAI_Seis_Envel_InstFreq_SeisInv[0].iloc[:,every_well]
        X1 = feature_list_RelAI_Seis_Envel_InstFreq_SeisInv[1].iloc[:,every_well]
        X2 = feature_list_RelAI_Seis_Envel_InstFreq_SeisInv[2].iloc[:,every_well]
        X3 = feature_list_RelAI_Seis_Envel_InstFreq_SeisInv[3].iloc[:,every_well]
        X4 = feature_list_RelAI_Seis_Envel_InstFreq_SeisInv[4].iloc[:,every_well]
        
        Y = df_facies_wells.iloc[:,every_well]

        XX0.append(X0)
        XX1.append(X1)
        XX2.append(X2)
        XX3.append(X3)
        XX4.append(X4)

        YY1.append(Y)

    XX0 = pd.concat(XX0, ignore_index=False, axis=0)
    XX1 = pd.concat(XX1, ignore_index=False, axis=0)
    XX2 = pd.concat(XX2, ignore_index=False, axis=0)
    XX3 = pd.concat(XX3, ignore_index=False, axis=0)
    XX4 = pd.concat(XX4, ignore_index=False, axis=0)

    facies = pd.concat(YY1, ignore_index=False)

    features = pd.concat([XX0, XX1, XX2, XX3, XX4], axis=1)

    features = features.rename(columns = {0:'relai', 
                                          1:'seis', 
                                          2: 'envel',
                                          3: 'inst_freq',
                                          4: 'seis_inv'
                                          })
    features

    facies = pd.DataFrame(facies).rename(columns={0:'facies'})
    facies

    facies_and_features = pd.concat([facies, features], axis=1)
    facies_and_features = facies_and_features.dropna(axis=0)
    
    return facies_and_features
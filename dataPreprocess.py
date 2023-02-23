from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def normalize_df(df, cols = ['latitude', 'longitude', 'depth', 'time'], scalerType = StandardScaler, ):
    """
    Normalized tabular data in specified columns of a dataframe

    ...

    Parameters
    ----------
    df : pandas dataframe 
        dataframe containing tabular data, with columns including [latitude, longtidue, depth]
    
    cols : list of str
        specifies columns in the dataframe to be normalized
    
    scalerType : class in sklearn.preprocessing
        Select between StandardScaler (unit variance, zero mean) or MinMaxScaler (rescale to range such as [0,1])

    """

    if not set(cols).issubset(df.columns):
        print('Some columns missing in dataframe among: ')
        print(*usefulColumns, sep = ", ")
        return
    
    cols_copy = cols.copy()
    # convert timecodes to year and month columns
    if 'time' in cols_copy:
        cols_copy.remove('time')
        datetimes = pd.to_datetime(df['time'])
        df['year'] = datetimes.dt.year
        cols_copy.append('year')
        
        df['month'] = datetimes.dt.month
        df['month_cyclic'] = 7 - abs(df['month'] - 7) # january -> 1, july ->7, august -> 6, december -> 2
        cols_copy.append('month_cyclic')
    
    data = df[cols_copy]
    scaler = scalerType()
    scaler.fit(data)
    normalized_columns = [s + '_norm' for s in cols_copy]
    df[normalized_columns] = scaler.transform(df[cols_copy])
    
    print(f'Dataframe has length {df.shape[0]}')
    print(f'Number of unique boreholes is {df.borehole.nunique()}')
    print(f'Latitude ranges from {df.latitude.min()} to {df.latitude.max()}')
    print(f'Longitude ranges from {df.longitude.min()} to {df.longitude.max()}')
    
    return scaler, normalized_columns

def filter_df_visibile_ice(df):
    
    """
    Prepare visible_ice column of dataframe, and generate visible_ice_code column
    """
    
    df['visible_ice'].replace(['None'], 'No visible ice', regex=True, inplace=True)
    print('visible_ice: \'None\' entries have been replaced by \'No visible ice\'')
    
    ordered_ice = ['No visible ice', 'Low', "Medium to high", 'High', 'Pure ice']
    df['visible_ice'] = pd.Series(pd.Categorical(df['visible_ice'], categories=ordered_ice, ordered=True))
    
    print('visible_ice column entries has been ordered:')
    print(df['visible_ice'].unique())
    
    df['visible_ice_code'] =  df['visible_ice'].cat.codes
    print('with corresponding codes in visible_ice_code column:')
    print(df['visible_ice_code'].unique())
    
    dm_visible_ice = pd.get_dummies(df.visible_ice)
    df['visible_ice_binary'] = (~dm_visible_ice['No visible ice'].astype(bool)).astype(int)
    print('visible_ice: binary column generated')
    
def filter_df_materials(df):
    
    """
    Prepare materials column of dataframe, and generate material_ice column with binary values indicating whether the material is ice
    """
    
    df['materials'].replace(['ICE'], 'Ice', regex=True, inplace=True)
    df['materials'].replace(['ice'], 'Ice', regex=True, inplace=True)
    print('materials: \'ICE\' and \'ice\' entries has been standardized into \'Ice\'')
    
    dm_materials = pd.get_dummies(df.materials)
    df['material_ice'] = dm_materials['Ice']
    print('\'material_ice\' column generated')

def create_chips_geo90(df, fpath, output_path):
    for for index, row in df.iterrows():
        
        lat_index_start = np.round((self.base_lat - lat) / pixel_len - self.chip_size/2).astype(int)
        lat_index_end = lat_index_start + self.chip_size
        
        lng_index_start = np.round((lng - self.base_lng) / pixel_len - self.chip_size/2).astype(int)
        lng_index_end = lng_index_start + self.chip_size
        
        image = self.preloaded[:, lat_index_start:lat_index_end,lng_index_start:lng_index_end]
        
    
        save_image(data['image'], os.path.join(data_root, 'chips', f'{i:04d}.png'))
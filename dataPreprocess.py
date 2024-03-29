from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from pyproj import Transformer
from osgeo import gdal, osr, ogr

def project_df(df):
    """
    Add epsg3413 x, y coordinates to dataframe, according to longitude latitude
    """
    # EPSG:3413
    # WGS 84 / NSIDC Sea Ice Polar Stereographic North
    transformer = Transformer.from_crs("epsg:4326", "epsg:3413")
    df['proj_x'], df['proj_y'] = transformer.transform(df.latitude,df.longitude)

def crop_df_hds(df):
    """
    filter dataframe to keep only boreholes inside hds_cropped area
    """
    
    ind = df[df['latitude'] > 68.692].index
    ind = ind.union(df[df['latitude'] <= 68.4881].index)
    ind = ind.union(df[df['longitude'] < -133.89358284312502].index)
    ind = ind.union(df[df['longitude'] > -133.443705313125].index)
    
    return df.drop(ind)
    



def crop_hds_discard_chips(df):
    # crops breohole areas to hds region, and discard areas where hds chips have too much invalid pixels
    df.borehole = df.borehole.str.replace('//', '--')

    file = r"C:\Users\mouju\Desktop\film\hds\geotiff_original\ldcorr_refine.tif"
    ds = gdal.Open(file)
    ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    df = crop_df_hds(df)
    
    df['borehole_group'] = df['borehole'].str.split('-').str[0]

    df=df.query("borehole_group != '2.45'")

    df=df.query("borehole_group != 'GSC3 '")

    df=df.query("borehole_group != 'GSC4 '")

    df=df.query("borehole != 'ENG.YARC03097-01--ITH-02'")
    df=df.query("borehole != 'ENG.YARC03097-01--ITH-03'")

    df=df.query("borehole != 'ENG.YARC03097-01--ITH-04'")

    df=df.query("borehole != 'W14103137-CR18N'")
    df=df.query("borehole != 'W14103137-CR18S'")
    df=df.query("borehole != 'W14103137-CR21N'")
    df=df.query("borehole != 'W14103137-CR21S'")
    print(f'Number of dataframe rows: {len(df)}')
    return df
    
def normalize_df(df, cols = ['latitude', 'longitude', 'depth', 'time'], scalerType = StandardScaler):
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
    
    print(f'List of columns normalized: {normalized_columns}')
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

    df['materials'].replace(['Pure Ice'], 'Ice', regex=True, inplace=True)
    
    df['materials'].replace(['Coarse till'], 'Till', regex=True, inplace=True)
    df['materials'].replace(['till'], 'Till', regex=True, inplace=True)
    df['materials'].replace(['Fine till'], 'Till', regex=True, inplace=True)
    df['materials'].replace(['Fine Till'], 'Till', regex=True, inplace=True)
    
    df['materials'].replace(['Cobbles'], 'Gravel', regex=True, inplace=True)
    df['materials'].replace(['Rock'], 'Gravel', regex=True, inplace=True)

    # -------------
    df['materials'].replace(['ICE'], 'Ice', regex=True, inplace=True)
    df['materials'].replace(['ice'], 'Ice', regex=True, inplace=True)
    print('materials: \'ICE\' and \'ice\' entries has been standardized into \'Ice\'')
    
    dm_materials = pd.get_dummies(df.materials)
    df['material_ice'] = dm_materials['Ice']
    print('\'material_ice\' column generated')

    
def prep_label_columns(df):
    
    """
    Prepare label columns: visible_ice, materials
    """
    #  Prepare visible_ice column of dataframe, and generate visible_ice_code column
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
    
    #Prepare materials column of dataframe, and generate material_ice column with binary values indicating whether the material is ice
    
    df['materials'].replace(['ICE'], 'Ice', regex=True, inplace=True)
    df['materials'].replace(['ice'], 'Ice', regex=True, inplace=True)
    print('materials: \'ICE\' and \'ice\' entries has been standardized into \'Ice\'')
    
    dm_materials = pd.get_dummies(df.materials)
    df['material_ice'] = dm_materials['Ice']
    print('\'material_ice\' column generated')
    
    df['materials'] = pd.Series(pd.Categorical(df['materials']))
    df['materials_code'] = df['materials'].cat.codes
    print("materials has been categorized into codes in materials_code")

def scale_column(column):
    scaler = StandardScaler()
    scaler.fit(column)
    return scaler.transform(column)
    
def prepare_df(df, list_cols, label):
    # prepare, and filter dataframe for specified label
    
    df['interval_length'] = df['bottom_of_interval'] - df['top_of_interval']
    df.borehole = df.borehole.str.replace('//', '--')
    project_df(df)
    df_scaler, list_of_columns_normalized = normalize_df(df, list_cols)
    prep_label_columns(df)
    
    df2 = df.dropna(subset=[label])
    print(f'Null entries of {label} dropped')
    
    n_classes = df2[label].nunique()
    return df2, n_classes
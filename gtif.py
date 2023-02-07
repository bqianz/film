from osgeo import gdal, osr, ogr # Python bindings for GDALa
import numpy as np

def create_geotiff(data, output_file, extent, data_type = gdal.GDT_Int16):
    
    """
    Convert numpy array to geotiff

    ...

    Parameters
    ----------
    data : numpy array 
        Array containing values to convert, dimension [nrows, ncols, nbands] or [nrows, ncols].
        Begin from top left corner of map, i.e smallest longitude, greatest latitude
    output_file: str
        Path of output tif file
    extent : list
        List of coordinates [longitude_min, latitude_min, longitude_max, latitude_max]
    data_type : gdal data type, optional
        Usually gdal.GDT_Int16 or gdal.GDT_Float32    
    """
    
    
    # Get GDAL driver GeoTiff
    driver = gdal.GetDriverByName('GTiff')

    # Get dimensions
    
    dim = [1,1,1]
    
    dim_n = len(data.shape)
    
    for n in range(dim_n):
        dim[n] = data.shape[n]
        
    for n in range(dim_n, 3):
        data = np.expand_dims(data, axis=-1)

    print(f'Dimensions of data is {dim[0]} rows, {dim[1]} columns, {dim[2]} layers')
    nlines = dim[0]
    ncols = dim[1]
    nbands = dim[2]
    

    # Create a temp grid
    #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
    grid_data = driver.Create('grid_data', ncols, nlines, nbands, data_type)#, options)

    # Write data for each bands
    for i in range(nbands):
        grid_data.GetRasterBand(i+1).WriteArray(data[:,:,i])

    # Lat/Lon WSG84 Spatial Reference System
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    # Setup projection and geo-transform
    grid_data.SetProjection(srs.ExportToWkt())
    
    def getGeoTransform(extent, nlines, ncols):
        resx = (extent[2] - extent[0]) / ncols
        resy = (extent[3] - extent[1]) / nlines
        return [extent[0], resx, 0, extent[3] , 0, -resy]

    grid_data.SetGeoTransform(getGeoTransform(extent, nlines, ncols))

    # Save the file
    print(f'Generated GeoTIFF: {output_file}')
    driver.CreateCopy(output_file, grid_data, 1)
    # third parameter: bStrict	TRUE if the copy must be strictly equivalent, 
    # or more normally FALSE indicating that the copy may adapt as needed for the output format. 

    # Close the file
    driver = None
    grid_data = None

    # Delete the temp grid                
    # os.remove('grid_data')

def read_geotiff(filename, bandId):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(bandId)
    arr = band.ReadAsArray()
    return arr, ds
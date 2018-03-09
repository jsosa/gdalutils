# inst: university of bristol
# auth: jeison sosa
# mail: j.sosa@bristol.ac.uk / sosa.jeison@gmail.com

import numpy as np
from osgeo import gdal
from osgeo import osr

def get_dataxy(filename,x,y,nx,ny):

    """ Import gdal part of data in a numpy array """
    
    ds   = gdal.Open(filename, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    data = np.float64(band.ReadAsArray(int(x),int(y),nx,ny))
    ds   = None
    return data

def get_data(filename):

    """ Import gdal data in a numpy array """
    
    ds   = gdal.Open(filename, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    data = np.float64(band.ReadAsArray())
    ds   = None
    return data

def get_geo(filename,proj4=None):

    """ Read geo info from raster file """

    ds   = gdal.Open(filename, gdal.GA_ReadOnly)
    bnd  = ds.GetRasterBand(1)
    noda = np.float64(bnd.GetNoDataValue())
    gt   = ds.GetGeoTransform()
    nx   = np.int64(ds.RasterXSize)
    ny   = np.int64(ds.RasterYSize)
    resx = np.float64(gt[1])
    resy = np.float64(gt[5])
    xmin = np.float64(gt[0])
    ymin = np.float64(gt[3]) + ny * resy
    xmax = np.float64(gt[0]) + nx * resx
    ymax = np.float64(gt[3])
    x    = np.linspace(xmin+resx/2,xmin+resx/2+resx*(nx-1),nx)
    y    = np.linspace(ymax+resy/2,ymax+resy/2+resy*(ny-1),ny)

    srs  = osr.SpatialReference()
    if proj4 == None:
        wkt  = ds.GetProjectionRef()
        srs.ImportFromWkt(wkt)
    else:
        srs.ImportFromProj4(proj4)

    ds   = None
    geo  = [xmin,ymin,xmax,ymax,nx,ny,resx,resy,x,y,srs,noda]
    return geo

def write_raster(myarray,myraster,geo,fmt,nodata):

    """ Write an array to Raster format defaul in GTiff format"""

    # available data types:
    if fmt == "Byte":
        ofmt = gdal.GDT_Byte
    elif fmt == "Float32":
        ofmt = gdal.GDT_Float32
    elif fmt == "Float64":
        ofmt = gdal.GDT_Float64
    elif fmt == "Int16":
        ofmt = gdal.GDT_Int16
    elif fmt == "Int32":
        ofmt = gdal.GDT_Int32

    xmin = geo[0]
    ymin = geo[1]
    xmax = geo[2]
    ymax = geo[3]
    nx   = geo[4]
    ny   = geo[5]
    resx = geo[6]
    resy = geo[7]
    x    = geo[8]
    y    = geo[9]
    srs  = geo[10]

    driver = gdal.GetDriverByName('GTiff')

    outRaster = driver.Create(myraster, int(nx), int(ny), 1, ofmt, ['COMPRESS=LZW'])
    outRaster.SetGeoTransform((xmin, resx, 0, ymax, 0, resy))
    outRaster.SetProjection(srs.ExportToWkt())
    
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(myarray)
    outband.FlushCache()
    outband.SetNoDataValue(nodata)
    
def clip_raster(fraster,xmin,ymin,xmax,ymax):

    geo = get_geo(fraster)
    
    xx = geo[8]
    yy = geo[9]

    xind0 = abs(xx-xmin).argmin()
    xind1 = abs(xx-xmax).argmin()+1 # inclusive slicing
    xind  = range(xind0,xind1)
    nx    = len(xind)
    x     = xx[xind] # returns degrees

    yind0 = abs(yy-ymax).argmin()
    yind1 = abs(yy-ymin).argmin()+1 # inclusive slicing
    yind  = range(yind0,yind1)
    ny    = len(yind)
    y     = yy[yind] # returns degrees

    newras = get_dataxy(fraster,xind0,yind0,nx,ny)

    # since this geoinfo was created intentionally, xmin and ymax should be
    # modified before writing. Removes error of 0.004 degress in final raster
    diffx = (geo[8][2]-geo[8][1])/2.
    diffy = (geo[9][2]-geo[9][1])/2.

    # find xmin, ymin, xmax and ymax for
    xmin = xx[xind0] - diffx
    xmax = xx[xind1-1]
    ymax = yy[yind0] - diffy
    ymin = yy[yind1-1]

    # create geo data for clipped raster
    resx   = geo[6]
    resy   = geo[7]
    srs    = geo[10]

    newgeo = [xmin,ymin,xmax,ymax,nx,ny,resx,resy,x,y,srs]

    return newras,newgeo

def to_xarray(filename):

    import xarray as xr

    dat = get_data(filename)
    geo = get_geo(filename)

    # Replace NaN values
    dat[dat==geo[11]] = np.nan

    var = 'myvar' 
    data = dat
    x = geo[8]
    y = geo[9]
    
    foo = xr.Dataset()
    foo[var] = (('y','x'), data)
    foo.coords['x'] = (('x'), x)
    foo.coords['y'] = (('y'), y)
    
    return foo

def to_pandas(filename,nodata=None):

    import pandas as pd

    dat = get_data(filename)
    geo = get_geo(filename)

    if nodata == None:
        nodata = geo[11]
        iy,ix = np.where(dat!=nodata)
    else:
        iy,ix = np.where(dat!=nodata)

    idx = np.ravel_multi_index((iy,ix),dat.shape)
    X_flat = geo[8][ix]
    Y_flat = geo[9][iy]
    dat_flat = dat.flatten()[idx]

    df = pd.DataFrame({'x':X_flat,
                       'y':Y_flat,
                       'z':dat_flat})

    return df

def to_ascii(filename,output,nodata=None,sep=None):

    df = to_pandas(filename,nodata=nodata)

    if sep != None:
        df.to_csv(output,index=False,sep=sep)
    else:
        df.to_csv(output,index=False)
    

def plot(filename,width=5,height=5,dpi=72,title=None):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    dat = get_data(filename)
    geo = get_geo(filename)

    dat[dat==geo[11]]=np.nan

    fig, ax = plt.subplots(figsize=(width,height), dpi=dpi)
    myplot=ax.imshow(dat, interpolation='none')
    
    if title is not None:
        plt.title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.1)
    plt.colorbar(myplot, cax=cax)

def save(filename,out,tile='no',width=5,height=5,dpi=72):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plot(filename,width=width,height=height,dpi=dpi)
    if tile == 'yes':
        plt.imsave(out, dat, dpi=dpi)
    else:
        plt.savefig(out, bbox_inches='tight')


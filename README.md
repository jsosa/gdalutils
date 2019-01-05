# gdalutils

`gdautils` is a small library to handle GDAL-based raster files

### Installation

Just run this line after installing all dependencies

``` pip install git+https://github.com/jsosa/gdalutils.git```

### Dependencies

- [Geopandas](http://geopandas.org/)
- [GDAL](https://www.gdal.org/)

### Usage

The module is loaded via

```python
import gdalutils as gu
```

To read a raster directly from disk it can be done by

```python
gu.get_data(filename)
```

It'll return a `numpy.array` object

Geographical information can be read via

```python
gu.get_geo(filename)
```

it'll return a Pyhton `list` containing

1. xmin
2. ymin
3. xmax
4. ymax
5. number of cells `x` direction
6. number of cells `y` direction
7. resolution `x` direction
8. resolution `y` direction
9. an array containing center coordiantes of `x` cells
9. an array containing center coordiantes of `y` cells
10. projection
11. nodata value

Writing `numpy.array` object is posible by calling

```python
gu.write_raster(myarray, myraster, geo, fmt, nodata)
```

Where `myarray` is a `numpy.array` object, `myraster` is a filename output, `geo` is a list with geographical information identical to the one obtained with `gu.get_geo`, `fmt` is the format output: `'Float32'`, `'Float64'`, etc and `nodata` is the NODATA value

Passing from a `numpy.array` object to a `pandas.Dataframe` object is posible by

```python
array_to_pandas(dat, geo, val, symbol)
```

where `dat` is the n`umpy.array` object, `geo` is the list containing geographical information, `val` is a value to be masked and `symbol` is the logical symbol to be applied

Passing from pandas to array is also possible via

```python
pandas_to_array(df, geo, nodata)
```

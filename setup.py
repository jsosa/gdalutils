from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules=[
    Extension("gdalutils.extras.haversine", ["gdalutils/extras/haversine.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp","-O3"],
        extra_link_args=["-fopenmp"],
    )
]

setup(name='gdalutils',
      version='0.1',
      description='GDAL utils',
      url='http://github.com/jsosa/gdalutils',
      author='Jeison Sosa',
      author_email='sosa.jeison@gmil.com',
      license='MIT',
      packages=['gdalutils'],
      zip_safe=False,
      ext_modules=cythonize(ext_modules),
      )

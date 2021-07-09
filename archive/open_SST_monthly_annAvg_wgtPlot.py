# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:00:32 2020
기말고사 step 1
@author: current
"""
# load necessary modules
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#from scipy.stats import linregress
import scipy.stats as stats
import myFunc
from mpl_toolkits.basemap import Basemap


DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data1 = nc.Dataset(DATA_PATH + '/sst/sst.197901.nc', 'r')
lat = data1.variables['lat'][:]
lon = data1.variables['lon'][:]


# Import monthly SST data
styear = 1979  # start year
edyear = 2018  # end year

SSTs = np.zeros([12*40, len(lat), len(lon)])

k = 0
for i in range(styear, edyear+1):
    for j in range(1, 13):  # loop for months

        year = str(i)
        month = str(j).zfill(2)  # two digit string
        filename = f"/sst/sst.{year}{month}.nc"  # file name
        data = nc.Dataset(DATA_PATH + filename)  # read SST file
        SST_month = data.variables['sst'][:, :]
        SSTs[k, :, :] = SST_month[0, :, :]

        k += 1
    print(i)

#idxNaN2d = np.where(SSTs[0,:,:] < 0)
idxNaN = np.where(SSTs < 0)
SSTs[idxNaN] = np.NaN

#plt.contour(lon,lat,SSTs[0])
print(np.nanmax(SSTs))
print(np.nanmin(SSTs))

# Annual mean

SST_year = np.zeros([40, len(lat), len(lon)])
for i in range(0, 40):
    SST_year[i, :, :] = np.nanmean(SSTs[12*i:12*(i+1), :, :], 0)

# Cosine weighting & area average
avg_SST_year = myFunc.cos_agerage(SST_year,lat,lon)

# Plotting
years = np.arange(styear, edyear+1)
r = stats.linregress(years, avg_SST_year)
beta = r.slope
alpha = r.intercept
y = beta*years + alpha

plt.plot(years, avg_SST_year - 273, years, y - 273, 'r')

#plt.plot(years, avg_SST_year - 273)

kim = range(10)
for x in kim:
    print(x)

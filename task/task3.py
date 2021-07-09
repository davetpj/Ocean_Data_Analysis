import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
EVAPdir = DATA_PATH + 'EVAPintp/'

data1 = nc.Dataset(EVAPdir + 'EVAP.201801.nc')
lat = data1.variables['lat'][:]
lon = data1.variables['lon'][:]
time = data1.variables['time'][:]

styear = 1979
edyear = 2019

evap_year = np.zeros([(edyear-styear+1), len(lat), len(lon)])
nday_list1 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
nday_list2 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

evaps_list = []
for i in range(styear, edyear+1):

    if i % 4 != 0:
        nday = nday_list1[:]
    else:
        nday = nday_list2[:]

    for j in range(1, 13):
        year = str(i)
        month = str(j).zfill(2)
        filename = EVAPdir+'EVAP.'+year+month+'.nc'
        data = nc.Dataset(filename)
        evap_month = data.variables['e'][:, :]
        evap_year[i-styear, :, :] += evap_month[0, :, :]*nday[j-1]
        evaps_list = evap_year[i-styear, :, :]
        np.append(evaps_list, evap_month[0, :, :]*nday[j-1])

    evaps = np.array(evaps_list)


cosarray = np.zeros((len(lat), len(lon)))

for x in range(0, len(lon)):
    cosarray[:, x] = np.cos(lat * np.pi / 180)

wgt_evap_year0 = np.zeros(evap_year.shape)
for t in range(0, len(evap_year)):
    wgt_evap_year0[t, :, :] = evap_year[t, :, :]*cosarray

wgt_evap_year = np.sum(wgt_evap_year0, 2)
avg_evap_year = np.sum(wgt_evap_year, 1) / np.sum(cosarray)
plt.plot(range(styear, edyear+1), -avg_evap_year)

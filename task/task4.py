import numpy as np
import netCDF4 as nc
import scipy.stats as stats
import matplotlib.pyplot as plt

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
EVAPdir = DATA_PATH + 'EVAPintp/'
PRECdir = DATA_PATH + '/PRECintp/'

data1 = nc.Dataset(EVAPdir + 'EVAP.201801.nc')
data2 = nc.Dataset(PRECdir + 'PREC.201801.nc')
lat = data1.variables['lat'][:]
lon = data1.variables['lon'][:]
time = data1.variables['time'][:]

styear = 1979
edyear = 2019

evap_year = np.zeros([(edyear-styear+1), len(lat), len(lon)])
prec_year = np.zeros([(edyear-styear+1), len(lat), len(lon)])
nday_list1 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
nday_list2 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

evaps_list = []
precs_list = []
for i in range(styear, edyear+1):

    if i % 4 != 0:
        nday = nday_list1[:]
    else:
        nday = nday_list2[:]

    for j in range(1, 13):
        year = str(i)
        month = str(j).zfill(2)
        filename_e = EVAPdir+'EVAP.'+year+month+'.nc'
        filename_p = PRECdir+'PREC.'+year+month+'.nc'
        data_e = nc.Dataset(filename_e)
        data_p = nc.Dataset(filename_p)
        evap_month = data_e.variables['e'][:, :]
        prec_month = data_p.variables['tp'][:, :]
        evap_year[i-styear, :, :] += evap_month[0, :, :]*nday[j-1]
        prec_year[i-styear, :, :] += prec_month[0, :, :]*nday[j-1]
        evaps_list = evap_year[i-styear, :, :]
        precs_list = prec_year[i-styear, :, :]
        np.append(evaps_list, evap_month[0, :, :]*nday[j-1])
        np.append(precs_list, prec_month[0, :, :]*nday[j-1])

    evaps = np.array(evaps_list)
    precs = np.array(precs_list)


cosarray = np.zeros((len(lat), len(lon)))

for x in range(0, len(lon)):
    cosarray[:, x] = np.cos(lat * np.pi / 180)

wgt_evap_year0 = np.zeros(evap_year.shape)
wgt_prec_year0 = np.zeros(evap_year.shape)
for t in range(0, len(evap_year)):
    wgt_evap_year0[t, :, :] = evap_year[t, :, :]*cosarray
    wgt_prec_year0[t, :, :] = prec_year[t, :, :]*cosarray
wgt_evap_year = np.sum(-wgt_evap_year0, 2)
wgt_prec_year = np.sum(wgt_prec_year0, 2)
avg_evap_year = np.sum(wgt_evap_year, 1) / np.sum(cosarray)
avg_prec_year = np.sum(wgt_prec_year, 1) / np.sum(cosarray)


x = avg_prec_year
r = stats.linregress(x, avg_evap_year)

beta = r.slope
alpha = r.intercept

y = beta*x + alpha
plt.plot(avg_prec_year, avg_evap_year, 'go', avg_prec_year, y, 'r')

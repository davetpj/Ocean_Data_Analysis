import numpy as np
import netCDF4 as nc
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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
evap_year_a = np.zeros([(edyear-styear+1), len(lat), len(lon)])
prec_year = np.zeros([(edyear-styear+1), len(lat), len(lon)])
nday_list1 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
nday_list2 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

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
        evap_year[i-styear, :, :] -= evap_month[0, :, :]*nday[j-1]
        evap_year_a[i-styear, :, :] += evap_month[0, :, :]*nday[j-1]
        prec_year[i-styear, :, :] += prec_month[0, :, :]*nday[j-1]

    print(i)

evap_trend = np.zeros((len(lat), len(lon)))
evap_sig = np.zeros(np.shape(evap_trend))
prec_trend = np.zeros((len(lat), len(lon)))
prec_sig = np.zeros(np.shape(prec_trend))


years = np.arange(styear, edyear+1)
for i in range(len(lat)):
    for j in range(len(lon)):
        evaps = evap_year[:, i, j]
        r_e = stats.linregress(years, evaps)
        precs = prec_year[:, i, j]
        r_p = stats.linregress(years, precs)

        evap_trend[i, j] = 100*r_e.slope*len(years)
        prec_trend[i, j] = 100*r_p.slope*len(years)
        if r_e.pvalue < 0.05:
            evap_sig[i, j] = 1
            prec_sig[i, j] = 1
        else:
            evap_sig[i, j] = np.NaN
            prec_sig[i, j] = 1


evap_trend_sig = evap_trend*evap_sig
prec_trend_sig = prec_trend*prec_sig

fig = plt.figure(figsize=(8, 5))

# subplot test
ax = fig.add_subplot(221)
ax.set_title('Surface Evaporation trend')

m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(0, 420, 60), label=[0, 0, 0, 1])
m.drawcoastlines()
levels = np.arange(-40, 41, 4.0)
draw = plt.contourf(lon, lat, evap_trend_sig, levels,
                    cmap='jet', extend='both', latlon=True)
plt.colorbar(draw, orientation='horizontal',
             fraction=0.05, pad=0.05, label=['cm'])


ax = fig.add_subplot(222)
ax.set_title('precipitaion Trend')
m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(0, 420, 60), label=[0, 0, 0, 1])
m.drawcoastlines()
levels = np.arange(-40, 41, 4.0)
draw = plt.contourf(lon, lat, prec_trend_sig, levels,
                    cmap='jet', extend='both', latlon=True)
plt.colorbar(draw, orientation='horizontal',
             fraction=0.05, pad=0.05, label=['cm'])


ax = fig.add_subplot(224)

cosarray = np.zeros((len(lat), len(lon)))

for x in range(0, len(lon)):
    cosarray[:, x] = np.cos(lat * np.pi / 180)

wgt_evap_year0 = np.zeros(evap_year_a.shape)
wgt_prec_year0 = np.zeros(evap_year_a.shape)
for t in range(0, len(evap_year_a)):
    wgt_evap_year0[t, :, :] = evap_year_a[t, :, :]*cosarray
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
# plt.colorbar(draw,orientation='horizontal',fraction=0.05,pad=0.08,label=['cm'])
#plt.title('Surface Evaporation trend',fontsize=15)

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc')

lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
T2m = data.variables['t2m'][:]

def surface_temp_trend(viw):
    T2m_trend = np.zeros((len(lat), len(lon)))
    T2m_sig = np.zeros((len(lat), len(lon)))
    years = np.arange(1979, 1979+len(time))

    for i in range(len(lat)):
        for j in range(len(lon)):
            t2 = T2m[:, i, j]
            r = stats.linregress(years, t2)
            # trend
            T2m_trend[i, j] = r.slope*len(time)  # 베타(r.slope)에 40년치
            if r.pvalue < 0.05:  # 95% 유의 수준
                T2m_sig[i, j] = 1
            else:
                T2m_sig[i, j] = viw

    t2m_trend_sig = T2m_trend*T2m_sig
    levels = np.arange(-2.25, 2.5, 0.25)
    r_value1 = [levels, t2m_trend_sig]
    return r_value1


def regression_analysis():
    cosarray = np.zeros((len(lat), len(lon)))
    RAD = np.pi/180.
    for x in range(0, len(lat)):
        cosarray[x, :] = np.cos(lat[x]*RAD)

    wgt_T2m_year0 = np.zeros(T2m.shape)
    for t in range(0, len(T2m)):
        wgt_T2m_year0[t, :, :] = T2m[t, :, :]*cosarray

    wgt_T2m_year = np.sum(wgt_T2m_year0, 2)
    avg_T2m_year = np.sum(wgt_T2m_year, 1)/np.sum(cosarray) - 273
    # 회귀분석
    x = np.arange(1979, 2018+1)
    r = stats.linregress(x, avg_T2m_year)
    beta = r.slope
    alpha = r.intercept
    y = beta*x + alpha

    r_value2 = [x, y, avg_T2m_year]
    return r_value2

def ct(lon, lat, lvl, sts, title):
    ax.set_title(title, fontsize=15)
    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
    m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(60, 420, 60), labels=[0, 0, 0, 1])
    m.drawcoastlines()
    draw = plt.contourf(lon, lat, sts, lvl,cmap='jet', extend='both', latlon=True)
    plt.colorbar(draw, orientation='horizontal',fraction=0.05, pad=0.05, label=['K'])

#
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(221)
ct(lon,lat,surface_temp_trend(1)[0],surface_temp_trend(1)[1],'Surface temperature trend')

ax = fig.add_subplot(222)
plt.plot(regression_analysis()[0], regression_analysis()[2], 'go--')


ax = fig.add_subplot(223)
ct(lon,lat,surface_temp_trend(np.NaN)[0],surface_temp_trend(np.NaN)[1],'Surface temperature trend [95%]')


ax = fig.add_subplot(224)
plt.plot(regression_analysis()[0], regression_analysis()[
         2], 'go--', regression_analysis()[0], regression_analysis()[1], 'r')


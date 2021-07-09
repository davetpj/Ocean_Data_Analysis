import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc')
# data 안에 lon 60 lat 30 time 40 연평균이 있음. 회기분석, 주성분분석에 집중

lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
T2m = data.variables['t2m'][:]

# print(T2m.shape) time lst lon

T2m.shape
# Perform linear regression and significance test
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
            T2m_sig[i, j] = np.NaN
        # 1 for significant grids and NaN eleswise

# 유의미한 지역만 시각화
t2m_trend_sig = T2m_trend*T2m_sig


# Mapping the trend
fig = plt.figure(figsize=(8, 5))  # open figure

m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(60, 420, 60), labels=[0, 0, 0, 1])
m.drawcoastlines()

levels = np.arange(-2.25, 2.5, 0.25)
draw = plt.contourf(lon, lat, t2m_trend_sig, levels,
                    cmap='jet', extend='both', latlon=True)

plt.colorbar(draw, orientation='horizontal',
             fraction=0.05, pad=0.05, label=['K'])
plt.title('Surface temperature trend', fontsize=15)

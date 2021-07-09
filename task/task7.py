import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc', 'r')

lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
T2m = data.variables['t2m'][:, :, :]
data.close()

# 노말라이즈 하면 달라질지 모르겟지만 똑같이 나옴

# T2m = np.zeros(T2m_orig)
# for x in range(0, len(lon)):
#     cosarray[:, x] = np.cos(lat * np.pi / 180)

# wgt_evap_year0 = np.zeros(evap_year.shape)

# for t in range(0, len(time)):
#     wgt_T2m0[t, :, :] = T2m[t, :, :]*cosarray

# wgt_T2m = np.sum(-wgt_T2m0, 2)
# avg_T2m = np.sum(wgt_T2m, 1) / np.sum(cosarray)

# mode2 는 pc2
# 저위도가 더 중요함.

# 시간에대한 프로젝션과 공간에 대한 프로젝션의 결과가 똑같음을 볼 수 있다.
# 왜 두가지 방법이 똑같이 나오는지 수식으로 증명해보라.


T2 = np.zeros(T2m.shape)
RAD = np.pi/180.
for x in range(0, len(lat)):
    cosarray[x, :] = np.cos(lat[x]*RAD)

wgt_T2m_year0 = np.zeros(T2m.shape)
for t in range(0, len(T2m)):
    wgt_T2m_year0[t, :, :] = T2m[t, :, :]*cosarray

wgt_T2m_year = np.sum(wgt_T2m_year0, 2)
avg_T2m_year = np.sum(wgt_T2m_year, 1)/np.sum(cosarray) - 273
# eof 와 pc 에서 - 제거
#########


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
    # 회기분석
    x = np.arange(1979, 2018+1)
    r = stats.linregress(x, avg_T2m_year)
    beta = r.slope
    alpha = r.intercept
    y = beta*x + alpha

    r_value2 = [x, y, avg_T2m_year]
    return r_value2


def ct(lon, lat, lvl, sts, title):
    ax.set_title(title, fontsize=15)
    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
                90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
    m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(60, 420, 60), labels=[0, 0, 0, 1])
    m.drawcoastlines()
    draw = plt.contourf(lon, lat, sts, lvl, cmap='jet',
                        extend='both', latlon=True)
    plt.colorbar(draw, orientation='horizontal',
                 fraction=0.05, pad=0.05, label=['K'])


###
T2m = avg_T2m_year
###
T2_mean = np.mean(T2m, 0)
T2a = np.array(T2m-T2_mean)
T2a_1d = np.reshape(T2a, (len(time), len(lon)*len(lat)))
cov_T2a_1d = np.matmul(T2a_1d.T, T2a_1d)/len(time)
#
eigen_vals, eigen_vecs = np.linalg.eig(cov_T2a_1d)
efrac = eigen_vals.real / np.sum(eigen_vals.real)
eigen_vals_sum = np.sum(eigen_vals)
# PCs = np.dot(T2a_1d, eigen_vecs)
Y = np.dot(T2a_1d, eigen_vecs)
MODE = 1
EOF = np.reshape(eigen_vecs[:, MODE-1], (len(lat), len(lon)))
Y1 = Y[:, MODE-1]
efrac1 = efrac[MODE-1]*100
# 35.28%
# normalization
n_eigen_vec = -EOF * np.sqrt(eigen_vals[MODE-1])
# Dimensionless PC Time-series & Normalization
dY = Y1 / np.sqrt(eigen_vals[MODE-1])
n_Y = -(dY - np.mean(dY) / np.std(dY))
levels = np.arange(-0.6, 0.7, 0.1)
#
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(221)
ct(lon, lat, surface_temp_trend(1)[0], surface_temp_trend(
    1)[1], 'Surface Temperature Trend')

ax = fig.add_subplot(222)
plt.plot(regression_analysis()[0], regression_analysis()[
         2], 'go--', regression_analysis()[0], regression_analysis()[1], 'r')


ax = fig.add_subplot(223)
ct(lon, lat, levels, n_eigen_vec, 'EOF1 [32.29%]')

ax = fig.add_subplot(224)
plt.plot(regression_analysis()[0], n_Y, 'k-', linewidth=2)

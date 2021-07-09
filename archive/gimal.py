import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import os

## 저위도 태평양 지역. 경도 120E ~ 290E, 위도 40S ~ 40N ##
region = [120, 290, -40, 40]

# Import Data
DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/sst_ERA5_1979_2018_month.nc', 'r')

lon = data.variables['lon'][:]  # Lon
lat = data.variables['lat'][:]  # Lat

# T2m monthly data
SST_month = data.variables['sst'][:, :, :]
data.close()

# 육지 지역 (= -32767 ) 을 결측값 (NaN) 처리
idxNaN = np.where(SST_month < 0)
SST_month[idxNaN] = np.NaN

# 겨울철 (12-1-2월) 평균값들을 SST_year 에 저장
years = np.arange(1980, 2019)
SST_year = []
for y in range(11, len(SST_month)-1, 12):
    SSTy = np.nanmean(SST_month[y:y+3, :, :], 0)
    SST_year.append(SSTy)

SST_year = np.array(SST_year)

# 저위도 태평양 지역만 골라냄 (위 region 변수를 이용)
lon_idx = (lon >= region[0]) & (lon <= region[1])
lat_idx = (lat >= region[2]) & (lat <= region[3])

lat_region = lat[lat_idx]
lon_region = lon[lon_idx]

SST_region = SST_year[:, lat_idx][..., lon_idx]
print(SST_region.shape)

## 2m 온도 데이터 불러오기 ##
data = nc.Dataset(DATA_PATH + "/T2m_ERA5_1979_2018_month.nc", 'r')
T2_month = data.variables['t2m'][:, :, :]  # T2m monthly data
data.close()

# T2m monthly data 불러오기
# 위 SST 데이터와 마찬가지로 12-1-2월 평균을 계산해서 T2_year 에 저장

T2_year = []

for y in range(11, len(T2_month)-1, 12):
    T2y = np.nanmean(T2_month[y:y+3, :, :], 0)
    T2_year.append(T2y)

T2_year = np.array(T2_year)


# Removing the climatology
SST_region_mean = np.nanmean(SST_region, 0)
SST_region_a = np.array(SST_region - SST_region_mean)

# Cosine weighting
SST_region_wgt = np.zeros(SST_region_a.shape)

# SST_region_a 에 cosign weighting 을 해서 SST_region_wgt 에 저장
RAD = np.pi/180
cosarray = np.zeros((len(lat_region), len(lon_region)))

for t in range(len(lat[lat_idx])):
    SST_region_wgt[:, t, :] = SST_region_a[:, t, :]*np.cos(lat[lat_idx][t]*RAD)


# Retrieving eigenvalues & eigenvectors
# Singular Value Decomposition (SVD)
idxNaN_region = np.isnan(SST_region_wgt)
SST_region_wgt[idxNaN_region] = 0

SST_region_wgt1d = np.reshape(
    SST_region_wgt, (len(years), len(lon_region)*len(lat_region)))

eigen_vec, s, PCs = np.linalg.svd(SST_region_wgt1d.T)

# Fraction of each Eigenvalue
eigen_val = s**2
efrac = eigen_val / np.sum(eigen_val)

MODE = 1
# Primary Modes
# 첫번째 공간 주성분을 뽑아서 EOF 라는 변수에 저장

EOF = np.reshape(eigen_vec[MODE-1, :], (len(lat_region), len(lon_region)))

# 첫번째 시간 주성분을 뽑아서 PC 라는 변수에 저장
PC = PCs[MODE-1, :]


# Normalize eigenvector
n_EOF = -1.2*EOF * np.sqrt(s[MODE-1])

# Normalize PC Time-series
n_PC = (1/1.2)*PC * np.sqrt(s[MODE-1])


######### Regression against T2m and Precip ##########
# n_PC 와 2m 온도 자료를 각 격자 (grid) 마다 회귀분석을 수행
# 즉, 기울기 (beta) 값을 각 격자마다 구해서 T2_beta 에 저장
# 우선, 2m 온도 자료의 경향성 (온난화 경향)을 제거 (signal.detrend)
T2_year_detrend = signal.detrend(T2_year, 0)

T2_beta = np.zeros((len(lat), len(lon)))
T2_sig = np.zeros(np.shape(T2_beta))

for i in range(len(lat)):
    for j in range(len(lon)):
        t2 = T2_year_detrend[:, i, j]
        r = stats.linregress(n_PC, t2)
        T2_beta[i, j] = r.slope
        if r.pvalue < 0.2:
            T2_sig[i, j] = r.slope
        else:
            T2_sig[i, j] = np.NaN

T2_beta_sig = T2_sig


# Figures
fig = plt.figure(figsize=(5.5, 5))

# Plotting the PC time-series
ax1 = plt.subplot(211)

ax1.plot(years, n_PC, 'k-', linewidth=2)

# Tick options
ax1.set_xticks(np.arange(1980, 2019, 5))  # set major xtick

# Axes options
ax1.set_xlabel('Year', fontsize=13)  # x-axis label
ax1.set_ylabel('PC1', fontsize=13)
ax1.set_xlim(1978, 2019)  # set x-axis limit

# ax2 Title
ax1.set_title('Principal Component time series', loc='left', fontsize=13)

# Mapping of Regression Slopes
ax2 = plt.subplot(212)

levels = np.arange(-0.7, 0.8, 0.1)

m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(0, 420, 60), labels=[0, 0, 0, 1])
m.drawcoastlines()

draw = plt.contourf(lon, lat, T2_beta_sig, levels, cmap='jet', extend='both')


# Colorbar for Shading
plt.colorbar(draw, orientation='vertical',
             fraction=0.02, pad=0.05, label='[K]')

plt.tight_layout()

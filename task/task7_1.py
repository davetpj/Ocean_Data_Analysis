import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc', 'r')

lon = data.variables['lon'][:]  # Lon
lat = data.variables['lat'][:]  # Lat
time = data.variables['time'][:]  # Time
T2_orig = data.variables['t2m'][:, :, :]  # T2m data
data.close()


T2 = T2_orig
RAD = np.pi/180.  # transforms degrees to radian
for x in range(0, len(lat)):
    T2[:, x, :] = T2_orig[:, x, :]*np.cos(lat[x]*np.pi/180)

# Removing the time mean value
# T2m time mean
T2_mean = np.mean(T2, 0)
# Subtracting the time mean
T2a = np.array(T2 - T2_mean)

# Covariance matrix for T2m data
# converting 3D data to 2D (from 2D space to 1D space)
T2a_1d = np.reshape(T2a, (len(time), len(lon)*len(lat)))

# Produce the covariance matrix
#cov_T2a_1d = np.matmul(T2a_1d.T, T2a_1d)/len(time)
# 이렇게하면 1800 1800


# 위 covariance matrix 를 아래와 같이 생성
cov_T2a_1d = np.matmul(T2a_1d, T2a_1d.T)/len(time)
# 40 1800
# 아래 코드를 위 matrix 에 맞게 고치세요

# Retrieving eigenvalues & eigenvectors
# Eigen values & vectors
eigen_val, eigen_vec = np.linalg.eig(cov_T2a_1d)
# 대각행렬 val(1800,1800), (1800,1800)
# 시간 고유벡터 40 40vec 공분산행렬 40 1800 공간 주성분백터 40 1800
# 이렇게 바꾸면 아이겐 벨류는 40 40 람다
# 아이겐 벡터도 40 40 (시간에 대한 주성분으로 나옴 시공간이 바뀐것 원래대로 하면 공간이였음)
#

# Fraction of each eigenvalue
efrac = eigen_val / np.sum(eigen_val)

# PC Time-series: projection of data to eigenvector
EOF = np.matmul(T2a_1d.T, eigen_vec)
# (40 1800) (40 1800) (1800 1800)
#PCs = eigen_vec * (eigen_val) ** 2
# 시간에 대한 주성분


# Primary Modes
MODE = 1

# EOF First mode
#PCs = np.reshape(eigen_vec[:, MODE-1], (len(lat), len(lon)))
#PCs = np.reshape(eigen_vec[:, MODE-1], (len(time)))

print(EOF.shape)

# First PC Time-series
PC = PCs[:, MODE-1]
# print(PC.shape)

efrac1 = efrac[MODE-1] * 100

# Normalize eigenvector
n_EOF = EOF * np.sqrt(eigen_val[MODE-1])

# Normalize PC Time-series
n_PC = PC/np.sqrt(eigen_val[MODE-1])
#n_PC = -PC
# Figures
fig = plt.figure(figsize=(5.2, 7))

# Mapping of EOF
levels = np.arange(-0.6, 0.7, 0.1)

ax1 = plt.subplot(211)  # ax1: configuration of the figure

# m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
#            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
#m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
#m.drawmeridians(np.arange(0, 420, 60), labels=[0, 0, 0, 1])
# m.drawcoastlines()

# Shading
#draw = plt.contourf(lon, lat, n_EOF, levels, cmap='jet', extend='both')

# Colorbar for Shading
# plt.colorbar(draw, orientation='horizontal',
#             fraction=0.05, pad=0.11, label='[K]')

# ax1 Title
ax1.set_title('EOF%d' % (MODE), loc='left', fontsize=15)
ax1.set_title('%1.2f%%' % (efrac1), loc='right', fontsize=15)

# Plotting the PC time-series
ax2 = plt.subplot(212)

years = np.arange(1979, 2019)

ax2.plot(years, n_PC, 'k-', linewidth=2)

# Tick options
ax2.set_xticks(np.arange(1980, 2019, 5))  # set major xtick

# Axes options
ax2.set_xlabel('Year', fontsize=15)  # x-axis label
ax2.set_ylabel('PC1', fontsize=15)  # y-axis label
ax2.set_xlim(1978, 2019)  # set x-axis limit

# ax2 Title
ax2.set_title('PC Time-series', loc='left', fontsize=15)
plt.tight_layout()

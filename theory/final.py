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

# 코싸인어레이
T2 = np.zeros(shape=(40, len(lat), len(lon)))
RAD = np.pi/180.  # transforms degrees to radian
for x in range(0, len(lat)):
    T2[:, x, :] = T2_orig[:, x, :]*np.cos(lat[x]*np.pi/180)

T2_mean = np.mean(T2, 0)
# Subtracting the time mean
T2a = np.array(T2 - T2_mean)

T2a_1d = np.reshape(T2a, (len(time), len(lon)*len(lat)))

print(T2a_1d.shape == (40, 1800))  # True

# 1번째 방법 (1800,1800) 만들기

cov_T2a_1d = np.matmul(T2a_1d.T, T2a_1d)/40
eigen_val, eigen_vec = np.linalg.eig(cov_T2a_1d)
test1 = np.matmul(cov_T2a_1d, eigen_vec[:, 0])
test2 = eigen_val[0]*eigen_vec[:, 0]
print(test1 == test2)
PCs = np.matmul(T2a_1d, eigen_vec)
# 40 1800 1800 1800
# = 40 1800
print(PCs.shape, "PCs")
PC1 = -PCs[:, 0]
PC1_test1 = PC1
PC1_test2 = -np.matmul(T2a_1d, eigen_vec[:, 0])
print(PC1_test1 == PC1_test2, "test")


EOF = np.reshape(eigen_vec[:, 0], (30, 60))
EOF = -EOF

# Normalization
n_eigen_vec = -EOF * np.sqrt(eigen_val[0])


dPC1 = PC1/np.sqrt(eigen_val[0])
print(np.sqrt(eigen_val[0]))

n_PC1 = -(dPC1 - np.mean(dPC1)) / np.std(dPC1)


fig = plt.figure(figsize=(5.2, 7))
# Mapping of EOF
levels = np.arange(-0.6, 0.7, 0.1)
ax1 = plt.subplot(211)  # ax1: configuration of the figure

m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(0, 420, 60), labels=[0, 0, 0, 1])
m.drawcoastlines()

# Shading
draw = plt.contourf(lon, lat, n_eigen_vec, levels, cmap='jet', extend='both')

# Colorbar for Shading
plt.colorbar(draw, orientation='horizontal',
             fraction=0.05, pad=0.11, label='[K]')

# ax1 Title
ax1.set_title('EOF%d' % (1), loc='left', fontsize=15)
ax1.set_title('%1.2f%%' % (1), loc='right', fontsize=15)

# Plotting the PC time-series
ax2 = plt.subplot(212)

years = np.arange(1979, 2019)

ax2.plot(years, n_PC1, 'k-', linewidth=2)

# Tick options
ax2.set_xticks(np.arange(1980, 2019, 5))  # set major xtick

# Axes options
ax2.set_xlabel('Year', fontsize=15)  # x-axis label
ax2.set_ylabel('PC1', fontsize=15)  # y-axis label
ax2.set_xlim(1978, 2019)  # set x-axis limit

# ax2 Title
ax2.set_title('PC Time-series', loc='left', fontsize=15)
plt.tight_layout()

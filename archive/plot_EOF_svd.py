# PC/EOF script for Week 12
import netCDF4 as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


# Import Data
data = nc.Dataset("T2m_ERA5_1979_2018_lowR.nc", 'r')
lon = data.variables['lon'][:]  # Lon
lat = data.variables['lat'][:]  # Lat
time = data.variables['time'][:]  # Time
T2_orig = data.variables['t2m'][:, :, :]  # T2m data
data.close()

#T2 = T2_orig
T2 = np.zeros(T2_orig.shape)
RAD = np.pi/180.  # transforms degrees to radian
for x in range(0, len(lat)):
    T2[:, x, :] = T2_orig[:, x, :]*np.cos(lat[x]*np.pi/180)

# Removing the time mean value
# T2 time mean
T2_mean = np.mean(T2, 0)

T2a = np.array(T2 - T2_mean)  # T2m anomaly

T2a_1d = np.reshape(T2a, (len(time), len(lon)*len(lat)))

# Retrieving eigenvalues & eigenvectors
# Singular Value Decomposition (SVD)
eigen_vec, s, PCs = np.linalg.svd(T2a_1d.T)

# Fraction of each Eigenvalue
eigen_val = s**2
efrac = eigen_val / np.sum(eigen_val)

# Primary Modes
MODE = 1

# EOF First mode
EOF = np.reshape(eigen_vec[:, MODE-1], (len(lat), len(lon)))
print(EOF.shape)

# First PC Time-series
PC = PCs.T[:, MODE-1]
print(PC.shape)

efrac1 = efrac[MODE-1] * 100

# Normalize eigenvector
n_EOF = -EOF * np.sqrt(s[MODE-1])

# Normalize PC Time-series
n_PC = -PC * np.sqrt(s[MODE-1])

# Figures
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
draw = plt.contourf(lon, lat, n_EOF, levels, cmap='jet', extend='both')

# Colorbar for Shading
plt.colorbar(draw, orientation='horizontal',
             fraction=0.05, pad=0.11, label='[K]')

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

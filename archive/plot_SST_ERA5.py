
# load necessary modules
import numpy as np
import netCDF4 as nc
import matplotlib
from matplotlib import interactive
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
####PART1: Make an array to mask land surface data#####
# load SST
# SSTdir='C:/python/ERA5/' # directory to SST file

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/ERSST.nc')

SSTfile = 'ERSST.nc'  # SST file name
# SSTread=Dataset(SSTdir+SSTfile)
SSTread = data

lat = SSTread['lat'][:].data
lon = SSTread['lon'][:].data
SST = SSTread['SST']  # extract variable named 'SST'

SSTdata = SST[0, :, :].data  # select 1st time step, and drop out attributes
#SSTdata1 = SST[40,:,:].data
#SSTanom = SSTdata1 - SSTdata0

## Replacing -999 as NaN ###
# find indices for grids where SST=-999, i.e. land
nan_land = np.where(SSTdata == -999)
SSTdata[nan_land] = np.NaN  # fill in land grids with nan value

# PART4: Draw on map
figname = 'sst0_ERA5.pdf'
fig = plt.figure(figsize=(16, 10))  # open figure

m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(0, 420, 60), labels=[0, 0, 0, 1])
m.drawcoastlines()
m.fillcontinents()

xx, yy = np.meshgrid(lon, lat)  # meshgrid of lon lat
# levels=np.arange(-1,1.1,0.1) # contour level for SST anomalies
levels = np.arange(272, 302, 1)  # contour level for SST anomalies

pc = m.pcolormesh(xx, yy, SSTdata, cmap='jet',
                  vmin=np.nanmin(SSTdata), vmax=np.nanmax(SSTdata))
# 컬러바 표출
# size 컬러바 가로 폭 크기 지정
cbar = m.colorbar(pc, size='2%')

# draw=m.contourf(xx,yy,SSTdata,levels,cmap=plt.cm.bwr,extend='both',latlon=True)
# plt.colorbar(draw,orientation='horizontal',fraction=0.05,pad=0.08,label='[K]')

plt.title('Sea Surface Temperature', fontsize=15)
# save figure
plt.savefig(figname)

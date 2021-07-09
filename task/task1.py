# 201907 evaporation data visualization
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap

DATA_PATH = "../DATA/"

data = nc.Dataset(DATA_PATH + "EVAP.201907.nc")
print(data.variables)
# lat = data.variables['lat'][:]
# lon = data.variables['lon'][:]
# evap0 = data.variables['e'][:]
# evap = -evap0*100*31


# m = Basemap(projection='cyl', llcrnrlat=-90,
#             urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)

# m.drawcoastlines()
# levels = np.arange(-4, 24, 4)
# draw = plt.contourf(lon, lat, evap[0, :, :], levels, cmap='jet', extend='both')
# plt.colorbar(draw, orientation="horizontal", label="[cm]")

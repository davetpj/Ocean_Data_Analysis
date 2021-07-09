import numpy as np
import netCDF4 as nc

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'

# Import monthly evaporation data
EVAPdir = DATA_PATH + 'EVAPintp/'
styear = 1979  # start year
edyear = 2019  # end year

evaps = np.zeros([12*(edyear-styear+1), 37, 72])
index = 0
for i in range(styear, edyear+1):
    for j in range(1, 13):
        year = str(i)
        if j > 9:
            j = j
        else:
            j = "0"+str(j)
        month = str(j)
        filename = f"{EVAPdir}EVAP.{year}{month}.nc"
        data = nc.Dataset(filename)
        evap_month = data.variables['e'][:, :]  # [m/day]
        evaps[index, :, :] = evap_month
        index = index+1

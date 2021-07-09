import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.stats as stats

DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc')

lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]

T2m = data.variables['t2m'][:]

T2m.shape

cosarray = np.zeros((len(lat), len(lon)))
RAD = np.pi/180.
for x in range(0, len(lat)):
    cosarray[x, :] = np.cos(lat[x]*np.pi/180)

wgt_T2m_year0 = np.zeros(T2m.shape)
for t in range(0, len(T2m)):
    wgt_T2m_year0[t, :, :] = T2m[t, :, :]*cosarray

wgt_T2m_year = np.sum(wgt_T2m_year0, 2)
avg_T2m_year = np.sum(wgt_T2m_year, 1)/np.sum(cosarray) - 273

# regression analysis
#[slope, intercept, r_value,pvalue, std_err] = stats.linregress
x = np.arange(1979, 2018+1)
r = stats.linregress(x, avg_T2m_year)
beta = r.slope
alpha = r.intercept

y = beta*x + alpha

# plot
fig = plt.figure(figsize=(5.5, 4.5))
plt.plot(x, avg_T2m_year, 'go--', x, y, 'r')
#plt.plot(x, avg_T2m_year, 'go--')

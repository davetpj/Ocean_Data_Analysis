import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#from m11f1109_2 import avg_T2m_year
DATA_PATH = '/Users/dhkim/gh/Ocean_Data_Analysis/DATA/'
data = nc.Dataset(DATA_PATH + '/T2m_ERA5_1979_2018_lowR.nc', 'r')

""" """
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
time = data.variables['time'][:]
T2 = data.variables['t2m'][:, :, :]# 40시간 30lon 60lat
data.close()

""" """
# 시간에 대한 평균
T2_mean = np.mean(T2, 0)
# 시간에 대한 평균 제거 (아노말리) (3d 온도데이터에서 평균을 빼준것)(평균값을 0축으로 센터링)
T2a = np.array(T2-T2_mean)
# T2a 가 x 메트릭스인거임 센터링된

# 3D 시 공간 데이터를 2차원으로 변환 (차원 축소)
T2a_1d = np.reshape(T2a, (len(time), len(lon)*len(lat)))
#(40,1800)


# 공분산 행렬 구해서 정방 행렬로 만들어 줘야 한다.
cov_T2a_1d = np.matmul(T2a_1d.T, T2a_1d)/len(time)
#1800,1800


""" eigen value & eigen vector """
eigen_vals, eigen_vecs = np.linalg.eig(cov_T2a_1d)
#eigen_val(1800,0)람다값으로 구성 
#eigen_vec(1800,1800)고유 벡터

## 공헌도 살펴보기
efrac = eigen_vals.real / np.sum(eigen_vals.real)
#람다1pc1의 공헌도가 35.29 람다2pc2의 공헌도가 7.89%
#pc1이 설명을 잘 해줌을 알려줌


eigen_vals_sum = np.sum(eigen_vals)

pcar = eigen_vals/eigen_vals_sum


#pc1 = eigen_vecs[:, 0]
#pc1 = eigen_vecs*eigen_vals

## PC Time-series 원 데이터를 프로젝션 해줘야함 
PCs = np.dot(T2a_1d, eigen_vecs)
Y = np.dot(T2a_1d, eigen_vecs)
#Y(40,1800)
MODE = 1



EOF= np.reshape(eigen_vecs[:,MODE-1],(len(lat),len(lon)))
#(30, 60)

Y1 = Y[:,MODE-1]
#Y1(40,)
efrac1 = efrac[MODE-1]*100
#35.28%

#normalization
n_eigen_vec = -EOF *np.sqrt(eigen_vals[MODE-1])

#Dimensionless PC Time-series & Normalization

dY = Y1 / np.sqrt(eigen_vals[MODE-1])
n_Y = -(dY - np.mean(dY) / np.std(dY))


fig = plt.figure(figsize=(5.5, 4.5))

levels = np.arange(-0.6,0.7,0.1)
ax1 = plt.subplot(211)

m=Basemap(projection='cyl',resolution='c',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360)
m.drawparallels(np.arange(-90,120,30), label=[1,0,0,0])
m.drawmeridians(np.arange(0,420,60),labels=[0,0,0,1])
m.drawcoastlines()

draw = plt.contourf(lon,lat,n_eigen_vec,levels,cmap='jet',extend='both')

plt.colorbar(draw, orientation='horizontal',fraction=0.05,pad=0.11,label='[K]')

ax1.set_title('EOF%d'%(MODE),loc='left',fontsize=15)
ax1.set_title('%1.2f%%'%(efrac1),loc='right',fontsize=15)

ax2 = plt.subplot(212)

years = np.arange(1979,2019)

ax2.plot(years, n_Y, 'k-', linewidth=2)

ax2.set_xticks(np.arange(1980,2019,5))

ax2.set_xlabel('Year',fontsize=15)
ax2.set_ylabel('PC1',fontsize=15)

ax2.set_xlim(1978,2019)
ax2.set_title('PC Time-series', loc='left', fontsize=15)

plt.tight_layout()
plt.show()



#print(EOF)
#print(EOF.shape)

## First PC Time-series
#PC = PCs[:,0]
#pc1=(PC - np.mean(PC)) / np.std(PC)



#x = np.arange(1979, 2018+1)
#fig = plt.figure(figsize=(5.5, 4.5))
#plt.plot(x, pc1, 'go--')

#fig = plt.figure(figsize=(8, 5))  # open figure

#m = Basemap(projection='cyl', resolution='c', llcrnrlat=-
#            90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
#m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
#m.drawmeridians(np.arange(60, 420, 60), labels=[0, 0, 0, 1])
#m.drawcoastlines()

#levels = np.arange(-2.25, 2.5, 0.25)
#draw = plt.contourf(lon, lat, EOF, levels,
#                    cmap='jet', extend='both', latlon=True)

#plt.colorbar(draw, orientation='horizontal',
#             fraction=0.05, pad=0.05, label=['K'])
#plt.title('Surface temperature trend', fontsize=15)

# 주성분분석
# A = np.array([[1, 2], [2, 1]])
# print(A)

# w, v = np.linalg.eig(A)
# print(w)
# print(v)
# 평균에 대한 회기분석보다는 발전된 대안

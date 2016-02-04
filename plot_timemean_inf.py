import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

plt.figure()
f1 = Dataset('/Volumes/Drobo/enkf_twolevel_test1.nc')
lats = f1.variables['lat'][:]; lons = f1.variables['lon'][:]
thetinf1 = (f1.variables['thetinflation'][:].mean(axis=0)).mean(axis=-1)
f1.close()
f2 = Dataset('/Volumes/Drobo/enkf_twolevel_test2.nc')
thetinf2 = (f2.variables['thetinflation'][:].mean(axis=0)).mean(axis=-1)
f2.close()
plt.plot(lats,thetinf1,'b',label='RTPS inf',linewidth=1)
plt.plot(lats,thetinf2,'r',label='obs dep inf',linewidth=1)
plt.title('time mean inflation')
plt.xlim(-90,90)
plt.ylabel('time mean inflation for theta')
plt.xlabel('latitude (degrees)')
plt.grid(True)
plt.legend(loc='lower right',fontsize=12)
plt.show()

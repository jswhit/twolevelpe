import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

plt.figure()
f = Dataset('truth_twolevel_t32_12h_moistfact1p0.nc')
lats = f.variables['lat'][:]; lons = f.variables['lon'][:]
umean = (f.variables['u'][200:,1,:,:].mean(axis=0)).mean(axis=-1)
plt.plot(lats,umean,color='b',label='dry')
f.close()
f = Dataset('truth_twolevel_t32_12h_moistfact0p01.nc')
umean = (f.variables['u'][200:,1,:,:].mean(axis=0)).mean(axis=-1)
plt.plot(lats,umean,color='r',label='moist')
plt.title('u mean')
plt.legend()
f.close()

plt.figure()
f = Dataset('truth_twolevel_t32_12h_moistfact1p0.nc')
lats = f.variables['lat'][:]; lons = f.variables['lon'][:]
vmean = (f.variables['v'][200:,1,:,:].mean(axis=0)).mean(axis=-1)
vvar = ((f.variables['v'][200:,1,:,:]-vmean[:,np.newaxis])**2).mean(axis=0)
plt.plot(lats,np.sqrt(vvar).mean(axis=-1),color='b',label='dry')
f.close()
f = Dataset('truth_twolevel_t32_12h_moistfact0p01.nc')
vmean = (f.variables['v'][200:,1,:,:].mean(axis=0)).mean(axis=-1)
vvar = ((f.variables['v'][200:,1,:,:]-vmean[:,np.newaxis])**2).mean(axis=0)
plt.plot(lats,np.sqrt(vvar).mean(axis=-1),color='r',label='moist')
plt.title('v stdev')
plt.legend()
plt.show()

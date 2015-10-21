import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

plt.figure()
f = Dataset('/Volumes/Drobo/truth_twolevel_t63_12h.nc')
lats = f.variables['lat'][:]; lons = f.variables['lon'][:]
umean = (f.variables['u'][:,1,:,:].mean(axis=0)).mean(axis=-1)
plt.plot(lats,umean,color='b',label='upper level')
umean = (f.variables['u'][:,0,:,:].mean(axis=0)).mean(axis=-1)
plt.plot(lats,umean,color='r',label='lower level')
plt.title('u mean')
plt.legend()

plt.figure()
wmean = (f.variables['w'][:].mean(axis=0)).mean(axis=-1)
wvar = ((f.variables['w'][:]-wmean[:,np.newaxis])**2).mean(axis=0)
plt.plot(lats,np.sqrt(wvar).mean(axis=-1),color='b',label='stdev of w')
plt.plot(lats,10.*wmean,color='r',label='time mean w (x 10)')
plt.title('w mean (x10) & stdev')
plt.legend()
plt.show()

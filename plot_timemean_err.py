import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

plt.figure()
f1 = Dataset('/Volumes/Drobo/enkf_twolevel_test1.nc')
lats = f1.variables['lat'][:]; lons = f1.variables['lon'][:]
thetensmean1 = f1.variables['thetensmeana'][200:]
thetsprd1 = f1.variables['thetsprda'][200:]
thettruth = f1.variables['thettruth'][200:]
f1.close()
f2 = Dataset('/Volumes/Drobo/enkf_twolevel_test2.nc')
thetensmean2 = f2.variables['thetensmeana'][200:]
thetsprd2 = f2.variables['thetsprda'][200:]
f2.close()
theterr1 = (((thetensmean1-thettruth)**2).mean(axis=0)).mean(axis=-1)
theterr2 = (((thetensmean2-thettruth)**2).mean(axis=0)).mean(axis=-1)
thetsprd1 = thetsprd1.mean(axis=0).mean(axis=-1)
thetsprd2 = thetsprd2.mean(axis=0).mean(axis=-1)
theterr1av = (theterr1*np.cos(np.radians(lats))).sum()/np.cos(np.radians(lats)).sum()
theterr2av = (theterr2*np.cos(np.radians(lats))).sum()/np.cos(np.radians(lats)).sum()
thetsprd1av = (thetsprd1*np.cos(np.radians(lats))).sum()/np.cos(np.radians(lats)).sum()
thetsprd2av = (thetsprd2*np.cos(np.radians(lats))).sum()/np.cos(np.radians(lats)).sum()
print 'min/max/mean error:',theterr1.min(), theterr2.max(), theterr1av, theterr2av
print 'min/max/mean sprd:',thetsprd1.min(), thetsprd2.max(), thetsprd1av, thetsprd2av
plt.plot(lats,np.sqrt(theterr1),'b',label='RTPS inf err (%4.2f)' %\
        np.sqrt(theterr1av),linewidth=1)
plt.plot(lats,np.sqrt(theterr2),'r',label='obs dep inf err (%4.2f)' %\
        np.sqrt(theterr2av),linewidth=2)
plt.plot(lats,np.sqrt(thetsprd1),'b:',label='RTPS inf sprd (%4.2f)' %\
        np.sqrt(thetsprd1av),linewidth=2)
plt.plot(lats,np.sqrt(thetsprd2),'r:',label='obs dep inf sprd (%4.2f)' %\
        np.sqrt(thetsprd2av),linewidth=2)
plt.title('analysis error/spread')
plt.xlim(-90,90)
plt.ylabel('RMS theta error (solid) and spread (dashed) in K')
plt.xlabel('latitude (degrees)')
plt.grid(True)
plt.legend(loc='lower right',fontsize=10)
plt.show()

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pyspharm import Spharmt

def getvarspectrum(vrtspec,divspec,norm,indxm,indxn,ntrunc):
    varspect = np.zeros(ntrunc+1,np.float)
    nlm = (ntrunc+1)*(ntrunc+2)/2
    for n in range(nlm):
        vrtmag = (vrtspec[n]*np.conj(vrtspec[n])).real
        divmag = (divspec[n]*np.conj(divspec[n])).real
        if indxm[n] == 0:
            varspect[indxn[n]] += norm[n]*vrtmag
            varspect[indxn[n]] += norm[n]*divmag
        else:
            varspect[indxn[n]] += 2.*norm[n]*vrtmag
            varspect[indxn[n]] += 2.*norm[n]*divmag
    return varspect

f = Dataset('/Volumes/Drobo/truth_twolevel_t63_12h.nc')
lats = f.variables['lat'][:]; lons = f.variables['lon'][:]
u = f.variables['u']; v = f.variables['v']
ntimes, nlevs, nlats, nlons = u.shape
print nlons, nlats,  f.ntrunc, f.rsphere
sp = Spharmt(nlons,nlats,int(f.ntrunc),f.rsphere,gridtype='gaussian')
kenorm = (-0.25*sp.invlap).astype(np.float)
kespecmean = None
nout = 0
for ntime in range(ntimes):
    print ntime
    vrtspec, divspec = sp.getvrtdivspec(u[ntime,1,...],v[ntime,1,...])
    kespec = getvarspectrum(vrtspec,divspec,kenorm,sp.order,sp.degree,sp.ntrunc)
    if kespecmean is None:
        kespecmean = kespec
    else:
        kespecmean += kespec
    nout += 1
kespecmean = kespecmean/nout
plt.loglog(np.arange(f.ntrunc+1),kespec,linewidth=2,\
        label='t63')

f.close()
f = Dataset('truth_twolevel_t32_12h_tst.nc')
u = f.variables['u']; v = f.variables['v']
ntimes, nlevs, nlats, nlons = u.shape
print nlons, nlats,  f.ntrunc, f.rsphere
sp = Spharmt(nlons,nlats,int(f.ntrunc),f.rsphere,gridtype='gaussian')
kespecmean = None
nout = 0
for ntime in range(ntimes):
    print ntime
    vrtspec, divspec = sp.getvrtdivspec(u[ntime,1,...],v[ntime,1,...])
    kespec = getvarspectrum(vrtspec,divspec,kenorm,sp.order,sp.degree,sp.ntrunc)
    if kespecmean is None:
        kespecmean = kespec
    else:
        kespecmean += kespec
    nout += 1
kespecmean = kespecmean/nout
plt.loglog(np.arange(f.ntrunc+1),kespec,linewidth=2,\
        label='t32')
plt.legend()
plt.show()
f.close()

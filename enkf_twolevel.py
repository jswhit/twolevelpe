from pyspharm import Spharmt
from twolevel import TwoLevel
import numpy as np
from netCDF4 import Dataset
import sys, time
from enkf_utils import  gcdist,bilintrp,serial_ensrf,gaspcohn,fibonacci_pts

# EnKF cycling for two-level model with mid-level temp obs

if len(sys.argv) == 1:
   msg="""
python enkf_twolevel.py covlocal_scale covinflate1 covinflate2
   """
   raise SystemExit(msg)
# covariance localization length scale in meters.
covlocal_scale = float(sys.argv[1])
# inflation parameters
# (covinflate2 <= 0 for RTPS, otherwise use Hodyss and Campbell)
if len(sys.argv) < 3: #
    # no inflation factor specified, use Hodyss and Campbell with a=b=1
    covinflate1 = 1.0; covinflate2 = 1.0
elif len(sys.argv) == 3:
    covinflate1 = float(sys.argv[2])
    covinflate2 = 0.0
else:
    covinflate1 = float(sys.argv[2])
    covinflate2 = float(sys.argv[3])

profile = False # turn on profiling?

nobs = 500 # number of obs to assimilate
# each ob time nobs ob locations are randomly sampled (without
# replacement) from an evenly spaced fibonacci grid of nominally nobsall points.
# if nobsall = nobs, a fixed observing network is used.
nobsall = 10*nobs
nanals = 20 # ensemble members
oberrstdev = 0.5 # ob error in L
nassim = 1501 # assimilation times to run
gaussian=True # if True, use Gaussian function similar to Gaspari-Cohn
              # polynomial for localization.

# grid, time step info
nlons = 96; nlats = nlons/2  # number of longitudes/latitudes
ntrunc = nlons/3 # spectral truncation (for alias-free computations)
gridtype = 'regular'
dt = 3600. # time step in seconds
rsphere = 6.37122e6 # earth radius

# fix random seed for reproducibility.
np.random.seed(42)

# model nature run to sample initial ensemble and draw additive noise.
modelclimo_file = 'truth_twolevel_t%s_12h.nc' % ntrunc
# 'truth' nature run to sample obs
# (these two files can be the same for perfect model expts)
# file to sample additive noise.
truth_file = 'truth_twolevel_t32_12h.nc'

# create spherical harmonic transform instance
sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype=gridtype)

models = []
for nanal in xrange(nanals):
    models.append(TwoLevel(sp,dt))

# read nature run, create obs.
nct = Dataset(truth_file)
lats = nct.variables['lat'][:]
lons = nct.variables['lon'][:]
# find fibonacci points between latmin and latmax
if nobs == 1:
    nobsall = 1
    # single ob test.
    oblonsall = np.array([180.], np.float)
    oblatsall = np.array([45.], np.float)
else:
    oblatsall,oblonsall =\
    fibonacci_pts(nobsall,np.degrees(sp.lats[-1]),np.degrees(sp.lats[0]))
nobsall = len(oblatsall) # reset nobsall

print '# %s obs to assimilate (out of %s) with ob err stdev = %s' % (nobs,nobsall,oberrstdev)
print '# covlocal_scale=%s km, covinflate1=%s covinflate2=%s' %\
(covlocal_scale/1000., covinflate1,covinflate2)
thetaobsall = np.empty((nassim,nobsall),np.float)
# keep truth upper layer winds interpolated to all ob locations for validation.
uobsall = np.empty((nassim,nobsall),np.float)
vobsall = np.empty((nassim,nobsall),np.float)
usobsall = np.empty((nassim,nobsall),np.float)
vsobsall = np.empty((nassim,nobsall),np.float)
oberrvar = np.empty(nobs,np.float); oberrvar[:] = oberrstdev**2
obtimes = np.empty((nassim),np.float)
for n in xrange(nassim):
    # flip latitude direction so lats are increasing (needed for interpolation)
    theta = nct.variables['theta'][n,::-1,:]
    obtimes[n] = nct.variables['t'][n]
    thetaobsall[n] = bilintrp(theta,lons,lats[::-1],oblonsall,oblatsall)
    uobsall[n] = bilintrp(nct.variables['u'][n,1,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    vobsall[n] = bilintrp(nct.variables['v'][n,1,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    usobsall[n] = bilintrp(nct.variables['u'][n,0,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    vsobsall[n] = bilintrp(nct.variables['v'][n,0,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
nct.close()

# create initial ensemble by randomly sampling climatology
# of forecast model.
ncm = Dataset(modelclimo_file)
indx = np.random.choice(np.arange(len(ncm.variables['t'])),nanals,replace=False)
#indx[:] = 0 # for testing forward operator
thetaens = np.empty((nanals,sp.nlats,sp.nlons),np.float)
uens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float)
vens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float)
theta_modelclim = ncm.variables['theta']
u_modelclim = ncm.variables['u']
v_modelclim = ncm.variables['v']

nanal=0
for n in indx:
    thetaens[nanal] = theta_modelclim[n]
    uens[nanal] = u_modelclim[n]
    vens[nanal] = v_modelclim[n]
    nanal += 1

# transform initial ensemble to spectral space
vrtspec = np.empty((nanals,2,sp.nlm),np.complex)
divspec = np.empty((nanals,2,sp.nlm),np.complex)
thetaspec = np.empty((nanals,sp.nlm),np.complex)
for nanal in xrange(nanals):
    vrtspec[nanal], divspec[nanal] = sp.getvrtdivspec(uens[nanal],vens[nanal])
    thetaspec[nanal] = sp.grdtospec(thetaens[nanal])
xens = np.empty((nanals,5*sp.nlons*sp.nlats),np.float) # empty 1d state vector array

# precompute covariance localization for fixed observation network.
covlocal1 = np.zeros((nobsall,sp.nlons*sp.nlats),np.float)
hcovlocal = np.zeros((nobsall,nobsall),np.float)
modellats = np.degrees(sp.lats)
modellons = np.degrees(sp.lons)
modellons,modellats = np.meshgrid(modellons,modellats)
for nob in xrange(nobsall):
    r = sp.rsphere*gcdist(np.radians(oblonsall[nob]),np.radians(oblatsall[nob]),
    np.radians(modellons.ravel()),np.radians(modellats.ravel()))
    taper = gaspcohn(r/covlocal_scale,gaussian=gaussian)
    covlocal1[nob,:] = taper
    r = sp.rsphere*gcdist(np.radians(oblonsall[nob]),np.radians(oblatsall[nob]),
    np.radians(oblonsall),np.radians(oblatsall))
    taper = gaspcohn(r/covlocal_scale,gaussian=gaussian)
    hcovlocal[nob,:] = taper
covlocal = np.tile(covlocal1,5)

fhassim = obtimes[1]-obtimes[0] # assim interval  (assumed constant)
nsteps = int(fhassim*3600/models[0].dt) # time steps in assim interval
print '# fhassim,nsteps = ',fhassim,nsteps

# initialize model clock
for nanal in xrange(nanals):
    models[nanal].t = obtimes[0]*3600.

for ntime in xrange(nassim):

    # check model clock
    if models[0].t/3600. != obtimes[ntime]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t/3600., obtimes[ntime]))

    # compute forward operator.
    t1 = time.clock()
    # ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float)
    hxensu = np.empty((nanals,nobsall),np.float)
    hxensv = np.empty((nanals,nobsall),np.float)
    hxensus = np.empty((nanals,nobsall),np.float)
    hxensvs = np.empty((nanals,nobsall),np.float)
    hxenstheta = np.empty((nanals,nobsall),np.float)
    if nobs == nobsall:
        oblats = oblatsall; oblons = oblonsall
        thetaobs = thetaobsall[ntime]
        obindx = np.arange(nobs)
        covlocal_tmp = covlocal; hcovlocal_tmp = hcovlocal
    elif nobsall > nobs:
        obindx = np.random.choice(np.arange(nobsall),size=nobs,replace=False)
        oblats = oblatsall[obindx]; oblons = oblonsall[obindx]
        thetaobs = np.ascontiguousarray(thetaobsall[ntime,obindx])
        covlocal_tmp = np.ascontiguousarray(covlocal[obindx,:])
        hcovlocal_tmp = np.ascontiguousarray(hcovlocal[obindx,:][:,obindx])
    else:
        raise ValueError('nobsall must be >= nobs')
    if oberrstdev > 0.: # add observation error
        thetaobs += np.random.normal(scale=oberrstdev,size=nobs) # add ob errors
    for nanal in xrange(nanals):
        # inverse transform to grid.
        uens[nanal],vens[nanal] = sp.getuv(vrtspec[nanal],divspec[nanal])
        thetaens[nanal] = sp.spectogrd(thetaspec[nanal])
        # forward operator calculation.
        theta = thetaens[nanal,::-1,:]
        hxens[nanal] = bilintrp(theta,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
        hxensu[nanal] =\
        bilintrp(uens[nanal,1,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensv[nanal] =\
        bilintrp(vens[nanal,1,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensus[nanal] =\
        bilintrp(uens[nanal,0,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensvs[nanal] =\
        bilintrp(vens[nanal,0,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxenstheta[nanal] =\
        bilintrp(theta,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
    hxensmean = hxens.mean(axis=0)
    obfits = ((thetaobs-hxensmean)**2).sum(axis=0)/(nobs-1)
    obbias = (thetaobs-hxensmean).mean(axis=0)
    obsprd = (((hxens-hxensmean)**2).sum(axis=0)/(nanals-1)).mean()
    hxensmeanu = hxensu.mean(axis=0)
    hxensmeanv = hxensv.mean(axis=0)
    hxensmeanus = hxensus.mean(axis=0)
    hxensmeanvs = hxensvs.mean(axis=0)
    hxensmeantheta = hxenstheta.mean(axis=0)
    obfitsuv =\
    ((uobsall[ntime]-hxensmeanu)**2+(vobsall[ntime]-hxensmeanv)**2).sum(axis=0)/(nobsall-1)
    obsprduv = (((hxensu-hxensmeanu)**2+(hxensv-hxensmeanv)**2).sum(axis=0)/(nanals-1)).mean()
    obfitsuvs =\
    ((usobsall[ntime]-hxensmeanus)**2+(vsobsall[ntime]-hxensmeanvs)**2).sum(axis=0)/(nobsall-1)
    obsprduvs = (((hxensus-hxensmeanus)**2+(hxensvs-hxensmeanvs)**2).sum(axis=0)/(nanals-1)).mean()
    obfitstheta =\
    ((thetaobsall[ntime]-hxensmeantheta)**2).sum(axis=0)/(nobsall-1)
    obsprdtheta = (((hxenstheta-hxensmeantheta)**2).sum(axis=0)/(nanals-1)).mean()
    t2 = time.clock()
    if profile: print 'cpu time for forward operator',t2-t1

    # print rms wind and temp errors (relative to truth) and spread at all ob locations.
    print "%s %g %g %g %g %g %g %g %g %g" %\
    (ntime,np.sqrt(obfitsuv),np.sqrt(obsprduv),np.sqrt(obfitsuvs),np.sqrt(obsprduvs),\
     np.sqrt(obfitstheta),np.sqrt(obsprdtheta),\
     np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias)

    # EnKF update
    t1 = time.clock()
    # create 1d state vector.
    for nanal in xrange(nanals):
        xens[nanal] = np.concatenate((uens[nanal,0,...],uens[nanal,1,...],\
                      vens[nanal,0,...],vens[nanal,1,...],thetaens[nanal])).ravel()
    # update state vector.
    xens =\
    serial_ensrf(xens,hxens,thetaobs,oberrvar,covlocal_tmp,hcovlocal_tmp,covinflate1,covinflate2)
    # 1d vector back to 3d arrays.
    for nanal in xrange(nanals):
        xsplit = np.split(xens[nanal],5)
        uens[nanal,0,...] = xsplit[0].reshape((sp.nlats,sp.nlons))
        uens[nanal,1,...] = xsplit[1].reshape((sp.nlats,sp.nlons))
        vens[nanal,0,...] = xsplit[2].reshape((sp.nlats,sp.nlons))
        vens[nanal,1,...] = xsplit[3].reshape((sp.nlats,sp.nlons))
        thetaens[nanal]   = xsplit[4].reshape((sp.nlats,sp.nlons))
        vrtspec[nanal], divspec[nanal] = sp.getvrtdivspec(uens[nanal],vens[nanal])
        thetaspec[nanal] = sp.grdtospec(thetaens[nanal])
    t2 = time.clock()
    if profile: print 'cpu time for EnKF update',t2-t1

    # run forecast ensemble to next analysis time
    t1 = time.clock()
    for nstep in xrange(nsteps):
        for nanal in xrange(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])
    t2 = time.clock()
    if profile:print 'cpu time for ens forecast',t2-t1

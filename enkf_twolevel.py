from pyspharm import Spharmt, regrid, regriduv
from twolevel import TwoLevel
import numpy as np
from netCDF4 import Dataset
import sys, time
from enkf_utils import  gcdist,bilintrp,serial_ensrf,gaspcohn,fibonacci_pts,\
                        letkf_calcwts,letkf_update

# EnKF cycling for two-level model with mid-level temp obs

if len(sys.argv) == 1:
   msg="""
python enkf_twolevel.py covlocal_scale covinflate1 (covinflate2)
   """
   raise SystemExit(msg)
# covariance localization length scale in meters.
covlocal_scale = float(sys.argv[1])
# covariance inflation parameter.
covinflate1 = float(sys.argv[2])
covinflate2 = 0.
# if covinflate2 not specified, RTPS inflation used.
# if covinflate2 given, use Hodyss & Campbell inflation
# with a = covinflate1, b = covinflate2
if len(sys.argv) > 3:
    covinflate2 = float(sys.argv[3])

profile = False # turn on profiling?
use_letkf = False # use LETKF?
if use_letkf:
    print('# using LETKF...')
else:
    print('# using serial EnSRF...')

nobs = 1024 # number of obs to assimilate
# each ob time nobs ob locations are randomly sampled (without
# replacement) from an evenly spaced fibonacci grid of nominally nobsall points.
# if nobsall = nobs, a fixed observing network is used.
nobsall = 10*nobs
#nobsall = nobs
nanals = 20 # ensemble members
oberrstdev = 1.0 # ob error in K
nassim = 2001 # assimilation times to run
gaussian=False # if True, use Gaussian function similar to Gaspari-Cohn
              # polynomial for localization.

# grid, time step info
nlons = 192; nlats = nlons//2  # number of longitudes/latitudes
ntrunc = nlons//3 # spectral truncation (for alias-free computations)
gridtype = 'gaussian'
dt = 3600. # time step in seconds
rsphere = 6.37122e6 # earth radius

# fix random seed for reproducibility.
np.random.seed(42)

# model nature run to sample initial ensemble and draw additive noise.
modelclimo_file = 'truth_twolevel_t%s_12h.nc' % ntrunc
# 'truth' nature run to sample obs
# (these two files can be the same for perfect model expts)
# file to sample additive noise.
truth_file = 'truth_twolevel_t%s_12h.nc' % ntrunc

# create spherical harmonic transform instance
sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype=gridtype)
spout = sp

models = []
for nanal in range(nanals):
    models.append(TwoLevel(sp,dt))

# weights for computing global means.
globalmeanwts = models[0].globalmeanwts

# read nature run, create obs.
nct = Dataset(truth_file)
lats = nct.variables['lat'][:]
lons = nct.variables['lon'][:]
spin = Spharmt(len(lons),len(lats),int(nct.ntrunc),rsphere,gridtype=gridtype)
samegrid = spin.nlons == spout.nlons and spin.nlats == spout.nlats
# find fibonacci points between latmin and latmax
if nobs == 1:
    nobsall = 1
    # single ob test.
    oblonsall = np.array([180.], np.float32)
    oblatsall = np.array([45.], np.float32)
else:
    oblatsall,oblonsall =\
    fibonacci_pts(nobsall,np.degrees(sp.lats[-1]),np.degrees(sp.lats[0]))

# reset nobsall
if nobs == nobsall:
    nobsall = len(oblatsall)
    nobs = nobsall
else:
    nobsall = len(oblatsall)

print('# %s obs to assimilate (out of %s) with ob err stdev = %s' % (nobs,nobsall,oberrstdev))
print('# covlocal_scale=%s km, covinflate1=%s covinflate2=%s' %\
(covlocal_scale/1000., covinflate1, covinflate2))
thetaobsall = np.empty((nassim,nobsall),np.float32)
utruth = np.empty((nassim,2,nlats,nlons),np.float32)
vtruth = np.empty((nassim,2,nlats,nlons),np.float32)
wtruth = np.empty((nassim,nlats,nlons),np.float32)
thetatruth = np.empty((nassim,nlats,nlons),np.float32)
oberrvar = np.empty(nobs,np.float32); oberrvar[:] = oberrstdev**2
obtimes = np.empty((nassim),np.float32)
for n in range(nassim):
    # flip latitude direction so lats are increasing (needed for interpolation)
    vrtspec_tmp,divspec_tmp =\
    spin.getvrtdivspec(nct.variables['u'][n],nct.variables['v'][n])
    w = models[0].dp*spin.spectogrd(divspec_tmp[1]-divspec_tmp[0])
    obtimes[n] = nct.variables['t'][n]
    thetaobsall[n] =\
    bilintrp(nct.variables['theta'][n,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    if samegrid:
       utruth[n] = nct.variables['u'][n]
       vtruth[n] = nct.variables['v'][n]
       thetatruth[n] = nct.variables['theta'][n]
       wtruth[n] = w
    else:
       utruth[n], vtruth[n] =\
       regriduv(spin,spout,nct.variables['u'][n],nct.variables['v'][n])
       thetatruth[n] = regrid(spin,spout,nct.variables['theta'][n],levs=1)
       wtruth[n] = regrid(spin,spout,w,levs=1)
nct.close()

# create initial ensemble by randomly sampling climatology
# of forecast model.
ncm = Dataset(modelclimo_file)
indx = np.random.choice(np.arange(len(ncm.variables['t'])),nanals,replace=False)
#indx[:] = 0 # for testing forward operator
thetaens = np.empty((nanals,sp.nlats,sp.nlons),np.float32)
wens = np.empty((nanals,sp.nlats,sp.nlons),np.float32)
uens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float32)
vens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float32)
thetinf = np.empty((sp.nlats,sp.nlons),np.float32)
uinf = np.empty((2,sp.nlats,sp.nlons),np.float32)
vinf = np.empty((2,sp.nlats,sp.nlons),np.float32)
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
vrtspec = np.empty((nanals,2,sp.nlm),np.complex128)
divspec = np.empty((nanals,2,sp.nlm),np.complex128)
thetaspec = np.empty((nanals,sp.nlm),np.complex128)
for nanal in range(nanals):
    vrtspec[nanal], divspec[nanal] = sp.getvrtdivspec(uens[nanal],vens[nanal])
    thetaspec[nanal] = sp.grdtospec(thetaens[nanal])
nvars = 5
ndim1 = sp.nlons*sp.nlats
ndim = nvars*ndim1
xens = np.empty((nanals,ndim),np.float32) # empty 1d state vector array

# precompute covariance localization for fixed observation network.
covlocal1 = np.zeros((nobsall,ndim1),np.float32)
hcovlocal = np.zeros((nobsall,nobsall),np.float32)
modellats = np.degrees(sp.lats)
modellons = np.degrees(sp.lons)
modellons,modellats = np.meshgrid(modellons,modellats)
for nob in range(nobsall):
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
print('# fhassim,nsteps = ',fhassim,nsteps)

savedata = None
#savedata = 'enkf_twolevel_test.nc'
nout = 0
if savedata is not None:
    ncout = Dataset(savedata,'w',format='NETCDF4_CLASSIC')
    ncout.rsphere = rsphere
    ncout.gridtype = gridtype
    ncout.ntrunc = ntrunc
    ncout.dt = dt
    ncout.nassim = nassim
    ncout.covinflate1 = covinflate1
    ncout.covinflate2 = covinflate2
    ncout.covlocal_scale = covlocal_scale
    ncout.truth_file = truth_file
    ncout.modelclimo_file = modelclimo_file
    ncout.nobs = nobs
    ncout.nobsall = nobsall
    ncout.oberrstdev = oberrstdev
    atts = ['grav','omega','cp','rgas','p0','ptop','delth','efold','ndiss','tdrag','tdiab','umax','jetexp']
    for att in atts:
        ncout.setncattr(att,models[0].__dict__[att])
    lat = ncout.createDimension('lat',sp.nlats)
    lon = ncout.createDimension('lon',sp.nlons)
    level = ncout.createDimension('level',2)
    timed = ncout.createDimension('t',None)
    u_ensmeanb = ncout.createVariable('uensmeanb',np.float3232,('t','level','lat','lon'),zlib=False)
    u_ensmeanb.units = 'meters per second'
    v_ensmeanb = ncout.createVariable('vensmeanb',np.float3232,('t','level','lat','lon'),zlib=False)
    v_ensmeanb.units = 'meters per second'
    thet_ensmeanb = ncout.createVariable('thetensmeanb',np.float3232,('t','lat','lon'),zlib=False)
    thet_ensmeanb.units = 'K'
    w_ensmeanb = ncout.createVariable('wensmeanb',np.float3232,('t','lat','lon'),zlib=False)
    w_ensmeanb.units = 'Pa per second'
    u_ensmeana = ncout.createVariable('uensmeana',np.float3232,('t','level','lat','lon'),zlib=False)
    u_ensmeana.units = 'meters per second'
    v_ensmeana = ncout.createVariable('vensmeana',np.float3232,('t','level','lat','lon'),zlib=False)
    v_ensmeana.units = 'meters per second'
    thet_ensmeana = ncout.createVariable('thetensmeana',np.float3232,('t','lat','lon'),zlib=False)
    thet_ensmeana.units = 'K'
    w_ensmeana = ncout.createVariable('wensmeana',np.float3232,('t','lat','lon'),zlib=False)
    w_ensmeana.units = 'Pa per second'
    u_truth = ncout.createVariable('utruth',np.float3232,('t','level','lat','lon'),zlib=False)
    u_truth.units = 'meters per second'
    v_truth = ncout.createVariable('vtruth',np.float3232,('t','level','lat','lon'),zlib=False)
    v_truth.units = 'meters per second'
    thet_truth = ncout.createVariable('thettruth',np.float3232,('t','lat','lon'),zlib=False)
    thet_truth.units = 'K'
    w_truth = ncout.createVariable('wtruth',np.float3232,('t','lat','lon'),zlib=False)
    w_truth.units = 'Pa per second'
    u_sprdb = ncout.createVariable('usprdb',np.float3232,('t','level','lat','lon'),zlib=False)
    u_sprdb.units = 'meters per second'
    v_sprdb = ncout.createVariable('vsprdb',np.float3232,('t','level','lat','lon'),zlib=False)
    v_sprdb.units = 'meters per second'
    thet_sprdb = ncout.createVariable('thetsprdb',np.float3232,('t','lat','lon'),zlib=False)
    thet_sprdb.units = 'K'
    w_sprdb = ncout.createVariable('wsprdb',np.float3232,('t','lat','lon'),zlib=False)
    w_sprdb.units = 'Pa per second'
    u_sprda = ncout.createVariable('usprda',np.float3232,('t','level','lat','lon'),zlib=False)
    u_sprda.units = 'meters per second'
    v_sprda = ncout.createVariable('vsprda',np.float3232,('t','level','lat','lon'),zlib=False)
    v_sprda.units = 'meters per second'
    thet_sprda = ncout.createVariable('thetsprda',np.float3232,('t','lat','lon'),zlib=False)
    thet_sprda.units = 'K'
    w_sprda = ncout.createVariable('wsprda',np.float3232,('t','lat','lon'),zlib=False)
    w_sprda.units = 'Pa per second'
    uinflation = ncout.createVariable('uinflation',np.float3232,('t','level','lat','lon'),zlib=False)
    uinflation.units = 'meters per second'
    vinflation = ncout.createVariable('vinflation',np.float3232,('t','level','lat','lon'),zlib=False)
    vinflation.units = 'meters per second'
    thetinflation = ncout.createVariable('thetinflation',np.float3232,('t','lat','lon'),zlib=False)
    thetinflation.units = 'K'
    times = ncout.createVariable('t',np.float32,('t',))
    lats = ncout.createVariable('lat',np.float32,('lat',))
    lats.units = 'degrees north'
    lats[:] = np.degrees(sp.lats)
    lons = ncout.createVariable('lon',np.float32,('lon',))
    lons.units = 'degrees east'
    lons[:] = np.degrees(sp.lons)

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[0]*3600.

for ntime in range(nassim):

    # check model clock
    if models[0].t/3600. != obtimes[ntime]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t/3600., obtimes[ntime]))

    # compute forward operator.
    t1 = time.time()
    # ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float32)
    if nobs == nobsall:
        oblats = oblatsall; oblons = oblonsall
        thetaobs = thetaobsall[ntime]
        obindx = np.arange(nobs)
        if use_letkf:
            covlocal_tmp = covlocal1
        else:
            covlocal_tmp = covlocal
            hcovlocal_tmp = hcovlocal
    elif nobsall > nobs:
        obindx = np.random.choice(np.arange(nobsall),size=nobs,replace=False)
        oblats = oblatsall[obindx]; oblons = oblonsall[obindx]
        thetaobs = np.ascontiguousarray(thetaobsall[ntime,obindx])
        if use_letkf:
            covlocal_tmp = np.ascontiguousarray(covlocal1[obindx,:])
        else:
            covlocal_tmp = np.ascontiguousarray(covlocal[obindx,:])
            hcovlocal_tmp = np.ascontiguousarray(hcovlocal[obindx,:][:,obindx])
    else:
        raise ValueError('nobsall must be >= nobs')
    if oberrstdev > 0.: # add observation error
        thetaobs += np.random.normal(scale=oberrstdev,size=nobs) # add ob errors
    for nanal in range(nanals):
        # inverse transform to grid.
        uens[nanal],vens[nanal] = sp.getuv(vrtspec[nanal],divspec[nanal])
        thetaens[nanal] = sp.spectogrd(thetaspec[nanal])
        wens[nanal] =\
        models[nanal].dp*sp.spectogrd(divspec[nanal,1,...]-divspec[nanal,0,...])
        # forward operator calculation.
        theta = thetaens[nanal,::-1,:]
        hxens[nanal] = bilintrp(theta,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
    hxensmean = hxens.mean(axis=0)
    obfits = ((thetaobs-hxensmean)**2).sum(axis=0)/(nobs-1)
    obbias = (thetaobs-hxensmean).mean(axis=0)
    obsprd = (((hxens-hxensmean)**2).sum(axis=0)/(nanals-1)).mean()
    uensmean = uens.mean(axis=0); vensmean = vens.mean(axis=0)
    thetensmean = thetaens.mean(axis=0)
    wensmean = wens.mean(axis=0)
    theterr = (thetatruth[ntime]-thetensmean)**2
    theterrb = np.sqrt((theterr*globalmeanwts).sum())
    werr = (wtruth[ntime]-wensmean)**2
    werrb = np.sqrt((werr*globalmeanwts).sum())
    thetsprd = ((thetaens-thetensmean)**2).sum(axis=0)/(nanals-1)
    thetsprdb = np.sqrt((thetsprd*globalmeanwts).sum())
    wsprd = ((wens-wensmean)**2).sum(axis=0)/(nanals-1)
    wsprdb = np.sqrt((wsprd*globalmeanwts).sum())
    uverr1 = 0.5*((utruth[ntime,1,:,:]-uensmean[1])**2+(vtruth[ntime,1,:,:]-vensmean[1])**2)
    uverr1b = np.sqrt((uverr1*globalmeanwts).sum())
    usprd = ((uens-uensmean)**2).sum(axis=0)/(nanals-1)
    vsprd = ((vens-vensmean)**2).sum(axis=0)/(nanals-1)
    uvsprd1 = 0.5*(usprd[1]+vsprd[1])
    uvsprd1b = np.sqrt((uvsprd1*globalmeanwts).sum())
    uverr0 = 0.5*((utruth[ntime,0,:,:]-uensmean[0])**2+(vtruth[ntime,0,:,:]-vensmean[0])**2)
    uverr0b = np.sqrt((uverr0*globalmeanwts).sum())
    uvsprd0 = 0.5*(usprd[0]+vsprd[0])
    uvsprd0b = np.sqrt((uvsprd0*globalmeanwts).sum())
    t2 = time.time()
    if profile: print('cpu time for forward operator',t2-t1)

    if savedata:
        u_ensmeanb[nout] = uensmean
        v_ensmeanb[nout] = vensmean
        thet_ensmeanb[nout] = thetensmean
        w_ensmeanb[nout] = wensmean
        u_sprdb[nout] = usprd
        v_sprdb[nout] = vsprd
        thet_sprdb[nout] = thetsprd
        w_sprdb[nout] = wsprd

    # EnKF update
    t1 = time.time()
    # create 1d state vector.
    for nanal in range(nanals):
        if use_letkf:
            uens1 = uens[nanal].reshape((2,ndim1))
            vens1 = vens[nanal].reshape((2,ndim1))
            thetaens1 = thetaens[nanal].reshape((ndim1,))
            for n in range(ndim1):
                xens[nanal,nvars*n] = uens1[0,n]
                xens[nanal,nvars*n+1] = uens1[1,n]
                xens[nanal,nvars*n+2] = vens1[0,n]
                xens[nanal,nvars*n+3] = vens1[1,n]
                xens[nanal,nvars*n+4] = thetaens1[n]
                n += 1
        else:
            xens[nanal] = np.concatenate((uens[nanal,0,...],uens[nanal,1,...],\
                          vens[nanal,0,...],vens[nanal,1,...],thetaens[nanal])).ravel()
    xmean_b = xens.mean(axis=0); xprime = xens-xmean_b
    # background spread.
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)
    # update state vector.
    if use_letkf:
        wts = letkf_calcwts(hxens,thetaobs-hxensmean,oberrvar,covlocal_ob=covlocal_tmp)
        xens = letkf_update(xens,wts)
    else:
        xens = serial_ensrf(xens,hxens,thetaobs,oberrvar,covlocal_tmp,hcovlocal_tmp)
    xmean = xens.mean(axis=0); xprime = xens-xmean
    # analysis spread
    asprd = (xprime**2).sum(axis=0)/(nanals-1)
    # posterior inflation
    if covinflate2 == 0:
        # relaxation to prior spread (Whitaker and Hamill)
        # relax variance
        #inflation_factor = np.sqrt(1.+covinflate1*(fsprd-asprd)/asprd)
        # relax st. deviation
        inflation_factor = 1.+covinflate1*(np.sqrt(fsprd)-np.sqrt(asprd))/np.sqrt(asprd)
    else:
        # Hodyss and Campbell
        inc = xmean - xmean_b
        inflation_factor = np.sqrt(covinflate1 + \
        (asprd/fsprd**2)*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1))))
    xprime = xprime*inflation_factor
    xens = xmean + xprime
    # 1d vector back to 3d arrays.
    for nanal in range(nanals):
        if use_letkf:
            for n in range(ndim1):
                uens1[0,n] = xens[nanal,nvars*n]
                uens1[1,n] = xens[nanal,nvars*n+1]
                vens1[0,n] = xens[nanal,nvars*n+2]
                vens1[1,n] = xens[nanal,nvars*n+3]
                thetaens1[n] = xens[nanal,nvars*n+4]
                n += 1
            uens[nanal] = uens1.reshape((2,sp.nlats,sp.nlons))
            vens[nanal] = vens1.reshape((2,sp.nlats,sp.nlons))
            thetaens[nanal] = thetaens1.reshape((sp.nlats,sp.nlons,))
        else:
            xsplit = np.split(xens[nanal],5)
            uens[nanal,0,...] = xsplit[0].reshape((sp.nlats,sp.nlons))
            uens[nanal,1,...] = xsplit[1].reshape((sp.nlats,sp.nlons))
            vens[nanal,0,...] = xsplit[2].reshape((sp.nlats,sp.nlons))
            vens[nanal,1,...] = xsplit[3].reshape((sp.nlats,sp.nlons))
            thetaens[nanal]   = xsplit[4].reshape((sp.nlats,sp.nlons))
        vrtspec[nanal], divspec[nanal] = sp.getvrtdivspec(uens[nanal],vens[nanal])
        thetaspec[nanal] = sp.grdtospec(thetaens[nanal])
        wens[nanal] =\
        models[nanal].dp*sp.spectogrd(divspec[nanal,1,...]-divspec[nanal,0,...])
    infsplit = np.split(inflation_factor,5)
    uinf[0,...] = infsplit[0].reshape((sp.nlats,sp.nlons))
    uinf[1,...] = infsplit[1].reshape((sp.nlats,sp.nlons))
    vinf[0,...] = infsplit[2].reshape((sp.nlats,sp.nlons))
    vinf[1,...] = infsplit[3].reshape((sp.nlats,sp.nlons))
    thetinf   = infsplit[4].reshape((sp.nlats,sp.nlons))
    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    uensmean = uens.mean(axis=0); vensmean = vens.mean(axis=0)
    thetensmean = thetaens.mean(axis=0)
    wensmean = wens.mean(axis=0)
    theterr = (thetatruth[ntime]-thetensmean)**2
    theterra = np.sqrt((theterr*globalmeanwts).sum())
    werr = (wtruth[ntime]-wensmean)**2
    werra = np.sqrt((werr*globalmeanwts).sum())
    thetsprd = ((thetaens-thetensmean)**2).sum(axis=0)/(nanals-1)
    thetsprda = np.sqrt((thetsprd*globalmeanwts).sum())
    wsprd = ((wens-wensmean)**2).sum(axis=0)/(nanals-1)
    wsprda = np.sqrt((wsprd*globalmeanwts).sum())
    uverr1 = 0.5*((utruth[ntime,1,:,:]-uensmean[1])**2+(vtruth[ntime,1,:,:]-vensmean[1])**2)
    uverr1a = np.sqrt((uverr1*globalmeanwts).sum())
    usprd = ((uens-uensmean)**2).sum(axis=0)/(nanals-1)
    vsprd = ((vens-vensmean)**2).sum(axis=0)/(nanals-1)
    uvsprd1 = 0.5*(usprd[1]+vsprd[1])
    uvsprd1a = np.sqrt((uvsprd1*globalmeanwts).sum())
    uverr0 = 0.5*((utruth[ntime,0,:,:]-uensmean[0])**2+(vtruth[ntime,0,:,:]-vensmean[0])**2)
    uverr0a = np.sqrt((uverr0*globalmeanwts).sum())
    uvsprd0 = 0.5*(usprd[0]+vsprd[0])
    uvsprd0a = np.sqrt((uvsprd0*globalmeanwts).sum())
    # print rms wind and temp error & spread (relative to truth for analysis
    # and background), plus innov stats for background.
    print("%s %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,theterra,thetsprda,theterrb,thetsprdb,\
           werra,wsprda,werrb,wsprdb,\
           uverr0a,uvsprd0a,uverr0b,uvsprd0b,\
           uverr1a,uvsprd1a,uverr1b,uvsprd1b,
           np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias))

    # write out data.
    if savedata:
        u_ensmeana[nout] = uensmean
        v_ensmeana[nout] = vensmean
        thet_ensmeana[nout] = thetensmean
        w_ensmeana[nout] = wensmean
        u_sprda[nout] = usprd
        v_sprda[nout] = vsprd
        thet_sprda[nout] = thetsprd
        w_sprda[nout] = wsprd
        u_truth[nout] = utruth[ntime]
        v_truth[nout] = vtruth[ntime]
        thet_truth[nout] = thetatruth[ntime]
        w_truth[nout] = wtruth[ntime]
        thetinflation[nout] = thetinf
        uinflation[nout] = uinf
        vinflation[nout] = vinf
        times = obtimes[ntime]
        nout += 1

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nstep in range(nsteps):
        for nanal in range(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])
    t2 = time.time()
    if profile:print('cpu time for ens forecast',t2-t1)

from pyspharm import Spharmt, regrid, regriduv
from twolevel import TwoLevel
import numpy as np
from netCDF4 import Dataset
import sys, time
from enkf_utils import  gcdist,bilintrp,serial_ensrf,gaspcohn,fibonacci_pts,\
                        letkf_calcwts,letkf_update

# EnKF cycling for two-level model with mid-level temp and vertical mean wind obs

if len(sys.argv) == 1:
   msg="""
python enkf_twolevel.py covlocal_scale covinflate
   """
   raise SystemExit(msg)
# covariance localization length scale in meters.
covlocal_scale = float(sys.argv[1])
# covariance inflation parameter (relaxation to prior spread if only one parameter given).
covinflate1 = float(sys.argv[2]) # large scale
covinflate2 = float(sys.argv[3]) # small scale
smoothfact = float(sys.argv[4]) # filter parameter

profile = False # turn on profiling?
use_letkf = False # use LETKF?
if use_letkf:
    print('# using LETKF...')
else:
    print('# using serial EnSRF...')

nobs = 1024 # number of ob locations to assimilate
# each ob time nobs ob locations are randomly sampled (without
# replacement) from an evenly spaced fibonacci grid of nominally nobsall points.
# if nobsall = nobs, a fixed observing network is used.
nobsall = 10*nobs
#nobsall = nobs
nanals = 20 # ensemble members
wind_obs = True # assimilate vertical mean winds also
oberrstdev = 1.0 # temp ob error in K
oberrstdevw = 2.5 # ob err for vertical mean wind in mps
nassim = 2101 # assimilation times to run
gaussian=False # if True, use Gaussian function similar to Gaspari-Cohn
               # polynomial for localization.

# grid, time step info
nlons = 192; nlats = nlons//2  # number of longitudes/latitudes
ntrunc = nlons//3 # spectral truncation (for alias-free computations)
gridtype = 'gaussian'
#div2_diff_efold = 1.e30
div2_diff_efold = 3600. # divergence diffusion to damp GW

# fix random seed for reproducibility.
np.random.seed(42)

# model nature run to sample initial ensemble and draw additive noise.
modelclimo_file = 'truth_twolevel_t%s_6h.nc' % ntrunc
ncm = Dataset(modelclimo_file)
dt = ncm.dt
rsphere = ncm.rsphere
# 'truth' nature run to sample obs
# (these two files can be the same for perfect model expts)
# file to sample additive noise.
truth_file = 'truth_twolevel_t%s_6h.nc' % ntrunc

# create spherical harmonic transform instance
sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype=gridtype)
spout = sp
smoothspec = np.exp(sp.lap*sp.rsphere**2/(smoothfact*(smoothfact+1.)))

models = []
for nanal in range(nanals):
    models.append(TwoLevel(sp,dt,div2_diff_efold=div2_diff_efold))

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

print('# %s obs to assimilate (out of %s) with ob err stdev = (%s, %s)' % (nobs,nobsall,oberrstdev,oberrstdevw))
print('# covlocal_scale=%s km, covinflate1=%s, covinflate2=%s, smoothfact=%s, wind_obs=%s' %\
(covlocal_scale/1000., covinflate1, covinflate2, smoothfact, wind_obs))
thetaobsall = np.empty((nassim,nobsall),np.float32)
if wind_obs:
    uobsall = np.empty((nassim,nobsall),np.float32)
    vobsall = np.empty((nassim,nobsall),np.float32)
utruth = np.empty((nassim,2,nlats,nlons),np.float32)
vtruth = np.empty((nassim,2,nlats,nlons),np.float32)
wtruth = np.empty((nassim,nlats,nlons),np.float32)
thetatruth = np.empty((nassim,nlats,nlons),np.float32)
if wind_obs:
    oberrvar = np.empty(3*nobs,np.float32); oberrvar[:nobs] = oberrstdev**2; oberrvar[nobs:] = oberrstdevw**2
else:
    oberrvar = np.empty(nobs,np.float32); oberrvar[:] = oberrstdev**2
obtimes = np.empty((nassim),np.float32)
for n in range(nassim):
    # flip latitude direction so lats are increasing (needed for interpolation)
    vrtspec_tmp,divspec_tmp =\
    spin.getvrtdivspec(nct.variables['u'][n],nct.variables['v'][n])
    um = (nct.variables['u'][n]).mean(axis=0)
    vm = (nct.variables['v'][n]).mean(axis=0)
    w = models[0].dp*spin.spectogrd(divspec_tmp[1]-divspec_tmp[0])
    obtimes[n] = nct.variables['t'][n]
    thetaobsall[n] =\
    bilintrp(nct.variables['theta'][n,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    if wind_obs:
        uobsall[n] =\
        bilintrp(um[::-1,:],lons,lats[::-1],oblonsall,oblatsall)
        vobsall[n] =\
        bilintrp(vm[::-1,:],lons,lats[::-1],oblonsall,oblatsall)
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
indx = np.random.choice(np.arange(len(ncm.variables['t'])),nanals,replace=False)
#indx[:] = 0 # for testing forward operator
thetaens = np.empty((nanals,sp.nlats,sp.nlons),np.float32)
wens = np.empty((nanals,sp.nlats,sp.nlons),np.float32)
uens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float32)
vens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float32)
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
xens_l  = np.empty_like(xens)
xens_s  = np.empty_like(xens)

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
    ncout.smoothfact = smoothfact
    ncout.covlocal_scale = covlocal_scale
    ncout.truth_file = truth_file
    ncout.modelclimo_file = modelclimo_file
    ncout.nobs = nobs
    ncout.nobsall = nobsall
    atts = ['grav','omega','cp','rgas','p0','ptop','delth','efold','ndiss','tdrag','tdiab','umax','jetexp']
    for att in atts:
        ncout.setncattr(att,models[0].__dict__[att])
    lat = ncout.createDimension('lat',sp.nlats)
    lon = ncout.createDimension('lon',sp.nlons)
    nob = ncout.createDimension('nobs',nobs)
    level = ncout.createDimension('level',2)
    ens = ncout.createDimension('ens',nanals)
    timed = ncout.createDimension('t',None)
    oblats_var = ncout.createVariable('oblats',np.float32,('t','nobs'),zlib=True)
    oblons_var = ncout.createVariable('oblons',np.float32,('t','nobs'),zlib=True)
    thetobs_var = ncout.createVariable('thetobs',np.float32,('t','nobs'),zlib=True)
    thetobs_var.oberrstdev = oberrstdev
    if wind_obs:
        uobs_var = ncout.createVariable('uobs',np.float32,('t','nobs'),zlib=True)
        uobs_var.oberrstdev = oberrstdevw
        vobs_var = ncout.createVariable('vobs',np.float32,('t','nobs'),zlib=True)
        vobs_var.oberrstdev = oberrstdevw
    u_ensb = ncout.createVariable('uensb',np.float32,('t','ens','level','lat','lon'),zlib=True)
    u_ensb.units = 'meters per second'
    v_ensb = ncout.createVariable('vensb',np.float32,('t','ens','level','lat','lon'),zlib=True)
    v_ensb.units = 'meters per second'
    thet_ensb = ncout.createVariable('thetensb',np.float32,('t','ens','lat','lon'),zlib=True)
    thet_ensb.units = 'K'
    w_ensb = ncout.createVariable('wensb',np.float32,('t','ens','lat','lon'),zlib=True)
    w_ensb.units = 'K'
    u_ensmeanb = ncout.createVariable('uensmeanb',np.float32,('t','level','lat','lon'),zlib=True)
    u_ensmeanb.units = 'meters per second'
    v_ensmeanb = ncout.createVariable('vensmeanb',np.float32,('t','level','lat','lon'),zlib=True)
    v_ensmeanb.units = 'meters per second'
    thet_ensmeanb = ncout.createVariable('thetensmeanb',np.float32,('t','lat','lon'),zlib=True)
    thet_ensmeanb.units = 'K'
    w_ensmeanb = ncout.createVariable('wensmeanb',np.float32,('t','lat','lon'),zlib=True)
    w_ensmeanb.units = 'Pa per second'
    u_ensa = ncout.createVariable('uensa',np.float32,('t','ens','level','lat','lon'),zlib=True)
    u_ensa.units = 'meters per second'
    v_ensa = ncout.createVariable('vensa',np.float32,('t','ens','level','lat','lon'),zlib=True)
    v_ensa.units = 'meters per second'
    thet_ensa = ncout.createVariable('thetensa',np.float32,('t','ens','lat','lon'),zlib=True)
    thet_ensa.units = 'K'
    w_ensa = ncout.createVariable('wensa',np.float32,('t','ens','lat','lon'),zlib=True)
    w_ensa.units = 'K'
    u_ensmeana = ncout.createVariable('uensmeana',np.float32,('t','level','lat','lon'),zlib=True)
    u_ensmeana.units = 'meters per second'
    v_ensmeana = ncout.createVariable('vensmeana',np.float32,('t','level','lat','lon'),zlib=True)
    v_ensmeana.units = 'meters per second'
    thet_ensmeana = ncout.createVariable('thetensmeana',np.float32,('t','lat','lon'),zlib=True)
    thet_ensmeana.units = 'K'
    w_ensmeana = ncout.createVariable('wensmeana',np.float32,('t','lat','lon'),zlib=True)
    w_ensmeana.units = 'Pa per second'
    u_truth = ncout.createVariable('utruth',np.float32,('t','level','lat','lon'),zlib=True)
    u_truth.units = 'meters per second'
    v_truth = ncout.createVariable('vtruth',np.float32,('t','level','lat','lon'),zlib=True)
    v_truth.units = 'meters per second'
    thet_truth = ncout.createVariable('thettruth',np.float32,('t','lat','lon'),zlib=True)
    thet_truth.units = 'K'
    w_truth = ncout.createVariable('wtruth',np.float32,('t','lat','lon'),zlib=True)
    w_truth.units = 'Pa per second'
    u_sprdb = ncout.createVariable('usprdb',np.float32,('t','level','lat','lon'),zlib=True)
    u_sprdb.units = 'meters per second'
    v_sprdb = ncout.createVariable('vsprdb',np.float32,('t','level','lat','lon'),zlib=True)
    v_sprdb.units = 'meters per second'
    thet_sprdb = ncout.createVariable('thetsprdb',np.float32,('t','lat','lon'),zlib=True)
    thet_sprdb.units = 'K'
    w_sprdb = ncout.createVariable('wsprdb',np.float32,('t','lat','lon'),zlib=True)
    w_sprdb.units = 'Pa per second'
    u_sprda = ncout.createVariable('usprda',np.float32,('t','level','lat','lon'),zlib=True)
    u_sprda.units = 'meters per second'
    v_sprda = ncout.createVariable('vsprda',np.float32,('t','level','lat','lon'),zlib=True)
    v_sprda.units = 'meters per second'
    thet_sprda = ncout.createVariable('thetsprda',np.float32,('t','lat','lon'),zlib=True)
    thet_sprda.units = 'K'
    w_sprda = ncout.createVariable('wsprda',np.float32,('t','lat','lon'),zlib=True)
    w_sprda.units = 'Pa per second'
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
    if wind_obs:
        hxens = np.empty((nanals,3*nobs),np.float32)
    else:
        hxens = np.empty((nanals,nobs),np.float32)
    if nobs == nobsall:
        oblats = oblatsall; oblons = oblonsall
        thetaobs = thetaobsall[ntime]
        if wind_obs:
            uobs = uobsall[ntime]
            vobs = vobsall[ntime]
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
        if wind_obs:
            uobs = np.ascontiguousarray(uobsall[ntime,obindx])
            vobs = np.ascontiguousarray(vobsall[ntime,obindx])
        if use_letkf:
            covlocal_tmp = np.ascontiguousarray(covlocal1[obindx,:])
        else:
            covlocal_tmp = np.ascontiguousarray(covlocal[obindx,:])
            hcovlocal_tmp = np.ascontiguousarray(hcovlocal[obindx,:][:,obindx])
    else:
        raise ValueError('nobsall must be >= nobs')
    thetaobs += np.random.normal(scale=oberrstdev,size=nobs) # add ob errors
    for nanal in range(nanals):
        # inverse transform to grid.
        uens[nanal],vens[nanal] = sp.getuv(vrtspec[nanal],divspec[nanal])
        thetaens[nanal] = sp.spectogrd(thetaspec[nanal])
        wens[nanal] =\
        models[nanal].dp*sp.spectogrd(divspec[nanal,1,...]-divspec[nanal,0,...])
        # forward operator calculation.
        theta = thetaens[nanal,::-1,:]
        hxens[nanal,:nobs] = bilintrp(theta,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
        if wind_obs:
            um = (uens[nanal,:,::-1,:]).mean(axis=0)
            vm = (vens[nanal,:,::-1,:]).mean(axis=0)
            hxens[nanal,nobs:2*nobs] = bilintrp(um,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
            hxens[nanal,2*nobs:] = bilintrp(vm,np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
    hxensmean = hxens.mean(axis=0)
    obfits = ((thetaobs-hxensmean[:nobs])**2).sum(axis=0)/(nobs-1)
    obbias = (thetaobs-hxensmean[:nobs]).mean(axis=0)
    obsprd = (((hxens[:,:nobs]-hxensmean[:nobs])**2).sum(axis=0)/(nanals-1)).mean()
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
        u_ensb[nout] = uens
        v_ensb[nout] = vens
        thet_ensb[nout] = thetaens
        w_ensb[nout] = wens
        u_ensmeanb[nout] = uensmean
        v_ensmeanb[nout] = vensmean
        thet_ensmeanb[nout] = thetensmean
        w_ensmeanb[nout] = wensmean
        u_sprdb[nout] = usprd
        v_sprdb[nout] = vsprd
        thet_sprdb[nout] = thetsprd
        w_sprdb[nout] = wsprd
        oblats_var[nout] = oblats
        oblons_var[nout] = oblons
        thetobs_var[nout] = thetaobs
        if wind_obs:
            uobs_var[nout] = uobs
            vobs_var[nout] = vobs

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
    for nanal in range(nanals):
        xsplit = np.split(xprime[nanal],5)
        uens[nanal,0,...] = xsplit[0].reshape((sp.nlats,sp.nlons))
        uens[nanal,1,...] = xsplit[1].reshape((sp.nlats,sp.nlons))
        vens[nanal,0,...] = xsplit[2].reshape((sp.nlats,sp.nlons))
        vens[nanal,1,...] = xsplit[3].reshape((sp.nlats,sp.nlons))
        thetaens[nanal]   = xsplit[4].reshape((sp.nlats,sp.nlons))
        vrtspec_tmp,divspec_tmp = sp.getvrtdivspec(uens[nanal],vens[nanal])
        uens_l,vens_l = sp.getuv(smoothspec*vrtspec_tmp, smoothspec*divspec_tmp)
        thetaspec_tmp = sp.grdtospec(thetaens[nanal])
        thetaens_l = sp.spectogrd(smoothspec*thetaspec_tmp)
        uens_s = uens[nanal]-uens_l
        vens_s = vens[nanal]-vens_l
        thetaens_s = thetaens[nanal]-thetaens_l
        #print(nanal, uens_l.min(), uens_l.max(), uens_s.min(), uens_s.max(), thetaens_l.min(), thetaens_l.max(), thetaens_s.min(), thetaens_s.max())
        xens_l[nanal] = np.concatenate((uens_l[0,...],uens_l[1,...],\
                      vens_l[0,...],vens_l[1,...],thetaens_l)).ravel()
        xens_s[nanal] = np.concatenate((uens_s[0,...],uens_s[1,...],\
                      vens_s[0,...],vens_s[1,...],thetaens_l)).ravel()
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)
    fsprd_l = (xens_l**2).sum(axis=0)/(nanals-1)
    fsprd_s = (xens_s**2).sum(axis=0)/(nanals-1)
    #print(fsprd.mean(), fsprd_l.mean(), fsprd_s.mean())
    # update state vector.
    if use_letkf:
        ominusf = np.empty_like(oberrvar)
        ominusf[:nobs] = thetaobs-hxensmean[:nobs]
        if wind_obs:
            ominusf[nobs:2*nobs] = uobs-hxensmean[nobs:2*nobs]
            ominusf[2*nobs:] = vobs-hxensmean[2*nobs:]
            wts = letkf_calcwts(hxens,ominusf,oberrvar,covlocal_ob=np.vstack((covlocal_tmp,covlocal_tmp,covlocal_tmp)))
        else:
            wts = letkf_calcwts(hxens,ominusf,oberrvar,covlocal_ob=covlocal_tmp)
        xens = letkf_update(xens,wts)
    else:
        obs = np.empty_like(oberrvar)
        obs[:nobs] = thetaobs[:nobs]
        if wind_obs:
            obs[nobs:2*nobs] = uobs
            obs[2*nobs:]     = vobs
            xens = serial_ensrf(xens,hxens,obs,oberrvar,np.vstack((covlocal_tmp,covlocal_tmp,covlocal_tmp)),\
                                np.block([[hcovlocal_tmp,hcovlocal_tmp,hcovlocal_tmp],\
                                          [hcovlocal_tmp,hcovlocal_tmp,hcovlocal_tmp],\
                                          [hcovlocal_tmp,hcovlocal_tmp,hcovlocal_tmp]]))
        else:
            xens = serial_ensrf(xens,hxens,obs,oberrvar,covlocal_tmp,hcovlocal_tmp)
    xmean = xens.mean(axis=0); xprime = xens-xmean

    asprd = (xprime**2).sum(axis=0)/(nanals-1)
    if use_letkf:
        raise SystemExit
    else:
        for nanal in range(nanals):
            xsplit = np.split(xprime[nanal],5)
            uens[nanal,0,...] = xsplit[0].reshape((sp.nlats,sp.nlons))
            uens[nanal,1,...] = xsplit[1].reshape((sp.nlats,sp.nlons))
            vens[nanal,0,...] = xsplit[2].reshape((sp.nlats,sp.nlons))
            vens[nanal,1,...] = xsplit[3].reshape((sp.nlats,sp.nlons))
            thetaens[nanal]   = xsplit[4].reshape((sp.nlats,sp.nlons))
            vrtspec_tmp,divspec_tmp = sp.getvrtdivspec(uens[nanal],vens[nanal])
            uens_l,vens_l = sp.getuv(smoothspec*vrtspec_tmp, smoothspec*divspec_tmp)
            thetaspec_tmp = sp.grdtospec(thetaens[nanal])
            thetaens_l = sp.spectogrd(smoothspec*thetaspec_tmp)
            uens_s = uens[nanal]-uens_l
            vens_s = vens[nanal]-vens_l
            thetaens_s = thetaens[nanal]-thetaens_l
            #print(nanal, uens_l.min(), uens_l.max(), uens_s.min(), uens_s.max(), thetaens_l.min(), thetaens_l.max(), thetaens_s.min(), thetaens_s.max())
            xens_l[nanal] = np.concatenate((uens_l[0,...],uens_l[1,...],\
                            vens_l[0,...],vens_l[1,...],thetaens_l)).ravel()
            xens_s[nanal] = np.concatenate((uens_s[0,...],uens_s[1,...],\
                            vens_s[0,...],vens_s[1,...],thetaens_l)).ravel()
    asprd_l = (xens_l**2).sum(axis=0)/(nanals-1)
    asprd_s = (xens_s**2).sum(axis=0)/(nanals-1)
    #print(asprd.mean(), asprd_l.mean(), asprd_s.mean())
    # posterior inflation
    # relaxation to prior spread (Whitaker and Hamill, https://doi.org/10.1175/MWR-D-11-00276.1)
    inflation_factor = 1.+covinflate1*(np.sqrt(fsprd_l)-np.sqrt(asprd_l))/np.sqrt(asprd_l)
    xprime_l = xens_l*inflation_factor
    # posterior inflation
    # relaxation to prior spread (Whitaker and Hamill, https://doi.org/10.1175/MWR-D-11-00276.1)
    inflation_factor = 1.+covinflate2*(np.sqrt(fsprd_s)-np.sqrt(asprd_s))/np.sqrt(asprd_s)
    xprime_s = xens_s*inflation_factor
    xprime = xprime_l + xprime_s

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
    #print("%s %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g" %\
    #(ntime,theterra,thetsprda,theterrb,thetsprdb,\
    #       werra,wsprda,werrb,wsprdb,\
    #       uverr0a,uvsprd0a,uverr0b,uvsprd0b,\
    #       uverr1a,uvsprd1a,uverr1b,uvsprd1b,
    #       np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias))
    print("%s %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,theterrb,thetsprdb,\
           werrb,wsprdb,\
           uverr0b,uvsprd0b,\
           uverr1b,uvsprd1b,\
           np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias))

    # write out data.
    if savedata:
        u_ensa[nout] = uens
        v_ensa[nout] = vens
        thet_ensa[nout] = thetaens
        w_ensa[nout] = wens
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

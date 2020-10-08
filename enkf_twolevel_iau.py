from pyspharm import Spharmt, regrid, regriduv
from twolevel import TwoLevel
import numpy as np
from netCDF4 import Dataset
import sys, time, os, cPickle
from enkf_utils import  gcdist,bilintrp,serial_ensrf,gaspcohn,fibonacci_pts,\
                        letkf_calcwts,letkf_update

# EnKF cycling with IAU for two-level model with mid-level temp obs

if len(sys.argv) == 1:
   msg="""
python enkf_twolevel.py covlocal_scale covinflate obshr_interval use_letkf (nstart)
   """
   raise SystemExit(msg)
# covariance localization length scale in meters.
covlocal_scale = float(sys.argv[1])
# covariance inflation parameter.
covinflate = float(sys.argv[2])
# interval to compute increments (in hours) within IAU window.
# 0 means 3DIAU, < 0 means no IAU.
obshr_interval = float(sys.argv[3])
use_letkf = bool(int(sys.argv[4]))
#use_letkf = True

profile = bool(os.getenv('PROFILE')) # turn on profiling?
if use_letkf:
    print('# using LETKF...')
else:
    print('# using serial EnSRF...')

nobs = 256 # number of obs to assimilate
# each ob time nobs ob locations are randomly sampled (without
# replacement) from an evenly spaced fibonacci grid of nominally nobsall points.
# if nobsall = nobs, a fixed observing network is used.
nobsall = nobs
nanals = 10 # ensemble members
oberrstdev = 1.0 # ob error in meters
nassim = 2201 # assimilation times to run
gaussian=True # if True, use Gaussian function similar to Gaspari-Cohn
              # polynomial for localization.

# grid, time step info
nlons = 96; nlats = nlons/2  # number of longitudes/latitudes
ntrunc = 32 # spectral truncation (for alias-free computations)
gridtype = 'gaussian'
dt = 3600. #  time step in seconds
rsphere = 6.37122e6 # earth radius

# set random states
rs1 = np.random.RandomState(seed=42) # reproduce obs
rs2 = np.random.RandomState(seed=None)  # different ensemble each run

# model nature run to sample initial ensemble and draw additive noise.
modelclimo_file = 'truth_twolevel_t%s_12h.nc' % ntrunc
# 'truth' nature run to sample obs
truth_file = 'truth_twolevel_t32_12h.nc'

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
    oblonsall = np.array([180.], np.float)
    oblatsall = np.array([45.], np.float)
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
print('# covlocal_scale=%s km, covinflate=%s obshr_interval=%s' %\
(covlocal_scale/1000., covinflate, obshr_interval))
thetaobsall = np.empty((nassim,nobsall),np.float)
utruth = np.empty((nassim,2,nlats,nlons),np.float)
vtruth = np.empty((nassim,2,nlats,nlons),np.float)
wtruth = np.empty((nassim,nlats,nlons),np.float)
thetatruth = np.empty((nassim,nlats,nlons),np.float)
oberrvar = np.empty(nobs,np.float); oberrvar[:] = oberrstdev**2
obtimes = np.empty((nassim),np.float)
for n in range(nassim):
    # flip latitude direction so lats are increasing (needed for interpolation)
    theta = nct.variables['theta'][n,::-1,:]
    vrtspec_tmp,divspec_tmp =\
    spin.getvrtdivspec(nct.variables['u'][n,...],nct.variables['v'][n,...])
    w = models[0].dp*spin.spectogrd(divspec_tmp[1]-divspec_tmp[0])
    obtimes[n] = nct.variables['t'][n]
    thetaobsall[n] = bilintrp(theta,lons,lats[::-1],oblonsall,oblatsall)
    if samegrid:
       utruth[n] = nct.variables['u'][n]
       vtruth[n] = nct.variables['v'][n]
       wtruth[n] = w
       thetatruth[n] = nct.variables['theta'][n]
    else:
       utruth[n], vtruth[n] =\
       regriduv(spin,spout,nct.variables['u'][n],nct.variables['v'][n])
       thetatruth[n] = regrid(spin,spout,nct.variables['theta'][n],levs=1)
       wtruth[n] = regrid(spin,spout,w,levs=1)
nct.close()

# create initial ensemble by randomly sampling climatology
# of forecast model.
ncm = Dataset(modelclimo_file)
thetaens = np.empty((nanals,sp.nlats,sp.nlons),np.float)
wens = np.empty((nanals,sp.nlats,sp.nlons),np.float)
uens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float)
vens = np.empty((nanals,2,sp.nlats,sp.nlons),np.float)
theta_modelclim = ncm.variables['theta']
u_modelclim = ncm.variables['u']
v_modelclim = ncm.variables['v']
fhassim = obtimes[1]-obtimes[0] # assim interval  (assumed constant)
nsteps = int(fhassim*3600/models[0].dt) # time steps in assim interval
if obshr_interval > 0:
    nsteps_iau = int(fhassim/obshr_interval)
    nsteps_periau = int(obshr_interval*3600./dt)
else:
    nsteps_iau = 0
    nsteps_periau = 0
vrtspec_fcst = np.empty((nsteps+1,nanals,2,sp.nlm),np.complex)
divspec_fcst = np.empty((nsteps+1,nanals,2,sp.nlm),np.complex)
thetaspec_fcst = np.empty((nsteps+1,nanals,sp.nlm),np.complex)
vrtspec_inc = np.empty((nsteps+1,nanals,2,sp.nlm),np.complex)
divspec_inc = np.empty((nsteps+1,nanals,2,sp.nlm),np.complex)
thetaspec_inc = np.empty((nsteps+1,nanals,sp.nlm),np.complex)
vrtspec_inc1 = np.empty((nsteps_iau+1,nanals,2,sp.nlm),np.complex)
divspec_inc1 = np.empty((nsteps_iau+1,nanals,2,sp.nlm),np.complex)
thetaspec_inc1 = np.empty((nsteps_iau+1,nanals,sp.nlm),np.complex)
indx = rs2.choice(np.arange(len(ncm.variables['t'])),nanals,replace=False)
print('# fhassim,nsteps,nsteps_iau = ',fhassim,nsteps,nsteps_iau)

nanal=0
for n in indx:
    thetag = theta_modelclim[n]
    ug = u_modelclim[n]
    vg = v_modelclim[n]
    vrtspec_fcst[0,nanal,...], divspec_fcst[0,nanal,...] = \
    sp.getvrtdivspec(ug,vg)
    thetaspec_fcst[0,nanal,...] = sp.grdtospec(thetag)
    for nstep in range(nsteps):
        vrtspec_fcst[nstep+1,nanal,...],divspec_fcst[nstep+1,nanal,...],thetaspec_fcst[nstep+1,nanal,...] = \
        models[nanal].rk4step(vrtspec_fcst[nstep,nanal,...],divspec_fcst[nstep,nanal,...],thetaspec_fcst[nstep,nanal,...])
    nanal += 1
ncm.close()

nvars = 5
ndim = nvars*sp.nlons*sp.nlats
xens = np.empty((nanals,ndim),np.float) # empty 1d state vector array

# precompute covariance localization for fixed observation network.
ndim1 = sp.nlons*sp.nlats
covlocal1 = np.zeros((nobsall,ndim1),np.float)
hcovlocal = np.zeros((nobsall,nobsall),np.float)
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
covlocal1 = np.where(covlocal1 < 1.e-13, 1.e-13, covlocal1)
covlocal = np.tile(covlocal1,5)
# simple constant IAU weights
wts_iau = np.ones(nsteps,np.float)/(3600.*fhassim)
if obshr_interval < 0.:
    # regular enkf (dump all the increment in at
    # one time step in the middle of the window).
    wts_iau[:] = 0.
    wts_iau[nsteps/2]=1./dt

for ntime in range(nassim):

    # compute forward operator.
    t1 = time.clock()
    # ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float)
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
        obindx = rs1.choice(np.arange(nobsall),size=nobs,replace=False)
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
        thetaobs += rs1.normal(scale=oberrstdev,size=nobs) # add ob errors
    for nanal in range(nanals):
        # inverse transform to grid at obs time (center of IAU window).
        uens[nanal],vens[nanal] = sp.getuv(vrtspec_fcst[nsteps/2,nanal,...],divspec_fcst[nsteps/2,nanal,...])
        thetaens[nanal] = sp.spectogrd(thetaspec_fcst[nsteps/2,nanal,...])
        wens[nanal] = models[nanal].dp*sp.spectogrd(divspec_fcst[nsteps/2,nanal,1,:]-divspec_fcst[nsteps/2,nanal,0,:])
        # forward operator calculation (obs all at center of assim window).
        hxens[nanal] = bilintrp(thetaens[nanal,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
    hxensmean = hxens.mean(axis=0)
    obfits = ((thetaobs-hxensmean)**2).sum(axis=0)/(nobs-1)
    obbias = (thetaobs-hxensmean).mean(axis=0)
    obsprd = (((hxens-hxensmean)**2).sum(axis=0)/(nanals-1)).mean()
    uensmean = uens.mean(axis=0); vensmean = vens.mean(axis=0)
    thetensmean = thetaens.mean(axis=0)
    wensmean = wens.mean(axis=0)
    theterr = (thetatruth[ntime]-thetensmean)**2
    theterr = np.sqrt((theterr*globalmeanwts).sum())
    werr = (wtruth[ntime]-wensmean)**2
    werr = np.sqrt((werr*globalmeanwts).sum())
    thetsprd = ((thetaens-thetensmean)**2).sum(axis=0)/(nanals-1)
    thetsprd = np.sqrt((thetsprd*globalmeanwts).sum())
    wsprd = ((wens-wensmean)**2).sum(axis=0)/(nanals-1)
    wsprd = np.sqrt((wsprd*globalmeanwts).sum())
    uverr1 = 0.5*((utruth[ntime,1,:,:]-uensmean[1])**2+(vtruth[ntime,1,:,:]-vensmean[1])**2)
    uverr1 = np.sqrt((uverr1*globalmeanwts).sum())
    usprd = ((uens-uensmean)**2).sum(axis=0)/(nanals-1)
    vsprd = ((vens-vensmean)**2).sum(axis=0)/(nanals-1)
    uvsprd1 = 0.5*(usprd[1]+vsprd[1])
    uvsprd1 = np.sqrt((uvsprd1*globalmeanwts).sum())
    uverr0 = 0.5*((utruth[ntime,0,:,:]-uensmean[0])**2+(vtruth[ntime,0,:,:]-vensmean[0])**2)
    uverr0 = np.sqrt((uverr0*globalmeanwts).sum())
    uvsprd0 = 0.5*(usprd[0]+vsprd[0])
    uvsprd0 = np.sqrt((uvsprd0*globalmeanwts).sum())
    # print rms wind, w and temp error & spread plus
    # plus innov stats for background only (at center of window).
    print("%s %5i %4.2f %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,int(covlocal_scale/1000),covinflate,theterr,thetsprd,werr,wsprd,uverr0,uvsprd0,uverr1,uvsprd1,\
           np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias))
    if ntime==nassim-1: break

    t2 = time.clock()
    if profile: print('cpu time for forward operator',t2-t1)

    # EnKF update
    t1 = time.clock()
    if use_letkf:
        wts = letkf_calcwts(hxens,thetaobs-hxensmean,oberrvar,covlocal_ob=covlocal_tmp)
    nstep_iau = 0
    for nstep in range(nsteps+1):
        if nsteps_periau != 0:
            # interpolated IAU
            calc_inc = nstep % nsteps_periau == 0
        else:
            # constant IAU
            calc_inc = nstep == nsteps/2
        if calc_inc:
            #print(nstep,nstep_iau)
            for nanal in range(nanals):
                # inverse transform to grid.
                ug,vg = sp.getuv(vrtspec_fcst[nstep,nanal,...],divspec_fcst[nstep,nanal,...])
                thetag = sp.spectogrd(thetaspec_fcst[nstep,nanal,...])
                # create 1d state vector.
                if use_letkf:
                    uens1 = ug.reshape((2,ndim1))
                    vens1 = vg.reshape((2,ndim1))
                    thetaens1 = thetag.reshape((ndim1,))
                    for n in range(ndim1):
                        xens[nanal,nvars*n] = uens1[0,n]
                        xens[nanal,nvars*n+1] = uens1[1,n]
                        xens[nanal,nvars*n+2] = vens1[0,n]
                        xens[nanal,nvars*n+3] = vens1[1,n]
                        xens[nanal,nvars*n+4] = thetaens1[n]
                        n += 1
                else:
                    xens[nanal] = np.concatenate((ug[0,...],ug[1,...],\
                                  vg[0,...],vg[1,...],thetag)).ravel()
            xens_b = xens.copy()
            xmean_b = xens_b.mean(axis=0); xprime_b = xens_b-xmean_b
            # background spread.
            fsprd = (xprime_b**2).sum(axis=0)/(nanals-1)
            # update state vector.
            if use_letkf:
                xens = letkf_update(xens,wts)
            else:
                xens =\
                serial_ensrf(xens,hxens,thetaobs,oberrvar,covlocal_tmp,hcovlocal_tmp)
            xmean = xens.mean(axis=0); xprime = xens-xmean
            # analysis spread
            asprd = (xprime**2).sum(axis=0)/(nanals-1)
            # posterior inflation
            if covinflate < 1:
                # relaxation to prior stdev (Whitaker and Hamill)
                asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
                inflation_factor = 1.+covinflate*(fsprd-asprd)/asprd
            else:
                # Hodyss and Campbell
                inc = xmean - xmean_b
                inflation_factor = np.sqrt(1. + \
                covinflate*(asprd/fsprd**2)*((fsprd/nanals) + (2.*inc**2/(nanals-1))))
                #inflation_factor = np.sqrt(covinflate1 + \
                #(asprd/fsprd**2)*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1))))
            xprime = xprime*inflation_factor
            xens = xmean + xprime
            #print((xens-xens_b).min(),(xens-xens_b).max())
            # 1d vector back to 3d arrays.
            for nanal in range(nanals):
                if use_letkf:
                    for n in range(ndim1):
                        uens1[0,n] = xens[nanal,nvars*n]-xens_b[nanal,nvars*n]
                        uens1[1,n] =\
                        xens[nanal,nvars*n+1]-xens_b[nanal,nvars*n+1]
                        vens1[0,n] =\
                        xens[nanal,nvars*n+2]-xens_b[nanal,nvars*n+2]
                        vens1[1,n] =\
                        xens[nanal,nvars*n+3]-xens_b[nanal,nvars*n+3]
                        thetaens1[n] =\
                        xens[nanal,nvars*n+4]-xens_b[nanal,nvars*n+4]
                        n += 1
                    ug = uens1.reshape((2,sp.nlats,sp.nlons))
                    vg = vens1.reshape((2,sp.nlats,sp.nlons))
                    thetag = thetaens1.reshape((sp.nlats,sp.nlons,))
                else:
                    xsplit = np.split(xens[nanal]-xens_b[nanal],5)
                    ug[0,...] = xsplit[0].reshape((sp.nlats,sp.nlons))
                    ug[1,...] = xsplit[1].reshape((sp.nlats,sp.nlons))
                    vg[0,...] = xsplit[2].reshape((sp.nlats,sp.nlons))
                    vg[1,...] = xsplit[3].reshape((sp.nlats,sp.nlons))
                    thetag    = xsplit[4].reshape((sp.nlats,sp.nlons))
                vrtspec_inc1[nstep_iau,nanal,...],  \
                divspec_inc1[nstep_iau,nanal,...] = \
                sp.getvrtdivspec(ug,vg)
                thetaspec_inc1[nstep_iau,nanal,...] = sp.grdtospec(thetag)
                # inverse transform to grid.
                #ug,vg =\
                #sp.getuv(vrtspec_inc1[nstep_iau,nanal,...],divspec_inc1[nstep_iau,nanal,...])
                #thetag = sp.spectogrd(thetaspec_inc1[nstep_iau,nanal,...])
                #print(nstep_iau,nanal,ug.min(),ug.max(),thetag.min(),thetag.max())
            nstep_iau += 1
    t2 = time.clock()
    if profile: print('cpu time for EnKF update',t2-t1)

    # linearly interpolate increments to every time in IAU window.
    if nsteps_iau == 0:
        for n in range(nsteps+1):
            vrtspec_inc[n] = vrtspec_inc1[0]
            divspec_inc[n] = divspec_inc1[0]
            thetaspec_inc[n] = thetaspec_inc1[0]
    else:
        nstep = 0
        for nstep_iau in range(nsteps_iau):
            for nn in range(nsteps/(nsteps_iau)):
                itime  = nstep_iau + float(nn*nsteps_iau)/float(nsteps)
                itimel = int(nstep_iau)
                alpha = itime - float(itimel)
                itimer = min(nsteps_iau,itimel+1)
                #print(nstep,itimel,1.-alpha,itimer,alpha)
                vrtspec_inc[nstep] =\
                ((1.0-alpha)*vrtspec_inc1[itimel]+alpha*vrtspec_inc1[itimer])
                divspec_inc[nstep] =\
                ((1.0-alpha)*divspec_inc1[itimel]+alpha*divspec_inc1[itimer])
                thetaspec_inc[nstep] =\
                ((1.0-alpha)*thetaspec_inc1[itimel]+alpha*thetaspec_inc1[itimer])
                #for nanal in range(nanals):
                #    # inverse transform to grid.
                #    ug,vg = sp.getuv(vrtspec_inc[nstep,nanal,...],divspec_inc[nstep,nanal,...])
                #    thetag = sp.spectogrd(thetaspec_inc[nstep,nanal,...])
                #    print(nstep,nanal,ug.min(),ug.max(),thetag.min(),thetag.max())
                nstep += 1
        vrtspec_inc[nsteps] = vrtspec_inc1[nsteps_iau]
        divspec_inc[nsteps] = divspec_inc1[nsteps_iau]
        thetaspec_inc[nsteps] = thetaspec_inc1[nsteps_iau]

    # run forecast ensemble over IAU window, adding increments
    # slowly.
    vrtspec = vrtspec_fcst[0].copy()
    divspec = divspec_fcst[0].copy()
    thetaspec = thetaspec_fcst[0].copy()
    # set model clocks at beginning of IAU window
    for nanal in range(nanals):
        models[nanal].t = obtimes[ntime]*3600. - 0.5*fhassim*3600.
    for nstep in range(nsteps):
        if nstep == nsteps/2:
            # check model clock
            if models[0].t/3600. != obtimes[ntime]:
                raise ValueError('model/ob time mismatch %s vs %s' %\
                (models[0].t/3600., obtimes[ntime]))
        # add of bit of the increment to the state before every time step.
        vrtspec += wts_iau[nstep]*dt*vrtspec_inc[nstep,:,...]
        divspec += wts_iau[nstep]*dt*divspec_inc[nstep,:,...]
        thetaspec += wts_iau[nstep]*dt*thetaspec_inc[nstep,:,...]
        # advance model one time step.
        for nanal in range(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])

    # run forecast ensemble from end of IAU interval
    # to beginning of next IAU window (no forcing)
    t1 = time.clock()
    for nstep in range(nsteps):
        vrtspec_fcst[nstep] = vrtspec[:]
        divspec_fcst[nstep] = divspec[:]
        thetaspec_fcst[nstep] = thetaspec[:]
        if nstep == nsteps/2:
            # check model clock
            if models[0].t/3600. != obtimes[ntime+1]:
                raise ValueError('model/ob time mismatch %s vs %s' %\
                (models[0].t/3600., obtimes[ntime+1]))
        for nanal in range(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])
    vrtspec_fcst[nsteps] = vrtspec[:]
    divspec_fcst[nsteps] = divspec[:]
    thetaspec_fcst[nsteps] = thetaspec[:]

    t2 = time.clock()
    if profile:print('cpu time for ens forecast',t2-t1)

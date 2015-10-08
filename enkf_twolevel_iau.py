from pyspharm import Spharmt
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
# relaxation to prior spread inflation parameter.
covinflate = float(sys.argv[2])
# interval to compute increments (in hours) within IAU window.
#obshr_interval = float(sys.argv[3])
#use_letkf = bool(int(sys.argv[4]))
obshr_interval = 0 # 0 for no IAU
use_letkf = False

profile = bool(os.getenv('PROFILE')) # turn on profiling?
if use_letkf:
    print '# using LETKF...'
else:
    print '# using serial EnSRF...'

# spinup parameters
covlocal_scale_spinup = 2000.e3
covinflate_spinup = 0.8
ntimes_spinup = 0

nobs = 500 # number of obs to assimilate
# each ob time nobs ob locations are randomly sampled (without
# replacement) from an evenly spaced fibonacci grid of nominally nobsall points.
# if nobsall = nobs, a fixed observing network is used.
nobsall = 10*nobs
nanals = 20 # ensemble members
oberrstdev = 0.5 # ob error in meters
nassim = 1501 # assimilation times to run
nassim_run = 1501
gaussian=True # if True, use Gaussian function similar to Gaspari-Cohn
              # polynomial for localization.

# grid, time step info
nlons = 96; nlats = nlons/2  # number of longitudes/latitudes
ntrunc = 32 # spectral truncation (for alias-free computations)
gridtype = 'regular'
dt = 3600. #  time step in seconds
rsphere = 6.37122e6 # earth radius

# fix random seed for reproducibility.
np.random.seed(42)

# model nature run to sample initial ensemble and draw additive noise.
modelclimo_file = 'truth_twolevel_t%s_12h.nc' % ntrunc
# 'truth' nature run to sample obs
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
spt = Spharmt(len(lons),len(lats),ntrunc,rsphere,gridtype=gridtype)
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
print '# covlocal_scale=%s km, covinflate=%s, obshr_interval=%s' %\
(covlocal_scale/1000., covinflate, obshr_interval)
thetaobsall = np.empty((nassim,nobsall),np.float)
# keep truth upper layer winds interpolated to all ob locations for validation.
uobsall = np.empty((nassim,nobsall),np.float)
vobsall = np.empty((nassim,nobsall),np.float)
wobsall = np.empty((nassim,nobsall),np.float)
usobsall = np.empty((nassim,nobsall),np.float)
vsobsall = np.empty((nassim,nobsall),np.float)
oberrvar = np.empty(nobs,np.float); oberrvar[:] = oberrstdev**2
obtimes = np.empty((nassim),np.float)
for n in xrange(nassim):
    # flip latitude direction so lats are increasing (needed for interpolation)
    theta = nct.variables['theta'][n,::-1,:]
    vrtspec_tmp,divspec_tmp =\
    spt.getvrtdivspec(nct.variables['u'][n,...],nct.variables['v'][n,...])
    w = models[0].dp*spt.spectogrd(divspec_tmp[1]-divspec_tmp[0])[::-1]
    obtimes[n] = nct.variables['t'][n]
    thetaobsall[n] = bilintrp(theta,lons,lats[::-1],oblonsall,oblatsall)
    wobsall[n] = bilintrp(w,lons,lats[::-1],oblonsall,oblatsall)
    uobsall[n] = bilintrp(nct.variables['u'][n,1,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    vobsall[n] = bilintrp(nct.variables['v'][n,1,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    usobsall[n] = bilintrp(nct.variables['u'][n,0,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
    vsobsall[n] = bilintrp(nct.variables['v'][n,0,::-1,:],lons,lats[::-1],oblonsall,oblatsall)
nct.close()

# create initial ensemble by randomly sampling climatology
# of forecast model.
ncm = Dataset(modelclimo_file)
thetag = np.empty((sp.nlats,sp.nlons),np.float)
ug = np.empty((2,sp.nlats,sp.nlons),np.float)
vg = np.empty((2,sp.nlats,sp.nlons),np.float)
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
indx = np.random.choice(np.arange(len(ncm.variables['t'])),nanals,replace=False)
print '# fhassim,nsteps,nsteps_iau = ',fhassim,nsteps,nsteps_iau

if len(sys.argv) > 5:
    nstart = int(sys.argv[5])
    nend = min(nassim, nstart + nassim_run)
    print '# restarting from saved ensemble at assim step %s' % nstart
    vrtspec_fcst = cPickle.load(open('vrtspec_fcst.pkl',mode='rb'))
    divspec_fcst = cPickle.load(open('divspec_fcst.pkl',mode='rb'))
    thetaspec_fcst = cPickle.load(open('thetaspec_fcst.pkl',mode='rb'))
else:
    nstart = 0
    nend = nassim_run
    nanal=0
    for n in indx:
        thetag = theta_modelclim[n]
        ug = u_modelclim[n]
        vg = v_modelclim[n]
        vrtspec_fcst[0,nanal,...], divspec_fcst[0,nanal,...] = \
        sp.getvrtdivspec(ug,vg)
        thetaspec_fcst[0,nanal,...] = sp.grdtospec(thetag)
        for nstep in xrange(nsteps):
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
covlocal1_spinup = np.zeros((nobsall,ndim1),np.float)
hcovlocal = np.zeros((nobsall,nobsall),np.float)
hcovlocal_spinup = np.zeros((nobsall,nobsall),np.float)
modellats = np.degrees(sp.lats)
modellons = np.degrees(sp.lons)
modellons,modellats = np.meshgrid(modellons,modellats)
for nob in xrange(nobsall):
    r = sp.rsphere*gcdist(np.radians(oblonsall[nob]),np.radians(oblatsall[nob]),
    np.radians(modellons.ravel()),np.radians(modellats.ravel()))
    taper = gaspcohn(r/covlocal_scale,gaussian=gaussian)
    taper_spinup = gaspcohn(r/covlocal_scale_spinup,gaussian=gaussian)
    covlocal1[nob,:] = taper
    covlocal1_spinup[nob,:] = taper_spinup
    r = sp.rsphere*gcdist(np.radians(oblonsall[nob]),np.radians(oblatsall[nob]),
    np.radians(oblonsall),np.radians(oblatsall))
    taper = gaspcohn(r/covlocal_scale,gaussian=gaussian)
    taper_spinup = gaspcohn(r/covlocal_scale_spinup,gaussian=gaussian)
    hcovlocal[nob,:] = taper
    hcovlocal_spinup[nob,:] = taper_spinup
covlocal1 = np.where(covlocal1 < 1.e-13, 1.e-13, covlocal1)
covlocal1_spinup = np.where(covlocal1_spinup < 1.e-13, 1.e-13, covlocal1_spinup)
covlocal = np.tile(covlocal1,5)
covlocal_spinup = np.tile(covlocal1_spinup,5)
# simple constant IAU weights
wts_iau = np.ones(nsteps,np.float)/(3600.*fhassim)
if obshr_interval < 0.:
    # regular enkf (dump all the increment in at
    # one time step in the middle of the window).
    wts_iau[:] = 0.
    wts_iau[nstep/2]=1./dt

if nstart > 0:
    # make sure random sequence is the same for a restart.
    for ntime in xrange(0,nstart):
        if nobsall > nobs:
            obindx = np.random.choice(np.arange(nobsall),size=nobs,replace=False)
        if oberrstdev > 0.: # add observation error
            oberrs = np.random.normal(scale=oberrstdev,size=nobs) # add ob errors

for ntime in xrange(nstart,nend):

    # compute forward operator.
    t1 = time.clock()
    # ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float)
    hxensu = np.empty((nanals,nobsall),np.float)
    hxensv = np.empty((nanals,nobsall),np.float)
    hxensw = np.empty((nanals,nobsall),np.float)
    hxensus = np.empty((nanals,nobsall),np.float)
    hxensvs = np.empty((nanals,nobsall),np.float)
    hxenstheta = np.empty((nanals,nobsall),np.float)
    if nobs == nobsall:
        oblats = oblatsall; oblons = oblonsall
        thetaobs = thetaobsall[ntime]
        obindx = np.arange(nobs)
        if use_letkf:
            if ntime < ntimes_spinup:
                covlocal_tmp = covlocal1_spinup
            else:
                covlocal_tmp = covlocal1
        else:
            if ntime < ntimes_spinup:
                covlocal_tmp = covlocal_spinup
                hcovlocal_tmp = hcovlocal_spinup
            else:
                covlocal_tmp = covlocal
                hcovlocal_tmp = hcovlocal
    elif nobsall > nobs:
        obindx = np.random.choice(np.arange(nobsall),size=nobs,replace=False)
        oblats = oblatsall[obindx]; oblons = oblonsall[obindx]
        thetaobs = np.ascontiguousarray(thetaobsall[ntime,obindx])
        if use_letkf:
            if ntime < ntimes_spinup:
                covlocal_tmp = np.ascontiguousarray(covlocal1_spinup[obindx,:])
            else:
                covlocal_tmp = np.ascontiguousarray(covlocal1[obindx,:])
        else:
            if ntime < ntimes_spinup:
                covlocal_tmp = np.ascontiguousarray(covlocal_spinup[obindx,:])
                hcovlocal_tmp = np.ascontiguousarray(hcovlocal_spinup[obindx,:][:,obindx])
            else:
                covlocal_tmp = np.ascontiguousarray(covlocal[obindx,:])
                hcovlocal_tmp = np.ascontiguousarray(hcovlocal[obindx,:][:,obindx])
    else:
        raise ValueError('nobsall must be >= nobs')
    if oberrstdev > 0.: # add observation error
        thetaobs += np.random.normal(scale=oberrstdev,size=nobs) # add ob errors
    for nanal in xrange(nanals):
        # inverse transform to grid at obs time (center of IAU window).
        ug,vg = sp.getuv(vrtspec_fcst[nsteps/2,nanal,...],divspec_fcst[nsteps/2,nanal,...])
        thetag = sp.spectogrd(thetaspec_fcst[nsteps/2,nanal,...])
        wg = models[nanal].dp*sp.spectogrd(divspec_fcst[nsteps/2,nanal,1,:]-divspec_fcst[nsteps/2,nanal,0,:])
        # forward operator calculation.
        hxens[nanal] = bilintrp(thetag[::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblons,oblats)
        hxensu[nanal] =\
        bilintrp(ug[1,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensv[nanal] =\
        bilintrp(vg[1,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensw[nanal] =\
        bilintrp(wg[::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensus[nanal] =\
        bilintrp(ug[0,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxensvs[nanal] =\
        bilintrp(vg[0,::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
        hxenstheta[nanal] =\
                bilintrp(thetag[::-1,:],np.degrees(sp.lons),np.degrees(sp.lats[::-1]),oblonsall,oblatsall)
    hxensmean = hxens.mean(axis=0)
    obfits = ((thetaobs-hxensmean)**2).sum(axis=0)/(nobs-1)
    obbias = (thetaobs-hxensmean).mean(axis=0)
    obsprd = (((hxens-hxensmean)**2).sum(axis=0)/(nanals-1)).mean()
    hxensmeanu = hxensu.mean(axis=0)
    hxensmeanv = hxensv.mean(axis=0)
    hxensmeanw = hxensw.mean(axis=0)
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
    obfitsw =\
    ((wobsall[ntime]-hxensmeanw)**2).sum(axis=0)/(nobsall-1)
    obsprdtheta = (((hxenstheta-hxensmeantheta)**2).sum(axis=0)/(nanals-1)).mean()
    obsprdw = (((hxensw-hxensmeanw)**2).sum(axis=0)/(nanals-1)).mean()
    t2 = time.clock()
    if profile: print 'cpu time for forward operator',t2-t1

    # print rms wind and temp errors (relative to truth) and spread at all ob locations.
    print "%s %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,np.sqrt(obfitsuv),np.sqrt(obsprduv),np.sqrt(obfitsuvs),np.sqrt(obsprduvs),\
     np.sqrt(obfitstheta),np.sqrt(obsprdtheta),np.sqrt(obfitsw),np.sqrt(obsprdw),\
     np.sqrt(obfits),np.sqrt(obsprd+oberrstdev**2),obbias)
    if ntime==nend-1: break

    # EnKF update
    if ntime < ntimes_spinup:
        covinf = covinflate_spinup
    else:
        covinf = covinflate
    t1 = time.clock()
    if use_letkf:
        wts = letkf_calcwts(hxens,thetaobs-hxensmean,oberrvar,covlocal_ob=covlocal_tmp)
    nstep_iau = 0
    for nstep in xrange(nsteps+1):
        if nsteps_periau != 0:
            # interpolated IAU
            calc_inc = nstep % nsteps_periau == 0
        else:
            # constant IAU
            calc_inc = nstep == nsteps/2
        if calc_inc:
            #print nstep,nstep_iau
            for nanal in xrange(nanals):
                # inverse transform to grid.
                ug,vg = sp.getuv(vrtspec_fcst[nstep,nanal,...],divspec_fcst[nstep,nanal,...])
                thetag = sp.spectogrd(thetaspec_fcst[nstep,nanal,...])
                # create 1d state vector.
                if use_letkf:
                    uens1 = ug.reshape((2,ndim1))
                    vens1 = vg.reshape((2,ndim1))
                    thetaens1 = thetag.reshape((ndim1,))
                    for n in xrange(ndim1):
                        xens[nanal,nvars*n] = uens1[0,n]
                        xens[nanal,nvars*n+1] = uens1[1,n]
                        xens[nanal,nvars*n+2] = vens1[0,n]
                        xens[nanal,nvars*n+3] = vens1[1,n]
                        xens[nanal,nvars*n+4] = thetaens1[n]
                        n += 1
                else:
                    xens[nanal] = np.concatenate((ug[0,...],ug[1,...],\
                                  vg[0,...],vg[1,...],thetag)).ravel()
            xens_fg = xens.copy()
            # update state vector.
            if use_letkf:
                xens = letkf_update(xens,wts,covinf)
            else:
                xens =\
                serial_ensrf(xens,hxens,thetaobs,oberrvar,covlocal_tmp,hcovlocal_tmp,covinf)
            #print (xens-xens_fg).min(),(xens-xens_fg).max()
            # 1d vector back to 3d arrays.
            for nanal in xrange(nanals):
                if use_letkf:
                    for n in xrange(ndim1):
                        uens1[0,n] = xens[nanal,nvars*n]
                        uens1[1,n] = xens[nanal,nvars*n+1]
                        vens1[0,n] = xens[nanal,nvars*n+2]
                        vens1[1,n] = xens[nanal,nvars*n+3]
                        thetaens1[n] = xens[nanal,nvars*n+4]
                        n += 1
                    ug = uens1.reshape((2,sp.nlats,sp.nlons))
                    vg = vens1.reshape((2,sp.nlats,sp.nlons))
                    thetag = thetaens1.reshape((sp.nlats,sp.nlons,))
                else:
                    xsplit = np.split(xens[nanal]-xens_fg[nanal],5)
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
                #print nstep_iau,nanal,ug.min(),ug.max(),thetag.min(),thetag.max()
            nstep_iau += 1
    t2 = time.clock()
    if profile: print 'cpu time for EnKF update',t2-t1

    # linearly interpolate increments to every time in IAU window.
    if nsteps_iau == 0:
        for n in xrange(nsteps+1):
            vrtspec_inc[n] = vrtspec_inc1[0]
            divspec_inc[n] = divspec_inc1[0]
            thetaspec_inc[n] = thetaspec_inc1[0]
    else:
        nstep = 0
        for nstep_iau in xrange(nsteps_iau):
            for nn in xrange(nsteps/(nsteps_iau)):
                itime  = nstep_iau + float(nn*nsteps_iau)/float(nsteps)
                itimel = int(nstep_iau)
                alpha = itime - float(itimel)
                itimer = min(nsteps_iau,itimel+1)
                #print nstep,itimel,1.-alpha,itimer,alpha
                vrtspec_inc[nstep] =\
                ((1.0-alpha)*vrtspec_inc1[itimel]+alpha*vrtspec_inc1[itimer])
                divspec_inc[nstep] =\
                ((1.0-alpha)*divspec_inc1[itimel]+alpha*divspec_inc1[itimer])
                thetaspec_inc[nstep] =\
                ((1.0-alpha)*thetaspec_inc1[itimel]+alpha*thetaspec_inc1[itimer])
                #for nanal in xrange(nanals):
                #    # inverse transform to grid.
                #    ug,vg = sp.getuv(vrtspec_inc[nstep,nanal,...],divspec_inc[nstep,nanal,...])
                #    thetag = sp.spectogrd(thetaspec_inc[nstep,nanal,...])
                #    print nstep,nanal,ug.min(),ug.max(),thetag.min(),thetag.max()
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
    for nanal in xrange(nanals):
        models[nanal].t = obtimes[ntime]*3600. - 0.5*fhassim*3600.
    for nstep in xrange(nsteps):
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
        for nanal in xrange(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])

    # run forecast ensemble from end of IAU interval
    # to beginning of next IAU window (no forcing)
    t1 = time.clock()
    for nstep in xrange(nsteps):
        vrtspec_fcst[nstep] = vrtspec[:]
        divspec_fcst[nstep] = divspec[:]
        thetaspec_fcst[nstep] = thetaspec[:]
        if nstep == nsteps/2:
            # check model clock
            if models[0].t/3600. != obtimes[ntime+1]:
                raise ValueError('model/ob time mismatch %s vs %s' %\
                (models[0].t/3600., obtimes[ntime+1]))
        for nanal in xrange(nanals):
            vrtspec[nanal],divspec[nanal],thetaspec[nanal] = \
            models[nanal].rk4step(vrtspec[nanal],divspec[nanal],thetaspec[nanal])
    vrtspec_fcst[nsteps] = vrtspec[:]
    divspec_fcst[nsteps] = divspec[:]
    thetaspec_fcst[nsteps] = thetaspec[:]

    t2 = time.clock()
    if profile:print 'cpu time for ens forecast',t2-t1

# dump out ensemble for restart
cPickle.dump(vrtspec_fcst,open('vrtspec_fcst.pkl',mode='wb'),protocol=2)
cPickle.dump(divspec_fcst,open('divspec_fcst.pkl',mode='wb'),protocol=2)
cPickle.dump(thetaspec_fcst,open('thetaspec_fcst.pkl',mode='wb'),protocol=2)

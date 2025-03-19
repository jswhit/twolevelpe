import numpy as np
from scipy.linalg import lapack
# function definitions.

def addcyclic_1d(data,cyclic_width=360.):
    # add longitudinal wrap-around point to 1-d array
    nlons, = data.shape
    data_out = np.empty((nlons+1,),data.dtype)
    data_out[0:-1] = data
    data_out[-1] = data[0]+cyclic_width # default assumes this is a longitude in degrees
    return data_out

def addcyclic_2d(data):
    # add longitudinal wrap-around point to 2-d array
    nlats,nlons = data.shape
    data_out = np.empty((nlats,nlons+1),data.dtype)
    data_out[:,0:-1] = data
    data_out[:,-1] = data[:,0]
    return data_out

def fibonacci_pts(npts,latmin,latmax):
    # return lats and lons of N=npts fibonacci grid on a sphere.
    pi = np.pi
    inc = pi * (3.0 - np.sqrt(5.0))
    off = 2. / npts
    lats = []; lons = []
    for k in range(npts):
       y = k*off - 1. + 0.5*off
       r = np.sqrt(1 - y**2)
       phi = k * inc
       x = np.cos(phi)*r
       z = np.sin(phi)*r
       theta = np.arctan2(np.sqrt(x**2+y**2),z)
       phi = np.arctan2(y,x)
       lat = np.degrees(0.5*pi-theta)
       if lat > latmin and lat < latmax:
           lats.append( lat )
           if phi < 0.: phi = 2.*pi+phi
           lons.append( np.degrees(phi) )
    return np.array(lats), np.array(lons)

def bilintrp(datain,xin,yin,xout,yout,checkbounds=True,cyclic_width=360.):
    """
    Interpolate data (datain) on a rectilinear grid (with x = xin
    y = yin) to a grid with x = xout, y = yout.

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    datain           a rank-2 array with 1st dimension corresponding to
                     y, 2nd dimension x.
    xin, yin         rank-1 arrays containing x and y of
                     datain grid in increasing order.
    xout, yout       rank-1 arrays containing x and y of desired output grid.
    checkbounds      if True, check to see that interpolation points
                     are inside input grid.
    ==============   ====================================================
    """
    # add wrap-around point to xin, datain
    datain = addcyclic_2d(datain)
    xin = addcyclic_1d(xin,cyclic_width=cyclic_width)
    # xin and yin must be monotonically increasing.
    if xin[-1]-xin[0] < 0 or yin[-1]-yin[0] < 0:
        raise ValueError('xin and yin must be increasing!')
    if xout.shape != yout.shape:
        raise ValueError('xout and yout must have same shape!')
    # check that xout,yout are
    # within region defined by xin,yin.
    if checkbounds:
        if xout.min() < xin.min() or \
           xout.max() > xin.max() or \
           yout.min() < yin.min() or \
           yout.max() > yin.max():
            raise ValueError('yout or xout outside range of yin or xin')
    # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]
    if max(delx)-min(delx) < 1.e-7 and max(dely)-min(dely) < 1.e-7:
        # regular input grid.
        xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
        ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])
    else:
        # irregular (but still rectilinear) input grid.
        xoutflat = xout.flatten(); youtflat = yout.flatten()
        ix = (np.searchsorted(xin,xoutflat)-1).tolist()
        iy = (np.searchsorted(yin,youtflat)-1).tolist()
        xoutflat = xoutflat.tolist(); xin = xin.tolist()
        youtflat = youtflat.tolist(); yin = yin.tolist()
        xcoords = []; ycoords = []
        for n,i in enumerate(ix):
            if i < 0:
                xcoords.append(-1) # outside of range on xin (lower end)
            elif i >= len(xin)-1:
                xcoords.append(len(xin)) # outside range on upper end.
            else:
                xcoords.append(float(i)+(xoutflat[n]-xin[i])/(xin[i+1]-xin[i]))
        for m,j in enumerate(iy):
            if j < 0:
                ycoords.append(-1) # outside of range of yin (on lower end)
            elif j >= len(yin)-1:
                ycoords.append(len(yin)) # outside range on upper end
            else:
                ycoords.append(float(j)+(youtflat[m]-yin[j])/(yin[j+1]-yin[j]))
        xcoords = np.reshape(xcoords,xout.shape)
        ycoords = np.reshape(ycoords,yout.shape)
    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)
    # interpolate to output grid using bilinear interpolation.
    xi = xcoords.astype(np.int32)
    yi = ycoords.astype(np.int32)
    xip1 = xi+1
    yip1 = yi+1
    xip1 = np.clip(xip1,0,len(xin)-1)
    yip1 = np.clip(yip1,0,len(yin)-1)
    delx = xcoords-xi.astype(np.float32)
    dely = ycoords-yi.astype(np.float32)
    dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
              delx*dely*datain[yip1,xip1] + \
              (1.-delx)*dely*datain[yip1,xi] + \
              delx*(1.-dely)*datain[yi,xip1]
    return dataout

def gcdist(lon1,lat1,lon2,lat2):
    # compute great circle distance in radians between (lon1,lat1) and
    # (lon2,lat2).
    # lon,lat pairs given in radians - returned distance is in radians.
    # uses Haversine formula
    # (see http://www.census.gov/cgi-bin/geo/gisfaq?Q5.1)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    # this can happen due to roundoff error, resulting in dist = NaN.
    a = a.clip(0.,1.)
    return 2.0 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

def gaspcohn(r,gaussian=False):
    # Gaspari-Cohn taper function.
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be normalized so taper = 0 at r = 1
    # r should be positive.
    if gaussian: # return equivalent gaussian (no compact support)
        taper = np.exp(-(r**2/0.15))
    else:
        rr = 2.*r
        rr = rr.clip(min=np.finfo(rr.dtype).eps)
        taper = np.where(r<=0.5, \
                ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
                np.zeros(r.shape,r.dtype))
        taper = np.where(np.logical_and(r>0.5,r<1.), \
                ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                   + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper

def serial_ensrf(xens,hxens,obs,oberrs,covlocal,hcovlocal):
    """serial potter method"""
    nanals, ndim = xens.shape; nobs = len(obs)
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
    xmean_b = xmean.copy()
    for nob,ob,oberr in zip(np.arange(nobs),obs,oberrs):
        hpbht = (hxprime[:,nob]**2).sum()/(nanals-1)
        # state space update
        pbht = np.sum(np.transpose(xprime)*hxprime[:,nob],1)/float(nanals-1)
        kfgain = (covlocal[nob,:]*pbht/(hpbht+oberr)).reshape((1,ndim))
        gainfact = ((hpbht+oberr)/hpbht*\
                   (1.-np.sqrt(oberr/(hpbht+oberr))))
        xmean = xmean + kfgain*(ob-hxmean[nob])
        hxprime_tmp = hxprime[:,nob].reshape((nanals, 1))
        xprime = xprime - gainfact*kfgain*hxprime_tmp
        # observation space update (only update obs not yet assimilated)
        pbht = np.sum(np.transpose(hxprime[:,nob:])*hxprime[:,nob],1)/float(nanals-1)
        kfgain = (hcovlocal[nob,nob:]*pbht/(hpbht+oberr)).reshape((1,nobs-nob))
        hxmean[nob:] = hxmean[nob:] + kfgain*(ob-hxmean[nob])
        hxprime[:,nob:] = hxprime[:,nob:] - gainfact*kfgain*hxprime_tmp
    xens = xmean + xprime
    return xens

def letkf_calcwts(hxens,ominusf,oberrs,covlocal_ob=None):
    """calculate analysis weights with local ensemble transform kalman filter
    (assuming no vertical localization, using observation error localization
    in the horizontal)"""
    nanals, nobs = hxens.shape
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
    def calcwts(hxprime,Rinv,ominusf):
        YbRinv = hxprime*Rinv
        YbSqrtRinv = hxprime*np.sqrt(Rinv)
        pa = (nanals-1)*np.eye(nanals) + np.dot(YbSqrtRinv, YbSqrtRinv.T)
        #evals, eigs = np.linalg.eigh(pa)
        evals, eigs, info = lapack.dsyevd(pa)
        pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1./evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(pasqrtinv, pasqrtinv.T), YbRinv), ominusf)
        wts = np.sqrt(nanals-1)*pasqrtinv + tmp[:,np.newaxis]
        return wts
    if covlocal_ob is not None: # LETKF (horizontal localization)
        ndim1 = covlocal_ob.shape[1]
        wts = np.empty((ndim1,nanals,nanals),np.float32)
        for n in range(ndim1):
            mask = covlocal_ob[:,n] > 1.e-7
            Rinv = covlocal_ob[mask,n]/oberrs[mask]
            wts[n] = calcwts(hxprime[:,mask],Rinv,ominusf[mask])
    else: # ETKF (no localization)
        Rinv = np.diag(1./oberrs)
        wts = calcwts(hxprime,Rinv,ominusf)
    return wts

def letkf_calcwts_corr(hxens,xens,ominusf,oberrs,corr_power=0,covlocal_ob=None):
    """calculate analysis weights with local ensemble transform kalman filter
    (assuming no vertical localization, using observation error localization
    in the horizontal)"""
    nanals, nobs = hxens.shape
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
    xmean = xens.mean(axis=0); xprime = xens-xmean
    normfact = 1./(nanals-1)
    def calcwts(hxprime,Rinv,ominusf):
        YbRinv = hxprime*Rinv
        YbSqrtRinv = hxprime*np.sqrt(Rinv)
        pa = (nanals-1)*np.eye(nanals) + np.dot(YbSqrtRinv, YbSqrtRinv.T)
        evals, eigs, info = lapack.dsyevd(pa)
        pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1./evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(pasqrtinv, pasqrtinv.T), YbRinv), ominusf)
        wts = np.sqrt(nanals-1)*pasqrtinv + tmp[:,np.newaxis]
        return wts
    ndim1 = covlocal_ob.shape[1]; nvars = 5
    wts = np.empty((nvars,ndim1,nanals,nanals),np.float32)
    for n in range(ndim1):
        for nv in range(nvars):
            mask = covlocal_ob[:,n] > 1.e-7
            hxprime_local = hxprime[:,mask]
            if corr_power > 0:
                varobs = (hxprime_local**2).sum(axis=0)*normfact
                varstate =  (xprime[:,nvars*n+nv]**2).sum(axis=0)*normfact
                pbht = (xprime[:,nvars*n+nv,np.newaxis]*hxprime_local).sum(axis=0)*normfact
                corr = np.abs(pbht/np.sqrt(varobs[np.newaxis,:]*varstate)).squeeze()
                #print(nv,corr.shape, corr.min(), corr.max())
                Rinv = (corr**corr_power)*covlocal_ob[mask,n]/oberrs[mask]
            else:
                Rinv = covlocal_ob[mask,n]/oberrs[mask]
            wts[nv,n] = calcwts(hxprime_local,Rinv,ominusf[mask])
    return wts

def letkf_update(xens,wts):
    """calculate increment (analysis - forecast) to state with LETKF
    using precomputed analysis weights (assuming no vertical localization)."""
    nanals, ndim = xens.shape
    xmean = xens.mean(axis=0); xprime = xens-xmean
    if len(wts.shape) == 2: # ETKF (no localization, global weights)
        xens = xmean + np.dot(wts.T, xprime)
    else: # LETKF (wts for every horizontal grid point)
        ndim1 = wts.shape[0]; nvars = ndim//ndim1
        for n in range(ndim1):
            xens[:,nvars*n:nvars*(n+1)] = xmean[nvars*n:nvars*(n+1)] +\
            np.dot(wts[n].T, xprime[:,nvars*n:nvars*(n+1)])
    return xens

def letkf_update_corr(xens,wts):
    """calculate increment (analysis - forecast) to state with LETKF
    using precomputed analysis weights (assuming no vertical localization)."""
    nanals, ndim = xens.shape
    xmean = xens.mean(axis=0); xprime = xens-xmean
    nvars = wts.shape[0]; ndim1 = wts.shape[1]
    for n in range(ndim1):
        for nv in range(nvars):
            xens[:,nvars*n+nv] = xmean[nvars*n+nv] +\
            np.dot(wts[nv,n].T, xprime[:,nvars*n+nv])
    return xens

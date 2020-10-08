import numpy as np
from pyspharm import Spharmt

# two-level baroclinic primitive equation model with constant
# static stability described in
# Lee, S. and I. M. Held: 1993: Baroclinic Wave Packets in Models and Observations
# J. Atmos. Sci., 50, 1413-1428.
# http://dx.doi.org/10.1175/1520-0469(1993)050<1413:BWPIMA>2.0.CO;2

class TwoLevel(object):

    def __init__(self,sp,dt,ptop=0.,p0=1.e5,grav=9.80616,omega=7.292e-5,cp=1004,\
            rgas=287.,efold=3600.,ndiss=8,tdrag=4.*86400,tdiab=20.*86400.,\
            umax=40,jetexp=2,delth=20):
        # set model parameters
        self.p0 = p0 # mean surface pressure
        self.ptop = ptop # model top pressure
        self.rgas = rgas # gas constant for dry air
        self.grav = grav # gravity
        self.omega = omega # rotation rate
        self.cp = cp # specific heat of dry air at constant pressure
        self.delth = delth # static stability
        dp = 0.5*(ptop-p0)
        self.dp = dp
        exnf1 = cp*((p0+0.5*dp)/p0)**(rgas/cp)
        exnf2 = cp*((p0+1.5*dp)/p0)**(rgas/cp)
        self.delta_exnf = exnf2-exnf1 # diff in exner function between 2 levs.
        # efolding time scale for hyperdiffusion at shortest wavenumber
        self.efold = efold
        self.ndiss = ndiss # order of hyperdiffusion (2 for laplacian)
        self.sp = sp # Spharmt instance
        self.ntrunc = sp.ntrunc; self.rsphere = self.sp.rsphere
        self.dt = dt # time step (secs)
        self.tdiab = tdiab # lower layer drag timescale
        self.tdrag = tdrag # interface relaxation timescale
        # create lat/lon arrays
        delta = 2.*np.pi/sp.nlons
        lons1d = np.arange(-np.pi,np.pi,delta)
        self.lons,self.lats = np.meshgrid(lons1d,sp.lats)
        self.nlat = self.sp.nlats; self.nlon = self.sp.nlons
        # weights for computing global means.
        if self.sp.gridtype == 'gaussian':
            self.globalmeanwts =\
            np.ones((self.nlat,self.nlon))*self.sp.gauwts[:,np.newaxis]
        else:
            self.globalmeanwts =\
            np.ones((self.nlat,self.nlon))*np.cos(self.lats)
        self.globalmeanwts = self.globalmeanwts/self.globalmeanwts.sum()
        self.f = 2.*omega*np.sin(self.lats) # coriolis
        # create laplacian operator and its inverse.
        self.lap = -sp.degree*(sp.degree+1.0).astype(np.complex)
        self.ilap = np.zeros(self.lap.shape, self.lap.dtype)
        self.ilap[1:] = 1./self.lap[1:]
        self.lap = self.lap/self.rsphere**2
        self.ilap = self.ilap*self.rsphere**2
        # hyperdiffusion operator
        indxn = sp.degree.astype(np.float)
        totwavenum = indxn*(indxn+1.0)
        self.hyperdiff = -(1./efold)*(totwavenum/totwavenum[-1])**(ndiss/2)
        # set equilibrium layer thicknes profile.
        self.jetexp = jetexp
        self.umax = umax
        self._interface_profile(umax,jetexp)
        self.t = 0.

    def _interface_profile(self,umax,jetexp):
        ug = np.zeros((2,self.nlat,self.nlon),np.float32)
        vg = np.zeros((2,self.nlat,self.nlon),np.float32)
        ug[1,:,:] = umax*np.sin(2.*self.lats)**jetexp
        vrtspec, divspec = self.sp.getvrtdivspec(ug,vg)
        thetaspec = self.nlbalance(vrtspec)
        self.thetarefspec = thetaspec
        self.thetaref = self.sp.spectogrd(thetaspec)
        self.uref = ug

    def nlbalance(self,vrtspec):
        # solve nonlinear balance eqn to get potential temp given vorticity.
        vrtg = self.sp.spectogrd(vrtspec)
        divspec2 = np.zeros(vrtspec.shape, vrtspec.dtype)
        ug,vg = self.sp.getuv(vrtspec,divspec2)
        # horizontal vorticity flux
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        # compute vort flux contributions to vorticity and divergence tend.
        tmpspec, dvrtdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2)
        ddivdtspec = tmpspec[1,:]-tmpspec[0,:]
        ke = 0.5*(ug**2+vg**2)
        tmpspec = self.sp.grdtospec(ke[1,:,:]-ke[0,:,:])
        return (tmpspec - self.ilap*ddivdtspec)/self.delta_exnf

    def gettend(self,vrtspec,divspec,thetaspec):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrtg = self.sp.spectogrd(vrtspec)
        # this is baroclinic div = divupper-divlower
        # omega = dp*divg/2
        # baroptropic div is zero.
        divg = self.sp.spectogrd(divspec)
        divspec2 = np.empty(vrtspec.shape, vrtspec.dtype)
        divspec2[0,:] = -0.5*divspec
        divspec2[1,:] = 0.5*divspec
        ug,vg = self.sp.getuv(vrtspec,divspec2)
        thetag = self.sp.spectogrd(thetaspec)
        self.u = ug; self.v = vg; self.divg = divg
        self.vrt = vrtg; self.theta = thetag
        self.w = self.dp*divg
        vadvu = 0.25*(divg*(ug[1,:,:]-ug[0,:,:]))
        vadvv = 0.25*(divg*(vg[1,:,:]-vg[0,:,:]))
        # horizontal vorticity flux
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        # add lower layer drag and vertical advection contributions
        tmpg1[0,:,:] += vadvv + vg[0,:,:]/self.tdrag
        tmpg2[0,:,:] += -vadvu - ug[0,:,:]/self.tdrag
        tmpg1[1,:,:] += vadvv
        tmpg2[1,:,:] += -vadvu
        # compute vort flux contributions to vorticity and divergence tend.
        tmpspec, dvrtdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2)
        ddivdtspec = tmpspec[1,:]-tmpspec[0,:]
        dvrtdtspec *= -1
        # vorticity hyperdiffusion
        dvrtdtspec += self.hyperdiff[np.newaxis,:]*vrtspec
        # add laplacian term and hyperdiffusion to div tend.
        ke = 0.5*(ug**2+vg**2)
        tmpspec = self.sp.grdtospec(ke[1,:,:]-ke[0,:,:])
        ddivdtspec += self.hyperdiff*divspec - \
                      self.lap*(tmpspec - self.delta_exnf*thetaspec)
        # tendency of pot. temp.
        umean = 0.5*(ug[1,:,:]+ug[0,:,:])
        vmean = 0.5*(vg[1,:,:]+vg[0,:,:])
        # temp eqn - flux term
        tmpg1 = -umean*thetag; tmpg2 = -vmean*thetag
        tmpspec1, dthetadtspec = self.sp.getvrtdivspec(tmpg1,tmpg2)
        # hyperdiffusion, vertical advection, thermal relaxation.
        dthetadtspec += self.hyperdiff*thetaspec +\
        (self.thetarefspec-thetaspec)/self.tdiab - 0.5*self.delth*divspec
        return dvrtdtspec,ddivdtspec,dthetadtspec

    def rk4step(self,vrtspec,divspecin,thetaspec):
        if divspecin.ndim == 2:
            divspec = divspecin[1]-divspecin[0]
        else:
            divspec = divspecin
        # update state using 4th order runge-kutta
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,thetaspec)
        k2vrt,k2div,k2thk = \
        self.gettend(vrtspec+0.5*dt*k1vrt,divspec+0.5*dt*k1div,thetaspec+0.5*dt*k1thk)
        k3vrt,k3div,k3thk = \
        self.gettend(vrtspec+0.5*dt*k2vrt,divspec+0.5*dt*k2div,thetaspec+0.5*dt*k2thk)
        k4vrt,k4div,k4thk = \
        self.gettend(vrtspec+dt*k3vrt,divspec+dt*k3div,thetaspec+dt*k3thk)
        vrtspec += dt*(k1vrt+2.*k2vrt+2.*k3vrt+k4vrt)/6.
        divspec += dt*(k1div+2.*k2div+2.*k3div+k4div)/6.
        thetaspec += dt*(k1thk+2.*k2thk+2.*k3thk+k4thk)/6.
        self.t += dt
        if divspecin.ndim == 2:
            divspecin[0,:] = -0.5*divspec
            divspecin[1,:] = 0.5*divspec
            return vrtspec,divspecin,thetaspec
        else:
            return vrtspec,divspec,thetaspec

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # grid, time step info
    nlons = 128 # number of longitudes
    ntrunc = 42  # spectral truncation (for alias-free computations)
    dt = 2700 # time step in seconds
    nlats = nlons//2  # for regular grid.
    gridtype = 'gaussian'

    # create spherical harmonic instance.
    rsphere = 6.37122e6 # earth radius
    sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype=gridtype)

    # create model instance using default parameters.
    model = TwoLevel(sp,dt)

    # vort, div initial conditions
    psipert = np.zeros((2,model.nlat,model.nlon),np.float)
    psipert[1,:,:] = 5.e6*np.sin((model.lons-np.pi))**12*np.sin(2.*model.lats)**12
    psipert = np.where(model.lons[np.newaxis,:,:] > 0., 0, psipert)
    psipert[1,:,:] += np.random.normal(scale=1.e6,size=(sp.nlats,sp.nlons))
    ug = np.zeros((2,model.nlat,model.nlon),np.float)
    vg = np.zeros((2,model.nlat,model.nlon),np.float)
    ug[1,:,:] = model.umax*np.sin(2.*model.lats)**model.jetexp
    vrtspec, divspec = sp.getvrtdivspec(ug,vg)
    vrtspec = vrtspec + model.lap*sp.grdtospec(psipert)
    thetaspec = model.nlbalance(vrtspec)
    divspec = np.zeros(thetaspec.shape, thetaspec.dtype)

    # animate potential temperature

    fig = plt.figure(figsize=(16,8))
    vrtspec, divspec, thetaspec = model.rk4step(vrtspec, divspec, thetaspec)
    thetamean = (model.theta*model.globalmeanwts).sum()
    varplot = 'theta'
    #varplot = 'w'
    #varplot = 'u'
    if varplot == 'theta':
        data = model.theta - thetamean
        vmax = 50; vmin = -vmax
        cmap = plt.cm.RdBu_r
    elif varplot == 'u':
        data = model.u[1]
        vmax = 120; vmin = -vmax
        cmap = plt.cm.RdBu_r
    else:
        data = model.w
        vmax = 10; vmin = -vmax
        cmap = plt.cm.RdBu_r
    vmin = -vmax
    ax = fig.add_subplot(111); ax.axis('off')
    plt.tight_layout()
    im=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest")
    txt=ax.text(0.5,0.95,'%s day %10.2f' % \
        (varplot,float(model.t/86400.)),ha='center',color='k',fontsize=18,transform=ax.transAxes)

    model.t = 0 # reset clock
    nout = int(3.*3600./model.dt) # plot interval
    def updatefig(*args):
        global vrtspec, divspec, thetaspec
        for n in range(nout):
            vrtspec, divspec, thetaspec = model.rk4step(vrtspec, divspec, thetaspec)
        if varplot == 'theta':
            thetamean = (model.theta*model.globalmeanwts).sum()
            im.set_data(model.theta - thetamean)
        elif varplot == 'u':
            im.set_data(model.u[1])
        else:
            im.set_data(model.w)
        txt.set_text('%s day %10.2f' % \
                     (varplot,float(model.t/86400.)))
        return im,txt,

    ani = animation.FuncAnimation(fig,updatefig,interval=0,blit=False)
    plt.show()

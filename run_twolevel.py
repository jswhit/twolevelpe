from twolevel import TwoLevel
from pyspharm import Spharmt
import numpy as np
from netCDF4 import Dataset

# grid, time step info
nlons = 192; nlats = nlons//2  # number of longitudes/latitudes
ntrunc = 64
dt = 1800. # time step in seconds

#nlons = 128; nlats = nlons//2  # number of longitudes/latitudes
#ntrunc = 42
#dt = 2700. # time step in seconds

#nlons = 96; nlats = nlons//2  # number of longitudes/latitudes
#ntrunc = 32
#dt = 3600. # time step in seconds

fhout = 12. # output interval (hours)

gridtype = 'gaussian' # 'regular' or 'gaussian'
output_file = 'truth_twolevel_t%s_%sh.nc' % (ntrunc,int(fhout))

# create spherical harmonic instance.
rsphere = 6.37122e6 # earth radius
sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype=gridtype)

nstart = int((200.*86400.)/dt) # end of spinup period
nmax = int((1200.*86400.)/dt) # total duration of run

# create model instance
model = TwoLevel(sp,dt,jetexp=4,umax=50,tdrag=2.*86400,tdiab=14.*86400.)
print('pole/equator temp diff = ', model.thetaref.max()-model.thetaref.min())

# vort, div initial conditions
psipert = np.zeros((2,model.nlat,model.nlon),np.float32)
psipert[1,:,:] = 5.e6*np.sin((model.lons-np.pi))**12*np.sin(2.*model.lats)**12
psipert = np.where(model.lons[np.newaxis,:,:] > 0., 0, psipert)
psipert[1,:,:] += np.random.normal(scale=1.e6,size=(sp.nlats,sp.nlons))
ug = np.zeros((2,model.nlat,model.nlon),np.float32)
vg = np.zeros((2,model.nlat,model.nlon),np.float32)
ug[1,:,:] = model.umax*np.sin(2.*model.lats)**model.jetexp
vrtspec, divspec = sp.getvrtdivspec(ug,vg)
vrtspec = vrtspec + model.lap*sp.grdtospec(psipert)
thetaspec = model.nlbalance(vrtspec)
divspec = np.zeros(thetaspec.shape, thetaspec.dtype)

# create netcdf file.
nc = Dataset(output_file,'w',format='NETCDF4_CLASSIC')
nc.rsphere = rsphere
nc.gridtype = gridtype
nc.ntrunc = ntrunc
nc.dt = dt
atts = ['grav','omega','cp','rgas','p0','ptop','delth','efold','ndiss','tdrag','tdiab','umax','jetexp']
for att in atts:
    nc.setncattr(att,model.__dict__[att])
lat = nc.createDimension('lat',sp.nlats)
lon = nc.createDimension('lon',sp.nlons)
layer = nc.createDimension('layer',2)
t = nc.createDimension('t',None)
u = nc.createVariable('u',np.float32,('t','layer','lat','lon'),zlib=True)
u.units = 'meters per second'
v = nc.createVariable('v',np.float32,('t','layer','lat','lon'),zlib=True)
u.units = 'meters per second'
w = nc.createVariable('w',np.float32,('t','lat','lon'),zlib=True)
w.units = 'Pa per second'
theta = nc.createVariable('theta',np.float32,('t','lat','lon'),zlib=True)
theta.units = 'K'
lats = nc.createVariable('lat',np.float32,('lat',))
lats.units = 'degrees north'
lats[:] = np.degrees(sp.lats)
lons = nc.createVariable('lon',np.float32,('lon',))
lons.units = 'degrees east'
lons[:] = np.degrees(sp.lons)
time = nc.createVariable('t',np.float32,('t',))
time.units = 'hours'

# run model, write out netcdf.
nn = 0
print("# timestep, hour, vmin, vmax")
for n in range(nmax):
    vrtspec, divspec, thetaspec = model.rk4step(vrtspec, divspec, thetaspec)
    if n >= nstart-1 and model.t % (fhout*3600.) == 0.:
        wout = model.dp*model.sp.spectogrd(divspec)
        print(n,model.t/86400.,model.v.min(), model.v.max(), wout.min(), wout.max())
        time[nn] = model.t/3600.
        u[nn] = model.u; v[nn] = model.v
        theta[nn] = model.theta
        w[nn] = wout
        nn += 1
nc.close()

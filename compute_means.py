import sys, numpy
data = numpy.genfromtxt(sys.argv[1],invalid_raise=False)
nskip = int(sys.argv[2])
if len(sys.argv) > 3:
    nmax = int(sys.argv[3])+1
else:
    nmax = data.shape[0]
data = data[nskip:nmax]
ntimes = data[-1,0]-data[0,0]
out = data[:,1:].mean(axis=0)
strout = '%s ' % (int(ntimes),)
for item in out:
    strout += '%6.4f ' % (item,)
print strout

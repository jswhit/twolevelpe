import sys, numpy
data = numpy.loadtxt(sys.argv[1])
nskip = int(sys.argv[2])
data = data[nskip:]
ntimes = data[-1,0]-data[0,0]
out = data[:,1:].mean(axis=0)
strout = '%s ' % (ntimes,)
for item in out:
    strout += '%6.4f ' % (item,)
print strout

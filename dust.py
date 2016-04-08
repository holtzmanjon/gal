import healpy
import h5py
import numpy as np

def setup():
    global pix_info, best_fit

    f = h5py.File('dust-map-3d.h5', 'r')
    pix_info = f['/pixel_info'][:]
    best_fit = f['/best_fit'][:]

def getebv(l,b) :
    phi = l * np.pi / 180.
    if b<0 :
        theta = (90+abs(b)) * np.pi / 180.
    else :
        theta = (90-b) * np.pi / 180.
    nside = 1024
    ind = np.array([])
    while len(ind) == 0 and nside>= 64:
        pix = healpy.ang2pix(nside,theta,phi,nest=True)
        ind = np.where((pix_info['healpix_index'] == pix) & (pix_info['nside'] == nside))[0]
        nside /= 2
    if len(ind) > 0 :
        return best_fit[ind[0]]
    else :
        print 'no value found'
        return None


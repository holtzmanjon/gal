import os
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import vstack
import astropy.table as table
import pdb
import astropy
import math
import matplotlib.pyplot as plt
from holtz.tools import plots

os.environ['ISOCHRONE_DIR'] = '/home/holtz/isochrones/'

def isoname(feh) :
    if feh < 0 :
        file = 'zm{:02d}'.format(int(round(-feh*10)))
    else :
        file = 'zp{:02d}'.format(int(round(feh*10)))
    return file


def basicread(infile) :
    """
    Routine to read a Padova isochrone file using low-level I/O and return
    a numpy structured array with the contents

    Args: 
        file name : input data file name

    Returns: 
        structured array with isochrone data
    """

    # open the file
    file = open(infile)

    # initialize the lists to hold the file contents
    z = [] ; age = [] ; mini = []; mact = []; logl = []; logte = []; logg = []
    mbol = []; u = []; b = []; v = []; r = []; i = []; j = []; h = []; k = [] 
    intimf = []; stage = []

    # loop through lines in file
    nlines = 0
    for line in file :
       # ignoring comment lines starting with #, add input data into list vars
       if line.startswith('#') is False :
           cols=line.split()
           z.append(cols[0])
           age.append(cols[1])
           mini.append(cols[2])
           mact.append(cols[3])
           logl.append(cols[4])
           logte.append(cols[5])
           logg.append(cols[6])
           mbol.append(cols[7])
           u.append(cols[8])
           b.append(cols[9])
           v.append(cols[10])
           r.append(cols[11])
           i.append(cols[12])
           j.append(cols[13])
           h.append(cols[14])
           k.append(cols[15])
           intimf.append(cols[16])
           stage.append(cols[17])
           nlines+=1

    # declare the output structured array
    data = np.recarray(nlines,dtype=[
                       ('z','f4'),('age','f4'),
                       ('mini','f4'),('mact','f4'),
                       ('logl','f4'),('logte','f4'),('logg','f4'),
                       ('mbol','f4'),('u','f4'),('b','f4'),
                       ('v','f4'),('r','f4'),('i','f4'),
                       ('j','f4'),('h','f4'),('k','f4'),
                       ('intimf','f4'),('stage','i4')
                       ])

    # fill the contents and return
    data['z'] = z
    data['age'] = age
    data['mini'] = mini
    data['mact'] = mact
    data['logl'] = logl
    data['logte'] = logte
    data['logg'] = logg
    data['mbol'] = mbol
    data['u'] = u
    data['b'] = b
    data['v'] = v
    data['r'] = r
    data['i'] = i
    data['j'] = j
    data['h'] = h
    data['k'] = k
    data['intimf'] = intimf
    data['stage'] = stage
    return data

def basicread2(infile) :
    """
    Routine to read a Padova isochrone file using low-level I/O and return
    a numpy structured array with the contents

    Args: 
        file name : input data file name

    Returns: 
        structured array with isochrone data
    """

    # open the file
    file = open(infile)

    # setup up the names and data types of the columns, and initialize 
    #   list of lists
    ncols= 18
    names = [('z','f4'),('age','f4'),
            ('mini','f4'),('mact','f4'),
            ('logl','f4'),('logte','f4'),('logg','f4'),
            ('mbol','f4'),('u','f4'),('b','f4'),
            ('v','f4'),('r','f4'),('i','f4'),
            ('j','f4'),('h','f4'),('k','f4'),
            ('intimf','f4'),('stage','i4')]
    listdata=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    # loop through lines, ignoring comments, filling listdata
    nlines = 0
    for line in file :
       # ignoring comment lines starting with #, add input data into list vars
       if line.startswith('#') is False :
          cols=line.split()
          for i in range(ncols) :
              listdata[i].append(cols[i])
          nlines+=1

    # define the numpy structrued array and fill the columns
    data = np.recarray(nlines,dtype=names)
    for i in range(ncols) :
       #pdb.set_trace()
       data[names[i][0]] = listdata[i]

    return data


def basicread3(infile) :
    """
    Routine to read a Padova isochrone file using low-level I/O and return
    a numpy structured array with the contents

    Args: 
        file name : input data file name

    Returns: 
        structured array with isochrone data
    """

    #use astropy.io.ascii.read!
    data=ascii.read(infile,names=['z','age','mini','mact','logl','logte','logg','mbol','u','b','v','r','i','j','h','k','intimf','stage'])

    return data


def read(infile,columns=None,agerange=[0,20.]) :
    """
    Routine to read a Padova isochrone file using low-level I/O and return
    a numpy structured array with the contents

    Args: 
        file name : input data file name
        columns=[list] : list of columns to extract
        age = age : single age to extract

    Returns: 
        structured array with isochrone data
    """

    if os.getenv('ISOCHRONE_DIR') != "" :
        data=ascii.read(os.getenv('ISOCHRONE_DIR')+'/'+infile,
             names=['z','age','mini','mact','logl','logte','logg',
                    'mbol','u','b','v','r','i','j','h','k','intimf','stage'])
    else :
        data=ascii.read(infile,
             names=['z','age','mini','mact','logl','logte','logg',
                    'mbol','u','b','v','r','i','j','h','k','intimf','stage'])

    # add some derived columns
    data.add_column(table.column.Column(name='feh',data=np.log10(data['z']/0.0152)))
    data.add_column(table.column.Column(name='teff',data=10.**(data['logte'])))
    data.add_column(table.column.Column(name='jk',data=data['j']-data['k']))

    # select ages within specified age range
    gd = np.where((data['age'] >=agerange[0]) & (data['age'] <= agerange[1]))
    data=data[gd]

    # default columns
    if columns is None:
        # can set default columns here, or keep all quantitites
        #data.keep_columns(['Z','age','logte','logl','intimf','stage'])
        pass
    # option to extract specified columns
    else:
        data.keep_columns(columns)

    return data

def radius(logl,logte) :
    """ 
    Get stellar radius given luminosity, effective temperature
    """
    teff=10.**logte * astropy.units.K
    lum = 10.**logl * astropy.units.Lsun * astropy.constants.L_sun.cgs
    return np.sqrt(lum / (4.*math.pi*astropy.constants.sigma_sb.cgs*teff**4.))

def mkhess(agerange=[0.,20.], files=['zp00.dat'], vals=['logte','logl'],xmin=[3.4,-6.],xmax=[4.5,5.],nbins=[200,200],norm=False,verbose=0,isoadj=False) :
    """
    Routine to make a Hess diagram of a particular age from a file
    """
    # read the isochrone
    first = True
    for file in files :
        print file
        tmp = read(file,agerange=agerange)
        if first :
            iso = tmp
        else :
            iso = vstack([iso,tmp])
        first = False

    # if isochrones are logarithmically spaced, convert for constant SFR
    iso['intimf'] *= 10**(iso['age']-9)
    if isoadj :
        iso['teff'] -= iso['feh']*150.

    # initialize Hess diagram and set bin sizes given limits and nbins
    hess = np.zeros(nbins[::-1],dtype=np.float32)
    ndim=len(nbins)
    dx = (np.array(xmax)-np.array(xmin))/((np.array(nbins)-1).astype('float'))

    # need to handle each age separately
    isoages = set(iso['age'])
    tot = 0
    for age in isoages :
        print 'load age: ', age
        gd = np.where(iso['age'] == age)[0]
        aiso = iso[gd]
        tot += aiso['intimf'][-1]

        # calculate bin locations
        npts = len(gd)
        bins = np.zeros([ndim,npts])
        for idim in range(ndim) :
            bins[idim,:] = ((aiso[vals[idim]]-xmin[idim])/dx[idim]).round()

        # loop over each pair of isochrone points
        for i in range(npts-1) :
          # number of stars in between these points
          nimf = aiso['intimf'][i+1]-aiso['intimf'][i]
          if nimf > 0 :
    
            # get min and max bin numbers, in all dimensions
            bmin = np.min(bins[:,i:i+2],axis=1)
            bmax = np.max(bins[:,i:i+2],axis=1)

            # number of bins over which stars are spread
            nbin = np.prod(bmax-bmin+1)
            if verbose > 0 :
                print 'line: ', i, ' nbins: ', nbin, ' nimf: ', nimf, verbose
                for idim in range(ndim) :
                    #print aiso[vals[idim]][i], bins[idim,i], xmin[idim], dx[idim],
                    print aiso[vals[idim]][i], bins[idim,i],
                print

            if nbin > 1 :
                # interpolate between points if we are covering more than one bin
                from scipy import interpolate
                data_interp = np.zeros([ndim,nbin])
                # interpolate with mass as independent variable
                xdata = aiso['mini'][i:i+2]
                #pdb.set_trace()
                x=np.linspace(xdata[0],xdata[1],num=nbin)
                for idim in range(0,ndim) :
                    data = aiso[vals[idim]][i:i+2]
                    intfunc = interpolate.interp1d(xdata,data,kind='linear')
                    data_interp[idim,:]=intfunc(x)  #returns interpolated value(s) at x
                ndata = aiso['intimf'][i:i+2]
                intfunc = interpolate.interp1d(xdata,ndata,kind='linear')
                nb=intfunc(x) 

                # get bin limits for interpolated points
                b = np.zeros([ndim,nbin])
                for idim in range(ndim) :
                    b[idim,:] = ((data_interp[idim,:]-xmin[idim])/dx[idim]).round()
            else :
                # otherwise just copy the isochrone points
                b=bins[:,i:i+2]
                nb=aiso['intimf'][i:i+2]

            #if verbose>0 :
                #print 'sum: ', nb[len(nb)-1]-nb[0]
                #print nb

            # loop over all the pairs of sub-bins
            #print 'nsubbins: ', len(nb)
            ntot = 0.
            for j in range(len(nb)-1):
                # get min and max bin numbers
                bmin = np.min(b[:,j:j+2],axis=1)
                bmax = np.max(b[:,j:j+2],axis=1)

                # number of bins over which stars are spread
                nbin = np.prod(bmax-bmin+1)
                nimf = nb[j+1]-nb[j]

                #print 'subbin: ', j, ' nbins: ', nbin, ' nimf: ', nimf
                #print bmin
                #print bmax
                #pdb.set_trace()

                # make sure that data doesn't fall entirely outside output array in any dimension
                if (bmin>=0).all() and (bmax<=(np.array(nbins)-1)).all() :
                    # don't let array go outside of bounds
                    bmin=np.max([bmin,np.zeros(ndim)+0],axis=0)
                    bmax=np.min([bmax,np.array(nbins)-1],axis=0)
                    #print bmin
                    #print bmax
                    # add the stars in!
                    if ndim == 1 :
                        hess[bmin[0]:bmax[0]+1] += nimf / nbin
                    elif ndim == 2 :
                        hess[bmin[1]:bmax[1]+1,bmin[0]:bmax[0]+1] += nimf / nbin
                    elif ndim == 3 :
                        hess[bmin[2]:bmax[2]+1,bmin[1]:bmax[1]+1,bmin[0]:bmax[0]+1] += nimf / nbin
                    elif ndim == 4 :
                        hess[bmin[3]:bmax[3]+1,bmin[2]:bmax[2]+1,bmin[1]:bmax[1]+1,bmin[0]:bmax[0]+1] += nimf / nbin
                    elif ndim == 5 :
                        hess[bmin[4]:bmax[4]+1,bmin[3]:bmax[3]+1,bmin[2]:bmax[2]+1,bmin[1]:bmax[1]+1,bmin[0]:bmax[0]+1] += nimf / nbin
                    ntot += nimf

            #if verbose > 0 :
            #    print 'added: ', ntot
    if norm :
       hess /= tot
 
    # include axes information in a FITS HDU
    hdu=fits.ImageHDU(hess)
    hdu.header['CRVAL1'] = xmin[0]
    hdu.header['CDELT1'] = dx[0]
    hdu.header['CTYPE1'] = vals[0]
    if ndim > 1:
        hdu.header['CRVAL2'] = xmin[1]
        hdu.header['CDELT2'] = dx[1]
        hdu.header['CTYPE2'] = vals[1]
    if ndim > 2:
        hdu.header['CRVAL3'] = xmin[2]
        hdu.header['CDELT3'] = dx[2]
        hdu.header['CTYPE3'] = vals[2]
    if ndim > 3:
        hdu.header['CRVAL4'] = xmin[3]
        hdu.header['CDELT4'] = dx[3]
        hdu.header['CTYPE4'] = vals[3]
    if ndim > 4:
        hdu.header['CRVAL5'] = xmin[4]
        hdu.header['CDELT5'] = dx[4]
        hdu.header['CTYPE5'] = vals[4]

    return hdu

def plot(ax,iso,x,y,xr=None,yr=None,color=None,dx=0.,dy=0.) :
    ''' plotting routine that handles tip of RGB '''
    mdiff = iso['mini'][0:-1]-iso['mini'][1:]
    j=np.where(abs(mdiff) < 1.e-8)[0]
    if len(j) > 0 :
        if len(j) > 1 :
            j=j[0]
        line = plots.plotl(ax,iso[x][0:j]+dx,iso[y][0:j]+dy,xr=xr,yr=yr,color=color)
        plots.plotl(ax,iso[x][j+1:]+dx,iso[y][j+1:]+dy,color=line[0].get_color())
    else :
        line = plots.plotl(ax,iso[x]+dx,iso[y]+dy,xr=xr,yr=yr,color=color)
    plt.draw()

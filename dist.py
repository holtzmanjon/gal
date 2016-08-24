import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb
import os
import sys
import subprocess
import copy
from holtz.apogee import apload
from holtz.apogee import apselect
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from holtz.gal import isochrones
from holtz.gal import galmodel
from holtz.gal import dust 
from holtz.tools import plots
from holtz.tools import struct
from holtz.tools import match

def getgrid(doage=None,clobber=False,isoadj=False,filt=['h'],prefix=None):
    """ 
    Initialize 4D (Teff, logg, [Fe/H], M_H) or 5D (with doage=True) isochrone grid 
  
    Args:

    Keyword args:

    Returns: nothing, but sets global variables: teff, logg, feh, mh, grid, dteff, dlogg, dfeh, age, dage
    """

    global teff, logg, feh, mh, grid, dteff, dlogg, dfeh, age, dage

    dteff=25
    dlogg=0.05
    dfeh=0.1
    teff=np.arange(3000.,6000.,dteff)
    logg=np.arange(-1.,5.0,dlogg)
    feh=np.arange(-2.1,0.6,dfeh)
    #feh=np.arange(-0.2,0.2,dfeh)
    mh=np.arange(-7.,7.,0.1)
    if doage is None :
        vals=['teff','logg','feh']
        vmin=[teff[0],logg[0],feh[0],mh[0]]
        nbin=[teff.shape[0], logg.shape[0], feh.shape[0], mh.shape[0]]
        vmax=[teff[nbin[0]-1], logg[nbin[1]-1], feh[nbin[2]-1], mh[nbin[3]-1]]
        file = 'grid_'
    else :
        dage=0.05
        age=np.arange(8.,10.15,dage)
        vals=['teff','logg','feh','age']
        vmin=[teff[0],logg[0],feh[0],age[0],mh[0]]
        nbin=[teff.shape[0], logg.shape[0], feh.shape[0], age.shape[0], mh.shape[0]]
        vmax=[teff[nbin[0]-1], logg[nbin[1]-1], feh[nbin[2]-1], age[nbin[3]-1], mh[nbin[4]-1]]
        file = 'gridage_'

    files=(['zm21.dat','zm20.dat','zm19.dat','zm18.dat','zm17.dat','zm16.dat','zm15.dat','zm14.dat','zm13.dat','zm12.dat','zm11.dat','zm10.dat',
       'zm09.dat','zm08.dat','zm07.dat','zm06.dat','zm05.dat','zm04.dat','zm03.dat','zm02.dat','zm01.dat',
       'zp00.dat','zp01.dat','zp02.dat','zp03.dat','zp04.dat','zp05.dat','zp06.dat'])
    agerange=[8.0,10.15]

    #files=['zp00.dat']
    #agerange=[9.39,9.41]
    #file='gridtest.fits'

    # output file name
    if prefix is not None:
        file = prefix+file
    for f in filt :
        file=file+f 

    if os.path.exists(file+'.fits') and not clobber:
        grid = fits.open(file+'.fits')
    else :
        hdu=fits.PrimaryHDU() 
        hdulist=[hdu]
        for f in filt :
            temp=copy.copy(vals)
            temp.append(f)
            grid=isochrones.mkhess(agerange=agerange,files=files,vals=temp,xmin=vmin,xmax=vmax,nbins=nbin,isoadj=isoadj)
            #hdulist.append(fits.ImageHDU(grid))
            hdulist.append(grid)
       
        grid.writeto(file+'.fits',clobber=True)
   

def getdist(obste,obslogg,obsfeh,obsmag,ext=[0.],glon=None,glat=None,errte=50,errlogg=0.1,errfeh=0.1,mlim=12.2,obsage=None,errage=10.,plot=None,disp=None,verbose=False):
    """ 
    Return distance estimates for input observed parameters, errors, limiting mag for single object
  
    Args:
       obste (float) : effective temperature
       obslogg (float) : surface gravity
       obsfeh (float) : metallicity
       obsmag (list[float]) : observed magnitudes

    Keyword args:
       errte (float) :  uncertainty for Teff (default=50)
       errlogg (float) :  uncertainty for logg (default=0.1)
       errfeh (float) :  uncertainty for metallicity (default=0.1)
       mlim (float]) : limiting magnitude (for distance prior)
       ext (list[float]) : extinction(s)   (for distance prior)
       glon : Galactic longitude   (for distance prior)
       glat : Galactic latitude   (for distance prior)

    Returns: structured array with distance output
    """

    if len(obsmag) != len(ext) or len(grid)-1 != len(obsmag) :
        print('obsmag, ext, and grid lengths must match!')
        pdb.set_trace()

    # get probability of observed Teff, and limits over which to marginalize
    probte=np.exp(-0.5*(obste-teff)**2/errte**2)
    itmin=min(len(teff)-1,max(0,np.floor((obste-3*errte-teff[0])/dteff).astype('int')))
    itmax=max(0,min(len(teff)-1,np.ceil((obste+3*errte-teff[0])/dteff).astype('int')))

    # get probability of observed logg, and limits over which to marginalize
    problogg=np.exp(-0.5*(obslogg-logg)**2/errlogg**2)
    iloggmin=min(len(logg)-1,max(0,np.floor((obslogg-3*errlogg-logg[0])/dlogg).astype('int')))
    iloggmax=min(len(logg)-1,np.ceil((obslogg+3*errlogg-logg[0])/dlogg).astype('int'))

    # get probability of observed [Fe/H], and limits over which to marginalize
    probfeh=np.exp(-0.5*(obsfeh-feh)**2/errfeh**2)
    ifehmin=min(len(feh)-1,max(0,np.floor((obsfeh-3*errfeh-feh[0])/dfeh).astype('int')))
    ifehmax=max(0,min(len(feh)-1,np.ceil((obsfeh+3*errfeh-feh[0])/dfeh).astype('int')))

    # get probability of age, and limits over which to marginalize
    if verbose: 
        print(obste, obslogg, obsfeh)
        print(ifehmin, ifehmax, iloggmin, iloggmax,itmin, itmax)
    if obsage is not None :
        probage=np.exp(-0.5*(obsage-age)**2/errage**2)
        iagemin=np.max([0,np.floor((obsage-3*errage-age[0])/dage).astype('int')])
        iagemax=np.min([len(age)-1,np.ceil((obsage+3*errage-age[0])/dage).astype('int')])

    # get the probability grid 
    #obs=grid.data*0.
    if obsage is None :
        try:
          obs=np.zeros([len(grid)-1,len(mh),ifehmax-ifehmin+1,iloggmax-iloggmin+1,itmax-itmin+1])
        except:
          pdb.set_trace()
    else :
        obs=np.zeros([len(grid)-1,len(mh),iagemax-iagemin+1,ifehmax-ifehmin+1,iloggmax-iloggmin+1,itmax-itmin+1])

    for it in range(itmin,itmax) :
        #print('t:', it, itmin,itmax)
        for ig in range(iloggmin,iloggmax) :
            #print('g:', ig, iloggmin, iloggmax)
            p1 = probte[it]*problogg[ig]
            for ifeh in range(ifehmin,ifehmax) :
                #print('fe: ',ifeh, ifehmin, ifehmax)
                prob = p1*probfeh[ifeh]
                if obsage is None :
                    for imh in range(len(mh)-1) :
                        for igrid in range(len(grid)-1) :
                            obs[igrid,imh,ifeh-ifehmin,ig-iloggmin,it-itmin] = grid[igrid+1].data[imh,ifeh,ig,it]*prob
                else :
                    for iage in range(iagemin,iagemax) :
                        p2 = prob*probage[iage]
                        #for imh in range(len(mh)-1) :
                        for igrid in range(len(grid)-1) :
                            #print(iage, iagemin, iagemax, obsage, errage, ifeh,ig,it)
                            #print(obs.shape, grid[igrid+1].data.shape)
                            obs[igrid,:,iage-iagemin,ifeh-ifehmin,ig-iloggmin,it-itmin] = grid[igrid+1].data[:,iage,ifeh,ig,it]*p2
                        #for imh in range(len(mh)-1) :
                        #    obs[imh,iage,ifeh,ig,it] = grid.data[imh,iage,ifeh,ig,it]*p2

    #marginalize and normalize
    diso=[]
    diso_dist=[]
    diso_pdf=[]
    diso_pdfmax=[]
    for igrid in range(len(obsmag)) :
        if obsage is None :
            probmh=obs[igrid,:,:,:,:].sum(axis=(1,2,3))
        else :
            probmh=obs[igrid,:,:,:,:,:].sum(axis=(1,2,3,4))
        tot=probmh.sum()

        # get distances corresponding to M_H
        distmod = obsmag[igrid]-mh-ext[igrid]
        dist=10.**((distmod+5)/5.)

        # get max, median, mean distance from M_H PDF, no prior
        diso_dist.append(dist)
        diso_pdf.append(copy.deepcopy(probmh))
        if tot > 0 :
            probmh/=tot
            diso.append([dist[probmh.argmax()],median_index(dist,probmh)[1],(probmh*dist).sum()/probmh.sum()])
            diso_pdfmax.append(probmh[probmh.argmax()]*tot)
        else :
            diso.append([np.nan,np.nan,np.nan])
            diso_pdfmax.append(0.)

        if plot is not None :
            print('M_H: ', mh[probmh.argmax()],median_index(mh,probmh)[1],(probmh*mh).sum()/probmh.sum())
            print('diso: ', diso[igrid])
            plot.plot(mh,probmh*tot)
            plt.draw()

    # OK, now compute spatial prior if we have coordinates and 3D extinction
    diso_gal = np.array([-1.,-1.,-1.])
    diso_gal_dist = dist*0.
    diso_gal_pdf = probmh*0.
    dext = -1.
    if glon is not None :
        p = galmodel.galmodel(dist,0.,glon,glat)
        p *= dist**2      # account for cone volume effect
        # get extinction from 3D dust map
        ebv = dust.getebv(glon,glat)  
        if ebv is not None :
            extinct = { 'distmod': np.arange(4.,19.5,0.5), 
                        'ah': ebv*.449 }  # for H band extinction from Schlafly&Finkbeiner
            # get the interpolating function 
            intfunc = interpolate.interp1d(extinct['distmod'],extinct['ah'],kind='linear',bounds_error=False)
            extmap = intfunc(distmod)
            # replace nans
            try:
                if np.nanargmax(extmap) > 0 :
                    extmap[0:np.nanargmax(extmap)] = np.nanmax(extmap)
                if np.nanargmin(extmap) < len(extmap) :
                    extmap[np.nanargmin(extmap):len(extmap)] = 0.
            except:
                extmap[:] = 0.

            # get "extinction" distance
            i=0
            try:
                hext=ext[1]
            except:
                hext=ext[0]
            while extmap[i] > hext  and i < len(extmap)-1:
                i+=1
            if i < len(extmap) :
                dext = distmod[i-1]+(distmod[i]-distmod[i-1])*(hext-extmap[i-1])/(extmap[i]-extmap[i-1])
            else :
                dext = -1.

            # get the density prior, accounting for selection function
            dprior=(select(mh,bulgelf,distmod,mlim,extmap=extmap)*p[2]+
                    select(mh,halolf,distmod,mlim,extmap=extmap)*p[3]+
                    select(mh,thinlf,distmod,mlim,extmap=extmap)*p[0]+
                    select(mh,thicklf,distmod,mlim,extmap=extmap)*p[1])
            probmh*=dprior
            tot = probmh.sum()
            diso_gal_dist = dist
            diso_gal_pdf = probmh
            if tot > 0 :
                probmh/=tot
                 # get max, median, mean distance from PDF with prior
                diso_gal = dist[probmh.argmax()],median_index(dist,probmh)[1],(probmh*dist).sum()/probmh.sum()
            else :
                diso_gal = [np.nan,np.nan,np.nan]

    # output
    out = np.recarray(1,dtype=[
                       ('diso','3f4',(len(obsmag))),
                       ('diso_dist','140f4',(len(obsmag))),
                       ('diso_pdf','140f4',(len(obsmag))),
                       ('diso_pdfmax','f4',(len(obsmag))),
                       ('diso_gal','3f4'),
                       ('diso_gal_dist','140f4'),
                       ('diso_gal_pdf','140f4'),
                       ('dext','f4'),
                       ])
    out.diso = diso
    out.diso_dist = diso_dist
    out.diso_pdf = diso_pdf
    out.diso_pdfmax = diso_pdfmax
    out.diso_gal = diso_gal
    out.diso_gal_dist = diso_gal_dist
    out.diso_gal_pdf = diso_gal_pdf
    out.dext = dext
    if disp is not None: 
        it,ig,ife,ia = index(obste,obslogg,obsfeh,obsage)
        if obsage is None:
            disp.tv(grid[1].data[:,ife,:,:].sum(axis=0)*1000.,min=0,max=10.)
            print(mh[grid[1].data[:,ife,ig,it].argmax()])
            oy,ox = np.unravel_index(obs[0,:,:,:,:].sum(axis=0).sum(axis=0).argmax(),obs[0,:,:,:,:].sum(axis=0).sum(axis=0).shape)
        else :
            disp.tv(grid[1].data[:,ia,ife,:,:].sum(axis=0)*1000.,min=0,max=10.)
            print(mh[grid[1].data[:,ia,ife,ig,it].argmax()])
            oy,ox = np.unravel_index(obs[0,:,:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0).argmax(),obs[0,:,:,:,:,:].sum(axis=0).sum(axis=0).sum(axis=0).shape)
        circ=plt.Circle((it,ig),radius=2,fill=False,color='g')
        disp.ax.add_patch(circ)
        ocirc=plt.Circle((ox+itmin,oy+iloggmin),radius=2,fill=False,color='r')
        disp.ax.add_patch(ocirc)
        pdb.set_trace()
        circ.remove()
        ocirc.remove()
    return out

def select(m,mhlf,distmod,mlim,extmap=None) :
    """ 
    Given a absolute magnitude luminosity function, mhlf, in units
    of fraction of total number of stars for absolute mags, m, an
    array of distances and an limiting apparent magnitude, return
    array of fraction of population sampled 
  
    Args:

    Keyword args:

    Returns:
    """

    s=[] 
    mhlim = mlim - distmod
    if extmap is not None :
        mhlim -= extmap
    for lim in mhlim :
        j=np.where(m < lim)
        s.append(mhlf[j].sum())
    return np.array(s)

def lfs(file='priorlfs.fits',clobber=False) :
    """ 
    Set up luminosity functions to be used for spatial priors
  
    Args:

    Keyword args:

    Returns:
    """

    global bulgelf, halolf, thinlf, thicklf
    # get LFs for components
    if os.path.exists(file) and not clobber:
        tmp = fits.open(file)
        thinlf=tmp[1].data.sum(axis=1)
        thicklf=tmp[2].data.sum(axis=1)
        bulgelf=tmp[3].data.sum(axis=1)
        halolf=tmp[4].data.sum(axis=1)
    else :
        mhmin=mh[0]
        mhmax=mh[len(mh)-1]
        jkmin=0.5
        jkmax=1.5
        # thin disk
        hdu1=isochrones.mkhess(files=['zp00.dat'],vals=['jk','h'],xmin=[jkmin,mhmin],xmax=[jkmax,mhmax],agerange=[8.3,10.],norm=True)
        thinlf=hdu1.data.sum(axis=1)
        # thick disk
        hdu2=isochrones.mkhess(files=['zm05.dat'],vals=['jk','h'],xmin=[jkmin,mhmin],xmax=[jkmax,mhmax],agerange=[9.6,10.],norm=True)
        thicklf=hdu2.data.sum(axis=1)
        # bulge
        hdu3=isochrones.mkhess(files=['zp00.dat'],vals=['jk','h'],xmin=[jkmin,mhmin],xmax=[jkmax,mhmax],agerange=[9.9,10.1],norm=True)
        bulgelf=hdu3.data.sum(axis=1)
        # halo
        hdu4=isochrones.mkhess(files=['zm13.dat'],vals=['jk','h'],xmin=[jkmin,mhmin],xmax=[jkmax,mhmax],agerange=[10.,10.1],norm=True)
        halolf=hdu4.data.sum(axis=1)
        hdu=fits.PrimaryHDU()
        out=fits.HDUList([hdu,hdu1,hdu2,hdu3,hdu4])
        out.writeto(file,clobber=True)
  

def init(doage=None,isoadj=False,clobber=False,prefix=None) :
    """ 
    Initialize isochrone grid and luminosity functions for spatial priors 
  
    Args:

    Keyword args:

    Returns:
    """
    print('Initilizing grid...')
    getgrid(doage=doage,isoadj=isoadj,clobber=clobber,prefix=prefix)
    print('Initilizing LFs for priors...')
    lfs()
    dust.setup()


def isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2,nskip=10,errage=None,dteff=0.,dwerrlogg=0.1) :
    """ 
    Test using input isochrone data: runs for full grid of isochrones!
  
    Args:

    Keyword args:
         nskip (int) : use every nskip'th line in isochrone data

    Returns:
    """

    m=10.
    first=True
    for feh0 in np.arange(-2,0.5,0.5) :

        a=isochrones.read(isochrones.isoname(feh0)+'.dat',agerange=[8.0 ,10])
        rec = np.recarray(1,dtype=[
                       ('age','f4'),
                       ('teff','f4'),
                       ('logg','f4'),
                       ('feh','f4'),
                       ('h','f4'),
                       ('dist','f4'),
                       ('diso','3f4'),
                       ])
        for i in range(len(a)) :
            #if (a['logg'][i] < 4)  & (a['teff'][i] < 5800) & (i%nskip == 0) :
            print(a['teff'][i],a['logg'][i],feh0,a['age'][i])
            if (a['teff'][i] < 5800) & (a['h'][i] < 7) & (i%nskip == 0) :
                # add simulated uncertainties
                teff = a['teff'][i] + np.random.normal(scale=errte) + dteff
                if a['logg'][i] < 4 :
                    logg = a['logg'][i] + np.random.normal(scale=errlogg)
                else :
                    logg = a['logg'][i] + np.random.normal(scale=dwerrlogg)
                feh = feh0 + np.random.normal(scale=errfeh)
                # get distance
                if errage is None :
                    if a['logg'][i] > 4 :
                      out = getdist(teff,logg,feh,[m],errte=errte,errlogg=dwerrlogg,errfeh=errfeh,mlim=12.2)
                    else :
                      out = getdist(teff,logg,feh,[m],errte=errte,errlogg=errlogg,errfeh=errfeh,mlim=12.2)
                else :
                    if a['logg'][i] > 4 :
                        out = getdist(teff,logg,feh,[m],errte=errte,errlogg=dwerrlogg,errfeh=errfeh,mlim=12.2,
                                  errage=errage,obsage=a['age'][i])
                    else :
                        out = getdist(teff,logg,feh,[m],errte=errte,errlogg=errlogg,errfeh=errfeh,mlim=12.2,
                                  errage=errage,obsage=a['age'][i])
                # true distance computed for input apparent magnitude
                true = 10.**((m-a['h'][i]+5)/5.)
                rec['age'] = a['age'][i]
                rec['teff'] = a['teff'][i]
                rec['logg'] = a['logg'][i]
                rec['feh'] = feh0
                rec['h'] = a['h'][i]
                rec['dist'] = true
                rec['diso'] = out.diso[0]
                print(a['age'][i], a['teff'][i], a['logg'][i], feh, true, out.diso[0])
                if first :
                    all = rec
                else :
                    all = np.append(all,rec)
                first = False

    return Table(all)

def plottest(all,file='test',index=2) :
    """ 
    Plots for a single test run
  
    Args:

    Keyword args:

    Returns:
    """   
    fig=plt.figure()
    ax=fig.add_subplot(2,3,1)
    plots.plotc(ax,all['teff'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='Teff')
    ax.set_xticks(np.arange(3000,6000,1000))
    ax=fig.add_subplot(2,3,2)
    plots.plotc(ax,all['feh'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='[M/H]')
    ax.set_xticks(np.arange(-2.5,1.0,1.0))
    ax=fig.add_subplot(2,3,3)
    plots.plotc(ax,all['logg'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='log g')
    ax=fig.add_subplot(2,3,4)
    plots.plotc(ax,all['h'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='H')
    ax=fig.add_subplot(2,3,5)
    plots.plotc(ax,all['age'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='Age')
    ax=fig.add_subplot(2,3,6)
    ax.hist((all['diso'][:,index]-all['dist'])/all['dist'],bins=np.arange(-1.,1.,0.05))
    ax.set_xlabel('Fractional distance error')
    #plots.plotc(ax,all['dist'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10],yr=[-1,1],xt='dist')
    #ax.set_xticks(np.arange(-1000,15000,3000))
    plt.tight_layout()
    plt.savefig(file+'.jpg')
    plt.close()
  
def alltest(clobber=False) :
    """ 
    Run several test runs with different input parameters
  
    Args:

    Keyword args:

    Returns:
    """

    init(prefix='bigraw')
    # test with typical uncertainties
    file = 'bigtest'
    try :
        all=fits.open(file+'.fits')[1].data
    except :
        all=isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2,dwerrlogg=0.75)
        all.write(file+'.fits',format='fits',overwrite=True)
    plottest(all,file=file)
    plottest(all,file=file+'med',index=1)

    # test with typical uncertainties, offset temperature
    file = 'bigtestoff'
    try :
        all=fits.open(file+'.fits')[1].data
    except :
        all=isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2,dteff=200,dwerrlogg=0.75)
        all.write(file+'.fits',format='fits',overwrite=True)
    plottest(all,file=file)
    plottest(all,file=file+'med',index=1)

    # test with very small uncertainties
    file = 'bigtest0'
    try :
        all=fits.open(file+'.fits')[1].data
    except :
        all=isotest(errte=5,errlogg=0.01,errfeh=0.005,mlim=12.,dwerrlogg=0.01)
        all.write(file+'.fits',format='fits',overwrite=True)
    plottest(all,file=file)
    plottest(all,file=file+'med',index=1)

    # test using age dimension
    #file = 'testage'
    #init(doage=True)
    #all=isotest(errte=50,errlogg=0.1,errfeh=0.05,errage=0.3,mlim=12.2) 
    #all.write(file+'.fits',format='fits',overwrite=True)
    #plottest(all,file=file)
    #plottest(all,file=file+'med',index=1)

def test(teff=4500.,logg=2.5,feh=0.,mag=10.,glon=10,glat=10,errte=50,errlogg=0.05,errfeh=0.05) :
    """ 
    Run a single test distance 
  
    Args:

    Keyword args:

    Returns:
    """
    init()
    out=getdist(teff,logg,feh,[mag],glon=glon,glat=glat,errte=errte,errlogg=errlogg,errfeh=errfeh)
    pdb.set_trace()
    return out

def distobj(data,obj) :
    """
    Distance run for specified object
  
    Args:

    Keyword args:

    Returns:
    """
    init()
    j=np.where(np.core.defchararray.strip(data['APOGEE_ID']) == obj)[0]
    rec=data[j[0]]
    distrec(rec)

def distrec(rec,param='PARAM',disp=None,plot=None,obsage=None,errage=10,verbose=False) :
    """
    Get distance for single record
  
    Args:
        rec (structured array) : standard APOGEE record

    Keyword args:
        param (char) : name of parameter array to use (PARAM or FPARAM)
        disp  : if set, use as display object for image display of data
        plot  : if set, use as axes object for plot

    Returns: structured array of output distance information
 
    """

    # get extinction and limiting mag for this object
    mlim = getmlim(rec)
    ext = bestext(rec)
    # compute distance given parameters within range
    if rec[param][1] > -1  and rec[param][0] < 5800 and rec[param][0] > 3000 and ext[1]  > -0.1 : 
        iso_fe_h = feh_corrected(rec[param][3],rec[param][6])
        if verbose : print('  ',rec[param][0],rec[param][1],iso_fe_h,rec['H'],ext,rec['GLON'],rec['GLAT'])
        out=getdist(rec[param][0],rec[param][1],iso_fe_h,[rec['H']],glon=rec['GLON'],glat=rec['GLAT'],
                    mlim=mlim, ext=[ext[1]],
                    errte=50, errlogg=0.1, errfeh = 0.05, disp=disp, plot=plot, obsage=obsage, errage=errage)
        return out
    elif rec['FPARAM'][1] > -1  and rec['FPARAM'][0] < 5800 and rec['FPARAM'][0] > 3000 and ext[1]  > -0.1 : 
        iso_fe_h = feh_corrected(rec['FPARAM'][3],rec['FPARAM'][6])
        print(' using FPARAM: ',rec['FPARAM'][0],rec['FPARAM'][1],iso_fe_h,rec['H'],ext,rec['GLON'],rec['GLAT'])
        if rec['FPARAM'][1] > 4 :
           errlogg =0.75
        else :
           errlogg =0.1
        out=getdist(rec['FPARAM'][0],rec['FPARAM'][1],iso_fe_h,[rec['H']],glon=rec['GLON'],glat=rec['GLAT'],
                    mlim=mlim, ext=[ext[1]],
                    errte=50, errlogg=errlogg, errfeh = 0.05, disp=disp, plot=plot, obsage=obsage, errage=errage)
        return out
    else :
        return None

def main(file=None,nmax=None,raw=False) :
    """
    Get distances for entire file and output
  
    Args:

    Keyword args:

    Returns:
    """
    # read input structure
    if file is None :
        a=apload.allStar()[1].data    
        file='allStar'
    else :
        a=fits.open(file+'.fits')[1].data

    if nmax is not None :
        a=a[0:nmax]

    # initialize and do dummy run to get output recarray to append to structure
    init(isoadj=True,prefix='bigadj')
    out=getdist(4500.,2.5,0.,[10.])
    data=struct.add_cols(a,out)

    if raw :
        param = 'FPARAM'
    else :
        param = 'PARAM'

    # loop over objects
    for rec in data :
        out=distrec(rec,param=param,disp=None,plot=None) 

        if out is not None:
            for tag in out.dtype.names :
                rec[tag] = out[tag]

    # write out file with extra distance tags
    tab=Table(data)
    tab.write(file+'+.fits',format='fits',overwrite=True)
 
def getmlim(rec) :
    """ 
    Get limiting magnitude for input record 
  
    Args:

    Keyword args:

    Returns:
    """

    # CURRENTLY DOES NOT ACCOUNT FOR NVISITS!
    #mlim=12.2
    #if (rec['APOGEE_TARGET2'] & 2**11) :
    #    mlim=12.2
    #elif (rec['APOGEE_TARGET2'] & 2**12) :
    #    mlim=12.8
    #elif (rec['APOGEE_TARGET2'] & 2**13) :
    #    mlim=13.3
    #print(rec['APOGEE_ID'],rec['FIELD'], mlim)
    #return mlim
    if rec['MAX_H'] > 0 :
        return rec['MAX_H']
    else :
        return 12.2
 
def bestext(rec,verbose=False) :
    """ 
    get "best" H band extinction from targetting info
  
    Args:

    Keyword args:

    Returns:
    """

    if type(rec['AK_TARG']) == np.float32 :
        # single object
        # use AK_TARG if IRAC was used, else AK_WISE
        if 'IRAC' in rec['AK_TARG_METHOD'] :
            ak = rec['AK_TARG']
        else :
            ak = rec['AK_WISE']
        # at |b|>16, use SFD if it is lower
        if (abs(rec['GLAT']) > 16) and ( (0.302*rec['SFD_EBV'] < 1.2*ak) or (ak <0) ):
            ak =  0.302 * rec['SFD_EBV']
        # assume no extinction for hipparcos sample
        if rec['FIELD'].strip() == 'hip' :
            ak = 0.
        if verbose : print('  AK: ', rec['AK_TARG_METHOD'], rec['AK_TARG'], rec['AK_WISE'], ak)
    else :
        # array input/output
        ak=rec['AK_WISE']
        gd=np.where(np.core.defchararray.find(rec['AK_TARG_METHOD'],'IRAC') >= 0)[0]
        ak[gd]=rec['AK_TARG']
        gd=np.where((abs(rec['GLAT']) > 16) & ((0.302*rec['SFD_EBV'] < 1.2*ak) | (ak<0)) )[0]
        ak[gd]=0.302 * rec['SFD_EBV'][gd]
        gd=np.where(np.core.defchararray.strip(rec['FIELD']) == 'hip')[0]
        ak[gd]=0.
    #ah = ak*.53/.36
    ah = 1.55*ak  # from Indebetouw 2005
    aj = 2.5*ak
    return [aj,ah,ak]
 
def feh_corrected(feh,alpham) : 
    """ 
    Return Salaris (1993) corrected metallicity given input metallicity and alpha-enhancement
  
    Args:

    Keyword args:

    Returns:
    """
    if alpham > 0 :
        f_alpha = 10.**alpham
        f = 0.638*f_alpha + .362
        return feh+np.log10(f)
    else :
        return feh

def merge(out='allStar+',n=15) :
    """ 
    Merge multiple files (from split) into a single one, and output
  
    Args:

    Keyword args:

    Returns:
    """
    for i in range(n) :
        infile = 's{:02d}+'.format(i)
        print(infile)
        f=fits.open(infile+'.fits')[1].data
        if i == 0 :
            all = f
        else :
            all = np.append(all,f)
    tab=Table(all)
    tab.write(out+'.fits',format='fits',overwrite=True)

def split(write=True,run=True,n=15) :
    """ 
    Split allStar file into muliple pieces, and run each in parallel 
  
    Args:

    Keyword args:

    Returns:
    """
    apload.dr13()
    #apload.aspcap='l30g'
    #apload.results='l30g'
    data=apload.allStar()[1].data    
    ntot=len(data)
    for i in range(n) :
        i1=i*ntot/n
        if i < n-1 :
            i2=(i+1)*ntot/n-1
        else :
            # on last one, get all the rest of the stars
            i2=ntot-1
        file = 's{:02d}'.format(i)
        print(file)
        if write :
            out=Table(data[i1:i2])
            out.write(file+'.fits',format='fits',overwrite=True)
        if run :
            log = open(file+'.log','w')
            subprocess.Popen(["python","/home/holtz/python/holtz/gal/dist.py",file],stdout=log,stderr=subprocess.STDOUT)

def testall() :
    testcat(plot=True)
    testcat(plot=True,index=1)
    testcat(plot=True,index=2)
    testcat(plot=True,bpg=True)
    testcat(plot=True,ms=True)
    testcat(plot=True,msbayes=True)
    testcat(plot=True,wang=True)
    testcat(plot=True,cannon=True)

def testcat(disp=None,plot=True,bpg=False,ms=False,msbayes=False,wang=False,cannon=False,ness=False,dr12=False,index=0) :
    """ 
    Compare distances with catalog distances 
  
    Args:

    Keyword args:

    Returns:
    """

    from esutil import htm
    # read catalog
    rc=fits.open('apogee-rc-DR13.fits')[1].data
    cat=fits.open('apogee-distances_DR12_v1.3.fits')
    cat.verify('fix')
    cat=cat[1].data
    if bpg :
        all=fits.open('apogee-BPGdistances_DR13_v2.0.fits')[1].data
        prefix='bpg'
    elif ms :
        msall=ascii.read('MS_Isochrone-Match_DR13-Dist-Cat_simple.csv')
        all=fits.open('allStar+.fits')[1].data
        prefix='ms'
    elif msbayes :
        msall=ascii.read('MS_Bayesian_DR13-Dist-Cat_simple.csv')
        all=fits.open('allStar+.fits')[1].data
        prefix='msbayes'
    elif wang :
        msall=ascii.read('wang.dat')
        all=fits.open('allStar+.fits')[1].data
        prefix='wang'
    elif cannon :
        msall=ascii.read('cannon-distances.csv')
        all=fits.open('allStar+.fits')[1].data
        prefix='cannon'
    elif ness :
        msall=ascii.read('ids_distances.txt')
        l=np.core.defchararray.split(msall['ids'],'.')
        for i in range(len(l)) :
            ll=l[i]
            msall['ids'][i]=ll[-1]
        all=fits.open('allStar+.fits')[1].data
        prefix='ness'
    elif dr12 :
        all=fits.open('allStar+-dr12.fits')[1].data
        prefix='dr12_diso{:1d}'.format(index)
    else :
        all=fits.open('allStar+.fits')[1].data
        prefix='diso{:1d}'.format(index)

    # match objects
    h=htm.HTM()
    maxrad=1./3600.
    if ms or msbayes or wang or cannon or ness:
        # replace distances in all with MS distances
        all['diso'][:,0] = -1
        if ms :
            m1ms,m2ms=match.match(msall['MS_Isochrone-Match_DR13 ID_2MASS'],all['APOGEE_ID'])
            all['diso'][m2ms,0] = msall['dist_kpc'][m1ms]*1000.
        elif msbayes :
            m1ms,m2ms=match.match(msall['MS_Bayesian_DR13 ID_2MASS'],all['APOGEE_ID'])
            all['diso'][m2ms,0] = msall['dist_kpc'][m1ms]*1000.
        elif wang :
            m1ms,m2ms=match.match(msall['col1'],all['APOGEE_ID'])
            all['diso'][m2ms,0] = msall['col4'][m1ms]*1000.
            all['ak_targ'][m2ms] = msall['col8'][m1ms]
        elif cannon :
            m1ms,m2ms=match.match(msall['APOGEE_ID'],all['APOGEE_ID'])
            all['diso'][m2ms,0] = msall['DIST_Padova'][m1ms]*1000.
        elif ness :
            m1ms,m2ms=match.match(msall['ids'],all['APOGEE_ID'])
            all['diso'][m2ms,0] = msall['distance'][m1ms]*1000.
            pdb.set_trace()
    m1,m2,rad=h.match(all['RA'],all['DEC'],cat['RA'],cat['DEC'],maxrad)
    m1rc,m2rc,rad=h.match(all['RA'],all['DEC'],rc['RA'],rc['DEC'],maxrad)

    # match clusters
    clust=apselect.clustdata()
    mc=np.array([],dtype=int)
    dc=np.array([])
    for i in range(len(clust.name)) :
        print(clust[i].name,clust[i].dist)
        j=apselect.clustmember(all,clust[i].name,logg=[-1,5],raw=True)
        if len(j) > 0 :
            mc=np.append(mc,j)
            dc=np.append(dc,np.zeros(len(j))+clust[i].dist*1000)

    # make various catalog comparison plots
    if plot :
        f=open(prefix+'.txt','w')
        f.write(prefix)
        n,m,s=plotcat(prefix+'_hip',all[m1],1000./(cat['HIP_PLX'][m2]),index=index)
        f.write('{:6d}{:6.2f}{:6.2f}'.format(n,m,s))
        n,m,s=plotcat(prefix+'_clust',all[mc],dc,index=index)
        f.write('{:6d}{:6.2f}{:6.2f}'.format(n,m,s))
        j=apselect.select(all,field='BAADEWIN_001-04')
        n,m,s = plotcat(prefix+'_apokasc',all[m1],1000.*cat['APOKASC_DIST_BAYES'][m2],index=index)
        f.write('{:6d}{:6.2f}{:6.2f}'.format(n,m,s))
        n,m,s=plotcat(prefix+'_rc',all[m1rc],1000.*rc['RC_DIST'][m2rc],index=index)
        f.write('{:6d}{:6.2f}{:6.2f}'.format(n,m,s))
        n,m,s=plotcat(prefix+'_bw',all[j],all[j]['RA']*0+8500.,index=index)
        f.write('{:6d}{:6.2f}{:6.2f}\n'.format(n,m,s))
        f.close()

    else :
     fig=plt.figure()
     ax=fig.add_subplot(211)
     ax2=fig.add_subplot(212)
     for i in range(len(m1)) :
      if np.isfinite(cat['HIP_PLX'][m2[i]]) :
        print(all[m1[i]]['apogee_id'])
        print(all[m1[i]]['aspcapflags'])
        print(all[m1[i]]['diso'],1000./cat['HIP_PLX'][m2[i]],5*np.log10(all[m1[i]]['diso'][0]/(1000./cat['HIP_PLX'][m2[i]])))
        print(all[m1[i]]['ak_targ'],all[m1[i]]['ak_wise'],all[m1[i]]['sfd_ebv']*0.302,all[m1[i]]['glat'])
        print(all[m1[i]]['param'])
        print('parallax, m-M, H: ',cat[m2[i]]['hip_plx'],all[m1[i]]['H']-(5*np.log10(1000./cat[m2[i]]['HIP_PLX'])-5),all[m1[i]]['H'])
        plots.plotl(ax,all[m1[i]]['diso_dist'],all[m1[i]]['diso_pdf'],xr=[0,1000])
        for j in range(3) :
          x=all[m1[i]]['diso'][j]
          ax.plot([x,x],[0,1])
        x=1000./cat['HIP_PLX'][m2[i]]
        ax.plot([x,x],[0,1],color='k')
        if abs(5*np.log10(all[m1[i]]['diso'][0]/(1000./cat['HIP_PLX'][m2[i]]))) > 0.0 :
          if disp is not None :
            out=distrec(all[m1[i]],plot=ax2,disp=disp)
        plt.figure(fig.number)
        ax.cla()
        ax2.cla()

def testclust(disp=None,suffix=None) :
    a=fits.open('allStar+'+suffix+'.fits')[1].data
    clust=apselect.clustdata()

    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
    alldata=np.array([])
    allmh=np.array([])
    for i in range(len(clust.name)) :
        print(clust[i].name,clust[i].dist)
        j=apselect.clustmember(a,clust[i].name)
        hdata=a['DISO'][j,1]/(clust[i].dist*1000)
        alldata=np.append(alldata,hdata)
        allmh=np.append(allmh,a[j]['M_H'])
        # individual clust plots? set minimum number of members to plot
        if len(j) > 3000 :
          plots.plotc(ax1,a[j]['TEFF'],a['DISO'][j,1],a[j]['M_H'],size=15,zr=[-2,0.5])
          ax1.plot([0,10000],[clust[i].dist*1000,clust[i].dist*1000])
          plots.plotc(ax2,a[j]['TEFF'],a[j]['LOGG'],a[j]['M_H'],size=15,zr=[-2,0.5],xr=[5500,3500],yr=[5,0])
          plots.plotc(ax3,a[j]['J']-a[j]['K'],a[j]['H'],a[j]['M_H'],size=15,zr=[-2,0.5],xr=[0,2],yr=[15,5])
          ax4.hist(np.log10(hdata[np.where(~np.isnan(hdata) & (hdata>0))]))
          #for i in range(len(j)) :
          #  if disp is not None :
          #    out=distrec(a[j[i]],plot=ax4,disp=disp)
          #pdb.set_trace()
          plt.savefig('dist_'+clust[i].name+'.jpg')
          ax1.cla()
          ax2.cla()
          ax3.cla()
          ax4.cla()
    plt.close()
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
    plots.plotc(ax1,allmh,alldata,allmh,xr=[-2.5,1.0],yr=[0,2.],zr=[-2.5,1.0])
    hist,bins=np.histogram(alldata,bins=np.arange(0.,2.,0.05))
    plots.plotl(ax2,bins[0:-1],hist)
    plt.savefig('dist_all'+suffix+'.jpg')
       
    
def plotcat(file,all,dist,index=2,inter=False):
    """ 
    Plots for catalog comparisons 
  
    Args:

    Keyword args:

    Returns:
    """
    try:
        m_h = all['M_H']
    except:
        m_h = all['PARAM_M_H']
    z=m_h
    zr=[-2.,0.5]
    #ak=bestext(all)[2]
    #z=ak
    #zr=[-0.05,0.2]
    try:
        apogeedist=all['diso'][:,index]
    except:
        apogeedist=all['d_med']*1000.

    fig, ax = plt.subplots(2,3)
    #fig.subplots_adjust(wspace=0.)
    plots.plotc(ax[0,0],all['teff'],(apogeedist-dist)/dist,z,zr=zr,xr=[2500,6500],yr=[-1,1],xt='Teff',yt='Fractional distance error')
    ax[0,0].set_xticks(np.arange(3000,7000,1000))

    plots.plotc(ax[0,1],m_h,(apogeedist-dist)/dist,z,zr=zr,xr=[-3,1],yr=[-1,1],xt='[M/H]')
    plt.setp(ax[0,1].get_yticklabels(),visible=False)
    ax[0,1].set_xticks(np.arange(-2.5,1.0,1.0))

    plots.plotc(ax[0,2],all['logg'],(apogeedist-dist)/dist,z,zr=zr,xr=[-1,5],yr=[-1,1],xt='log g')
    plt.setp(ax[0,2].get_yticklabels(),visible=False)

    plots.plotc(ax[1,0],all['h'],(apogeedist-dist)/dist,z,zr=zr,yr=[-1,1],xt='H',yt='Fractional distance error')

    #plots.plotc(ax[1,1],all['diso'][:,index],(all['diso'][:,index]-dist)/dist,m_h,zr=[-2,0.5],yr=[-1,1],xt='Distance')
    #plt.setp(ax[0,1].get_yticklabels(),visible=False)
    #xmin,xmax= ax[1,1].get_xlim()
    #ax[1,1].set_xticks(np.linspace(xmin,xmax,5)[1:-1])
    ax[1,1].hist((apogeedist-dist)/dist,bins=np.arange(-1.,1,0.05),range=[-1.,1.])
    ax[1,1].set_xlabel('Fractional dist error')
    frac=(apogeedist-dist)/dist
    gd=np.where((frac>-1) & (frac<1) )[0]
    ymin,ymax=ax[1,1].get_ylim()
    ax[1,1].text(-0.9,ymax-0.05*(ymax-ymin),'{:3d}{:5.2f}{:5.2f}'.format(len(gd),frac[gd].mean(),frac[gd].std()))
   
    #plots.plotc(ax[1,2],5*np.log10(dist)-5,5*np.log10(all['diso'][:,index])-5,all['logg'],zr=[0,5],xt='m-M')
    #plots.plotc(ax[1,2],5*np.log10(dist)-5,5*np.log10(all['diso'][:,index])-5*np.log10(dist),bestext(all),zr=[0,1],xt='m-M',yr=[-1.5,1.5])
    #ax[1,2].yaxis.tick_right()
    plots.plotc(ax[1,2],all['teff'],all['logg'],5*np.log10(apogeedist)-5*np.log10(dist),zr=[-1,1],xt='Teff',xr=[6500,3500],yr=[5,0],yt='log g')
    ax[1,2].set_xticks(np.arange(3000,7000,1000))

    if inter :
        x = 5*np.log10(dist)-5
        y = 5*np.log10(apogeedist)-5
        gd, = np.where(np.isfinite(x) & np.isfinite(y))
        plots._data_x = x[gd]
        plots._data_y = y[gd]
        plots._data = all[gd]
        plots.event(fig)
    else :
        plt.tight_layout()
        plt.savefig(file+'.jpg')
        plt.close()
    #f=open(file+'.txt','w')
    #f.write('{:3d}{:5.2f}{:5.2f}'.format(len(gd),frac[gd].mean(),frac[gd].std()))
    return len(gd), frac[gd].mean(),frac[gd].std()

def median_index(x,p) :
    """ 
    Return the index and value of the median value for input NORMALIZED probability array
  
    Args:

    Keyword args:

    Returns:
    """
    tot =0.
    i=0
    while tot < 0.5 :
       tot0=tot
       tot += p[i]
       i +=1
    xmed = x[i-1]+(x[i]-x[i-1])*(0.5-tot0)/(tot-tot0)
    return i,xmed

def index(t,g,f,a) :
    """ 
    Return array indices for input values
  
    Args:

    Keyword args:

    Returns:
    """
    it= int((t-teff[0])/dteff)
    ig = int((g-logg[0])/dlogg)
    ifeh = int((f-feh[0])/dfeh)
    try:
        ia= int((a-age[0])/dage )
    except:
        ia=0
    return it,ig,ifeh,ia
   



if __name__ == '__main__' :
    main(sys.argv[1])

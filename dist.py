import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pdb
import os
import sys
import subprocess
from sdss.apogee import apload
from astropy.io import fits
from astropy.table import Table
import isochrones
import galmodel
import dust
from holtz.tools import plots
from holtz.tools import struct

def getgrid(doage=None,clobber=False):
    ''' Initialize 4D (Teff, logg, [Fe/H], M_H) or 5D (with doage=True) isochrone grid '''

    global teff, logg, feh, mh, grid, dteff, dlogg, dfeh, age, dage

    dteff=25
    dlogg=0.05
    dfeh=0.05
    teff=np.arange(3000.,6000.,dteff)
    logg=np.arange(-1.,5.0,dlogg)
    feh=np.arange(-2.1,0.6,dfeh)
    #feh=np.arange(-0.2,0.2,dfeh)
    mh=np.arange(-7.,3.,0.1)
    if doage is None :
        vals=['teff','logg','feh','h']
        vmin=[teff[0],logg[0],feh[0],mh[0]]
        nbin=[teff.shape[0], logg.shape[0], feh.shape[0], mh.shape[0]]
        vmax=[teff[nbin[0]-1], logg[nbin[1]-1], feh[nbin[2]-1], mh[nbin[3]-1]]
        file = 'grid.fits'
    else :
        dage=0.05
        age=np.arange(8.,10.15,dage)
        vals=['teff','logg','feh','age','h']
        vmin=[teff[0],logg[0],feh[0],age[0],mh[0]]
        nbin=[teff.shape[0], logg.shape[0], feh.shape[0], age.shape[0], mh.shape[0]]
        vmax=[teff[nbin[0]-1], logg[nbin[1]-1], feh[nbin[2]-1], age[nbin[3]-1], mh[nbin[4]-1]]
        file = 'gridage.fits'

    files=(['zm21.dat','zm20.dat','zm19.dat','zm18.dat','zm17.dat','zm16.dat','zm15.dat','zm14.dat','zm13.dat','zm12.dat','zm11.dat','zm10.dat',
       'zm09.dat','zm08.dat','zm07.dat','zm06.dat','zm05.dat','zm04.dat','zm03.dat','zm02.dat','zm01.dat',
       'zp00.dat','zp01.dat','zp02.dat','zp03.dat','zp04.dat','zp05.dat','zp06.dat'])
    agerange=[8.0,10.15]

    #files=['zp00.dat']
    #agerange=[9.39,9.41]
    #file='gridtest.fits'

    if os.path.exists(file) and not clobber:
        grid = fits.open(file)[0]
    else :
        grid=isochrones.mkhess(agerange=agerange,files=files,vals=vals,xmin=vmin,xmax=vmax,nbins=nbin)
        grid.writeto(file,clobber=True)
   

def getdist(obste,obslogg,obsfeh,obsmag,ext=0.,glon=None,glat=None,errte=50,errlogg=0.1,errfeh=0.1,mlim=12.2,obsage=None,errage=10.,plot=False):
    ''' Return distance estimates for input observed parameters, errors, limiting mag for single object'''

    # get probability of observed Teff, and limits over which to marginalize
    probte=np.exp(-0.5*(obste-teff)**2/errte**2)
    itmin=max(0,np.floor((obste-3*errte-teff[0])/dteff).astype('int'))
    itmax=min(len(teff)-1,np.ceil((obste+3*errte-teff[0])/dteff).astype('int'))

    # get probability of observed logg, and limits over which to marginalize
    problogg=np.exp(-0.5*(obslogg-logg)**2/errlogg**2)
    iloggmin=max(0,np.floor((obslogg-3*errlogg-logg[0])/dlogg).astype('int'))
    iloggmax=min(len(logg)-1,np.ceil((obslogg+3*errlogg-logg[0])/dlogg).astype('int'))

    # get probability of observed [Fe/H], and limits over which to marginalize
    probfeh=np.exp(-0.5*(obsfeh-feh)**2/errfeh**2)
    ifehmin=min(len(feh)-1,max(0,np.floor((obsfeh-3*errfeh-feh[0])/dfeh).astype('int')))
    ifehmax=max(0,min(len(feh)-1,np.ceil((obsfeh+3*errfeh-feh[0])/dfeh).astype('int')))

    # get probability of age, and limits over which to marginalize
    if obsage is not None :
        probage=np.exp(-0.5*(obsage-age)**2/errage**2)
        iagemin=np.max([0,np.floor((obsage-3*errage-age[0])/dage).astype('int')])
        iagemax=np.min([len(age)-1,np.ceil((obsage+3*errage-age[0])/dage).astype('int')])

    # get the probability grid 
    #obs=grid.data*0.
    print itmin, itmax, iloggmin, iloggmax, ifehmin, ifehmax
    if obsage is None :
        obs=np.zeros([len(mh),ifehmax-ifehmin+1,iloggmax-iloggmin+1,itmax-itmin+1])
    else :
        obs=np.zeros([len(mh),iagemax-iagemin+1,ifehmax-ifehmin+1,iloggmax-iloggmin+1,itmax-itmin+1])

    for it in range(itmin,itmax) :
        #print 't:', it, itmin,itmax
        for ig in range(iloggmin,iloggmax) :
            #print 'g:', ig, iloggmin, iloggmax
            p1 = probte[it]*problogg[ig]
            for ifeh in range(ifehmin,ifehmax) :
                #print 'fe: ',ifeh, ifehmin, ifehmax
                prob = p1*probfeh[ifeh]
                if obsage is None :
                    for imh in range(len(mh)-1) :
                        obs[imh,ifeh-ifehmin,ig-iloggmin,it-itmin] = grid.data[imh,ifeh,ig,it]*prob
                else :
                    for iage in range(iagemin,iagemax) :
                        p2 = prob*probage[iage]
                        #print iage, iagemin, iagemax, obsage, errage, ifeh,ig,it
                        #print obs.shape, grid.data.shape
                        obs[:,iage-iagemin,ifeh-ifehmin,ig-iloggmin,it-itmin] = grid.data[:,iage,ifeh,ig,it]*p2
                        #for imh in range(len(mh)-1) :
                        #    obs[imh,iage,ifeh,ig,it] = grid.data[imh,iage,ifeh,ig,it]*p2

    #marginalize and normalize
    if obsage is None :
        probmh=obs.sum(axis=(1,2,3))
    else :
        probmh=obs.sum(axis=(1,2,3,4))
    probmh/=probmh.sum()
    if plot == True :
        print mh[probmh.argmax()],median_index(mh,probmh)[1],(probmh*mh).sum()/probmh.sum()
        plt.plot(mh,probmh)
        pdb.set_trace()

    # get distances corresponding to M_H
    distmod = obsmag-mh-ext
    dist=10.**((distmod+5)/5.)

    # get max, median, mean distance from M_H PDF, no prior
    diso = dist[probmh.argmax()],median_index(dist,probmh)[1],(probmh*dist).sum()/probmh.sum()
    diso_dist = dist
    diso_pdf = probmh

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

            # get "extinction" distance
            i=0
            while extmap[i] > ext  and i < len(extmap)-1:
                i+=1
            if i < len(extmap) :
                dext = distmod[i-1]+(distmod[i]-distmod[i-1])*(ext-extmap[i-1])/(extmap[i]-extmap[i-1])
            else :
                dext = -1.

            # get the density prior, accounting for selection function
            dprior=(select(mh,bulgelf,distmod,mlim,extmap=extmap)*p[2]+
                    select(mh,halolf,distmod,mlim,extmap=extmap)*p[3]+
                    select(mh,thinlf,distmod,mlim,extmap=extmap)*p[0]+
                    select(mh,thicklf,distmod,mlim,extmap=extmap)*p[1])
            probmh*=dprior
            probmh/=probmh.sum()

            # get max, median, mean distance from PDF with prior
            diso_gal = dist[probmh.argmax()],median_index(dist,probmh)[1],(probmh*dist).sum()/probmh.sum()
            diso_gal_dist = dist
            diso_gal_pdf = probmh

    # output
    out = np.recarray(1,dtype=[
                       ('diso','3f4'),
                       ('diso_dist','100f4'),
                       ('diso_pdf','100f4'),
                       ('diso_gal','3f4'),
                       ('diso_gal_dist','100f4'),
                       ('diso_gal_pdf','100f4'),
                       ('dext','f4'),
                       ])
    out.diso = diso
    out.diso_dist = diso_dist
    out.diso_pdf = diso_pdf
    out.diso_gal = diso_gal
    out.diso_gal_dist = diso_gal_dist
    out.diso_gal_pdf = diso_gal_pdf
    return out

def select(m,mhlf,distmod,mlim,extmap=None) :
    ''' 
    Given a absolute magnitude luminosity function, mhlf, in units
    of fraction of total number of stars for absolute mags, m, an
    array of distances and an limiting apparent magnitude, return
    array of fraction of population sampled 
    '''

    s=[] 
    mhlim = mlim - distmod
    if extmap is not None :
        mhlim -= extmap
    for lim in mhlim :
        j=np.where(m < lim)
        s.append(mhlf[j].sum())
    return np.array(s)

def lfs(file='priorlfs.fits',clobber=False) :
    ''' Set up luminosity functions to be used for spatial priors'''

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
  

def init(doage=None) :
    ''' Initialize isochrone grid and luminosity functions for spatial priors '''
    print 'Initilizing grid...'
    getgrid(doage=doage)
    print 'Initilizing LFs for priors...'
    lfs()


def isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2,nskip=10,errage=None,dteff=0.) :
    ''' Test using input isochrone data'''

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
            if (a['logg'][i] < 4)  & (a['teff'][i] < 5800) & (i%nskip == 0) :
                # add simulated uncertainties
                teff = a['teff'][i] + np.random.normal(scale=errte) + dteff
                logg = a['logg'][i] + np.random.normal(scale=errlogg)
                feh = feh0 + np.random.normal(scale=errfeh)
                # get distance
                if errage is None :
                    out = getdist(teff,logg,feh,m,errte=errte,errlogg=errlogg,errfeh=errfeh,mlim=12.2)
                else :
                    out = getdist(teff,logg,feh,m,errte=errte,errlogg=errlogg,errfeh=errfeh,mlim=12.2,
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
                print a['age'][i], a['teff'][i], a['logg'][i], feh, true, out.diso[0]
                if first :
                    all = rec
                else :
                    all = np.append(all,rec)
                first = False

    return Table(all)

def plottest(all,file='test',index=2) :
    ''' Plots for a single test run'''   
    fig=plt.figure()
    ax=fig.add_subplot(2,3,1)
    plots.plotc(ax,all['teff'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    ax=fig.add_subplot(2,3,2)
    plots.plotc(ax,all['feh'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    ax=fig.add_subplot(2,3,3)
    plots.plotc(ax,all['logg'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    ax=fig.add_subplot(2,3,4)
    plots.plotc(ax,all['h'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    ax=fig.add_subplot(2,3,5)
    plots.plotc(ax,all['age'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    ax=fig.add_subplot(2,3,6)
    plots.plotc(ax,all['dist'],(all['diso'][:,index]-all['dist'])/all['dist'],all['age'],zr=[9,10])
    plt.tight_layout()
    plt.savefig(file+'.jpg')
  
def alltest() :
    ''' Run several test runs with different input parameters'''

    init()
    # test with typical uncertainties
    #file = 'test'
    #all=isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2) 
    #all.write(file+'.fits',format='fits',overwrite=True)
    #plottest(all,file=file)
    #plottest(all,file=file+'med',index=1)

    # test with typical uncertainties, offset temperature
    file = 'testoff'
    all=isotest(errte=50,errlogg=0.1,errfeh=0.05,mlim=12.2,dteff=200) 
    all.write(file+'.fits',format='fits',overwrite=True)
    plottest(all,file=file)
    plottest(all,file=file+'med',index=1)

    # test with very small uncertainties
    #file = 'test0'
    #all=isotest(errte=5,errlogg=0.01,errfeh=0.005,mlim=12.2)
    #all.write(file+'.fits',format='fits',overwrite=True)
    #plottest(all,file=file)
    #plottest(all,file=file+'med',index=1)

    # test using age dimension
    #file = 'testage'
    #init(doage=True)
    #all=isotest(errte=50,errlogg=0.1,errfeh=0.05,errage=0.3,mlim=12.2) 
    #all.write(file+'.fits',format='fits',overwrite=True)
    #plottest(all,file=file)
    #plottest(all,file=file+'med',index=1)

def test(teff=4500.,logg=2.5,feh=0.,mag=10.,glon=10,glat=10,errte=50,errlogg=0.05,errfeh=0.05) :
    ''' Run a single test distance '''
    init()
    out=getdist(teff,logg,feh,mag,glon=glon,glat=glat,errte=errte,errlogg=errlogg,errfeh=errfeh)
    pdb.set_trace()
    return out

def distrec(rec) :
    ''' get distance for single "record"'''

    mlim = getmlim(rec)
    ext = bestext(rec)
    try:
        iso_fe_h = feh_corrected(rec['M_H'],rec['ALPHA_M'])
    except:
        iso_fe_h = feh_corrected(rec['PARAM_M_H'],rec['PARAM_ALPHA_M'])
    print rec['TEFF'],rec['LOGG'],iso_fe_h,rec['H'],rec['GLON'],rec['GLAT'],mlim,ext

    out=getdist(rec['TEFF'],rec['LOGG'],iso_fe_h,rec['H'],glon=rec['GLON'],glat=rec['GLAT'],
                mlim=mlim, ext=ext,
                errte=50, errlogg=0.1, errfeh = 0.05)
    return out

def main(file=None,nmax=None,raw=False) :

    dust.setup()
    # read input structure
    if file is None :
        a=apload.allStar()[1].data    
        file='allStar'
    else :
        a=fits.open(file+'.fits')[1].data

    if nmax is not None :
        a=a[0:nmax]

    # initialize and do dummy run to get output recarray to append to structure
    init()
    out=getdist(4500.,2.5,0.,10.)
    data=struct.add_cols(a,out)

    if raw :
        param = 'FPARAM'
    else :
        param = 'PARAM'

    # loop over objects
    for rec in data :
        # get extinction and limiting mag for this object
        mlim = getmlim(rec)
        ext = bestext(rec)

        # compute distance given parameters within range
        if rec[param][1] > -1  and rec[param][0] < 5800 and rec[param][0] > 3000 and ext  > -0.1 : 
            iso_fe_h = feh_corrected(rec[param][3],rec[param][6])
            print '  ',rec[param][0],rec[param][1],iso_fe_h,rec['H'],ext,rec['GLON'],rec['GLAT']
            out=getdist(rec[param][0],rec[param][1],iso_fe_h,rec['H'],glon=rec['GLON'],glat=rec['GLAT'],
                        mlim=mlim, ext=ext,
                        errte=50, errlogg=0.1, errfeh = 0.05)
            for tag in out.dtype.names :
                rec[tag] = out[tag]

    # write out file with extra distance tags
    tab=Table(data)
    tab.write(file+'+.fits',format='fits',overwrite=True)
 
def getmlim(rec) :
    ''' Get limiting magnitude for input record '''

    # CURRENTLY DOES NOT ACCOUNT FOR NVISITS!
    #mlim=12.2
    #if (rec['APOGEE_TARGET2'] & 2**11) :
    #    mlim=12.2
    #elif (rec['APOGEE_TARGET2'] & 2**12) :
    #    mlim=12.8
    #elif (rec['APOGEE_TARGET2'] & 2**13) :
    #    mlim=13.3
    #print rec['APOGEE_ID'],rec['FIELD'], mlim
    #return mlim
    if rec['MAX_H'] > 0 :
        return rec['MAX_H']
    else :
        return 12.2
 
def bestext(rec) :
    ''' get "best" H band extinction from targetting info'''
    if (rec['FIELD']) == 'hip' :
        ak = 0.
    elif type(rec['AK_TARG']) == np.float32 :
        if 'IRAC' in rec['AK_TARG_METHOD'] :
            ak = rec['AK_TARG']
        elif abs(rec['GLAT']) > 16 :
            ak =  0.302 * rec['SFD_EBV']
        else :
            ak = rec['AK_WISE']
        print '  AK: ', rec['AK_TARG_METHOD'], rec['AK_TARG'], rec['AK_WISE'], ak
    else :
        ak=rec['AK_WISE']
        gd=np.where(abs(rec['GLAT']) > 16)[0]
        ak[gd]=0.302 * rec['SFD_EBV'][gd]
        gd=np.where(np.core.defchararray.find(rec['AK_TARG_METHOD'],'IRAC') >= 0)[0]
        ak[gd]=rec['AK_TARG']
    return ak*.53/.36
 
def feh_corrected(feh,alpham) : 
    ''' Return Salaris (1993) corrected metallicity given input metallicity and alpha-enhancement'''
    if alpham > 0 :
        f_alpha = 10.**alpham
        f = 0.638*f_alpha + .362
        return feh+np.log10(f)
    else :
        return feh

def merge(out='allStar+',n=15) :
    ''' Merge multiple files (from split) into a single one, and output'''
    for i in range(n) :
        infile = 's{:02d}+'.format(i)
        print infile
        f=fits.open(infile+'.fits')[1].data
        if i == 0 :
            all = f
        else :
            all = np.append(all,f)
    tab=Table(all)
    tab.write(out+'.fits',format='fits',overwrite=True)

def split(write=True,run=True,n=15) :
    ''' Split allStar file into muliple pieces, and run each in parallel '''
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
        print file
        if write :
            out=Table(data[i1:i2])
            out.write(file+'.fits',format='fits',overwrite=True)
        if run :
            log = open(file+'.log','w')
            subprocess.Popen(["python","/home/holtz/python/dist.py",file],stdout=log,stderr=subprocess.STDOUT)

def testcat() :
    ''' Compare distances with catalog distances '''

    from esutil import htm
    # read catalog
    cat=fits.open('apogee-distances_DR12_v1.3.fits')
    cat.verify('fix')
    cat=cat[1].data
    all=fits.open('allStar+.fits')[1].data

    # match objects
    h=htm.HTM()
    maxrad=1./3600.
    m1,m2,rad=h.match(all['RA'],all['DEC'],cat['RA'],cat['DEC'],maxrad)

    # make various catalog comparison plots
    catplot('cat_apokasc',all[m1],1000.*cat['APOKASC_DIST_BAYES'][m2])
    catplot('cat_hip',all[m1],1000./(cat['HIP_PLX'][m2]))
    catplot('cat_rc',all[m1],1000.*cat['RC_DIST'][m2])

def catplot(file,all,dist,index=2,inter=False):
    ''' Plots for catalog comparisons '''
    fig, ax = plt.subplots(2,3)
    fig.subplots_adjust(wspace=0.)
    plots.plotc(ax[0,0],all['teff'],(all['diso'][:,index]-dist)/dist,all['logg'],zr=[0,5],xr=[2500,6500],yr=[-1,1],xt='Teff')
    ax[0,0].set_xticks(np.arange(3000,6000,1000))

    plots.plotc(ax[0,1],all['param_m_h'],(all['diso'][:,index]-dist)/dist,all['logg'],zr=[0,5],xr=[-3,1],yr=[-1,1],xt='[M/H]')
    plt.setp(ax[0,1].get_yticklabels(),visible=False)
    ax[0,1].set_xticks(np.arange(-2.5,0.5,1.0))

    plots.plotc(ax[0,2],all['logg'],(all['diso'][:,index]-dist)/dist,all['logg'],zr=[0,5],xr=[-1,5],yr=[-1,1],xt='log g')
    plt.setp(ax[0,2].get_yticklabels(),visible=False)

    plots.plotc(ax[1,0],all['h'],(all['diso'][:,index]-dist)/dist,all['logg'],zr=[0,5],yr=[-1,1],xt='H')

    plots.plotc(ax[1,1],all['diso'][:,index],(all['diso'][:,index]-dist)/dist,all['logg'],zr=[0,5],yr=[-1,1],xt='Distance')
    plt.setp(ax[0,1].get_yticklabels(),visible=False)
    xmin,xmax= ax[1,1].get_xlim()
    ax[1,1].set_xticks(np.linspace(xmin,xmax,5)[1:-1])

    #plots.plotc(ax[1,2],5*np.log10(dist)-5,5*np.log10(all['diso'][:,index])-5,all['logg'],zr=[0,5],xt='m-M')
    plots.plotc(ax[1,2],5*np.log10(dist)-5,5*np.log10(all['diso'][:,index])-5,bestext(all),zr=[0,1],xt='m-M')
    ax[1,2].yaxis.tick_right()

    if inter :
        x = 5*np.log10(dist)-5
        y = 5*np.log10(all['diso'][:,index])-5
        gd, = np.where(np.isfinite(x) & np.isfinite(y))
        plots._data_x = x[gd]
        plots._data_y = y[gd]
        plots._data = all[gd]
        plots.event(fig)
    else :
        plt.savefig(file+'.jpg')
        plt.close()

def median_index(x,p) :
    ''' Return the index and value of the median value for input NORMALIZED probability array'''
    tot =0.
    i=0
    while tot < 0.5 :
       tot0=tot
       tot += p[i]
       i +=1
    xmed = x[i-1]+(x[i]-x[i-1])*(0.5-tot0)/(tot-tot0)
    return i,xmed

def index(t,g,f,a) :
    ''' Return array indices for input values'''
    print t, (t-teff[0])/dteff
    print g, (g-logg[0])/dlogg
    print f, (f-feh[0])/dfeh
    print a, (a-age[0])/dage



if __name__ == '__main__' :
    main(sys.argv[1])

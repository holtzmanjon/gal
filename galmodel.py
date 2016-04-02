import numpy as np
import pdb
import matplotlib.pyplot as plt

def galmodel(dist,feh,glong,glat,nomet=True,plot=False) :

    ''' Return probabilities in each galactic component for a range of distances at given l, b'''

    #solar distance from galactic center (pc)
    soldistgc = np.double(8000.)

    gallat = np.double(glat)*np.pi/180.
    gallong = np.double(glong)*np.pi/180.

    # GALAXY PARAMETERS FOR PRIOR ASSUMPTIONS  

    rthin = 2900. #thin disk scale length, pc
    zthin = 300.  #thin disk scale height,pc
    zhthin = 94.69*2**1.666
    rthick = 2400. #thick disk scale length, pc
    zthick = 800.  #thick disk scale height, pc
    zhthick= 800.
    bulgescale = 2500. # bulge scale length,pc
    bulgetrunc = 95.   # bulge inner truncation radius, pc
    bulgeangle = -15.*np.pi/180. #15 degree major axis angle relative to the sun-gc LoS, converted to radians
    bulgezeta = 0.31  # z axis ratio for bulge
    bulgeeta = 0.68 #y axis ratio for bulge

    #metallicity distribution for our priors.  the metallicity
    #distribution will probably need to be revisisted, most studies of
    #these (other than the bulge) take place at the solar neighborhood.
    #might need tweaking (larger standard deviations, or perhaps as a function of R_gc).

    bulgemetal = -0.1  # [Fe/H]
    bulgesig = 0.5  #  standard deviation of metallicity for bulge
    bulgenorm = 406. # msol/pc**3. at GC
    halometal = -1.6 # [Fe/H]
    halosig = 0.5 # standard devation of metallicity for halo
    halonorm = 0.03 # msol/pc**3 for GC (the normalization was truncated at 1 kpc, as it blows up as you approach GC)
    thinmetal = 0. # [Fe/H]
    thinsig = 0.20  # Standard Deviation of metallicity for thin disk
    thinnorm = 0.12 # Msol/pc**3 at GC
    thinhnorm = 55.4082/1202.43 # Msol/pc**3 at solar circle
    thickmetal = -0.6 # [Fe/H]
    thicksig = 0.5 # standard deviation in metallicity for the thick disk
    thicknorm = 0.03 # msol/pc**3 at GC
    thickhnorm = 0.028 # msol/pc**3 at GC

    # distance above plane
    tempz = np.sin(gallat)*dist 
    # distance to target in plane
    tempprojection = np.cos(gallat)*dist 
    # radial distance from the galactic center
    tempr = np.sqrt(soldistgc**2.+tempprojection**2.-2.*soldistgc*tempprojection*np.cos(gallong))

    # distance from galactic center for halo stars (i.e. those out of the plane of the galaxy)
    temphalodist = np.sqrt(tempr**2.+tempz**2.)
         
    # the following loop calculates X & Y
    # coordinates for each star. (0,0) is at
    # GC, (0,-8 kpc) for the sun
    x=temphalodist*0.
    y=temphalodist*0.
    tmp = (tempprojection**2.-tempr**2.-8000.**2)/(-2.*8000.*tempr)
    bd = np.where(tmp > 1.)
    tmp[bd] = 1.
    bd = np.where(tmp < -1.)
    tmp[bd] = -1.
    theta = np.arccos(tmp)
    if gallong >= np.pi :
        j=np.where(theta < np.pi/2.)
        if len(j) > 0 :
          x[j] = tempr[j]*np.sin(theta[j])
          beta = np.pi/2.-theta[j]
          y[j] = -tempr[j]*np.sin(beta)
        j2=np.where(theta >= np.pi/2.)
        if len(j2) > 0 :
          thetaprime = np.pi-theta[j2]
          x[j2] = tempr[j2]*np.sin(thetaprime)
          beta = np.pi/2. - thetaprime
          y[j2] = tempr[j2]*np.sin(beta)

    if gallong < np.pi :
        j=np.where(theta > np.pi/2.)
        if len(j) > 0 :
          thetaprime = np.pi-theta[j]
          x[j] = -tempr[j]*np.sin(theta[j])
          beta = np.pi/2.-thetaprime
          y[j] = tempr[j]*np.sin(beta)
        j2=np.where(theta <= np.pi/2.)
        if len(j2) > 0 :
          x[j2] = -tempr[j2]*np.sin(theta[j2])
          beta = np.pi/2.-theta[j2]
          y[j2] = -tempr[j2]*np.sin(beta)       

    #convert X & Y to bulge X & Y, assuming a 15 degree shift for the bar.  
    bulgecoordx = x*np.cos(bulgeangle)+y*np.sin(bulgeangle)
    bulgecoordy = -x*np.sin(bulgeangle)+y*np.cos(bulgeangle)

    # no bulge past 30 kpc
    bd=np.where(tempr > 30000.)
    if len(bd) > 0 :
      bulgecoordx[bd] = 100000.
      bulgecoordy[bd] = 100000.

    # triaxial bulge from Binney et al
    coords = (bulgecoordx**2.+(bulgecoordy**2.)/(bulgeeta**2.)+(tempz**2.)/(bulgezeta**2.))**(1./2.)

    #prior probabilities for the spatial and metallicity distribution functions. Separated the SFH into
    #its own probability as it is most likely to change.                             
    #triaxial bulge
    pbulge = bulgenorm*((np.exp(-(coords**2.)/(bulgescale**2.)))/((1.+coords/bulgetrunc)**1.8))
    # Burnett & Binney halo
    phalo= halonorm*(temphalodist/1000.)**(-3.39)
    # TRILEGAL halo
    halonorm = 1.e-4/(np.exp(-7.67*((8000./2698.)**0.25-1.)))
    phalo= halonorm*(np.exp(-7.67*((temphalodist/2698.)**0.25-1.)))

    # exponential relations for vertical disk distribituion
    # pthick = thicknorm*np.exp(-tempr/rthick-abs(tempz)/zthick)
    # pthin = thinnorm*np.exp(-tempr/rthin-abs(tempz)/zthin)
    # hyperbolic secant relations
    pthin = thinhnorm*(np.exp(-tempr/rthin)*1./np.cosh(0.5*tempz/zhthin)**2)
    pthick = thickhnorm*(np.exp(-tempr/rthick)*1./np.cosh(0.5*tempz/zhthick)**2)
    # metallicity priors
    if not nomet :
      pbulge*=np.exp(-(feh-bulgemetal)**2./(2.*bulgesig**2.))*1./(bulgesig*np.sqrt(2.*np.pi))
      phalo *= 1./(halosig*np.sqrt(2.*np.pi))*np.exp(-(feh-halometal)**2./(2.*halosig**2.))
      pthick *= 1./(thicksig*np.sqrt(2.*np.pi))*np.exp(-(feh-thickmetal)**2./(2.*thicksig**2.))
      pthin *= 1./(thinsig*np.sqrt(2.*np.pi))*np.exp(-(feh-thinmetal)**2./(2.*thinsig**2.))

    if plot :
        plt.plot(dist,np.log10(pthin))
        plt.plot(dist,np.log10(pthick))
        plt.plot(dist,np.log10(pbulge))
        plt.plot(dist,np.log10(phalo))
    return pthin,pthick,pbulge,phalo


def test() :
    dist=np.arange(0.,20.,0.1)
    out=galmodel(dist,0.,0.,0.)
    return out

def lbd2xyz(l,b,d,R0=8.5) :
    ''' Angular coordinates + distance -> galactocentry x,y,z '''

    brad = b*np.pi/180.
    lrad = l*np.pi/180.

    x = d*np.sin(0.5*np.pi-brad)*np.cos(lrad)-R0
    y = d*np.sin(0.5*np.pi-brad)*np.sin(lrad)
    z = d*np.cos(0.5*np.pi-brad)
    r = np.sqrt(x**2+y**2) 
    return x, y, z, r


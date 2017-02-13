def ghb(jk0,feh,dwarf=False) :
    """
    Color-temperature relation from Gonzalez Hernandez & Bonifacio (2009):  (J-K)_0 - Teff
    """
    if dwarf :
      b0=0.6524 ; b1=0.5813 ; b2=0.1225 ; b3=-0.0646 ; b4=0.0370 ; b5=0.0016 # dwarfs
    else :
      b0=0.6517 ; b1=0.6312 ; b2=0.0168 ; b3=-0.0381 ; b4=0.0256 ; b5=0.0013 # giants
    theta=b0+b1*jk0+b2*jk0**2+b3*jk0*feh+b4*feh+b5*feh**2
    dtheta_djk = b1+2*b2*jk0+b3*feh
    dt_djk= -5040./theta**2*dtheta_djk

    return 5040./theta, dt_djk


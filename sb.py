# Program to supply Sersic function and create plot of profiles for various
#   Sersic indices

import numpy as np
import matplotlib.pyplot as plt

def sersic(r, I_e, R_e, n=1, mag=False) :
    '''
    routine to return Sersic profile

    Args:
        r:  input array of radii at which to calculate profile
       I_e:  surface brightness at effective radius, in intensity or magnitude
            units, depending on mag
       R_e:  effective radius

    Keyword args:
        n:  sersic index (default=1, i.e. exponential)
      mag:  False fo profile in intensity, true for profile in mags
    '''
    bn=1.99*n-0.327
    I_r = np.exp (-bn * ((r/R_e)**(1./n) - 1) )           # Sersic function
    if mag :
        return I_e - 2.5 *np.log10(I_r)
    else :
        return I_e * I_r

if __name__ == '__main__' :
    # input radii
    r=np.arange(0.,5.,0.01)

    # loop over Sersic indices
    I_e=0.
    R_e=1.
    for n in [0.5,1.,2.,3.,4.,5.,6.] :
        plt.plot(r,sersic(r,I_e,R_e,n=n,mag=True),label='n={:4.1f}'.format(n))

    # annotate plot
    plt.legend()
    plt.xlabel('Radius (r/R_e)')
    plt.ylabel('Surface brightness (m-m_e)')
    plt.ylim(sersic(r[-1],I_e,R_e,n=0.5,mag=True),sersic(0.,I_e,R_e,n=6,mag=True))


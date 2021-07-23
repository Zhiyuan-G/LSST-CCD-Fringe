# Modified code from TMM package https://github.com/sbyrnes321/tmm
# Using numba to speed up code

from numba import jit
import numba
import sys
import numba_scipy
from numpy import cos, inf, zeros, array, exp, conj, nan, isnan, pi, sin
import numpy as np
EPSILON = sys.float_info.epsilon

from Getindex import load_refraction_data
index_of_refraction = load_refraction_data(Epoxy_ind=1.6,Temp = 183.)


@jit(nopython = True,nogil=True)
def make_2x2_array(a, b, c, d, dtype=numba.complex64):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = np.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

@jit(nopython = True,nogil=True)
def is_forward_angle(n, theta):

    ncostheta = n * cos(theta)
    if abs(ncostheta.imag) > 100 * EPSILON:
        answer = (ncostheta.imag > 0)
    else:
        answer = (ncostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    return answer


@jit(nopython = True,nogil=True)
def snell(n_1, n_2, th_1):

    # Important that the arcsin here is scipy.arcsin, not numpy.arcsin! (They
    # give different results e.g. for arcsin(2).)
    th_2_guess = np.arcsin(n_1*np.sin(th_1) / n_2)
    if is_forward_angle(n_2, th_2_guess):
        return th_2_guess
    else:
        return pi - th_2_guess
    
@jit(nopython = True,nogil=True)
def list_snell(n_list, th_0):
    angles = np.arcsin(n_list[0]*np.sin(th_0) / n_list)
    if not is_forward_angle(n_list[0], angles[0]):
        angles[0] = pi - angles[0]
    if not is_forward_angle(n_list[-1], angles[-1]):
        angles[-1] = pi - angles[-1]
    return angles

@jit(nopython = True,nogil=True)
def interface_r(polarization, n_i, n_f, th_i, th_f):
    """
    reflection amplitude (from Fresnel equations)

    polarization is either "s" or "p" for polarization

    n_i, n_f are (complex) refractive index for incident and final

    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        return ((n_i * cos(th_i) - n_f * cos(th_f)) /
                (n_i * cos(th_i) + n_f * cos(th_f)))
    elif polarization == 'p':
        return ((n_f * cos(th_i) - n_i * cos(th_f)) /
                (n_f * cos(th_i) + n_i * cos(th_f)))
    else:
        raise ValueError("Polarization must be 's' or 'p'")
        
        
@jit(nopython = True)
def interface_t(polarization, n_i, n_f, th_i, th_f):
    """
    transmission amplitude (frem Fresnel equations)

    polarization is either "s" or "p" for polarization

    n_i, n_f are (complex) refractive index for incident and final

    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        return 2 * n_i * cos(th_i) / (n_i * cos(th_i) + n_f * cos(th_f))
    elif polarization == 'p':
        return 2 * n_i * cos(th_i) / (n_f * cos(th_i) + n_i * cos(th_f))
    else:
        raise ValueError("Polarization must be 's' or 'p'")
        
@jit(nopython = True,nogil=True)
def R_from_r(r):
    """
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return abs(r)**2


@jit(nopython = True,nogil=True)
def T_from_t(pol, t, n_i, n_f, th_i, th_f):
    if pol == 's':
        return abs(t**2) * (((n_f*cos(th_f)).real) / (n_i*cos(th_i)).real)
    elif pol == 'p':
        return abs(t**2) * (((n_f*conj(cos(th_f))).real) /
                                (n_i*conj(cos(th_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")
        
        
@jit(nopython = True,nogil=True)
def coh_tmm(pol, n_list, d_list, th_0, lam_vac):
    num_layers = n_list.size
    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * cos(th_list) / lam_vac


    delta = kz_list * d_list



    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    t_list = zeros((num_layers, num_layers), dtype=numba.complex64)
    r_list = zeros((num_layers, num_layers), dtype=numba.complex64)
    for i in range(num_layers-1):
        t_list[i,i+1] = interface_t(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        r_list[i,i+1] = interface_r(pol, n_list[i], n_list[i+1],
                                    th_list[i], th_list[i+1])
        
    M_list = zeros((num_layers, 2, 2), dtype=numba.complex64)
    
    for i in range(1, num_layers-1):
        M_list[i] = (1/t_list[i,i+1]) * np.dot(
            make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
                           dtype=numba.complex64),
            make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=numba.complex64))
        
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=numba.complex64)
    for i in range(1, num_layers-1):
        Mtilde = np.dot(Mtilde, M_list[i])
    
    Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=numba.complex64)/t_list[0,1], Mtilde)
    
    # Net complex transmission and reflection amplitudes
    r = Mtilde[1,0]/Mtilde[0,0]
    t = 1/Mtilde[0,0]
    R = R_from_r(r)
    T = T_from_t(pol, t, n_list[0], n_list[-1], th_0, th_list[-1])
    return (R,T)




@jit(nopython = True,nogil=True)
def E2V_model(d_si2,n_list,wlen):
    
    thickness_um = np.array([np.inf, 0.1221,0.0441,100., 0.1, 0.3,1.,14.8+d_si2,130-d_si2, np.inf])

    # Assume normal incidence.
    theta = 0.
    R = T = 0
    for pol in 'sp':

        # Using tmm package to do multilayer thin-film simulation
        rr,tt = coh_tmm(pol, n_list, thickness_um, theta, 1e-3 * wlen)
        # Store R and T results
        R += 0.5*rr
        T += 0.5*tt

    A = 1 - R - T

    return(A)

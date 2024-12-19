# Author: J Asplet

# Some simple routines for calculating BODY WAVE radiaion patterns.

# Equations used are taken from some old GEOSC 488 notes (a course run by
# Chuck Ammon at Penn State)
# These expressions (or simialr versions) can be found in:
#    - Aki and Richards (1980)
#    - H. Dufumier and L. Rivera.,
#    On the resolution of the isotropic component in moment tensor inversion,
#    Geophys. J . Int. (1997) 131, 595-606
#    - Ammon, C.J., Velasco, A.A., Lay, T., Wallace, T.C., 2021.
#    Chapter 17 - Seismic point-source radiation patterns,
#    in: Ammon, C.J., Velasco, A.A., Lay, T., Wallace, T.C. (Eds.),
#    Foundations of Modern Global Seismology (Second Edition). Academic Press,
#    pp. 463–484. https://doi.org/10.1016/B978-0-12-815679-7.00025-2

# Chuck's textbook is probably the best first port-of-call for an interested
# reader. It is very clear and easy to read.

# Routines here calculate radiation patterns,
# functions in plots.py plots radiation patterns.

# direct import these functions to keep things concise
from numpy import sin, cos, deg2rad
import numpy as np
import pygc
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from obspy.taup import TauPyModel


def model_ray_param_ak135(lat1, lon1, lat2, lon2, vp=5, vs=3):

    dist_km = pygc.great_distance(start_latitude=lat1,
                                  end_latitude=lat2,
                                  start_longitude=lon1,
                                  end_longitude=lon2)['distance'] * 1e-3
    dist_deg = kilometers2degrees(dist_km)
    model = TauPyModel(model="ak135")
    arrivals = model.get_ray_paths(source_depth_in_km=1,
                                   distance_in_degree=dist_deg,
                                   phase_list=["P", "S"])
    p = False
    s = False
    for arrival in arrivals:
        if (arrival.name == 'P') and (p == False):
            print(p)
            ray_param_p = 1/degrees2kilometers(1/arrival.ray_param_sec_degree) #in s/km
            print(ray_param_p)
            takeoff_p = np.rad2deg(np.arcsin(ray_param_p * vp))
            p = True
        elif arrival.name == 'S' and ~s:
            ray_param_s = 1/degrees2kilometers(1/arrival.ray_param_sec_degree) #in s/km
            takeoff_s = np.rad2deg(np.arcsin(ray_param_s * vs))
            s = True

    return takeoff_p, takeoff_s


def calc_coefficiants(rake, dip, strike, reciever_azi):
    '''
    Calculate co-efficiants used in Ammon et al., (2021)
    '''
    theta = reciever_azi - strike
#   sr
    s_r = sin(rake)*sin(dip)*cos(dip)
#   qr
    q_r_1 = sin(rake)*cos(2*dip)*sin(theta) 
    q_r_2 = cos(rake)*cos(dip)*cos(theta)
    q_r = q_r_1 + q_r_2
#   pr
    p_r_1 = cos(rake)*sin(dip)*sin(2*theta)
    p_r_2 = sin(rake)*sin(dip)*cos(dip)*cos(2*theta)
    p_r = p_r_1 - p_r_2
#   ql
    q_l_1 = -(cos(rake)*cos(dip))*sin(theta) 
    q_l_2 = (sin(rake)*cos(2*dip))*cos(theta)
    q_l = q_l_1 + q_l_2
#   pl
    p_l_1 = sin(rake)*sin(dip)*cos(dip)*sin(2*theta)
    p_l_2 = cos(rake)*sin(dip)*cos(2*theta)
    p_l = p_l_1 + p_l_2
    return s_r, q_r, p_r, q_l, p_l


def calc_rad_patterns(rake, dip, strike, reciever_azi, takeoff_angle):

    s_r, q_r, p_r, q_l, p_l = calc_coefficiants(rake,
                                                dip,
                                                strike,
                                                reciever_azi)

    # Calc P radiation pattern
    rad_p_1 = s_r*(3*(cos(takeoff_angle)**2) - 1) 
    rad_p_2 = q_r*sin(2*takeoff_angle)
    rad_p_3 = p_r*(sin(takeoff_angle)**2)
    rad_p = rad_p_1 - rad_p_2 - rad_p_3

    # Calc SV radiation
    rad_sv_1 = 1.5*s_r*sin(2*takeoff_angle)
    rad_sv_2 = q_r*cos(2*takeoff_angle)
    rad_sv_3 = 0.5*p_r*sin(2*takeoff_angle)
    rad_sv = rad_sv_1 + rad_sv_2 + rad_sv_3

    # Calc SH radiation
    rad_sh = q_l*cos(takeoff_angle) + p_l*sin(takeoff_angle)
    return rad_p, rad_sv, rad_sh


def radiation_patterns_2d(fault, n_azis, n_takeoffs):

    takeoffs = np.linspace(0, np.pi/2, n_takeoffs)
    azimuths = np.linpace(0, 2*np.pi, n_azis)
    r_p_2d = np.zeros((len(takeoffs), len(azimuths)))
    r_sh_2d = np.zeros((len(takeoffs), len(azimuths)))
    r_sv_2d = np.zeros((len(takeoffs), len(azimuths)))

    for t in range(0, n_takeoffs):
        # we could possibly vecotirze this loop...?
        p_rad, sv_rad, sh_rad = calc_rad_patterns(rake=deg2rad(fault['rake']),
                                                  dip=deg2rad(fault['dip']),
                                                  strike=deg2rad(fault['strike']),
                                                  reciever_azi=azimuths,
                                                  takeoff_angle=takeoffs[t])
        r_p_2d[t, :] = p_rad
        r_sv_2d[t,:] = sv_rad
        r_sh_2d[t,:] = sh_rad

    return r_p_2d, r_sv_2d, r_sh_2d

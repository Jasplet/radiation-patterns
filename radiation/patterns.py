# Author: J Asplet

# Some simple routines for calculating BODY WAVE radiaion patterns.

# Equations used are taken from some old GEOSC 488 notes (a course run by Chuck Ammon at Penn State)
# These expressions (or simialr versions) can be found in:
#          - Aki and Richards (1980)
#          - H. Dufumier and L. Rivera., On the resolution of the isotropic component in moment tensor inversion,
#            Geophys. J . Int. (1997) 131, 595-606
#          - Ammon, C.J., Velasco, A.A., Lay, T., Wallace, T.C., 2021. 
#            Chapter 17 - Seismic point-source radiation patterns, 
#            in: Ammon, C.J., Velasco, A.A., Lay, T., Wallace, T.C. (Eds.),
#            Foundations of Modern Global Seismology (Second Edition). Academic Press,
#            pp. 463–484. https://doi.org/10.1016/B978-0-12-815679-7.00025-2 
#
# Chuck's textbook is probably the best first port-of-call for an interested reader. It is very clear and easy to read.

# Routines here calculate radiation patterns, functions in plots.py plots radiation patterns.

from numpy import sin, cos, deg2rad



def calc_coefficiants(rake, dip, strike, reciever_azi):

    theta = reciever_azi - strike
    #sr
    s_r = sin(rake)*sin(dip)*cos(dip)
    #qr
    q_r_1 = sin(rake)*cos(2*dip)*sin(theta) 
    q_r_2 = cos(rake)*cos(dip)*cos(theta)
    q_r = q_r_1 + q_r_2
    #pr
    p_r_1 = cos(rake)*sin(dip)*sin(2*theta)
    p_r_2 = sin(rake)*sin(dip)*cos(dip)*cos(2*theta)
    p_r = p_r_1 - p_r_2
    #ql
    q_l_1 = -(cos(rake)*cos(dip))*sin(theta) 
    q_l_2 = (sin(rake)*cos(2*dip))*cos(theta)
    q_l = q_l_1 + q_l_2
    #pl
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
    rad_p_1 = s_r*(3*(np.cos(takeoff_angle)**2) - 1) 
    rad_p_2 = q_r*np.sin(2*takeoff_angle)
    rad_p_3 = p_r*(np.sin(takeoff_angle)**2)
    rad_p = rad_p_1 - rad_p_2 - rad_p_3

    # Calc SV radiation
    rad_sv_1 = 1.5*s_r*np.sin(2*takeoff_angle)
    rad_sv_2 = q_r*np.cos(2*takeoff_angle)
    rad_sv_3 = 0.5*p_r*np.sin(2*takeoff_angle)
    rad_sv = rad_sv_1 + rad_sv_2 + rad_sv_3

    # Calc SH radiation
    rad_sh = q_l*np.cos(takeoff_angle) + p_l*np.sin(takeoff_angle)
    return rad_p, rad_sv, rad_sh


def p_radiation_2d(azimuths, takeoffs, fault):
    rp = np.zeros((len(theta), len(phi)))
    for t in range(0,len(takeoffs)):
        p_rad, _, _ = calc_rad_patterns(rake=np.deg2rad(fault['rake']),
                                          dip=np.deg2rad(fault['dip']),
                                          strike=np.deg2rad(fault['strike']),
                                          reciever_azi=azimuths,
                                          takeoff_angle=takeoffs[t])
        rp[:, t] = p_rad
    return rp

def sh_radiation_2d(azimuths, takeoffs, fault):
    rp = np.zeros((len(theta), len(phi)))
    for t in range(0,len(takeoffs)):
        p_rad, _, _ = calc_rad_patterns(rake=np.deg2rad(fault['rake']),
                                          dip=np.deg2rad(fault['dip']),
                                          strike=np.deg2rad(fault['strike']),
                                          reciever_azi=azimuths,
                                          takeoff_angle=takeoffs[t])
        rp[:, t] = p_rad
    return rp

def sv_radiation_2d(azimuths, takeoffs, fault):
rp = np.zeros((len(theta), len(phi)))
for t in range(0,len(takeoffs)):
    p_rad, _, _ = calc_rad_patterns(rake=np.deg2rad(fault['rake']),
                                      dip=np.deg2rad(fault['dip']),
                                      strike=np.deg2rad(fault['strike']),
                                      reciever_azi=azimuths,
                                      takeoff_angle=takeoffs[t])
    rp[:, t] = p_rad
return rp
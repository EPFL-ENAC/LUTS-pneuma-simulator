from math import atan2, cos, sin, sqrt

import numpy as np
from numba import jit


@jit(nopython=True)
def calc_dtc(
    l_j: float,
    w_j: float,
    l_i: float,
    w_i: float,
    x_j: float,
    y_j: float,
    x_i: float,
    y_i: float,
    theta_j: float,
    theta_i: float,
) -> float:
    """
    Calculates the distance to closest approach (DTC) between two objects.

    Args:
        l_j (float): Length of object j.
        w_j (float): Width of object j.
        l_i (float): Length of object i.
        w_i (float): Width of object i.
        x_j (float): x-coordinate of object j.
        y_j (float): y-coordinate of object j.
        x_i (float): x-coordinate of object i.
        y_i (float): y-coordinate of object i.
        theta_j (float): Orientation angle of object j.
        theta_i (float): Orientation angle of object i.

    Returns:
        float: The distance to closest approach (DTC) between the two objects.
    """

    # distance from i to j
    s_i_j = sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
    # distance of closest approach between i and j
    min_d = ellipses(l_j, w_j, l_i, w_i, x_j, y_j, x_i, y_i, theta_j, theta_i)
    dtc = s_i_j - min_d
    return dtc


@jit(nopython=True)
def ellipses(
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    theta1: float,
    theta2: float,
) -> float:
    """Subroutine to calculate the distance of closest approach of two ellipses

    Original code written in Fortran90:
    https://www.math.kent.edu/~zheng/ellipse.html
    https://www.math.kent.edu/~zheng/ellipses.f90

    This is my Pythonic implementation powered by JIT.

    Args:
        a1 (float): length of major semiaxis of first ellipse
        b1 (float): length of minor semiaxis of first ellipse
        a2 (float): length of major semiaxis of second ellipse
        b2 (float): length of minor semiaxis of second ellipse
        x1 (float): x coordinate of the center of the first ellipse
        y1 (float): y coordinate of the center of the first ellipse
        x2 (float): x coordinate of the center of the second ellipse
        y2 (float): y coordinate of the center of the second ellipse
        theta1 (float): angle associated with the major axis of first ellipse
        theta2 (float): angle associated with the major axis of second ellipse

    Returns:
        float: distance between the centers when two ellipses are externally tangent
    """
    theta3 = atan2(y2 - y1, x2 - x1)
    cs1, sn1 = cos(theta1), sin(theta1)
    cs3, sn3 = cos(theta3), sin(theta3)
    k1d = cos(theta3 - theta1)
    k2d = cos(theta3 - theta2)
    k1k2 = cos(theta2 - theta1)
    # eccentricity of ellipses
    e1 = 1 - (b1**2 / a1**2)
    e2 = 1 - (b2**2 / a2**2)
    # component of A'
    eta = a1 / b1 - 1
    a11 = b1**2 / b2**2 * (1 + 0.5 * (1 + k1k2) * (eta * (2 + eta) - e2 * (1 + eta * k1k2) ** 2))
    a12 = b1**2 / b2**2 * 0.5 * sqrt(1 - k1k2**2) * (eta * (2 + eta) + e2 * (1 - eta**2 * k1k2**2))
    a22 = b1**2 / b2**2 * (1 + 0.5 * (1 - k1k2) * (eta * (2 + eta) - e2 * (1 - eta * k1k2) ** 2))
    # eigenvalues of A'
    lambda1 = 0.5 * (a11 + a22) + 0.5 * sqrt((a11 - a22) ** 2 + 4 * a12**2)
    lambda2 = 0.5 * (a11 + a22) - 0.5 * sqrt((a11 - a22) ** 2 + 4 * a12**2)
    # major and minor axes of transformed ellipse
    b2p = 1 / sqrt(lambda1)
    a2p = 1 / sqrt(lambda2)
    deltap = a2p**2 / b2p**2 - 1
    if abs(k1k2) == 1:
        if a11 > a22:
            kpmp = 1 / sqrt(1 - e1 * k1d**2) * b1 / a1 * k1d
        else:
            kpmp = (sn3 * cs1 - cs3 * sn1) / sqrt(1 - e1 * k1d**2)
    elif deltap != 0:
        kpmp = (
            a12 / sqrt(1 + k1k2) * (b1 / a1 * k1d + k2d + (b1 / a1 - 1) * k1d * k1k2)
            + (lambda1 - a11) / sqrt(1 - k1k2) * (b1 / a1 * k1d - k2d - (b1 / a1 - 1) * k1d * k1k2)
        ) / sqrt(2 * (a12**2 + (lambda1 - a11) ** 2) * (1 - e1 * k1d**2))
    else:
        kpmp = 0
    if kpmp == 0 or deltap == 0:
        Rc = a2p + 1
        # The distance of closest approach
        dist = Rc * b1 / sqrt(1 - e1 * k1d**2)
        return dist
    else:
        # coefficients of quartic for q
        t = 1 / kpmp**2 - 1
        A = -1 / b2p**2 * (1 + t)
        B = -2 / b2p * (1 + t + deltap)
        C = -t - (1 + deltap) ** 2 + 1 / b2p**2 * (1 + t + deltap * t)
        D = 2 / b2p * (1 + t) * (1 + deltap)
        E = (1 + t + deltap) * (1 + deltap)
        # https://github.com/numba/numba/issues/3568
        # coeffs are now in an array, and the dtype is complex space.
        rts = np.real(np.roots(np.array([A, B, C, D, E], dtype=np.complex128)))
        qq = rts[rts > 0][0]
        # substitute for R'
        Rc = sqrt(
            (qq**2 - 1) / deltap * (1 + b2p * (1 + deltap) / qq) ** 2 + (1 - (qq**2 - 1) / deltap) * (1 + b2p / qq) ** 2
        )
        # The distance of closest approach
        dist = Rc * b1 / sqrt(1 - e1 * k1d**2)
        return dist

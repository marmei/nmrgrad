#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the Biot-Savart line integral.

"""
from multiprocessing import Pool
from functools import partial
import numpy as np

def calc_biot_savart(field_pt, delta_l, wire_elem_c, current=1.0,
                     mu_not=1.25663706212e-06):
    """
    Calculates the Biot-Savart line integral at a given point, ``field_pt``.

    Parameters
    ----------
    field_pt : ndarray
        The point were to compute the B-field.
    delta_l : ndarray
        The length for each wire element.
    wire_elem_c : ndarray
        The wire element center which has n number of points of shape (n, 3).

    Returns
    -------
    ndarray
        The magnetic flux density at the given point, field_pt.

    """
    disp_vect = field_pt - wire_elem_c
    cross_p = np.cross(delta_l, disp_vect)
    f_vector = (np.divide(cross_p.T, (disp_vect**2).sum(axis=1)**1.5).T).sum(axis=0)

    return mu_not * current / (4.0 * np.pi) * f_vector

def biot_line_integral(wire_elem_pt, r_field, current=1.0):
    """
    Calculates the magnetic flux density from a list of points.

    ``wire_elem_pt`` represent the discretized points on the conductor.
    The magnetic flux density is calculated for all r_field positions,
    and is returned in the same shape.

    All coordinates are in m.

    Parameters
    ----------
    wire_elem_pt : ndarray
        The points of the wire element in the format \
        array([[x_0, y_0, z_0], ... , [x_n, y_n, z_n]]).
    r_field : ndarray
        The positions were to compute the field vector.
    current : float, optional
        The current in Ampere that is flowing through the wire. The default is 1.0.

    Returns
    -------
    bfield_out : ndarray
        Magnetric flux density, with the same shape as r_field.

    """
    pool = Pool()

    delta_l = wire_elem_pt[1:] - wire_elem_pt[:-1]
    bfield_out = np.zeros_like(r_field)

    wire_elem_centre_pt = wire_elem_pt[:-1] + delta_l * 0.5

    part_calc_biot_savart = partial(calc_biot_savart,
                                    delta_l=delta_l,
                                    wire_elem_c=wire_elem_centre_pt,
                                    current=current)

    bfield_out[:] = pool.map(part_calc_biot_savart,
                             r_field[range(r_field.shape[0])])

    return bfield_out

def mgrid_2d_roi(axis_x, axis_y, axis_const,
                 roi=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]) * 1e-3,
                 pos_axis_const=0.0, ndots=50):
    """
    Create the roi mesh grid on a 2D plane.

    Parameters
    ----------
    axis_x : str
        Selects the respective axis for the resulting 2D roi.
    axis_y : str
        Selects the respective axis for the resulting 2D roi.
    axis_const : str
        Selects the constant axis.
    roi : ndarray, optional
        The cubic definition of the region of interest. \
        The default is np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]) * 1e-3.
    pos_axis_const : float, optional
        The offset of the constant axis in metre. The default is 0.0.
    ndots : int, optional
        The number of grid points. The default is 50.

    Returns
    -------
    x_pts : mgrid
        Meshgrid for the x positions.
    y_pts : mgrid
        Meshgrid for the y positions.
    r_p : mgrid
        Meshgrid for the field vectors.

    """
    axes = {'x': 0, 'y': 1, 'z': 2}

    x_pts, y_pts = np.mgrid[roi[:, axes[axis_x]].min():roi[:, axes[axis_x]].max():ndots*1j,
                            roi[:, axes[axis_y]].min():roi[:, axes[axis_y]].max():ndots*1j]

    r_p = np.zeros([x_pts.ravel().shape[0], 3])
    r_p[:, axes[axis_x]] = x_pts.ravel()
    r_p[:, axes[axis_y]] = y_pts.ravel()
    r_p[:, axes[axis_const]] = np.zeros_like(x_pts.ravel()) + pos_axis_const

    return x_pts, y_pts, r_p

if __name__ == "__main__":
    import time

    Z_NOT = 1.312e-3

    x = np.linspace(-.1, .1, 200)
    y = np.zeros_like(x) + Z_NOT
    z = np.zeros_like(x)

    wire_elem_points = np.array([x, y, z]).swapaxes(0, 1)

    t0 = time.time()

    _, _, RP = mgrid_2d_roi("x", "y", "z", ndots=9)
    bfield_rp = biot_line_integral(wire_elem_points, RP)

    bfield_centre = biot_line_integral(wire_elem_points, np.array([[0.0, 0.0, 0.0]]))

    t1 = time.time()
    print(bfield_rp)

    # infinite wire, analytical solution:
    b_mag_analytical = np.abs(np.pi * 4e-7 / (2 * np.pi * Z_NOT))

    print("error of bfield compared to analytical solution: {:2.3f} %".format( \
          np.abs(1 - np.linalg.norm(bfield_centre) / b_mag_analytical) * 100))

    print("took %f seconds" %(t1 - t0))

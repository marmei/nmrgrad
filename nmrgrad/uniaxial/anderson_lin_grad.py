#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the y- and z-axial linear current shim base on straight conductors.

[1] Anderson, 1961, Electrical Current Shims for Correcting Magnetic Fields, \
    doi: 10.1063/1.1717338

"""
import numpy as np
from skimage import measure

def idx_nearest(array, value):
    """
    Find the index within an array where the value matches closely.

    Parameters
    ----------
    array : ndarray
        1d numpy array.
    value : float
        The value to search within the array.

    Returns
    -------
    int
        Index of the element closest to ``value``.

    """
    return (np.absolute(array - value)).argmin()

class AndersonLinGradY:
    """
    Compute the Anderson y-gradient.

    """
    n_contour_pts = 2000
    _y_w = None
    _y_c = None
    mu_not = 1.25663706212e-06

    def __init__(self):
        grid_x, grid_y = np.mgrid[0.01:10.0:self.n_contour_pts*1j,
                                  0.01:10.0:self.n_contour_pts*1j]

        values = self._w_yc_zero_f(grid_x.ravel(), grid_y.ravel())
        values.shape = grid_x.shape

        # get the max z_0, w, y_c, see Fig 4. in Anderson 1961
        w_yc_z_not_ratio = measure.find_contours(values, 0.0)
        w_yc_z_not_ratio.sort(key=lambda x:x.T[1].min())

        w_yc_z_not_ratio = w_yc_z_not_ratio[2] / (self.n_contour_pts-1) * 9.99 + 0.01
        self._y_w = w_yc_z_not_ratio[:, 0]
        self._y_c = w_yc_z_not_ratio[:, 1]

    @staticmethod
    def _w_yc_zero_f(y_w, y_c):
        return -16*((y_w + 2*y_c)**4 - 24*(y_w + 2*y_c)**2 + 16) \
                / ((y_w + 2*y_c)**8 + 16*(y_w + 2*y_c)**6 + 96*(y_w + 2*y_c)**4 \
                + 256*(y_w + 2*y_c)**2 + 256) \
                + 16*((y_w -2*y_c)**4 - 24*(y_w - 2*y_c)**2 + 16) \
                / ((y_w - 2*y_c)**8 + 16*(y_w - 2*y_c)**6 + 96*(y_w - 2*y_c)**4 \
                + 256*(y_w - 2*y_c)**2 + 256)

    def grad_mag(self, y_w, y_c, z_zero, current=1.0):
        """
        Returns the gradient magnitude in the centre for a given configuration.

        Parameters
        ----------
        y_w : float
            Wire separation.
        y_c : TYPE
            Centre position of the wires.
        z_zero : float
            The plate separation in mm.
        current : float, optional
            The conductor current. The default is 1.0.

        Returns
        -------
        float
            DESCRIPTION.

        """
        delta_b_dash_z1 = 4*((y_w + 2*y_c)**2 - 4)/((y_w + 2*y_c)**2 + 4)**2 \
                          - 4*((y_w - 2*y_c)**2 - 4)/((y_w - 2*y_c)**2 + 4)**2

        return (4 * self.mu_not * current) \
                / (2 * np.pi * (z_zero * 10**(-3))**2) * delta_b_dash_z1

    def calc_grad_y_ratio(self, z_zero):
        """
        Returns the y2-ordered values of gradient conductor configuration.

        This function is used to select a certain gradient strength.

        Parameters
        ----------
        z_zero : float
            The plate separation in mm.

        Returns
        -------
        list(list)
            [[G in T/(mA), y1, y2], ... ].T.

        """
        grad_relation = np.array([np.asarray(self.grad_mag(self._y_w,
                                                           self._y_c,
                                                           z_zero)),
                                  self._y_w * z_zero,
                                  self._y_c * z_zero])
        # sort it
        sort = np.array([grad_relation[0],
                         grad_relation[2] - 0.5 * grad_relation[1], # y_1
                         grad_relation[2] + 0.5 * grad_relation[1] # y_2
                         ]).transpose()

        order = sort[:, 1].argsort() # sort by y_2, steadily growing

        return np.take(sort, order, 0).T

    def set_inter_coordinates(self, z_zero, wire_dist):
        """
        Set the inter coordinates.

        Parameters
        ----------
        z_zero : float
            The plate separation in mm.
        wire_dist : float
            The distance between the wires in mm.

        Returns
        -------
        list
            Returns the gradient magnitude and the y_1 and y_2 coordinates. \
            [G in T/(mA), y1, y2].

        """
        grad_y_ratio = self.calc_grad_y_ratio(z_zero)

        maxgrad = grad_y_ratio[0][grad_y_ratio.argmax(axis=1)[0]]
        maxgrad_y1 = grad_y_ratio[1][grad_y_ratio.argmax(axis=1)[0]]
        maxgrad_y2 = grad_y_ratio[2][grad_y_ratio.argmax(axis=1)[0]]

        grad_gy1y2_points = list()
        grad_gy1y2_points.append([maxgrad, maxgrad_y1, maxgrad_y2])

        lastpos = np.array([maxgrad_y1, maxgrad_y2]) # start pos

        # nearest neighbor
        for n_neighbor in range(grad_y_ratio[1].shape[0]):
            if (grad_y_ratio[1][n_neighbor] >= (lastpos[0] + wire_dist)) \
                and (grad_y_ratio[1][n_neighbor] <= (grad_y_ratio[2][n_neighbor] - wire_dist)):

                grad_gy1y2_points.append([grad_y_ratio[0][n_neighbor],
                                          grad_y_ratio[1][n_neighbor],
                                          grad_y_ratio[2][n_neighbor]])

                lastpos = np.array([grad_y_ratio[1][n_neighbor], grad_y_ratio[2][n_neighbor]])
        return np.asarray(grad_gy1y2_points)

    def max_grad_z_comb(self, z_zero, separation=0.1):
        """
        Calculates a combined z- & y-gradient coil for a given wire separation.
        @todo: maxGrad_ZgradComb

        Where the conductors are separated by a given separation.

        Parameters
        ----------
        z_zero : float
            The plate separation in mm.
        separation : float, optional
            The separation gap between the z-gradient conductor and the \
            y-gradient conductors. The default is 0.1.

        Returns
        -------
        list
            Returns [G in T/(mA), y1, y2].

        """
        grad_y_values = self.calc_grad_y_ratio(z_zero)

        ## get the closest gradient to match with Gz
        idx_grad_y_nearest = idx_nearest(grad_y_values[0], anderson_lin_grad_z(z_zero))
        if (grad_y_values[2][idx_grad_y_nearest] - z_zero) < separation:
            while (grad_y_values[2][idx_grad_y_nearest] - z_zero) <= separation:
                idx_grad_y_nearest -= 1

        return np.swapaxes(grad_y_values, 1,0)[idx_grad_y_nearest]

def anderson_lin_grad_z(z_zero, current=1.0, mu_not=1.25663706212e-06):
    """
    Returns the magnitude of the Anderson linear z grad.

    The wire spacing in y-direction, with respect to the centre \
    is equal to the z_zero separation.

    Parameters
    ----------
    z_zero : float
        The plate separation in mm.
    current : float, optional
        The current through the conductor. The default is 1.0.

    Returns
    -------
    float
        The gradient strength in T/(m A).

    """
    return (mu_not * current) / (np.pi * (z_zero * 1e-3)**2)

if __name__ == "__main__":
    y_gradient = AndersonLinGradY()
    print(y_gradient.calc_grad_y_ratio(1.0))
    print("GmaxPos: [G in T/(mA), y1, y2]")
    print(y_gradient.set_inter_coordinates(1.0, 10))

    import matplotlib.pyplot as plt

    # plot the y_w/y_c ratio
    plt.plot(y_gradient._y_w, y_gradient._y_c)
    plt.xlabel("yw / z0")
    plt.ylabel("yc / z0")
    plt.show()
